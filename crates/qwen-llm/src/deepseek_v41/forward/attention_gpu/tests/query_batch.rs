use super::super::query_batch::{Scratch, TILE, rope_rows};
use super::*;

fn config() -> Config {
    Config {
        hidden: 32,
        heads: 8,
        head_dim: 32,
        query_rank: 32,
        output_rank: 32,
        groups: 1,
        index_heads: 2,
        index_dim: 32,
        index_topk: 4,
        window: 128,
        rope_dim: 8,
        candidate_block: 2,
        candidate_topk: 4,
        norm_eps: 1e-6,
        ..Config::default()
    }
}

fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * seed + i / 17 + 13) % 101) as f32 - 50.0) / 103.0)
        .collect()
}

fn bound<'a>(ctx: &'a MetalContext, c: &Config, s: &Schedule) -> Vec<Weights<'a>> {
    (0..s.layers)
        .map(|id| {
            let mut w = scheduled_weights(ctx, c, s, id, GgmlType::F32);
            w.frequencies = (0..c.rope_dim / 2).map(|i| 0.19 / (i + 1) as f32).collect();
            w
        })
        .collect()
}

fn seed(ctx: &MetalContext, w: &mut Workspace, position: usize) {
    // Identical clean-boundary states, including an odd prefix's pending pair.
    // No dependence on a long prefill run to exercise the 16385 boundary.
    let fill = |t: &MetalTensor, salt| {
        upload(ctx, &values(t.n_elements() as usize, salt), t.shape.clone()).unwrap()
    };
    w.raw = fill(&w.raw, 17);
    for (i, owner) in w.owners.iter_mut().enumerate() {
        owner.compressed = fill(&owner.compressed, 19 + i * 2);
        owner.keys = fill(&owner.keys, 31 + i * 2);
        owner.pending_value = fill(&owner.pending_value, 43 + i * 2);
        owner.pending_gate = fill(&owner.pending_gate, 53 + i * 2);
    }
    w.next_position = position;
}

fn sink_spans(c: &Config, ratio: usize, start: usize, n: usize) -> Vec<(usize, usize, bool)> {
    let matrix = |row: usize| {
        let end = start + row + 1;
        let selected = if ratio == 0 {
            0
        } else {
            (end / ratio).min(c.index_topk)
        };
        c.head_dim.is_multiple_of(32) && c.heads >= 8 && end.min(c.window) + selected >= 256
    };
    let mut spans = Vec::new();
    for row in 0..n {
        let branch = matrix(row);
        if row % TILE != 0 && spans.last().is_some_and(|&(_, _, prior)| prior == branch) {
            spans.last_mut().unwrap().1 += 1;
        } else {
            spans.push((row, 1, branch));
        }
    }
    spans
}

fn compare_case(ctx: &MetalContext, c: &Config, schedule: &Schedule, start: usize, n: usize) {
    let weights = bound(ctx, c, schedule);
    let refs = weights.iter().collect::<Vec<_>>();
    let capacity = start + n + 1;
    let mut scalar = Workspace::new_with_schedule(ctx, c, capacity, &refs, schedule).unwrap();
    let mut batched = Workspace::new_with_schedule(ctx, c, capacity, &refs, schedule).unwrap();
    seed(ctx, &mut scalar, start);
    seed(ctx, &mut batched, start);
    let inputs: Vec<_> = (0..schedule.layers)
        .map(|id| {
            let mut x = values(n * c.hidden, 11 + id * 2);
            for row in x.chunks_exact_mut(c.hidden) {
                let rms =
                    (row.iter().map(|x| x * x).sum::<f32>() / c.hidden as f32 + c.norm_eps).sqrt();
                for x in row {
                    *x /= rms;
                }
            }
            upload(ctx, &x, vec![c.hidden as u64, n as u64]).unwrap()
        })
        .collect();
    let blocks = capacity.div_ceil(c.candidate_block);
    let expected: Vec<_> = (0..schedule.layers)
        .map(|_| {
            [
                prefill::allocate(ctx, c.hidden, n, GgmlType::F32).unwrap(),
                prefill::allocate(ctx, c.index_topk, n, GgmlType::I32).unwrap(),
                prefill::allocate(ctx, blocks, n, GgmlType::I32).unwrap(),
            ]
        })
        .collect();
    execute(ctx, |enc| {
        for row in 0..n {
            for id in 0..schedule.layers {
                scalar
                    .encode(
                        ctx,
                        enc,
                        id,
                        start + row,
                        &weights[id],
                        &prefill::row(&inputs[id], c.hidden, row),
                    )
                    .unwrap();
                for (source, dest, width) in [
                    (&scalar.output, &expected[id][0], c.hidden),
                    (&scalar.selected, &expected[id][1], c.index_topk),
                    (&scalar.candidates, &expected[id][2], blocks),
                ] {
                    copy(ctx, enc, source, &prefill::row(dest, width, row), width).unwrap();
                }
            }
        }
    });
    let rows = Rows::new(ctx, c, capacity, n).unwrap();
    let output = prefill::allocate(ctx, c.hidden, n, GgmlType::F32).unwrap();
    batched.begin_batch(start, n).unwrap();
    assert!(batched.validate_snapshot_position(start).is_err());
    for id in 0..schedule.layers {
        execute(ctx, |enc| {
            metal::dispatch_census_begin();
            batched
                .encode_rows(ctx, enc, id, start, &weights[id], &inputs[id], &rows)
                .unwrap();
            let census = metal::dispatch_census_take();
            for suffix in ["score_rows", "softmax_rows"] {
                assert_eq!(
                    census
                        .iter()
                        .filter(|d| d.kernel == format!("kernel_deepseek_v41_attention_{suffix}"))
                        .count(),
                    sink_spans(c, schedule.ratio(id), start, n).len()
                );
            }
            assert!(
                !census
                    .iter()
                    .any(|d| d.kernel == "kernel_deepseek_v41_attention_score")
            );
            if c.heads == 64 && c.head_dim == 512 {
                assert_eq!(
                    census
                        .iter()
                        .filter(|d| d.kernel == "kernel_latent_grouped_f32_f32_m16n64k32")
                        .count(),
                    sink_spans(c, schedule.ratio(id), start, n)
                        .iter()
                        .filter(|&&(_, queries, matrix)| matrix && queries >= 8)
                        .count()
                );
            }
            batched
                .finish_rows(ctx, enc, &weights[id], &rows, &output)
                .unwrap();
        });
        close(
            f32_slice(&output).unwrap(),
            f32_slice(&expected[id][0]).unwrap(),
        );
        for row in 0..n {
            let count = batched.selected_count(id, start + row);
            assert_eq!(
                ids(&prefill::row(&rows.selected, c.index_topk, row), count),
                ids(&prefill::row(&expected[id][1], c.index_topk, row), count),
                "IDs layer={id} row={row} n={n} start={start}"
            );
            if schedule
                .candidate_source_layer
                .is_some_and(|source| id >= source)
            {
                let visible = (start + row + 1).div_ceil(c.candidate_block);
                assert_eq!(
                    ids(&prefill::row(&rows.candidates, blocks, row), visible),
                    ids(&prefill::row(&expected[id][2], blocks, row), visible),
                    "mask layer={id} row={row}"
                );
            }
        }
    }
    batched.validate_snapshot_position(start + n).unwrap();
    for ((a, ac), (b, bc)) in batched
        .causal_regions(start + n)
        .unwrap()
        .into_iter()
        .zip(scalar.causal_regions(start + n).unwrap())
    {
        assert_eq!(ac, bc);
        close(&f32_slice(a).unwrap()[..ac], &f32_slice(b).unwrap()[..bc]);
    }
    assert_eq!(ids(batched.status(), 1), [0]);
    assert_eq!(ids(scalar.status(), 1), [0]);
    // The first decode following a batch must use unchanged persistent state.
    for id in 0..schedule.layers {
        let input = prefill::row(&inputs[id], c.hidden, n - 1);
        execute(ctx, |enc| {
            scalar
                .encode(ctx, enc, id, start + n, &weights[id], &input)
                .unwrap();
            batched
                .encode(ctx, enc, id, start + n, &weights[id], &input)
                .unwrap();
        });
        close(
            f32_slice(&batched.output).unwrap(),
            f32_slice(&scalar.output).unwrap(),
        );
        let count = batched.selected_count(id, start + n);
        assert_eq!(ids(&batched.selected, count), ids(&scalar.selected, count));
    }
}

#[test]
fn query_batch_normalized_inputs_awkward_rows_wrap_pair_phase_and_followers() {
    let ctx = MetalContext::new().unwrap();
    let c = config();
    let schedule = Schedule {
        layers: 7,
        kv_source_layers: vec![1, 3],
        index_source_layers: vec![1, 3, 5],
        compression_ratios: vec![0, 2, 2, 1, 1, 1, 1],
        candidate_source_layer: Some(3),
    };
    for n in [1, 2, 31, 32, 33, 127, 128, 129, 255, 256] {
        compare_case(&ctx, &c, &schedule, 129, n);
    }
    compare_case(&ctx, &c, &schedule, 0, 33);
    let nonuniform = Schedule {
        layers: 7,
        kv_source_layers: vec![0, 2, 5],
        index_source_layers: vec![0, 2, 4, 5],
        compression_ratios: vec![1; 7],
        candidate_source_layer: Some(0),
    };
    compare_case(&ctx, &c, &nonuniform, 3, 129);
}

#[test]
fn query_batch_f32_mma_ring_pair_followers_and_decode_continuation() {
    let ctx = MetalContext::new().unwrap();
    let c = Config {
        heads: 64,
        head_dim: 512,
        index_topk: 512,
        ..config()
    };
    let schedule = Schedule {
        layers: 2,
        kv_source_layers: vec![0],
        index_source_layers: vec![0],
        compression_ratios: vec![2, 2],
        candidate_source_layer: None,
    };
    compare_case(&ctx, &c, &schedule, 257, 33);
}

#[test]
fn query_batch_per_query_publication_and_pruning_at_16385() {
    let ctx = MetalContext::new().unwrap();
    let mut c = config();
    c.index_topk = 512;
    c.candidate_block = 8;
    c.candidate_topk = 2048;
    let schedule = Schedule {
        layers: 4,
        kv_source_layers: vec![0, 2],
        index_source_layers: vec![0, 2, 3],
        compression_ratios: vec![1; 4],
        candidate_source_layer: Some(0),
    };
    compare_case(&ctx, &c, &schedule, 16383, 2);
}

#[test]
fn query_batch_pruned_follower_excludes_otherwise_winning_block() {
    let ctx = MetalContext::new().unwrap();
    let c = Config {
        index_heads: 1,
        index_topk: 512,
        candidate_block: 8,
        candidate_topk: 2048,
        rope_dim: 0,
        ..config()
    };
    let schedule = Schedule {
        layers: 4,
        kv_source_layers: vec![0, 2],
        index_source_layers: vec![0, 2, 3],
        compression_ratios: vec![1; 4],
        candidate_source_layer: Some(0),
    };
    let (start, n, capacity) = (16383usize, 2, 16385usize);
    let mut weights = bound(&ctx, &c, &schedule);
    let channel = |w: &mut Linear<'_>, output: usize| {
        let mut data = vec![0.0; w.input * w.output];
        data[output * w.input] = 1.0;
        w.tensor = upload(&ctx, &data, w.tensor.shape.clone()).unwrap();
    };
    for (id, w) in weights.iter_mut().enumerate() {
        channel(&mut w.query_a, 0);
        if let Some(index) = &mut w.index {
            channel(&mut index.query, usize::from(id != 0));
            channel(&mut index.weights, 0);
        }
        if let Some(compressor) = &mut w.compressor {
            compressor.kv.tensor = upload(
                &ctx,
                &vec![0.0; compressor.kv.input * compressor.kv.output],
                compressor.kv.tensor.shape.clone(),
            )
            .unwrap();
        }
    }
    let mut workspace = Workspace::new_with_schedule(
        &ctx,
        &c,
        capacity,
        &weights.iter().collect::<Vec<_>>(),
        &schedule,
    )
    .unwrap();
    seed(&ctx, &mut workspace, start);
    // Source channel 0 uniquely drops block 0 when the 2049th block appears.
    // Follower channel 1 ranks IDs 0..511 highest without that mask, versus
    // 8..519 with it. Newly published zero keys cannot win either ranking.
    let mut keys = vec![0.0; capacity * c.index_dim];
    for id in 0..start {
        keys[id * c.index_dim] = if id < c.candidate_block { 0.0 } else { 1.0 };
        keys[id * c.index_dim + 1] = (capacity - id) as f32;
    }
    for owner in &mut workspace.owners {
        owner.keys = upload(&ctx, &keys, owner.keys.shape.clone()).unwrap();
    }
    let mut input = vec![0.0; n * c.hidden];
    for row in 0..n {
        input[row * c.hidden] = (c.hidden as f32).sqrt();
    }
    let input = upload(&ctx, &input, vec![c.hidden as u64, n as u64]).unwrap();
    let rows = Rows::new(&ctx, &c, capacity, n).unwrap();
    let blocks = capacity.div_ceil(c.candidate_block);
    workspace.begin_batch(start, n).unwrap();
    for (id, w) in weights.iter().enumerate() {
        execute(&ctx, |enc| {
            workspace
                .encode_rows(&ctx, enc, id, start, w, &input, &rows)
                .unwrap()
        });
        assert_eq!(
            ids(&prefill::row(&rows.candidates, blocks, 0), 2048),
            vec![1; 2048]
        );
        let mask = ids(&prefill::row(&rows.candidates, blocks, 1), 2049);
        assert_eq!(mask[0], 0);
        assert!(mask[1..].iter().all(|&keep| keep == 1));
        if id >= 2 {
            assert_eq!(
                ids(&prefill::row(&rows.selected, c.index_topk, 0), c.index_topk),
                (0..512).collect::<Vec<_>>()
            );
            assert_eq!(
                ids(&prefill::row(&rows.selected, c.index_topk, 1), c.index_topk),
                (8..520).collect::<Vec<_>>()
            );
        }
    }
    // Same final follower scores, same selector, deliberately without pruning:
    // block 0 really does contain winners; agreement is not vacuous.
    execute(&ctx, |enc| {
        workspace
            .sort(&ctx, enc, &workspace.index_scores, capacity, false)
            .unwrap()
    });
    let unmasked = ids(&workspace.sort_ids, c.index_topk);
    assert_eq!(unmasked, (0..512).collect::<Vec<_>>());
    let mask = ids(&prefill::row(&rows.candidates, blocks, 1), blocks);
    assert!(unmasked.iter().any(|&id| mask[id / c.candidate_block] == 0));
    assert_eq!(ids(workspace.status(), 1), [0]);
}

#[test]
fn query_batch_error_restores_all_six_temporary_binding_identities() {
    use objc2::rc::Retained;
    let ctx = MetalContext::new().unwrap();
    let c = config();
    let schedule = Schedule {
        layers: 1,
        kv_source_layers: vec![0],
        index_source_layers: vec![0],
        compression_ratios: vec![2],
        candidate_source_layer: None,
    };
    let weights = bound(&ctx, &c, &schedule);
    let mut workspace =
        Workspace::new_with_schedule(&ctx, &c, 8, &[&weights[0]], &schedule).unwrap();
    for (i, binding) in [
        &mut workspace.selected,
        &mut workspace.candidates,
        &mut workspace.latent,
        &mut workspace.gate,
        &mut workspace.index_query,
        &mut workspace.index_weights,
    ]
    .into_iter()
    .enumerate()
    {
        let offset = i + 1;
        let backing = tensor(&ctx, binding.n_elements() as usize + offset, binding.dtype).unwrap();
        *binding = backing.view_subrange(offset as u64, binding.shape.clone());
    }
    let bindings = |w: &Workspace| {
        [
            w.selected.clone(),
            w.candidates.clone(),
            w.latent.clone(),
            w.gate.clone(),
            w.index_query.clone(),
            w.index_weights.clone(),
        ]
    };
    let before = bindings(&workspace);
    let rows = Rows::new(&ctx, &c, 8, 2).unwrap();
    workspace.begin_batch(0, 2).unwrap();
    let invalid_input = tensor(&ctx, c.hidden - 1, GgmlType::F32).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    // Row order passes. valid_view fails inside encode_inner, after all six
    // replacements, rather than at encode_batch_row's pre-replacement guard.
    metal::dispatch_census_begin();
    let error = workspace
        .encode_batch_row(&ctx, &enc, 0, 0, &weights[0], &invalid_input, &rows, 0)
        .unwrap_err();
    assert_eq!(error, "GPU attention buffer bounds/dtype");
    assert!(metal::dispatch_census_take().is_empty());
    enc.end(); // No submission is needed to witness an encode-time error.
    for (i, (actual, expected)) in bindings(&workspace).iter().zip(&before).enumerate() {
        assert_eq!(
            Retained::as_ptr(&actual.buffer),
            Retained::as_ptr(&expected.buffer),
            "binding {i} buffer"
        );
        assert_eq!(actual.offset, expected.offset, "binding {i} offset");
        assert_eq!(actual.shape, expected.shape, "binding {i} shape");
        assert_eq!(actual.dtype, expected.dtype, "binding {i} dtype");
    }
    let cursor = workspace.batch.as_ref().unwrap();
    assert_eq!(
        (cursor.start, cursor.rows, cursor.layer, cursor.row),
        (0, 2, 0, 0)
    );
    assert_eq!(ids(workspace.status(), 1), [0]);
}

#[test]
fn query_batch_memory_pricing_and_rope_canaries() {
    let ctx = MetalContext::new().unwrap();
    let c = config();
    for n in [1, 2, 31, 32, 33, 127, 128, 129, 255, 256] {
        let sizes = Scratch::shapes(&c, n).unwrap();
        let priced = Rows::tile_widths(&c, n).unwrap();
        assert_eq!(sizes.len(), priced.len());
        for (&actual, &width) in sizes.iter().zip(&priced) {
            assert!(width * n >= actual && width * n - actual < n);
            assert!(
                ctx.price_shared_buffer_upper((width * n * 4) as u64)
                    .unwrap()
                    .priced_upper_bytes
                    >= (actual * 4) as u64
            );
        }
        assert_eq!(sizes[0], (c.window + n) * c.head_dim);
        assert_eq!(
            sizes[1],
            n.min(TILE) * c.heads * (c.window + c.index_topk + 1)
        );
        for start in [0usize, 3, 129] {
            let dim = 7;
            let ring_data = values(c.window * dim, 17);
            let new_data = values(n * dim, 23);
            let old = start.min(c.window);
            let (ring_backing, ring) = guarded_dispatch_tensor(&ctx, &ring_data);
            let (new_backing, new_rows) = guarded_dispatch_tensor(&ctx, &new_data);
            let (history_backing, history) =
                guarded_dispatch_tensor(&ctx, &vec![DISPATCH_CANARY; (c.window + n) * dim]);
            execute(&ctx, |enc| {
                let pso = pipeline(&ctx, enc, "history").unwrap();
                enc.set_bytes(0, &[dim as u32, c.window as u32, start as u32, n as u32]);
                enc.set_tensor(1, &ring);
                enc.set_tensor(2, &new_rows);
                enc.set_tensor(3, &history);
                dispatch(enc, &pso, (old + n) * dim);
                let pso = pipeline(&ctx, enc, "publish_raw").unwrap();
                dispatch(enc, &pso, n.min(c.window) * dim);
            });
            let mut expected_history = vec![DISPATCH_CANARY; (c.window + n) * dim];
            for row in 0..old {
                let slot = (start - old + row) % c.window;
                expected_history[row * dim..(row + 1) * dim]
                    .copy_from_slice(&ring_data[slot * dim..(slot + 1) * dim]);
            }
            expected_history[old * dim..(old + n) * dim].copy_from_slice(&new_data);
            let mut expected_ring = ring_data;
            for row in 0..n {
                let slot = (start + row) % c.window;
                expected_ring[slot * dim..(slot + 1) * dim]
                    .copy_from_slice(&new_data[row * dim..(row + 1) * dim]);
            }
            assert_dispatch_contents(&history_backing, &expected_history);
            assert_dispatch_contents(&ring_backing, &expected_ring);
            assert_dispatch_contents(&new_backing, &new_data);
        }
    }
    assert!(Scratch::shapes(&c, 0).is_err());
    assert!(Scratch::shapes(&c, 1025).is_err());
    for (dim, heads, n, pairs) in [(7, 3, 1, 2), (32, 8, 31, 4), (32, 8, 32, 0)] {
        let v = values(dim * heads * n, 37);
        let (backing, actual) = guarded_dispatch_tensor(&ctx, &v);
        let expected = upload(&ctx, &v, vec![(dim * heads * n) as u64]).unwrap();
        let freq = upload(&ctx, &vec![0.07; pairs.max(1)], vec![pairs.max(1) as u64]).unwrap();
        execute(&ctx, |enc| {
            rope_rows(
                &ctx, enc, &actual, &actual, &freq, dim, heads, pairs, 129, n, false,
            )
            .unwrap();
            for row in 0..n {
                rope(
                    &ctx,
                    enc,
                    &prefill::row(&expected, dim * heads, row),
                    &freq,
                    dim,
                    heads,
                    pairs,
                    129 + row,
                    false,
                )
                .unwrap();
            }
        });
        assert_dispatch_contents(&backing, f32_slice(&expected).unwrap());
    }
}

fn guard(ctx: &MetalContext, tensor: &mut MetalTensor, data: &[f32]) -> MetalTensor {
    assert_eq!(tensor.n_elements() as usize, data.len());
    let (backing, view) = guarded_dispatch_tensor(ctx, data);
    *tensor = view.view_bytes(0, tensor.shape.clone());
    backing
}

#[test]
fn query_batch_hc_projected_rms_same_reduction_and_one_launch() {
    let ctx = MetalContext::new().unwrap();
    for n in [1, 2, 31, 32, 33, 127, 128, 129, 255, 256] {
        let width = 128;
        let input_data = values(width * n, 17);
        let projected = values(24 * n, 19);
        let (input_backing, input) = guarded_dispatch_tensor(&ctx, &input_data);
        let (output_backing, output) = guarded_dispatch_tensor(&ctx, &projected);
        let scalar = upload(&ctx, &projected, vec![24, n as u64]).unwrap();
        execute(&ctx, |enc| {
            enc.set_pipeline(&ctx.pipeline("kernel_deepseek_v41_projected_rms").unwrap());
            enc.set_bytes(0, &(width as u32));
            enc.set_bytes(1, &1e-6f32);
            enc.set_tensor(2, &input);
            enc.set_tensor(3, &output);
            metal::dispatch_census_begin();
            enc.dispatch(
                MTLSize {
                    width: n,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            let census = metal::dispatch_census_take();
            assert_eq!(census.len(), 1);
            assert_eq!(census[0].grid_width, n as u64);
            assert_eq!(census[0].kernel, "kernel_deepseek_v41_projected_rms");
            for row in 0..n {
                enc.set_tensor(2, &prefill::row(&input, width, row));
                enc.set_tensor(3, &prefill::row(&scalar, 24, row));
                enc.dispatch(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        });
        assert_dispatch_contents(&input_backing, &input_data);
        assert_dispatch_contents(&output_backing, f32_slice(&scalar).unwrap());
    }
}

fn sink_case(
    ctx: &MetalContext,
    c: &Config,
    start: usize,
    n: usize,
    ratio: usize,
) -> (Vec<u64>, Vec<u64>) {
    let schedule = Schedule {
        layers: 1,
        kv_source_layers: vec![0],
        index_source_layers: vec![0],
        compression_ratios: vec![ratio],
        candidate_source_layer: (ratio == 1).then_some(0),
    };
    let weights = bound(ctx, c, &schedule);
    let capacity = start + n;
    let mut workspace =
        Workspace::new_with_schedule(ctx, c, capacity, &[&weights[0]], &schedule).unwrap();
    let mut rows = Rows::new(ctx, c, capacity, n).unwrap();
    let width = c.heads * c.head_dim;
    let queries = values(n * width, 37);
    let mut history = values((c.window + n) * c.head_dim, 19);
    for row in history.chunks_exact_mut(c.head_dim) {
        row[0] = -0.0;
    }
    let mut compressed = values(capacity.div_ceil(ratio) * c.head_dim, 23);
    for row in compressed.chunks_exact_mut(c.head_dim) {
        row[0] = -0.0;
    }
    workspace.owners[0].compressed =
        upload(ctx, &compressed, vec![compressed.len() as u64]).unwrap();
    let mut guards = vec![
        guard(ctx, &mut rows.query, &queries),
        guard(ctx, &mut rows.tile.history, &history),
    ];
    for t in [&mut rows.tile.scores, &mut rows.tile.attended] {
        let data = vec![f32::NAN; t.n_elements() as usize];
        guards.push(guard(ctx, t, &data));
    }
    if let Some(matrix) = &mut rows.tile.matrix {
        for t in [&mut matrix.values, &mut matrix.probabilities] {
            let data = vec![f32::NAN; t.n_elements() as usize];
            guards.push(guard(ctx, t, &data));
        }
    }
    let selections: Vec<u32> = (0..n)
        .flat_map(|row| {
            let length = (start + row + 1) / ratio;
            let count = length.min(c.index_topk);
            (0..c.index_topk).map(move |i| {
                if i < count {
                    ((length - 1 - i + row * 7) % length) as u32
                } else {
                    u32::MAX
                }
            })
        })
        .collect();
    let selected = |ids: &[u32]| {
        MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(ids),
            vec![c.index_topk as u64, n as u64],
            GgmlType::I32,
        )
        .unwrap()
    };
    rows.selected = selected(&selections);
    let original = upload(ctx, &queries, vec![width as u64, n as u64]).unwrap();
    let expected = prefill::allocate(ctx, width, n, GgmlType::F32).unwrap();
    execute(ctx, |enc| {
        for row in 0..n {
            let position = start + row;
            let nr = (position + 1).min(c.window);
            let length = (position + 1) / ratio;
            let nk = length.min(c.index_topk);
            enc.set_bytes(
                0,
                &[
                    c.head_dim as u32,
                    nr as u32,
                    (position + 1 - nr - (start - start.min(c.window))) as u32,
                    (start.min(c.window) + n) as u32,
                ],
            );
            enc.set_bytes(1, &[nk as u32, (c.window + c.index_topk + 1) as u32]);
            enc.set_tensor(2, &prefill::row(&original, width, row));
            enc.set_tensor(3, &rows.tile.history);
            enc.set_tensor(4, &workspace.owners[0].compressed);
            enc.set_tensor(5, &prefill::row(&rows.selected, c.index_topk, row));
            enc.set_tensor(6, &workspace.constants[0].sink);
            enc.set_tensor(7, &workspace.attention_scores);
            enc.set_tensor(8, &workspace.attended);
            enc.set_bytes(9, &(length as u32));
            enc.set_tensor(10, &workspace.status);
            sink_passes(
                ctx,
                enc,
                c.heads,
                c.head_dim,
                nr + nk,
                workspace
                    .value_matrix
                    .as_ref()
                    .map(|m| (m, &workspace.attended, &workspace.status)),
            )
            .unwrap();
            rope(
                ctx,
                enc,
                &workspace.attended,
                &workspace.constants[0].frequencies,
                c.head_dim,
                c.heads,
                c.rope_dim / 2,
                position,
                true,
            )
            .unwrap();
            copy(
                ctx,
                enc,
                &workspace.attended,
                &prefill::row(&expected, width, row),
                width,
            )
            .unwrap();
        }
    });
    let spans = sink_spans(c, ratio, start, n);
    let run = |rows: &Rows| {
        let mut depths = (Vec::new(), Vec::new());
        execute(ctx, |enc| {
            copy(ctx, enc, &original, &rows.query, n * width).unwrap();
            metal::dispatch_census_begin();
            for first in (0..n).step_by(TILE) {
                workspace
                    .sink_tile(ctx, enc, 0, start, rows, first, (n - first).min(TILE))
                    .unwrap();
            }
            let census = metal::dispatch_census_take();
            let count = |suffix: &str| {
                census
                    .iter()
                    .filter(|d| d.kernel == format!("kernel_deepseek_v41_attention_{suffix}"))
                    .count()
            };
            assert_eq!(count("score_rows"), spans.len());
            assert_eq!(count("softmax_rows"), spans.len());
            assert_eq!(count("rope_rows"), spans.len());
            let grouped = census
                .iter()
                .filter(|d| d.kernel.contains("grouped") && d.kernel.contains("f32"))
                .collect::<Vec<_>>();
            assert_eq!(grouped.len(), count("matrix_values_rows"));
            assert_eq!(count("finite"), grouped.len());
            assert_eq!(count("output_rows") + grouped.len(), spans.len());
            depths.0 = census
                .iter()
                .filter(|d| d.kernel == "kernel_deepseek_v41_attention_score_rows")
                .map(|d| d.grid_depth)
                .collect();
            depths.1 = grouped.iter().map(|d| d.grid_depth).collect();
            assert_eq!(
                depths.0,
                spans.iter().map(|&(_, n, _)| n as u64).collect::<Vec<_>>()
            );
            assert_eq!(
                depths.1,
                spans
                    .iter()
                    .filter(|&&(_, _, matrix)| matrix)
                    .map(|&(_, n, _)| n as u64)
                    .collect::<Vec<_>>()
            );
            for (dispatch, &(_, queries, _)) in grouped.iter().zip(spans.iter().filter(|s| s.2)) {
                assert!(dispatch.grid_depth <= TILE as u64);
                let mma = queries >= 8
                    && c.head_dim.is_multiple_of(16)
                    && c.heads >= 64
                    && (c.head_dim / 16) * (c.heads / 64) * queries >= 256;
                assert_eq!(
                    dispatch.kernel,
                    if mma {
                        "kernel_latent_grouped_f32_f32_m16n64k32"
                    } else {
                        "kernel_latent_grouped_f32_f32"
                    }
                );
                assert_eq!(
                    (dispatch.grid_width, dispatch.grid_height),
                    if mma {
                        ((c.heads / 64) as u64, (c.head_dim / 16) as u64)
                    } else {
                        (c.head_dim.div_ceil(4) as u64, c.heads.div_ceil(4) as u64)
                    }
                );
            }
        });
        depths
    };
    let depths = run(&rows);
    close(
        f32_slice(&rows.query).unwrap(),
        f32_slice(&expected).unwrap(),
    );
    assert_eq!(ids(&workspace.status, 1), [0]);
    let &(first, tile, matrix) = spans.last().unwrap();
    if let Some(matrix) = rows.tile.matrix.as_ref().filter(|_| matrix) {
        let padded =
            ((start + n).min(c.window) + ((start + n) / ratio).min(c.index_topk)).div_ceil(32) * 32;
        let v = f32_slice(&matrix.values).unwrap();
        let p = f32_slice(&matrix.probabilities).unwrap();
        for q in 0..tile {
            let row = first + q;
            let position = start + row;
            let nr = (position + 1).min(c.window);
            let nk = ((position + 1) / ratio).min(c.index_topk);
            for j in 0..c.head_dim {
                for i in 0..padded {
                    let expected = if i < nr {
                        history[(position + 1 - nr - (start - start.min(c.window)) + i)
                            * c.head_dim
                            + j]
                    } else if i < nr + nk {
                        compressed
                            [selections[row * c.index_topk + i - nr] as usize * c.head_dim + j]
                    } else {
                        0.0
                    };
                    assert_eq!(
                        v[(q * c.head_dim + j) * padded + i].to_bits(),
                        expected.to_bits()
                    );
                }
            }
            for head in 0..c.heads {
                for i in nr + nk..padded {
                    assert_eq!(
                        p[(head * tile + q) * padded + i].to_bits(),
                        0.0f32.to_bits()
                    );
                }
            }
        }
    }
    // For multi-row cases the injected entry exists in the completed source
    // cache, but is not yet visible to this query. A final-length check is wrong.
    let bad_row = if n > 2 { 1 } else { 0 };
    let visible = (start + bad_row + 1) / ratio;
    let populated = (start + n) / ratio;
    if n > 1 {
        assert!(visible < populated);
    }
    for row in f32_slice(&expected).unwrap().chunks_exact(width) {
        assert!(row.iter().any(|&value| value != 0.0));
    }
    let mut bad = selections.clone();
    bad[bad_row * c.index_topk] = if n > 1 { visible } else { capacity } as u32;
    rows.selected = selected(&bad);
    run(&rows);
    assert_eq!(ids(&workspace.status, 1), [2]);
    assert!(
        f32_slice(&rows.query).unwrap()[bad_row * width..(bad_row + 1) * width]
            .iter()
            .all(|&v| v == 0.0)
    );
    close(
        &f32_slice(&rows.query).unwrap()[..bad_row * width],
        &f32_slice(&expected).unwrap()[..bad_row * width],
    );
    close(
        &f32_slice(&rows.query).unwrap()[(bad_row + 1) * width..],
        &f32_slice(&expected).unwrap()[(bad_row + 1) * width..],
    );
    rows.selected = selected(&selections);
    run(&rows);
    assert_eq!(ids(&workspace.status, 1), [2]);
    let mut bad_history = history.clone();
    bad_history[c.head_dim] = f32::NAN;
    let poisoned = upload(ctx, &bad_history, vec![bad_history.len() as u64]).unwrap();
    execute(ctx, |enc| {
        copy(ctx, enc, &poisoned, &rows.tile.history, bad_history.len()).unwrap()
    });
    run(&rows);
    assert_eq!(ids(&workspace.status, 1), [6]);
    for backing in &guards {
        let data = f32_slice(backing).unwrap();
        for &v in data[..DISPATCH_GUARD]
            .iter()
            .chain(&data[data.len() - DISPATCH_GUARD..])
        {
            assert_eq!(v.to_bits(), DISPATCH_CANARY.to_bits());
        }
    }
    depths
}

#[test]
fn query_batch_group_specific_pv_padding_canaries_and_sticky_errors() {
    let ctx = MetalContext::new().unwrap();
    let mut c = config();
    c.index_topk = 133;
    for n in [1, 2, 31, 32, 33] {
        sink_case(&ctx, &c, 129, n, 1);
    }
    c.index_topk = 128;
    sink_case(&ctx, &c, 126, 33, 1);
    c.heads = 2;
    c.index_topk = 4;
    sink_case(&ctx, &c, 129, 33, 1); // Unsupported grouped geometry remains tiled.
    c.heads = 64;
    c.head_dim = 512;
    c.index_topk = 512;
    for n in [1, 7, 8, 32, 33] {
        sink_case(&ctx, &c, 513, n, 1);
    }
    // Inverse RoPE, moving raw window and pair visibility on the production MMA
    // branch, followed by a one-query scalar tail. No tolerance changes.
    sink_case(&ctx, &c, 257, 33, 2);
}

#[test]
fn query_batch_native_mixed_threshold_preserves_pv_branch_and_split_census() {
    let ctx = MetalContext::new().unwrap();
    let c = Config {
        heads: 64,
        head_dim: 512,
        index_topk: 512,
        ..config()
    };
    assert_eq!(sink_spans(&c, 1, 126, 32), [(0, 1, false), (1, 31, true)]);
    let (score, pv) = sink_case(&ctx, &c, 126, 32, 1);
    assert_eq!(score, [1, 31]);
    assert_eq!(pv, [31]);
    let (score, pv) = sink_case(&ctx, &c, 126, 33, 1);
    assert_eq!(score, [1, 31, 1]);
    assert_eq!(pv, [31, 1]);
}

#[test]
fn query_batch_threshold_split_source_census_expectation() {
    let c = Config::default();
    let schedule = Schedule::default();
    let mut tiles = 0;
    let mut splits = 0;
    for start in (0..2048).step_by(256) {
        for id in 0..schedule.layers {
            let spans = sink_spans(&c, schedule.ratio(id), start, 256);
            assert!((8..=9).contains(&spans.len()));
            tiles += spans.len();
            splits += spans.len() - 8;
        }
    }
    // Source expectation only: each of the 38 compressed-attention layers
    // crosses the threshold once, inside a fixed tile of the first chunk.
    assert_eq!((tiles, splits), (2598, 38));
}

#[test]
fn query_batch_rejects_populated_but_not_yet_visible_ids() {
    let ctx = MetalContext::new().unwrap();
    for heads in [2, 8] {
        let c = Config {
            heads,
            index_topk: 133,
            ..config()
        };
        sink_case(&ctx, &c, 129, 33, 1);
        // Query 1 is position 258: its pair is incomplete, so ID 129 is
        // forbidden there even though query 2 and the completed source see it.
        let (start, bad_row, rows, ratio) = (257usize, 1, 33, 2);
        assert_eq!((start + bad_row + 1) % ratio, 1);
        assert_eq!((start + bad_row + 1) / ratio, 129);
        assert!((start + bad_row + 1) / ratio < (start + rows) / ratio);
        sink_case(&ctx, &c, start, rows, ratio);
    }
}
