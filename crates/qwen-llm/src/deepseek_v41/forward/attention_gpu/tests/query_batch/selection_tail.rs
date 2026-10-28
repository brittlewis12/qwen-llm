use super::*;
use crate::deepseek_v41::forward::attention_gpu::query_batch::selection_shapes;

const LIMIT: u64 = 2 << 30;
const CPU_RESERVE: u64 = 512 << 20;
const METAL_RESERVE: u64 = 256 << 20;

fn buffer_bytes(c: &Config, s: &Schedule, capacity: usize, n: usize) -> Vec<u64> {
    let mut bytes = Vec::new();
    // Two workspaces, plus old backing and upload staging during seed replacement.
    for _ in 0..4 {
        bytes.extend(
            shapes(c, capacity, s)
                .unwrap()
                .into_iter()
                .map(|v| v as u64 * 4),
        );
    }
    for _ in 0..2 {
        bytes.extend(
            Rows::widths(c, capacity)
                .into_iter()
                .map(|v| (v * n) as u64 * 4),
        );
        bytes.extend(
            Scratch::shapes(c, n)
                .unwrap()
                .into_iter()
                .map(|v| v as u64 * 4),
        );
        bytes.extend(
            selection_shapes(c, capacity, n, TILE)
                .unwrap()
                .into_iter()
                .filter(|&v| v > 0)
                .map(|v| v as u64 * 4),
        );
        bytes.push(prefill::attention_half_scratch_elements(c, n).unwrap() as u64 * 2);
    }
    // Shared input, two complete output buffers, and input upload staging.
    bytes.extend([c.hidden * n; 4].map(|v| v as u64 * 4));
    for id in 0..s.layers {
        let mut elements = vec![
            c.hidden * c.query_rank,
            c.query_rank * c.heads * c.head_dim,
            c.hidden * c.head_dim,
            c.heads * c.head_dim * c.output_rank,
            c.groups * c.output_rank * c.hidden,
        ];
        if s.owns_kv(id) {
            elements.push(c.hidden * c.head_dim);
            if s.ratio(id) == 2 {
                elements.push(c.hidden * c.head_dim);
            }
        }
        if s.produces_index(id) {
            elements.extend([
                c.query_rank * c.index_heads * c.index_dim,
                c.hidden * c.index_heads,
            ]);
            if s.owns_kv(id) {
                elements.push(c.head_dim * c.index_dim);
            }
        }
        for v in elements {
            bytes.extend([v as u64 * 4; 2]); // F32 weights and upload staging.
            bytes.push(4096); // Per-linear native control allowance.
        }
    }
    bytes.retain(|&v| v > 0);
    bytes
}

fn assert_bits(a: &MetalTensor, b: &MetalTensor, count: usize) {
    assert_eq!(a.dtype, b.dtype);
    if a.dtype == GgmlType::I32 {
        assert_eq!(ids(a, count), ids(b, count));
    } else {
        assert_eq!(a.dtype, GgmlType::F32);
        let a: &[u32] = bytemuck::cast_slice(&f32_slice(a).unwrap()[..count]);
        let b: &[u32] = bytemuck::cast_slice(&f32_slice(b).unwrap()[..count]);
        if let Some(i) = a.iter().zip(b).position(|(a, b)| a != b) {
            assert_eq!(a[i], b[i], "F32 bits at element {i} of {count}");
        }
    }
}

fn assert_state(a: &Workspace, b: &Workspace, position: usize) {
    assert_eq!(a.batch, b.batch);
    assert_eq!(
        (a.next_layer, a.next_position),
        (b.next_layer, b.next_position)
    );
    assert!(a.ced.is_none() && b.ced.is_none());
    assert_eq!(ids(a.status(), 1), [0]);
    assert_eq!(ids(b.status(), 1), [0]);
    let a = a.causal_regions(position).unwrap();
    let b = b.causal_regions(position).unwrap();
    assert_eq!(a.len(), b.len());
    for ((a, ac), (b, bc)) in a.into_iter().zip(b) {
        assert_eq!(ac, bc);
        assert_bits(a, b, ac);
    }
}

#[test]
#[ignore = "self-leased GPU correctness; MTL_DEBUG_LAYER=1; shape-priced <=2 GiB CPU+Metal, not measured peak; no model or timing claim"]
fn long_selection_short_tails_match_prepared_per_row_and_continue() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    for (key, _) in std::env::vars_os() {
        let key = key.to_string_lossy();
        assert!(
            !key.starts_with("QWEN_V41_SELECT_")
                && !key.starts_with("QWEN_V41_RADIX")
                && !key.starts_with("QWEN_V41_COMPRESS_")
                && key != "QWEN_V41_INDEX_TENSOR"
                && key != "QWEN_V41_SINK_TILE",
            "remove route override {key}"
        );
    }
    let _lease = metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let c = Config {
        index_heads: 32,
        index_dim: 128,
        index_topk: 512,
        candidate_block: 8,
        candidate_topk: 2048,
        rope_dim: 0,
        ..config()
    };
    let s = Schedule {
        layers: 3,
        kv_source_layers: vec![0, 1],
        index_source_layers: vec![0, 1, 2],
        compression_ratios: vec![2, 1, 1],
        candidate_source_layer: Some(1),
    };
    s.validate().unwrap();
    assert_eq!(s.owner_slot(1), s.owner_slot(2));
    for (start, n, fits) in [
        (32768usize, 2usize, [true, false, false]),
        (32769, 2, [true, false, false]),
        (32768, 3, [true, true, true]),
        (65536, 2, [false, false, false]),
        (65536, 5, [true, true, true]),
    ] {
        objc2::rc::autoreleasepool(|_| {
            let capacity = start + n + 1;
            let sizes = shapes(&c, capacity, &s).unwrap();
            // Overbound two full CPU snapshots plus the largest seed upload Vec.
            let snapshots = 2
                * (sizes.iter().sum::<usize>()
                    + Rows::widths(&c, capacity).iter().sum::<usize>() * n
                    + c.hidden * n) as u64
                * 4;
            let staging = *sizes.iter().max().unwrap() as u64 * 4;
            assert!(snapshots + staging <= CPU_RESERVE);
            let priced: u64 = buffer_bytes(&c, &s, capacity, n)
                .into_iter()
                .map(|v| ctx.price_shared_buffer_upper(v).unwrap().priced_upper_bytes)
                .sum();
            assert!(priced + METAL_RESERVE + CPU_RESERVE <= LIMIT);
            assert!(
                metal::evaluate_metal_memory_admission_with_cpu_bytes(
                    priced + METAL_RESERVE,
                    CPU_RESERVE,
                    0,
                    ctx.memory_signals(),
                    true,
                )
                .is_admitted()
            );
            eprintln!(
                "selection tail correctness start={start} n={n} priced_metal={priced} metal_reserve={METAL_RESERVE} cpu_reserve={CPU_RESERVE} limit={LIMIT}; admission bound, not measured peak"
            );
            let weights = bound(&ctx, &c, &s);
            let refs = weights.iter().collect::<Vec<_>>();
            let mut reference =
                Workspace::new_with_schedule(&ctx, &c, capacity, &refs, &s).unwrap();
            let mut candidate =
                Workspace::new_with_schedule(&ctx, &c, capacity, &refs, &s).unwrap();
            seed(&ctx, &mut reference, start);
            seed(&ctx, &mut candidate, start);
            let reference_rows = Rows::new(&ctx, &c, capacity, n).unwrap();
            let candidate_rows = Rows::new(&ctx, &c, capacity, n).unwrap();
            assert_eq!(reference_rows.rows, n);
            assert_eq!(candidate_rows.rows, n);
            assert_eq!(
                reference_rows.select_scores.n_elements(),
                candidate_rows.select_scores.n_elements()
            );
            let mut input = values(c.hidden * n, 11);
            for row in input.chunks_exact_mut(c.hidden) {
                let rms =
                    (row.iter().map(|x| x * x).sum::<f32>() / c.hidden as f32 + c.norm_eps).sqrt();
                for x in row {
                    *x /= rms;
                }
            }
            let input = upload(&ctx, &input, vec![c.hidden as u64, n as u64]).unwrap();
            let reference_output = prefill::allocate(&ctx, c.hidden, n, GgmlType::F32).unwrap();
            let candidate_output = prefill::allocate(&ctx, c.hidden, n, GgmlType::F32).unwrap();
            reference.begin_batch(start, n).unwrap();
            candidate.begin_batch(start, n).unwrap();
            let blocks = capacity.div_ceil(c.candidate_block);
            let mut published = Vec::new();
            for (id, weight) in weights.iter().enumerate() {
                // Both arms use identical batched projection arithmetic, not scalar linear.
                execute(&ctx, |enc| {
                    reference
                        .encode_rows_prepare(&ctx, enc, id, start, weight, &input, &reference_rows)
                        .unwrap();
                    candidate
                        .encode_rows_prepare(&ctx, enc, id, start, weight, &input, &candidate_rows)
                        .unwrap();
                });
                for (a, b) in [
                    (&reference_rows.query, &candidate_rows.query),
                    (&reference_rows.latent, &candidate_rows.latent),
                    (
                        &reference_rows.compressor_value,
                        &candidate_rows.compressor_value,
                    ),
                    (
                        &reference_rows.compressor_gate,
                        &candidate_rows.compressor_gate,
                    ),
                    (&reference_rows.index_query, &candidate_rows.index_query),
                    (&reference_rows.index_weights, &candidate_rows.index_weights),
                ] {
                    assert_bits(a, b, a.n_elements() as usize);
                }
                let width = (start + n) / s.ratio(id);
                assert!(width > 16384, "stay outside the unpruned tensor-index path");
                let source = s.candidate_source_layer == Some(id);
                let required = width
                    + if source {
                        width.div_ceil(c.candidate_block)
                    } else {
                        0
                    };
                let available = candidate_rows.select_scores.n_elements() as usize;
                assert_eq!(available >= required, fits[id]);
                let tile = if available >= required {
                    (available / required).min(n)
                } else {
                    0
                };
                for (is_candidate, workspace, rows, output) in [
                    (false, &mut reference, &reference_rows, &reference_output),
                    (true, &mut candidate, &candidate_rows, &candidate_output),
                ] {
                    execute(&ctx, |enc| {
                        metal::dispatch_census_begin();
                        metal::allocation_census_begin();
                        if is_candidate {
                            workspace
                                .encode_rows_serial(&ctx, enc, id, start, weight, &input, rows)
                                .unwrap();
                        } else {
                            for row in 0..n {
                                workspace
                                    .encode_batch_row_mode(
                                        &ctx,
                                        enc,
                                        id,
                                        start + row,
                                        weight,
                                        &prefill::row(&input, c.hidden, row),
                                        rows,
                                        row,
                                        true,
                                        true,
                                        GlobalKvMode::Ordinary,
                                    )
                                    .unwrap();
                            }
                        }
                        let allocations = metal::allocation_census_take();
                        let census = metal::dispatch_census_take();
                        assert!(
                            allocations.is_empty(),
                            "serial must reuse existing buffers: {allocations:?}"
                        );
                        let heights = |suffix: &str| {
                            census
                                .iter()
                                .filter(|d| {
                                    d.kernel == format!("kernel_deepseek_v41_attention_{suffix}")
                                })
                                .map(|d| d.grid_height)
                                .collect::<Vec<_>>()
                        };
                        let batched = is_candidate && tile > 0;
                        let tiles = if batched {
                            (0..n)
                                .step_by(tile)
                                .map(|first| (n - first).min(tile) as u64)
                                .collect::<Vec<_>>()
                        } else {
                            vec![1; n]
                        };
                        assert_eq!(
                            heights("index_heads_rows_mma"),
                            tiles,
                            "MMA tile sizes start={start} n={n} layer={id} candidate={is_candidate}"
                        );
                        assert_eq!(
                            heights("blocks").len(),
                            if source && !batched { n } else { 0 }
                        );
                        assert_eq!(
                            heights("select_block_scores_rows"),
                            if source && batched {
                                tiles.clone()
                            } else {
                                vec![]
                            }
                        );
                        // Batched tiles keep select_radix_rows; per-row selection
                        // is the decode path's multi-threadgroup selector.
                        let radix = if !batched {
                            vec![]
                        } else if source {
                            tiles.iter().flat_map(|&h| [h, h]).collect::<Vec<_>>()
                        } else {
                            tiles.clone()
                        };
                        assert_eq!(heights("select_radix_rows"), radix);
                        let per_row = if batched { 0 } else { n };
                        assert_eq!(heights("select_radix_grid_hist"), vec![1; 3 * per_row]);
                        assert_eq!(heights("select_radix_grid_place"), vec![1; per_row]);
                        assert_eq!(
                            heights("selection_prepare_rows"),
                            if batched { tiles } else { vec![] }
                        );
                        for suffix in [
                            "index_heads",
                            "index_heads_rows",
                            "index_heads_rows_ilp",
                            "index_scores_tensor",
                        ] {
                            assert!(heights(suffix).is_empty(), "unexpected scorer {suffix}");
                        }
                        workspace
                            .encode_rows_tiles(&ctx, enc, id, start, rows)
                            .unwrap();
                        workspace
                            .finish_rows(&ctx, enc, weight, rows, output)
                            .unwrap();
                    });
                }
                assert_bits(&reference_output, &candidate_output, c.hidden * n);
                assert_bits(
                    &reference_rows.query,
                    &candidate_rows.query,
                    c.heads * c.head_dim * n,
                );
                assert_bits(
                    &reference_rows.selected,
                    &candidate_rows.selected,
                    c.index_topk * n,
                );
                if source {
                    published.clear();
                }
                for row in 0..n {
                    let length = (start + row + 1) / s.ratio(id);
                    let selected = ids(
                        &prefill::row(&candidate_rows.selected, c.index_topk, row),
                        c.index_topk,
                    );
                    assert!(selected.windows(2).all(|pair| pair[0] < pair[1]));
                    assert!(selected.iter().all(|&key| key < length));
                    if id >= s.candidate_source_layer.unwrap() {
                        let visible = length.div_ceil(c.candidate_block);
                        assert_bits(
                            &prefill::row(&reference_rows.candidates, blocks, row),
                            &prefill::row(&candidate_rows.candidates, blocks, row),
                            visible,
                        );
                        let mask = ids(
                            &prefill::row(&candidate_rows.candidates, blocks, row),
                            visible,
                        );
                        assert!(mask.iter().all(|&keep| keep <= 1));
                        assert_eq!(mask.iter().sum::<usize>(), c.candidate_topk);
                        assert_eq!(mask[visible - 1], 1);
                        if source {
                            published.push(mask);
                        } else {
                            assert_eq!(mask, published[row], "reuse source visible mask unchanged");
                            assert!(
                                selected
                                    .iter()
                                    .all(|&key| mask[key / c.candidate_block] == 1)
                            );
                        }
                    }
                }
                if id > s.candidate_source_layer.unwrap() {
                    // The per-row reference leaves the final follower's actual scores
                    // in index_scores. Re-select without its mask using existing sort
                    // scratch: agreement must not be vacuous on an already-winning set.
                    execute(&ctx, |enc| {
                        reference
                            .sort(&ctx, enc, &reference.index_scores, width, false)
                            .unwrap();
                    });
                    let unmasked = ids(&reference.sort_ids, c.index_topk);
                    assert!(
                        unmasked
                            .iter()
                            .any(|&key| published[n - 1][key / c.candidate_block] == 0),
                        "fixture needs an otherwise-winning masked-out follower key start={start} n={n}"
                    );
                }
                assert_state(&reference, &candidate, start + n);
                if id + 1 < s.layers {
                    let cursor = candidate.batch.unwrap();
                    assert_eq!(
                        (cursor.start, cursor.rows, cursor.layer, cursor.row),
                        (start, n, id + 1, 0)
                    );
                    assert_eq!((candidate.next_layer, candidate.next_position), (0, start));
                } else {
                    assert!(candidate.batch.is_none());
                    candidate.validate_snapshot_position(start + n).unwrap();
                    reference.validate_snapshot_position(start + n).unwrap();
                }
            }
            // One ordinary decode after the complete batch, on the same authoritative caches.
            let next_input = prefill::row(&input, c.hidden, n - 1);
            for (id, weight) in weights.iter().enumerate() {
                execute(&ctx, |enc| {
                    reference
                        .encode(&ctx, enc, id, start + n, weight, &next_input)
                        .unwrap();
                    candidate
                        .encode(&ctx, enc, id, start + n, weight, &next_input)
                        .unwrap();
                });
                assert_bits(&reference.output, &candidate.output, c.hidden);
                assert_bits(&reference.selected, &candidate.selected, c.index_topk);
                if id >= s.candidate_source_layer.unwrap() {
                    let visible = ((start + n + 1) / s.ratio(id)).div_ceil(c.candidate_block);
                    assert_bits(&reference.candidates, &candidate.candidates, visible);
                }
                assert_state(&reference, &candidate, capacity);
                assert!(candidate.batch.is_none());
                assert_eq!(candidate.next_layer, (id + 1) % s.layers);
                assert_eq!(
                    candidate.next_position,
                    start + n + usize::from(id + 1 == s.layers)
                );
            }
            reference.validate_snapshot_position(capacity).unwrap();
            candidate.validate_snapshot_position(capacity).unwrap();
        });
    }
}
