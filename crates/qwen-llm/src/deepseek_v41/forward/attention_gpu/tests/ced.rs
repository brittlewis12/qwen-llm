use super::super::super::ced_plan::{CedPlan, CedRequest};
use super::*;

fn plan(c: &Config, n: usize, capacity: usize) -> CedPlan {
    CedPlan::shrinking(
        c,
        &Schedule::default(),
        &engram::HashConfig {
            layers: Vec::new(),
            ..engram::HashConfig::default()
        },
        &CedRequest::cold(n, capacity),
    )
    .unwrap()
}

fn bits(t: &MetalTensor) -> Vec<u32> {
    f32_slice(t).unwrap().iter().map(|v| v.to_bits()).collect()
}

fn values(c: &Config, n: usize, layer: usize) -> Vec<f32> {
    (0..n * c.hidden)
        .map(|i| (((i * 17 + layer * 13 + i / c.hidden * 7) % 83) as f32 - 41.0) / 31.0)
        .collect()
}

fn owner_bits(w: &Workspace) -> Vec<Vec<u32>> {
    w.owners
        .iter()
        .flat_map(|o| [&o.compressed, &o.keys, &o.pending_value, &o.pending_gate])
        .map(bits)
        .collect()
}

fn assert_status(w: &Workspace) {
    assert_eq!(ids(w.status(), 1), [0]);
}

fn attention_case(ctx: &MetalContext, scalar: bool) {
    let data = fixture();
    let cpu: Vec<_> = (0..40).map(|id| cpu_weights(&data, id)).collect();
    let weights: Vec<_> = cpu.iter().map(|w| resident(ctx, w)).collect();
    let refs = weights.iter().collect::<Vec<_>>();
    let mut c = super::super::super::toy::config();
    c.hidden = 8;
    let (n, capacity) = (47, 48);
    let plan = plan(&c, n, capacity);
    let input: Vec<_> = (0..40)
        .map(|id| {
            upload(
                ctx,
                &values(&c, capacity, id),
                vec![c.hidden as u64, capacity as u64],
            )
            .unwrap()
        })
        .collect();
    let mut ordinary = Workspace::new(ctx, &c, capacity, &refs).unwrap();
    let mut ced = Workspace::new(ctx, &c, capacity, &refs).unwrap();
    let blocks = capacity.div_ceil(c.candidate_block);
    let expected: Vec<_> = (0..40)
        .map(|_| {
            [
                prefill::allocate(ctx, c.hidden, n, GgmlType::F32).unwrap(),
                prefill::allocate(ctx, c.index_topk, n, GgmlType::I32).unwrap(),
                prefill::allocate(ctx, blocks, n, GgmlType::I32).unwrap(),
            ]
        })
        .collect();
    metal::dispatch_census_begin();
    if scalar {
        execute(ctx, |enc| {
            for p in 0..n {
                for id in 0..40 {
                    ordinary
                        .encode(
                            ctx,
                            enc,
                            id,
                            p,
                            &weights[id],
                            &prefill::row(&input[id], c.hidden, p),
                        )
                        .unwrap();
                    for (source, output, width) in [
                        (&ordinary.output, &expected[id][0], c.hidden),
                        (&ordinary.selected, &expected[id][1], c.index_topk),
                        (&ordinary.candidates, &expected[id][2], blocks),
                    ] {
                        copy(ctx, enc, source, &prefill::row(output, width, p), width).unwrap();
                    }
                }
            }
        });
    } else {
        let rows = Rows::new(ctx, &c, capacity, n).unwrap();
        ordinary.begin_batch(0, n).unwrap();
        execute(ctx, |enc| {
            for id in 0..40 {
                ordinary
                    .encode_rows(
                        ctx,
                        enc,
                        id,
                        0,
                        &weights[id],
                        &input[id].view_subrange(0, vec![c.hidden as u64, n as u64]),
                        &rows,
                    )
                    .unwrap();
                ordinary
                    .finish_rows(ctx, enc, &weights[id], &rows, &expected[id][0])
                    .unwrap();
                copy(ctx, enc, &rows.selected, &expected[id][1], n * c.index_topk).unwrap();
                copy(ctx, enc, &rows.candidates, &expected[id][2], n * blocks).unwrap();
            }
        });
    }
    let ordinary_census = metal::dispatch_census_take();
    assert_eq!(
        ordinary_census
            .iter()
            .filter(|r| r.kernel == "kernel_deepseek_v41_attention_finite")
            .count(),
        if scalar { n * 40 } else { 40 },
        "ordinary publication must not add CED finite-check dispatches",
    );
    assert_status(&ordinary);
    ced.begin_ced(&plan).unwrap();
    assert!(ced.validate_snapshot_position(0).is_err());
    assert!(ced.begin_batch(0, 1).is_err());
    assert!(ced.commit_ced(n).is_err());
    if scalar {
        for p in 0..n {
            ced.begin_ced_encoder(p, 1).unwrap();
            execute(ctx, |enc| {
                for id in 0..20 {
                    ced.encode(
                        ctx,
                        enc,
                        id,
                        p,
                        &weights[id],
                        &prefill::row(&input[id], c.hidden, p),
                    )
                    .unwrap();
                }
                ced.publish_ced_scalar(
                    ctx,
                    enc,
                    20,
                    p,
                    &weights[20],
                    &prefill::row(&input[20], c.hidden, p),
                )
                .unwrap();
            });
            assert_eq!((ced.next_layer, ced.next_position), (0, 0));
            ced.complete_ced_encoder(p, 1).unwrap();
        }
    } else {
        let rows = Rows::new(ctx, &c, capacity, n).unwrap();
        let publication = PublicationRows::new(ctx, &c, n).unwrap();
        ced.begin_ced_encoder(0, n).unwrap();
        execute(ctx, |enc| {
            for id in 0..20 {
                ced.encode_rows(
                    ctx,
                    enc,
                    id,
                    0,
                    &weights[id],
                    &input[id].view_subrange(0, vec![c.hidden as u64, n as u64]),
                    &rows,
                )
                .unwrap();
                ced.finish_rows(ctx, enc, &weights[id], &rows, &expected[id][0])
                    .unwrap();
            }
        });
        let before: Vec<_> = [
            &ced.raw,
            &ced.query_rank,
            &ced.query,
            &ced.index_query,
            &ced.index_weights,
            &ced.latent,
            &ced.gate,
            &ced.selected,
            &ced.candidates,
        ]
        .into_iter()
        .map(|t| {
            (
                t.clone(),
                unsafe {
                    std::slice::from_raw_parts(
                        t.buffer
                            .contents()
                            .as_ptr()
                            .cast::<u8>()
                            .add(t.offset as usize),
                        t.n_elements() as usize * 4,
                    )
                }
                .to_vec(),
            )
        })
        .collect();
        execute(ctx, |enc| {
            metal::dispatch_census_begin();
            ced.publish_ced_rows(
                ctx,
                enc,
                20,
                0,
                &weights[20],
                &input[20].view_subrange(0, vec![c.hidden as u64, n as u64]),
                &publication,
            )
            .unwrap();
            let census = metal::dispatch_census_take();
            for suffix in [
                "history",
                "publish_raw",
                "score",
                "index_heads",
                "selection_prepare",
                "pool",
            ] {
                assert!(
                    !census
                        .iter()
                        .any(|d| d.kernel == format!("kernel_deepseek_v41_attention_{suffix}"))
                );
            }
        });
        for (tensor, before) in before {
            let after = unsafe {
                std::slice::from_raw_parts(
                    tensor
                        .buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(tensor.offset as usize),
                    before.len(),
                )
            };
            assert_eq!(after, before);
        }
        ced.complete_ced_encoder(0, n).unwrap();
    }
    assert_status(&ced);
    let published = owner_bits(&ced);
    if scalar {
        assert_eq!(published, owner_bits(&ordinary));
    }
    // Poison every decoder slot. A first output may read only the W-1 seeded
    // prefix, never the extra old slot that the ordinary history wrapper gathers.
    let raw = ced.raw.buffer.contents().as_ptr().cast::<f32>();
    for i in 20 * c.window * c.head_dim..40 * c.window * c.head_dim {
        unsafe {
            raw.add(i).write(f32::NAN);
        }
    }
    let controls = QueryControls::new(ctx, &c, capacity, plan.candidate_rows.clone()).unwrap();
    assert!(
        controls
            .checked_tile_view(plan.candidate_rows.start - 1, 1)
            .is_err()
    );
    for layer in &plan.decoder {
        let id = layer.layer;
        if scalar {
            for p in layer.seed_rows.clone() {
                ced.begin_ced_seed(id, p, 1).unwrap();
                execute(ctx, |enc| {
                    ced.seed_ced_scalar(
                        ctx,
                        enc,
                        id,
                        p,
                        &weights[id],
                        &prefill::row(&input[id], c.hidden, p),
                    )
                    .unwrap()
                });
                ced.complete_ced_seed(id, p, 1).unwrap();
            }
            for p in layer.output_rows.clone() {
                let view = controls.checked_tile_view(p, 1).unwrap();
                ced.begin_ced_decoder_scalar(id, p, &view).unwrap();
                execute(ctx, |enc| {
                    ced.encode(
                        ctx,
                        enc,
                        id,
                        p,
                        &weights[id],
                        &prefill::row(&input[id], c.hidden, p),
                    )
                    .unwrap();
                });
                assert!(ced.validate_snapshot_position(n).is_err());
                ced.complete_ced_decoder_chunk(id, p, 1).unwrap();
                assert_eq!(
                    bits(&ced.output),
                    bits(&prefill::row(&expected[id][0], c.hidden, p)),
                    "scalar layer={id} p={p}"
                );
                assert_eq!(
                    ids(&view.selected, c.index_topk),
                    ids(
                        &prefill::row(&expected[id][1], c.index_topk, p),
                        c.index_topk
                    )
                );
                let visible = (p + 1).div_ceil(c.candidate_block);
                assert_eq!(
                    ids(&view.candidates, visible),
                    ids(&prefill::row(&expected[id][2], blocks, p), visible)
                );
            }
        } else {
            let seeds = &layer.seed_rows;
            if !seeds.is_empty() {
                let scratch = SeedRows::new(ctx, &c, seeds.len()).unwrap();
                ced.begin_ced_seed(id, seeds.start, seeds.len()).unwrap();
                execute(ctx, |enc| {
                    ced.seed_ced_rows(
                        ctx,
                        enc,
                        id,
                        seeds.start,
                        &weights[id],
                        &input[id].view_subrange(
                            (seeds.start * c.hidden) as u64,
                            vec![c.hidden as u64, seeds.len() as u64],
                        ),
                        &scratch,
                    )
                    .unwrap()
                });
                ced.complete_ced_seed(id, seeds.start, seeds.len()).unwrap();
            }
            let outputs = &layer.output_rows;
            let view = controls
                .checked_tile_view(outputs.start, outputs.len())
                .unwrap();
            let rows = Rows::with_controls(ctx, &c, capacity, &view).unwrap();
            rows.reset_grouped_layouts();
            let output = prefill::allocate(ctx, c.hidden, outputs.len(), GgmlType::F32).unwrap();
            ced.begin_ced_decoder_chunk(id, outputs.start, outputs.len())
                .unwrap();
            execute(ctx, |enc| {
                ced.encode_rows(
                    ctx,
                    enc,
                    id,
                    outputs.start,
                    &weights[id],
                    &input[id].view_subrange(
                        (outputs.start * c.hidden) as u64,
                        vec![c.hidden as u64, outputs.len() as u64],
                    ),
                    &rows,
                )
                .unwrap();
                ced.finish_rows(ctx, enc, &weights[id], &rows, &output)
                    .unwrap();
            });
            ced.complete_ced_decoder_chunk(id, outputs.start, outputs.len())
                .unwrap();
            let expected_output = expected[id][0].view_subrange(
                (outputs.start * c.hidden) as u64,
                vec![c.hidden as u64, outputs.len() as u64],
            );
            close(
                f32_slice(&output).unwrap(),
                f32_slice(&expected_output).unwrap(),
            );
            for (row, p) in outputs.clone().enumerate() {
                assert_eq!(
                    ids(
                        &prefill::row(&view.selected, c.index_topk, row),
                        c.index_topk
                    ),
                    ids(
                        &prefill::row(&expected[id][1], c.index_topk, p),
                        c.index_topk
                    )
                );
                let visible = (p + 1).div_ceil(c.candidate_block);
                assert_eq!(
                    ids(&prefill::row(&view.candidates, blocks, row), visible),
                    ids(&prefill::row(&expected[id][2], blocks, p), visible)
                );
            }
        }
        assert_status(&ced);
        assert_eq!(
            owner_bits(&ced),
            published,
            "replay changed globals layer={id}"
        );
    }
    ced.commit_ced(n).unwrap();
    ced.validate_snapshot_position(n).unwrap();
    if scalar {
        assert_eq!(bits(&ced.raw), bits(&ordinary.raw));
    } else {
        close(
            f32_slice(&ced.raw).unwrap(),
            f32_slice(&ordinary.raw).unwrap(),
        );
    }
    for id in 0..40 {
        execute(ctx, |enc| {
            let x = prefill::row(&input[id], c.hidden, n);
            ordinary.encode(ctx, enc, id, n, &weights[id], &x).unwrap();
            ced.encode(ctx, enc, id, n, &weights[id], &x).unwrap();
        });
        if scalar {
            assert_eq!(bits(&ced.output), bits(&ordinary.output));
        } else {
            close(
                f32_slice(&ced.output).unwrap(),
                f32_slice(&ordinary.output).unwrap(),
            );
        }
    }
    ced.validate_snapshot_position(n + 1).unwrap();
    assert_status(&ced);
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn ced_scalar_native_publication_seed_controls_reuse_and_continuation() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("CED GPU lease");
    let ctx = MetalContext::new().unwrap();
    attention_case(&ctx, true);
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn ced_rows_publication_only_poisoned_history_and_borrowed_controls() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("CED GPU lease");
    let ctx = MetalContext::new().unwrap();
    attention_case(&ctx, false);
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn ced_owner_rows_reuse_preserves_published_cache_with_ordinary_batch_control() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    assert!(std::env::var_os("QWEN_V41_SELECT_PER_ROW").is_none());
    let _lease = metal::acquire_metal_benchmark_lease().expect("CED GPU lease");
    let ctx = MetalContext::new().unwrap();
    let old = super::super::query_batch::with_old_host_loop(|| owner_rows_reuse_case(&ctx));
    let range = owner_rows_reuse_case(&ctx);
    assert_eq!(
        old, range,
        "old host vs range: payloads and ordered full dispatch census"
    );
}

fn owner_rows_reuse_case(ctx: &MetalContext) -> (Vec<Vec<u32>>, Vec<String>) {
    assert!(
        metal::evaluate_metal_memory_admission_with_cpu_bytes(
            64 << 20,
            32 << 20,
            0,
            ctx.memory_signals(),
            true
        )
        .is_admitted()
    );
    let mut snapshots = Vec::new();
    let mut launches = Vec::new();
    let c = Config {
        hidden: 32,
        heads: 1,
        head_dim: 512,
        query_rank: 32,
        output_rank: 32,
        groups: 1,
        index_heads: 1,
        index_dim: 128,
        index_topk: 8,
        window: 2,
        rope_dim: 64,
        candidate_block: 2,
        candidate_topk: 4,
        norm_eps: 1e-6,
        ..Config::default()
    };
    let schedule = Schedule {
        layers: 3,
        kv_source_layers: vec![0, 1],
        index_source_layers: vec![0, 1, 2],
        compression_ratios: vec![2, 1, 1],
        candidate_source_layer: Some(1),
    };
    let (n, capacity) = (7, 8);
    let plan = CedPlan::shrinking(
        &c,
        &schedule,
        &engram::HashConfig {
            layers: Vec::new(),
            ..engram::HashConfig::default()
        },
        &CedRequest::cold(n, capacity),
    )
    .unwrap();
    let frequencies = op::rotary_frequencies(c.rope_dim, 160000.0, 65536, 16.0, 32.0, 1.0).unwrap();
    let weights: Vec<_> = (0..schedule.layers)
        .map(|id| {
            let mut weights = scheduled_weights(&ctx, &c, &schedule, id, GgmlType::BF16);
            weights.frequencies = frequencies.clone();
            weights
        })
        .collect();
    let refs = weights.iter().collect::<Vec<_>>();
    let input: Vec<_> = (0..schedule.layers)
        .map(|id| upload(&ctx, &values(&c, n, id), vec![c.hidden as u64, n as u64]).unwrap())
        .collect();
    let chunk = |id: usize, range: std::ops::Range<usize>| {
        input[id].view_subrange(
            (range.start * c.hidden) as u64,
            vec![c.hidden as u64, range.len() as u64],
        )
    };
    let kernel = "kernel_deepseek_v41_attention_compress_rows";
    let mut ordinary = Workspace::new_with_schedule(&ctx, &c, capacity, &refs, &schedule).unwrap();
    metal::dispatch_census_begin();
    // The second chunk has both an unaligned ratio-two head and tail.
    for range in [0..3, 3..n] {
        let rows = Rows::new(&ctx, &c, capacity, range.len()).unwrap();
        ordinary.begin_batch(range.start, range.len()).unwrap();
        execute(&ctx, |enc| {
            for (id, w) in weights.iter().enumerate() {
                ordinary
                    .encode_rows(
                        &ctx,
                        enc,
                        id,
                        range.start,
                        w,
                        &chunk(id, range.clone()),
                        &rows,
                    )
                    .unwrap();
            }
        });
    }
    let census = metal::dispatch_census_take();
    launches.push(format!("{census:?}"));
    assert_eq!(
        census
            .iter()
            .filter(|d| d.kernel == kernel)
            .map(|d| (d.grid_width, d.grid_height, d.grid_depth, d.tg_threads))
            .collect::<Vec<_>>(),
        [
            (1, 1, 1, 256),
            (3, 1, 1, 256),
            (1, 1, 1, 256),
            (4, 1, 1, 256)
        ],
        "Ordinary ratio-one and ratio-two owners must retain batched publication",
    );
    assert_status(&ordinary);
    snapshots.extend(owner_bits(&ordinary));
    snapshots.push(bits(&ordinary.raw));

    let mut ced = Workspace::new_with_schedule(&ctx, &c, capacity, &refs, &schedule).unwrap();
    let rows = Rows::new(&ctx, &c, capacity, n).unwrap();
    let publication = PublicationRows::new(&ctx, &c, n).unwrap();
    ced.begin_ced(&plan).unwrap();
    ced.begin_ced_encoder(0, n).unwrap();
    metal::dispatch_census_begin();
    execute(&ctx, |enc| {
        ced.encode_rows(&ctx, enc, 0, 0, &weights[0], &input[0], &rows)
            .unwrap();
        ced.publish_ced_rows(&ctx, enc, 1, 0, &weights[1], &input[1], &publication)
            .unwrap();
    });
    launches.push(format!("{:?}", metal::dispatch_census_take()));
    ced.complete_ced_encoder(0, n).unwrap();
    assert_status(&ced);
    let published = owner_bits(&ced);
    snapshots.extend(published.clone());
    assert!(ced.owners.iter().all(|o| {
        [&o.compressed, &o.keys]
            .into_iter()
            .all(|t| f32_slice(t).unwrap().iter().any(|v| *v != 0.0))
    }));
    let controls = QueryControls::new(&ctx, &c, capacity, plan.candidate_rows.clone()).unwrap();
    for layer in &plan.decoder {
        let id = layer.layer;
        let seeds = &layer.seed_rows;
        let scratch = SeedRows::new(&ctx, &c, seeds.len()).unwrap();
        ced.begin_ced_seed(id, seeds.start, seeds.len()).unwrap();
        metal::dispatch_census_begin();
        execute(&ctx, |enc| {
            ced.seed_ced_rows(
                &ctx,
                enc,
                id,
                seeds.start,
                &weights[id],
                &chunk(id, seeds.clone()),
                &scratch,
            )
            .unwrap();
        });
        launches.push(format!("{:?}", metal::dispatch_census_take()));
        ced.complete_ced_seed(id, seeds.start, seeds.len()).unwrap();
        let outputs = &layer.output_rows;
        let view = controls
            .checked_tile_view(outputs.start, outputs.len())
            .unwrap();
        let rows = Rows::with_controls(&ctx, &c, capacity, &view).unwrap();
        // ReusePublished must neither prepare nor publish these stale rows.
        for t in [&rows.compressor_value, &rows.compressor_gate] {
            let ptr = t.buffer.contents().as_ptr().cast::<f32>();
            for i in 0..t.n_elements() as usize {
                unsafe { ptr.add(t.offset as usize / 4 + i).write(f32::NAN) };
            }
        }
        let output = prefill::allocate(&ctx, c.hidden, outputs.len(), GgmlType::F32).unwrap();
        ced.begin_ced_decoder_chunk(id, outputs.start, outputs.len())
            .unwrap();
        assert_eq!(
            ced.global_mode(id, outputs.start, outputs.len()).unwrap(),
            GlobalKvMode::ReusePublished
        );
        let progress = ced.ced_public_decoder_progress();
        metal::dispatch_census_begin();
        execute(&ctx, |enc| {
            ced.encode_rows(
                &ctx,
                enc,
                id,
                outputs.start,
                &weights[id],
                &chunk(id, outputs.clone()),
                &rows,
            )
            .unwrap();
            assert_eq!(ced.ced_pending_encoding(), Some((true, false)));
            assert_eq!(ced.ced_public_decoder_progress(), progress);
            ced.finish_rows(&ctx, enc, &weights[id], &rows, &output)
                .unwrap();
        });
        let census = metal::dispatch_census_take();
        launches.push(format!("{census:?}"));
        snapshots.extend([
            bits(&output),
            bits(&rows.query),
            ids(&rows.selected, outputs.len() * c.index_topk)
                .into_iter()
                .map(|v| v as u32)
                .collect(),
            ids(
                &rows.candidates,
                outputs.len() * capacity.div_ceil(c.candidate_block),
            )
            .into_iter()
            .map(|v| v as u32)
            .collect(),
            bits(&ced.raw),
        ]);
        assert_eq!(
            owner_bits(&ced),
            published,
            "replay changed globals layer={id}"
        );
        assert!(
            !census.iter().any(|d| d.kernel == kernel),
            "ReusePublished must not dispatch batched owner publication layer={id}",
        );
        assert_status(&ced);
        ced.complete_ced_decoder_chunk(id, outputs.start, outputs.len())
            .unwrap();
    }
    ced.commit_ced(n).unwrap();
    ced.validate_snapshot_position(n).unwrap();
    (snapshots, launches)
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn ced_scalar_error_restores_control_bindings_and_finite_publication_blocks_completion() {
    use objc2::rc::Retained;
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("CED GPU lease");
    let ctx = MetalContext::new().unwrap();
    let data = fixture();
    let cpu: Vec<_> = (0..40).map(|id| cpu_weights(&data, id)).collect();
    let weights: Vec<_> = cpu.iter().map(|w| resident(&ctx, w)).collect();
    let mut c = super::super::super::toy::config();
    c.hidden = 8;
    let mut ced = Workspace::new(&ctx, &c, 2, &weights.iter().collect::<Vec<_>>()).unwrap();
    let input = upload(&ctx, &[1.0; 8], vec![8]).unwrap();
    ced.begin_ced(&plan(&c, 1, 2)).unwrap();
    ced.begin_ced_encoder(0, 1).unwrap();
    execute(&ctx, |enc| {
        for (id, weight) in weights[..20].iter().enumerate() {
            ced.encode(&ctx, enc, id, 0, weight, &input).unwrap();
        }
        ced.publish_ced_scalar(&ctx, enc, 20, 0, &weights[20], &input)
            .unwrap();
    });
    ced.complete_ced_encoder(0, 1).unwrap();
    let controls = QueryControls::new(&ctx, &c, 2, 0..1).unwrap();
    ced.begin_ced_decoder_scalar(20, 0, &controls.checked_tile_view(0, 1).unwrap())
        .unwrap();
    let before = [ced.selected.clone(), ced.candidates.clone()];
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    assert!(
        ced.encode(
            &ctx,
            &enc,
            20,
            0,
            &weights[20],
            &input.view_subrange(0, vec![7])
        )
        .is_err()
    );
    enc.end();
    for (a, b) in before.iter().zip([&ced.selected, &ced.candidates]) {
        assert_eq!(Retained::as_ptr(&a.buffer), Retained::as_ptr(&b.buffer));
        assert_eq!(a.offset, b.offset);
    }
    assert!(ced.complete_ced_decoder_chunk(20, 0, 1).is_err());
    assert_eq!((ced.next_layer, ced.next_position), (0, 0));

    let mut bad = Workspace::new(&ctx, &c, 2, &weights.iter().collect::<Vec<_>>()).unwrap();
    bad.begin_ced(&plan(&c, 1, 2)).unwrap();
    bad.begin_ced_encoder(0, 1).unwrap();
    let nan = upload(&ctx, &[f32::NAN; 8], vec![8]).unwrap();
    execute(&ctx, |enc| {
        for (id, weight) in weights[..20].iter().enumerate() {
            bad.encode(&ctx, enc, id, 0, weight, &input).unwrap();
        }
        bad.publish_ced_scalar(&ctx, enc, 20, 0, &weights[20], &nan)
            .unwrap();
    });
    assert_ne!(ids(bad.status(), 1), [0]);
    assert!(bad.complete_ced_encoder(0, 1).is_err());
    assert!(bad.begin_ced_decoder_chunk(20, 0, 1).is_err());
}
