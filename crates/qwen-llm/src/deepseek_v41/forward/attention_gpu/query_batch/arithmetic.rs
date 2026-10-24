use super::*;

#[test]
fn f32_attention_scope_nests_is_thread_local_and_restores_on_unwind() {
    let c = Config::default();
    assert!(mma_geometry(&c));
    let ordinary = Scratch::shapes(&c, 2048).unwrap();
    assert_eq!(ordinary.len(), 3);
    with_f32_attention(|| {
        assert!(!mma_geometry(&c));
        with_f32_attention(|| assert!(!mma_geometry(&c)));
        assert!(!mma_geometry(&c));
        assert!(
            std::thread::spawn(|| mma_geometry(&Config::default()))
                .join()
                .unwrap()
        );
    });
    assert!(mma_geometry(&c));
    assert!(std::panic::catch_unwind(|| with_f32_attention(|| panic!("restore"))).is_err());
    assert!(mma_geometry(&c));
    assert_eq!(Scratch::shapes(&c, 2048).unwrap(), ordinary);
}

#[test]
fn f32_attention_planning_prices_the_released_geometry_gathered_matrices() {
    let c = Config::default();
    for n in [1, 33, 129, 2048] {
        let ordinary = Scratch::shapes(&c, n).unwrap();
        prefill::with_batched_f32(|| {
            let shapes = Scratch::shapes(&c, n).unwrap();
            let tile = n.min(tile_rows());
            let padded = (c.window + c.index_topk).div_ceil(32) * 32;
            assert_eq!(&shapes[..3], ordinary);
            assert_eq!(shapes[3], tile * padded * c.head_dim);
            assert_eq!(shapes[4], tile * padded * c.heads);
            let widths = Rows::tile_widths(&c, n).unwrap();
            assert_eq!(shapes.len(), widths.len());
            for (&actual, width) in shapes.iter().zip(widths) {
                assert!(width * n >= actual && width * n - actual < n);
            }
            if tile == 512 {
                assert_eq!((shapes[3] + shapes[4]) * 4, 754_974_720);
            }
        });
        assert_eq!(Scratch::shapes(&c, n).unwrap(), ordinary);
    }
    prefill::with_batched_f32(|| {
        assert!(Scratch::shapes(&c, 0).is_err());
        assert!(Scratch::shapes(&c, prefill::MAX_CHUNK_ROWS + 1).is_err());
    });
}

fn config() -> Config {
    Config {
        hidden: 32,
        query_rank: 32,
        output_rank: 32,
        heads: 16,
        head_dim: 512,
        groups: 1,
        index_heads: 2,
        index_dim: 32,
        index_topk: 512,
        window: 128,
        rope_dim: 0,
        candidate_block: 2,
        candidate_topk: 512,
        norm_eps: 1e-6,
        ..Config::default()
    }
}

fn schedule() -> Schedule {
    Schedule {
        layers: 1,
        kv_source_layers: vec![0],
        index_source_layers: vec![0],
        compression_ratios: vec![1],
        candidate_source_layer: Some(0),
    }
}

fn weights<'a>(ctx: &'a MetalContext, c: &Config) -> Weights<'a> {
    let linear = |input: usize, output: usize| {
        let values: Vec<_> = (0..input * output)
            .map(|i| half::bf16::from_f32(((i * 19 % 101) as f32 - 50.0) / 8192.0).to_bits())
            .collect();
        Linear {
            ctx,
            tensor: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&values),
                vec![input as u64, output as u64],
                GgmlType::BF16,
            )
            .unwrap(),
            native_control: None,
            input,
            output,
        }
    };
    Weights {
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
        compressor: Some(op::CompressorWeights {
            ratio: 1,
            kv: linear(c.hidden, c.head_dim),
            gate: None,
            norm: vec![1.0; c.head_dim],
        }),
        index: Some(op::IndexWeights {
            query: linear(c.query_rank, c.index_heads * c.index_dim),
            weights: linear(c.hidden, c.index_heads),
            key: Some(linear(c.head_dim, c.index_dim)),
            key_norm: vec![1.0; c.index_dim],
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

fn input(ctx: &MetalContext, c: &Config, n: usize) -> MetalTensor {
    upload(
        ctx,
        &(0..c.hidden * n)
            .map(|i| ((i * 31 % 113) as f32 - 56.0) / 71.0)
            .collect::<Vec<_>>(),
        vec![c.hidden as u64, n as u64],
    )
    .unwrap()
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn packed_attention_propagates_to_all_seven_query_and_output_projections() {
    let _lease = metal::acquire_metal_benchmark_lease().expect("arithmetic GPU lease");
    let ctx = MetalContext::new().unwrap();
    let c = config();
    let s = schedule();
    let w = weights(&ctx, &c);
    let workspace = Workspace::new_with_schedule(&ctx, &c, 256, &[&w], &s).unwrap();
    for n in [1, 33] {
        let controls = QueryControls::new(&ctx, &c, 256, 0..n).unwrap();
        let view = controls.checked_tile_view(0, n).unwrap();
        let x = input(&ctx, &c, n);
        let output = prefill::allocate(&ctx, c.hidden, n, GgmlType::F32).unwrap();
        for packed in [false, true] {
            let rows = if packed {
                Rows::with_controls(&ctx, &c, 256, &view).unwrap()
            } else {
                Rows::new(&ctx, &c, 256, n).unwrap()
            };
            assert_eq!(
                rows.arithmetic,
                if packed {
                    RowsArithmetic::Packed
                } else {
                    RowsArithmetic::Ordinary
                }
            );
            metal::dispatch_census_begin();
            let result = submit(
                &ctx,
                "attention_projection_arithmetic",
                [n, 0, 0, 0],
                |enc| {
                    workspace.prepare_rows_mode(
                        &ctx,
                        enc,
                        0,
                        &w,
                        &x,
                        &rows,
                        GlobalKvMode::ReusePublished,
                    )?;
                    workspace
                        .finish_rows(&ctx, enc, &w, &rows, &output)
                        .map_err(|detail| metal::MetalError::BadShape {
                            kernel: "arithmetic_test",
                            detail,
                        })
                },
            );
            let census = metal::dispatch_census_take();
            result.unwrap();
            assert_eq!(
                census
                    .iter()
                    .filter(|d| d.kernel.starts_with("kernel_mat_mat_bf16"))
                    .count(),
                if packed || n > 1 { 7 } else { 0 }
            );
            assert_eq!(rows.query.n_elements(), (c.heads * c.head_dim * n) as u64);
        }
    }
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn controlled_f32_attention_census_proves_both_pv_routes_and_matching_allocations() {
    let _lease = metal::acquire_metal_benchmark_lease().expect("arithmetic GPU lease");
    let ctx = MetalContext::new().unwrap();
    let c = config();
    let s = schedule();
    let w = weights(&ctx, &c);
    for start in [0, 129] {
        for n in [1, 33] {
            prefill::with_batched_f32(|| {
                let mut workspace =
                    Workspace::new_with_schedule(&ctx, &c, start + n, &[&w], &s).unwrap();
                workspace.next_position = start;
                let rows = Rows::new(&ctx, &c, start + n, n).unwrap();
                let shapes = Scratch::shapes(&c, n).unwrap();
                let matrix = rows.tile.matrix.as_ref().unwrap();
                for (tensor, &size) in [
                    &rows.tile.history,
                    &rows.tile.scores,
                    &rows.tile.attended,
                    &matrix.values,
                    &matrix.probabilities,
                ]
                .into_iter()
                .zip(&shapes)
                {
                    assert_eq!(tensor.n_elements(), size as u64);
                }
                let x = input(&ctx, &c, n);
                let output = prefill::allocate(&ctx, c.hidden, n, GgmlType::F32).unwrap();
                workspace.begin_batch(start, n).unwrap();
                metal::dispatch_census_begin();
                let result = submit(&ctx, "attention_f32_arithmetic", [n, start, 0, 0], |enc| {
                    workspace
                        .encode_rows(&ctx, enc, 0, start, &w, &x, &rows)
                        .and_then(|()| workspace.finish_rows(&ctx, enc, &w, &rows, &output))
                        .map_err(|detail| metal::MetalError::BadShape {
                            kernel: "arithmetic_test",
                            detail,
                        })
                });
                let census = metal::dispatch_census_take();
                result.unwrap();
                let has = |suffix: &str| {
                    census
                        .iter()
                        .any(|d| d.kernel == format!("kernel_deepseek_v41_attention_{suffix}"))
                };
                assert!(has("score_rows_lane8"));
                assert!(has("softmax_rows"));
                assert!(!has("score_mma") && !has("pv_mma"));
                assert!(
                    !census
                        .iter()
                        .any(|d| d.kernel.starts_with("kernel_mat_mat_"))
                );
                assert!(census.iter().any(|d| Some(&d.kernel)
                    == metal::grouped_projection_pipeline_name(GgmlType::BF16).as_ref()));
                assert_eq!(has("matrix_values_rows"), start > 0);
                assert_eq!(has("output_rows"), start == 0);
                assert_eq!(
                    census
                        .iter()
                        .any(|d| d.kernel == "kernel_latent_grouped_f32_f32"),
                    start > 0
                );
                workspace.validate_snapshot_position(start + n).unwrap();
                assert_eq!(
                    unsafe { *workspace.status.buffer.contents().as_ptr().cast::<i32>() },
                    0
                );
                assert!(f32_slice(&output).unwrap().iter().all(|v| v.is_finite()));
            });
        }
    }
    let mut workspace = Workspace::new_with_schedule(&ctx, &c, 1, &[&w], &s).unwrap();
    let rows = Rows::new(&ctx, &c, 1, 1).unwrap();
    workspace.begin_batch(0, 1).unwrap();
    prefill::with_batched_f32(|| {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        metal::dispatch_census_begin();
        let error = workspace
            .sink_tile(&ctx, &enc, 0, 0, &rows, 0, 1)
            .unwrap_err();
        assert!(error.contains("planned and allocated"), "{error}");
        assert!(metal::dispatch_census_take().is_empty());
        enc.end();
    });
}
