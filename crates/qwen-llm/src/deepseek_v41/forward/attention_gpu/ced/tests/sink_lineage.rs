use super::super::super::{query_batch, tests::scheduled_weights};
use super::*;
use objc2_metal::{MTLDevice, MTLGPUFamily};

const WIDTH: usize = 128;
const LIMIT: u64 = 32 * 1024 * 1024;

fn config() -> Config {
    Config {
        hidden: 32,
        query_rank: 32,
        output_rank: 32,
        heads: 16,
        head_dim: 512,
        groups: 1,
        index_heads: 1,
        index_dim: 32,
        index_topk: 4,
        window: 19,
        rope_dim: 8,
        candidate_block: 16,
        candidate_topk: 16,
        ..Config::default()
    }
}

fn schedule() -> Schedule {
    Schedule {
        layers: 3,
        kv_source_layers: vec![1],
        index_source_layers: vec![1],
        compression_ratios: vec![0, 1, 1],
        candidate_source_layer: Some(1),
    }
}

fn plan(c: &Config, s: &Schedule, n: usize) -> CedPlan {
    CedPlan::shrinking(
        c,
        s,
        &engram::HashConfig {
            layers: Vec::new(),
            ..engram::HashConfig::default()
        },
        &CedRequest::cold(n, n),
    )
    .unwrap()
}

// Sum all explicit allocations, even weights dropped before Rows allocation,
// with 16 KiB slack per buffer and optional half staging on either GPU tier.
fn buffer_budget(c: &Config, s: &Schedule, n: usize, compact: usize) -> u64 {
    let mut buffers = shapes(c, n, s).unwrap();
    for id in 0..s.layers {
        buffers.extend([
            c.hidden * c.query_rank,
            c.query_rank * c.heads * c.head_dim,
            c.hidden * c.head_dim,
            c.heads * c.head_dim * c.output_rank,
            c.groups * c.output_rank * c.hidden,
        ]);
        if s.owns_kv(id) {
            buffers.push(c.hidden * c.head_dim);
        }
        if s.produces_index(id) {
            buffers.extend([
                c.query_rank * c.index_heads * c.index_dim,
                c.hidden * c.index_heads,
                c.head_dim * c.index_dim,
            ]);
        }
    }
    for (rows, controls) in [(n, false), (compact, true)] {
        let widths = if controls {
            Rows::widths_with_controls(c, n)
        } else {
            Rows::widths(c, n)
        };
        buffers.extend(widths.into_iter().map(|w| w * rows));
        // Use rows rather than WIDTH as a conservative tile bound, independent
        // of the process-wide override in the CPU-only budget test.
        buffers.extend([
            (c.window + rows) * c.head_dim,
            rows * c.heads * (c.window + c.index_topk + 1),
            rows * c.heads * c.head_dim,
            rows * Rows::select_width(c, n),
        ]);
        // Price half scratch as F32 as well; it is actually two bytes/element.
        buffers.push(prefill::attention_half_scratch_elements(c, rows).unwrap());
    }
    buffers.extend(QueryControls::shapes(c, n, n - c.window..n).unwrap());
    buffers.into_iter().map(|n| n as u64 * 4 + 16_384).sum()
}

#[test]
fn ced_sink_lineage_fixture_budget_and_plan() {
    let c = config();
    let s = schedule();
    for (n, decoder, range) in [(128, 1, 127..128), (145, 1, 144..145), (145, 0, 126..145)] {
        let p = plan(&c, &s, n);
        assert_eq!(p.decoder[decoder].output_rows, range);
        assert!(prefill::original_scalar_tail(n).is_none());
        let bytes = buffer_budget(&c, &s, n, range.len());
        assert!(bytes < LIMIT, "explicit buffer upper bound: {bytes}");
    }
}

fn sample(position: usize, element: usize, salt: usize) -> f32 {
    (((position * 37 + element * salt + element / 7 * 11) % 257) as f32 - 128.0) / 103.0
}

fn write(t: &MetalTensor, values: &[f32]) -> Result<()> {
    op::require(
        f32_slice(t)?.len() == values.len(),
        "sink fixture upload shape",
    )?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            t.buffer
                .contents()
                .as_ptr()
                .cast::<f32>()
                .add((t.offset / 4) as usize),
            values.len(),
        );
    }
    Ok(())
}

fn prepare(w: &Workspace, id: usize, start: usize, rows: &Rows) {
    let c = &w.config;
    let width = c.heads * c.head_dim;
    let q: Vec<_> = (0..rows.rows * width)
        .map(|i| sample(start + i / width, i % width, 17))
        .collect();
    write(&rows.query, &q).unwrap();
    let history = w.row_history(id, start, rows.rows).unwrap();
    let mut raw = vec![0.0; rows.tile.history.n_elements() as usize];
    for (i, v) in raw[..history.extent * c.head_dim].iter_mut().enumerate() {
        *v = sample(history.floor + i / c.head_dim, i % c.head_dim, 29);
    }
    write(&rows.tile.history, &raw).unwrap();
    rows.history.set(Some(history));
    let ids: Vec<u32> = (start..start + rows.rows)
        .flat_map(|p| {
            let count = (p + 1).min(c.index_topk);
            (0..c.index_topk).map(move |i| {
                if i < count {
                    (p + 1 - count + i) as u32
                } else {
                    u32::MAX
                }
            })
        })
        .collect();
    assert_eq!(rows.selected.n_elements(), ids.len() as u64);
    unsafe {
        let dest = rows
            .selected
            .buffer
            .contents()
            .as_ptr()
            .cast::<u32>()
            .add((rows.selected.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(ids.as_ptr(), dest, ids.len());
    }
}

struct Census;

impl Drop for Census {
    fn drop(&mut self) {
        let _ = metal::dispatch_census_take();
        let _ = metal::allocation_census_take();
    }
}

fn dispatch_rows(
    ctx: &MetalContext,
    w: &Workspace,
    id: usize,
    start: usize,
    rows: &Rows,
    spans: &[(usize, bool)],
) {
    assert!(!metal::dispatch_census_is_active());
    metal::dispatch_census_begin();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    w.encode_rows_tiles(ctx, &enc, id, start, rows).unwrap();
    enc.end();
    let census = metal::dispatch_census_take();
    assert_eq!(census.len(), 5 * spans.len(), "{census:?}");
    for (launches, &(n, tensor)) in census.chunks_exact(5).zip(spans) {
        for (launch, suffix) in launches.iter().zip([
            if tensor {
                if std::env::var_os("QWEN_V41_SINK_TENSOR32").is_none_or(|v| v != "0") {
                    "score_tensor32"
                } else {
                    "score_tensor"
                }
            } else {
                "score_mma"
            },
            "softmax_rows",
            if tensor { "pv_tensor" } else { "pv_mma" },
            "normalize_rows",
            "rope_rows",
        ]) {
            assert_eq!(
                launch.kernel,
                format!("kernel_deepseek_v41_attention_{suffix}")
            );
            assert!(!launch.encoder_concurrent);
            assert_eq!(
                (
                    launch.grid_depth,
                    launch.threads_height,
                    launch.threads_depth
                ),
                (1, 1, 1)
            );
        }
        for (launch, heads, threads) in [
            (&launches[0], 1, 128),
            (&launches[1], 16, 32),
            (&launches[2], 1, if tensor { 128 } else { 256 }),
        ] {
            assert_eq!(
                (launch.grid_width, launch.grid_height, launch.threads_width),
                (heads, n as u64, threads)
            );
        }
        for launch in &launches[3..] {
            assert_eq!(launch.grid_height, 1);
            assert_eq!(launch.grid_tgs * launch.tg_threads, (n * 16 * 512) as u64);
        }
    }
    metal::commit_and_wait(&command).unwrap();
    assert_eq!(
        unsafe { *w.status.buffer.contents().as_ptr().cast::<u32>() },
        0
    );
}

fn run_case(ctx: &MetalContext, n: usize, decoder: usize, range: Range<usize>, tensor: bool) {
    metal::allocation_census_begin();
    let _census = Census;
    let c = config();
    let s = schedule();
    let p = plan(&c, &s, n);
    assert_eq!(p.decoder[decoder].output_rows, range);
    let budget = buffer_budget(&c, &s, n, range.len());
    assert!(budget < LIMIT, "explicit buffer upper bound: {budget}");
    let id = p.decoder[decoder].layer;
    let mut w = {
        let weights: Vec<_> = (0..s.layers)
            .map(|id| {
                let mut w = scheduled_weights(ctx, &c, &s, id, GgmlType::F32);
                w.frequencies = (0..c.rope_dim / 2)
                    .map(|i| 0.007 / (i + 1) as f32)
                    .collect();
                w
            })
            .collect();
        Workspace::new_with_schedule(ctx, &c, n, &weights.iter().collect::<Vec<_>>(), &s).unwrap()
    };
    let compressed: Vec<_> = (0..n * c.head_dim)
        .map(|i| sample(i / c.head_dim, i % c.head_dim, 43))
        .collect();
    write(&w.owners[0].compressed, &compressed).unwrap();
    let full = Rows::new(ctx, &c, n, n).unwrap();
    assert_eq!(full.arithmetic, RowsArithmetic::Ordinary);
    prepare(&w, id, 0, &full);
    let full_spans = if n == WIDTH {
        vec![(WIDTH, tensor)]
    } else {
        vec![(WIDTH, tensor), (17, false)]
    };
    dispatch_rows(ctx, &w, id, 0, &full, &full_spans);

    w.begin_ced(&p).unwrap();
    // Sink-only witness: synthetic published KV/history/control payloads and
    // completed frontiers, not execution of the encoder or earlier decoders.
    {
        let cursor = w.ced.as_mut().unwrap();
        cursor.encoder_completed = n;
        cursor.publication_completed = n;
        cursor.decoder_next = decoder;
        for j in 0..=decoder {
            cursor.seed_completed[j] = p.decoder[j].seed_rows.end;
            if j < decoder {
                cursor.output_completed[j] = n;
            }
        }
    }
    let controls = QueryControls::new(ctx, &c, n, p.candidate_rows.clone()).unwrap();
    controls
        .coverage
        .lock()
        .unwrap()
        .complete(p.candidate_rows.clone(), &s, 1);
    let view = controls
        .checked_tile_view(range.start, range.len())
        .unwrap();
    let compact = Rows::with_controls(ctx, &c, n, &view).unwrap();
    assert_eq!(compact.arithmetic, RowsArithmetic::Packed);
    w.begin_ced_decoder_chunk(id, range.start, range.len())
        .unwrap();
    w.bind_ced_rows(id, range.start, &compact).unwrap();
    prepare(&w, id, range.start, &compact);
    for _ in range.clone() {
        w.advance_ced_row().unwrap();
    }
    assert_eq!(
        w.ced_decoder_prompt_rows(id, range.start, range.len())
            .unwrap(),
        Some(n)
    );
    let spans = match range.len() {
        1 => vec![(1, tensor && n == WIDTH)],
        19 => vec![(2, tensor), (17, false)],
        _ => unreachable!(),
    };
    dispatch_rows(ctx, &w, id, range.start, &compact, &spans);
    let width = c.heads * c.head_dim;
    let expected = &f32_slice(&full.query).unwrap()[range.start * width..range.end * width];
    let actual = f32_slice(&compact.query).unwrap();
    assert_eq!(actual.len(), expected.len());
    assert!(expected.iter().any(|v| v.abs() > 1e-4));
    // Same per-query pipeline and inputs must be bitwise stable across physical
    // arities; do not use the looser Tensor-vs-MMA or half-vs-F32 tolerances.
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "N={n}, range={range:?}, element={i}"
        );
    }
    let allocations = metal::allocation_census_take();
    assert!(!allocations.is_empty());
    let explicit_bytes: u64 = allocations.iter().map(|a| a.buffer_length).sum();
    assert!(
        explicit_bytes <= budget,
        "explicit={explicit_bytes}, bound={budget}"
    );
    eprintln!(
        "CED sink lineage N={n} range={range:?}: explicit cumulative buffer bytes={explicit_bytes}, bound={budget}; context/pipeline/driver overhead excluded"
    );
}

#[test]
#[ignore = "requires explicit Metal benchmark lease, debug layer and QWEN_V41_SINK_TILE=128"]
fn ced_sink_lineage_gpu_singletons_and_boundary_bitwise() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    assert_eq!(
        query_batch::tile_rows(),
        WIDTH,
        "set QWEN_V41_SINK_TILE=128 in a fresh process"
    );
    assert!(std::env::var_os("QWEN_V41_SINK_TENSOR").is_none_or(|v| v != "0"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("bounded CED sink GPU lease");
    let ctx = MetalContext::new().unwrap();
    let tensor = metal::moe_grouped_tensor_available(&ctx);
    if ctx.device.supportsFamily(MTLGPUFamily::Apple10)
        && ctx.device.supportsFamily(MTLGPUFamily::Metal4)
    {
        assert!(
            tensor,
            "Tensor-capable device must load the Tensor tier: {:?}",
            ctx.tensor_ops_status()
        );
    }
    if tensor {
        assert!(
            metal::moe_grouped_tensor_selected(&ctx),
            "enable QWEN_EXPERT_TENSOR_TILE for Tensor coverage"
        );
    } else {
        eprintln!(
            "Tensor tier unsupported: {:?}; all cases still validate MMA fallback",
            ctx.tensor_ops_status()
        );
    }
    for (n, decoder, range) in [(128, 1, 127..128), (145, 1, 144..145), (145, 0, 126..145)] {
        objc2::rc::autoreleasepool(|_| run_case(&ctx, n, decoder, range, tensor));
    }
}
