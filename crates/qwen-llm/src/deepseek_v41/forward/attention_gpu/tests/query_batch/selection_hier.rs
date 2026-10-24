//! Bounded proof fixtures, not a model witness or a duration benchmark.
use super::super::super::super::gpu_step;
use super::super::super::query_batch::{
    SelectionOverrides, SelectionRoute, select_hier_rows, selection_route, selection_shapes,
};
use super::*;

const K: usize = 512;
const LIMIT: u64 = 128 * 1024 * 1024;

#[path = "selection_hier/tile_reuse.rs"]
mod tile_reuse;

#[test]
fn hier_top512_shapes_price_two_allocations_and_capacity_not_prompt() {
    let c = Config::default();
    for (capacity, partitions) in [
        (4096, 0),
        (4097, 2),
        (8192, 2),
        (8448, 4),
        (16384, 4),
        (32768, 4),
    ] {
        for rows in [
            1,
            3,
            511,
            512,
            513,
            prefill::DEFAULT_CHUNK_ROWS,
            prefill::MAX_CHUNK_ROWS,
        ] {
            let shapes = selection_shapes(&c, capacity, rows, 512).unwrap();
            let t = rows.min(512);
            assert_eq!(shapes, [t * capacity.min(16384), t * partitions * K * 2]);
            let widths = Rows::select_widths(&c, capacity, rows).unwrap();
            let actual = selection_shapes(
                &c,
                capacity,
                rows,
                super::super::super::query_batch::tile_rows(),
            )
            .unwrap();
            assert_eq!(widths.len(), 2);
            for i in 0..2 {
                assert_eq!(widths[i], actual[i].div_ceil(rows));
                assert!(widths[i] * rows >= actual[i]);
            }
        }
    }
    assert_eq!(
        selection_shapes(&c, 8192, 512, 512).unwrap()[1] * 4,
        4 * 1024 * 1024
    );
    assert_eq!(
        selection_shapes(&c, 8448, 512, 512).unwrap()[1] * 4,
        8 * 1024 * 1024
    );
    let mut other = c.clone();
    other.index_topk = 256;
    assert_eq!(selection_shapes(&other, 16384, 512, 512).unwrap()[1], 0);
    assert!(selection_shapes(&c, 16384, 0, 512).is_err());
    assert!(selection_shapes(&c, 16384, 1, 0).is_err());
    assert!(selection_shapes(&c, 16384, prefill::MAX_CHUNK_ROWS + 1, 512).is_err());
    other.candidate_topk = usize::MAX;
    assert!(selection_shapes(&other, 8192, 512, 512).is_err());
    other.candidate_block = 1;
    assert!(selection_shapes(&other, usize::MAX, 512, 512).is_err());
}

#[test]
fn hier_top512_route_preserves_boundaries_overrides_and_pruning() {
    use SelectionRoute::*;
    let c = Config::default();
    let flags = SelectionOverrides::default();
    for (length, expected) in [
        (0, All),
        (512, All),
        (513, Fused2k),
        (2048, Fused2k),
        (2049, Fused4k),
        (4095, Fused4k),
        (4096, Fused4k),
        (4097, Hier(2)),
        (8191, Hier(2)),
        (8192, Hier(2)),
        (8193, Hier(4)),
        (16384, Hier(4)),
        (16385, Legacy),
    ] {
        assert_eq!(selection_route(&c, length, flags), expected, "L={length}");
    }
    for flags in [
        SelectionOverrides {
            per_row: true,
            ..flags
        },
        SelectionOverrides {
            split_sort: true,
            ..flags
        },
        SelectionOverrides {
            fuse_2048: true,
            ..flags
        },
        SelectionOverrides {
            no_hier: true,
            ..flags
        },
    ] {
        for length in [4097, 8192, 8193, 16384, 16385] {
            assert_eq!(selection_route(&c, length, flags), Legacy);
        }
    }
    let rollback = SelectionOverrides {
        no_hier: true,
        ..flags
    };
    assert_eq!(selection_route(&c, 2048, rollback), Fused2k);
    assert_eq!(selection_route(&c, 4096, rollback), Fused4k);
    let fuse2 = SelectionOverrides {
        fuse_2048: true,
        ..flags
    };
    assert_eq!(selection_route(&c, 2048, fuse2), Fused2k);
    assert_eq!(selection_route(&c, 2049, fuse2), Legacy);
    let mut unsupported = c.clone();
    unsupported.index_topk = 511;
    assert_eq!(selection_route(&unsupported, 4097, flags), Legacy);
    unsupported.candidate_topk = 1;
    assert_eq!(selection_route(&unsupported, 4096, flags), Legacy);
    // A tile's last length picks P; every row still has its own floor division.
    for ratio in [1, 2] {
        let (first, n) = (7, 3);
        let start = 4097 * ratio - first - n;
        let visible: Vec<_> = (first..first + n)
            .map(|row| (start + row + 1) / ratio)
            .collect();
        assert!(visible[0] <= 4096 && visible[2] == 4097);
        assert_eq!(
            selection_route(&c, (start + first + n) / ratio, flags),
            Hier(2)
        );
    }
    assert_eq!(16384usize.div_ceil(c.candidate_block), c.candidate_topk);
    assert!(16385usize.div_ceil(c.candidate_block) > c.candidate_topk);
}

#[test]
fn hier_top512_full_and_ced_budget_consume_separately_rounded_shapes() {
    let c = Config::default();
    let (capacity, rows) = (8448, prefill::DEFAULT_CHUNK_ROWS);
    let widths = Rows::select_widths(&c, capacity, rows).unwrap();
    assert_eq!(widths.len(), 2);
    let full = gpu_step::chunk_widths_from_geometry(&c, 1, &[], capacity, rows).unwrap();
    let ced = gpu_step::ced::decoder_chunk_widths_from_geometry(&c, 1, capacity, rows).unwrap();
    assert!(full.windows(2).any(|pair| pair == widths.as_slice()));
    assert!(ced.windows(2).any(|pair| pair == widths.as_slice()));
    for budget in [full, ced] {
        assert!(budget.iter().all(|&width| width > 0));
    }
    let mut small = c.clone();
    small.index_topk = 4;
    assert!(
        gpu_step::chunk_widths_from_geometry(&small, 1, &[], capacity, rows)
            .unwrap()
            .iter()
            .all(|&width| width > 0)
    );
    assert!(
        gpu_step::ced::decoder_chunk_widths_from_geometry(&small, 1, capacity, rows)
            .unwrap()
            .iter()
            .all(|&width| width > 0)
    );
}

#[test]
fn hier_top512_n8192_full_source_dispatch_census_not_duration() {
    let c = Config::default();
    let schedule = Schedule::default();
    let (mut rows, mut tiles) = (0, 0);
    for id in 0..schedule.layers {
        let ratio = schedule.ratio(id);
        if !schedule.produces_index(id) || schedule.owner_slot(id).is_none() || ratio == 0 {
            continue;
        }
        for first in (0..8192).step_by(512) {
            let length = (first + 512) / ratio;
            if matches!(
                selection_route(&c, length, SelectionOverrides::default()),
                SelectionRoute::Hier(_)
            ) {
                assert_eq!(ratio, 1);
                rows += 512;
                tiles += 1;
            }
        }
    }
    assert_eq!((rows, tiles), (20480, 40));
    assert_eq!(rows * 12, 245760); // incumbent sort + chronology, not scoring/prep
    assert_eq!(tiles * 2, 80); // hierarchical local + merge dispatches only
}

// Deliberately independent of the network's boolean comparator and float total_cmp
// (whose signed-zero order is not this selector's order).
fn oracle(scores: &[f32]) -> (Vec<usize>, u32) {
    let status = u32::from(scores.iter().any(|v| v.is_nan()));
    let normalized = |v: f32| if v.is_nan() { f32::NEG_INFINITY } else { v };
    let mut ids: Vec<_> = (0..scores.len()).collect();
    ids.sort_unstable_by(|&a, &b| {
        let l = normalized(scores[a]);
        let r = normalized(scores[b]);
        r.partial_cmp(&l)
            .unwrap()
            .then_with(|| l.to_bits().cmp(&r.to_bits()))
            .then(a.cmp(&b))
    });
    ids.truncate(K);
    ids.sort_unstable();
    (ids, status)
}

fn pattern(length: usize, family: usize) -> Vec<f32> {
    (0..length)
        .map(|id| match family {
            0 => ((id * 7919 + id / 17) % 1021) as f32 - 510.0,
            1 => 1.0,
            2 => {
                if id % 3 == 0 {
                    -0.0
                } else {
                    0.0
                }
            }
            3 => match id % 7 {
                0 => f32::INFINITY,
                1 => 0.0,
                _ => f32::NEG_INFINITY,
            },
            4 => {
                if id + 1 == length {
                    f32::from_bits(0x7fc01234)
                } else {
                    id as f32
                }
            }
            5 => f32::NEG_INFINITY,
            _ => f32::from_bits(0xffc05678),
        })
        .collect()
}

fn network(pairs: &mut [(f32, usize)]) {
    let mut width = 2;
    while width <= pairs.len() {
        let mut stride = width / 2;
        while stride > 0 {
            for j in 0..pairs.len() {
                let other = j ^ stride;
                if other <= j {
                    continue;
                }
                let (l, li) = pairs[j];
                let (r, ri) = pairs[other];
                let better = l > r
                    || (l == r
                        && if l.to_bits() == r.to_bits() {
                            li < ri
                        } else {
                            l.to_bits() < r.to_bits()
                        });
                if better != (j & width == 0) {
                    pairs.swap(j, other);
                }
            }
            stride /= 2;
        }
        width *= 2;
    }
}

#[test]
fn hier_top512_independent_partition_network_and_cpu_total_order_proof() {
    // Any omitted local item has >=512 strictly better items in its own
    // partition under this total order, so it cannot be in the global top512.
    for length in [4095, 4096, 4097, 8191, 8192, 8193, 16384] {
        let p = if length <= 8192 { 2 } else { 4 };
        for family in 0..7 {
            let values = pattern(length, family);
            let mut candidates = Vec::new();
            for part in 0..p {
                let mut local: Vec<_> = (part * 4096..(part + 1) * 4096)
                    .map(|id| {
                        (
                            values
                                .get(id)
                                .copied()
                                .filter(|v| !v.is_nan())
                                .unwrap_or(f32::NEG_INFINITY),
                            id,
                        )
                    })
                    .collect();
                network(&mut local);
                candidates.extend_from_slice(&local[..K]);
            }
            network(&mut candidates);
            let mut ids: Vec<_> = candidates[..K].iter().map(|&(_, id)| id).collect();
            ids.sort_unstable();
            assert_eq!(ids, oracle(&values).0, "L={length} family={family}");
        }
    }
    let zeros = vec![-0.0; 513].into_iter().chain([0.0]).collect::<Vec<_>>();
    assert!(oracle(&zeros).0.contains(&513));
    assert!(!oracle(&zeros).0.contains(&511));
}

fn fixture_config() -> Config {
    Config {
        heads: 2,
        index_topk: K,
        candidate_block: 8,
        candidate_topk: 2048,
        rope_dim: 0,
        ..config()
    }
}

fn fixture_schedule(ratio: usize) -> Schedule {
    Schedule {
        layers: 1,
        kv_source_layers: vec![0],
        index_source_layers: vec![0],
        compression_ratios: vec![ratio],
        candidate_source_layer: (ratio == 1).then_some(0),
    }
}

// All explicit buffers are page-priced before allocation. Host vectors are
// bounded by the same shapes; context/library/PSO/driver memory is separate.
fn ledger(
    ctx: &MetalContext,
    c: &Config,
    capacity: usize,
    schedule: &Schedule,
    extras: &[usize],
    copies: usize,
) {
    let price = |n: usize| {
        ctx.price_shared_buffer_upper((n * 4) as u64)
            .unwrap()
            .priced_upper_bytes
    };
    let workspace: u64 = shapes(c, capacity, schedule)
        .unwrap()
        .into_iter()
        .map(price)
        .sum();
    // scheduled_weights F32, one layer, one output group, optional pair gate.
    let weights = fixture_weight_shapes(c).into_iter().map(price).sum::<u64>();
    let extra: u64 = extras.iter().copied().filter(|&n| n != 0).map(price).sum();
    let total = workspace * copies as u64 + weights + extra;
    eprintln!(
        "hier explicit ledger: workspaces={} weights={weights} extras={extra} total={total} limit={LIMIT}; context/driver separate",
        workspace * copies as u64
    );
    assert!(total <= LIMIT);
}

fn fixture_weight_shapes(c: &Config) -> [usize; 10] {
    [
        c.hidden * c.query_rank,
        c.query_rank * c.heads * c.head_dim,
        c.hidden * c.head_dim,
        c.heads * c.head_dim * c.output_rank,
        c.output_rank * c.hidden,
        c.hidden * c.head_dim,
        c.hidden * c.head_dim,
        c.query_rank * c.index_heads * c.index_dim,
        c.hidden * c.index_heads,
        c.head_dim * c.index_dim,
    ]
}

fn guarded_ids(ctx: &MetalContext, words: usize) -> (MetalTensor, MetalTensor) {
    let bits = vec![DISPATCH_CANARY.to_bits(); words + 2 * DISPATCH_GUARD];
    let backing = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&bits),
        vec![bits.len() as u64],
        GgmlType::I32,
    )
    .unwrap();
    let view = backing.view_subrange(DISPATCH_GUARD as u64, vec![words as u64]);
    (backing, view)
}

fn check_guards(t: &MetalTensor) {
    let n = t.n_elements() as usize;
    let data = unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u32>()
                .add(t.offset as usize / 4),
            n,
        )
    };
    for &v in data[..DISPATCH_GUARD]
        .iter()
        .chain(&data[n - DISPATCH_GUARD..])
    {
        assert_eq!(v, DISPATCH_CANARY.to_bits());
    }
}

fn write_words(t: &MetalTensor, words: &[u32]) {
    assert_eq!(t.n_elements() as usize, words.len());
    // Only called before submission or after checked completion.
    unsafe {
        std::ptr::copy_nonoverlapping(
            words.as_ptr(),
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u32>()
                .add(t.offset as usize / 4),
            words.len(),
        );
    }
}

fn gpu_score_case(ctx: &MetalContext, length: usize, ratio: usize) {
    let c = fixture_config();
    let schedule = fixture_schedule(ratio);
    let (first, n) = (2, 3);
    let start = length * ratio - first - n;
    let capacity = start + first + n + 256;
    let width = Rows::select_width(&c, capacity).max(length);
    let p = if length <= 8192 { 2 } else { 4 };
    // Deliberately allocate P4 at capacity8448 even when this tile needs P2.
    let pair_words = selection_shapes(&c, capacity.max(8448), n, 512).unwrap()[1];
    let out_words = (first + n + 1) * K;
    let guard = 2 * DISPATCH_GUARD;
    ledger(
        ctx,
        &c,
        capacity,
        &schedule,
        &[
            n * width + guard,
            pair_words + guard,
            out_words + guard,
            out_words + guard,
            1 + guard,
        ],
        1,
    );
    let weights = bound(ctx, &c, &schedule);
    let workspace =
        Workspace::new_with_schedule(ctx, &c, capacity, &[&weights[0]], &schedule).unwrap();
    let (score_backing, scores) = guarded_dispatch_tensor(ctx, &vec![f32::NAN; n * width]);
    let (pair_backing, pairs) = guarded_ids(ctx, pair_words);
    let (out_backing, out) = guarded_ids(ctx, out_words);
    let (legacy_backing, legacy) = guarded_ids(ctx, out_words);
    let (status_backing, status) = guarded_ids(ctx, 1);
    // Preexisting status bits survive; first pass has only FUTURE NaNs.
    write_words(&status, &[8]);
    for (target, family) in [
        (length, 0),
        (length, 1),
        (length, 2),
        (length, 3),
        (length, 5),
        (length, 4),
        (length, 6),
        (length.min(8192), 0),
        (length.min(4097), 0),
        (length, 0),
    ] {
        // Same allocation, different absolute origins and compact P4/P2 layouts.
        let start = target * ratio - first - n;
        let p = if target <= 8192 { 2 } else { 4 };
        let mut data = vec![f32::from_bits(0x7fc0beef); n * width];
        let mut expected = Vec::new();
        let mut expected_status = ids(&status, 1)[0] as u32;
        for local in 0..n {
            let visible = (start + first + local + 1) / ratio;
            let values = pattern(visible, family);
            data[local * width..local * width + visible].copy_from_slice(&values);
            let (answer, bits) = oracle(&values);
            if visible > K {
                expected_status |= bits;
            }
            expected.push(answer);
        }
        write_words(
            &scores,
            &data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
        write_words(&pairs, &vec![0xdeadbeef; pair_words]);
        write_words(&out, &vec![DISPATCH_CANARY.to_bits(); out_words]);
        write_words(&legacy, &vec![DISPATCH_CANARY.to_bits(); out_words]);
        execute(ctx, |enc| {
            // Use the actual prepare kernel for the 511/512/513 straddle.
            pipeline(ctx, enc, "selection_prepare_rows").unwrap();
            enc.set_bytes(0, &[start as u32, ratio as u32, K as u32, 8]);
            enc.set_bytes(1, &[K as u32, 1, n as u32, 0]);
            enc.set_tensor(2, &out);
            enc.set_tensor(3, &workspace.candidates);
            enc.set_bytes(4, &(first as u32));
            enc.dispatch(
                MTLSize {
                    width: 2,
                    height: n,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            metal::dispatch_census_begin();
            select_hier_rows(
                ctx, enc, &scores, &pairs, &out, &status, start, ratio, first, n, width, p,
            )
            .unwrap();
            let census = metal::dispatch_census_take();
            assert_eq!(census.len(), 2);
            assert_eq!(
                census[0].kernel,
                "kernel_deepseek_v41_attention_select_hier_local"
            );
            assert_eq!(
                census[1].kernel,
                "kernel_deepseek_v41_attention_select_hier_merge"
            );
            assert_eq!(
                (census[0].grid_width, census[0].grid_height),
                (p as u64, n as u64)
            );
            assert_eq!((census[1].grid_width, census[1].grid_height), (1, n as u64));
            assert!(
                census
                    .iter()
                    .all(|d| !d.encoder_concurrent && d.tg_threads == 256)
            );
            for local in 0..n {
                let visible = (start + first + local + 1) / ratio;
                if visible <= K {
                    continue;
                }
                let score_row = scores.view_subrange((local * width) as u64, vec![width as u64]);
                metal::dispatch_census_begin();
                workspace
                    .sort(ctx, enc, &score_row, visible, false)
                    .unwrap();
                let pso = pipeline(ctx, enc, "chronological").unwrap();
                enc.set_bytes(0, &(K as u32));
                enc.set_tensor(1, &workspace.sort_ids);
                enc.set_tensor(2, &prefill::row(&legacy, K, first + local));
                dispatch(enc, &pso, K);
                let incumbent = metal::dispatch_census_take();
                if visible == 8192 {
                    assert_eq!(incumbent.len(), 12);
                }
                if visible == 16384 {
                    assert_eq!(incumbent.len(), 17);
                }
            }
        });
        assert_eq!(ids(&status, 1), [expected_status as usize]);
        assert_eq!(
            ids(&workspace.status, 1)[0] & 1,
            (expected_status & 1) as usize
        );
        for local in 0..n {
            let visible = (start + first + local + 1) / ratio;
            let actual = ids(&prefill::row(&out, K, first + local), visible.min(K));
            assert_eq!(
                actual, expected[local],
                "hier L={visible} ratio={ratio} family={family}"
            );
            if visible > K {
                assert_eq!(actual, ids(&prefill::row(&legacy, K, first + local), K));
            }
            assert!(actual.iter().all(|&id| id < visible));
        }
        for row in [0, 1, first + n] {
            assert!(
                ids(&prefill::row(&out, K, row), K)
                    .iter()
                    .all(|&v| v == DISPATCH_CANARY.to_bits() as usize)
            );
        }
        assert_dispatch_contents(&score_backing, &data);
        for t in [
            &pair_backing,
            &out_backing,
            &legacy_backing,
            &status_backing,
        ] {
            check_guards(t);
        }
        let compact_words = n * p * K * 2;
        assert!(
            ids(&pairs, pair_words)[compact_words..]
                .iter()
                .all(|&v| v == 0xdeadbeef)
        );
    }
    // Geometry failures must occur before any selector dispatch, not retry.
    execute(ctx, |enc| {
        metal::dispatch_census_begin();
        let tiny_pairs = pairs.view_subrange(0, vec![2]);
        assert!(
            select_hier_rows(
                ctx,
                enc,
                &scores,
                &tiny_pairs,
                &out,
                &status,
                start,
                ratio,
                first,
                n,
                width,
                p
            )
            .is_err()
        );
        assert!(metal::dispatch_census_take().is_empty());
    });
}

fn require_bounded_gpu_environment() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    for name in [
        "QWEN_METAL_LEASE_SKIP_WIRED_GATE",
        "QWEN_V41_SELECT_PER_ROW",
        "QWEN_V41_SELECT_SPLIT_SORT",
        "QWEN_V41_SELECT_FUSE_2048",
        "QWEN_V41_SELECT_NO_HIER",
        "QWEN_V41_SINK_TILE",
    ] {
        assert!(
            std::env::var_os(name).is_none(),
            "unset {name} for bounded proof"
        );
    }
}

#[test]
#[ignore = "self-leased bounded selector GPU proof; MTL_DEBUG_LAYER=1, no memory bypass, no model"]
fn hier_top512_bounded_gpu_exact_legacy_cpu_canaries_and_reuse() {
    require_bounded_gpu_environment();
    let _lease =
        metal::acquire_metal_benchmark_lease().expect("exclusive production GPU lease required");
    let ctx = MetalContext::new().unwrap();
    for ratio in [1, 2] {
        for length in [512, 513, 4095, 4096, 4097, 8191, 8192, 8193, 16384] {
            objc2::rc::autoreleasepool(|_| gpu_score_case(&ctx, length, ratio));
        }
    }
}

#[test]
#[ignore = "self-leased bounded query selector integration/pruning proof; MTL_DEBUG_LAYER=1, no model"]
fn hier_top512_bounded_gpu_query_route_and_pruning_boundary() {
    require_bounded_gpu_environment();
    let _lease =
        metal::acquire_metal_benchmark_lease().expect("exclusive production GPU lease required");
    let ctx = MetalContext::new().unwrap();
    let c = fixture_config();
    for (ratio, start, n) in [
        (1, 507, 5),
        (1, 510, 3),
        (1, 4094, 3),
        (2, 8189, 5),
        (1, 8190, 3),
        (1, 16383, 2),
    ] {
        objc2::rc::autoreleasepool(|_| {
            let schedule = fixture_schedule(ratio);
            let capacity = start + n + 1;
            let mut extras: Vec<_> = Rows::widths(&c, capacity)
                .into_iter()
                .map(|w| w * n)
                .collect();
            extras.extend(
                selection_shapes(
                    &c,
                    capacity,
                    n,
                    super::super::super::query_batch::tile_rows(),
                )
                .unwrap(),
            );
            extras.extend(Scratch::shapes(&c, n).unwrap());
            extras.push(
                prefill::attention_half_scratch_elements(&c, n)
                    .unwrap()
                    .div_ceil(2),
            );
            extras.extend([
                c.hidden * n,
                c.hidden * n,
                c.hidden * n,
                K * n,
                capacity.div_ceil(8) * n,
            ]);
            // seed replaces each cache while its initial allocation may still
            // be autoreleased; retain another two workspace bounds as slack.
            ledger(&ctx, &c, capacity, &schedule, &extras, 4);
            compare_case(&ctx, &c, &schedule, start, n);
        });
    }
}
