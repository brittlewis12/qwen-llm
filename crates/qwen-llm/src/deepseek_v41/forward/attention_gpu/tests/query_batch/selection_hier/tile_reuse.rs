//! Full-width scratch reuse and CED buffer binding, without a CED/model replay.
use super::*;

const ROWS: usize = 513;
const WIDTH: usize = 16384;
const START: usize = 7680;
const PREFIX: usize = 3;
const SUFFIX: usize = 2;
const POISON: u32 = 0x7fc0beef;

fn tile_config() -> Config {
    Config {
        heads: 1,
        index_heads: 1,
        window: 8,
        ..fixture_config()
    }
}

fn retained_range() -> std::ops::Range<usize> {
    START - PREFIX..START + ROWS + SUFFIX
}

fn allocation_counts(c: &Config, borrowed: bool) -> Vec<usize> {
    let guard = 2 * DISPATCH_GUARD;
    let selection = selection_shapes(c, WIDTH, ROWS, TILE).unwrap();
    let mut counts: Vec<_> = if borrowed {
        Rows::widths_with_controls(c, WIDTH)
    } else {
        Rows::widths(c, WIDTH)
    }
    .into_iter()
    .map(|w| w.checked_mul(ROWS).unwrap())
    .collect();
    if borrowed {
        counts.extend(QueryControls::shapes(c, WIDTH, retained_range()).unwrap());
    } else {
        counts.extend([
            K * ROWS + guard,
            WIDTH.div_ceil(c.candidate_block) * ROWS + guard,
        ]);
    }
    counts.extend(Scratch::shapes(c, ROWS).unwrap());
    counts.push(
        prefill::attention_half_scratch_elements(c, ROWS)
            .unwrap()
            .div_ceil(2),
    );
    // Include both original Rows slabs and guarded replacements: autoreleased
    // originals may survive until the enclosing case's pool drains.
    counts.extend(selection);
    counts.extend(selection.map(|n| n + guard));
    counts.extend([K * ROWS + guard, 1 + guard, 2 + guard, 1 + guard]);
    counts
}

#[test]
fn hier_top512_513_rows_fixture_explicit_buffers_below_128_mib() {
    let c = tile_config();
    let schedule = fixture_schedule(1);
    assert_eq!(
        selection_shapes(&c, WIDTH, ROWS, TILE).unwrap(),
        [512 * WIDTH, 512 * 4 * K * 2]
    );
    for borrowed in [false, true] {
        let counts = shapes(&c, WIDTH, &schedule)
            .unwrap()
            .into_iter()
            .chain(fixture_weight_shapes(&c))
            .chain(allocation_counts(&c, borrowed));
        // The runtime fixture also checks actual device/page pricing before
        // allocating. This pure ceiling charges 16 KiB slack per allocation.
        let upper: u64 = counts
            .filter(|&n| n != 0)
            .map(|n| n as u64 * 4 + 16384)
            .sum();
        assert!(
            upper <= 100 * 1024 * 1024,
            "borrowed={borrowed} explicit upper={upper}"
        );
        assert!(upper <= LIMIT);
    }
}

fn fill_words(t: &MetalTensor, value: u32) {
    // All host writes in this fixture precede its first command submission.
    unsafe {
        std::slice::from_raw_parts_mut(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u32>()
                .add(t.offset as usize / 4),
            t.n_elements() as usize,
        )
        .fill(value);
    }
}

fn words(t: &MetalTensor) -> &[u32] {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u32>()
                .add(t.offset as usize / 4),
            t.n_elements() as usize,
        )
    }
}

fn guarded_slab(ctx: &MetalContext, count: usize, dtype: GgmlType) -> (MetalTensor, MetalTensor) {
    let backing = prefill::allocate(ctx, count + 2 * DISPATCH_GUARD, 1, dtype).unwrap();
    fill_words(&backing, DISPATCH_CANARY.to_bits());
    let view = backing.view_subrange(DISPATCH_GUARD as u64, vec![count as u64]);
    (backing, view)
}

fn key_value(id: usize, component: usize) -> f32 {
    if component == 0 {
        (1 + id % 1021) as f32
    } else {
        (1 + (id * 37) % 1031) as f32
    }
}

fn query_factor(row: usize) -> f32 {
    [0.5, 1.0, 2.0][row % 3]
}

fn initialize_index_inputs(workspace: &Workspace, rows: &Rows, c: &Config) {
    let keys = &workspace.owners[0].keys;
    fill_words(keys, POISON);
    for id in 0..8193 {
        let key = prefill::row(keys, c.index_dim, id);
        fill_words(&key, 0);
        write_words(
            &key.view_subrange(0, vec![2]),
            &[key_value(id, 0).to_bits(), key_value(id, 1).to_bits()],
        );
    }
    fill_words(&rows.index_query, 0);
    fill_words(&rows.index_weights, 1.0f32.to_bits());
    for row in 0..ROWS {
        write_words(
            &rows
                .index_query
                .view_subrange((row * c.index_dim + row % 2) as u64, vec![1]),
            &[query_factor(row).to_bits()],
        );
    }
    fill_words(&workspace.index_scores, POISON);
    write_words(&workspace.status, &[8]);
}

fn encode_prepare_and_scores(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &Workspace,
    rows: &Rows,
    c: &Config,
    first: usize,
    n: usize,
) {
    let length = START + first + n;
    let stride = WIDTH.div_ceil(c.candidate_block);
    pipeline(ctx, enc, "selection_prepare_rows").unwrap();
    enc.set_bytes(0, &[START as u32, 1, K as u32, c.candidate_block as u32]);
    enc.set_bytes(1, &[K as u32, stride as u32, n as u32, 1]);
    enc.set_tensor(2, &rows.selected);
    enc.set_tensor(3, &rows.candidates);
    enc.set_bytes(4, &(first as u32));
    enc.dispatch(
        MTLSize {
            width: length.div_ceil(c.candidate_block).div_ceil(256),
            height: n,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    pipeline(ctx, enc, "index_heads_rows").unwrap();
    enc.set_bytes(0, &[START as u32, 1, K as u32, length as u32]);
    enc.set_bytes(1, &[c.index_dim as u32, 1, c.index_dim as u32, 1]);
    enc.set_bytes(2, &[WIDTH as u32, n as u32, 0, 0]);
    enc.set_tensor(3, &rows.index_query);
    enc.set_tensor(4, &rows.index_weights);
    enc.set_tensor(5, &workspace.owners[0].keys);
    enc.set_tensor(6, &rows.select_scores);
    enc.set_bytes(7, &(first as u32));
    enc.dispatch(
        MTLSize {
            width: length.div_ceil(8),
            height: n,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_legacy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &Workspace,
    rows: &Rows,
    c: &Config,
    poison: &MetalTensor,
    legacy: &MetalTensor,
) {
    for row in 0..ROWS {
        let visible = START + row + 1;
        pipeline(ctx, enc, "index_heads").unwrap();
        enc.set_bytes(0, &[visible as u32, c.index_dim as u32, 1, 0]);
        enc.set_tensor(1, &prefill::row(&rows.index_query, c.index_dim, row));
        enc.set_tensor(2, &prefill::row(&rows.index_weights, 1, row));
        enc.set_tensor(3, &workspace.owners[0].keys);
        enc.set_tensor(4, &workspace.index_scores);
        enc.dispatch(
            MTLSize {
                width: visible.div_ceil(8),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        if row == TILE {
            copy(
                ctx,
                enc,
                poison,
                &workspace.index_scores.view_subrange(8192, vec![1]),
                1,
            )
            .unwrap();
        }
        workspace
            .sort(ctx, enc, &workspace.index_scores, visible, false)
            .unwrap();
        let pso = pipeline(ctx, enc, "chronological").unwrap();
        enc.set_bytes(0, &(K as u32));
        enc.set_tensor(1, &workspace.sort_ids);
        enc.set_tensor(2, &prefill::row(legacy, K, row));
        dispatch(enc, &pso, K);
    }
}

fn gpu_tile_reuse_case(ctx: &MetalContext, borrowed: bool) {
    let c = tile_config();
    let schedule = fixture_schedule(1);
    assert_eq!(super::super::super::super::query_batch::tile_rows(), TILE);
    let counts = allocation_counts(&c, borrowed);
    ledger(ctx, &c, WIDTH, &schedule, &counts, 1);
    let weights = bound(ctx, &c, &schedule);
    let workspace =
        Workspace::new_with_schedule(ctx, &c, WIDTH, &[&weights[0]], &schedule).unwrap();
    let controls = borrowed.then(|| QueryControls::new(ctx, &c, WIDTH, retained_range()).unwrap());
    let whole = controls.as_ref().map(|controls| {
        controls
            .checked_tile_view(retained_range().start, retained_range().len())
            .unwrap()
    });
    let mut guards = Vec::new();
    let mut rows = if let Some(controls) = &controls {
        let view = controls.checked_tile_view(START, ROWS).unwrap();
        assert_eq!(view.absolute_rows(), START..START + ROWS);
        assert_eq!(view.selected.offset, (PREFIX * K * 4) as u64);
        assert_eq!(
            view.candidates.offset,
            (PREFIX * WIDTH.div_ceil(c.candidate_block) * 4) as u64
        );
        let rows = Rows::with_controls(ctx, &c, WIDTH, &view).unwrap();
        assert!(std::ptr::eq(&*rows.selected.buffer, &*view.selected.buffer));
        assert!(std::ptr::eq(
            &*rows.candidates.buffer,
            &*view.candidates.buffer
        ));
        assert_eq!(rows.selected.offset, view.selected.offset);
        assert_eq!(rows.candidates.offset, view.candidates.offset);
        rows
    } else {
        let mut rows = Rows::new(ctx, &c, WIDTH, ROWS).unwrap();
        for t in [&mut rows.selected, &mut rows.candidates] {
            let (backing, view) = guarded_slab(ctx, t.n_elements() as usize, GgmlType::I32);
            *t = view;
            guards.push(backing);
        }
        rows
    };
    if let Some(whole) = &whole {
        fill_words(&whole.selected, DISPATCH_CANARY.to_bits());
        fill_words(&whole.candidates, DISPATCH_CANARY.to_bits());
    }
    for t in [&mut rows.select_scores, rows.select_pairs.as_mut().unwrap()] {
        let (backing, view) = guarded_slab(ctx, t.n_elements() as usize, t.dtype);
        *t = view;
        guards.push(backing);
    }
    fill_words(&rows.select_scores, POISON);
    fill_words(rows.select_pairs.as_ref().unwrap(), 0xdeadbeef);
    initialize_index_inputs(&workspace, &rows, &c);
    let (legacy_guard, legacy) = guarded_slab(ctx, ROWS * K, GgmlType::I32);
    let (status_guard, status) = guarded_slab(ctx, 1, GgmlType::I32);
    let (checkpoints_guard, checkpoints) = guarded_slab(ctx, 2, GgmlType::I32);
    let (poison_guard, poison) = guarded_slab(ctx, 1, GgmlType::F32);
    write_words(&status, &[8]);
    write_words(&poison, &[POISON]);
    guards.extend([legacy_guard, status_guard, checkpoints_guard, poison_guard]);
    execute(ctx, |enc| {
        metal::dispatch_census_begin();
        for (tile, (first, n, p)) in [(0, TILE, 2), (TILE, 1, 4)].into_iter().enumerate() {
            assert_eq!(
                selection_route(&c, START + first + n, SelectionOverrides::default()),
                SelectionRoute::Hier(p)
            );
            encode_prepare_and_scores(ctx, enc, &workspace, &rows, &c, first, n);
            if first == TILE {
                // A losing visible NaN only on the tail; all future poison on
                // the full first tile must leave its status checkpoint clean.
                copy(
                    ctx,
                    enc,
                    &poison,
                    &rows.select_scores.view_subrange(8192, vec![1]),
                    1,
                )
                .unwrap();
            }
            select_hier_rows(
                ctx,
                enc,
                &rows.select_scores,
                rows.select_pairs.as_ref().unwrap(),
                &rows.selected,
                &status,
                START,
                1,
                first,
                n,
                WIDTH,
                p,
            )
            .unwrap();
            copy(
                ctx,
                enc,
                &status,
                &checkpoints.view_subrange(tile as u64, vec![1]),
                1,
            )
            .unwrap();
        }
        let census = metal::dispatch_census_take();
        let hierarchy: Vec<_> = census
            .iter()
            .filter(|d| d.kernel.contains("select_hier_"))
            .collect();
        assert_eq!(hierarchy.len(), 4);
        for (d, (suffix, width, height)) in hierarchy.iter().zip([
            ("local", 2, 512),
            ("merge", 1, 512),
            ("local", 4, 1),
            ("merge", 1, 1),
        ]) {
            assert_eq!(
                d.kernel,
                format!("kernel_deepseek_v41_attention_select_hier_{suffix}")
            );
            assert_eq!(
                (d.grid_width, d.grid_height, d.grid_depth),
                (width, height, 1)
            );
            assert_eq!(d.tg_threads, 256);
            assert!(!d.encoder_concurrent);
        }
        assert_eq!(
            census
                .iter()
                .filter(|d| d.kernel.ends_with("index_heads_rows"))
                .count(),
            2
        );
        assert_eq!(
            census
                .iter()
                .filter(|d| d.kernel.ends_with("selection_prepare_rows"))
                .count(),
            2
        );
        assert!(
            !census
                .iter()
                .any(|d| d.kernel.contains("sort_") || d.kernel.ends_with("chronological"))
        );
    });
    assert_eq!(ids(&checkpoints, 2), [8, 9]);
    assert_eq!(ids(&status, 1), [9]);
    execute(ctx, |enc| {
        metal::dispatch_census_begin();
        encode_legacy(ctx, enc, &workspace, &rows, &c, &poison, &legacy);
        let census = metal::dispatch_census_take();
        assert_eq!(
            census
                .iter()
                .filter(|d| d.kernel.ends_with("chronological"))
                .count(),
            ROWS
        );
        assert_eq!(
            census
                .iter()
                .filter(|d| d.kernel.contains("sort_") || d.kernel.ends_with("chronological"))
                .count(),
            512 * 12 + 17
        );
    });
    assert_eq!(ids(&workspace.status, 1), ids(&status, 1));
    let stride = WIDTH.div_ceil(c.candidate_block);
    let mut cpu_status = 8;
    for row in 0..ROWS {
        let visible = START + row + 1;
        // One positive integer key component times a positive common scale.
        // Small integer spacing prevents rounded-score ties; powers of two
        // vary row scores without changing numerical or bit tie ordering.
        let mut scores: Vec<_> = (0..visible)
            .map(|id| key_value(id, row % 2) * query_factor(row))
            .collect();
        if row == TILE {
            scores[8192] = f32::from_bits(POISON);
        }
        let (expected, bits) = oracle(&scores);
        cpu_status |= bits;
        let actual = ids(&prefill::row(&rows.selected, K, row), K);
        assert_eq!(actual, expected, "CPU borrowed={borrowed} row={row}");
        assert_eq!(
            actual,
            ids(&prefill::row(&legacy, K, row), K),
            "legacy row={row}"
        );
        assert!(actual.iter().all(|&id| id < visible));
        assert!(actual.iter().any(|&id| id > 4096));
        if row == TILE {
            assert!(actual.iter().any(|&id| id >= 7681));
        }
        let candidates = prefill::row(&rows.candidates, stride, row);
        let mask = words(&candidates);
        let blocks = visible.div_ceil(c.candidate_block);
        assert!(
            mask[..blocks].iter().all(|&v| v == 1),
            "candidate visibility row={row}"
        );
        assert!(
            mask[blocks..]
                .iter()
                .all(|&v| v == DISPATCH_CANARY.to_bits()),
            "candidate future row={row}"
        );
    }
    assert_eq!(ids(&status, 1), [cpu_status as usize]);
    // The tail reuses scratch row zero, not physical output row 512. Other
    // score rows survive; poison beyond each row's own L stays untouched.
    let slab = words(&rows.select_scores);
    for local in 0..TILE {
        let row = if local == 0 { TILE } else { local };
        let visible = START + row + 1;
        assert!(
            slab[local * WIDTH + visible..(local + 1) * WIDTH]
                .iter()
                .all(|&v| v == POISON)
        );
    }
    let pairs = words(rows.select_pairs.as_ref().unwrap());
    assert!(pairs[TILE * 2 * K * 2..].iter().all(|&v| v == 0xdeadbeef));
    if let Some(whole) = &whole {
        for (t, width) in [(&whole.selected, K), (&whole.candidates, stride)] {
            let data = words(t);
            assert!(
                data[..PREFIX * width]
                    .iter()
                    .chain(&data[(PREFIX + ROWS) * width..])
                    .all(|&v| v == DISPATCH_CANARY.to_bits())
            );
        }
    }
    for backing in &guards {
        check_guards(backing);
    }
}

#[test]
#[ignore = "self-leased T512/513-row selector proof; MTL_DEBUG_LAYER=1, explicit buffers <=128 MiB"]
fn hier_top512_bounded_gpu_513_rows_tile_scratch_reuse() {
    require_bounded_gpu_environment();
    let _lease =
        metal::acquire_metal_benchmark_lease().expect("exclusive production GPU lease required");
    let ctx = MetalContext::new().unwrap();
    objc2::rc::autoreleasepool(|_| gpu_tile_reuse_case(&ctx, false));
}

#[test]
#[ignore = "self-leased T512/513-row CED borrowed offset selector proof; MTL_DEBUG_LAYER=1, <=128 MiB"]
fn hier_top512_bounded_gpu_513_rows_ced_offset_controls() {
    require_bounded_gpu_environment();
    let _lease =
        metal::acquire_metal_benchmark_lease().expect("exclusive production GPU lease required");
    let ctx = MetalContext::new().unwrap();
    objc2::rc::autoreleasepool(|_| gpu_tile_reuse_case(&ctx, true));
}
