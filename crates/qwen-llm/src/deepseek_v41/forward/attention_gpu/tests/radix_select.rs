use super::*;

/// The sort path's total order: numerically descending, equal values with
/// smaller bits first (+0 before -0), then smaller ID; masked keys and NaNs
/// are -Inf. Returns the top-k IDs in ascending (chronological) order.
fn reference_select(
    scores: &[f32],
    length: usize,
    k: usize,
    mask: Option<(&[u32], usize)>,
) -> Vec<u32> {
    let mut keys: Vec<(f32, u32)> = (0..length)
        .map(|t| {
            let masked = mask.is_some_and(|(m, block)| m[t / block] == 0);
            let v = if masked || scores[t].is_nan() {
                f32::NEG_INFINITY
            } else {
                scores[t]
            };
            (v, t as u32)
        })
        .collect();
    let better = |(l, li): (f32, u32), (r, ri): (f32, u32)| {
        l > r
            || (l == r
                && if l.to_bits() == r.to_bits() {
                    li < ri
                } else {
                    l.to_bits() < r.to_bits()
                })
    };
    keys.sort_by(|&a, &b| {
        if better(a, b) {
            std::cmp::Ordering::Less
        } else if better(b, a) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    let mut ids: Vec<u32> = keys[..k].iter().map(|&(_, id)| id).collect();
    ids.sort_unstable();
    ids
}

fn scores_for(pattern: &str, n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..n)
        .map(|i| match pattern {
            // Continuous values.
            "random" => (next() % 1_000_000) as f32 / 1000.0 - 500.0,
            // Few distinct values: large ties at the threshold.
            "ties" => (next() % 7) as f32 * 0.25,
            // Signed zeros, infinities and NaNs around a tie-heavy core.
            "special" => match next() % 11 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::NEG_INFINITY,
                3 => f32::INFINITY,
                4 => f32::NAN,
                5 => -1.5,
                _ => (next() % 5) as f32,
            },
            // Every key equal: the lowest IDs win.
            "flat" => 3.0,
            // Half exact zeros (relu-dead keys) among positive values.
            "sparse" => {
                if next().is_multiple_of(2) {
                    0.0
                } else {
                    (next() % 100_000) as f32 / 1000.0
                }
            }
            // Monotone ramp with a hot tail.
            _ => i as f32 * 1e-3,
        })
        .collect()
}

#[test]
fn radix_selection_matches_sort_order_exactly() {
    let ctx = MetalContext::new().unwrap();
    let k = 512usize;
    // Real candidate geometry: 8-key blocks, 2048 kept, so masks apply only to
    // rows longer than 16K keys.
    let config = Config::default();
    let block = config.candidate_block;
    for (length, pattern, masked, seed) in [
        (513usize, "random", false, 1u64),
        (1000, "ties", false, 2),
        (4097, "special", false, 3),
        (20_000, "random", true, 4),
        (40_000, "ties", true, 5),
        (40_000, "special", true, 6),
        (33_000, "flat", false, 7),
        (65_536, "ramp", true, 8),
    ] {
        let scores = scores_for(pattern, length, seed);
        let blocks = length.div_ceil(block);
        // Candidate mask: about half the blocks plus the last one, as the
        // published candidates force the newest block.
        let mask: Vec<u32> = (0..blocks)
            .map(|b| u32::from(b + 1 == blocks || (b * 7 + seed as usize).is_multiple_of(2)))
            .collect();
        let expected = reference_select(&scores, length, k, masked.then_some((&mask[..], block)));
        let s_t = upload(&ctx, &scores, vec![length as u64]).unwrap();
        let m_t = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&mask),
            vec![blocks as u64],
            GgmlType::I32,
        )
        .unwrap();
        let out = tensor(&ctx, k, GgmlType::I32).unwrap();
        let status = tensor(&ctx, 1, GgmlType::I32).unwrap();
        assert_eq!(config.index_topk, k);
        execute(&ctx, |enc| {
            encode_select_radix_row(
                &ctx,
                enc,
                &config,
                length - 1,
                1,
                &s_t,
                &m_t,
                &out,
                &status,
                masked,
            )
            .unwrap();
        });
        let got = unsafe {
            std::slice::from_raw_parts(out.buffer.contents().as_ptr().cast::<u32>(), k).to_vec()
        };
        assert_eq!(
            got, expected,
            "length {length} pattern {pattern} masked {masked}"
        );
        let status = unsafe { *status.buffer.contents().as_ptr().cast::<u32>() };
        let visible_nan =
            (0..length).any(|t| scores[t].is_nan() && !(masked && mask[t / block] == 0));
        assert_eq!(status & 1 != 0, visible_nan, "NaN status, length {length}");
    }
}

#[test]
fn ilp_index_scores_match_index_heads_rows() {
    index_rows_match_index_heads_rows(
        encode_index_scores_rows_ilp,
        &[
            (20_000usize, 1usize, 3usize, 0usize),
            (2_100, 4, 5, 2),
            (509, 1, 4, 0),
        ],
    );
}

#[test]
fn mma_index_scores_match_index_heads_rows() {
    // As the ILP cases (partial 8-key blocks and 256-key threadgroups, a
    // row at or below top-k), plus two rows of a 128K-key cache.
    index_rows_match_index_heads_rows(
        encode_index_scores_rows_mma,
        &[
            (20_000usize, 1usize, 3usize, 0usize),
            (2_100, 4, 5, 2),
            (509, 1, 4, 0),
            (131_066, 1, 2, 3),
        ],
    );
}

type IndexRowsEncoder = fn(
    &MetalContext,
    &KernelEncoder,
    &Config,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    &MetalTensor,
    &MetalTensor,
    &MetalTensor,
    &MetalTensor,
) -> Result<()>;

/// `encode` against the scalar `index_heads_rows` kernel for each (start,
/// ratio, rows, first) case.
fn index_rows_match_index_heads_rows(
    encode: IndexRowsEncoder,
    cases: &[(usize, usize, usize, usize)],
) {
    let ctx = MetalContext::new().unwrap();
    let (heads, dim) = (32usize, 128usize);
    let config = Config::default();
    assert!(index_ilp_supported(&config));
    let topk = config.index_topk;
    // Rows at increasing visible lengths (one partial 8-key group, one partial
    // 64-key threadgroup) plus a row at or below top-k that must not write.
    for &(start, ratio, rows, first) in cases {
        let last = (start + first + rows) / ratio;
        let stride = last + 3;
        let make = |n: usize, seed: usize| {
            (0..n)
                .map(|i| ((i * 2_654_435_761usize + seed) % 1_000_003) as f32 / 500_000.0 - 1.0)
                .collect::<Vec<_>>()
        };
        let q_rows = first + rows;
        let q_t = upload(
            &ctx,
            &make(q_rows * heads * dim, 3),
            vec![(q_rows * heads * dim) as u64],
        )
        .unwrap();
        let w_t = upload(
            &ctx,
            &make(q_rows * heads, 5),
            vec![(q_rows * heads) as u64],
        )
        .unwrap();
        let k_t = upload(&ctx, &make(last * dim, 7), vec![(last * dim) as u64]).unwrap();
        let poison = vec![-12345.0f32; rows * stride];
        let old = upload(&ctx, &poison, vec![(rows * stride) as u64]).unwrap();
        let new = upload(&ctx, &poison, vec![(rows * stride) as u64]).unwrap();
        execute(&ctx, |enc| {
            pipeline(&ctx, enc, "index_heads_rows").unwrap();
            enc.set_bytes(0, &[start as u32, ratio as u32, topk as u32, last as u32]);
            enc.set_bytes(
                1,
                &[dim as u32, heads as u32, (heads * dim) as u32, heads as u32],
            );
            enc.set_bytes(2, &[stride as u32, rows as u32, 0, 0]);
            enc.set_tensor(3, &q_t);
            enc.set_tensor(4, &w_t);
            enc.set_tensor(5, &k_t);
            enc.set_tensor(6, &old);
            enc.set_bytes(7, &(first as u32));
            enc.dispatch(
                MTLSize {
                    width: last.div_ceil(8),
                    height: rows,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encode(
                &ctx, enc, &config, start, ratio, last, first, rows, stride, &q_t, &w_t, &k_t, &new,
            )
            .unwrap();
        });
        let a = f32_slice(&old).unwrap();
        let b = f32_slice(&new).unwrap();
        // Same F32 operations; fast-math may contract/reorder them differently
        // per kernel, so allow a few ulps (the half tensor tile is ~1e-3 off).
        // Rows at or below top-k keep their poison in both.
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= 2e-6 + 1e-5 * x.abs(),
                "start {start} element {i}: {x} vs {y}"
            );
        }
    }
}

#[test]
fn radix_candidate_publication_matches_block_sort() {
    let ctx = MetalContext::new().unwrap();
    let config = Config::default();
    let (block, keep) = (config.candidate_block, config.candidate_topk);
    for (length, pattern, seed) in [
        (16_392usize, "random", 11u64),
        (40_000, "ties", 12),
        (131_072, "random", 13),
        (90_001, "flat", 14),
    ] {
        let scores = scores_for(pattern, length, seed);
        let blocks = length.div_ceil(block);
        // Reference: per-block maxima, newest block forced to +Inf, keep the
        // first `keep` blocks in the sort order.
        let block_scores: Vec<f32> = (0..blocks)
            .map(|b| {
                if b + 1 == blocks {
                    return f32::INFINITY;
                }
                scores[b * block..((b + 1) * block).min(length)]
                    .iter()
                    .fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m })
            })
            .collect();
        let kept = reference_select(&block_scores, blocks, keep, None);
        let mut expected = vec![0u32; blocks];
        for id in kept {
            expected[id as usize] = 1;
        }
        let s_t = upload(&ctx, &scores, vec![length as u64]).unwrap();
        let b_t = tensor(&ctx, blocks, GgmlType::F32).unwrap();
        let mask = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&vec![7u32; blocks]),
            vec![blocks as u64],
            GgmlType::I32,
        )
        .unwrap();
        let unused = tensor(&ctx, 1, GgmlType::I32).unwrap();
        let status = tensor(&ctx, 1, GgmlType::I32).unwrap();
        let position = length - 1;
        execute(&ctx, |enc| {
            pipeline(&ctx, enc, "select_block_scores_rows").unwrap();
            enc.set_bytes(0, &[position as u32, 1, block as u32, keep as u32]);
            enc.set_bytes(1, &[length as u32, blocks as u32, 1, 0]);
            enc.set_tensor(2, &s_t);
            enc.set_tensor(3, &b_t);
            enc.dispatch(
                MTLSize {
                    width: blocks.div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            pipeline(&ctx, enc, "select_radix_rows").unwrap();
            enc.set_bytes(
                0,
                &[position as u32, 1, config.index_topk as u32, block as u32],
            );
            enc.set_bytes(1, &[blocks as u32, 0, 1, 0]);
            enc.set_bytes(2, &[config.index_topk as u32, 0, keep as u32, 1]);
            enc.set_tensor(3, &b_t);
            enc.set_tensor(4, &mask);
            enc.set_tensor(5, &unused);
            enc.set_tensor(6, &status);
            enc.dispatch(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
            );
        });
        let got_blocks = f32_slice(&b_t).unwrap();
        assert!(
            got_blocks
                .iter()
                .zip(&block_scores)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "block scores, length {length}"
        );
        let got = unsafe {
            std::slice::from_raw_parts(mask.buffer.contents().as_ptr().cast::<u32>(), blocks)
                .to_vec()
        };
        assert_eq!(
            got, expected,
            "candidate mask, length {length} pattern {pattern}"
        );
    }
}

fn u32_tensor(ctx: &MetalContext, values: &[u32]) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(values),
        vec![values.len() as u64],
        GgmlType::I32,
    )
    .unwrap()
}

fn u32_vec(t: &MetalTensor, n: usize) -> Vec<u32> {
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
        .to_vec()
    }
}

/// About half the blocks plus the newest one, as published candidates keep.
fn candidate_mask(blocks: usize, seed: u64) -> Vec<u32> {
    (0..blocks)
        .map(|b| u32::from(b + 1 == blocks || (b * 7 + seed as usize).is_multiple_of(2)))
        .collect()
}

/// A radix selector implementation under test.
#[derive(Clone, Copy, Debug)]
enum Selector {
    /// `select_radix_rows`: one threadgroup per row.
    Rows,
    /// `encode_select_radix_wide` (the decode choice by length).
    Wide,
    /// The multi-threadgroup passes at a chunk size.
    Grid(usize),
}

/// One selection by `selector` on fresh outputs: the selected IDs
/// (`selected_len` words), the candidates buffer afterwards and the status.
#[allow(clippy::too_many_arguments)]
fn run_selector(
    ctx: &MetalContext,
    selector: Selector,
    abc: ([u32; 4], [u32; 4], [u32; 4]),
    domain: usize,
    scores: &MetalTensor,
    candidates: &[u32],
    selected_len: usize,
    scratch: &RadixGridScratch,
) -> (Vec<u32>, Vec<u32>, u32) {
    let (a, b, c) = abc;
    let m_t = u32_tensor(ctx, candidates);
    let out = tensor(ctx, selected_len, GgmlType::I32).unwrap();
    let status = tensor(ctx, 1, GgmlType::I32).unwrap();
    execute(ctx, |enc| match selector {
        Selector::Rows => {
            pipeline(ctx, enc, "select_radix_rows").unwrap();
            enc.set_bytes(0, &a);
            enc.set_bytes(1, &b);
            enc.set_bytes(2, &c);
            enc.set_tensor(3, scores);
            enc.set_tensor(4, &m_t);
            enc.set_tensor(5, &out);
            enc.set_tensor(6, &status);
            enc.dispatch(
                MTLSize {
                    width: 1,
                    height: b[2] as usize,
                    depth: 1,
                },
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Selector::Wide => encode_select_radix_wide(
            ctx, enc, a, b, c, domain, scores, &m_t, &out, &status, scratch,
        )
        .unwrap(),
        Selector::Grid(chunk) => encode_select_radix_grid(
            ctx, enc, a, b, c, domain, chunk, scores, &m_t, &out, &status, scratch,
        )
        .unwrap(),
    });
    (
        u32_vec(&out, selected_len),
        u32_vec(&m_t, candidates.len()),
        u32_vec(&status, 1)[0],
    )
}

/// Decode's choice and the grid at other chunk sizes, including chunks that
/// end inside a thread's key slot (1000, 1500).
const WIDE_SELECTORS: [Selector; 5] = [
    Selector::Wide,
    Selector::Grid(512),
    Selector::Grid(1000),
    Selector::Grid(1500),
    Selector::Grid(2048),
];

/// Scratch for one or more rows at the smallest tested chunk.
fn grid_scratch(ctx: &MetalContext, rows: usize, domain: usize) -> RadixGridScratch {
    RadixGridScratch::new(ctx, rows, domain.div_ceil(512)).unwrap()
}

/// The wide selectors (decode's choice, the multi-threadgroup passes at
/// several chunk sizes) against the
/// CPU order and `select_radix_rows`, over every pattern, masked and
/// unmasked, at short to 128K-key rows, through the decode wrapper too. One
/// grid scratch serves every case, so each selection also checks that the
/// previous one left it cleared (status bit 8).
#[test]
fn radix_wide_selection_matches_reference() {
    let ctx = MetalContext::new().unwrap();
    let config = Config::default();
    let (k, block) = (config.index_topk, config.candidate_block);
    let scratch = grid_scratch(&ctx, 1, 131_072);
    let mut seed = 100u64;
    for length in [513usize, 4097, 20_000, 40_000, 65_536, 131_072] {
        for pattern in ["random", "ties", "special", "flat", "ramp", "sparse"] {
            for masked in [false, true] {
                seed += 1;
                let scores = scores_for(pattern, length, seed);
                let blocks = length.div_ceil(block);
                let mask = candidate_mask(blocks, seed);
                // Rows within the kept block count are never masked.
                let applied = masked && blocks > config.candidate_topk;
                let expected =
                    reference_select(&scores, length, k, applied.then_some((&mask[..], block)));
                let visible_nan =
                    (0..length).any(|t| scores[t].is_nan() && !(applied && mask[t / block] == 0));
                let s_t = upload(&ctx, &scores, vec![length as u64]).unwrap();
                let case = format!("length {length} pattern {pattern} masked {masked}");
                let abc = (
                    [(length - 1) as u32, 1, k as u32, block as u32],
                    [0, 0, 1, masked as u32],
                    [k as u32, 0, config.candidate_topk as u32, 0],
                );
                let (old, _, old_status) =
                    run_selector(&ctx, Selector::Rows, abc, length, &s_t, &mask, k, &scratch);
                assert_eq!(old, expected, "select_radix_rows, {case}");
                assert_eq!(old_status, u32::from(visible_nan), "{case}");
                for selector in WIDE_SELECTORS {
                    let (got, _, status) =
                        run_selector(&ctx, selector, abc, length, &s_t, &mask, k, &scratch);
                    assert_eq!(got, expected, "{selector:?}, {case}");
                    assert_eq!(
                        status,
                        u32::from(visible_nan),
                        "{selector:?} status, {case}"
                    );
                }
                // The decode wrapper.
                let m_t = u32_tensor(&ctx, &mask);
                let out = tensor(&ctx, k, GgmlType::I32).unwrap();
                let status = tensor(&ctx, 1, GgmlType::I32).unwrap();
                execute(&ctx, |enc| {
                    encode_select_radix_row_wide(
                        &ctx,
                        enc,
                        &config,
                        length - 1,
                        1,
                        &s_t,
                        &m_t,
                        &out,
                        &status,
                        &scratch,
                        masked,
                    )
                    .unwrap();
                });
                assert_eq!(u32_vec(&out, k), expected, "decode wrapper, {case}");
                assert_eq!(u32_vec(&status, 1)[0], u32::from(visible_nan), "{case}");
            }
        }
    }
}

/// Several rows of one chunk in one dispatch (mode 0, masked, and the mode 1
/// candidate publication of the source layer): every wide selector that fits
/// equals `select_radix_rows` (IDs, untouched rows, masks, status) and the
/// CPU reference.
#[test]
fn radix_wide_rows_match_single_threadgroup_rows() {
    let ctx = MetalContext::new().unwrap();
    let config = Config::default();
    let (k, block, keep) = (
        config.index_topk,
        config.candidate_block,
        config.candidate_topk,
    );
    // Rows 1..6 of a chunk at 39,990 (pruned and masked), with a score
    // stride wider than the longest row.
    let (start, rows, first) = (39_990usize, 5usize, 1usize);
    let width = start + first + rows + 3;
    let stride_blocks = width.div_ceil(block);
    let scratch = grid_scratch(&ctx, rows, width);
    for (pattern, seed) in [("random", 41u64), ("ties", 42), ("special", 43)] {
        let scores: Vec<f32> = (0..rows)
            .flat_map(|r| scores_for(pattern, width, seed * 10 + r as u64))
            .collect();
        let masks: Vec<u32> = (0..first + rows)
            .flat_map(|r| candidate_mask(stride_blocks, seed + r as u64))
            .collect();
        let s_t = upload(&ctx, &scores, vec![scores.len() as u64]).unwrap();
        let abc = (
            [start as u32, 1, k as u32, block as u32],
            [width as u32, stride_blocks as u32, rows as u32, 1],
            [k as u32, first as u32, keep as u32, 0],
        );
        let len = (first + rows) * k;
        let (old, _, old_status) = run_selector(
            &ctx,
            Selector::Rows,
            abc,
            width,
            &s_t,
            &masks,
            len,
            &scratch,
        );
        for local in 0..rows {
            let row = first + local;
            let length = start + row + 1;
            let mask = &masks[row * stride_blocks..(row + 1) * stride_blocks];
            let expected =
                reference_select(&scores[local * width..], length, k, Some((mask, block)));
            assert_eq!(&old[row * k..(row + 1) * k], expected, "row {row}");
        }
        for selector in WIDE_SELECTORS {
            let (got, _, status) =
                run_selector(&ctx, selector, abc, width, &s_t, &masks, len, &scratch);
            assert_eq!(
                (got, status),
                (old.clone(), old_status),
                "{selector:?} {pattern}"
            );
        }
    }
    // Mode 1: per-row block scores in, every block's mask entry out.
    let blocks_of = |row: usize| (start + row + 1).div_ceil(block);
    let block_scores: Vec<f32> = (0..rows)
        .flat_map(|local| {
            let mut v = scores_for("ties", stride_blocks, 77 + local as u64);
            v[blocks_of(first + local) - 1] = f32::INFINITY;
            v
        })
        .collect();
    let s_t = upload(&ctx, &block_scores, vec![block_scores.len() as u64]).unwrap();
    let abc = (
        [start as u32, 1, k as u32, block as u32],
        [stride_blocks as u32, stride_blocks as u32, rows as u32, 0],
        [k as u32, first as u32, keep as u32, 1],
    );
    let poison = vec![7u32; (first + rows) * stride_blocks];
    let (_, old, old_status) = run_selector(
        &ctx,
        Selector::Rows,
        abc,
        stride_blocks,
        &s_t,
        &poison,
        1,
        &scratch,
    );
    assert_eq!(old_status, 0);
    for local in 0..rows {
        let row = first + local;
        let blocks = blocks_of(row);
        let kept = reference_select(&block_scores[local * stride_blocks..], blocks, keep, None);
        let mut expected = vec![7u32; stride_blocks];
        expected[..blocks].fill(0);
        for id in kept {
            expected[id as usize] = 1;
        }
        assert_eq!(
            &old[row * stride_blocks..(row + 1) * stride_blocks],
            expected,
            "candidate mask row {row}"
        );
    }
    for selector in WIDE_SELECTORS {
        let (_, got, status) = run_selector(
            &ctx,
            selector,
            abc,
            stride_blocks,
            &s_t,
            &poison,
            1,
            &scratch,
        );
        assert_eq!(
            (got, status),
            (old.clone(), 0),
            "{selector:?} candidate masks"
        );
    }
}

/// GPU microseconds per encoding of `f`: `iters` encodings in one command
/// buffer, median of five such buffers after one warm-up.
fn gpu_us_per(ctx: &MetalContext, iters: usize, f: &dyn Fn(&KernelEncoder)) -> f64 {
    let mut samples = Vec::new();
    for round in 0..6 {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        for _ in 0..iters {
            f(&enc);
        }
        enc.end();
        metal::commit_and_wait(&command).unwrap();
        if round > 0 {
            samples.push((command.GPUEndTime() - command.GPUStartTime()) * 1e6 / iters as f64);
        }
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// Opt-in switch for `select_kernels_timing`: any value runs it (it prints
/// ILP vs MMA index and single- vs multi-threadgroup radix timings and
/// asserts only agreement); unset, the test returns at once.
const SELECT_TIMING_ENV: &str = "QWEN_V41_SELECT_TIMING";

/// Opt-in timing of the long-row index and selection kernels for one decode
/// row (32 x 128 F32 index heads, top-512): ILP vs MMA index scores, and
/// `select_radix_rows` vs the wide selectors; 200 dispatches per command
/// buffer, median of five buffers. Prints; asserts only that the kernels
/// agree. `QWEN_V41_SELECT_TIMING=1`.
#[test]
fn select_kernels_timing() {
    if std::env::var_os(SELECT_TIMING_ENV).is_none() {
        return;
    }
    let ctx = MetalContext::new().unwrap();
    let config = Config::default();
    let (heads, dim, k) = (32usize, 128usize, config.index_topk);
    let iters = 200;
    let make = |n: usize, seed: usize| {
        (0..n)
            .map(|i| ((i * 2_654_435_761usize + seed) % 1_000_003) as f32 / 500_000.0 - 1.0)
            .collect::<Vec<_>>()
    };
    let longest = 131_072usize;
    let q_t = upload(&ctx, &make(heads * dim, 3), vec![(heads * dim) as u64]).unwrap();
    let w_t = upload(&ctx, &make(heads, 5), vec![heads as u64]).unwrap();
    let k_t = upload(&ctx, &make(longest * dim, 7), vec![(longest * dim) as u64]).unwrap();
    let scores = tensor(&ctx, longest, GgmlType::F32).unwrap();
    let index_us = |encode: IndexRowsEncoder, length: usize| {
        gpu_us_per(&ctx, iters, &|enc| {
            encode(
                &ctx,
                enc,
                &config,
                length - 1,
                1,
                length,
                0,
                1,
                length,
                &q_t,
                &w_t,
                &k_t,
                &scores,
            )
            .unwrap();
        })
    };
    for length in [32_768usize, 131_072] {
        let ilp = index_us(encode_index_scores_rows_ilp, length);
        let mma = index_us(encode_index_scores_rows_mma, length);
        println!(
            "index keys={length} ilp_us={ilp:.1} mma_us={mma:.1} speedup={:.2}x",
            ilp / mma
        );
    }
    // Batched rows (prefill tiles): 8 rows of 32K keys.
    let rows = 8;
    let q_rows = upload(
        &ctx,
        &make(rows * heads * dim, 11),
        vec![(rows * heads * dim) as u64],
    )
    .unwrap();
    let w_rows = upload(&ctx, &make(rows * heads, 13), vec![(rows * heads) as u64]).unwrap();
    let batch = tensor(&ctx, rows * 32_768, GgmlType::F32).unwrap();
    let rows_us = |encode: IndexRowsEncoder| {
        gpu_us_per(&ctx, 50, &|enc| {
            encode(
                &ctx, enc, &config, 32_760, 1, 32_768, 0, rows, 32_768, &q_rows, &w_rows, &k_t,
                &batch,
            )
            .unwrap();
        })
    };
    let ilp = rows_us(encode_index_scores_rows_ilp);
    let mma = rows_us(encode_index_scores_rows_mma);
    println!(
        "index rows={rows} keys=32761..32768 ilp_us={ilp:.1} mma_us={mma:.1} speedup={:.2}x",
        ilp / mma
    );
    // Selection over MMA index scores, a masked random row and half-zero rows.
    let scratch = grid_scratch(&ctx, 1, longest);
    let old_out = tensor(&ctx, k, GgmlType::I32).unwrap();
    let new_out = tensor(&ctx, k, GgmlType::I32).unwrap();
    let status = tensor(&ctx, 1, GgmlType::I32).unwrap();
    for length in [16_384usize, 32_768, 65_536, 131_072] {
        let _ = index_us(encode_index_scores_rows_mma, length);
        let blocks = length.div_ceil(config.candidate_block);
        let mask = u32_tensor(&ctx, &candidate_mask(blocks, 9));
        let random = upload(&ctx, &scores_for("random", length, 9), vec![length as u64]).unwrap();
        let sparse = upload(&ctx, &scores_for("sparse", length, 9), vec![length as u64]).unwrap();
        let a = [
            (length - 1) as u32,
            1,
            k as u32,
            config.candidate_block as u32,
        ];
        let c = [k as u32, 0, config.candidate_topk as u32, 0];
        for (name, input, masked) in [
            ("index", &scores, false),
            ("random_masked", &random, true),
            ("sparse", &sparse, false),
            ("sparse_masked", &sparse, true),
        ] {
            let b = [0, 0, 1, masked as u32];
            let old = gpu_us_per(&ctx, iters, &|enc| {
                encode_select_radix_row(
                    &ctx,
                    enc,
                    &config,
                    length - 1,
                    1,
                    input,
                    &mask,
                    &old_out,
                    &status,
                    masked,
                )
                .unwrap();
            });
            let selectors = [
                Selector::Wide,
                Selector::Grid(512),
                Selector::Grid(1024),
                Selector::Grid(2048),
            ];
            for selector in selectors {
                let us = gpu_us_per(&ctx, iters, &|enc| match selector {
                    Selector::Wide => encode_select_radix_wide(
                        &ctx, enc, a, b, c, length, input, &mask, &new_out, &status, &scratch,
                    )
                    .unwrap(),
                    Selector::Grid(chunk) => encode_select_radix_grid(
                        &ctx, enc, a, b, c, length, chunk, input, &mask, &new_out, &status,
                        &scratch,
                    )
                    .unwrap(),
                    Selector::Rows => unreachable!(),
                });
                println!(
                    "select keys={length} scores={name} {selector:?} radix_us={old:.1} new_us={us:.1} speedup={:.2}x",
                    old / us
                );
                assert_eq!(u32_vec(&old_out, k), u32_vec(&new_out, k), "{name}");
                assert_eq!(u32_vec(&status, 1)[0], 0);
            }
        }
    }
    // Dependent-dispatch floor: an empty one-thread dispatch.
    let empty = tensor(&ctx, 1, GgmlType::F32).unwrap();
    let floor = gpu_us_per(&ctx, iters, &|enc| {
        pipeline(&ctx, enc, "copy").unwrap();
        enc.set_bytes(0, &0u32);
        enc.set_tensor(1, &empty);
        enc.set_tensor(2, &empty);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
    });
    println!("empty dispatch us={floor:.2}");
}
