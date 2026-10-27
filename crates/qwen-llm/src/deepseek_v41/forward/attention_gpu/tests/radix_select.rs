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
    let ctx = MetalContext::new().unwrap();
    let (heads, dim) = (32usize, 128usize);
    let config = Config::default();
    assert!(index_ilp_supported(&config));
    let topk = config.index_topk;
    // Rows at increasing visible lengths (one partial 8-key group, one partial
    // 64-key threadgroup) plus a row at or below top-k that must not write.
    for (start, ratio, rows, first) in [
        (20_000usize, 1usize, 3usize, 0usize),
        (2_100, 4, 5, 2),
        (509, 1, 4, 0),
    ] {
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
            encode_index_scores_rows_ilp(
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
