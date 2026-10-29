use super::*;
use std::ops::Range;

const PROMPTS: [usize; 13] = [
    1, 17, 127, 128, 129, 513, 2047, 2048, 2049, 2065, 2575, 4097, 8192,
];
const WIDTHS: [usize; 7] = [32, 128, 129, 300, 512, 1024, 2048];

#[test]
fn encoder8k_scope_keeps_original_tiles_and_compacted_singleton_lineage() {
    assert!(super::original_query_tile(8192, 8192, 0, 512).is_err());
    prefill::with_full_chunk_ceiling_for_test(8192, || {
        for prompt in [8191, 8192, 8193] {
            for tile in [129, 300, 512, 4096] {
                let mut oracle = Vec::new();
                for first in (0..prompt).step_by(8192) {
                    let end = (first + 8192).min(prompt);
                    for start in (first..end).step_by(tile) {
                        let range = start..(start + tile).min(end);
                        oracle.extend(vec![range.clone(); range.len()]);
                    }
                }
                for (position, expected) in oracle.iter().enumerate() {
                    assert_eq!(
                        super::original_query_tile(prompt, 8192, position, tile).unwrap(),
                        *expected
                    );
                    for physical in [1, (prompt - position).min(2048)] {
                        let (rows, arity) =
                            super::query_tile_span(Some((prompt, 8192)), position, physical, tile)
                                .unwrap();
                        assert_eq!(rows, physical.min(expected.end - position));
                        assert_eq!(arity, expected.len());
                        assert_eq!(tensor_arithmetic(true, arity), expected.len() >= 128);
                        assert!(!tensor_arithmetic(false, arity));
                    }
                }
            }
        }
        let c = Config::default();
        let mut tiles = Vec::new();
        for position in (0..8192).step_by(512) {
            let tile = super::original_query_tile(8192, 8192, position, 512).unwrap();
            assert_eq!(
                tile,
                super::original_query_tile(8192, 4096, position, 512).unwrap()
            );
            assert_eq!(tile.len(), 512);
            tiles.push(tile);
        }
        assert_eq!(tiles.len(), 16);
        assert_eq!(
            checked_product(&[8192, c.heads, c.head_dim]).unwrap(),
            268435456
        );
        assert!(checked_product(&[8192, c.heads, c.head_dim, 4]).is_err());
        assert_eq!(
            selection_shapes(&c, 8448, 8192, 512).unwrap(),
            [512 * 8448, 512 * 4 * 512 * 2]
        );
        assert!(Scratch::shapes(&c, 8192).is_ok());
        assert!(Scratch::shapes(&c, 8193).is_err());
        assert_eq!(
            super::query_tile_span(None, 0, 8192, 512).unwrap(),
            (512, 512)
        );
        assert!(super::query_tile_span(None, usize::MAX, 8192, 512).is_err());
        assert!(super::original_query_tile(8192, 8192, 8192, 512).is_err());
        assert!(super::original_query_tile(8192, 8192, 0, 8192).is_err());
        assert!(mma_geometry(&c));
        with_f32_attention(|| {
            assert!(!mma_geometry(&c));
            assert!(!tensor_arithmetic(mma_geometry(&c), 512));
            assert!(Scratch::shapes(&c, 8192).unwrap().len() > 3);
        });
        for c in [
            Config {
                head_dim: 256,
                ..c.clone()
            },
            Config { heads: 17, ..c },
        ] {
            assert!(!mma_geometry(&c));
            assert!(!tensor_arithmetic(mma_geometry(&c), 512));
        }
        assert!(prefill::RowsArithmetic::Packed.packed_operands(1));
        assert_eq!(
            super::query_tile_span(Some((8193, 8192)), 8192, 1, 512).unwrap(),
            (1, 1)
        );
        // A nondivisor query tile exposes the different chunk boundary policy;
        // preserving lineage is not a promise that 4K and 8K compute equal bits.
        assert_eq!(
            super::original_query_tile(8192, 4096, 4096, 300).unwrap(),
            4096..4396
        );
        assert_eq!(
            super::original_query_tile(8192, 8192, 4096, 300).unwrap(),
            3900..4200
        );
    })
    .unwrap();
    assert!(Scratch::shapes(&Config::default(), 8192).is_err());
    assert!(super::original_query_tile(8192, 8192, 0, 512).is_err());
}

#[test]
fn selected_full_chunks_preserve_every_compacted_query_tile_and_singleton_route() {
    for chunk in [2048, 4096] {
        for n in [2047, 2048, 2049, 4095, 4096, 4097] {
            for tile in [128, 129, 300, 512, 2048, 4096] {
                let mut oracle = Vec::new();
                for first in (0..n).step_by(chunk) {
                    let end = (first + chunk).min(n);
                    for start in (first..end).step_by(tile) {
                        let range = start..(start + tile).min(end);
                        oracle.extend(vec![range.clone(); range.len()]);
                    }
                }
                for (position, expected) in oracle.iter().enumerate() {
                    assert_eq!(
                        super::original_query_tile(n, chunk, position, tile).unwrap(),
                        *expected
                    );
                    for physical in [1, (n - position).min(2048)] {
                        let (rows, arity) =
                            super::query_tile_span(Some((n, chunk)), position, physical, tile)
                                .unwrap();
                        assert_eq!(rows, physical.min(expected.end - position));
                        assert_eq!(arity, expected.len());
                        assert_eq!(tensor_arithmetic(true, arity), expected.len() >= 128);
                    }
                }
            }
        }
    }
    assert_eq!(
        super::query_tile_span(Some((2049, 4096)), 2048, 1, 4096).unwrap(),
        (1, 2049)
    );
    assert_eq!(
        super::query_tile_span(Some((2049, 2048)), 2048, 1, 4096).unwrap(),
        (1, 1)
    );
}

// Retain the original 2K witnesses independently of the new selected-width cases.
fn original_query_tile(prompt: usize, position: usize, width: usize) -> Result<Range<usize>> {
    super::original_query_tile(prompt, prefill::DEFAULT_CHUNK_ROWS, position, width)
}

fn query_tile_span(
    prompt: Option<usize>,
    position: usize,
    remaining: usize,
    width: usize,
) -> Result<(usize, usize)> {
    super::query_tile_span(
        prompt.map(|n| (n, prefill::DEFAULT_CHUNK_ROWS)),
        position,
        remaining,
        width,
    )
}

// Independent oracle: enumerate Full's chunk-relative tiles from cold zero,
// rather than deriving a tile from an absolute position as production does.
fn full_tiles(prompt_rows: usize, width: usize) -> Vec<Range<usize>> {
    let mut tiles = Vec::new();
    for chunk in (0..prompt_rows).step_by(prefill::DEFAULT_CHUNK_ROWS) {
        let end = (chunk + prefill::DEFAULT_CHUNK_ROWS).min(prompt_rows);
        for start in (chunk..end).step_by(width) {
            tiles.push(start..(start + width).min(end));
        }
    }
    tiles
}

fn per_row_tiles(prompt_rows: usize, width: usize) -> Vec<Range<usize>> {
    full_tiles(prompt_rows, width)
        .into_iter()
        .flat_map(|tile| vec![tile.clone(); tile.len()])
        .collect()
}

#[test]
fn original_tiles_match_cold_full_at_every_row() {
    for prompt_rows in PROMPTS {
        for width in WIDTHS.into_iter().chain([1]) {
            let full = per_row_tiles(prompt_rows, width);
            for (position, expected) in full.iter().enumerate() {
                let actual = original_query_tile(prompt_rows, position, width).unwrap();
                assert_eq!(
                    &actual, expected,
                    "N={prompt_rows}, p={position}, w={width}"
                );
                let (n, arity) = query_tile_span(Some(prompt_rows), position, 1, width).unwrap();
                assert_eq!(n, 1);
                assert_eq!(arity, expected.len());
                assert_eq!(
                    tensor_arithmetic(true, arity),
                    expected.len() >= 128,
                    "singleton must inherit Full's per-row policy"
                );
            }
        }
    }
}

fn check_subrange(prompt_rows: usize, width: usize, range: Range<usize>, full: &[Range<usize>]) {
    let mut position = range.start;
    while position < range.end {
        let (n, arity) =
            query_tile_span(Some(prompt_rows), position, range.end - position, width).unwrap();
        let expected = &full[position];
        assert!(n > 0 && n <= width && n <= range.end - position);
        assert_eq!(n, (range.end - position).min(expected.end - position));
        assert_eq!(arity, expected.len());
        // Containment in one oracle tile proves the policy for every row in
        // the span, including when both sides of a physical chunk are partial.
        assert!(expected.start <= position && position + n <= expected.end);
        assert_eq!(&full[position + n - 1], expected);
        assert_eq!(tensor_arithmetic(true, arity), expected.len() >= 128);
        position += n;
    }
    assert_eq!(position, range.end);
}

#[test]
fn compacted_subranges_keep_full_per_row_arithmetic_and_exact_coverage() {
    for prompt_rows in PROMPTS {
        for width in WIDTHS {
            let full = per_row_tiles(prompt_rows, width);
            for start in 0..prompt_rows {
                let remaining = (prompt_rows - start).min(prefill::DEFAULT_CHUNK_ROWS);
                // Exhaust all subranges for small prompts. For larger ones,
                // cover every absolute start, both sides of the next original
                // boundary, and short/long arbitrary physical chunk extents.
                let lengths: Vec<_> = if prompt_rows <= 129 {
                    (1..=remaining).collect()
                } else {
                    let boundary = full[start].end - start;
                    [
                        1,
                        2,
                        17,
                        127,
                        128,
                        129,
                        width - 1,
                        width,
                        width + 1,
                        boundary.saturating_sub(1),
                        boundary,
                        boundary + 1,
                        2047,
                        2048,
                        remaining,
                    ]
                    .into_iter()
                    .filter(|&n| n > 0 && n <= remaining)
                    .collect()
                };
                for n in lengths {
                    check_subrange(prompt_rows, width, start..start + n, &full);
                }
            }
        }
    }
}

#[test]
fn ordinary_rows_remain_physical_and_chunk_relative() {
    for prompt_rows in PROMPTS {
        for width in WIDTHS {
            for chunk in (0..prompt_rows).step_by(prefill::DEFAULT_CHUNK_ROWS) {
                let end = (chunk + prefill::DEFAULT_CHUNK_ROWS).min(prompt_rows);
                let mut position = chunk;
                while position < end {
                    let expected = (end - position).min(width);
                    assert_eq!(
                        query_tile_span(None, position, end - position, width).unwrap(),
                        (expected, expected)
                    );
                    position += expected;
                }
            }
        }
    }
    assert_eq!(query_tile_span(None, 511, 513, 512).unwrap(), (512, 512));
    assert_eq!(query_tile_span(None, 8191, 1, 512).unwrap(), (1, 1));
}

#[test]
fn packed_singletons_and_short_original_tails_are_not_blanket_tensor() {
    assert!(prefill::RowsArithmetic::Packed.packed_operands(1));
    for (prompt_rows, position, arity) in [
        (17, 16, 17),
        (127, 126, 127),
        (128, 127, 128),
        (129, 128, 129),
        (513, 512, 1),
        (2047, 2046, 511),
        (2048, 2047, 512),
        (2065, 2064, 17),
        (2575, 2574, 15),
        (8192, 8191, 512),
    ] {
        assert_eq!(
            query_tile_span(Some(prompt_rows), position, 1, 512).unwrap(),
            (1, arity)
        );
        assert_eq!(tensor_arithmetic(true, arity), arity >= 128);
    }
    assert_eq!(
        query_tile_span(Some(2065), 2048, 17, 512).unwrap(),
        (17, 17)
    );
    assert_eq!(
        query_tile_span(Some(2575), 2558, 17, 512).unwrap(),
        (2, 512)
    );
    assert_eq!(
        query_tile_span(Some(2575), 2560, 15, 512).unwrap(),
        (15, 15)
    );
}

#[test]
fn nondivisor_width_restarts_at_original_chunks_not_physical_starts() {
    assert_eq!(original_query_tile(8192, 2047, 300).unwrap(), 1800..2048);
    assert_eq!(original_query_tile(8192, 2048, 300).unwrap(), 2048..2348);
    assert_eq!(
        query_tile_span(Some(8192), 2047, 2048, 300).unwrap(),
        (1, 248)
    );
    assert_eq!(
        query_tile_span(Some(8192), 2048, 2047, 300).unwrap(),
        (300, 300)
    );
    assert_eq!(original_query_tile(8192, 2047, 129).unwrap(), 1935..2048);
    assert!(!tensor_arithmetic(true, 113));
    assert_eq!(
        query_tile_span(Some(2065), 2048, 17, 300).unwrap(),
        (17, 17)
    );
}

#[test]
fn genuine_whole_chunk_scalar_tails_remain_outside_rows() {
    for prompt_rows in PROMPTS {
        let scalar = prefill::original_scalar_tail(prompt_rows, prefill::DEFAULT_CHUNK_ROWS);
        assert_eq!(
            scalar,
            (prompt_rows % prefill::DEFAULT_CHUNK_ROWS == 1).then_some(prompt_rows - 1)
        );
        for width in WIDTHS {
            let full = per_row_tiles(prompt_rows, width);
            let rows_end = scalar.unwrap_or(prompt_rows);
            // Mirror the decoder's Rows/scalar division without changing the
            // original scalar Native path or treating a compacted n=1 as it.
            for start in (0..rows_end).step_by(prefill::DEFAULT_CHUNK_ROWS) {
                let end = (start + prefill::DEFAULT_CHUNK_ROWS).min(rows_end);
                check_subrange(prompt_rows, width, start..end, &full);
                assert!(scalar.is_none_or(|tail| end <= tail));
            }
            if let Some(tail) = scalar {
                assert_eq!(full[tail], tail..prompt_rows);
                assert_eq!(
                    original_query_tile(prompt_rows, tail, width).unwrap().len(),
                    1
                );
            }
        }
    }
    assert_eq!(
        prefill::original_scalar_tail(513, prefill::DEFAULT_CHUNK_ROWS),
        None
    );
    assert_eq!(
        prefill::original_scalar_tail(2065, prefill::DEFAULT_CHUNK_ROWS),
        None
    );
    assert_eq!(
        prefill::original_scalar_tail(2575, prefill::DEFAULT_CHUNK_ROWS),
        None
    );
}

#[test]
fn lineage_does_not_bypass_f32_or_unsupported_mma_geometry() {
    let c = Config::default();
    assert!(tensor_arithmetic(mma_geometry(&c), 512));
    with_f32_attention(|| {
        assert!(!mma_geometry(&c));
        for arity in [1, 17, 127, 128, 512, 2048] {
            assert!(!tensor_arithmetic(mma_geometry(&c), arity));
        }
    });
    for c in [
        Config {
            head_dim: 256,
            ..Config::default()
        },
        Config {
            heads: 17,
            ..Config::default()
        },
    ] {
        assert!(!tensor_arithmetic(mma_geometry(&c), 512));
    }
}

#[test]
fn checked_bounds_reject_invalid_views_without_overflowing_valid_extremes() {
    for (prompt_rows, position, width) in [
        (0, 0, 512),
        (17, 17, 512),
        (17, 18, 512),
        (usize::MAX, usize::MAX, 512),
        (8192, 0, 0),
        (8192, 0, 4097),
        (8192, 0, usize::MAX),
    ] {
        assert!(original_query_tile(prompt_rows, position, width).is_err());
    }
    for prompt_rows in [None, Some(usize::MAX)] {
        for (position, remaining, width) in [
            (0, 0, 512),
            (0, 4097, 512),
            (0, usize::MAX, 512),
            (0, 1, 0),
            (0, 1, 4097),
            (usize::MAX, 1, 512),
            (usize::MAX - 1, 2, 512),
        ] {
            assert!(query_tile_span(prompt_rows, position, remaining, width).is_err());
        }
    }
    assert!(query_tile_span(Some(17), 16, 2, 512).is_err());
    assert!(query_tile_span(Some(17), 17, 1, 512).is_err());
    assert!(query_tile_span(Some(0), 0, 1, 512).is_err());
    for width in WIDTHS.into_iter().chain([1]) {
        let position = usize::MAX - 1;
        let tile = original_query_tile(usize::MAX, position, width).unwrap();
        assert!(tile.start <= position && position < tile.end);
        assert_eq!(tile.end, usize::MAX);
        assert_eq!(
            query_tile_span(Some(usize::MAX), position, 1, width).unwrap(),
            (1, tile.len())
        );
    }
}
