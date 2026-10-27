use super::*;

#[test]
fn cursor_range_matches_repeated_rows_partial_exact_and_final() {
    for rows in [1, 3, 128, 4096] {
        for first in [0, rows / 2, rows - 1] {
            let initial = BatchCursor {
                start: 17,
                rows,
                layer: 3,
                row: first,
            };
            for count in [1, rows - first] {
                let mut scalar = initial;
                for _ in 0..count {
                    scalar = scalar.advanced(3, scalar.start + scalar.row, 1).unwrap();
                }
                let range = initial.advanced(3, 17 + first, count).unwrap();
                assert_eq!(range, scalar);
                if count == rows - first {
                    assert_eq!((range.layer, range.row), (4, 0));
                } else {
                    assert_eq!((range.layer, range.row), (3, first + count));
                }
            }
        }
    }
}

#[test]
fn cursor_range_refusals_are_atomic() {
    let initial = BatchCursor {
        start: 17,
        rows: 3,
        layer: 2,
        row: 1,
    };
    for (layer, position, count) in [
        (2, 18, 0),
        (2, 18, 3),
        (2, 18, usize::MAX),
        (3, 18, 1),
        (2, 17, 1),
    ] {
        assert!(initial.advanced(layer, position, count).is_err());
        assert_eq!(
            initial,
            BatchCursor {
                start: 17,
                rows: 3,
                layer: 2,
                row: 1
            }
        );
    }
    for b in [
        BatchCursor {
            start: usize::MAX,
            rows: 2,
            ..initial
        },
        BatchCursor {
            layer: usize::MAX,
            ..initial
        },
        BatchCursor { row: 3, ..initial },
        BatchCursor {
            rows: 0,
            row: 0,
            ..initial
        },
    ] {
        let before = b;
        assert!(
            b.advanced(b.layer, b.start.saturating_add(b.row), 2)
                .is_err()
        );
        assert_eq!(b, before);
    }
}

#[test]
fn prepared_range_eligibility_preserves_publication_and_selection_fallbacks() {
    for ordinary in [false, true] {
        for owns in [false, true] {
            for publish in [false, true] {
                for index in [false, true] {
                    for select in [false, true] {
                        let commands =
                            usize::from(ordinary && owns && publish) + usize::from(index && select);
                        assert_eq!(
                            command_free_prepared(ordinary, owns, publish, index, select),
                            commands == 0
                        );
                    }
                }
            }
        }
    }
    assert!(command_free_prepared(true, false, true, false, true)); // follower/ownerless
    assert!(command_free_prepared(false, true, true, true, false)); // CED batched selection
    assert!(command_free_prepared(true, true, false, true, false)); // published body
    assert!(!command_free_prepared(true, true, true, true, false)); // unsupported publisher/head/tail
    assert!(!command_free_prepared(false, false, true, true, true)); // pruned/per-row selection
}

#[test]
fn ratio_head_body_tail_cursor_parity() {
    for ratio in [1, 2, 3, 8] {
        for start in 0..17 {
            for rows in 1..33 {
                let body_start = start + (ratio - start % ratio) % ratio;
                let body_end = (start + rows) / ratio * ratio;
                let (head, body) = if body_end > body_start {
                    (body_start - start, body_end - body_start)
                } else {
                    (0, 0)
                };
                let initial = BatchCursor {
                    start,
                    rows,
                    layer: 0,
                    row: 0,
                };
                let mut scalar = initial;
                let mut range = initial;
                for row in 0..rows {
                    scalar = scalar.advanced(0, start + row, 1).unwrap();
                }
                for row in 0..head {
                    range = range.advanced(0, start + row, 1).unwrap();
                }
                if body > 0 {
                    range = range.advanced(0, start + head, body).unwrap();
                }
                for row in head + body..rows {
                    range = range.advanced(0, start + row, 1).unwrap();
                }
                assert_eq!(range, scalar, "ratio={ratio} start={start} rows={rows}");
            }
        }
    }
}

#[test]
fn old_host_override_is_nested_panic_safe_and_thread_local() {
    assert!(range_host_enabled());
    with_old_host_loop(|| {
        assert!(!range_host_enabled());
        let panic = std::panic::catch_unwind(|| with_old_host_loop(|| panic!("restore")));
        assert!(panic.is_err());
        assert!(!range_host_enabled());
        assert!(std::thread::spawn(range_host_enabled).join().unwrap());
    });
    assert!(range_host_enabled());
}
