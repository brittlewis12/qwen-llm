use super::fixture::*;
use super::*;

fn attended(groups: usize) -> Vec<f32> {
    (0..groups * WIDTH)
        .map(|i| (i / WIDTH + 1) as f32 / 32.0 + (i % 19 + 1) as f32 / 128.0)
        .collect()
}

#[test]
fn stored_group_matrices_cover_offsets_strides_tails_and_distinct_ids() {
    for groups in GROUPS {
        for dtype in DTYPES {
            let stored = StoredBank::new(groups, dtype);
            assert_eq!(
                stored.bytes.len(),
                GUARD * 2 + groups * RANK * stored_row_bytes(dtype)
            );
            assert_eq!(&stored.bytes[..GUARD], &[0xa7; GUARD]);
            assert_eq!(&stored.bytes[stored.bytes.len() - GUARD..], &[0x5c; GUARD]);
            assert_eq!(RANK % 4, 3);
            for group in 0..groups {
                assert_eq!((GUARD + group * stored.group_bytes()) % 16, 0);
                let row = &stored.decoded[group * RANK * WIDTH..(group + 1) * RANK * WIDTH];
                assert!(row.iter().all(|v| v.is_finite()));
                assert!(row.iter().any(|v| *v != 0.0));
                for prior in 0..group {
                    assert_ne!(
                        row,
                        &stored.decoded[prior * RANK * WIDTH..(prior + 1) * RANK * WIDTH]
                    );
                }
            }
            assert_eq!(
                output_a_native_layout(&case_config(groups), dtype, GUARD as u64).is_some(),
                groups <= 16
            );
        }
    }
}

#[test]
fn independent_stored_precision_decoder_has_known_q8_and_bfloat_values() {
    let q8 = StoredBank::new(8, GgmlType::Q8_0);
    assert_eq!(&q8.bytes[GUARD..GUARD + 5], &[0x01, 0x18, 0xfa, 0x05, 0xff]);
    let scale = half::f16::from_bits(0x1801).to_f32();
    assert_eq!(&q8.decoded[..3], &[-6.0 * scale, 5.0 * scale, -scale]);
    let group = 7;
    let row = 6;
    let block = 7;
    let bits = 0x1801 + (group * 32 + row * 3 + block) as u16;
    let offset = GUARD + (group * RANK + row) * stored_row_bytes(GgmlType::Q8_0) + block * 34;
    assert_eq!(&q8.bytes[offset..offset + 2], &bits.to_le_bytes());
    let bf16 = StoredBank::new(1, GgmlType::BF16);
    let target = dense_coefficient(0, 0, 0);
    let bits = half::bf16::from_f32(target).to_bits();
    assert_eq!(&bf16.bytes[GUARD..GUARD + 2], &bits.to_le_bytes());
    assert_eq!(bf16.decoded[0], half::bf16::from_bits(bits).to_f32());
    assert_ne!(bf16.decoded[0], target);
    assert_ne!(bf16.decoded[0], half::f16::from_bits(bits).to_f32());
    let f32 = StoredBank::new(1, GgmlType::F32);
    assert_eq!(f32.decoded[0].to_bits(), target.to_bits());
}

#[test]
fn independent_oracle_rejects_noop_zero_bank_permutations_and_nonfinite_values() {
    for groups in GROUPS {
        for dtype in DTYPES {
            let stored = StoredBank::new(groups, dtype);
            let x = attended(groups);
            let expected = stored.oracle(&x, &(0..groups).collect::<Vec<_>>());
            let correct: Vec<_> = expected.iter().map(|d| d.value as f32).collect();
            check_dots(&correct, &expected, RANK).unwrap();
            for group in 0..groups {
                assert!(
                    expected[group * RANK..(group + 1) * RANK]
                        .iter()
                        .all(|d| d.value.abs() > 1e-4)
                );
            }
            for value in [0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                assert!(check_dots(&vec![value; correct.len()], &expected, RANK).is_err());
            }
            if groups > 1 {
                let bank_zero = stored.oracle(&x, &vec![0; groups]);
                let rotated = stored.oracle(
                    &x,
                    &(0..groups).map(|g| (g + 1) % groups).collect::<Vec<_>>(),
                );
                for bad in [bank_zero, rotated] {
                    let actual: Vec<_> = bad.iter().map(|d| d.value as f32).collect();
                    assert!(check_dots(&actual, &expected, RANK).is_err());
                }
                let mut permuted_low = correct.clone();
                permuted_low.rotate_left(RANK);
                assert!(check_dots(&permuted_low, &expected, RANK).is_err());
                let mut permuted_input = x.clone();
                permuted_input.rotate_left(WIDTH);
                let bad: Vec<_> = stored
                    .oracle(&permuted_input, &(0..groups).collect::<Vec<_>>())
                    .iter()
                    .map(|d| d.value as f32)
                    .collect();
                assert!(check_dots(&bad, &expected, RANK).is_err());
            }
        }
    }
}

fn launch(name: &str, grid: [u64; 3]) -> metal::DispatchCensusRow {
    metal::DispatchCensusRow {
        family: "fixture",
        tag: None,
        encoder_ordinal: 0,
        encoder_concurrent: false,
        kernel: name.into(),
        grid_width: grid[0],
        grid_height: grid[1],
        grid_depth: grid[2],
        threads_width: 128,
        threads_height: 1,
        threads_depth: 1,
        grid_tgs: grid.iter().product(),
        tg_threads: 128,
    }
}

#[test]
fn exact_dispatch_contract_rejects_missing_extra_or_malformed_fallback_launches() {
    for groups in GROUPS {
        for dtype in DTYPES {
            for r2 in [false, true] {
                let mut census = vec![launch(
                    "kernel_deepseek_v41_attention_output",
                    [groups as u64 * 8, 1, 1],
                )];
                let a = a_launches(groups, dtype, r2);
                assert_eq!(a.len(), if groups <= 16 { 1 } else { groups });
                let expected_grid = if groups <= 16 {
                    [2, groups as u64, 1]
                } else if dtype == GgmlType::F32 && r2 {
                    [4, 1, 1]
                } else {
                    [2, 1, 1]
                };
                assert!(a.iter().all(|(_, grid)| *grid == expected_grid));
                census.extend(a.iter().map(|(name, grid)| launch(name, *grid)));
                census.push(launch(
                    if r2 {
                        "kernel_mat_vec_f32_f32_lcpp_r2"
                    } else {
                        "kernel_mat_vec_f32_f32"
                    },
                    [1, 1, 1],
                ));
                census.push(launch("kernel_deepseek_v41_attention_finite", [1, 1, 1]));
                check_census(&census, groups, dtype).unwrap();
                let mut missing = census.clone();
                missing.remove(1);
                assert!(check_census(&missing, groups, dtype).is_err());
                let mut extra = census.clone();
                extra.insert(1, extra[1].clone());
                assert!(check_census(&extra, groups, dtype).is_err());
                for field in 0..6 {
                    let mut bad = census.clone();
                    match field {
                        0 => bad[1].grid_width += 1,
                        1 => bad[1].grid_height += 1,
                        2 => bad[1].grid_depth += 1,
                        3 => bad[1].threads_width = 64,
                        4 => bad[1].encoder_concurrent = true,
                        5 => bad[1].kernel = "no_op".into(),
                        _ => unreachable!(),
                    }
                    assert!(check_census(&bad, groups, dtype).is_err());
                }
            }
        }
    }
}

#[test]
fn bank_and_low_guard_controls_detect_mutation() {
    for dtype in DTYPES {
        let stored = StoredBank::new(17, dtype);
        assert!(stored.unchanged(&stored.bytes));
        for index in [
            0,
            GUARD - 1,
            GUARD,
            stored.bytes.len() - GUARD,
            stored.bytes.len() - 1,
        ] {
            let mut changed = stored.bytes.clone();
            changed[index] ^= 1;
            assert!(!stored.unchanged(&changed));
        }
    }
    let count = 17 * RANK;
    let low = vec![12345.25; count + 2 * GUARD / 4];
    assert!(low_guards_intact(&low, count));
    for index in [0, GUARD / 4 - 1, GUARD / 4 + count, low.len() - 1] {
        let mut changed = low.clone();
        changed[index] = f32::NAN;
        assert!(!low_guards_intact(&changed, count));
    }
}

#[test]
fn fixture_prices_bound_every_case_before_gpu_allocation() {
    for groups in GROUPS {
        for dtype in DTYPES {
            let price = price(groups, dtype);
            assert_eq!(price.logical, price.requests.iter().sum::<u64>());
            assert!(price.requests.iter().all(|&n| n > 0));
            assert_eq!(
                price.metal_upper,
                price.logical + price.requests.len() as u64 * 16384
            );
            assert_eq!(
                price.cpu_upper,
                4 * price.logical + (1 << 20) + LIFETIME_CPU_BYTES
            );
            assert!(
                LIFETIME_CPU_BYTES
                    >= 2 * 16384
                        + std::mem::size_of::<BufferReleaseProbes>() as u64
                        + 2 * std::mem::size_of::<usize>() as u64
            );
            let total = price.metal_upper + price.cpu_upper + RESERVE;
            assert!(total <= BUDGET);
            eprintln!(
                "fixture_price groups={groups} dtype={dtype:?} buffers={} logical={} metal_upper={} cpu_upper={} reserve={RESERVE} total={total} cap={BUDGET}",
                price.requests.len(),
                price.logical,
                price.metal_upper,
                price.cpu_upper
            );
        }
    }
}

#[test]
fn nested_autorelease_pools_release_position_objects_before_case_objects() {
    use objc2::rc::{Retained, Weak, autoreleasepool};
    use objc2::runtime::NSObject;

    let case_weak = autoreleasepool(|_| {
        let object = NSObject::new();
        let case_weak = Weak::from_retained(&object);
        let _ = Retained::autorelease_ptr(object);
        for _ in 0..2 {
            let position_weak = autoreleasepool(|_| {
                let object = NSObject::new();
                let weak = Weak::from_retained(&object);
                let _ = Retained::autorelease_ptr(object);
                assert!(
                    weak.load().is_some(),
                    "explicit autorelease must survive until drain"
                );
                weak
            });
            assert!(
                position_weak.load().is_none(),
                "position pool must drain independently"
            );
            assert!(
                case_weak.load().is_some(),
                "position drain must not release case objects"
            );
        }
        case_weak
    });
    assert!(
        case_weak.load().is_none(),
        "case pool must release its autoreleased objects"
    );
}

#[test]
fn weak_release_control_detects_strong_owner_escaping_pool_drain() {
    use objc2::rc::{Retained, Weak, autoreleasepool};
    use objc2::runtime::NSObject;

    let (weak, escaped) = autoreleasepool(|_| {
        let object = NSObject::new();
        let weak = Weak::from_retained(&object);
        let escaped = object.clone();
        let _ = Retained::autorelease_ptr(object);
        (weak, escaped)
    });
    assert!(
        weak.load().is_some(),
        "weak proof must detect an escaped strong owner"
    );
    drop(escaped);
    assert!(
        weak.load().is_none(),
        "weak observation must not keep the object alive"
    );
}
