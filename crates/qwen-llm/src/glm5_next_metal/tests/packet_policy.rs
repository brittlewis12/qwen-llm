//! Production packet selectors and synthetic dispatch witnesses; no checkpoint.
use super::*;

const HALF: Arith = Arith {
    lineage: PackedLineage::Fast,
    precision: FastPrecision::Half,
};

#[test]
fn packet_expert_policy_checks_precision_rows_and_all_three_banks() {
    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        for precision in FastPrecision::ALL {
            let arith = Arith { lineage, precision };
            for rows in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 32, 512, usize::MAX] {
                for gate in [
                    GgmlType::IQ2_S,
                    GgmlType::IQ3_S,
                    GgmlType::IQ2_XS,
                    GgmlType::F32,
                ] {
                    for up in [
                        GgmlType::IQ2_S,
                        GgmlType::IQ3_S,
                        GgmlType::IQ3_XXS,
                        GgmlType::Q8_0,
                    ] {
                        for down in [
                            GgmlType::IQ3_S,
                            GgmlType::IQ4_XS,
                            GgmlType::Q4_K,
                            GgmlType::F32,
                        ] {
                            let mode = expert_mode(arith, gate, up, down);
                            let expected = arith == HALF
                                && (1..=8).contains(&rows)
                                && matches!(gate, GgmlType::IQ2_S | GgmlType::IQ3_S)
                                && matches!(up, GgmlType::IQ2_S | GgmlType::IQ3_S)
                                && matches!(down, GgmlType::IQ3_S | GgmlType::IQ4_XS);
                            assert_eq!(
                                packet_half_experts_selected(arith, mode, gate, up, down, rows),
                                expected,
                                "{arith:?}/{rows}/{gate:?}/{up:?}/{down:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn packet_projection_policy_bounds_router_and_q8_without_tile_alignment_gates() {
    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        for precision in FastPrecision::ALL {
            let arith = Arith { lineage, precision };
            for stage in Stage::ALL {
                for rows in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 32, 512, usize::MAX] {
                    for dtype in [GgmlType::F32, GgmlType::F16, GgmlType::Q8_0, GgmlType::Q6_K] {
                        for n_in in [0, 31, 32, 64, 96, 256] {
                            for n_out in [1, 7, 16, 288] {
                                let mode = projection_mode(arith, stage, dtype, n_in, n_out);
                                assert_eq!(
                                    packet_router_rows_selected(arith, mode, stage, dtype, rows),
                                    arith == HALF
                                        && stage == Stage::Router
                                        && dtype == GgmlType::F32
                                        && (1..=8).contains(&rows)
                                );
                                assert_eq!(
                                    packet_q8_fewrow_selected(arith, mode, dtype, n_in, rows),
                                    arith == HALF
                                        && dtype == GgmlType::Q8_0
                                        && (2..=4).contains(&rows)
                                        && n_in > 0
                                        && n_in.is_multiple_of(32)
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn packet_diagnostics_restore_production_and_do_not_override_stage_precision() {
    let selections = |rows| {
        [
            packet_half_experts_selected(
                HALF,
                expert_mode(HALF, GgmlType::IQ2_S, GgmlType::IQ3_S, GgmlType::IQ4_XS),
                GgmlType::IQ2_S,
                GgmlType::IQ3_S,
                GgmlType::IQ4_XS,
                rows,
            ),
            packet_router_rows_selected(
                HALF,
                stage_mode(HALF, Stage::Router),
                Stage::Router,
                GgmlType::F32,
                rows,
            ),
            packet_q8_fewrow_selected(
                HALF,
                stage_mode(HALF, Stage::DenseFfn),
                GgmlType::Q8_0,
                96,
                rows,
            ),
        ]
    };
    assert_eq!(selections(4), [true; 3]);
    assert_eq!(selections(8), [true, true, false]);
    {
        let _legacy = PacketPolicy::legacy();
        assert_eq!(selections(4), [false; 3]);
        {
            let _production = PacketPolicy::production();
            assert_eq!(selections(4), [true; 3]);
            assert_eq!(selections(8), [true, true, false]);
        }
        assert_eq!(selections(4), [false; 3]);
        {
            let _half = PacketHalfExperts::set();
            let _q8 = PacketQ8Fewrow::set();
            assert_eq!(selections(8), [true, false, true]);
            assert_eq!(selections(3), [true, false, true]);
            let _exact = ExactStages::set(&[Stage::RoutedExperts, Stage::DenseFfn]);
            assert_eq!(selections(4), [false; 3]);
        }
        assert_eq!(selections(4), [false; 3]);
    }
    assert_eq!(selections(4), [true; 3]);
    {
        let _f32 = F32Stages::with_tiles(&Stage::ALL, &F32Tile::ALL);
        assert_eq!(selections(4), [false; 3]);
    }
    let unwind = std::panic::catch_unwind(|| {
        let _legacy = PacketPolicy::legacy();
        let _half = PacketHalfExperts::set();
        panic!("test scoped restoration");
    });
    assert!(unwind.is_err());
    assert_eq!(selections(4), [true; 3]);
    let _legacy = PacketPolicy::legacy();
    std::thread::spawn(|| {
        assert!(packet_q8_fewrow_selected(
            HALF,
            StageMode::Fast,
            GgmlType::Q8_0,
            32,
            2
        ));
    })
    .join()
    .unwrap();
}

fn tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32).unwrap()
}

fn ints(ctx: &MetalContext, values: &[i32], shape: Vec<u64>) -> MetalTensor {
    MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::I32).unwrap()
}

fn empty(ctx: &MetalContext, shape: Vec<u64>) -> MetalTensor {
    tensor(
        ctx,
        &vec![f32::NAN; shape.iter().product::<u64>() as usize],
        shape,
    )
}

fn run(ctx: &MetalContext, encode: impl FnOnce(&KernelEncoder)) -> Vec<String> {
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    crate::metal::dispatch_census_begin();
    encode(&enc);
    let kernels = crate::metal::dispatch_census_take()
        .into_iter()
        .map(|r| r.kernel.to_string())
        .collect();
    enc.end();
    command.commit();
    wait_completed(&command).unwrap();
    kernels
}

// Independent GGML decoding of finite synthetic compressed payloads.
fn bank(
    ctx: &MetalContext,
    dtype: GgmlType,
    shape: Vec<u64>,
    seed: usize,
) -> (MetalTensor, Vec<f32>) {
    let n = shape.iter().product::<u64>() as usize;
    let (block, size) = dtype.storage_layout().unwrap();
    let mut bytes = Vec::new();
    for ordinal in 0..n / block as usize {
        let mut b: Vec<u8> = (0..size as usize)
            .map(|i| (ordinal * 73 + i * 29 + seed * 11) as u8)
            .collect();
        let scale = (1 + (ordinal + seed) % 7) as f32 * 0.00013;
        b[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        bytes.extend(b);
    }
    let desc = TensorDesc {
        name: "packet_policy".into(),
        shape: shape.clone(),
        dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: bytes.len() as u64,
    };
    let decoded = crate::codec::dequant_to_f32(&desc, &bytes).unwrap();
    (
        MetalTensor::from_bytes(ctx, &bytes, shape, dtype).unwrap(),
        decoded,
    )
}

fn close(label: &str, actual: &[f32], expected: &[f32], relative: f32) {
    assert_eq!(actual.len(), expected.len());
    let scale = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    assert!(scale > 0.0, "degenerate {label} fixture");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= relative * scale,
            "{label}[{i}]: {a} vs {b}, scale={scale}"
        );
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    assert!(values.iter().all(|v| v.is_finite()));
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "Metal dispatch witness; production lease, API validation, serial execution"]
fn packet_half_experts_default_dispatch_census_and_cpu_oracle() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (h, f, e, k) = (256usize, 256usize, 3usize, 2usize);
    let clamp = 0.2;
    let half = |x: f32| half::f16::from_f32(x).to_f32();
    for gate_type in [GgmlType::IQ2_S, GgmlType::IQ3_S] {
        for up_type in [GgmlType::IQ2_S, GgmlType::IQ3_S] {
            for down_type in [GgmlType::IQ3_S, GgmlType::IQ4_XS] {
                let (gate, gc) = bank(&ctx, gate_type, vec![h as u64, f as u64, e as u64], 1);
                let (up, uc) = bank(&ctx, up_type, vec![h as u64, f as u64, e as u64], 3);
                let (down, dc) = bank(&ctx, down_type, vec![f as u64, h as u64, e as u64], 7);
                for rows in 1..=8 {
                    let values: Vec<f32> = (0..h * rows)
                        .map(|i| (i % 71) as f32 * 0.01713 - 0.5)
                        .collect();
                    let route_ids: Vec<i32> = (0..rows * k)
                        .map(|i| ((i + 2 * (i / k)) % e) as i32)
                        .collect();
                    let route_weights: Vec<f32> = (0..rows * k)
                        .map(|i| if i % k == 0 { 0.7 } else { 0.3 })
                        .collect();
                    let input = tensor(&ctx, &values, vec![h as u64, rows as u64]);
                    let ids = ints(&ctx, &route_ids, vec![k as u64, rows as u64]);
                    let weights = tensor(&ctx, &route_weights, vec![k as u64, rows as u64]);
                    let status = ints(&ctx, &vec![ROUTE_STATUS_READY; rows], vec![rows as u64]);
                    let counts = ints(&ctx, &vec![-7; e], vec![e as u64]);
                    let slots = ints(&ctx, &vec![-7; e * rows], vec![(e * rows) as u64]);
                    let inner = empty(&ctx, vec![f as u64, (k * rows) as u64]);
                    let slot_out = empty(&ctx, vec![h as u64, (k * rows) as u64]);
                    let output = empty(&ctx, vec![h as u64, rows as u64]);
                    let row_inner = empty(&ctx, vec![f as u64, k as u64]);
                    let row_out = empty(&ctx, vec![h as u64, k as u64]);
                    let b = GroupedExperts {
                        gate_bank: &gate,
                        up_bank: &up,
                        down_bank: &down,
                        input: &input,
                        ids: &ids,
                        weights: &weights,
                        counts: &counts,
                        slots: &slots,
                        inner: &inner,
                        slot_out: &slot_out,
                        output: &output,
                    };
                    let census = F32Census::begin();
                    let kernels = run(&ctx, |enc| {
                        assert_eq!(
                            encode_experts(
                                &ctx, enc, HALF, &b, &status, &row_inner, &row_out, h, f, e, k,
                                rows, clamp
                            )
                            .unwrap(),
                            StageMode::Fast
                        );
                    });
                    assert_eq!(
                        kernels
                            .iter()
                            .filter(|n| n.starts_with("kernel_expert_native_"))
                            .count(),
                        3,
                        "{kernels:?}"
                    );
                    assert!(
                        !kernels
                            .iter()
                            .any(|n| n.contains("grouped") || n.contains("bucket")),
                        "{kernels:?}"
                    );
                    let calls = census.take();
                    assert_eq!(calls.len(), 1);
                    assert_eq!(
                        calls.get(&(Stage::RoutedExperts, "experts".into(), CensusPath::Half)),
                        Some(&1)
                    );
                    assert_eq!(read_i32(&counts).unwrap(), vec![-7; e]);
                    assert_eq!(read_i32(&slots).unwrap(), vec![-7; e * rows]);

                    // F64 dots over half-rounded operands, F32 clamped SwiGLU,
                    // then half-rounded down operands and F32 weighted output.
                    // No claim of bit identity with the grouped MMA reduction.
                    let mut expected_inner = vec![0.0; f * k * rows];
                    let mut expected_slots = vec![0.0; h * k * rows];
                    let mut expected = vec![0.0; h * rows];
                    for row in 0..rows {
                        for slot in 0..k {
                            let index = row * k + slot;
                            let expert = route_ids[index] as usize;
                            let dot = |w: &[f32], o: usize| {
                                (0..h)
                                    .map(|i| {
                                        f64::from(half(w[(expert * f + o) * h + i]))
                                            * f64::from(half(values[row * h + i]))
                                    })
                                    .sum::<f64>() as f32
                            };
                            let g: Vec<f32> = (0..f).map(|o| dot(&gc, o)).collect();
                            let u: Vec<f32> = (0..f).map(|o| dot(&uc, o)).collect();
                            let act =
                                crate::deepseek_v4_oracle::clamped_swiglu(&g, &u, clamp).unwrap();
                            expected_inner[index * f..][..f].copy_from_slice(&act);
                            for o in 0..h {
                                let value = (0..f)
                                    .map(|i| {
                                        f64::from(half(dc[(expert * h + o) * f + i]))
                                            * f64::from(half(act[i]))
                                    })
                                    .sum::<f64>()
                                    as f32;
                                expected_slots[index * h + o] = value;
                                expected[row * h + o] += route_weights[index] * value;
                            }
                        }
                    }
                    close(
                        "native half inner",
                        &read_f32(&inner).unwrap(),
                        &expected_inner,
                        2e-5,
                    );
                    // F32 gate reduction can cross a down-input half rounding
                    // boundary. The leaf's cancellation-aware tests bound this;
                    // this integration witness checks composition and indexing.
                    close(
                        "native half slots",
                        &read_f32(&slot_out).unwrap(),
                        &expected_slots,
                        2e-3,
                    );
                    close(
                        "native half weighted",
                        &read_f32(&output).unwrap(),
                        &expected,
                        2e-3,
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "Metal dispatch witness; production lease, API validation, serial execution"]
fn packet_experts_preserve_legacy_dispatch_outside_half_packet_capability() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (h, f, e, k) = (256usize, 256usize, 3usize, 2usize);
    for (arith, rows, gate_type, down_type) in [
        (HALF, 9, GgmlType::IQ2_S, GgmlType::IQ3_S),
        (HALF, 32, GgmlType::IQ2_S, GgmlType::IQ3_S),
        (HALF, 4, GgmlType::IQ2_XS, GgmlType::IQ3_S),
        (HALF, 4, GgmlType::IQ2_S, GgmlType::IQ3_XXS),
        (
            Arith {
                lineage: PackedLineage::Exact,
                precision: FastPrecision::Half,
            },
            4,
            GgmlType::IQ2_S,
            GgmlType::IQ3_S,
        ),
        (
            Arith {
                lineage: PackedLineage::Fast,
                precision: FastPrecision::F32,
            },
            4,
            GgmlType::IQ2_S,
            GgmlType::IQ3_S,
        ),
        (
            Arith {
                lineage: PackedLineage::Fast,
                precision: FastPrecision::DenseF32,
            },
            4,
            GgmlType::IQ2_S,
            GgmlType::IQ3_S,
        ),
    ] {
        let (gate, _) = bank(&ctx, gate_type, vec![h as u64, f as u64, e as u64], 1);
        let (up, _) = bank(&ctx, gate_type, vec![h as u64, f as u64, e as u64], 3);
        let (down, _) = bank(&ctx, down_type, vec![f as u64, h as u64, e as u64], 7);
        let values: Vec<f32> = (0..h * rows)
            .map(|i| (i % 71) as f32 * 0.01713 - 0.5)
            .collect();
        let route_ids: Vec<i32> = (0..rows * k)
            .map(|i| ((i + 2 * (i / k)) % e) as i32)
            .collect();
        let input = tensor(&ctx, &values, vec![h as u64, rows as u64]);
        let ids = ints(&ctx, &route_ids, vec![k as u64, rows as u64]);
        let weights = tensor(&ctx, &vec![0.5; rows * k], vec![k as u64, rows as u64]);
        let status = ints(&ctx, &vec![ROUTE_STATUS_READY; rows], vec![rows as u64]);
        let counts = MetalTensor::zeros_i32(&ctx, vec![e as u64]).unwrap();
        let slots = MetalTensor::zeros_i32(&ctx, vec![(e * rows) as u64]).unwrap();
        let inner = empty(&ctx, vec![f as u64, (k * rows) as u64]);
        let slot_out = empty(&ctx, vec![h as u64, (k * rows) as u64]);
        let output = empty(&ctx, vec![h as u64, rows as u64]);
        let row_inner = empty(&ctx, vec![f as u64, k as u64]);
        let row_out = empty(&ctx, vec![h as u64, k as u64]);
        let b = GroupedExperts {
            gate_bank: &gate,
            up_bank: &up,
            down_bank: &down,
            input: &input,
            ids: &ids,
            weights: &weights,
            counts: &counts,
            slots: &slots,
            inner: &inner,
            slot_out: &slot_out,
            output: &output,
        };
        let encode = |enc: &KernelEncoder| {
            encode_experts(
                &ctx, enc, arith, &b, &status, &row_inner, &row_out, h, f, e, k, rows, 0.2,
            )
            .unwrap();
        };
        let kernels = run(&ctx, encode);
        let actual = read_f32(&output).unwrap();
        let _legacy = PacketPolicy::legacy();
        let reference_kernels = run(&ctx, encode);
        assert_eq!(
            kernels, reference_kernels,
            "{arith:?}/{rows}/{gate_type:?}/{down_type:?}"
        );
        assert_eq!(bits(&actual), bits(&read_f32(&output).unwrap()));
    }
}

#[test]
#[ignore = "Metal dispatch witness; production lease, API validation, serial execution"]
fn packet_q8_default_dispatch_matches_f32_oracle_and_preserves_other_modes() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    for (n_in, n_out) in [(32usize, 7usize), (96, 17), (256, 32)] {
        let (weight, decoded) = bank(&ctx, GgmlType::Q8_0, vec![n_in as u64, n_out as u64], 4);
        for rows in [1, 2, 3, 4, 5, 8, 9] {
            let values: Vec<f32> = (0..n_in * rows)
                .map(|i| (i % 71) as f32 * 0.01317 - 0.4)
                .collect();
            // Backing remains valid for the original padded FastF32 tile too.
            let input = MetalTensor::zeros_f32(&ctx, vec![n_in as u64, 32]).unwrap();
            let x = rows_view(&input, rows);
            write_f32(&x, &values).unwrap();
            let output = empty(&ctx, vec![n_out as u64, rows as u64]);
            for arith in [
                HALF,
                Arith {
                    lineage: PackedLineage::Exact,
                    precision: FastPrecision::Half,
                },
                Arith {
                    lineage: PackedLineage::Fast,
                    precision: FastPrecision::F32,
                },
                Arith {
                    lineage: PackedLineage::Fast,
                    precision: FastPrecision::DenseF32,
                },
            ] {
                let census = F32Census::begin();
                let encode = |enc: &KernelEncoder| {
                    matmat(
                        &ctx,
                        enc,
                        arith,
                        Stage::DenseFfn,
                        &weight,
                        &x,
                        &output,
                        n_in,
                        n_out,
                        rows,
                    )
                    .unwrap()
                };
                let kernels = run(&ctx, encode);
                let actual = read_f32(&output).unwrap();
                let selected = arith == HALF && (2..=4).contains(&rows);
                assert_eq!(
                    kernels.iter().any(|n| n.contains("q8_0_fewrow")),
                    selected,
                    "{arith:?}/{rows}: {kernels:?}"
                );
                if selected {
                    assert_eq!(kernels.len(), 1, "{kernels:?}");
                    assert_eq!(
                        census
                            .take()
                            .get(&(Stage::DenseFfn, "Q8_0".into(), CensusPath::F32Tile)),
                        Some(&1)
                    );
                    let mut expected = Vec::new();
                    for row in 0..rows {
                        for out in 0..n_out {
                            expected.push(
                                (0..n_in)
                                    .map(|i| {
                                        f64::from(decoded[out * n_in + i])
                                            * f64::from(values[row * n_in + i])
                                    })
                                    .sum::<f64>() as f32,
                            );
                        }
                    }
                    close("Q8 fewrow F32 operands", &actual, &expected, 2e-5);
                } else {
                    let _legacy = PacketPolicy::legacy();
                    let legacy_kernels = run(&ctx, encode);
                    assert_eq!(kernels, legacy_kernels);
                    assert_eq!(bits(&actual), bits(&read_f32(&output).unwrap()));
                }
            }
        }
    }
}

#[test]
#[ignore = "Metal dispatch witness; production lease, API validation, serial execution"]
fn packet_f32_router_default_dispatch_matches_rows_and_preserves_other_modes() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (n_in, n_out) = (64usize, 17usize);
    let w: Vec<f32> = (0..n_in * n_out)
        .map(|i| (i % 37) as f32 * 0.01713 - 0.2)
        .collect();
    let weight = tensor(&ctx, &w, vec![n_in as u64, n_out as u64]);
    for rows in 1..=9 {
        let x: Vec<f32> = (0..n_in * rows)
            .map(|i| (i % 71) as f32 * 0.01317 - 0.4)
            .collect();
        let input = tensor(&ctx, &x, vec![n_in as u64, rows as u64]);
        let output = empty(&ctx, vec![n_out as u64, rows as u64]);
        for arith in [
            HALF,
            Arith {
                lineage: PackedLineage::Exact,
                precision: FastPrecision::Half,
            },
            Arith {
                lineage: PackedLineage::Fast,
                precision: FastPrecision::F32,
            },
            Arith {
                lineage: PackedLineage::Fast,
                precision: FastPrecision::DenseF32,
            },
        ] {
            for stage in [Stage::Router, Stage::IndexerProjection] {
                let census = F32Census::begin();
                let encode = |enc: &KernelEncoder| {
                    matmat(
                        &ctx, enc, arith, stage, &weight, &input, &output, n_in, n_out, rows,
                    )
                    .unwrap()
                };
                let kernels = run(&ctx, encode);
                let actual = read_f32(&output).unwrap();
                let selected = arith == HALF && stage == Stage::Router && rows <= 8;
                if selected {
                    assert_eq!(
                        census
                            .take()
                            .get(&(stage, "F32".into(), CensusPath::RowBaseline)),
                        Some(&1)
                    );
                    let reference_kernels = run(&ctx, |enc| {
                        for row in 0..rows {
                            crate::metal_forward::encode_mat_vec_dispatch(
                                &ctx,
                                enc,
                                &weight,
                                &input.view_subrange((row * n_in) as u64, vec![n_in as u64]),
                                &output.view_subrange((row * n_out) as u64, vec![n_out as u64]),
                                n_in,
                                n_out,
                            )
                            .unwrap();
                        }
                    });
                    assert_eq!(kernels.len(), rows);
                    assert_eq!(kernels, reference_kernels);
                    assert_eq!(bits(&actual), bits(&read_f32(&output).unwrap()));
                    let mut expected = Vec::new();
                    for row in 0..rows {
                        for out in 0..n_out {
                            expected.push(
                                (0..n_in)
                                    .map(|i| {
                                        f64::from(w[out * n_in + i]) * f64::from(x[row * n_in + i])
                                    })
                                    .sum::<f64>() as f32,
                            );
                        }
                    }
                    close("F32 row router", &actual, &expected, 2e-5);
                } else {
                    let _legacy = PacketPolicy::legacy();
                    let legacy_kernels = run(&ctx, encode);
                    assert_eq!(kernels, legacy_kernels);
                    assert_eq!(bits(&actual), bits(&read_f32(&output).unwrap()));
                }
            }
        }
    }
}
