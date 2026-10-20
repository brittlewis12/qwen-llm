use super::*;

fn plan(taps: &[u32], features: usize, logits: usize) -> Glm5NextRowBankPlan {
    Glm5NextRowBankPlan::from_dimensions(4096, 154880, 45, taps, features, logits).unwrap()
}

#[test]
fn cpu_plan_preserves_caller_order_and_prices_only_optional_storage() {
    let p = plan(&[42, 5, 33, 14, 24], 8, 8);
    assert_eq!(p.taps(), &[42, 5, 33, 14, 24]);
    assert_eq!(p.feature_capacity(), 8);
    assert_eq!(p.logit_capacity(), 8);
    let specs = p.allocation_specs();
    assert_eq!(specs[0].shape, [4096, 5, 8]);
    assert_eq!(specs[1].shape, [154880, 8]);
    assert_eq!(
        p.priced_bytes(&Some).unwrap(),
        (8 * 5 * 4096 + 8 * 154880) * 4
    );
    assert!(p.priced_bytes(&|_| None).is_err());
    assert!(p.priced_bytes(&|_| Some(u64::MAX)).is_err());
    assert_eq!(plan(&[5], 2048, 0).allocation_specs().len(), 1);
    assert_eq!(plan(&[], 0, 8).allocation_specs().len(), 1);
}

#[test]
fn cpu_plan_refuses_invalid_sites_capacities_and_overflow() {
    let build =
        |taps: &[u32], f, l| Glm5NextRowBankPlan::from_dimensions(4096, 154880, 45, taps, f, l);
    for taps in [&[45][..], &[u32::MAX][..], &[5, 5][..]] {
        assert!(build(taps, 8, 8).is_err());
    }
    for (taps, f, l) in [
        (&[][..], 0, 0),
        (&[][..], 8, 1),
        (&[5][..], 0, 1),
        (&[5][..], usize::MAX, 0),
        (&[][..], 0, usize::MAX),
    ] {
        assert!(build(taps, f, l).is_err());
    }
    for (h, v, b) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
        assert!(Glm5NextRowBankPlan::from_dimensions(h, v, b, &[], 0, 1).is_err());
    }
}

#[test]
fn cpu_trailing_window_is_contiguous_across_chunk_boundaries() {
    let p = plan(&[42, 5], 5, 1);
    let ranges = p.ranges(100, 11, false).unwrap();
    assert_eq!(ranges.input, 100..111);
    assert_eq!(ranges.features, 106..111);
    assert_eq!(ranges.logits, 110..111);
    for chunk_size in [1, 2, 4, 8, 11] {
        let mut offsets = Vec::new();
        for start in (100..111).step_by(chunk_size) {
            for position in
                start.max(ranges.features.start)..(start + chunk_size).min(ranges.features.end)
            {
                for tap in 0..2 {
                    offsets.push(p.feature_offset(position - ranges.features.start, tap));
                }
            }
        }
        assert_eq!(offsets, (0..10).map(|i| i * 4096).collect::<Vec<_>>());
    }
    assert_eq!(p.ranges(900, 2, false).unwrap().features, 900..902);
    assert!(
        p.ranges(0, 2, true).is_err(),
        "all-row logits must not truncate"
    );
    assert!(
        plan(&[5], 1, 8).ranges(0, 2, true).is_err(),
        "verification features must not truncate"
    );
    assert!(p.ranges(usize::MAX, 1, false).is_err());
    assert!(p.ranges(0, 0, false).is_err());
    assert_eq!(plan(&[], 0, 8).ranges(3, 8, true).unwrap().logits, 3..11);
    assert_eq!(plan(&[5], 8, 0).ranges(3, 8, false).unwrap().logits, 11..11);
}

#[test]
#[ignore = "synthetic Metal capture layout gate; explicitly opt in with an idle GPU"]
fn metal_mean_capture_orders_taps_and_leaves_unused_bank_rows_untouched() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let p = Glm5NextRowBankPlan::from_dimensions(8, 16, 3, &[2, 0], 5, 0).unwrap();
    let features = MetalTensor::zeros_f32(&ctx, vec![8, 2, 5]).unwrap();
    write_f32(&features, &[f32::NAN; 80]).unwrap();
    let bank = Glm5NextRowBank {
        plan: p,
        weights_instance: 0,
        device: ctx.device.registryID(),
        features: Some(features),
        logits: None,
        priced_bytes: 0,
    };
    let quarter = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[0.25f32; 4]),
        vec![4],
        GgmlType::F32,
    )
    .unwrap();
    let output = RowOutput {
        bank: &bank,
        ranges: bank.plan.ranges(100, 3, false).unwrap(),
    };
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    let mut inputs = Vec::new();
    for block in 0..3 {
        let values: Vec<f32> = (0..3)
            .flat_map(|row| {
                (0..4).flat_map(move |stream| {
                    (0..8).map(move |d| (block * 1000 + row * 100 + stream * 10 + d) as f32)
                })
            })
            .collect();
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&values),
            vec![8, 4, 3],
            GgmlType::F32,
        )
        .unwrap();
        output
            .encode_capture(&ctx, &enc, &quarter, block, 100, 3, &input)
            .unwrap();
        inputs.push(input);
    }
    enc.end();
    command.commit();
    wait_completed(&command).unwrap();
    let captured = output.completed();
    let expected: Vec<f32> = (0..3)
        .flat_map(|row| {
            [2, 0].into_iter().flat_map(move |block| {
                (0..8).map(move |d| (block * 1000 + row * 100 + 15 + d) as f32)
            })
        })
        .collect();
    assert_eq!(read_f32(captured.features().unwrap()).unwrap(), expected);
    assert!(
        read_f32(bank.features.as_ref().unwrap()).unwrap()[48..]
            .iter()
            .all(|v| v.is_nan())
    );
    drop(inputs);
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "GLM capture cancellation: target weights, MTL_DEBUG_LAYER=1 and production lease"]
fn metal_capture_prefill_checkpoint_preserves_native_chunk_boundary_and_window() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required()).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, 64, 8).unwrap();
    let mut bank = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &[42, 5], 5, 1).unwrap(),
    )
    .unwrap();
    let tokens: Vec<u32> = (1..=19).collect();
    session.prefill_packed(&ctx, &tokens[..8]).unwrap();
    let prefix = session.capture_snapshot().unwrap();
    let future = session.forward(&ctx, 31).unwrap();
    let future_state = session.capture_snapshot().unwrap();
    session.reset().unwrap();
    let mut checkpoints = 0;
    let result =
        session.prefill_with_captures_and_checkpoint(&ctx, &tokens, &mut bank, &mut || {
            checkpoints += 1;
            if checkpoints == 2 {
                Err("cancel before second chunk".into())
            } else {
                Ok(())
            }
        });
    assert!(matches!(result, Err(Glm5NextMetalError::Cancelled(_))));
    assert_eq!(checkpoints, 2);
    assert_eq!(session.position(), 8);
    assert!(!session.is_poisoned());
    assert!(session.capture_snapshot().unwrap().same_state(&prefix));
    assert_eq!(bits(&session.forward(&ctx, 31).unwrap()), bits(&future));
    assert!(
        session
            .capture_snapshot()
            .unwrap()
            .same_state(&future_state)
    );

    session.reset().unwrap();
    let ordinary = session
        .prefill_with_captures(&ctx, &tokens, &mut bank)
        .unwrap();
    let features = read_f32(ordinary.features().unwrap()).unwrap();
    let logits = read_f32(ordinary.logits().unwrap()).unwrap();
    let end = session.capture_snapshot().unwrap();
    session.reset().unwrap();
    checkpoints = 0;
    let observed = session
        .prefill_with_captures_and_checkpoint(&ctx, &tokens, &mut bank, &mut || {
            checkpoints += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(checkpoints, 3, "native packed widths remain 8 + 8 + 3");
    assert_eq!(observed.feature_positions(), 14..19);
    assert_eq!(
        bits(&read_f32(observed.features().unwrap()).unwrap()),
        bits(&features)
    );
    assert_eq!(
        bits(&read_f32(observed.logits().unwrap()).unwrap()),
        bits(&logits)
    );
    assert!(session.capture_snapshot().unwrap().same_state(&end));
}

#[test]
#[ignore = "GLM same-width capture oracle: target weights, MTL_DEBUG_LAYER=1 and production lease"]
fn metal_capture_same_width_oracles_1_2_8() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required()).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, 64, 8).unwrap();
    assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
    assert_eq!(
        session.effective_fast_precision(),
        Some(FastPrecision::Half)
    );
    let taps = [42, 5, 33, 14, 24];
    let v = weights.config.vocab_size as usize;
    let f = taps.len() * weights.config.hidden_size as usize;
    let mut bank = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &taps, 11, 8).unwrap(),
    )
    .unwrap();
    let mut logits_only = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &[], 0, 8).unwrap(),
    )
    .unwrap();
    let mut tail = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &taps, 3, 1).unwrap(),
    )
    .unwrap();
    let ledger = *session.ledger();
    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        session.reset().unwrap();
        session.set_packed_lineage(lineage).unwrap();
        session.prefill_packed(&ctx, &[1, 2, 3]).unwrap();
        let base = session.capture_snapshot().unwrap();
        let allocated = ctx.current_allocated_size();
        for width in [1, 2, 8] {
            let tokens: Vec<u32> = (10..10 + width as u32).collect();
            // Scalar capture must not change scalar arithmetic or state.
            session.restore_snapshot(&base).unwrap();
            let mut scalar_logits = Vec::new();
            for &token in &tokens {
                scalar_logits.extend(session.forward(&ctx, token).unwrap());
            }
            let scalar_end = session.capture_snapshot().unwrap();
            let scalar_future = session.forward(&ctx, 31).unwrap();
            let scalar_future_state = session.capture_snapshot().unwrap();
            session.restore_snapshot(&base).unwrap();
            for (row, &token) in tokens.iter().enumerate() {
                let captured = session
                    .forward_with_captures(&ctx, token, &mut bank)
                    .unwrap();
                assert_eq!(
                    bits(&read_f32(captured.logits().unwrap()).unwrap()),
                    bits(&scalar_logits[row * v..(row + 1) * v])
                );
            }
            assert!(session.capture_snapshot().unwrap().same_state(&scalar_end));
            assert_eq!(
                bits(&session.forward(&ctx, 31).unwrap()),
                bits(&scalar_future)
            );
            assert!(
                session
                    .capture_snapshot()
                    .unwrap()
                    .same_state(&scalar_future_state)
            );

            // The exact packed oracle uses the SAME row count and lineage.
            session.restore_snapshot(&base).unwrap();
            let ordinary = session.prefill_packed(&ctx, &tokens).unwrap();
            let packed_end = session.capture_snapshot().unwrap();
            let packed_future = session.forward(&ctx, 31).unwrap();
            let packed_future_state = session.capture_snapshot().unwrap();
            session.restore_snapshot(&base).unwrap();
            let captured = session
                .verify_packed_with_captures(&ctx, &tokens, &mut bank)
                .unwrap();
            assert_eq!(captured.input_positions(), 3..3 + width);
            assert_eq!(captured.logit_positions(), 3..3 + width);
            assert_eq!(captured.taps(), taps);
            let all_logits = read_f32(captured.logits().unwrap()).unwrap();
            let all_features = read_f32(captured.features().unwrap()).unwrap();
            assert_eq!(bits(&all_logits[(width - 1) * v..]), bits(&ordinary));
            assert!(session.capture_snapshot().unwrap().same_state(&packed_end));
            assert_eq!(
                bits(&session.forward(&ctx, 31).unwrap()),
                bits(&packed_future)
            );
            assert!(
                session
                    .capture_snapshot()
                    .unwrap()
                    .same_state(&packed_future_state)
            );

            session.restore_snapshot(&base).unwrap();
            let no_features = session
                .verify_packed_with_captures(&ctx, &tokens, &mut logits_only)
                .unwrap();
            assert!(no_features.features().is_none());
            assert_eq!(
                bits(&read_f32(no_features.logits().unwrap()).unwrap()),
                bits(&all_logits)
            );
            assert!(session.capture_snapshot().unwrap().same_state(&packed_end));
            assert_eq!(
                bits(&session.forward(&ctx, 31).unwrap()),
                bits(&packed_future)
            );
            assert!(
                session
                    .capture_snapshot()
                    .unwrap()
                    .same_state(&packed_future_state)
            );

            session.restore_snapshot(&base).unwrap();
            let last_only = session
                .prefill_with_captures(&ctx, &tokens, &mut bank)
                .unwrap();
            assert_eq!(
                bits(&read_f32(last_only.logits().unwrap()).unwrap()),
                bits(&ordinary)
            );
            assert_eq!(
                bits(&read_f32(last_only.features().unwrap()).unwrap()),
                bits(&all_features)
            );
            assert!(session.capture_snapshot().unwrap().same_state(&packed_end));
            assert_eq!(
                bits(&session.forward(&ctx, 31).unwrap()),
                bits(&packed_future)
            );
            assert!(
                session
                    .capture_snapshot()
                    .unwrap()
                    .same_state(&packed_future_state)
            );
        }

        // Trailing capture crosses the same 8+3 packed schedule, not scalar decode.
        let tokens = [7; 11];
        session.restore_snapshot(&base).unwrap();
        let ordinary = session.prefill_packed(&ctx, &tokens).unwrap();
        let ordinary_end = session.capture_snapshot().unwrap();
        let future = session.forward(&ctx, 31).unwrap();
        let future_state = session.capture_snapshot().unwrap();
        session.restore_snapshot(&base).unwrap();
        let full = session
            .prefill_with_captures(&ctx, &tokens, &mut bank)
            .unwrap();
        let full_features = read_f32(full.features().unwrap()).unwrap();
        assert_eq!(
            bits(&read_f32(full.logits().unwrap()).unwrap()),
            bits(&ordinary)
        );
        assert!(
            session
                .capture_snapshot()
                .unwrap()
                .same_state(&ordinary_end)
        );
        session.restore_snapshot(&base).unwrap();
        let captured = session
            .prefill_with_captures(&ctx, &tokens, &mut tail)
            .unwrap();
        assert_eq!(captured.feature_positions(), 11..14);
        assert_eq!(captured.logit_positions(), 13..14);
        assert_eq!(
            bits(&read_f32(captured.features().unwrap()).unwrap()),
            bits(&full_features[8 * f..])
        );
        assert_eq!(
            bits(&read_f32(captured.logits().unwrap()).unwrap()),
            bits(&ordinary)
        );
        assert!(
            session
                .capture_snapshot()
                .unwrap()
                .same_state(&ordinary_end)
        );
        assert_eq!(bits(&session.forward(&ctx, 31).unwrap()), bits(&future));
        assert!(
            session
                .capture_snapshot()
                .unwrap()
                .same_state(&future_state)
        );

        session.restore_snapshot(&base).unwrap();
        assert!(
            session
                .verify_packed_with_captures(&ctx, &[1; 9], &mut bank)
                .is_err()
        );
        assert!(
            session
                .forward_with_captures(&ctx, weights.config.vocab_size, &mut bank)
                .is_err()
        );
        assert!(session.capture_snapshot().unwrap().same_state(&base));
        assert_eq!(
            ctx.current_allocated_size(),
            allocated,
            "capture/verify allocated Metal storage"
        );
    }
    assert_eq!(*session.ledger(), ledger);
    let before = session.capture_snapshot().unwrap();
    bank.weights_instance = u64::MAX;
    assert!(session.forward_with_captures(&ctx, 1, &mut bank).is_err());
    assert!(session.capture_snapshot().unwrap().same_state(&before));
}

// Numerical characterization is intentionally separate from capture invariance.
// These metrics are not an equivalence gate or an artifact qualification list.
fn report_numeric(label: &str, packed: &[f32], serial: &[f32], row_width: usize) {
    assert_eq!(packed.len(), serial.len());
    assert!(packed.iter().chain(serial).all(|x| x.is_finite()));
    let (max_abs, sum_squared) =
        packed
            .iter()
            .zip(serial)
            .fold((0.0f64, 0.0f64), |(max, sum), (&a, &b)| {
                let d = a as f64 - b as f64;
                (max.max(d.abs()), sum + d * d)
            });
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let top1_changes = packed
        .chunks_exact(row_width)
        .zip(serial.chunks_exact(row_width))
        .filter(|(a, b)| argmax(a) != argmax(b))
        .count();
    eprintln!(
        "{label}: max_abs={max_abs:.8e} rms={:.8e} row_top1_changes={top1_changes}",
        (sum_squared / packed.len() as f64).sqrt()
    );
}

#[test]
#[ignore = "GLM numerical characterization only: target weights, production lease, MTL_DEBUG_LAYER=1"]
fn metal_packed_vs_scalar_numeric_widths_1_2_8() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required()).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, 64, 8).unwrap();
    let taps = [42, 5, 33, 14, 24];
    let mut bank = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &taps, 8, 8).unwrap(),
    )
    .unwrap();
    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        session.reset().unwrap();
        session.set_packed_lineage(lineage).unwrap();
        session.prefill_packed(&ctx, &[1, 2, 3]).unwrap();
        let base = session.capture_snapshot().unwrap();
        for width in [1, 2, 8] {
            let tokens: Vec<u32> = (10..10 + width as u32).collect();
            session.restore_snapshot(&base).unwrap();
            let scalar = session
                .scalar_rows_with_captures(&ctx, &tokens, &mut bank, true)
                .unwrap();
            let scalar_logits = read_f32(scalar.logits().unwrap()).unwrap();
            let scalar_features = read_f32(scalar.features().unwrap()).unwrap();
            session.restore_snapshot(&base).unwrap();
            let packed = session
                .verify_packed_with_captures(&ctx, &tokens, &mut bank)
                .unwrap();
            report_numeric(
                &format!("{lineage:?} width={width} logits"),
                &read_f32(packed.logits().unwrap()).unwrap(),
                &scalar_logits,
                weights.config.vocab_size as usize,
            );
            report_numeric(
                &format!("{lineage:?} width={width} features"),
                &read_f32(packed.features().unwrap()).unwrap(),
                &scalar_features,
                weights.config.hidden_size as usize,
            );
        }
    }
}

#[test]
#[ignore = "GLM sparse failure gate: loads target weights; requires MTL_DEBUG_LAYER=1 and production lease"]
fn metal_all_row_failure_poisons_and_snapshot_restores() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required()).unwrap();
    let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
    let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, 2080, 128).unwrap();
    session.set_packed_lineage(PackedLineage::Exact).unwrap();
    session.prefill_packed(&ctx, &[1; 2048]).unwrap();
    let base = session.capture_snapshot().unwrap();
    let mut bank = Glm5NextRowBank::new(
        &ctx,
        &weights,
        Glm5NextRowBankPlan::new(&weights.config, &[42, 5], 8, 8).unwrap(),
    )
    .unwrap();
    session.corrupt_sparse_row = Some(4);
    let result = session.verify_packed_with_captures(&ctx, &[2; 8], &mut bank);
    assert!(matches!(
        result,
        Err(Glm5NextMetalError::KernelValidation { row: Some(4), .. })
    ));
    assert!(session.is_poisoned());
    assert_eq!(session.position(), 2048);
    assert!(session.capture_snapshot().is_err());
    session.restore_snapshot(&base).unwrap();
    assert!(!session.is_poisoned());
    let result = session
        .verify_packed_with_captures(&ctx, &[2; 8], &mut bank)
        .unwrap();
    assert_eq!(result.logit_positions(), 2048..2056);
}
