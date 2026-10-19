//! Conditional attention qualification and separate native continuation self-parity.
use anyhow::{Context, Result, ensure};
use qwen_llm::deepseek_v41::forward::{Forward, State};
use serde_json::{Value, json};

pub(super) const RING_POSITIONS: [usize; 9] = [126, 127, 128, 254, 255, 256, 381, 382, 383];

pub(super) fn ring_census(calls: &[Vec<qwen_llm::metal::DispatchCensusRow>]) -> Result<Value> {
    ensure!(calls.len() == 2, "ring census needs both capture calls");
    let names = ["score_rows", "softmax_rows", "score", "softmax"]
        .map(|s| format!("kernel_deepseek_v41_attention_{s}"));
    let mut totals = [0usize; 4];
    let mut reports = Vec::new();
    for (call, rows) in calls.iter().enumerate() {
        let counts = std::array::from_fn(|i| rows.iter().filter(|r| r.kernel == names[i]).count());
        let last_batch = rows
            .iter()
            .filter(|r| names[..2].contains(&r.kernel))
            .map(|r| r.encoder_ordinal)
            .max()
            .context("missing batched attention census")?;
        let decode_separate = rows
            .iter()
            .filter(|r| names[2..].contains(&r.kernel))
            .all(|r| r.encoder_ordinal > last_batch);
        check_ring_dispatches(call, counts, decode_separate)?;
        for (total, count) in totals.iter_mut().zip(counts) {
            *total += count;
        }
        let mut shapes = std::collections::BTreeMap::new();
        for r in rows {
            let grid = [r.grid_width, r.grid_height, r.grid_depth];
            let tg = [r.threads_width, r.threads_height, r.threads_depth];
            *shapes
                .entry((r.kernel.clone(), grid, tg, r.grid_tgs, r.tg_threads))
                .or_insert(0usize) += 1;
        }
        let shapes: Vec<_> = shapes
            .into_iter()
            .map(|((kernel, grid, tg, grid_tgs, tg_threads), count)| {
                json!({
                    "kernel":kernel, "grid":grid, "threadgroup":tg, "dispatches":count,
                    "total_grid_tgs":count as u64 * grid_tgs, "tg_threads":tg_threads
                })
            })
            .collect();
        reports.push(json!({"start":([0,127][call]), "prefill_rows":([127,257][call]),
            "attention_dispatches":counts, "decode_encoders_after_batch":decode_separate, "kernel_shapes":shapes}));
    }
    Ok(
        json!({"kind":"dispatch_census", "scope":"ring_capture_prefill_only_not_timing",
        "attention_kernels":names, "attention_dispatch_totals":totals, "calls":reports}),
    )
}

fn check_ring_dispatches(call: usize, counts: [usize; 4], decode_separate: bool) -> Result<()> {
    // run_captured uses [127] then [256, 1]; 32-query tiles do not straddle PV thresholds here.
    let expected = [[160, 160, 0, 0], [320, 320, 40, 40]];
    ensure!(
        expected.get(call) == Some(&counts),
        "ring call {call}: attention dispatches {counts:?}"
    );
    ensure!(
        decode_separate,
        "old row score/softmax appeared inside batched encoders"
    );
    Ok(())
}

#[test]
fn ring_dispatch_census_negative_controls() {
    for (call, valid) in [(0, [160, 160, 0, 0]), (1, [320, 320, 40, 40])] {
        check_ring_dispatches(call, valid, true).unwrap();
        assert!(check_ring_dispatches(call, valid, false).is_err());
        assert!(check_ring_dispatches(call, [0; 4], true).is_err());
        for i in 0..4 {
            let mut bad = valid;
            bad[i] += 1;
            assert!(check_ring_dispatches(call, bad, true).is_err());
        }
    }
    assert!(ring_census(&[]).is_err());
    assert!(ring_census(&[vec![], vec![]]).is_err());
}

pub(super) fn check_schedule(positions: &[usize], calls: &[usize]) -> Result<()> {
    ensure!(
        !positions.is_empty() && positions[0] >= 1 && positions.windows(2).all(|p| p[0] < p[1]),
        "checkpoints must be nonempty and strictly increasing"
    );
    let count = calls.iter().try_fold(0usize, |n, &rows| {
        ensure!(rows > 0, "empty prefill call");
        n.checked_add(rows).context("prefill schedule overflow")
    })?;
    ensure!(
        positions.last().unwrap().checked_add(1) == Some(count),
        "prefill schedule must end at final checkpoint"
    );
    Ok(())
}

pub(super) fn qualify(
    positions: &[usize],
    calls: &[usize],
    check_wrapped_continuation: bool,
) -> Result<()> {
    let (output, report) =
        super::batched_attention_replay(positions, calls, check_wrapped_continuation)?;
    check_report(&report, positions).context("suffix attention qualification failed")?;
    let evidence = json!({
        "kind":"qualification", "scope":"attention_only_captured_normalized_inputs",
        "weight_backend":"independent_gguf", "prefill_call_rows":calls,
        "requested_checkpoints":positions, "matched_checkpoints":positions.len(),
        "attention_rows":positions.len() * super::LONG_LAYERS.len(),
        "candidate_rows":positions.len() * super::LONG_LAYERS.iter().filter(|&&l| l >= 20).count(),
        "pruned_candidate_rows":positions.iter().filter(|&&p| p >= 16384).count()
            * super::LONG_LAYERS.iter().filter(|&&l| l >= 20).count(),
        "output_elements":positions.len() * super::LONG_LAYERS.len() * 5120,
        "atol":1e-4, "rtol":0, "status":"pass",
        "note":"Not independent full-model or logit acceptance",
        "metrics":report,
    });
    let text = serde_json::to_string_pretty(&evidence)?;
    std::fs::write(output.join("qualification.json"), format!("{text}\n"))?;
    println!("{text}");
    Ok(())
}

pub(super) fn wrapped_continuation(
    forward: &Forward<'_>,
    uninterrupted: &mut State,
    tokens: &[u32],
) -> Result<Value> {
    let start = uninterrupted.position();
    ensure!(
        start == 384 && tokens.len() == 3,
        "wrong wrapped continuation extent"
    );
    let end = start + tokens.len();
    let original_capacity = uninterrupted.capacity();
    ensure!(
        original_capacity == start,
        "continuation must require growth"
    );
    let snapshot = forward.capture_snapshot(uninterrupted)?;
    ensure!(snapshot.position() == start, "snapshot position mismatch");
    let mut restored = forward.new_state_with_gpu_step(original_capacity)?;
    forward.restore_snapshot(&mut restored, &snapshot)?;
    ensure!(restored.position() == start, "restore position mismatch");
    forward.ensure_capacity(uninterrupted, end)?;
    forward.ensure_capacity(&mut restored, end)?;
    for state in [&*uninterrupted, &restored] {
        ensure!(
            state.capacity() > original_capacity
                && state.capacity() >= end
                && state.position() == start,
            "growth did not preserve the wrapped prefix or increase capacity"
        );
    }
    let cancel = Some(qwen_cli::shutdown::cancellation_flag());
    let expected = forward.prefill(uninterrupted, tokens, cancel)?;
    let actual = forward.prefill(&mut restored, tokens, cancel)?;
    check_continuation_logits(
        &actual,
        &expected,
        restored.position(),
        uninterrupted.position(),
        end,
    )?;
    qwen_cli::shutdown::checkpoint()?;
    Ok(json!({
        "kind":"self_parity", "scope":"native_wrapped_prefix_snapshot_restore_growth_continuation",
        "reference":"uninterrupted_native_state_after_growth", "mode":"NativeF32", "status":"pass",
        "captured_tokens":start, "snapshot_position":snapshot.position(),
        "continuation_token_ids":tokens, "prefill_call_rows":[tokens.len()],
        "original_capacity":original_capacity,
        "uninterrupted_grown_capacity":uninterrupted.capacity(), "restored_grown_capacity":restored.capacity(),
        "uninterrupted_next_position":uninterrupted.position(), "restored_next_position":restored.position(),
        "compared_logit_positions":[end - 1], "logit_vectors":1, "logit_elements":actual.len(),
        "logits_bitwise_equal":true, "positions_exact":true,
        "note":"Same-path native self-parity only; both states grow. Endpoint logits only, not independent full-model acceptance or additional attention capture"
    }))
}

fn check_continuation_logits(
    actual: &[f32],
    expected: &[f32],
    actual_position: usize,
    expected_position: usize,
    end: usize,
) -> Result<()> {
    ensure!(
        actual_position == end && expected_position == end,
        "continuation position mismatch"
    );
    ensure!(
        actual.len() == 129280 && expected.len() == actual.len(),
        "continuation logit shape mismatch"
    );
    ensure!(
        actual.iter().chain(expected).all(|v| v.is_finite()),
        "nonfinite continuation logits"
    );
    ensure!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "continuation logits differ bitwise"
    );
    Ok(())
}

fn finite(value: &Value, key: &str) -> Result<f64> {
    let n = value[key]
        .as_f64()
        .with_context(|| format!("missing {key}"))?;
    ensure!(n.is_finite() && n >= 0., "invalid {key}: {n}");
    Ok(n)
}

fn check_report(report: &Value, positions: &[usize]) -> Result<()> {
    ensure!(
        !positions.is_empty() && positions[0] >= 1 && positions.windows(2).all(|p| p[0] < p[1]),
        "invalid qualification checkpoints"
    );
    let rows = report.as_array().context("missing attention reports")?;
    ensure!(
        rows.len() == positions.len() * super::LONG_LAYERS.len(),
        "incomplete attention inventory"
    );
    for (row, (position, layer)) in rows.iter().zip(
        positions
            .iter()
            .copied()
            .flat_map(|p| super::LONG_LAYERS.into_iter().map(move |l| (p, l))),
    ) {
        ensure!(
            row["position"] == position && row["layer"] == layer,
            "wrong checkpoint inventory"
        );
        ensure!(
            row["weight_backend"] == "independent_gguf"
                && row["comparison_scope"] == "attention_only_captured_normalized_inputs",
            "wrong replay backend or scope"
        );
        ensure!(
            row["owner"] == layer.min(20) && row["index_source"] == layer,
            "wrong owners"
        );
        for key in ["selection_exact", "owner_exact"] {
            ensure!(
                row[key] == true,
                "position={position} layer={layer}: {key} mismatch"
            );
        }
        for key in ["selection_boundary_tie", "candidate_boundary_tie"] {
            ensure!(
                row[key] == false,
                "position={position} layer={layer}: missing or tied {key}"
            );
        }
        let count = (position + 1) / if layer < 20 { 2 } else { 1 };
        let discarded = count.saturating_sub(512);
        ensure!(
            row["selection_discard"] == discarded,
            "wrong selection denominator"
        );
        if discarded == 0 {
            ensure!(
                row.get("selection_boundary_gap") == Some(&Value::Null),
                "unexpected selection boundary"
            );
        } else {
            ensure!(
                finite(row, "selection_boundary_gap")? > 0.,
                "nonpositive selection gap"
            );
        }
        if layer >= 20 {
            ensure!(
                row["candidates_exact"] == true,
                "position={position} layer={layer}: candidate mismatch"
            );
            // Newest (possibly partial) block is always retained; discarded blocks are full.
            let discarded = count.div_ceil(8).saturating_sub(2048) * 8;
            ensure!(
                row["candidate_discard"] == discarded,
                "wrong candidate denominator"
            );
        }
        let output = &row["output"];
        ensure!(output["elements"] == 5120, "wrong output denominator");
        ensure!(
            output["atol"] == 1e-4 && output["rtol"] == 0.,
            "wrong output tolerance"
        );
        ensure!(
            output["pass"] == true,
            "position={position} layer={layer}: attention comparison failed"
        );
        ensure!(
            finite(output, "max_abs")? <= 1e-4,
            "attention max_abs exceeds 1e-4"
        );
        ensure!(
            finite(output, "rmse")? <= 1e-4,
            "attention rmse exceeds 1e-4"
        );
        finite(output, "relative_l2")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapped_continuation_self_parity_rejects_vacuous_and_nonbitwise_results() -> Result<()> {
        let valid = vec![0.; 129280];
        check_continuation_logits(&valid, &valid, 387, 387, 387)?;
        for (actual, expected) in [
            (&[][..], &[][..]),
            (&valid[..1], &valid[..1]),
            (&valid[..129279], &valid[..]),
        ] {
            assert!(check_continuation_logits(actual, expected, 387, 387, 387).is_err());
        }
        for (actual, expected) in [(386, 387), (387, 386), (386, 386)] {
            assert!(check_continuation_logits(&valid, &valid, actual, expected, 387).is_err());
        }
        for value in [-0., f32::from_bits(1), f32::NAN, f32::INFINITY] {
            let mut bad = valid.clone();
            bad[129279] = value;
            assert!(check_continuation_logits(&bad, &valid, 387, 387, 387).is_err());
            assert!(check_continuation_logits(&valid, &bad, 387, 387, 387).is_err());
        }
        let mut bad = valid;
        bad[0] = f32::INFINITY;
        assert!(check_continuation_logits(&bad, &bad, 387, 387, 387).is_err());
        Ok(())
    }

    fn report(positions: &[usize]) -> Value {
        json!(positions.iter().flat_map(|&p| super::super::LONG_LAYERS.into_iter().map(move |l| {
            let count = (p + 1) / if l < 20 { 2 } else { 1 };
            json!({"position":p, "layer":l, "owner":l.min(20), "index_source":l,
                "weight_backend":"independent_gguf", "comparison_scope":"attention_only_captured_normalized_inputs",
                "selection_exact":true, "owner_exact":true, "candidates_exact":true,
                "selection_discard":count.saturating_sub(512),
                "candidate_discard":count.div_ceil(8).saturating_sub(2048) * 8,
                "selection_boundary_gap":(count > 512).then_some(0.01),
                "selection_boundary_tie":false, "candidate_boundary_tie":false,
                "output":{"elements":5120,"atol":1e-4,"rtol":0.,"pass":true,"max_abs":1e-4,"rmse":1e-5,"relative_l2":0.01}})
        })).collect::<Vec<_>>())
    }

    #[test]
    fn suffix_schedule_and_report_inventory_are_nonvacuous() -> Result<()> {
        for (positions, calls) in [
            (&super::super::BATCH_POSITIONS[..], &[16383, 3][..]),
            (&RING_POSITIONS[..], &[127, 257][..]),
        ] {
            check_schedule(positions, calls)?;
            let valid = report(positions);
            check_report(&valid, positions)?;
            for key in [
                "position",
                "layer",
                "owner",
                "index_source",
                "weight_backend",
                "comparison_scope",
                "selection_exact",
                "owner_exact",
                "candidates_exact",
                "selection_discard",
                "candidate_discard",
                "selection_boundary_gap",
                "selection_boundary_tie",
                "candidate_boundary_tie",
                "output",
            ] {
                let mut bad = valid.clone();
                bad.as_array_mut()
                    .unwrap()
                    .last_mut()
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .remove(key);
                assert!(
                    check_report(&bad, positions).is_err(),
                    "accepted missing {key}"
                );
            }
            let mut rows = valid.as_array().unwrap().clone();
            rows.pop();
            assert!(check_report(&json!(rows), positions).is_err());
            rows.push(valid[0].clone());
            assert!(check_report(&json!(rows), positions).is_err());
            rows.extend(valid.as_array().unwrap().iter().cloned());
            assert!(check_report(&json!(rows), positions).is_err());
        }
        assert!(check_report(&json!([]), &[]).is_err());
        assert!(check_report(&json!([]), &RING_POSITIONS).is_err());
        for (positions, calls) in [
            (&[][..], &[1][..]),
            (&[1, 1], &[2]),
            (&[2, 1], &[2]),
            (&[0], &[1]),
            (&[383], &[127, 256]),
            (&[383], &[127, 257, 1]),
            (&[383], &[0, 384]),
            (&[383], &[]),
            (&[383], &[usize::MAX, 1]),
        ] {
            assert!(check_schedule(positions, calls).is_err());
        }
        Ok(())
    }

    #[test]
    fn suffix_qualification_rejects_disagreement_and_overbound_even_with_pass_flag() {
        let positions = &super::super::BATCH_POSITIONS;
        let valid = report(positions);
        let last = valid.as_array().unwrap().len() - 1;
        for (key, value) in [
            ("selection_exact", json!(false)),
            ("owner_exact", json!(false)),
            ("candidates_exact", json!(false)),
            ("selection_discard", json!(0)),
            ("candidate_discard", json!(0)),
            ("selection_boundary_tie", json!(true)),
            ("candidate_boundary_tie", json!(true)),
            ("selection_boundary_gap", json!(0)),
        ] {
            let mut bad = valid.clone();
            bad[last][key] = value;
            assert!(check_report(&bad, positions).is_err(), "accepted {key}");
        }
        for (key, value) in [
            ("pass", json!(false)),
            ("elements", json!(0)),
            ("elements", json!(5119)),
            ("atol", json!(0.02)),
            ("rtol", json!(0.02)),
            ("max_abs", json!(0.00010001)),
            ("max_abs", json!(-1)),
            ("max_abs", json!("NaN")),
            ("max_abs", json!("inf")),
            ("max_abs", Value::Null),
            ("rmse", json!(0.00010001)),
            ("relative_l2", json!(-1)),
        ] {
            let mut bad = valid.clone();
            bad[last]["output"][key] = value;
            assert!(check_report(&bad, positions).is_err(), "accepted {key}");
        }
        for key in [
            "pass",
            "elements",
            "atol",
            "rtol",
            "max_abs",
            "rmse",
            "relative_l2",
        ] {
            let mut bad = valid.clone();
            bad[last]["output"].as_object_mut().unwrap().remove(key);
            assert!(
                check_report(&bad, positions).is_err(),
                "accepted missing {key}"
            );
        }
        // The old characterization still measures, rather than accepts, numeric failures.
        let mut bad = valid;
        bad[last]["output"]["pass"] = json!(false);
        bad[last]["output"]["max_abs"] = json!(10.);
        super::super::characterization::check_attention_characterization(&bad, positions).unwrap();
        assert!(check_report(&bad, positions).is_err());
    }
}
