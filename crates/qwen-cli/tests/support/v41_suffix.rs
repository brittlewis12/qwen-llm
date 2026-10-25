//! Conditional attention qualification and separate native continuation self-parity.
use anyhow::{Context, Result, ensure};
use objc2_metal::MTLComputePipelineState;
use qwen_llm::deepseek_v41::forward::{Forward, State};
use serde_json::{Value, json};

pub(super) const RING_CALLS: [usize; 2] = [127, 513];
pub(super) const RING_END: usize = 640;
pub(super) const RING_POSITIONS: [usize; 15] = [
    126, 127, 128, 254, 255, 256, 381, 382, 383, 510, 511, 512, 637, 638, 639,
];
const PREFIX: &str = "kernel_deepseek_v41_attention_";

#[derive(Clone, Copy, Debug)]
pub(super) struct RingPlan {
    tensor: bool,
    score32: bool,
}

impl RingPlan {
    pub(super) fn selected(
        ctx: &qwen_llm::metal::MetalContext,
        weights: &qwen_llm::deepseek_v41::gguf::ResidentWeights,
    ) -> Result<Self> {
        let c = &weights.config;
        ensure!(
            c.heads == 64
                && c.head_dim == 512
                && c.window == 128
                && c.index_topk == 512
                && weights.schedule.layers == 40,
            "ring census requires released 40-layer/64-head/512-latent geometry"
        );
        ensure!(
            std::env::var_os("QWEN_V41_SINK_TILE").is_none(),
            "ring boundary witness requires the ordinary 512-query tile; unset QWEN_V41_SINK_TILE"
        );
        ensure!(
            ctx.pipeline("kernel_deepseek_v41_attention_normalize_rows")?
                .maxTotalThreadsPerThreadgroup()
                >= 128,
            "ring census requires the released 128-thread normalization geometry"
        );
        let enabled = |name| std::env::var_os(name).is_none_or(|v| v != "0");
        Ok(Self {
            tensor: qwen_llm::metal::moe_grouped_tensor_selected(ctx)
                && enabled("QWEN_V41_SINK_TENSOR"),
            score32: enabled("QWEN_V41_SINK_TENSOR32"),
        })
    }

    fn kernels(self, queries: usize) -> [&'static str; 4] {
        if self.tensor && queries >= 128 {
            [
                if self.score32 {
                    "score_tensor32"
                } else {
                    "score_tensor"
                },
                "softmax_rows",
                "pv_tensor",
                "normalize_rows",
            ]
        } else {
            ["score_mma", "softmax_rows", "pv_mma", "normalize_rows"]
        }
    }

    pub(super) fn identity(self) -> Value {
        json!({"layers":40, "heads":64, "head_dim":512, "window":128,
            "index_topk":512, "query_tile_rows":512,
            "tensor_selected":self.tensor, "tensor_score32_enabled":self.score32,
            "source":"loaded_geometry_and_context_selection_not_inferred_from_census",
            "limits":"census has no buffer bindings, query offsets, dispatch API kind or layer IDs"})
    }
}

fn sink_kernel(kernel: &str) -> bool {
    kernel.strip_prefix(PREFIX).is_some_and(|s| {
        [
            "score",
            "softmax",
            "pv",
            "normalize",
            "matrix_values",
            "attend",
        ]
        .iter()
        .any(|prefix| s.starts_with(prefix))
    })
}

fn check_ring_call(
    rows: &[qwen_llm::metal::DispatchCensusRow],
    queries: usize,
    plan: RingPlan,
) -> Result<Value> {
    ensure!(queries > 1, "ring call must be batched");
    let sink: Vec<_> = rows.iter().filter(|r| sink_kernel(&r.kernel)).collect();
    let tiles: Vec<_> = (0..queries)
        .step_by(512)
        .map(|q| (queries - q).min(512))
        .collect();
    ensure!(
        sink.len() == 40 * tiles.len() * 4,
        "wrong sink dispatch count: {} for {queries} queries",
        sink.len()
    );
    let mut previous_encoder = None;
    let mut cursor = 0;
    for layer in 0..40 {
        let encoder = sink[cursor].encoder_ordinal;
        ensure!(
            previous_encoder.is_none_or(|old| encoder >= old),
            "layer {layer}: sink encoders not ordered"
        );
        previous_encoder = Some(encoder);
        for &n in &tiles {
            for (stage, suffix) in plan.kernels(n).iter().enumerate() {
                let row = sink[cursor];
                cursor += 1;
                ensure!(
                    row.kernel == format!("{PREFIX}{suffix}"),
                    "layer {layer} queries {n} stage {stage}: wrong kernel {}",
                    row.kernel
                );
                ensure!(
                    !row.encoder_concurrent && row.encoder_ordinal == encoder,
                    "layer {layer}: wrong encoder type/order"
                );
                let grid = [row.grid_width, row.grid_height, row.grid_depth];
                let tg = [row.threads_width, row.threads_height, row.threads_depth];
                if stage == 3 {
                    ensure!(
                        grid == [n as u64 * 256, 1, 1] && tg == [128, 1, 1],
                        "wrong normalize geometry"
                    );
                } else {
                    let expected_grid = if stage == 1 {
                        [64, n as u64, 1]
                    } else {
                        [4, n as u64, 1]
                    };
                    let threads = if stage == 1 {
                        32
                    } else if stage == 2 && *suffix == "pv_mma" {
                        256
                    } else {
                        128
                    };
                    ensure!(
                        grid == expected_grid && tg == [threads, 1, 1],
                        "layer {layer} stage {stage}: wrong grid/threadgroup {grid:?}/{tg:?}"
                    );
                }
                ensure!(
                    row.grid_tgs == grid.iter().product::<u64>()
                        && row.tg_threads == tg.iter().product::<u64>(),
                    "inconsistent census totals"
                );
            }
        }
    }
    let mut shapes = std::collections::BTreeMap::new();
    for row in &sink {
        *shapes
            .entry((
                row.kernel.clone(),
                [row.grid_width, row.grid_height, row.grid_depth],
                [row.threads_width, row.threads_height, row.threads_depth],
            ))
            .or_insert(0usize) += 1;
    }
    let shapes: Vec<_> = shapes.into_iter().map(|((kernel, grid, threadgroup), dispatches)|
        json!({"kernel":kernel, "grid":grid, "threadgroup":threadgroup, "dispatches":dispatches})
    ).collect();
    Ok(json!({"prefill_rows":queries, "query_tiles":tiles,
        "sink_dispatches":sink.len(), "layer_sink_groups":40,
        "decode_dispatches":0, "serial_stage_order_exact":true, "kernel_shapes":shapes}))
}

pub(super) fn ring_census(
    calls: &[Vec<qwen_llm::metal::DispatchCensusRow>],
    plan: RingPlan,
) -> Result<Value> {
    ensure!(
        calls.len() == RING_CALLS.len(),
        "ring census needs both capture calls"
    );
    let reports = calls
        .iter()
        .zip(RING_CALLS)
        .map(|(rows, n)| check_ring_call(rows, n, plan))
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"kind":"dispatch_census", "scope":"ring_capture_prefill_only_not_timing",
        "selected_geometry":plan.identity(), "prefill_call_rows":RING_CALLS,
        "calls":reports, "boundary_positions":[638,639],
        "note":"Exact released-geometry sink sequence, not independent full-model numerical acceptance"}),
    )
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
        start == RING_END && tokens.len() == 3,
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

    // These literal launch shapes are an independent witness fixture, not
    // generated by RingPlan::kernels or the validator's tile calculation.
    fn census_fixture(tiles: &[(u64, &str, &str, u64)]) -> Vec<qwen_llm::metal::DispatchCensusRow> {
        let mut rows = Vec::new();
        for layer in 0..40 {
            for &(n, score, pv, pv_threads) in tiles {
                for (name, grid, threads) in [
                    (score, [4, n, 1], 128),
                    ("softmax_rows", [64, n, 1], 32),
                    (pv, [4, n, 1], pv_threads),
                    ("normalize_rows", [n * 256, 1, 1], 128),
                ] {
                    rows.push(qwen_llm::metal::DispatchCensusRow {
                        family: "",
                        tag: None,
                        encoder_ordinal: layer,
                        encoder_concurrent: false,
                        kernel: format!("{PREFIX}{name}"),
                        grid_width: grid[0],
                        grid_height: grid[1],
                        grid_depth: grid[2],
                        threads_width: threads,
                        threads_height: 1,
                        threads_depth: 1,
                        grid_tgs: grid.iter().product(),
                        tg_threads: threads,
                    });
                }
            }
        }
        rows
    }

    #[test]
    fn ring_census_positive_selected_tiers_and_real_query_boundary() -> Result<()> {
        let short = census_fixture(&[(127, "score_mma", "pv_mma", 256)]);
        for (tensor, score32, score, pv, threads) in [
            (false, false, "score_mma", "pv_mma", 256),
            (true, false, "score_tensor", "pv_tensor", 128),
            (true, true, "score_tensor32", "pv_tensor", 128),
        ] {
            let plan = RingPlan { tensor, score32 };
            let packed = census_fixture(&[(257, score, pv, threads)]);
            check_ring_call(&packed, 257, plan)?;
            let split =
                census_fixture(&[(512, score, pv, threads), (1, "score_mma", "pv_mma", 256)]);
            let report = ring_census(&[short.clone(), split], plan)?;
            assert_eq!(report["calls"][0]["sink_dispatches"], 160);
            assert_eq!(report["calls"][1]["sink_dispatches"], 320);
            assert_eq!(report["calls"][1]["query_tiles"], json!([512, 1]));
            // Ordinary unsampled execution can keep all layers in one serial encoder.
            let mut single = packed;
            for row in &mut single {
                row.encoder_ordinal = 0;
            }
            check_ring_call(&single, 257, plan)?;
        }
        check_schedule(&RING_POSITIONS, &RING_CALLS)?;
        assert!(RING_POSITIONS.contains(&638) && RING_POSITIONS.contains(&639));
        assert_eq!(RING_POSITIONS.last().copied(), Some(RING_END - 1));
        Ok(())
    }

    #[test]
    fn ring_census_rejects_missing_wrong_kernel_type_count_order_and_geometry() {
        let plan = RingPlan {
            tensor: true,
            score32: true,
        };
        let valid = census_fixture(&[
            (512, "score_tensor32", "pv_tensor", 128),
            (1, "score_mma", "pv_mma", 256),
        ]);
        check_ring_call(&valid, 513, plan).unwrap();
        for index in [0, 1, 2, 3, 4, 7, valid.len() - 1] {
            let mut bad = valid.clone();
            bad.remove(index);
            assert!(check_ring_call(&bad, 513, plan).is_err());
            let mut bad = valid.clone();
            bad.push(valid[index].clone());
            assert!(check_ring_call(&bad, 513, plan).is_err());
            for kernel in [
                "score_rows",
                "score_mma",
                "score_tensor",
                "pv_tensor",
                "softmax",
                "unknown",
            ] {
                let mut bad = valid.clone();
                bad[index].kernel = format!("{PREFIX}{kernel}");
                if bad[index].kernel != valid[index].kernel {
                    assert!(check_ring_call(&bad, 513, plan).is_err());
                }
            }
            let mut bad = valid.clone();
            bad[index].encoder_concurrent = true;
            assert!(check_ring_call(&bad, 513, plan).is_err());
            let mut bad = valid.clone();
            bad[index].grid_height += 1;
            assert!(check_ring_call(&bad, 513, plan).is_err());
            let mut bad = valid.clone();
            bad[index].threads_width *= 2;
            assert!(check_ring_call(&bad, 513, plan).is_err());
            let mut bad = valid.clone();
            bad[index].grid_tgs += 1;
            assert!(check_ring_call(&bad, 513, plan).is_err());
            let mut bad = valid.clone();
            bad[index].tg_threads += 1;
            assert!(check_ring_call(&bad, 513, plan).is_err());
        }
        for (a, b) in [(0, 1), (2, 3), (0, 4), (3, 4)] {
            let mut bad = valid.clone();
            bad.swap(a, b);
            assert!(check_ring_call(&bad, 513, plan).is_err());
        }
        let mut bad = valid.clone();
        bad[8].encoder_ordinal = 0;
        bad[16].encoder_ordinal = 0; // precedes layer 1's encoder
        bad[8].encoder_ordinal = 2;
        assert!(check_ring_call(&bad, 513, plan).is_err());
        let mut bad = valid.clone();
        bad[3].grid_width *= 2;
        bad[3].grid_tgs *= 2;
        bad[3].threads_width /= 2;
        bad[3].tg_threads /= 2;
        assert!(check_ring_call(&bad, 513, plan).is_err());
        assert!(ring_census(&[], plan).is_err());
        assert!(ring_census(&[vec![], vec![]], plan).is_err());
    }

    #[test]
    fn ring_census_stale_suffix_red_control() {
        let plan = RingPlan {
            tensor: false,
            score32: false,
        };
        // The historical [256, 1] dispatch hypothesis for a 257-row call is
        // rejected even if every individual kernel is otherwise legitimate.
        let stale_split = census_fixture(&[
            (256, "score_mma", "pv_mma", 256),
            (1, "score_mma", "pv_mma", 256),
        ]);
        assert!(check_ring_call(&stale_split, 257, plan).is_err());
        let stale_tile = census_fixture(&[
            (32, "score_mma", "pv_mma", 256),
            (95, "score_mma", "pv_mma", 256),
        ]);
        assert!(check_ring_call(&stale_tile, 127, plan).is_err());
        let stale_kernel = census_fixture(&[(127, "score_rows", "pv_mma", 256)]);
        assert!(check_ring_call(&stale_kernel, 127, plan).is_err());
        let packed = census_fixture(&[(257, "score_mma", "pv_mma", 256)]);
        check_ring_call(&packed, 257, plan).unwrap();
        // A packed 513-row launch does not prove the actual query boundary.
        let unsplit = census_fixture(&[(513, "score_mma", "pv_mma", 256)]);
        assert!(check_ring_call(&unsplit, 513, plan).is_err());
    }

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
            (&RING_POSITIONS[..], &RING_CALLS[..]),
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
