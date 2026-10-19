//! Data validity only. Error magnitudes and disagreements are measurements.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::path::Path;

fn finite(value: &Value, key: &str) -> Result<()> {
    let number = value[key]
        .as_f64()
        .with_context(|| format!("missing numeric {key}"))?;
    ensure!(
        number.is_finite() && number >= 0.,
        "invalid finite metric {key}: {number}"
    );
    Ok(())
}

fn tensor_metrics(value: &Value) -> Result<()> {
    for key in ["maxabs", "rmse", "normrelative"] {
        finite(value, key)?;
    }
    Ok(())
}

fn tensor_record(value: &Value) -> Result<()> {
    ensure!(
        value["position"].as_u64().is_some_and(|n| n < 7),
        "invalid tensor position"
    );
    match value["tensor"].as_str() {
        Some("logits") => ensure!(value["layer"].is_null(), "logits must have no layer"),
        Some("hidden" | "pre") => ensure!(
            value["layer"].as_u64().is_some_and(|n| n < 40),
            "invalid tensor layer"
        ),
        _ => anyhow::bail!("invalid tensor kind"),
    }
    tensor_metrics(value)
}

pub(super) fn check_characterization(report: &Value) -> Result<()> {
    ensure!(
        report["mode"] == "NativeF32" && report["comparison"] == "characterization_only",
        "wrong comparison mode"
    );
    ensure!(
        report["positions"] == 7 && report["layers"] == 40 && report["tensors_compared"] == 567,
        "incomplete tensor inventory"
    );
    ensure!(
        report["token_check"] == "complete" && report["token_positions_verified"] == 7,
        "incomplete consumed-token agreement"
    );
    ensure!(
        report["shapes"] == json!({"hidden":[4,5120],"pre":[4],"logits":[129280]}),
        "unexpected shapes"
    );
    tensor_record(&report["worst_maxabs"])?;
    let first = report
        .get("first_divergence")
        .context("missing first-divergence record")?;
    if !first.is_null() {
        tensor_record(first)?;
    }
    let logits = report["logits"]
        .as_array()
        .context("missing logits reports")?;
    ensure!(logits.len() == 7, "incomplete logits inventory");
    for (position, row) in logits.iter().enumerate() {
        ensure!(row["position"] == position, "wrong logits position");
        tensor_metrics(row)?;
        finite(row, "kl_reference_to_runtime")?;
        let reference = row["reference_top1"]
            .as_u64()
            .context("missing reference argmax")?;
        let runtime = row["runtime_top1"]
            .as_u64()
            .context("missing runtime argmax")?;
        ensure!(
            reference < 129280 && runtime < 129280,
            "argmax outside vocabulary"
        );
        ensure!(
            row["top1_match"].as_bool() == Some(reference == runtime),
            "inconsistent argmax report"
        );
    }
    Ok(())
}

pub(super) fn check_long_characterization(report: &Value) -> Result<()> {
    check_attention_characterization(report, &super::LONG_POSITIONS)
}

pub(super) fn check_attention_characterization(report: &Value, positions: &[usize]) -> Result<()> {
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
            "wrong attention inventory"
        );
        ensure!(
            row["owner"] == layer.min(20) && row["index_source"] == layer,
            "wrong source owner inventory"
        );
        for key in [
            "selection_exact",
            "owner_exact",
            "selection_boundary_tie",
            "candidate_boundary_tie",
        ] {
            ensure!(row[key].is_boolean(), "missing diagnostic {key}");
        }
        finite(row, "selection_boundary_gap")?;
        ensure!(
            row["selection_discard"].as_u64().is_some(),
            "missing selection discard count"
        );
        if layer >= 20 {
            ensure!(
                row["candidates_exact"].is_boolean(),
                "missing candidate comparison"
            );
            ensure!(
                row["candidate_discard"].as_u64().is_some(),
                "missing candidate discard count"
            );
        }
        ensure!(
            row["output"]["pass"].is_boolean(),
            "missing legacy diagnostic flag"
        );
        for key in ["max_abs", "rmse", "relative_l2"] {
            finite(&row["output"], key)?;
        }
    }
    Ok(())
}

pub(super) fn emit_characterization(output: &Path, scope: &str, report: &Value) -> Result<()> {
    let evidence = json!({
        "kind":"characterization", "scope":scope,
        "report_validity":"complete_finite", "numerical_acceptance":"not_evaluated",
        "note":"This report characterizes only its stated scope; finite metrics and legacy pass flags do not certify full-model numerical correctness",
        "metrics":report,
    });
    let text = serde_json::to_string_pretty(&evidence)?;
    std::fs::write(output.join("characterization.json"), format!("{text}\n"))?;
    println!("{text}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn short_report() -> Value {
        json!({"mode":"NativeF32", "comparison":"characterization_only", "positions":7,
            "layers":40,"tensors_compared":567,"token_check":"complete","token_positions_verified":7,
            "shapes":{"hidden":[4,5120],"pre":[4],"logits":[129280]},
            "worst_maxabs":{"position":6,"layer":39,"tensor":"hidden","maxabs":50.,"rmse":10.,"normrelative":2.},
            "first_divergence":null,
            "logits":(0..7).map(|p| json!({"position":p,"maxabs":5.,"rmse":2.,"normrelative":1.,
                "kl_reference_to_runtime":0.5,"reference_top1":5,"runtime_top1":6,"top1_match":false})).collect::<Vec<_>>()})
    }

    #[test]
    fn finite_conversion_error_and_argmax_changes_are_not_acceptance_failures() {
        let report = short_report();
        check_characterization(&report).unwrap();
        assert!(super::super::check_legacy_tight_report(&report).is_err());
    }

    #[test]
    fn characterization_rejects_incomplete_nonfinite_and_inconsistent_reports() {
        let valid = short_report();
        for key in [
            "mode",
            "comparison",
            "positions",
            "layers",
            "tensors_compared",
            "token_check",
            "token_positions_verified",
            "shapes",
            "worst_maxabs",
            "first_divergence",
            "logits",
        ] {
            let mut bad = valid.clone();
            bad.as_object_mut().unwrap().remove(key);
            assert!(
                check_characterization(&bad).is_err(),
                "accepted missing {key}"
            );
        }
        for key in ["maxabs", "rmse", "normrelative", "kl_reference_to_runtime"] {
            for value in [Value::Null, json!("inf"), json!("NaN"), json!(-1)] {
                let mut bad = valid.clone();
                bad["logits"][0][key] = value;
                assert!(check_characterization(&bad).is_err(), "accepted {key}");
            }
        }
        let mut bad = valid.clone();
        bad["logits"][6] = bad["logits"][0].clone();
        assert!(check_characterization(&bad).is_err());
        let mut bad = valid.clone();
        bad["logits"][0]["top1_match"] = json!(true);
        assert!(check_characterization(&bad).is_err());
        let mut bad = valid.clone();
        bad["logits"][0]["runtime_top1"] = json!(129280);
        assert!(check_characterization(&bad).is_err());
        let mut bad = valid;
        bad["worst_maxabs"]["normrelative"] = json!("inf");
        assert!(check_characterization(&bad).is_err());
    }

    fn long_report() -> Value {
        Value::Array(
            super::super::LONG_POSITIONS
                .into_iter()
                .flat_map(|p| {
                    super::super::LONG_LAYERS.into_iter().map(move |l| json!({
                "position":p,"layer":l,"owner":l.min(20),"index_source":l,
                "selection_exact":false,"owner_exact":true,"selection_boundary_tie":true,
                "candidate_boundary_tie":false,"selection_boundary_gap":0.,"selection_discard":17,
                "candidates_exact":false,"candidate_discard":8,
                "output":{"pass":false,"max_abs":10.,"rmse":3.,"relative_l2":2.}
            }))
                })
                .collect(),
        )
    }

    #[test]
    fn long_characterization_reports_differences_without_relaxing_legacy_bounds() {
        let valid = long_report();
        check_long_characterization(&valid).unwrap();
        assert!(super::super::check_legacy_tight_long_report(&valid).is_err());
        for key in ["max_abs", "rmse", "relative_l2"] {
            let mut bad = valid.clone();
            bad[31]["output"][key] = Value::Null;
            assert!(check_long_characterization(&bad).is_err());
        }
        let mut bad = valid.clone();
        bad[31] = bad[0].clone();
        assert!(check_long_characterization(&bad).is_err());
        let mut bad = valid;
        bad.as_array_mut().unwrap().pop();
        assert!(check_long_characterization(&bad).is_err());
    }

    #[test]
    fn batched_attention_inventory_requires_pruning_checkpoints() {
        let base = long_report();
        let mut rows = Vec::new();
        for position in super::super::BATCH_POSITIONS {
            for row in &base.as_array().unwrap()[..super::super::LONG_LAYERS.len()] {
                let mut row = row.clone();
                row["position"] = json!(position);
                rows.push(row);
            }
        }
        let mut report = Value::Array(rows);
        check_attention_characterization(&report, &super::super::BATCH_POSITIONS).unwrap();
        assert!(check_long_characterization(&report).is_err());
        report.as_array_mut().unwrap().pop();
        assert!(check_attention_characterization(&report, &super::super::BATCH_POSITIONS).is_err());
    }
}
