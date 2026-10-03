//! Bounded first-site oracle: no prior intervention can change the reference input.

use super::*;
use crate::serve::http::GenerationBackend;
use anyhow::{Context, Result, ensure};
use qwen_llm::{
    gguf::GgufFile,
    runtime::{LoadedModelConfig, Runtime},
};

pub(super) fn vector(value: &Value) -> Result<Vec<f64>> {
    value
        .as_array()
        .context("oracle vector")?
        .iter()
        .map(|v| {
            let x = v.as_f64().context("oracle numeric component")?;
            ensure!(x.is_finite(), "nonfinite oracle component");
            Ok(x)
        })
        .collect()
}

fn tolerance(value: f64) -> f64 {
    1e-3 + 1e-4 * value.abs()
}

pub(super) fn distinguish_orders(
    a: &[f64],
    b: &[f64],
    actual_a: &[f64],
    actual_b: &[f64],
) -> Result<()> {
    ensure!(
        !a.is_empty()
            && [b.len(), actual_a.len(), actual_b.len()]
                .iter()
                .all(|&n| n == a.len()),
        "order witness shape"
    );
    ensure!(
        a.iter()
            .chain(b)
            .chain(actual_a)
            .chain(actual_b)
            .all(|v| v.is_finite()),
        "nonfinite order witness"
    );
    ensure!(
        a.iter()
            .zip(b)
            .any(|(&x, &y)| (x - y).abs() > tolerance(x) + tolerance(y)),
        "reference order tolerance bands overlap"
    );
    for (actual, intended, opposite) in [(actual_a, a, b), (actual_b, b, a)] {
        ensure!(
            actual
                .iter()
                .zip(intended)
                .all(|(&x, &y)| (x - y).abs() <= tolerance(y)),
            "wrong intended order"
        );
        ensure!(
            actual
                .iter()
                .zip(opposite)
                .any(|(&x, &y)| (x - y).abs() > tolerance(y)),
            "opposite order was not rejected"
        );
    }
    Ok(())
}

#[test]
fn order_witness_requires_disjoint_tolerances_and_rejects_swapped_results() {
    assert!(distinguish_orders(&[1.], &[1.0005], &[1.], &[1.0005]).is_err());
    assert!(distinguish_orders(&[1.], &[2.], &[1.], &[2.]).is_ok());
    assert!(distinguish_orders(&[1.], &[2.], &[2.], &[1.]).is_err());
}

pub(super) fn transform(
    mut source: Vec<f64>,
    directions: &std::collections::BTreeMap<&str, Vec<f64>>,
    operations: &[Value],
) -> Result<Vec<f64>> {
    for operation in operations {
        let action = &operation["action"];
        let direction = &directions[action["direction"].as_str().context("direction id")?];
        ensure!(direction.len() == source.len(), "direction shape");
        let coefficient = action["coefficient"].as_f64().context("coefficient")?;
        let scale = match action["kind"].as_str().context("operator")? {
            "fixed_add" => coefficient,
            "projection_ablate" => {
                -coefficient
                    * source
                        .iter()
                        .zip(direction)
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
            }
            "residual_l2_fraction" => {
                coefficient * source.iter().map(|x| x * x).sum::<f64>().sqrt()
            }
            _ => anyhow::bail!("unexpected operator"),
        };
        for (x, d) in source.iter_mut().zip(direction) {
            *x += scale * d;
        }
    }
    Ok(source)
}

#[test]
#[ignore = "Metal: QWEN_LENS_ORACLE_MODE=interventions scripts/serve/lens_fitted_check.ts"]
fn ordered_interventions_cpu_oracle() -> Result<()> {
    crate::shutdown::install()?;
    let paired = std::env::var("QWEN_LENS_ORACLE_MODE").as_deref() == Ok("pairs");
    let model = std::path::PathBuf::from(std::env::var("QWEN_LENS_TEST_MODEL")?);
    let output = std::path::PathBuf::from(std::env::var("QWEN_LENS_TEST_OUTPUT")?);
    ensure!(!output.exists(), "fresh oracle output required");
    std::fs::create_dir(&output)?;
    let gguf = GgufFile::open(&model)?;
    let (layer, last) = super::fitted_live::synthetic_assets(&output, &gguf)?;
    let protocol = crate::prompt_template::identify_qwen_release_for_gguf(&gguf)?
        .template
        .serve_template();
    let registry = Arc::new(registry::Registry::open(
        &output.join("config.json"),
        &gguf,
        &mut crate::shutdown::checkpoint,
    )?);
    let store = Arc::new(JobStore::open(
        &output.join("jobs"),
        crate::serve::jobs::store::Limits::default(),
    )?);
    let runtime = Runtime::metal()?;
    let started = std::time::Instant::now();
    let loaded = runtime.load_opened_gguf_with_config(
        gguf,
        model,
        LoadedModelConfig {
            prefix_cache_max_bytes: 0,
            ..Default::default()
        },
    )?;
    if paired {
        super::paired_live::validate_destinations(&loaded)?;
    }
    let mut backend = crate::serve::backend::EngineBackend::new(
        loaded,
        "intervention-oracle".into(),
        3,
        Some(256),
        256,
        None,
        protocol,
        true,
    )?;
    backend.attach_lens_registry(Some(registry))?;
    let profile = backend.native_profile()?.context("native profile")?;
    let scope = json!({"layers":{"kind":"values","values":[layer]},"prefill":{"kind":"values","values":[0]}});
    let readout =
        json!({"id":"after","lens":"plain","mode":"full_vocabulary","top_k":5,"scope":scope});
    let mut value = json!({"schema_version":1,"idempotency_key":"baseline",
        "input":{"kind":"messages","messages":[{"role":"user","content":"Name an animal."}],"generation_mode":"thinking","assistant_prefill":{"channel":"reasoning","text":"Let me"}},
        "generation":{"max_new_tokens":3,"sampling":{"temperature":0,"top_k":0,"top_p":1,"min_p":0,"seed":7}},
        "diagnostics":{"directions":[],"operations":[],"readouts":[readout]}});
    let direction = |id, normalization| json!({"id":id,"lens":"oracle","row":{"kind":"token_id","token_id":1},"normalization":normalization});
    let scope = if paired {
        json!({"layers":{"kind":"values","values":[layer]},"prefill":{"kind":"values","values":[0,1]},"decode":{"kind":"values","values":[0]}})
    } else {
        scope
    };
    let operation = |id, kind, direction, coefficient| json!({"id":id,"scope":scope,"action":{"kind":kind,"direction":direction,"coefficient":coefficient}});
    let actions = vec![
        operation("add", "fixed_add", "raw", 0.25),
        operation("ablate", "projection_ablate", "unit", 0.5),
        operation("relative", "residual_l2_fraction", "unit", 0.1),
    ];
    let mut ids = Vec::new();
    let mut authored = Vec::new();
    let kinds = if paired {
        vec![
            "baseline",
            "zero",
            "active",
            "operation-only",
            "reordered",
            "pair-only",
        ]
    } else {
        vec!["baseline", "zero", "active", "operation-only", "reordered"]
    };
    for kind in kinds {
        value["idempotency_key"] = json!(kind);
        value["diagnostics"]["directions"] = json!([]);
        value["diagnostics"]["operations"] = json!([]);
        if kind != "baseline" && kind != "pair-only" {
            value["diagnostics"]["directions"] =
                json!([direction("raw", "as_stored"), direction("unit", "unit_l2")]);
            let mut operations = actions.clone();
            if kind == "zero" {
                for op in &mut operations {
                    op["action"]["coefficient"] = json!(0);
                }
            }
            if kind == "reordered" {
                operations.swap(0, 1);
            }
            value["diagnostics"]["operations"] = json!(operations);
        }
        value["diagnostics"]["readouts"] = if kind == "operation-only" || kind == "pair-only" {
            json!([])
        } else {
            json!([readout.clone()])
        };
        if paired {
            value["diagnostics"]["residual_pairs"] = if kind == "operation-only" {
                json!([])
            } else {
                json!([{"id":"whole","scope":{"layers":{"kind":"values","values":[layer,last]},"prefill":{"kind":"values","values":[1]},"decode":{"kind":"values","values":[0]}}}])
            };
        }
        let prepared = profile
            .prepare(&Request::parse(&value)?)
            .map_err(|e| anyhow::anyhow!("{}", e.error.message))?;
        if kind == "zero" {
            ensure!(
                prepared.staging.keys.is_empty() && prepared.interventions.rows.is_empty(),
                "zero control allocated work"
            );
        }
        let id = store
            .accept_with_archive(kind, &value, true, prepared.readouts.archive_bytes)?
            .status
            .id;
        let writer = writer::Writer::spawn(
            store.clone(),
            id.clone(),
            store.control(&id)?,
            &prepared,
            Default::default(),
        )?;
        ensure!(
            matches!(writer.wait_ready()?, writer::Readiness::Execute),
            "settled before oracle"
        );
        let outcome = backend.generate_native(&prepared, writer.sink());
        writer.finish(outcome)?;
        ids.push(id);
        authored.push(value.clone());
    }
    drop(backend);
    drop(runtime);
    let residency_ms = started.elapsed().as_millis();
    println!("intervention oracle GPU released after {residency_ms}ms");
    let mut pages = Vec::new();
    for id in &ids {
        let status = store.status(id)?;
        ensure!(
            status.state == crate::serve::jobs::state::JobState::Completed
                && status.result.complete
                && status.result.error.is_none(),
            "failed oracle: {status:?}"
        );
        ensure!(
            status.generation.counters.consumed_generated_tokens > 0,
            "oracle requires decode"
        );
        pages.push(store.result(id, None, 128)?.records);
    }
    if paired {
        return super::paired_live::verify(
            &store,
            &ids,
            &pages,
            &authored,
            layer,
            last,
            profile.hidden,
            &output,
            residency_ms,
        );
    }
    let samples = |records: &[Value]| {
        records
            .iter()
            .filter(|r| r["kind"] == "sampled_token")
            .cloned()
            .map(|mut r| {
                r.as_object_mut().unwrap().remove("seq");
                r
            })
            .collect::<Vec<_>>()
    };
    ensure!(
        samples(&pages[0]) == samples(&pages[1]),
        "zero control changed samples"
    );
    ensure!(
        samples(&pages[2]) == samples(&pages[3]),
        "observation changed intervention samples"
    );
    let read = |index: usize| -> Result<&Value> {
        let rows = pages[index]
            .iter()
            .filter(|r| r["kind"] == "readout")
            .collect::<Vec<_>>();
        ensure!(rows.len() == 1, "one readout required");
        Ok(rows[0])
    };
    let before = vector(&read(0)?["test_source_values"])?;
    ensure!(before.len() == profile.hidden, "baseline source shape");
    for index in [0, 1] {
        let row = read(index)?;
        ensure!(
            row["phase"] == "prefill"
                && row["index"] == 0
                && row["position"] == 0
                && row["predicts_position"] == 1
                && row["source_layer"] == layer
                && row["input_token_id"] == pages[index][0]["token_ids"][0]
                && row["applied_operation_ids"] == json!([]),
            "baseline/zero capture coordinates"
        );
    }
    ensure!(
        read(0)?["scores"] == read(1)?["scores"]
            && read(0)?["test_source_values"] == read(1)?["test_source_values"],
        "zero control changed readout"
    );
    for records in &pages[..2] {
        ensure!(
            !records
                .iter()
                .any(|r| r["kind"] == "operation_application" || r["kind"] == "direction_prepared"),
            "baseline/zero fabricated work"
        );
    }
    let mut projection_witnesses = 0;
    let mut errors = Vec::new();
    let mut references = Vec::new();
    let mut common_directions = None;
    for index in 2..5 {
        let records = &pages[index];
        let directions = records
            .iter()
            .filter(|r| r["kind"] == "direction_prepared")
            .collect::<Vec<_>>();
        ensure!(
            directions.len() == 2,
            "two normalized direction rows required"
        );
        let mut values = std::collections::BTreeMap::new();
        for row in directions {
            ensure!(
                row["source_layer"] == layer
                    && row["target_covector"] == "deployed_logit_numerator",
                "direction identity"
            );
            let witness = &row["test_projection_witness"];
            ensure!(
                witness["basis"] == "cpu_f64_f16_matrix_transpose_times_deployed_covector"
                    && witness["within_tolerance"] == true
                    && witness["bypass_tolerance_ratio"]
                        .as_f64()
                        .is_some_and(|v| v.is_finite() && v > 10.),
                "projection did not prove nonidentity"
            );
            ensure!(
                row["test_normalization_witness"]["within_tolerance"] == true,
                "normalized projection mismatch"
            );
            values.insert(
                row["direction_id"].as_str().context("direction id")?,
                vector(&row["test_direction_values"])?,
            );
            projection_witnesses += 1;
        }
        if let Some(common) = &common_directions {
            ensure!(
                &values == common,
                "prepared directions changed across orders"
            );
        } else {
            common_directions = Some(values.clone());
        }
        let operations = authored[index]["diagnostics"]["operations"]
            .as_array()
            .unwrap();
        let apps = records
            .iter()
            .filter(|r| r["kind"] == "operation_application")
            .collect::<Vec<_>>();
        ensure!(apps.len() == 3, "three applications required");
        for (order, (app, op)) in apps.iter().zip(operations).enumerate() {
            ensure!(
                app["id"] == op["id"]
                    && app["action"] == op["action"]
                    && app["order"] == order
                    && app["position"] == 0
                    && app["layer"] == layer,
                "application order/site mismatch"
            );
        }
        if index == 3 {
            ensure!(
                !records.iter().any(|r| r["kind"] == "readout"),
                "operation-only produced head"
            );
            continue;
        }
        let actual = read(index)?;
        ensure!(
            actual["position"] == 0
                && actual["source_layer"] == layer
                && actual["applied_operation_ids"]
                    == json!(operations.iter().map(|o| &o["id"]).collect::<Vec<_>>()),
            "after-state provenance"
        );
        let expected = transform(before.clone(), &values, operations)?;
        let after = vector(&actual["test_source_values"])?;
        ensure!(after.len() == expected.len(), "after shape");
        let mut max_abs = 0f64;
        for (want, got) in expected.iter().zip(&after) {
            let delta = (want - got).abs();
            ensure!(
                delta <= tolerance(*want),
                "sequential CPU transform mismatch {want} vs {got}"
            );
            max_abs = max_abs.max(delta);
        }
        errors.push(max_abs);
        references.push(expected);
    }
    let active = vector(&read(2)?["test_source_values"])?;
    let reordered = vector(&read(4)?["test_source_values"])?;
    distinguish_orders(&references[0], &references[1], &active, &reordered)?;
    ensure!(
        projection_witnesses == 6 && errors.len() == 2,
        "oracle incomplete"
    );
    std::fs::write(
        output.join("witnesses.json"),
        serde_json::to_vec_pretty(
            &json!({"zero_control_equal":true,"operation_only_samples_equal":true,
        "projection_witness_records":projection_witnesses,"transformation_sites_checked":2,"ordered_transform_max_abs_errors":errors,
        "noncommuting_order_distinguished":true,"jobs":ids,"residency_ms":residency_ms}),
        )?,
    )?;
    Ok(())
}
