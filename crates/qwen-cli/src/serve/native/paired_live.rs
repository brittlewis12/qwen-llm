//! Actual-before numerical witnesses for later whole-site intervention programs.

use super::intervention_live::{distinguish_orders, transform, vector};
use super::*;
use qwen_llm::{
    metal::{MetalTensor, PostBlockIntervention},
    runtime::{LoadedModel, SequenceConfig},
};

fn verify_sites(records: &[Value], layer: u32, last: u32) -> Result<()> {
    let prepared = records
        .iter()
        .find(|r| r["kind"] == "prepared_input")
        .context("prepared tokens")?;
    let tokens = prepared["token_ids"].as_array().context("prompt tokens")?;
    ensure!(tokens.len() > 1, "later prefill site required");
    let sample = records
        .iter()
        .find(|r| r["kind"] == "sampled_token" && r["index"] == 0)
        .context("decode input sample")?;
    ensure!(sample["consumed"] == true, "decode input not consumed");
    let prompt = u64::try_from(tokens.len())?;
    let mut expected = std::collections::BTreeSet::new();
    for position in [1, prompt] {
        for layer in [layer, last] {
            expected.insert((position, u64::from(layer)));
        }
    }
    ensure!(expected.len() == 4, "distinct capture layers required");
    for pair in records.iter().filter(|r| r["kind"] == "residual_pair") {
        let position = pair["position"].as_u64().context("pair position")?;
        let layer = pair["source_layer"].as_u64().context("pair layer")?;
        ensure!(
            pair["id"] == "whole" && expected.remove(&(position, layer)),
            "duplicate/unexpected pair site"
        );
        let (phase, index, token) = if position < prompt {
            ("prefill", position, &tokens[position as usize])
        } else {
            ("decode", position - prompt, &sample["token_id"])
        };
        ensure!(
            pair["phase"] == phase && pair["index"] == index && pair["input_token_id"] == *token,
            "pair phase/position/input mismatch"
        );
    }
    ensure!(expected.is_empty(), "missing pair sites");
    Ok(())
}

#[test]
fn site_oracle_rejects_duplicates_and_consistent_coordinate_mislabeling() {
    let mut records = vec![
        json!({"kind":"prepared_input","token_ids":[10,11,12]}),
        json!({"kind":"sampled_token","index":0,"token_id":20,"consumed":true}),
    ];
    for (position, phase, index, token) in [(1, "prefill", 1, 11), (3, "decode", 0, 20)] {
        for layer in [1, 3] {
            records.push(json!({"kind":"residual_pair","id":"whole","position":position,"phase":phase,"index":index,"input_token_id":token,"source_layer":layer}));
        }
    }
    verify_sites(&records, 1, 3).unwrap();
    let mut duplicate = records.clone();
    duplicate[3] = duplicate[2].clone();
    assert!(verify_sites(&duplicate, 1, 3).is_err());
    for (field, value) in [
        ("phase", json!("decode")),
        ("position", json!(2)),
        ("index", json!(0)),
        ("input_token_id", json!(99)),
        ("id", json!("different")),
    ] {
        let mut wrong = records.clone();
        wrong[2][field] = value;
        assert!(verify_sites(&wrong, 1, 3).is_err(), "{field}");
    }
    records.pop();
    assert!(verify_sites(&records, 1, 3).is_err());
}

pub(super) fn validate_destinations(loaded: &LoadedModel) -> Result<()> {
    let mut sequence = loaded.create_sequence(SequenceConfig::new(1))?;
    let session = unsafe { sequence.metal_session_mut() };
    let h = loaded.arch().hidden_size as u64;
    let arena = MetalTensor::zeros_f32(loaded.context(), vec![3 * h])?;
    let before = arena.view_subrange(0, vec![h]);
    let after = arena.view_subrange(h, vec![h]);
    let direction = arena.view_subrange(2 * h, vec![h]);
    let forward = loaded.forward();
    let mut rejected = |b: &MetalTensor,
                        a: &MetalTensor,
                        layers: &[u32],
                        ops: &[PostBlockIntervention<'_>]|
     -> Result<()> {
        ensure!(
            forward
                .single_token_with_post_block_measurements(
                    1,
                    0,
                    session,
                    (layers, b),
                    (&[0], a),
                    ops,
                    false
                )
                .is_err(),
            "invalid capture accepted: before {:?} {:?} offset={}, after offset={}, layers={layers:?}, operations={}",
            b.dtype,
            b.shape,
            b.offset,
            a.offset,
            ops.len()
        );
        Ok(())
    };
    rejected(&before, &before, &[0], &[])?;
    rejected(&arena.view_subrange(1, vec![h]), &after, &[0], &[])?;
    rejected(&before, &after, &[loaded.arch().n_layer], &[])?;
    rejected(&arena.view_subrange(0, vec![h - 1]), &after, &[0], &[])?;
    let mut misaligned = before.clone();
    misaligned.offset = 1;
    rejected(&misaligned, &after, &[0], &[])?;
    let mut overflow = before.clone();
    overflow.offset = u64::MAX - 3;
    rejected(&overflow, &after, &[0], &[])?;
    let mut outside = before.clone();
    outside.offset = 3 * h * 4;
    rejected(&outside, &after, &[0], &[])?;
    let half = MetalTensor::zeros_f16(loaded.context(), vec![h])?;
    rejected(&half, &after, &[0], &[])?;
    for tensor in [&before, &after] {
        rejected(
            &before,
            &after,
            &[0],
            &[PostBlockIntervention::Fixed {
                layer: 0,
                direction: tensor,
                coefficient: 1.,
            }],
        )?;
        rejected(
            &before,
            &after,
            &[0],
            &[PostBlockIntervention::SourceToTarget {
                layer: 0,
                source: tensor,
                target: &direction,
                coefficient: 1.,
            }],
        )?;
        rejected(
            &before,
            &after,
            &[0],
            &[PostBlockIntervention::SourceToTarget {
                layer: 0,
                source: &direction,
                target: tensor,
                coefficient: 1.,
            }],
        )?;
    }
    let mutable_alias = session.x.clone();
    ensure!(
        forward
            .single_token_with_post_block_measurements(
                1,
                0,
                session,
                (&[0], &mutable_alias),
                (&[0], &after),
                &[],
                false
            )
            .is_err(),
        "session alias accepted"
    );
    // Disjoint views, including a direction in the same allocation, are valid.
    let logits = forward.single_token_with_post_block_measurements(
        1,
        0,
        session,
        (&[0], &before),
        (&[0], &after),
        &[PostBlockIntervention::Fixed {
            layer: 0,
            direction: &direction,
            coefficient: 1.,
        }],
        false,
    )?;
    ensure!(logits.is_empty(), "no-tail validation ran a head");
    Ok(())
}

pub(super) fn verify(
    store: &JobStore,
    ids: &[String],
    pages: &[Vec<Value>],
    authored: &[Value],
    layer: u32,
    last: u32,
    hidden: usize,
    output: &std::path::Path,
    residency_ms: u128,
) -> Result<()> {
    let samples = |records: &[Value]| {
        records
            .iter()
            .filter(|r| r["kind"] == "sampled_token")
            .map(|r| {
                let mut r = r.clone();
                r.as_object_mut().unwrap().remove("seq");
                r
            })
            .collect::<Vec<_>>()
    };
    ensure!(
        samples(&pages[0]) == samples(&pages[1]),
        "zero changed samples"
    );
    ensure!(
        samples(&pages[2]) == samples(&pages[3]),
        "pairs changed intervention samples"
    );
    ensure!(
        samples(&pages[0]) == samples(&pages[5]),
        "pair-only changed samples"
    );
    let counters = |i: usize| -> Result<_> { Ok(store.status(&ids[i])?.generation.counters) };
    ensure!(
        serde_json::to_value(counters(2)?)? == serde_json::to_value(counters(3)?)?,
        "paired consumption changed"
    );
    let mut checked = 0;
    let mut errors = Vec::new();
    let mut control_arrays = Vec::new();
    for (index, records) in pages.iter().enumerate() {
        if index != 3 {
            verify_sites(records, layer, last)?;
        }
        let pairs = records
            .iter()
            .filter(|r| r["kind"] == "residual_pair")
            .collect::<Vec<_>>();
        ensure!(
            pairs.len() == if index == 3 { 0 } else { 4 },
            "pair site count"
        );
        if index == 3 || index == 5 {
            ensure!(
                !records.iter().any(|r| r["kind"] == "readout"),
                "pair-only/operation-only produced heads"
            );
        }
        if [0, 1, 5].contains(&index) {
            ensure!(
                !records
                    .iter()
                    .any(|r| r["kind"] == "operation_application"
                        || r["kind"] == "direction_prepared"),
                "zero/no-op staged directions"
            );
        }
        let directions = records
            .iter()
            .filter(|r| r["kind"] == "direction_prepared")
            .map(|r| {
                Ok((
                    r["direction_id"].as_str().context("direction id")?,
                    vector(&r["test_direction_values"])?,
                ))
            })
            .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
        let mut arrays = Vec::new();
        for pair in pairs {
            ensure!(
                pair["capture_stage"] == "whole_post_block_program"
                    && pair["provenance"] == "original_forward",
                "pair provenance"
            );
            let get = |field: &str, quantity: &str| -> Result<Vec<f32>> {
                let record = records
                    .iter()
                    .find(|r| r["kind"] == "retained_array" && r["key"] == pair[field])
                    .context("missing paired array")?;
                ensure!(
                    record["quantity"] == quantity && record["seq"].as_u64() < pair["seq"].as_u64(),
                    "array dependency/order"
                );
                for field in [
                    "phase",
                    "index",
                    "position",
                    "source_layer",
                    "input_token_id",
                ] {
                    ensure!(record[field] == pair[field], "array coordinate {field}");
                }
                let offset = record["array"]["url"]
                    .as_str()
                    .context("array url")?
                    .rsplit('/')
                    .next()
                    .unwrap()
                    .parse()?;
                let bytes = store.array(&ids[index], offset)?;
                let values = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
                ensure!(
                    values.len() == hidden && values.iter().all(|v| v.is_finite()),
                    "pair shape/finite"
                );
                Ok(values)
            };
            let before = get("before_key", "source_residual_before")?;
            let after = get("after_key", "source_residual")?;
            // Independent metric calculation from downloaded F32 arrays.
            let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            let delta = before
                .iter()
                .zip(&after)
                .map(|(&a, &b)| (f64::from(b) - f64::from(a)).powi(2))
                .sum::<f64>()
                .sqrt();
            for (key, want) in [
                ("norm_before", norm(&before)),
                ("norm_after", norm(&after)),
                ("delta_norm", delta),
                ("relative_delta", delta / norm(&before)),
            ] {
                let got = pair["metrics"][key].as_f64().context("numeric metric")?;
                ensure!(
                    (got - want).abs() <= 1e-12 * (1. + want.abs()),
                    "metric mismatch"
                );
            }
            checked += 1;
            if [2, 4].contains(&index) && pair["source_layer"] == layer {
                ensure!(
                    (pair["phase"] == "prefill" && pair["index"] == 1)
                        || (pair["phase"] == "decode" && pair["index"] == 0),
                    "later original site required"
                );
                let ops = authored[index]["diagnostics"]["operations"]
                    .as_array()
                    .unwrap();
                ensure!(
                    pair["applied_operation_ids"]
                        == json!(ops.iter().map(|o| &o["id"]).collect::<Vec<_>>()),
                    "paired authored order"
                );
                let source = before.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
                let expected = transform(source.clone(), &directions, ops)?;
                let mut opposite = ops.clone();
                opposite.swap(0, 1);
                let opposite = transform(source, &directions, &opposite)?;
                let actual = after.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
                distinguish_orders(&expected, &opposite, &actual, &opposite)?;
                errors.push(qwen_llm::compare::assert_max_abs_diff_f64(
                    concat!(file!(), ":", line!()),
                    expected.iter(),
                    &actual,
                ));
            } else {
                ensure!(before == after && delta == 0., "no-op site changed");
                ensure!(
                    pair["applied_operation_ids"] == json!([]),
                    "no-op fabricated applications"
                );
            }
            arrays.push((before, after));
        }
        if [0, 1, 5].contains(&index) {
            control_arrays.push(arrays);
        }
    }
    ensure!(
        control_arrays.windows(2).all(|p| p[0] == p[1]),
        "zero/no-op controls changed original arrays"
    );
    ensure!(errors.len() == 4 && checked == 20, "incomplete pair oracle");
    std::fs::write(
        output.join("witnesses.json"),
        serde_json::to_vec_pretty(&json!({
            "zero_control_equal":true,"operation_only_samples_equal":true,"pair_only_samples_equal":true,
            "transformation_sites_checked":errors.len(),"metrics_checked":checked,"ordered_transform_max_abs_errors":errors,
            "noncommuting_order_distinguished":true,"destination_validation":true,"jobs":ids,"residency_ms":residency_ms,
        }))?,
    )?;
    Ok(())
}
