use super::*;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Decision {
    Exact,
    Indeterminate,
    Mismatch,
    Invalid,
}

pub(super) fn witness(actual: &[f32], reference: &[f32]) -> serde_json::Value {
    assert_eq!(actual.len(), reference.len());
    let a = super::runner::argmax(actual) as usize;
    let r = super::runner::argmax(reference) as usize;
    let (a_max, a_at_r) = (f64::from(actual[a]), f64::from(actual[r]));
    let (r_max, r_at_a) = (f64::from(reference[r]), f64::from(reference[a]));
    json!({"vocabulary_size":actual.len(),"actual_top1":a,"reference_top1":r,
        "native_max_logit":a_max,"native_at_reference_choice":a_at_r,
        "reference_max_logit":r_max,"reference_at_native_choice":r_at_a,
        "native_regret_of_reference_choice":a_max-a_at_r,
        "reference_regret_of_native_choice":r_max-r_at_a})
}

pub(super) fn classify(metrics: &serde_json::Value, exact_required: bool, limit: f64) -> Decision {
    let w = &metrics["ranking"];
    let Some(width) = w["vocabulary_size"]
        .as_u64()
        .filter(|&n| n > 0 && n <= 250624)
    else {
        return Decision::Invalid;
    };
    let Some(a) = w["actual_top1"].as_u64().filter(|&id| id < width) else {
        return Decision::Invalid;
    };
    let Some(r) = w["reference_top1"].as_u64().filter(|&id| id < width) else {
        return Decision::Invalid;
    };
    if metrics["logits"]["actual_top1"].as_u64() != Some(a)
        || metrics["logits"]["reference_top1"].as_u64() != Some(r)
        || !limit.is_finite()
        || limit <= 0.
    {
        return Decision::Invalid;
    }
    let values = [
        "native_max_logit",
        "native_at_reference_choice",
        "reference_max_logit",
        "reference_at_native_choice",
        "native_regret_of_reference_choice",
        "reference_regret_of_native_choice",
    ]
    .map(|key| w[key].as_f64().filter(|v| v.is_finite()));
    let [Some(am), Some(ar), Some(rm), Some(ra), Some(ad), Some(rd)] = values else {
        return Decision::Invalid;
    };
    if ad < 0. || rd < 0. || am - ar != ad || rm - ra != rd {
        return Decision::Invalid;
    }
    if a == r {
        return if ad == 0. && rd == 0. {
            Decision::Exact
        } else {
            Decision::Invalid
        };
    }
    // Equal candidate logits must resolve to the higher ID in each distribution.
    if (ad == 0. && a < r) || (rd == 0. && r < a) {
        return Decision::Invalid;
    }
    if exact_required || ad >= limit || rd >= limit {
        Decision::Mismatch
    } else {
        Decision::Indeterminate
    }
}

#[test]
fn reciprocal_regrets_are_live_vector_evidence_and_exact_rows_stay_exact() {
    let a = [1.0, 1.0005];
    let r = [1.0005, 1.0];
    let w = witness(&a, &r);
    let m = json!({"logits":{"actual_top1":1,"reference_top1":0},"ranking":w});
    assert_eq!(classify(&m, false, 0.001), Decision::Indeterminate);
    assert_eq!(classify(&m, true, 0.001), Decision::Mismatch);
    let tied = witness(&[0., -0.], &[-0., 0.]);
    let tied = json!({"logits":{"actual_top1":1,"reference_top1":1},"ranking":tied});
    assert_eq!(classify(&tied, true, 0.001), Decision::Exact);
    for values in [vec![], vec![f32::NAN], vec![f32::INFINITY]] {
        assert!(std::panic::catch_unwind(|| witness(&values, &values)).is_err());
    }
}

#[test]
fn reciprocal_regret_boundary_and_corrupt_witnesses_fail_closed() {
    for (gap, expected) in [
        (
            f32::from_bits(0.001_f32.to_bits() - 1),
            Decision::Indeterminate,
        ),
        (0.001_f32, Decision::Mismatch),
    ] {
        let w = witness(&[0., gap], &[gap, 0.]);
        let m = json!({"logits":{"actual_top1":1,"reference_top1":0},"ranking":w});
        assert_eq!(classify(&m, false, 0.001), expected);
    }
    let good = json!({"logits":{"actual_top1":1,"reference_top1":0},"ranking":{
        "vocabulary_size":2,"actual_top1":1,"reference_top1":0,
        "native_max_logit":0.0005,"native_at_reference_choice":0.,
        "reference_max_logit":0.0005,"reference_at_native_choice":0.,
        "native_regret_of_reference_choice":0.0005,"reference_regret_of_native_choice":0.0005}});
    assert_eq!(classify(&good, false, 0.001), Decision::Indeterminate);
    for (maximum, regret) in [
        ("native_max_logit", "native_regret_of_reference_choice"),
        ("reference_max_logit", "reference_regret_of_native_choice"),
    ] {
        let mut m = good.clone();
        m["ranking"][maximum] = json!(0.001);
        m["ranking"][regret] = json!(0.001);
        assert_eq!(classify(&m, false, 0.001), Decision::Mismatch);
    }
    for (key, value) in [
        ("vocabulary_size", json!(0)),
        ("actual_top1", json!(2)),
        ("reference_top1", json!(1)),
        ("native_max_logit", serde_json::Value::Null),
        ("reference_at_native_choice", json!(1.)),
        ("native_regret_of_reference_choice", json!(-0.1)),
        ("reference_regret_of_native_choice", json!(0.0004)),
    ] {
        let mut m = good.clone();
        m["ranking"][key] = value;
        assert_eq!(classify(&m, false, 0.001), Decision::Invalid, "{key}");
    }
    let mut wrong_id = good.clone();
    wrong_id["logits"]["actual_top1"] = json!(0);
    assert_eq!(classify(&wrong_id, false, 0.001), Decision::Invalid);
    let mut wrong_tie = good;
    wrong_tie["ranking"]["reference_max_logit"] = json!(0.);
    wrong_tie["ranking"]["reference_regret_of_native_choice"] = json!(0.);
    assert_eq!(classify(&wrong_tie, false, 0.001), Decision::Invalid);
}
