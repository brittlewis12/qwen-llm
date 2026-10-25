use super::*;
use crate::deepseek_v41::execution::UnsupportedExecution;

#[test]
fn activation_aliases_each_and_equal_pair_preserve_silu_default() {
    let (kv, tensors) = fixture_parts(32, GgmlType::F32, 2, false);
    for keys in [
        vec![],
        vec!["deepseek41.hidden_act"],
        vec!["deepseek41.hidden_activation"],
        vec!["deepseek41.hidden_act", "deepseek41.hidden_activation"],
    ] {
        let mut kv = kv.clone();
        kv.extend(keys.into_iter().map(|key| (key, Value::Str("silu"))));
        plan(&write_fixture(kv, tensors.clone()))
            .unwrap()
            .validate_execution()
            .unwrap();
    }
}

#[test]
fn activation_aliases_conflict_in_either_order() {
    for values in [["silu", "gelu"], ["gelu", "silu"]] {
        let (mut kv, tensors) = fixture_parts(32, GgmlType::F32, 2, false);
        kv.extend([
            ("deepseek41.hidden_act", Value::Str(values[0])),
            ("deepseek41.hidden_activation", Value::Str(values[1])),
        ]);
        assert!(
            plan(&write_fixture(kv, tensors))
                .unwrap_err()
                .to_string()
                .contains("conflicting aliases")
        );
    }
}

#[test]
fn activation_aliases_reject_unsupported_single_and_equal_pair() {
    for keys in [
        vec!["deepseek41.hidden_act"],
        vec!["deepseek41.hidden_activation"],
        vec!["deepseek41.hidden_act", "deepseek41.hidden_activation"],
    ] {
        let (mut kv, tensors) = fixture_parts(32, GgmlType::F32, 2, false);
        kv.extend(keys.into_iter().map(|key| (key, Value::Str("gelu"))));
        assert!(
            plan(&write_fixture(kv, tensors))
                .unwrap_err()
                .to_string()
                .contains("unsupported deepseek41.hidden_")
        );
    }
}

#[test]
fn activation_aliases_wrong_types_never_fall_back_to_missing_default() {
    for key in ["deepseek41.hidden_act", "deepseek41.hidden_activation"] {
        for value in [
            Value::U32(0),
            Value::U32(1),
            Value::I32(-1),
            Value::I32(1),
            Value::Bool(true),
            Value::F32(1.0),
        ] {
            for paired in [false, true] {
                let (mut kv, tensors) = fixture_parts(32, GgmlType::F32, 2, false);
                if paired {
                    kv.push((
                        if key.ends_with("hidden_act") {
                            "deepseek41.hidden_activation"
                        } else {
                            "deepseek41.hidden_act"
                        },
                        Value::Str("silu"),
                    ));
                }
                kv.push((key, value.clone()));
                let error = plan(&write_fixture(kv, tensors)).unwrap_err().to_string();
                assert!(
                    error.contains(key) && error.contains("must be string"),
                    "{error}"
                );
            }
        }
    }
}

#[test]
fn positive_sinkhorn_counts_are_valid_plans_but_only_twenty_is_executable() {
    for iterations in [1, 3, 19, 20, 21] {
        let (mut kv, tensors) = fixture_parts(32, GgmlType::F32, 2, false);
        kv.retain(|(key, _)| *key != "deepseek41.hc_sinkhorn_iters");
        kv.push((
            "deepseek41.hyper_connection.sinkhorn_iterations",
            Value::U32(iterations),
        ));
        let p = plan(&write_fixture(kv, tensors)).unwrap();
        assert_eq!(p.config.hc_iterations, iterations as usize);
        assert_eq!(
            p.validate_execution(),
            if iterations == 20 {
                Ok(())
            } else {
                Err(UnsupportedExecution::SinkhornIterations(
                    iterations as usize,
                ))
            }
        );
    }
}

#[test]
fn routing_topk_above_native_limit_is_not_malformed_metadata() {
    for topk in [1, 6, 16, 17, 32] {
        let (mut kv, tensors) = fixture_parts(32, GgmlType::F32, 2, false);
        kv.retain(|(key, _)| *key != "deepseek41.expert_used_count");
        kv.push(("deepseek41.expert_used_count", Value::U32(topk)));
        let p = plan(&write_fixture(kv, tensors)).unwrap();
        assert_eq!(
            p.validate_execution(),
            if topk <= 16 {
                Ok(())
            } else {
                Err(UnsupportedExecution::RoutingTopK(topk as usize))
            }
        );
    }
}

#[test]
fn zero_sinkhorn_and_topk_outside_expert_count_remain_invalid_plans() {
    for (key, value) in [
        ("deepseek41.hc_sinkhorn_iters", 0),
        ("deepseek41.expert_used_count", 0),
        ("deepseek41.expert_used_count", 33),
    ] {
        let (mut kv, tensors) = fixture_parts(32, GgmlType::F32, 2, false);
        kv.retain(|(old, _)| *old != key);
        kv.push((key, Value::U32(value)));
        assert!(matches!(
            plan(&write_fixture(kv, tensors)),
            Err(LoadError::Invalid(_))
        ));
    }
}
