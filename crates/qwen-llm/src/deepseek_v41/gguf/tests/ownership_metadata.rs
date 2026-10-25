use super::*;

const KEYS: [(&str, &str); 3] = [
    (
        "deepseek41.kv_source_layer_ids",
        "deepseek41.attention.kv_source_layers",
    ),
    (
        "deepseek41.index_source_layer_ids",
        "deepseek41.attention.indexer.source_layers",
    ),
    (
        "deepseek41.candidate_source_layer_id",
        "deepseek41.attention.indexer.candidate_source_layer",
    ),
];

type Metadata = Vec<(&'static str, Value<'static>)>;
type Tensors = Vec<(String, Vec<u64>, GgmlType)>;

fn fixture_for(schedule: &Schedule) -> (Metadata, Tensors) {
    let config = metadata::parse(&fixture(17, GgmlType::F32, 2, false))
        .unwrap()
        .config;
    let (mut kv, _) = fixture_parts(17, GgmlType::F32, 2, false);
    kv.retain(|(key, _)| !KEYS.iter().any(|(short, long)| key == short || key == long));
    for (key, value) in &mut kv {
        match *key {
            "deepseek41.block_count" => *value = Value::U32(schedule.layers as u32),
            "deepseek41.compress_ratios" => {
                *value = Value::U64s(
                    schedule
                        .compression_ratios
                        .iter()
                        .map(|&n| n as u64)
                        .collect(),
                );
            }
            _ => (),
        }
    }
    let mut tensors: Tensors = names::required_dense(schedule, &[])
        .into_iter()
        .map(|name| {
            let shape = names::expected_shape(&name, &config).unwrap();
            (name, shape, GgmlType::F32)
        })
        .collect();
    for layer in 0..schedule.layers {
        for part in ["gate", "up", "down"] {
            tensors.push((
                format!("blk.{layer}.ffn_{part}_exps.weight"),
                vec![2, 2, 17],
                GgmlType::F32,
            ));
        }
    }
    (kv, tensors)
}

fn values(schedule: &Schedule) -> [Value<'static>; 3] {
    [
        Value::U64s(
            schedule
                .kv_source_layers
                .iter()
                .map(|&n| n as u64)
                .collect(),
        ),
        Value::U64s(
            schedule
                .index_source_layers
                .iter()
                .map(|&n| n as u64)
                .collect(),
        ),
        Value::U32(schedule.candidate_source_layer.unwrap() as u32),
    ]
}

fn declare(kv: &mut Metadata, schedule: &Schedule, spelling: usize) {
    for ((short, long), value) in KEYS.into_iter().zip(values(schedule)) {
        if spelling != 1 {
            kv.push((short, value.clone()));
        }
        if spelling != 0 {
            kv.push((long, value));
        }
    }
}

fn explicit_layout() -> Schedule {
    Schedule {
        kv_source_layers: vec![2, 9, 14, 20, 24],
        index_source_layers: vec![2, 9, 14, 20, 24, 28, 32, 36],
        candidate_source_layer: Some(24),
        ..Schedule::default()
    }
}

fn assert_invalid(error: LoadError, message: &str) {
    assert!(
        matches!(&error, LoadError::Invalid(text) if text.contains(message)),
        "expected Invalid containing {message:?}, got {error:?}"
    );
}

#[test]
fn long_short_and_consistent_aliases_override_fallback() {
    let schedule = explicit_layout();
    let (kv, tensors) = fixture_for(&schedule);
    for spelling in 0..3 {
        let mut declared = kv.clone();
        declare(&mut declared, &schedule, spelling);
        let p = plan(&write_fixture(declared, tensors.clone())).unwrap();
        assert_eq!(p.schedule, schedule);
        assert_eq!(p.kv_source_layers, vec![2, 9, 14, 20, 24]);
        assert_eq!(p.index_source_layers, vec![2, 9, 14, 20, 24, 28, 32, 36]);
    }
}

#[test]
fn gsq_attention_candidate_declarations_override_defaults_and_check_conflicts() {
    let schedule = explicit_layout();
    let (mut kv, tensors) = fixture_for(&schedule);
    declare(&mut kv, &schedule, 0);
    kv.retain(|(key, _)| *key != "deepseek41.candidate_source_layer_id");
    kv.extend([
        (
            "deepseek41.attention.candidate_source_layer_id",
            Value::U32(24),
        ),
        ("deepseek41.attention.candidate_block_size", Value::U32(16)),
        ("deepseek41.attention.candidate_topk_blocks", Value::U32(17)),
        ("deepseek41.expert_scoring_func", Value::Str("sqrtsoftplus")),
    ]);
    let p = plan(&write_fixture(kv.clone(), tensors.clone())).unwrap();
    assert_eq!(p.schedule, schedule);
    assert_eq!(p.config.candidate_block, 16);
    assert_eq!(p.config.candidate_topk, 17);
    for (key, value) in [
        ("deepseek41.candidate_source_layer_id", Value::U32(20)),
        ("deepseek41.candidate_block_size", Value::U32(8)),
        ("deepseek41.candidate_topk_blocks", Value::U32(2048)),
        ("deepseek41.scoring_func", Value::Str("softmax")),
    ] {
        let mut bad = kv.clone();
        bad.push((key, value));
        assert!(plan(&write_fixture(bad, tensors.clone())).is_err(), "{key}");
    }
    for key in [
        "deepseek41.attention.candidate_block_size",
        "deepseek41.attention.candidate_topk_blocks",
        "deepseek41.attention.candidate_source_layer_id",
        "deepseek41.expert_scoring_func",
    ] {
        let mut bad = kv.clone();
        bad.iter_mut().find(|(k, _)| *k == key).unwrap().1 = Value::F32(f32::NAN);
        assert!(plan(&write_fixture(bad, tensors.clone())).is_err(), "{key}");
    }
}

#[test]
fn explicit_nonstandard_schedule_keeps_existing_structural_support() {
    let schedule = Schedule {
        layers: 6,
        kv_source_layers: vec![1, 4],
        index_source_layers: vec![1, 3, 4, 5],
        compression_ratios: vec![0, 2, 2, 2, 1, 1],
        candidate_source_layer: Some(4),
    };
    let (kv, tensors) = fixture_for(&schedule);
    for spelling in 0..3 {
        let mut declared = kv.clone();
        declare(&mut declared, &schedule, spelling);
        assert_eq!(
            plan(&write_fixture(declared, tensors.clone()))
                .unwrap()
                .schedule,
            schedule
        );
    }
}

#[test]
fn each_conflicting_alias_is_rejected_in_either_value_order() {
    let standard = Schedule::default();
    let explicit = explicit_layout();
    let (kv, tensors) = fixture_for(&standard);
    for (i, (short, long)) in KEYS.into_iter().enumerate() {
        for reverse in [false, true] {
            let mut declared = kv.clone();
            declare(&mut declared, &standard, 0);
            if reverse {
                declared.iter_mut().find(|(k, _)| *k == short).unwrap().1 =
                    values(&explicit)[i].clone();
                declared.push((long, values(&standard)[i].clone()));
            } else {
                declared.push((long, values(&explicit)[i].clone()));
            }
            assert_invalid(
                plan(&write_fixture(declared, tensors.clone())).unwrap_err(),
                "conflicting",
            );
        }
    }
}

#[test]
fn explicit_invalid_layout_never_silently_uses_fallback() {
    let standard = Schedule::default();
    let (kv, tensors) = fixture_for(&standard);
    for (i, bad, message) in [
        (
            0,
            Value::U64s(vec![]),
            "compressed attention has no KV owner",
        ),
        (
            0,
            Value::U64s(vec![2, 8, 14]),
            "reuse layer compression differs",
        ),
        (0, Value::U64s(vec![2, 8, 14, 40]), "invalid layer schedule"),
        (1, Value::U64s(vec![]), "KV owner must produce index"),
        (
            2,
            Value::U32(14),
            "candidate source must own uncompressed KV",
        ),
        (
            2,
            Value::U32(40),
            "candidate source must own uncompressed KV",
        ),
    ] {
        for spelling in 0..3 {
            let mut declared = kv.clone();
            declare(&mut declared, &standard, spelling);
            for (key, value) in &mut declared {
                if *key == KEYS[i].0 || *key == KEYS[i].1 {
                    *value = bad.clone();
                }
            }
            assert_invalid(
                plan(&write_fixture(declared, tensors.clone())).unwrap_err(),
                message,
            );
        }
    }
}

#[test]
fn explicit_layout_must_bind_its_own_tensors_not_fallback_inventory() {
    let (kv, tensors) = fixture_for(&Schedule::default());
    for spelling in 0..3 {
        let mut declared = kv.clone();
        declare(&mut declared, &explicit_layout(), spelling);
        let g = write_fixture(declared, tensors.clone());
        assert_eq!(metadata::parse(&g).unwrap().schedule, explicit_layout());
        assert_invalid(
            plan(&g).unwrap_err(),
            "missing dense tensor blk.9.attn_compressor_kv.weight",
        );
    }
}

#[test]
fn malformed_long_alias_cannot_be_treated_as_missing() {
    let (kv, tensors) = fixture_for(&Schedule::default());
    for (i, (_, long)) in KEYS.into_iter().enumerate() {
        let mut declared = kv.clone();
        declared.push((long, Value::Str("invalid")));
        let error = plan(&write_fixture(declared, tensors.clone())).unwrap_err();
        if i == 2 {
            assert_invalid(error, "must be unsigned integer");
        } else {
            assert!(matches!(error, LoadError::Gguf(_)), "{error:?}");
        }
    }
}

#[test]
fn missing_ownership_falls_back_only_for_matching_standard_operations() {
    let standard = Schedule::default();
    let (kv, tensors) = fixture_for(&standard);
    assert_eq!(
        plan(&write_fixture(kv.clone(), tensors.clone()))
            .unwrap()
            .schedule,
        standard
    );
    for spelling in 0..2 {
        for (short, long) in KEYS {
            let mut partial = kv.clone();
            declare(&mut partial, &standard, spelling);
            partial.retain(|(key, _)| *key != short && *key != long);
            assert_eq!(
                plan(&write_fixture(partial, tensors.clone()))
                    .unwrap()
                    .schedule,
                standard
            );
        }
    }

    let mut wrong_ratios = standard.clone();
    wrong_ratios.compression_ratios[19] = 1;
    let mut extended = standard.clone();
    extended.layers = 41;
    extended.compression_ratios.push(1);
    let mut shorter = standard;
    shorter.layers = 39;
    shorter.compression_ratios.pop();
    for schedule in [wrong_ratios, extended, shorter] {
        let (kv, tensors) = fixture_for(&schedule);
        assert_invalid(
            plan(&write_fixture(kv.clone(), tensors.clone())).unwrap_err(),
            "non-default layout",
        );
        for spelling in 0..2 {
            for (short, long) in KEYS {
                let mut partial = kv.clone();
                declare(&mut partial, &schedule, spelling);
                partial.retain(|(key, _)| *key != short && *key != long);
                assert_invalid(
                    plan(&write_fixture(partial, tensors.clone())).unwrap_err(),
                    "non-default layout",
                );
            }
        }
    }
}
