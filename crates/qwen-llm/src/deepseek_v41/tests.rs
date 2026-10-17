use super::*;

const PINNED: &str =
    include_str!("../../../../scripts/reference/deepseek_v41/fixtures/config.json");

fn config() -> TextConfig {
    TextConfig::from_hf_json(PINNED, ExpertGeometry::Base384).unwrap()
}

#[test]
fn pinned_schedule_and_ownership() {
    let c = config();
    assert_eq!(c.layers().len(), 40);
    let owners: Vec<_> = c
        .layers()
        .iter()
        .filter(|p| p.kind() == LayerKind::Full)
        .map(LayerPlan::layer)
        .collect();
    assert_eq!(owners, KV_OWNERS);
    let producers: Vec<_> = c
        .layers()
        .iter()
        .filter(|p| matches!(p.kind(), LayerKind::Full | LayerKind::Reindex))
        .map(LayerPlan::layer)
        .collect();
    assert_eq!(producers, SELECTION_PRODUCERS);
    for (layer, owner, producer) in [
        (1, None, None),
        (2, Some(2), Some(2)),
        (7, Some(2), Some(2)),
        (8, Some(8), Some(8)),
        (13, Some(8), Some(8)),
        (14, Some(14), Some(14)),
        (19, Some(14), Some(14)),
        (20, Some(20), Some(20)),
        (23, Some(20), Some(20)),
        (24, Some(20), Some(24)),
        (27, Some(20), Some(24)),
        (28, Some(20), Some(28)),
        (31, Some(20), Some(28)),
        (32, Some(20), Some(32)),
        (35, Some(20), Some(32)),
        (36, Some(20), Some(36)),
        (39, Some(20), Some(36)),
    ] {
        assert_eq!(c.layers()[layer].kv_index_owner(), owner);
        assert_eq!(c.layers()[layer].selection_producer(), producer);
    }
    assert_eq!(c.layers()[0].kind(), LayerKind::Swa);
    assert_eq!(c.layers()[1].kv_index_owner(), None);
    assert_eq!(c.layers()[19].compression_ratio(), 2);
    assert_eq!(c.layers()[19].candidate_owner(), None);
    assert_eq!(c.layers()[20].compression_ratio(), 1);
    assert_eq!(c.layers()[20].candidate_owner(), Some(20));
    assert_eq!(c.layers()[24].kv_index_owner(), Some(20));
}

#[test]
fn rejects_text_mutations_and_numeric_types() {
    let original: Value = serde_json::from_str(PINNED).unwrap();
    for key in original["text_config"].as_object().unwrap().keys() {
        let mut v = original.clone();
        v["text_config"].as_object_mut().unwrap().remove(key);
        assert!(
            TextConfig::from_hf_json(&v.to_string(), ExpertGeometry::Base384).is_err(),
            "missing {key}"
        );
    }
    for (key, value) in [
        ("architectures", json!(["DeepseekV4ForCausalLM"])),
        ("model_type", json!("deepseek_v4")),
        ("dtype", json!("float32")),
        ("bos_token_id", json!(1)),
        ("eos_token_id", json!(0)),
        ("pad_token_id", Value::Null),
        ("quantization_config", json!({})),
    ] {
        let mut v = original.clone();
        v[key] = value;
        assert!(
            TextConfig::from_hf_json(&v.to_string(), ExpertGeometry::Base384).is_err(),
            "{key}"
        );
    }
    for (key, value) in [
        ("hidden_size", json!(4096)),
        ("moe_intermediate_size", json!(2048)),
        ("num_hidden_layers", json!(41)),
        ("hidden_size", json!(5120.0)),
        ("hidden_size", json!("5120")),
        ("hidden_size", json!(-5120)),
        ("rms_norm_eps", json!("NaN")),
        ("rms_norm_eps", Value::Null),
        ("kv_source_layer_ids", json!([2, 8, 14, 19])),
        (
            "index_source_layer_ids",
            json!([2, 8, 14, 20, 24, 28, 36, 32]),
        ),
        ("candidate_source_layer_id", json!(19)),
        ("scoring_func", json!("softmax")),
        ("engram_layer_ids", json!([1, 15])),
        ("dspark_target_layer_ids", json!([36, 38, 39])),
        ("unknown_semantics", json!(true)),
    ] {
        let mut v = original.clone();
        v["text_config"][key] = value;
        assert!(
            TextConfig::from_hf_json(&v.to_string(), ExpertGeometry::Base384).is_err(),
            "{key}"
        );
    }
    for i in 0..43 {
        let mut v = original.clone();
        v["text_config"]["compress_ratios"][i] = json!(3);
        assert!(TextConfig::from_hf_json(&v.to_string(), ExpertGeometry::Base384).is_err());
    }
    let mut v = original.clone();
    v["text_config"]["compress_ratios"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(TextConfig::from_hf_json(&v.to_string(), ExpertGeometry::Base384).is_err());
    for invalid in ["NaN", "Infinity", "1e999", "{\"text_config\":NaN}"] {
        assert!(TextConfig::from_hf_json(invalid, ExpertGeometry::Base384).is_err());
    }
}

#[test]
fn explicit_reap_geometry_only() {
    for geometry in [
        ExpertGeometry::Base384,
        ExpertGeometry::PotentialReap272,
        ExpertGeometry::PotentialReap256,
    ] {
        let mut v: Value = serde_json::from_str(PINNED).unwrap();
        v["text_config"]["n_routed_experts"] = json!(geometry.expert_count());
        let c = TextConfig::from_hf_json(&v.to_string(), geometry).unwrap();
        assert_eq!(c.geometry(), geometry);
        if geometry != ExpertGeometry::Base384 {
            assert!(TextConfig::from_hf_json(&v.to_string(), ExpertGeometry::Base384).is_err());
        }
        v["text_config"]["n_routed_experts"] = json!(271);
        assert!(TextConfig::from_hf_json(&v.to_string(), geometry).is_err());
    }
}

#[test]
fn real_parameters_allow_equivalent_exponent_spellings() {
    let mut value: Value = serde_json::from_str(PINNED).unwrap();
    value["text_config"]["hc_eps"] = serde_json::from_str("0.000001").unwrap();
    value["text_config"]["rms_norm_eps"] = serde_json::from_str("10e-21").unwrap();
    value["text_config"]["attention_dropout"] = json!(0);
    TextConfig::from_hf_json(&value.to_string(), ExpertGeometry::Base384).unwrap();
    value["text_config"]["hc_eps"] = json!(0.000002);
    assert!(
        TextConfig::from_hf_json(&value.to_string(), ExpertGeometry::Base384)
            .unwrap_err()
            .contains("text_config.hc_eps")
    );
}

#[test]
fn expert_storage_geometry_is_not_a_runtime_admission() {
    use ExpertProjection::{Down, Gate, Up};
    assert_eq!(Gate.source_shapes(), ([2304, 2560], [2304, 160]));
    assert_eq!(Up.source_shapes(), Gate.source_shapes());
    assert_eq!(Down.source_shapes(), ([5120, 1152], [5120, 72]));
    for (geometry, bytes) in [
        (ExpertGeometry::Base384, 2_406_481_920),
        (ExpertGeometry::PotentialReap272, 1_704_591_360),
        (ExpertGeometry::PotentialReap256, 1_604_321_280),
    ] {
        for projection in [Gate, Up, Down] {
            assert_eq!(geometry.projection_bytes(projection), bytes);
        }
    }
}

#[test]
fn position_boundaries_are_eligibility_not_completion() {
    let c = config();
    for (pos, completed_pairs) in [
        (0, 0..0),
        (1, 0..1),
        (2, 1..1),
        (127, 63..64),
        (128, 64..64),
        (129, 64..65),
    ] {
        let p = c.publication(pos..pos + 1).unwrap();
        assert_eq!(p.swa_context(), (pos + 1).saturating_sub(128)..pos + 1);
        for owner in p.owners() {
            assert_eq!(
                owner.rows(),
                if owner.owner() < 20 {
                    completed_pairs.clone()
                } else {
                    pos..pos + 1
                }
            );
        }
    }
    assert_eq!(c.publication(0..0).unwrap().owners()[0].rows(), 0..0);
    for end in [16383, 16384, 16385, MAX_CONTEXT] {
        assert_eq!(
            c.publication(0..end).unwrap().candidate_capacity(),
            end.min(16384)
        );
    }
    assert!(c.publication(2..1).is_err());
    assert!(c.publication(0..MAX_CONTEXT + 1).is_err());
    assert!(c.publication(usize::MAX..usize::MAX).is_err());
    assert_ne!(
        ActivationArithmetic::NativeF32,
        ActivationArithmetic::ReferenceFp8
    );
}
