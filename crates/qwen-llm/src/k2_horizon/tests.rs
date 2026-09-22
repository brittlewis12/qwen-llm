use super::*;
use serde_json::json;

fn metadata(context: u32, theta: f32) -> BTreeMap<String, Value> {
    let mut metadata: BTreeMap<String, Value> = serde_json::from_value(json!({
        "general.architecture": "k2-horizon",
        "k2-horizon.block_count": 36,
        "k2-horizon.context_length": context,
        "k2-horizon.embedding_length": 4096,
        "k2-horizon.feed_forward_length": 12288,
        "k2-horizon.attention.head_count": 32,
        "k2-horizon.attention.head_count_kv": 8,
        "k2-horizon.attention.key_length": 128,
        "k2-horizon.attention.value_length": 128,
        "k2-horizon.attention.group_norm_groups": 4,
        "k2-horizon.attention.layer_norm_rms_epsilon": 1e-6,
        "k2-horizon.rope.dimension_count": 128,
        "k2-horizon.rope.freq_base": theta,
        "k2-horizon.expert_count": 0,
        "k2-horizon.expert_used_count": 0,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "k2-horizon"
    }))
    .unwrap();
    // Structural fixture only, deliberately not a conforming tokenizer.
    metadata.insert(
        "tokenizer.ggml.tokens".into(),
        json!(vec!["fixture"; 250624]),
    );
    metadata
}

fn config() -> K2HorizonConfig {
    K2HorizonConfig::from_metadata(&metadata(524288, 10_000_000.0)).unwrap()
}

fn tensor(name: &str, shape: &[u64], dtype: GgmlType) -> TensorDesc {
    let (block, size) = dtype.storage_layout().unwrap();
    TensorDesc {
        name: name.into(),
        shape: shape.into(),
        dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: shape.iter().product::<u64>() / block * size,
    }
}

fn inventory(dtype: GgmlType) -> Vec<TensorDesc> {
    let mut tensors = vec![
        tensor("token_embd.weight", &[4096, 250624], dtype),
        tensor("output_norm.weight", &[4096], GgmlType::F32),
        tensor("output.weight", &[4096, 250624], dtype),
    ];
    for layer in 0..36 {
        for suffix in ["attn_norm", "ffn_norm"] {
            tensors.push(tensor(
                &format!("blk.{layer}.{suffix}.weight"),
                &[4096],
                GgmlType::F32,
            ));
        }
        for (suffix, shape) in [
            ("attn_q", [4096, 4096]),
            ("attn_k", [4096, 1024]),
            ("attn_v", [4096, 1024]),
            ("attn_output", [4096, 4096]),
            ("ffn_gate", [4096, 12288]),
            ("ffn_up", [4096, 12288]),
            ("ffn_down", [12288, 4096]),
        ] {
            tensors.push(tensor(
                &format!("blk.{layer}.{suffix}.weight"),
                &shape,
                dtype,
            ));
        }
    }
    tensors
}

#[test]
fn stage_positions_are_preserved_without_names_or_chat() {
    for (context, theta) in [
        (8192, 500_000.0),
        (32768, 1_000_000.0),
        (131072, 10_000_000.0),
        (524288, 10_000_000.0),
    ] {
        let mut metadata = metadata(context, theta);
        let expected = K2HorizonConfig::from_metadata(&metadata).unwrap();
        assert_eq!(expected.context_length, context);
        assert_eq!(expected.rope_theta, theta);
        metadata.insert(
            "general.name".into(),
            json!("arbitrary research checkpoint"),
        );
        metadata.insert(
            "tokenizer.chat_template".into(),
            json!("not a validated template"),
        );
        assert_eq!(K2HorizonConfig::from_metadata(&metadata).unwrap(), expected);
        assert_eq!(
            crate::model_family::ModelFamily::from_architecture_name(ARCHITECTURE_NAME),
            Some(crate::model_family::ModelFamily::K2Horizon)
        );
    }
}

#[test]
fn distinguishes_missing_invalid_and_unsupported_metadata() {
    let mut metadata = metadata(8192, 500_000.0);
    metadata.remove("k2-horizon.rope.freq_base");
    assert!(matches!(
        K2HorizonConfig::from_metadata(&metadata),
        Err(K2HorizonError::MissingMetadata(_))
    ));
    metadata.insert("k2-horizon.rope.freq_base".into(), json!("500000"));
    assert!(matches!(
        K2HorizonConfig::from_metadata(&metadata),
        Err(K2HorizonError::InvalidMetadata { .. })
    ));
    metadata.insert("k2-horizon.rope.freq_base".into(), json!(500000));
    metadata.insert("k2-horizon.rope.dimension_count".into(), json!(64));
    assert!(matches!(
        K2HorizonConfig::from_metadata(&metadata),
        Err(K2HorizonError::Unsupported { .. })
    ));
}

#[test]
fn rejects_incompatible_graph_metadata() {
    let base = metadata(8192, 500_000.0);
    for (key, value) in [
        ("general.architecture", json!("qwen35")),
        ("general.type", json!("adapter")),
        ("k2-horizon.block_count", json!(48)),
        ("k2-horizon.embedding_length", json!(2560)),
        ("k2-horizon.feed_forward_length", json!(768)),
        ("k2-horizon.attention.head_count", json!(0)),
        ("k2-horizon.attention.head_count_kv", json!(2)),
        ("k2-horizon.attention.key_length", json!(256)),
        ("k2-horizon.attention.value_length", json!(64)),
        ("k2-horizon.attention.group_norm_groups", json!(1)),
        ("k2-horizon.attention.layer_norm_rms_epsilon", json!(1e-5)),
        ("k2-horizon.expert_count", json!(100)),
        ("k2-horizon.attention.value_expert_count", json!(64)),
        ("k2-horizon.attention.sliding_window", json!(2048)),
        ("k2-horizon.rope.scaling.type", json!("yarn")),
        ("k2-horizon.rope.scaling.factor", json!(16.0)),
        ("k2-horizon.future_graph_feature", json!(true)),
        ("tokenizer.ggml.pre", json!("qwen35")),
    ] {
        let mut metadata = base.clone();
        metadata.insert(key.into(), value);
        assert!(
            K2HorizonConfig::from_metadata(&metadata).is_err(),
            "accepted {key}"
        );
    }
}

#[test]
fn malformed_sizes_positions_and_vocabulary_fail() {
    let base = metadata(8192, 500_000.0);
    for (key, value) in [
        ("k2-horizon.block_count", json!(u64::MAX)),
        ("k2-horizon.context_length", json!(0)),
        ("k2-horizon.context_length", json!(u64::from(u32::MAX) + 1)),
        ("k2-horizon.context_length", json!(-1)),
        ("k2-horizon.rope.freq_base", json!(0)),
        ("k2-horizon.rope.freq_base", json!(-1)),
        ("k2-horizon.rope.freq_base", json!(1e100)),
        ("k2-horizon.rope.freq_base", json!(1e-100)),
        ("tokenizer.ggml.tokens", json!(42)),
        ("tokenizer.ggml.tokens", json!(["a"])),
        ("k2-horizon.expert_count", json!(false)),
    ] {
        let mut metadata = base.clone();
        metadata.insert(key.into(), value);
        assert!(
            K2HorizonConfig::from_metadata(&metadata).is_err(),
            "accepted {key}"
        );
    }
    let mut malformed = base;
    malformed
        .get_mut("tokenizer.ggml.tokens")
        .unwrap()
        .as_array_mut()
        .unwrap()[123] = json!(123);
    assert!(K2HorizonConfig::from_metadata(&malformed).is_err());
}

#[test]
fn logical_cache_bytes_include_both_kv_rows_and_q8_scales() {
    let config = config();
    assert_eq!(
        config.kv_storage_bytes(1, K2KvStorage::F16).unwrap(),
        147456
    );
    assert_eq!(
        config.kv_storage_bytes(1, K2KvStorage::Q8_0).unwrap(),
        78336
    );
    assert_eq!(
        config.kv_storage_bytes(524288, K2KvStorage::F16).unwrap(),
        72 * 1024u64.pow(3)
    );
    assert_eq!(
        config.kv_storage_bytes(524288, K2KvStorage::Q8_0).unwrap(),
        41_070_624_768
    );
    for capacity in [0, 524289, u64::MAX] {
        assert!(config.kv_storage_bytes(capacity, K2KvStorage::F16).is_err());
    }
    assert!(K2KvStorage::F16.row_bytes(u64::MAX).is_err());
    assert!(K2KvStorage::Q8_0.row_bytes(1023).is_err());
    assert!(K2KvStorage::Q8_0.row_bytes(u64::MAX - 31).is_err());
}

#[test]
fn structural_inventory_binds_without_weight_materialization() {
    for dtype in [
        GgmlType::Q8_0,
        GgmlType::BF16,
        GgmlType::F16,
        GgmlType::F32,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
    ] {
        let tensors = inventory(dtype);
        assert_eq!(tensors.len(), DENSE_7B_TENSOR_COUNT);
        let bound = K2HorizonModel::bind(config(), &tensors).unwrap();
        assert_eq!(bound.layers.len(), 36);
        assert_eq!(bound.output.shape, [4096, 250624]);
        assert_eq!(bound.layers[35].key.shape, [4096, 1024]);
        assert_eq!(
            bound.weight_payload_bytes,
            tensors.iter().map(|t| t.n_bytes).sum::<u64>()
        );
        if dtype == GgmlType::Q8_0 {
            assert_eq!(bound.weight_payload_bytes, 9_562_505_216);
        }
    }
}

#[test]
fn missing_duplicate_extra_and_wrong_tensors_fail() {
    let base = inventory(GgmlType::Q8_0);
    let mut missing = base.clone();
    missing.remove(2);
    assert!(K2HorizonModel::bind(config(), &missing).is_err());
    let mut duplicate = base.clone();
    duplicate.push(base[0].clone());
    assert!(K2HorizonModel::bind(config(), &duplicate).is_err());
    let mut extra = base.clone();
    extra.push(tensor(
        "blk.0.attn_gate.weight",
        &[4096, 4096],
        GgmlType::Q8_0,
    ));
    assert!(K2HorizonModel::bind(config(), &extra).is_err());
    for index in [0, 1, 2, 4, 326] {
        let mut wrong_shape = base.clone();
        wrong_shape[index].shape[0] += 1;
        assert!(K2HorizonModel::bind(config(), &wrong_shape).is_err());
        let mut wrong_bytes = base.clone();
        wrong_bytes[index].n_bytes += 1;
        assert!(K2HorizonModel::bind(config(), &wrong_bytes).is_err());
        let mut wrong_type = base.clone();
        wrong_type[index].dtype = GgmlType::I8;
        assert!(K2HorizonModel::bind(config(), &wrong_type).is_err());
    }
}

#[test]
fn storage_checks_rows_and_overflow_independently() {
    let mut t = tensor("test", &[4096, 32], GgmlType::Q8_0);
    t.shape[0] = 4095;
    assert!(validate_storage(&t, false).is_err());
    t.shape = vec![4096, u64::MAX];
    assert!(validate_storage(&t, false).is_err());
    t.shape.clear();
    assert!(validate_storage(&t, false).is_err());
    t = tensor("test", &[4096, 32], GgmlType::Q8_0);
    t.data_offset = u64::MAX;
    assert!(validate_storage(&t, false).is_err());
    assert!(validate_storage(&tensor("norm", &[4096], GgmlType::BF16), true).is_err());
}

#[test]
#[ignore = "CPU/header-only inspection; requires a downloaded K2 GGUF, never executes the model"]
fn inspect_local_q8_header() {
    let path = std::env::var("K2_GGUF").expect("set K2_GGUF explicitly");
    let gguf = GgufFile::open(path).unwrap();
    let model = K2HorizonModel::from_gguf(&gguf).unwrap();
    assert_eq!(model.layers.len(), 36);
    assert_eq!(model.weight_payload_bytes, 9_562_505_216);
    assert_eq!(model.config.context_length, 524288);
    assert_eq!(model.config.rope_theta, 10_000_000.0);
}
