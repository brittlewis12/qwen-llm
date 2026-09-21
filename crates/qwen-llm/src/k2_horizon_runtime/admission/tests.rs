use super::*;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn metadata() -> BTreeMap<String, Value> {
    let mut tokens = vec!["<bos>".to_owned(), "<eos>".to_owned()];
    let mut extra = 0;
    for byte in 0u32..256 {
        let code = if (33..=126).contains(&byte)
            || (161..=172).contains(&byte)
            || (174..=255).contains(&byte)
        {
            byte
        } else {
            extra += 1;
            255 + extra
        };
        tokens.push(char::from_u32(code).unwrap().to_string());
    }
    tokens.extend((tokens.len()..250624).map(|i| format!("fixture{i}")));
    let mut types = vec![1; tokens.len()];
    types[0] = 3;
    types[1] = 3;
    serde_json::from_value(json!({
        "general.architecture":"k2-horizon", "k2-horizon.block_count":36,
        "k2-horizon.context_length":8192,"k2-horizon.embedding_length":4096,
        "k2-horizon.feed_forward_length":12288,"k2-horizon.attention.head_count":32,
        "k2-horizon.attention.head_count_kv":8,"k2-horizon.attention.key_length":128,
        "k2-horizon.attention.value_length":128,"k2-horizon.attention.group_norm_groups":4,
        "k2-horizon.attention.layer_norm_rms_epsilon":1e-6,
        "k2-horizon.rope.dimension_count":128,"k2-horizon.rope.freq_base":500000.0,
        "tokenizer.ggml.model":"gpt2","tokenizer.ggml.pre":"k2-horizon",
        "tokenizer.ggml.tokens":tokens,"tokenizer.ggml.token_type":types,
        "tokenizer.ggml.merges":[],"tokenizer.ggml.bos_token_id":0,
        "tokenizer.ggml.eos_token_id":1,"tokenizer.ggml.add_bos_token":true,
        "tokenizer.ggml.add_eos_token":false
    }))
    .unwrap()
}

fn inventory() -> Vec<(String, Vec<u64>, GgmlType)> {
    let mut rows = vec![
        (
            "token_embd.weight".into(),
            vec![4096, 250624],
            GgmlType::Q8_0,
        ),
        ("output.weight".into(), vec![4096, 250624], GgmlType::Q8_0),
        ("output_norm.weight".into(), vec![4096], GgmlType::F32),
    ];
    for layer in 0..36 {
        for suffix in ["attn_norm", "ffn_norm"] {
            rows.push((
                format!("blk.{layer}.{suffix}.weight"),
                vec![4096],
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
            rows.push((
                format!("blk.{layer}.{suffix}.weight"),
                shape.to_vec(),
                GgmlType::Q8_0,
            ));
        }
    }
    rows
}

fn string(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}
fn kind(value: &Value) -> u32 {
    if value.is_string() {
        8
    } else if value.is_array() {
        9
    } else if value.is_boolean() {
        7
    } else if value.is_u64() {
        4
    } else {
        6
    }
}
fn value(out: &mut Vec<u8>, v: &Value) {
    match kind(v) {
        8 => string(out, v.as_str().unwrap()),
        7 => out.push(u8::from(v.as_bool().unwrap())),
        4 => out.extend_from_slice(&(v.as_u64().unwrap() as u32).to_le_bytes()),
        6 => out.extend_from_slice(&(v.as_f64().unwrap() as f32).to_le_bytes()),
        9 => {
            let array = v.as_array().unwrap();
            out.extend_from_slice(&array.first().map_or(8, kind).to_le_bytes());
            out.extend_from_slice(&(array.len() as u64).to_le_bytes());
            for item in array {
                value(out, item);
            }
        }
        _ => unreachable!(),
    }
}

fn fixture(
    metadata: &BTreeMap<String, Value>,
    tensors: &[(String, Vec<u64>, GgmlType)],
) -> Fixture {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, v) in metadata {
        string(&mut bytes, key);
        bytes.extend_from_slice(&kind(v).to_le_bytes());
        value(&mut bytes, v);
    }
    let mut extent = 0u64;
    for (name, shape, dtype) in tensors {
        string(&mut bytes, name);
        bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in shape {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&(*dtype as u32).to_le_bytes());
        bytes.extend_from_slice(&extent.to_le_bytes());
        let (block, size) = dtype.storage_layout().unwrap();
        extent += (shape.iter().product::<u64>() / block * size).next_multiple_of(32);
    }
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    let path = std::env::temp_dir().join(format!(
        "k2-admission-{}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .unwrap();
    file.write_all(&bytes).unwrap();
    // Real nonoverlapping tensor ranges in a sparse file. No weight payload is
    // written, read, authenticated, executed, or represented as a trained model.
    file.set_len(bytes.len() as u64 + extent).unwrap();
    Fixture(path)
}

#[test]
fn k2_cpu_preparation_admits_raw_and_separates_stop_policy_from_lens() {
    let mut meta = metadata();
    let tensors = inventory();
    for stops in [json!(1), json!([1, 2])] {
        meta.insert("tokenizer.ggml.eos_token_id".into(), stops.clone());
        let f = fixture(&meta, &tensors);
        let source = GgufFile::open(&f.0).unwrap();
        let prepared = K2PreparedArtifact::inspect(&source).unwrap();
        assert_eq!(prepared.config().context_length, 8192);
        assert_eq!(prepared.tokenizer().n_vocab(), 250624);
        assert_eq!(prepared.tokenizer().encode("a", true).unwrap(), [0, 99]);
        assert_eq!(prepared.generation_stops().is_ok(), stops == json!(1));
        if stops != json!(1) {
            assert_eq!(
                prepared.generation_stops().unwrap_err().code(),
                "k2_generation_stops"
            );
        }
        assert!(crate::k2_horizon_chat::verify_profile(&source).is_err());
    }
}

#[test]
fn k2_cpu_preparation_rejects_geometry_inventory_storage_tokenizer_and_ranges() {
    let base = metadata();
    let tensors = inventory();
    for (case, expected) in [
        ("geometry", "k2_configuration"),
        ("missing", "k2_tensor_inventory"),
        ("extra", "k2_tensor_inventory"),
        ("norm_storage", "k2_tensor_inventory"),
        ("embedding", "k2_embedding_storage"),
        ("tokenizer", "k2_tokenizer"),
        ("extent", "k2_tensor_inventory"),
        ("overlap", "k2_retained_storage"),
    ] {
        let mut meta = base.clone();
        let mut rows = tensors.clone();
        match case {
            "geometry" => {
                meta.insert("k2-horizon.embedding_length".into(), json!(2048));
            }
            "missing" => {
                rows.pop();
            }
            "extra" => {
                rows.push(("extra.weight".into(), vec![4096], GgmlType::F32));
            }
            "norm_storage" => rows[2].2 = GgmlType::F16,
            "embedding" => rows[0].2 = GgmlType::Q5_K,
            "tokenizer" => {
                meta.insert("tokenizer.ggml.token_type".into(), json!([1, 2]));
            }
            _ => {}
        }
        let f = fixture(&meta, &rows);
        let mut source = GgufFile::open(&f.0).unwrap();
        if case == "extent" {
            source.tensors[0].data_offset = u64::MAX - 8;
        }
        if case == "overlap" {
            source.tensors[1].data_offset = source.tensors[0].data_offset;
        }
        let error = match K2PreparedArtifact::inspect(&source) {
            Ok(_) => panic!("admitted {case}"),
            Err(e) => e,
        };
        assert_eq!(error.code(), expected, "{case}: {error}");
    }
}
