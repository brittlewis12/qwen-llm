use super::*;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend((value.len() as u64).to_le_bytes());
    out.extend(value.as_bytes());
}

// Tiny, real GGUF with one draft block and an independently sized capture list.
fn fixture(arch: &str, ids: &[u32], causal: Option<(u32, Vec<u8>)>) -> GgufFile {
    let v2 = arch == "dflash";
    let mut metadata = Vec::new();
    let mut arch_value = Vec::new();
    string(&mut arch_value, arch);
    metadata.push(("general.architecture".into(), 8u32, arch_value));
    for (key, value) in [
        ("block_count", 1u32),
        ("embedding_length", 32),
        ("feed_forward_length", 32),
        ("attention.head_count", 1),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 32),
        ("attention.sliding_window", 4),
        (
            if v2 {
                "block_size"
            } else {
                "dflash.block_size"
            },
            8,
        ),
    ] {
        metadata.push((format!("{arch}.{key}"), 4, value.to_le_bytes().to_vec()));
    }
    metadata.push((
        if v2 {
            "tokenizer.ggml.mask_token_id".into()
        } else {
            format!("{arch}.dflash.mask_token_id")
        },
        4,
        1u32.to_le_bytes().to_vec(),
    ));
    let mut array = 4u32.to_le_bytes().to_vec();
    array.extend((ids.len() as u64).to_le_bytes());
    for id in ids {
        array.extend(id.to_le_bytes());
    }
    metadata.push((
        format!(
            "{arch}.{}",
            if v2 {
                "target_layers"
            } else {
                "dflash.target_layer_ids"
            }
        ),
        9,
        array,
    ));
    if !v2 {
        metadata.push((
            format!("{arch}.dflash.n_target_features"),
            4,
            (ids.len() as u32 * 32).to_le_bytes().to_vec(),
        ));
    }
    let mut pattern = 7u32.to_le_bytes().to_vec();
    pattern.extend(1u64.to_le_bytes());
    pattern.push(1);
    metadata.push((
        format!("{arch}.attention.sliding_window_pattern"),
        9,
        pattern,
    ));
    if let Some((kind, value)) = causal {
        metadata.push((format!("{arch}.attention.causal"), kind, value));
    }

    let mut tensors = vec![
        (
            if v2 {
                "fc.weight".into()
            } else {
                "dflash_fc.weight".into()
            },
            vec![ids.len() as u64 * 32, 32],
        ),
        (
            if v2 {
                "enc.output_norm.weight".into()
            } else {
                "dflash_hidden_norm.weight".into()
            },
            vec![32],
        ),
        ("output_norm.weight".into(), vec![32]),
        ("token_embd.weight".into(), vec![32, 2]),
    ];
    for suffix in [
        "attn_norm",
        "attn_q_norm",
        "attn_k_norm",
        if v2 {
            "ffn_norm"
        } else {
            "post_attention_norm"
        },
    ] {
        tensors.push((format!("blk.0.{suffix}.weight"), vec![32]));
    }
    for suffix in [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ] {
        tensors.push((format!("blk.0.{suffix}.weight"), vec![32, 32]));
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend((tensors.len() as u64).to_le_bytes());
    bytes.extend((metadata.len() as u64).to_le_bytes());
    for (key, kind, value) in metadata {
        string(&mut bytes, &key);
        bytes.extend(kind.to_le_bytes());
        bytes.extend(value);
    }
    let mut offset = 0u64;
    let mut payload = Vec::new();
    for (name, shape) in tensors {
        string(&mut bytes, &name);
        bytes.extend((shape.len() as u32).to_le_bytes());
        for dim in &shape {
            bytes.extend(dim.to_le_bytes());
        }
        bytes.extend(0u32.to_le_bytes()); // F32
        bytes.extend(offset.to_le_bytes());
        offset += shape.iter().product::<u64>() * 4;
        for index in 0..shape.iter().product::<u64>() {
            let value: f32 = if shape.len() == 1
                || (name == "token_embd.weight" && (index == 0 || index == 33))
                || (matches!(
                    name.as_str(),
                    "blk.0.attn_v.weight" | "blk.0.attn_output.weight"
                ) && index / 32 == index % 32)
            {
                1.0
            } else {
                0.0
            };
            payload.extend(value.to_le_bytes());
        }
    }
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes.extend(payload);
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "dflash-metadata-{}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    file.write_all(&bytes).unwrap();
    std::fs::remove_file(&path).unwrap();
    GgufFile::from_opened_file(file, path).unwrap()
}

fn target(g: &GgufFile) -> Model<'_> {
    let tensor = g.find("output_norm.weight").unwrap();
    Model {
        arch: Arch {
            hidden_size: 32,
            vocab_size: 2,
            ..crate::model::QWEN3_27B
        },
        token_embd: g.find("token_embd.weight").unwrap(),
        output_norm: tensor,
        lm_head: g.find("token_embd.weight").unwrap(),
        blocks: vec![],
        tied_embeddings: true,
        mtp: None,
    }
}

#[test]
fn dflash_legacy_cpu_forward_obeys_model_causality_and_window() {
    for causal in [false, true] {
        let g = fixture("dflash-draft", &[1], Some((7, vec![u8::from(causal)])));
        let model = target(&g);
        let mut head = open_dflash_drafter(&g, &model).unwrap();
        let forward = crate::forward::Forward::new(&g, &model);
        for swa in [false, true] {
            head.layers[0].is_swa = swa;
            let run = |ids: &[i32]| {
                forward
                    .dflash_draft(&head, &g, ids, &[], 0, &[], 10)
                    .unwrap()
            };
            let baseline = run(&[0; 8]);
            for future_index in [1, 3, 4] {
                let mut ids = [0; 8];
                ids[future_index] = 1;
                let changed = run(&ids);
                let visible = !causal && (!swa || future_index < head.config.swa_window as usize);
                let delta = (changed[1] - baseline[1]).abs();
                if visible {
                    assert!(delta > 0.1, "future token must affect first-row logits");
                } else {
                    assert_eq!(&changed[..2], &baseline[..2]);
                }
            }
        }
    }
}

#[test]
fn dflash_metadata_capture_conventions_preserve_order_and_tap_count() {
    for (arch, stored, expected) in [
        ("dflash-draft", vec![0, 31, 1], vec![0, 31, 1]),
        ("dflash", vec![6, 20, 34, 48, 62], vec![5, 19, 33, 47, 61]),
        (
            "dflash",
            vec![2, 7, 12, 17, 23, 28, 33, 38],
            vec![1, 6, 11, 16, 22, 27, 32, 37],
        ),
        ("dflash", vec![64, 1, 4], vec![63, 0, 3]),
    ] {
        let g = fixture(arch, &stored, None);
        let model = target(&g);
        let head = open_dflash_drafter(&g, &model).unwrap();
        assert_eq!(head.target_layer_ids, expected);
        assert_eq!(head.config.n_layer, 1);
        assert_eq!(head.config.n_target_features_layers as usize, stored.len());
        assert_eq!(
            head.config.selector_top_k, 0,
            "format semantics, not selector presence"
        );
    }
}

#[test]
fn dflash_metadata_rejects_layer_input_zero() {
    let g = fixture("dflash", &[0, 1], None);
    assert!(
        matches!(open_dflash_drafter(&g, &target(&g)), Err(LoadError::BadMetadata(message)) if message.contains("layer input 0"))
    );
}

#[test]
fn dflash_metadata_causality_is_per_model_and_defaults_to_legacy() {
    for arch in ["dflash-draft", "dflash"] {
        for flag in [None, Some(false), Some(true)] {
            let g = fixture(arch, &[1, 3], flag.map(|v| (7, vec![u8::from(v)])));
            let model = target(&g);
            let head = open_dflash_drafter(&g, &model).unwrap();
            assert_eq!(head.config.causal, flag.unwrap_or(true));
        }
        let g = fixture(
            arch,
            &[1],
            Some((8, {
                let mut s = Vec::new();
                string(&mut s, "false");
                s
            })),
        );
        assert!(matches!(
            open_dflash_drafter(&g, &target(&g)),
            Err(LoadError::BadMetadata(_))
        ));
    }
}
