use std::path::Path;

/// Small real GGUF descriptors for CPU preflight; never loaded onto Metal.
pub(crate) fn write_cpu_gguf(
    path: &Path,
    family: &str,
    hidden: u64,
    token_claim: &str,
    bad_head: bool,
) {
    fn string(out: &mut Vec<u8>, text: &str) {
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }
    let mut metadata = Vec::new();
    for (key, text) in [
        ("general.architecture", family),
        ("tokenizer.ggml.model", token_claim),
    ] {
        let mut entry = Vec::new();
        string(&mut entry, key);
        entry.extend_from_slice(&8u32.to_le_bytes());
        string(&mut entry, text);
        metadata.push(entry);
    }
    for (key, value) in [
        ("block_count", 3),
        ("embedding_length", hidden),
        ("feed_forward_length", 4),
        ("attention.head_count", 1),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 2),
        ("full_attention_interval", 1),
        ("ssm.state_size", 2),
        ("ssm.time_step_rank", 1),
        ("ssm.group_count", 1),
        ("ssm.conv_kernel", 4),
        ("expert_count", 2),
        ("expert_used_count", 1),
        ("expert_feed_forward_length", 4),
        ("expert_shared_feed_forward_length", 4),
    ] {
        let mut entry = Vec::new();
        string(&mut entry, &format!("{family}.{key}"));
        entry.extend_from_slice(&10u32.to_le_bytes());
        entry.extend_from_slice(&value.to_le_bytes());
        metadata.push(entry);
    }
    let mut tensors = vec![
        ("token_embd.weight".to_owned(), vec![hidden, 32]),
        ("output_norm.weight".into(), vec![hidden]),
        (
            "output.weight".into(),
            vec![hidden + u64::from(bad_head), 32],
        ),
    ];
    for layer in 0..3 {
        for (name, shape) in [
            ("attn_norm", vec![hidden]),
            ("post_attention_norm", vec![hidden]),
            ("attn_q", vec![hidden, 4]),
            ("attn_k", vec![hidden, 2]),
            ("attn_v", vec![hidden, 2]),
            ("attn_output", vec![2, hidden]),
            ("attn_q_norm", vec![2]),
            ("attn_k_norm", vec![2]),
        ] {
            tensors.push((format!("blk.{layer}.{name}.weight"), shape));
        }
        let ffn = if family == "qwen35moe" {
            vec![
                ("ffn_gate_inp", vec![hidden, 2]),
                ("ffn_gate_exps", vec![hidden, 4, 2]),
                ("ffn_up_exps", vec![hidden, 4, 2]),
                ("ffn_down_exps", vec![4, hidden, 2]),
                ("ffn_gate_inp_shexp", vec![hidden]),
                ("ffn_gate_shexp", vec![hidden, 4]),
                ("ffn_up_shexp", vec![hidden, 4]),
                ("ffn_down_shexp", vec![4, hidden]),
            ]
        } else {
            vec![
                ("ffn_gate", vec![hidden, 4]),
                ("ffn_up", vec![hidden, 4]),
                ("ffn_down", vec![4, hidden]),
            ]
        };
        for (name, shape) in ffn {
            tensors.push((format!("blk.{layer}.{name}.weight"), shape));
        }
    }
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for entry in metadata {
        out.extend(entry);
    }
    let mut payload = Vec::new();
    for (name, shape) in tensors {
        payload.resize(payload.len().div_ceil(32) * 32, 0);
        string(&mut out, &name);
        out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in &shape {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        for _ in 0..shape.iter().product::<u64>() {
            payload.extend_from_slice(&1.0f32.to_le_bytes());
        }
    }
    out.resize(out.len().div_ceil(32) * 32, 0);
    out.extend(payload);
    std::fs::write(path, out).unwrap();
}
