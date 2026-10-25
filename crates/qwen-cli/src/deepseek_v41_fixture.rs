//! Tiny header fixtures shared by CLI unit tests; never loaded on Metal.
use qwen_llm::gguf::GgufFile;
use qwen_llm::tensor::GgmlType;
use std::io::Write;

enum Value {
    Number(u32),
    Float(f32),
    Text(String),
    Numbers(Vec<u32>),
    Texts(Vec<String>),
}

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend((value.len() as u64).to_le_bytes());
    out.extend(value.as_bytes());
}

pub(crate) fn gguf(experts: u32, context_key: &str) -> GgufFile {
    gguf_with_banks(experts, context_key, 2, [GgmlType::F32; 3])
}

pub(crate) fn gguf_with_banks(
    experts: u32,
    context_key: &str,
    width: u32,
    banks: [GgmlType; 3],
) -> GgufFile {
    use Value::*;
    let h = u64::from(width);
    let mut tokens = Vec::new();
    let mut extra = 256;
    for byte in 0..=255u32 {
        let code = if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174 {
            byte
        } else {
            extra += 1;
            extra - 1
        };
        tokens.push(char::from_u32(code).unwrap().to_string());
    }
    tokens.push(deepseek_recipe_encoding::v4::EOS_TOKEN.into());
    let kv = vec![
        ("general.architecture", Text("deepseek41".into())),
        ("deepseek41.expert_count", Number(experts)),
        ("deepseek41.expert_used_count", Number(1)),
        ("deepseek41.block_count", Number(1)),
        ("deepseek41.vocab_size", Number(257)),
        ("deepseek41.embedding_length", Number(width)),
        ("deepseek41.expert_feed_forward_length", Number(width)),
        ("deepseek41.attention.head_count_kv", Number(1)),
        ("deepseek41.expert_shared_count", Number(1)),
        ("deepseek41.hc_mult", Number(4)),
        ("deepseek41.hc_sinkhorn_iters", Number(20)),
        ("deepseek41.index_topk", Number(512)),
        ("deepseek41.sliding_window", Number(128)),
        ("deepseek41.rms_norm_eps", Float(1e-20)),
        ("deepseek41.hc_eps", Float(1e-6)),
        ("deepseek41.swiglu_limit", Float(10.0)),
        ("deepseek41.routed_scaling_factor", Float(1.5)),
        ("deepseek41.rope_theta", Float(10000.0)),
        ("deepseek41.compress_rope_theta", Float(160000.0)),
        ("deepseek41.rope_scaling.factor", Float(16.0)),
        ("deepseek41.rope_scaling.beta_fast", Float(32.0)),
        ("deepseek41.rope_scaling.beta_slow", Float(1.0)),
        (
            "deepseek41.rope_scaling.original_max_position_embeddings",
            Number(65536),
        ),
        ("deepseek41.compress_ratios", Numbers(vec![0])),
        ("deepseek41.kv_source_layer_ids", Numbers(vec![])),
        ("deepseek41.index_source_layer_ids", Numbers(vec![])),
        ("deepseek41.num_attention_heads", Number(1)),
        ("deepseek41.head_dim", Number(2)),
        ("deepseek41.qk_rope_head_dim", Number(2)),
        ("deepseek41.q_lora_rank", Number(2)),
        ("deepseek41.o_lora_rank", Number(2)),
        ("deepseek41.o_groups", Number(1)),
        ("deepseek41.index_n_heads", Number(1)),
        ("deepseek41.index_head_dim", Number(2)),
        (context_key, Number(64)),
        ("deepseek41.engram.token_map", Numbers((0..257).collect())),
        ("deepseek41.engram.pad_id", Number(1)),
        ("deepseek41.engram.layers", Numbers(vec![])),
        ("tokenizer.ggml.model", Text("gpt2".into())),
        ("tokenizer.ggml.pre", Text("joyai-llm".into())),
        ("tokenizer.ggml.tokens", Texts(tokens)),
        ("tokenizer.ggml.merges", Texts(vec![])),
        (
            "tokenizer.ggml.token_type",
            Numbers((0..257).map(|i| if i == 256 { 3 } else { 1 }).collect()),
        ),
        ("tokenizer.ggml.bos_token_id", Number(256)),
        ("tokenizer.ggml.eos_token_id", Number(256)),
    ];
    let mut tensors: Vec<(String, Vec<u64>, GgmlType)> = vec![
        ("token_embd.weight".into(), vec![h, 257], GgmlType::F32),
        ("output.weight".into(), vec![h, 257], GgmlType::F32),
        ("output_norm.weight".into(), vec![h], GgmlType::F32),
    ];
    for (name, shape) in [
        ("attn_norm.weight", vec![h]),
        ("ffn_norm.weight", vec![h]),
        ("attn_sinks.weight", vec![1]),
        ("attn_q_a.weight", vec![h, 2]),
        ("attn_q_b.weight", vec![2, 2]),
        ("attn_q_a_norm.weight", vec![2]),
        ("attn_kv.weight", vec![h, 2]),
        ("attn_kv_a_norm.weight", vec![2]),
        ("attn_output_a.weight", vec![2, 2]),
        ("attn_output_b.weight", vec![2, h]),
        ("ffn_gate_inp.weight", vec![h, experts as u64]),
        ("exp_probs_b.bias", vec![experts as u64]),
        ("ffn_gate_shexp.weight", vec![h, h]),
        ("ffn_up_shexp.weight", vec![h, h]),
        ("ffn_down_shexp.weight", vec![h, h]),
        ("hc_attn_fn.weight", vec![4 * h, 24]),
        ("hc_attn_scale.weight", vec![3]),
        ("hc_attn_base.weight", vec![24]),
        ("hc_ffn_fn.weight", vec![4 * h, 24]),
        ("hc_ffn_scale.weight", vec![3]),
        ("hc_ffn_base.weight", vec![24]),
        ("ffn_gate_exps.weight", vec![h, h, experts as u64]),
        ("ffn_up_exps.weight", vec![h, h, experts as u64]),
        ("ffn_down_exps.weight", vec![h, h, experts as u64]),
    ] {
        let dtype = match name {
            "ffn_gate_exps.weight" => banks[0],
            "ffn_up_exps.weight" => banks[1],
            "ffn_down_exps.weight" => banks[2],
            _ => GgmlType::F32,
        };
        tensors.push((format!("blk.0.{name}"), shape, dtype));
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((kv.len() as u64).to_le_bytes());
    for (key, value) in kv {
        string(&mut out, key);
        match value {
            Float(n) => {
                out.extend(6u32.to_le_bytes());
                out.extend(n.to_le_bytes());
            }
            Number(n) => {
                out.extend(4u32.to_le_bytes());
                out.extend(n.to_le_bytes());
            }
            Text(s) => {
                out.extend(8u32.to_le_bytes());
                string(&mut out, &s);
            }
            Numbers(values) => {
                out.extend(9u32.to_le_bytes());
                out.extend(4u32.to_le_bytes());
                out.extend((values.len() as u64).to_le_bytes());
                for n in values {
                    out.extend(n.to_le_bytes());
                }
            }
            Texts(values) => {
                out.extend(9u32.to_le_bytes());
                out.extend(8u32.to_le_bytes());
                out.extend((values.len() as u64).to_le_bytes());
                for s in values {
                    string(&mut out, &s);
                }
            }
        }
    }
    let mut offset = 0u64;
    for (name, shape, dtype) in tensors {
        string(&mut out, &name);
        out.extend((shape.len() as u32).to_le_bytes());
        for n in &shape {
            out.extend(n.to_le_bytes());
        }
        out.extend((dtype as u32).to_le_bytes());
        out.extend(offset.to_le_bytes());
        let (block, bytes) = dtype.storage_layout().unwrap();
        offset = (offset + shape.iter().product::<u64>() / block * bytes).next_multiple_of(32);
    }
    out.resize(out.len().next_multiple_of(32), 0);
    let path = std::env::temp_dir().join(format!(
        "v41-cli-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    file.write_all(&out).unwrap();
    file.set_len(out.len() as u64 + offset).unwrap();
    GgufFile::from_opened_file(file, "v41-synthetic.gguf").unwrap()
}
