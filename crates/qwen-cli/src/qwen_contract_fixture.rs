//! Header-only CPU fixture; no weights, usable tokenizer, or engine claim.
use qwen_llm::gguf::GgufFile;
use std::io::Write;

pub(crate) const TEMPLATE: &str =
    include_str!("../tests/fixtures/templates/qwen_xml_preserve_chat_template.jinja");

#[derive(Clone)]
pub(crate) enum Value<'a> {
    Text(&'a str),
    Float(f64),
    Signed(i64),
    Unsigned(u64),
    UnsignedArray(Vec<u64>),
}

fn string(out: &mut Vec<u8>, text: &str) {
    out.extend((text.len() as u64).to_le_bytes());
    out.extend(text.as_bytes());
}

pub(crate) fn temp_file() -> (std::path::PathBuf, std::fs::File) {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "qwen-contract-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ));
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    (path, file)
}

pub(crate) fn gguf(name: &str, template: &str, extra: &[(&str, Value<'_>)]) -> GgufFile {
    use Value::*;
    let mut entries = vec![
        ("general.architecture", Text("qwen35moe")),
        ("general.name", Text(name)),
        ("general.basename", Text(name)),
        ("tokenizer.ggml.model", Text("gpt2")),
        ("tokenizer.ggml.pre", Text("qwen35")),
        ("tokenizer.chat_template", Text(template)),
        ("qwen35moe.block_count", Unsigned(48)),
        ("qwen35moe.embedding_length", Unsigned(3072)),
        ("qwen35moe.expert_count", Unsigned(256)),
        ("qwen35moe.expert_used_count", Unsigned(8)),
        ("qwen35moe.context_length", Unsigned(32768)),
    ];
    entries.retain(|(key, _)| !extra.iter().any(|(other, _)| key == other));
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(((entries.len() + extra.len() + 1) as u64).to_le_bytes());
    for (key, value) in entries.iter().chain(extra) {
        string(&mut bytes, key);
        match value {
            Text(value) => {
                bytes.extend(8u32.to_le_bytes());
                string(&mut bytes, value);
            }
            Float(value) => {
                bytes.extend(12u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
            Signed(value) => {
                bytes.extend(11u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
            Unsigned(value) => {
                bytes.extend(10u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
            UnsignedArray(values) => {
                bytes.extend(9u32.to_le_bytes());
                bytes.extend(10u32.to_le_bytes());
                bytes.extend((values.len() as u64).to_le_bytes());
                for value in values {
                    bytes.extend(value.to_le_bytes());
                }
            }
        }
    }
    string(&mut bytes, "tokenizer.ggml.tokens");
    bytes.extend(9u32.to_le_bytes());
    bytes.extend(8u32.to_le_bytes());
    bytes.extend(248_320u64.to_le_bytes());
    for _ in 0..248_320 {
        string(&mut bytes, "");
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    let (path, mut file) = temp_file();
    std::fs::remove_file(path).unwrap();
    file.write_all(&bytes).unwrap();
    GgufFile::from_opened_file(file, "qwen-contract-header.gguf").unwrap()
}

/// Minimal Qwen4 alias header. With `control_tokens`, ids 248044..=248046 hold
/// the Qwen control strings and EOS points at `<|im_end|>`; otherwise the
/// vocabulary is the same length but carries no usable control mapping.
pub(crate) fn qwen4_gguf(
    architecture: &str,
    name: &str,
    template: &str,
    control_tokens: bool,
) -> GgufFile {
    qwen4_gguf_with_stops(architecture, name, template, control_tokens, 248_046, None)
}

pub(crate) fn qwen4_gguf_with_stops(
    architecture: &str,
    name: &str,
    template: &str,
    control_tokens: bool,
    eos: u64,
    eot: Option<u64>,
) -> GgufFile {
    qwen4_gguf_inner(
        architecture,
        name,
        template,
        control_tokens,
        eos,
        eot,
        false,
    )
}

pub(crate) fn qwen4_tokenizer_gguf(architecture: &str, template: &str) -> GgufFile {
    qwen4_gguf_inner(
        architecture,
        "future Qwen",
        template,
        true,
        248_046,
        None,
        true,
    )
}

fn qwen4_gguf_inner(
    architecture: &str,
    name: &str,
    template: &str,
    control_tokens: bool,
    eos: u64,
    eot: Option<u64>,
    usable: bool,
) -> GgufFile {
    use Value::*;
    let mut entries: Vec<(&str, Value<'_>)> = vec![
        ("general.architecture", Text(architecture)),
        ("general.name", Text(name)),
        ("general.basename", Text(name)),
        ("tokenizer.ggml.model", Text("gpt2")),
        ("tokenizer.ggml.pre", Text("qwen35")),
        ("tokenizer.chat_template", Text(template)),
    ];
    if control_tokens {
        entries.push(("tokenizer.ggml.eos_token_id", Unsigned(eos)));
        if let Some(eot) = eot {
            entries.push(("tokenizer.ggml.eot_token_id", Unsigned(eot)));
        }
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(((entries.len() + 1 + 2 * usize::from(usable)) as u64).to_le_bytes());
    for (key, value) in &entries {
        string(&mut bytes, key);
        match value {
            Text(value) => {
                bytes.extend(8u32.to_le_bytes());
                string(&mut bytes, value);
            }
            Float(value) => {
                bytes.extend(12u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
            Signed(value) => {
                bytes.extend(11u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
            Unsigned(value) => {
                bytes.extend(10u32.to_le_bytes());
                bytes.extend(value.to_le_bytes());
            }
            UnsignedArray(values) => {
                bytes.extend(9u32.to_le_bytes());
                bytes.extend(10u32.to_le_bytes());
                bytes.extend((values.len() as u64).to_le_bytes());
                for value in values {
                    bytes.extend(value.to_le_bytes());
                }
            }
        }
    }
    string(&mut bytes, "tokenizer.ggml.tokens");
    bytes.extend(9u32.to_le_bytes());
    bytes.extend(8u32.to_le_bytes());
    bytes.extend(248_320u64.to_le_bytes());
    for id in 0..248_320u64 {
        let text = if control_tokens {
            match id {
                248_044 => "<|endoftext|>",
                248_045 => "<|im_start|>",
                248_046 => "<|im_end|>",
                _ => "",
            }
        } else {
            ""
        };
        if usable && id < 256 {
            let b = id as u8;
            let ordinary = |b: u8| matches!(b,33..=126 | 161..=172 | 174..=255);
            let c = if ordinary(b) {
                u32::from(b)
            } else {
                256 + (0..b).filter(|b| !ordinary(*b)).count() as u32
            };
            string(&mut bytes, &char::from_u32(c).unwrap().to_string());
        } else if usable && text.is_empty() {
            string(&mut bytes, &format!("<unused_{id}>"));
        } else {
            string(&mut bytes, text);
        }
    }
    if usable {
        string(&mut bytes, "tokenizer.ggml.token_type");
        bytes.extend(9u32.to_le_bytes());
        bytes.extend(5u32.to_le_bytes());
        bytes.extend(248_320u64.to_le_bytes());
        for id in 0..248_320 {
            bytes.extend(
                (if (248_044..=248_046).contains(&id) {
                    3i32
                } else {
                    1i32
                })
                .to_le_bytes(),
            );
        }
        string(&mut bytes, "tokenizer.ggml.merges");
        bytes.extend(9u32.to_le_bytes());
        bytes.extend(8u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    let (path, mut file) = temp_file();
    std::fs::remove_file(path).unwrap();
    file.write_all(&bytes).unwrap();
    GgufFile::from_opened_file(file, "qwen4-alias-header.gguf").unwrap()
}

/// Complete reference-profile metadata for testing production release detection.
pub(crate) fn qwen4_reference_gguf(
    architecture: &str,
    name: &str,
    mutate: impl FnOnce(&mut qwen_llm::qwen4exp::Qwen4ExpConfig),
) -> GgufFile {
    qwen4_reference_gguf_with_extra(architecture, name, mutate, &[])
}

pub(crate) fn qwen4_reference_gguf_with_extra(
    architecture: &str,
    name: &str,
    mutate: impl FnOnce(&mut qwen_llm::qwen4exp::Qwen4ExpConfig),
    extra: &[(&str, Value<'_>)],
) -> GgufFile {
    use Value::*;
    let mut c = qwen_llm::qwen4exp::Qwen4ExpConfig::flash_next_reference();
    mutate(&mut c);
    let ple = c.ple.as_ref().unwrap();
    let mut fields = vec![("general.architecture".into(), Text(architecture))];
    for (key, value) in [
        ("context_length", c.context_length),
        ("block_count", c.layer_count),
        ("embedding_length", c.hidden_size),
        ("full_attention_interval", c.full_attention_interval),
        ("hyper_connection.count", c.hyper_connection.count),
        ("hyper_connection.low_rank", c.hyper_connection.low_rank),
        ("attention.head_count", c.attention.query_heads),
        ("attention.head_count_kv", c.attention.kv_heads),
        ("attention.key_length", c.attention.key_head_dim),
        ("attention.value_length", c.attention.value_head_dim),
        ("rope.dimension_count", c.attention.rotary_dim),
        ("ssm.group_count", c.gated_delta_net.key_heads),
        ("ssm.time_step_rank", c.gated_delta_net.value_heads),
        ("ssm.state_size", c.gated_delta_net.key_head_dim),
        ("ssm.conv_kernel", c.gated_delta_net.conv_kernel),
        ("ssm.inner_size", c.gated_delta_net.inner_size),
        ("expert_count", c.moe.expert_count),
        ("expert_used_count", c.moe.experts_per_token),
        ("expert_feed_forward_length", c.moe.expert_intermediate_size),
        (
            "expert_shared_feed_forward_length",
            c.moe.shared_expert_intermediate_size,
        ),
        ("attention.indexer.head_count", c.qsa.query_heads),
        ("attention.indexer.key_length", c.qsa.head_dim),
        ("attention.indexer.top_k", c.qsa.token_budget),
        ("ple.ngram_size", ple.ngram_size),
        ("ple.heads_per_ngram", ple.heads_per_ngram),
        ("embedding_length_per_layer_input", ple.embedding_head_dim),
        ("ple.conv_kernel", ple.conv_kernel),
        ("ple.eos_token_id", ple.eos_token_id),
        ("ple.image_token_id", ple.image_token_id.unwrap()),
        ("nextn_predict_layers", 0),
    ] {
        fields.push((format!("{architecture}.{key}"), Unsigned(u64::from(value))));
    }
    for (key, value) in [
        ("attention.layer_norm_rms_epsilon", c.rms_norm_eps),
        ("rope.freq_base", c.attention.rope_theta),
    ] {
        fields.push((format!("{architecture}.{key}"), Float(f64::from(value))));
    }
    for (key, values) in [
        (
            "rope.dimension_sections",
            c.attention.rope_sections.map(u64::from).to_vec(),
        ),
        (
            "attention.compress_ratios",
            c.compress_ratios.iter().copied().map(u64::from).collect(),
        ),
        (
            "ple.layers",
            c.ple
                .as_ref()
                .map(|ple| ple.layers.iter().copied().map(u64::from).collect())
                .unwrap_or_default(),
        ),
        ("ple.layer_multipliers", ple.multipliers.clone()),
        ("ple.head_offsets", ple.head_offsets.clone()),
        ("ple.head_vocab_sizes", ple.head_vocab_sizes.clone()),
    ] {
        fields.push((format!("{architecture}.{key}"), UnsignedArray(values)));
    }
    fields.retain(|(key, _)| !extra.iter().any(|(other, _)| key == other));
    fields.extend(
        extra
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone())),
    );
    let extra: Vec<_> = fields
        .iter()
        .map(|(key, value)| (key.as_str(), value.clone()))
        .collect();
    gguf(name, TEMPLATE, &extra)
}
