//! Header-only CPU fixture; no weights, usable tokenizer, or engine claim.
use qwen_llm::gguf::GgufFile;
use std::io::Write;

pub(crate) const TEMPLATE: &str =
    include_str!("../tests/fixtures/templates/qwen_xml_preserve_chat_template.jinja");

pub(crate) enum Value<'a> {
    Text(&'a str),
    Float(f64),
    Signed(i64),
    Unsigned(u64),
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
    bytes.extend(((entries.len() + 1) as u64).to_le_bytes());
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
        string(&mut bytes, text);
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    let (path, mut file) = temp_file();
    std::fs::remove_file(path).unwrap();
    file.write_all(&bytes).unwrap();
    GgufFile::from_opened_file(file, "qwen4-alias-header.gguf").unwrap()
}
