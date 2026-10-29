use qwen_llm::{gguf::GgufFile, tokenizer::NativeTokenizer};
use std::io::Write;

pub(super) const EOS: u32 = 256;
pub(super) const EMISSION: u32 = 257;

pub(super) fn tokenizer(emission: Option<&str>) -> NativeTokenizer {
    fn string(out: &mut Vec<u8>, text: &str) {
        out.extend((text.len() as u64).to_le_bytes());
        out.extend(text.as_bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(8u64.to_le_bytes());
    for (key, value) in [
        ("general.architecture", "deepseek41"),
        ("tokenizer.ggml.model", "gpt2"),
        ("tokenizer.ggml.pre", "joyai-llm"),
    ] {
        string(&mut out, key);
        out.extend(8u32.to_le_bytes());
        string(&mut out, value);
    }
    let mut extra = 256;
    let mut tokens: Vec<_> = (0..256u32)
        .map(|byte| {
            let code = if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174 {
                byte
            } else {
                extra += 1;
                extra - 1
            };
            char::from_u32(code).unwrap().to_string()
        })
        .collect();
    tokens.push(deepseek_recipe_encoding::v4::EOS_TOKEN.into());
    if let Some(emission) = emission {
        tokens.push(emission.into());
    }
    for (key, values) in [
        ("tokenizer.ggml.tokens", tokens.as_slice()),
        ("tokenizer.ggml.merges", &[][..]),
    ] {
        string(&mut out, key);
        out.extend(9u32.to_le_bytes());
        out.extend(8u32.to_le_bytes());
        out.extend((values.len() as u64).to_le_bytes());
        for value in values {
            string(&mut out, value);
        }
    }
    string(&mut out, "tokenizer.ggml.token_type");
    out.extend(9u32.to_le_bytes());
    out.extend(5u32.to_le_bytes());
    out.extend((tokens.len() as u64).to_le_bytes());
    for id in 0..tokens.len() {
        out.extend((if id < 256 { 1i32 } else { 3i32 }).to_le_bytes());
    }
    string(&mut out, "tokenizer.ggml.eos_token_id");
    out.extend(4u32.to_le_bytes());
    out.extend(EOS.to_le_bytes());
    string(&mut out, "tokenizer.ggml.bos_token_id");
    out.extend(4u32.to_le_bytes());
    out.extend(EOS.to_le_bytes());
    out.resize(out.len().next_multiple_of(32), 0);
    // Parallel tests can read the same clock value; the counter keeps
    // `create_new` from colliding within one process.
    static FIXTURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "v41-native-tokenizer-{}-{}-{}.gguf",
        std::process::id(),
        FIXTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
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
    std::fs::remove_file(path).unwrap();
    file.write_all(&out).unwrap();
    let gguf = GgufFile::from_opened_file(file, "v41-native-tokenizer-fixture.gguf").unwrap();
    NativeTokenizer::from_gguf(&gguf).unwrap()
}

#[test]
fn fixture_uses_native_joyai_encoding_and_exact_piece_bytes() {
    let tokenizer = tokenizer(Some("answer"));
    assert_eq!(tokenizer.n_vocab(), 258);
    assert_eq!(
        tokenizer.encode("prompt", false).unwrap(),
        [112, 114, 111, 109, 112, 116]
    );
    assert_eq!(
        tokenizer
            .encode(deepseek_recipe_encoding::v4::EOS_TOKEN, false)
            .unwrap(),
        [EOS as i32]
    );
    assert_eq!(
        tokenizer
            .try_decode_piece_bytes_exact(EMISSION as i32)
            .unwrap(),
        b"answer"
    );
    assert_eq!(
        tokenizer.try_decode_piece_bytes_exact(0xe2).unwrap(),
        &[0xe2]
    );
}
