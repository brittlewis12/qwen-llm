//! Offline oracle helper; GGUF runtime consumes the embedded Engram token map.
//! Tokenizers-native port of build_compressed_token_map in upstream inference/engram.py.
//! Pinned source: scripts/reference/deepseek_v41/fixtures/inference/engram.py.
use anyhow::{Context, Result, anyhow, ensure};
use std::collections::HashMap;
use tokenizers::{NormalizedString, Normalizer, Tokenizer, normalizers::NormalizerWrapper};

pub struct CompressedTokenMap {
    /// Raw token ID -> compressed ID, in ascending raw ID order.
    pub lookup: Vec<u32>,
    pub vocab_size: u32,
    pub pad_id: u32,
}

fn normalizer() -> Result<NormalizerWrapper> {
    Ok(serde_json::from_value(
        serde_json::json!({"type":"Sequence","normalizers":[
            {"type":"NFKC"}, {"type":"NFD"}, {"type":"StripAccents"}, {"type":"Lowercase"},
            {"type":"Replace","pattern":{"Regex":"[ \t\r\n]+"},"content":" "},
            {"type":"Replace","pattern":{"Regex":"^ $"},"content":"\u{e000}"},
            {"type":"Strip","strip_left":true,"strip_right":true},
            {"type":"Replace","pattern":{"String":"\u{e000}"},"content":" "}
        ]}),
    )?)
}

pub fn build_compressed_token_map(tokenizer: &Tokenizer) -> Result<(Vec<u32>, u32)> {
    let normalizer = normalizer()?;
    let size = tokenizer.get_vocab_size(true);
    ensure!(size <= u32::MAX as usize, "tokenizer vocabulary too large");
    let mut keys = HashMap::new();
    let mut lookup = Vec::with_capacity(size);
    for id in 0..size as u32 {
        if id % 1024 == 0 {
            crate::shutdown::checkpoint()?;
        }
        let raw = tokenizer
            .id_to_token(id)
            .with_context(|| format!("tokenizer vocabulary has a hole at {id}"))?;
        let text = tokenizer
            .decode(&[id], false)
            .map_err(|e| anyhow!(e.to_string()))?;
        let key = if text.contains('\u{fffd}') {
            raw
        } else {
            let mut normalized = NormalizedString::from(text.as_str());
            normalizer
                .normalize(&mut normalized)
                .map_err(|e| anyhow!(e.to_string()))?;
            if normalized.get().is_empty() {
                text
            } else {
                normalized.get().to_owned()
            }
        };
        let next = keys.len() as u32;
        lookup.push(*keys.entry(key).or_insert(next));
    }
    Ok((lookup, keys.len() as u32))
}

impl CompressedTokenMap {
    /// Forward receives this lookup and compressed pad, never raw pad token 2.
    pub fn for_config(tokenizer: &Tokenizer, config: &serde_json::Value) -> Result<Self> {
        let text = &config["text_config"];
        let (lookup, vocab_size) = build_compressed_token_map(tokenizer)?;
        ensure!(
            text["vocab_size"].as_u64() == Some(lookup.len() as u64),
            "raw tokenizer/config vocabulary mismatch"
        );
        ensure!(
            text["engram_compressed_vocab_size"].as_u64() == Some(vocab_size as u64),
            "compressed tokenizer/config vocabulary mismatch: {vocab_size}"
        );
        ensure!(
            vocab_size == qwen_llm::deepseek_v41::engram::COMPRESSED_VOCAB_SIZE,
            "compressed vocabulary does not match pinned hash multipliers"
        );
        let pad = text["engram_pad_token_id"]
            .as_u64()
            .context("missing engram_pad_token_id")?;
        let pad_id = *lookup
            .get(usize::try_from(pad)?)
            .context("Engram pad outside raw vocabulary")?;
        Ok(Self {
            lookup,
            vocab_size,
            pad_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENIZER: qwen_llm::test_fixtures::Fixture = qwen_llm::test_fixtures::Fixture {
        id: "deepseek-v4.1-reap-272e-tokenizer",
        env: &[],
        filename: "DeepSeek-V4.1-Flash-REAP-272E-tokenizer.json",
    };

    #[test]
    #[ignore = "model witness: deepseek-v4.1-reap-272e-tokenizer"]
    fn v41_full_tokenizer_matches_python_digest() {
        use sha2::{Digest, Sha256};
        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/v41_engram_full_summary.json"
        ))
        .unwrap();
        let bytes = std::fs::read(TOKENIZER.required()).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            expected["tokenizer_sha256"].as_str().unwrap()
        );
        let tokenizer = Tokenizer::from_bytes(bytes).unwrap();
        let config = serde_json::json!({"text_config": {"vocab_size":129280, "engram_compressed_vocab_size":99092, "engram_pad_token_id":2}});
        let map = CompressedTokenMap::for_config(&tokenizer, &config).unwrap();
        let mut hash = Sha256::new();
        for id in &map.lookup {
            hash.update(id.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", hash.finalize()),
            expected["lookup_u32le_sha256"].as_str().unwrap()
        );
        assert_eq!(
            map.lookup.len() as u64,
            expected["raw_vocab"].as_u64().unwrap()
        );
        assert_eq!(
            map.vocab_size as u64,
            expected["compressed_vocab"].as_u64().unwrap()
        );
        assert_eq!(
            map.pad_id as u64,
            expected["compressed_pad_id"].as_u64().unwrap()
        );
    }
    #[test]
    fn v41_engram_matches_upstream_python_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/v41_engram_token_map.json"))
                .unwrap();
        let tokenizer =
            Tokenizer::from_bytes(serde_json::to_vec(&fixture["tokenizer"]).unwrap()).unwrap();
        let (lookup, size) = build_compressed_token_map(&tokenizer).unwrap();
        assert_eq!(serde_json::json!(lookup), fixture["lookup"]);
        assert_eq!(
            size,
            fixture["compressed_vocab_size"].as_u64().unwrap() as u32
        );
    }
}
