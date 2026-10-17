use super::Result;
use std::collections::HashMap;

/// Dependency-free interning half of upstream build_compressed_token_map.
/// Callbacks must use the official tokenizers backend:
/// decode([id], skip_special_tokens=false), id_to_token(id), and normalization:
/// NFKC -> NFD -> StripAccents -> Lowercase -> replace [ \t\r\n]+ with space
/// -> replace ^ $ with U+E000 -> Strip -> replace U+E000 with space.
/// Unicode normalization is deliberately delegated, never approximated by ASCII.
pub fn build_compressed_token_map(
    vocab: usize,
    mut decode: impl FnMut(u32) -> Result<String>,
    mut raw_token: impl FnMut(u32) -> Result<String>,
    mut normalize: impl FnMut(&str) -> Result<String>,
) -> Result<(Vec<u32>, usize)> {
    if vocab > u32::MAX as usize {
        return Err("tokenizer vocabulary too large".into());
    }
    let mut keys = HashMap::new();
    let mut result = Vec::with_capacity(vocab);
    for id in 0..vocab as u32 {
        let text = decode(id)?;
        let key = if text.contains('\u{fffd}') {
            raw_token(id)?
        } else {
            let normalized = normalize(&text)?;
            if normalized.is_empty() {
                text
            } else {
                normalized
            }
        };
        let next = keys.len() as u32;
        result.push(*keys.entry(key).or_insert(next));
    }
    Ok((result, keys.len()))
}
