//! Exact, bounded per-conversation CPU tokenization reuse. No inference state.
use super::checkpoint_ladder::{Conversation, conversation};
use super::items::{ServeError, ServeRequest};
use qwen_llm::tokenizer::{NativeTokenizer, PartitionedEncoding, TokError};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

qwen_llm::env_flag!(
    /// Reuse V4.1 full prompt tokenization at proven special-token cuts, with exact text-prefix verification and full-encode fallback. Four conversations / 8 MiB retained; default-off qualification switch, 0 restores full and boundary encodes.
    default_off enabled,
    "QWEN_SERVE_TOKEN_PREFIX_CACHE"
);
pub(super) fn is_enabled() -> bool {
    enabled()
}
pub(super) const MAX_BYTES: usize = 8 * 1024 * 1024;
struct Entry {
    key: Conversation,
    encoding: PartitionedEncoding,
}
pub(crate) struct Cache {
    native: Arc<NativeTokenizer>,
    entries: Mutex<VecDeque<Arc<Entry>>>,
}
impl Cache {
    pub(crate) fn new(native: Arc<NativeTokenizer>) -> Self {
        Self {
            native,
            entries: Mutex::default(),
        }
    }
    pub(crate) fn encode(
        &self,
        request: &ServeRequest,
        prompt: &str,
    ) -> Result<Vec<u32>, ServeError> {
        let matching = super::lifecycle_timing::Span::current("tokenize_prefix_match", 0, 0);
        let key = conversation(request);
        let previous = key.and_then(|key| {
            self.entries
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .find(|e| e.key == key)
                .cloned()
        });
        drop(matching);
        let encoded = self
            .native
            .encode_partitioned(prompt, previous.as_ref().map(|e| &e.encoding))
            .map_err(|e| match e {
                TokError::NativeInputTooLong { .. }
                | TokError::InputBytesTooLong(_)
                | TokError::InputTooLong(_) => {
                    ServeError::invalid_request(Some("input"), format!("V4.1 prompt: {e}"))
                }
                e => ServeError::server_error(format!("tokenize V4.1 prompt: {e}")),
            })?;
        let Some(encoded) = encoded else {
            return super::decode_loop::encode_checked(
                self.native.as_ref(),
                prompt,
                false,
                self.native.n_vocab(),
                "V4.1",
            );
        };
        if encoded.ids().is_empty() {
            return Err(ServeError::invalid_request(
                Some("input"),
                "V4.1 prompt tokenized to zero tokens",
            ));
        }
        let ids = encoded
            .ids()
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                crate::checked_token_id(id, self.native.n_vocab(), &format!("prompt[{i}]"))
                    .map_err(|e| ServeError::server_error(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let _count = super::lifecycle_timing::Span::current(
            "tokenize_reused_prefix",
            encoded.reused_tokens(),
            encoded.ids().len() - encoded.reused_tokens(),
        );
        if let Some(key) = key
            && encoded.memory_bytes() <= MAX_BYTES
        {
            let entry = Arc::new(Entry {
                key,
                encoding: encoded,
            });
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            entries.retain(|e| e.key != key);
            while entries.len() >= 4
                || entries
                    .iter()
                    .map(|e| e.encoding.memory_bytes())
                    .sum::<usize>()
                    + entry.encoding.memory_bytes()
                    > MAX_BYTES
            {
                entries.pop_front();
            }
            entries.push_back(entry);
        }
        Ok(ids)
    }
    pub(crate) fn prefix(&self, prefix: &str) -> Option<Vec<i32>> {
        // Never hold this lock while scanning text, encoding, or running GPU work.
        let entries: Vec<_> = self
            .entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect();
        entries
            .iter()
            .rev()
            .find_map(|e| e.encoding.exact_prefix(prefix).map(<[i32]>::to_vec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tokenizer() -> Arc<NativeTokenizer> {
        Arc::new(
            crate::serve::backend_v41::tokenizer_fixture::tokenizer_with_markers(&[
                "<|s|>",
                "<|s|>long",
                "<｜User｜>",
                "<｜Assistant｜>",
                deepseek_recipe_encoding::v4::ASSISTANT_SP_TOKEN,
                deepseek_recipe_encoding::v4::THINKING_START_TOKEN,
                deepseek_recipe_encoding::v4::THINKING_END_TOKEN,
                "</think>",
            ]),
        )
    }
    #[test]
    fn plan_transcript_properties_keep_ids_and_boundary_prefixes_exact() {
        let native = tokenizer();
        let csv = include_str!("../../../../docs/bench/2026-11-08-real-opencode-agent-calls.csv");
        let source = include_str!("transport.rs");
        let mut text = format!("<｜User｜>{}<｜Assistant｜></think>", &source[..4096]);
        let mut previous = None;
        let mut hits = 0;
        for (i, row) in csv
            .lines()
            .skip(1)
            .filter(|r| r.starts_with("v41/") && r.split(',').nth(4) == Some("primary"))
            .take(64)
            .enumerate()
        {
            let n = row
                .split(',')
                .nth(7)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap()
                .min(2048);
            if text.len() > 32768 {
                let mut end = 4096;
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                text.truncate(end);
                text.push_str("<｜Assistant｜></think>");
                previous = None;
            }
            let adversarial = [
                "ab",
                "e\u{301}👩‍💻",
                "<|s|>long",
                "<|s",
                "|>",
                "\r\n  ",
                "中\0x",
                "<｜Assistant｜>",
                deepseek_recipe_encoding::v4::ASSISTANT_SP_TOKEN,
                deepseek_recipe_encoding::v4::THINKING_START_TOKEN,
                deepseek_recipe_encoding::v4::THINKING_END_TOKEN,
            ][i % 8];
            text.push_str(&format!(
                "answer-{i}<|s|><｜User｜>{}{adversarial}<｜Assistant｜></think>",
                &source[..n]
            ));
            let current = native
                .encode_partitioned(&text, previous.as_ref())
                .unwrap()
                .unwrap();
            assert_eq!(
                current.ids(),
                native.encode(&text, false).unwrap(),
                "plan row {i}"
            );
            hits += usize::from(current.reused_tokens() > 0);
            for (at, _) in text.match_indices("<｜Assistant｜>") {
                assert_eq!(
                    current.exact_prefix(&text[..at]).unwrap(),
                    native.encode(&text[..at], false).unwrap()
                );
            }
            previous = Some(current);
        }
        assert!(hits > 10);
    }
    #[test]
    fn rendered_agent_plan_cache_matches_full_encoding_and_every_cut() {
        let native = tokenizer();
        let cache = Cache::new(Arc::clone(&native));
        let profile = super::super::request_profile::RequestProfile::DeepSeekV41 {
            limits: super::super::request_profile::OutputLimits::TEST,
            sampling: None,
        };
        let source = include_str!("transport.rs");
        let mut histories:Vec<Vec<serde_json::Value>>=(0..3).map(|i|vec![serde_json::json!({"role":"user","content":format!("Independent agent {i}: {}", &source[..4096])})]).collect();
        let csv = include_str!("../../../../docs/bench/2026-11-08-real-opencode-agent-calls.csv");
        for (i, row) in csv
            .lines()
            .skip(1)
            .filter(|r| r.starts_with("v41/") && r.split(',').nth(4) == Some("primary"))
            .take(48)
            .enumerate()
        {
            let c = i % 3;
            let n: usize = row.split(',').nth(7).unwrap().parse().unwrap();
            let mut n = n.min(2048);
            while !source.is_char_boundary(n) {
                n -= 1;
            }
            histories[c].push(
                serde_json::json!({"role":"assistant","content":"previous exact reply 😀 ab"}),
            );
            histories[c].push(serde_json::json!({"type":"function_call","call_id":format!("read{i}"),"name":"read_file","arguments":"{}"}));
            histories[c].push(serde_json::json!({"type":"function_call_output","call_id":format!("read{i}"),"output":format!("{} e\u{301}<|s|>long\r\n",&source[..n])}));
            let request=profile.parse(&serde_json::json!({"model":"test","input":histories[c],"tools":[{"type":"function","name":"read_file","parameters":{"type":"object","properties":{}}}]})).unwrap();
            let (prompt, b) = profile.render_prepared(&request).unwrap();
            assert_eq!(
                cache.encode(&request, &prompt).unwrap(),
                native
                    .encode(&prompt, false)
                    .unwrap()
                    .into_iter()
                    .map(|x| x as u32)
                    .collect::<Vec<_>>(),
                "rendered plan row {i}"
            );
            let b = b.unwrap();
            let at = b.generation_header_start.unwrap();
            assert_eq!(
                cache
                    .prefix(&prompt[..at])
                    .expect("recipe generation boundary is cached"),
                native.encode(&prompt[..at], false).unwrap()
            );
            for at in [b.shared_prefix_end, b.generation_header_start]
                .into_iter()
                .flatten()
            {
                if let Some(ids) = cache.prefix(&prompt[..at]) {
                    assert_eq!(ids, native.encode(&prompt[..at], false).unwrap());
                }
            }
        }
    }
    #[test]
    fn adversarial_extensions_forks_and_tokenizer_identity_fall_back_exactly() {
        let native = tokenizer();
        for (old, new) in [
            ("a", "ab"),
            ("x<|s|>", "x<|s|>long"),
            ("x<|s", "x<|s|>long"),
            ("x<|s|> tail", "x<|s|> tâche"),
            ("<|s|>ab<|s|>long", "<|s|>ac<|s|>long"),
            ("e\u{301}<|s|>x", "é<|s|>x"),
        ] {
            let old = native.encode_partitioned(old, None).unwrap().unwrap();
            let current = native.encode_partitioned(new, Some(&old)).unwrap().unwrap();
            assert_eq!(current.ids(), native.encode(new, false).unwrap());
            let other = tokenizer()
                .encode_partitioned(new, Some(&old))
                .unwrap()
                .unwrap();
            assert_eq!(other.reused_tokens(), 0);
            assert_eq!(other.ids(), current.ids());
        }
    }
    #[test]
    fn conversation_cache_is_bounded_and_exact_prefix_lookup_never_guesses() {
        let native = tokenizer();
        let cache = Cache::new(Arc::clone(&native));
        let profile = super::super::request_profile::RequestProfile::DeepSeekV41 {
            limits: super::super::request_profile::OutputLimits::TEST,
            sampling: None,
        };
        for i in 0..7 {
            let request = profile
                .parse(&serde_json::json!({"model":"test","input":format!("root-{i}")}))
                .unwrap();
            let text = format!("<｜User｜>root-{i}<｜Assistant｜></think>");
            assert_eq!(
                cache.encode(&request, &text).unwrap(),
                native
                    .encode(&text, false)
                    .unwrap()
                    .into_iter()
                    .map(|x| x as u32)
                    .collect::<Vec<_>>()
            );
        }
        let entries = cache.entries.lock().unwrap();
        assert_eq!(entries.len(), 4);
        assert!(
            entries
                .iter()
                .map(|e| e.encoding.memory_bytes())
                .sum::<usize>()
                <= MAX_BYTES
        );
        drop(entries);
        assert!(cache.prefix("<｜User｜>root-0<｜Assistant｜>").is_none());
        assert!(cache.prefix("<｜User｜>root-6<｜Assistant｜>").is_some());
        assert!(cache.prefix("<｜User｜>roo").is_none());
    }
}
