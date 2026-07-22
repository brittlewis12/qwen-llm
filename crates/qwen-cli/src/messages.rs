use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    pub(crate) content: String,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum MessagesThinkingMode {
    Auto,
    Preserve,
    Strip,
}

pub(crate) fn messages_thinking_mode(preserve: bool, strip: bool) -> MessagesThinkingMode {
    if preserve {
        MessagesThinkingMode::Preserve
    } else if strip {
        MessagesThinkingMode::Strip
    } else {
        MessagesThinkingMode::Auto
    }
}

#[allow(dead_code)]
pub(crate) fn load_messages_prompt(
    path: &Path,
    max_messages: Option<usize>,
    thinking_mode: MessagesThinkingMode,
    append_generation_prompt: bool,
) -> Result<String> {
    load_messages_prompt_with_policy(path, max_messages, thinking_mode, append_generation_prompt)
        .map(|(prompt, _)| prompt)
}

pub(crate) fn load_messages_prompt_with_policy(
    path: &Path,
    max_messages: Option<usize>,
    thinking_mode: MessagesThinkingMode,
    append_generation_prompt: bool,
) -> Result<(String, bool)> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse messages input {}", path.display()))?;
    let (mut messages, meta) = parse_messages_input(value)?;
    if let Some(max) = max_messages {
        messages.truncate(max);
    }
    if messages.is_empty() {
        bail!("messages input {} contains no messages", path.display());
    }
    let preserve_thinking = match thinking_mode {
        MessagesThinkingMode::Preserve => true,
        MessagesThinkingMode::Strip => false,
        MessagesThinkingMode::Auto => messages_auto_preserve_thinking(&meta),
    };
    Ok((
        render_qwen_messages_prompt(&messages, preserve_thinking, append_generation_prompt),
        preserve_thinking,
    ))
}

pub(crate) fn messages_auto_preserve_thinking(meta: &serde_json::Value) -> bool {
    if meta
        .get("preserve_thinking")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
    {
        return true;
    }
    meta.get("model")
        .and_then(|value| value.as_str())
        .map(|model| model.to_ascii_lowercase().contains("qwen3.6"))
        .unwrap_or(false)
}

pub(crate) fn parse_messages_input(
    value: serde_json::Value,
) -> Result<(Vec<ChatMessage>, serde_json::Value)> {
    match value {
        serde_json::Value::Array(_) => {
            let messages = serde_json::from_value(value).context("parse bare messages array")?;
            Ok((messages, serde_json::Value::Null))
        }
        serde_json::Value::Object(mut object) => {
            let messages_value = object.remove("messages").ok_or_else(|| {
                anyhow!("wrapped messages input must contain a top-level `messages` array")
            })?;
            let messages =
                serde_json::from_value(messages_value).context("parse wrapped messages array")?;

            let mut merged = serde_json::Map::new();
            if let Some(meta_value) = object.remove("meta") {
                match meta_value {
                    serde_json::Value::Object(map) => merged.extend(map),
                    serde_json::Value::Null => {}
                    other => {
                        merged.insert("meta".into(), other);
                    }
                }
            }
            merged.extend(object);
            Ok((messages, serde_json::Value::Object(merged)))
        }
        other => Err(anyhow!(
            "messages input must be a message array or wrapped object, got {other}"
        )),
    }
}

pub(crate) fn render_qwen_messages_prompt(
    messages: &[ChatMessage],
    preserve_thinking: bool,
    append_generation_prompt: bool,
) -> String {
    let mut output = String::new();
    for message in messages {
        output.push_str("<|im_start|>");
        output.push_str(&message.role);
        output.push('\n');
        if message.role == "assistant" && !preserve_thinking {
            output.push_str(&strip_think(&message.content));
        } else {
            output.push_str(&message.content);
        }
        output.push_str("<|im_end|>\n");
    }
    if append_generation_prompt {
        output.push_str("<|im_start|>assistant\n");
    }
    output
}

pub(crate) fn strip_think(text: &str) -> String {
    let trimmed = text.trim_start();
    if let Some(rest) = trimmed.strip_prefix("<think>")
        && let Some((_, tail)) = rest.split_once("</think>")
    {
        return tail.trim().to_string();
    }
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn auto_preserves_thinking_only_for_qwen36() {
        assert!(messages_auto_preserve_thinking(
            &json!({ "model": "/models/Qwen3.6-27B-Q4_K_M.gguf" })
        ));
        assert!(messages_auto_preserve_thinking(
            &json!({ "preserve_thinking": true, "model": "anything" })
        ));
        assert!(!messages_auto_preserve_thinking(
            &json!({ "model": "/models/Qwen3.5-27B-Q4_K_M.gguf" })
        ));
        assert!(!messages_auto_preserve_thinking(&serde_json::Value::Null));
    }

    #[test]
    fn strip_think_only_strips_leading_qwen_block() {
        assert_eq!(strip_think("<think>hidden</think>shown"), "shown");
        assert_eq!(strip_think("plain text"), "plain text");
        assert_eq!(strip_think("  plain text  "), "  plain text  ");
        assert_eq!(
            strip_think("prefix </think> shown"),
            "prefix </think> shown"
        );
    }

    #[test]
    fn render_messages_prompt_respects_thinking_mode() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "<think>hidden</think>shown".into(),
            },
        ];
        let stripped = render_qwen_messages_prompt(&messages, false, true);
        let preserved = render_qwen_messages_prompt(&messages, true, true);
        assert!(stripped.contains("shown<|im_end|>"));
        assert!(!stripped.contains("hidden"));
        assert!(preserved.contains("<think>hidden</think>shown"));
        assert!(preserved.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn parse_messages_input_accepts_top_level_metadata() {
        let value = json!({
            "model": "/models/Qwen3.6-27B-Q4_K_M.gguf",
            "preserve_thinking": true,
            "messages": [
                {"role": "user", "content": "hi"}
            ]
        });
        let (messages, meta) = parse_messages_input(value).expect("parse messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(
            meta.get("model").and_then(|value| value.as_str()),
            Some("/models/Qwen3.6-27B-Q4_K_M.gguf")
        );
        assert_eq!(
            meta.get("preserve_thinking")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
    }
}
