use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    pub(crate) content: String,
    #[serde(default, flatten)]
    #[allow(dead_code)]
    pub(crate) extra: BTreeMap<String, serde_json::Value>,
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
    let (messages, meta) = load_messages_input(path, max_messages)?;
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

#[allow(dead_code)]
pub(crate) fn load_deepseek_v4_0731_messages_prompt(
    path: &Path,
    max_messages: Option<usize>,
) -> Result<String> {
    let (messages, meta) = load_messages_input(path, max_messages)?;
    validate_deepseek_v4_0731_wrapper_metadata(&meta)?;
    render_deepseek_v4_0731_messages_prompt(&messages)
}

fn validate_deepseek_v4_0731_wrapper_metadata(meta: &serde_json::Value) -> Result<()> {
    match meta {
        serde_json::Value::Null => Ok(()),
        serde_json::Value::Object(fields) if fields.is_empty() => Ok(()),
        serde_json::Value::Object(fields) => bail!(
            "DeepSeek V4 messages have unsupported wrapper fields: {}",
            fields
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => bail!("DeepSeek V4 messages have malformed wrapper metadata"),
    }
}

fn load_messages_input(
    path: &Path,
    max_messages: Option<usize>,
) -> Result<(Vec<ChatMessage>, serde_json::Value)> {
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
    Ok((messages, meta))
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

/// Render the ordinary chat subset of the DeepSeek V4 release encoder.
///
/// This intentionally excludes reasoning records, tools, developer messages,
/// and task modes until their richer schemas have independent byte fixtures.
#[allow(dead_code)]
pub(crate) fn render_deepseek_v4_0731_messages_prompt(messages: &[ChatMessage]) -> Result<String> {
    if messages.is_empty() {
        bail!("DeepSeek V4 messages contain no messages");
    }

    let mut output = String::from("<｜begin▁of▁sentence｜>");
    let mut expect_user = true;
    let mut saw_user = false;

    for (index, message) in messages.iter().enumerate() {
        if !message.extra.is_empty() {
            bail!(
                "DeepSeek V4 message {index} has unsupported fields: {}",
                message
                    .extra
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        match message.role.as_str() {
            "system" if index == 0 && expect_user => {
                output.push_str(&message.content);
            }
            "user" if expect_user => {
                output.push_str("<｜User｜>");
                output.push_str(&message.content);
                // In the release encoder the assistant transition belongs to
                // the preceding user turn, including historical turns.
                output.push_str("<｜Assistant｜></think>");
                expect_user = false;
                saw_user = true;
            }
            "assistant" if !expect_user => {
                output.push_str(&message.content);
                output.push_str("<｜end▁of▁sentence｜>");
                expect_user = true;
            }
            role => {
                let expected = if expect_user { "user" } else { "assistant" };
                bail!(
                    "DeepSeek V4 message {index} has role {role:?}; expected {expected:?} in the ordinary 0731 chat subset"
                );
            }
        }
    }

    if !saw_user {
        bail!("DeepSeek V4 messages require at least one user turn");
    }
    if expect_user {
        bail!("DeepSeek V4 messages must end with a user turn before generation");
    }
    Ok(output)
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

    fn message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            extra: BTreeMap::new(),
        }
    }

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
            message("user", "hi"),
            message("assistant", "<think>hidden</think>shown"),
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

    #[test]
    fn deepseek_v4_ordinary_chat_matches_release_derived_references() {
        assert_eq!(
            render_deepseek_v4_0731_messages_prompt(&[message("user", "Hello")]).unwrap(),
            "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
        );
        assert_eq!(
            render_deepseek_v4_0731_messages_prompt(&[
                message("system", "Be exact."),
                message("user", "Hello"),
            ])
            .unwrap(),
            "<｜begin▁of▁sentence｜>Be exact.<｜User｜>Hello<｜Assistant｜></think>"
        );
        assert_eq!(
            render_deepseek_v4_0731_messages_prompt(&[
                message("system", "Be exact."),
                message("user", "Hello"),
                message("assistant", "Hi!"),
                message("user", "上海 🙂"),
            ])
            .unwrap(),
            concat!(
                "<｜begin▁of▁sentence｜>Be exact.",
                "<｜User｜>Hello<｜Assistant｜></think>",
                "Hi!<｜end▁of▁sentence｜>",
                "<｜User｜>上海 🙂<｜Assistant｜></think>"
            )
        );
    }

    #[test]
    fn deepseek_v4_ordinary_chat_rejects_unrepresented_semantics() {
        for messages in [
            vec![message("assistant", "hello")],
            vec![message("user", "one"), message("user", "two")],
            vec![message("user", "hello"), message("assistant", "done")],
            vec![message("developer", "hello")],
            vec![message("system", "one"), message("system", "two")],
        ] {
            assert!(render_deepseek_v4_0731_messages_prompt(&messages).is_err());
        }

        let (messages, _) = parse_messages_input(json!([
            {"role": "user", "content": "hello", "reasoning": "hidden"}
        ]))
        .unwrap();
        let error = render_deepseek_v4_0731_messages_prompt(&messages)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsupported fields: reasoning"));
    }

    #[test]
    fn deepseek_v4_loader_rejects_wrapper_semantics_and_pins_truncation() {
        let path = std::env::temp_dir().join(format!(
            "qwen-dsv4-message-policy-{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"{
                "messages": [{"role":"user","content":"hello"}],
                "meta": {"reasoning_effort":"max"},
                "tools": []
            }"#,
        )
        .unwrap();
        let error = load_deepseek_v4_0731_messages_prompt(&path, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("reasoning_effort"));
        assert!(error.contains("tools"));

        std::fs::write(
            &path,
            r#"{
                "messages": [
                    {"role":"user","content":"one"},
                    {"role":"assistant","content":"done"},
                    {"role":"user","content":"two"},
                    {"role":"assistant","content":"hidden","reasoning":"not represented"}
                ]
            }"#,
        )
        .unwrap();
        assert!(load_deepseek_v4_0731_messages_prompt(&path, Some(1)).is_ok());
        let error = load_deepseek_v4_0731_messages_prompt(&path, Some(2))
            .unwrap_err()
            .to_string();
        assert!(error.contains("must end with a user turn"));
        assert!(load_deepseek_v4_0731_messages_prompt(&path, Some(3)).is_ok());
        let error = load_deepseek_v4_0731_messages_prompt(&path, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsupported fields: reasoning"));
        std::fs::remove_file(path).unwrap();
    }
}
