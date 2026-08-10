use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Default, Deserialize)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    pub(crate) content: String,
    /// DeepSeek V4 release-encoder assistant reasoning field (vLLM name).
    #[serde(default)]
    pub(crate) reasoning: Option<String>,
    /// OpenAI/SGLang-style alias for the same assistant reasoning field.
    #[serde(default)]
    pub(crate) reasoning_content: Option<String>,
    #[serde(default, flatten)]
    #[allow(dead_code)]
    pub(crate) extra: BTreeMap<String, serde_json::Value>,
}

/// DeepSeek V4 0731 reasoning selection for `--messages` encoding.
///
/// Follows the three-tier release contract (vLLM `77434861`): `None` is
/// chat mode; thinking tiers consult the effort-prompt table, where `Low`
/// contributes no bytes
/// (the release thinking default), `High` prepends the "Absolute maximum"
/// text (labeled max in the earlier two-tier encoders), and `Max` prepends the
/// stronger "Beyond maximum" text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// Shared across binaries; qwen-bench never constructs the thinking modes.
#[allow(dead_code)]
pub(crate) enum DeepSeekV4Reasoning {
    None,
    Low,
    High,
    Max,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DeepSeekV4EncodeOptions {
    pub(crate) reasoning: DeepSeekV4Reasoning,
    /// Maps to the release encoder's `drop_thinking=False`. The official
    /// trigger is declared tool schemas; this explicit knob exists for
    /// no-tools workloads that want interleaved reasoning retention.
    pub(crate) preserve_reasoning: bool,
}

impl Default for DeepSeekV4EncodeOptions {
    fn default() -> Self {
        Self {
            reasoning: DeepSeekV4Reasoning::None,
            preserve_reasoning: false,
        }
    }
}

const DEEPSEEK_V4_BOS: &str = "<｜begin▁of▁sentence｜>";
const DEEPSEEK_V4_EOS: &str = "<｜end▁of▁sentence｜>";
const DEEPSEEK_V4_USER: &str = "<｜User｜>";
const DEEPSEEK_V4_ASSISTANT: &str = "<｜Assistant｜>";
const DEEPSEEK_V4_THINK_START: &str = "<think>";
const DEEPSEEK_V4_THINK_END: &str = "</think>";
/// Byte-exact "high" effort instruction from the 0731 release contract
/// (vLLM `REASONING_EFFORT_PROMPTS["high"]` at `77434861`; identical bytes
/// appeared as the max-tier text in the earlier two-tier encoders, e.g. the
/// pinned SGLang revision).
const DEEPSEEK_V4_REASONING_EFFORT_HIGH: &str = concat!(
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n",
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n",
);
/// Byte-exact "max" effort instruction from the 0731 release contract
/// (vLLM `REASONING_EFFORT_PROMPTS["max"]` at `77434861`).
const DEEPSEEK_V4_REASONING_EFFORT_MAX: &str = concat!(
    "Reasoning Effort: Beyond maximum \u{2014} exhaustive, relentless, and uncompromising.\n",
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n",
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n",
);

impl DeepSeekV4Reasoning {
    /// Effort prompt contributed before the first message's content in
    /// thinking mode (vLLM `render_message` at `77434861`: consulted for
    /// every thinking-tier request; the low tier maps to the empty string).
    fn effort_prompt(self) -> &'static str {
        match self {
            DeepSeekV4Reasoning::None | DeepSeekV4Reasoning::Low => "",
            DeepSeekV4Reasoning::High => DEEPSEEK_V4_REASONING_EFFORT_HIGH,
            DeepSeekV4Reasoning::Max => DEEPSEEK_V4_REASONING_EFFORT_MAX,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum MessagesThinkingMode {
    Auto,
    Preserve,
    Strip,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum QwenGenerationMode {
    #[default]
    Auto,
    #[allow(dead_code)]
    Thinking,
    NoThinking,
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
    load_messages_prompt_with_policy_and_generation(
        path,
        max_messages,
        thinking_mode,
        append_generation_prompt,
        QwenGenerationMode::Auto,
    )
}

pub(crate) fn load_messages_prompt_with_policy_and_generation(
    path: &Path,
    max_messages: Option<usize>,
    thinking_mode: MessagesThinkingMode,
    append_generation_prompt: bool,
    generation_mode: QwenGenerationMode,
) -> Result<(String, bool)> {
    let (messages, meta) = load_messages_input(path, max_messages)?;
    render_loaded_qwen_messages(
        &messages,
        &meta,
        thinking_mode,
        append_generation_prompt,
        generation_mode,
    )
}

fn render_loaded_qwen_messages(
    messages: &[ChatMessage],
    meta: &serde_json::Value,
    thinking_mode: MessagesThinkingMode,
    append_generation_prompt: bool,
    generation_mode: QwenGenerationMode,
) -> Result<(String, bool)> {
    let preserve_thinking = match thinking_mode {
        MessagesThinkingMode::Preserve => true,
        MessagesThinkingMode::Strip => false,
        MessagesThinkingMode::Auto => messages_auto_preserve_thinking(meta),
    };
    let prompt = if generation_mode == QwenGenerationMode::Auto {
        render_qwen_messages_prompt(messages, preserve_thinking, append_generation_prompt)
    } else {
        render_qwen_messages_prompt_with_generation(
            messages,
            preserve_thinking,
            append_generation_prompt,
            generation_mode,
        )
    };
    Ok((prompt, preserve_thinking))
}

#[allow(dead_code)]
pub(crate) fn load_deepseek_v4_0731_messages_prompt(
    path: &Path,
    max_messages: Option<usize>,
    options: DeepSeekV4EncodeOptions,
) -> Result<String> {
    let (messages, meta) = load_messages_input(path, max_messages)?;
    validate_deepseek_v4_0731_wrapper_metadata(&meta)?;
    render_deepseek_v4_0731_messages_prompt(&messages, options)
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
    load_messages_input_from_str(&raw, &path.display().to_string(), max_messages)
}

fn load_messages_input_from_str(
    raw: &str,
    source: &str,
    max_messages: Option<usize>,
) -> Result<(Vec<ChatMessage>, serde_json::Value)> {
    let value: serde_json::Value =
        serde_json::from_str(raw).with_context(|| format!("parse messages input {source}"))?;
    let (mut messages, meta) = parse_messages_input(value)?;
    if let Some(max) = max_messages {
        messages.truncate(max);
    }
    if messages.is_empty() {
        bail!("messages input {source} contains no messages");
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct StrictChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct StrictMessagesWrapper {
    messages: Vec<StrictChatMessage>,
}

#[allow(dead_code)]
pub(crate) fn parse_strict_messages_input(raw: &str, source: &str) -> Result<Vec<ChatMessage>> {
    let value: serde_json::Value = serde_json::from_str(raw)
        .map_err(|error| anyhow!("parse messages input {source}: {error}"))?;
    let strict = match value {
        serde_json::Value::Array(_) => serde_json::from_value(value)
            .map_err(|error| anyhow!("parse bare messages array from {source}: {error}"))?,
        serde_json::Value::Object(_) => {
            let wrapper: StrictMessagesWrapper = serde_json::from_value(value)
                .map_err(|error| anyhow!("parse messages wrapper from {source}: {error}"))?;
            wrapper.messages
        }
        other => bail!(
            "messages input {source} must be a message array or {{\"messages\":[...]}}, got {other}"
        ),
    };
    validate_strict_messages(strict, source)
}

#[allow(dead_code)]
fn validate_strict_messages(
    messages: Vec<StrictChatMessage>,
    source: &str,
) -> Result<Vec<ChatMessage>> {
    if messages.is_empty() {
        bail!("messages input {source} contains no messages");
    }

    let mut expect_user = true;
    let mut saw_user = false;
    let mut validated = Vec::with_capacity(messages.len());
    for (index, message) in messages.into_iter().enumerate() {
        match message.role.as_str() {
            "system" if index == 0 && expect_user => {}
            "user" if expect_user => {
                expect_user = false;
                saw_user = true;
            }
            "assistant" if !expect_user => {
                if message.content.trim_start().starts_with("<think>") {
                    bail!(
                        "message {index} in {source} contains structured assistant thinking; modern `qwen run --messages` does not yet represent reasoning history. Use the legacy --messages interface if those semantics are intentional"
                    );
                }
                expect_user = true;
            }
            role => {
                let expected = if expect_user { "user" } else { "assistant" };
                bail!(
                    "message {index} in {source} has role {role:?}; expected {expected:?} in the strict ordinary-chat subset"
                );
            }
        }
        validated.push(ChatMessage {
            role: message.role,
            content: message.content,
            ..Default::default()
        });
    }

    if !saw_user {
        bail!("messages input {source} requires at least one user turn");
    }
    if expect_user {
        bail!("messages input {source} must end with a user turn before generation");
    }
    Ok(validated)
}

pub(crate) fn render_qwen_messages_prompt(
    messages: &[ChatMessage],
    preserve_thinking: bool,
    append_generation_prompt: bool,
) -> String {
    render_qwen_messages_prompt_with_generation(
        messages,
        preserve_thinking,
        append_generation_prompt,
        QwenGenerationMode::Auto,
    )
}

pub(crate) fn render_qwen_messages_prompt_with_generation(
    messages: &[ChatMessage],
    preserve_thinking: bool,
    append_generation_prompt: bool,
    generation_mode: QwenGenerationMode,
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
        match generation_mode {
            QwenGenerationMode::Auto => {}
            QwenGenerationMode::Thinking => output.push_str("<think>\n"),
            QwenGenerationMode::NoThinking => output.push_str("<think>\n\n</think>\n\n"),
        }
    }
    output
}

pub(crate) fn render_qwen_single_turn_prompt(
    user: &str,
    system: Option<&str>,
    generation_mode: QwenGenerationMode,
) -> String {
    let mut messages = Vec::with_capacity(usize::from(system.is_some()) + 1);
    if let Some(system) = system {
        messages.push(ChatMessage {
            role: "system".into(),
            content: system.into(),
            ..Default::default()
        });
    }
    messages.push(ChatMessage {
        role: "user".into(),
        content: user.into(),
        ..Default::default()
    });
    render_qwen_messages_prompt_with_generation(&messages, false, true, generation_mode)
}

#[allow(dead_code)]
pub(crate) fn render_deepseek_v4_0731_single_turn_prompt(
    user: &str,
    system: Option<&str>,
    options: DeepSeekV4EncodeOptions,
) -> Result<String> {
    let mut messages = Vec::with_capacity(usize::from(system.is_some()) + 1);
    if let Some(system) = system {
        messages.push(ChatMessage {
            role: "system".into(),
            content: system.into(),
            ..Default::default()
        });
    }
    messages.push(ChatMessage {
        role: "user".into(),
        content: user.into(),
        ..Default::default()
    });
    render_deepseek_v4_0731_messages_prompt(&messages, options)
}

/// Render the ordinary chat subset of the DeepSeek V4 release encoder,
/// including the release thinking modes.
///
/// Semantics are ported from the pinned vLLM (`deepseek_v4_encoding.py`) and
/// SGLang (`encoding_dsv4.py`) release-derived encoders. They agree byte for
/// byte for chat plus the default/low/high thinking tiers; the newer max tier
/// is pinned to vLLM separately:
/// - chat mode: every assistant transition is `</think>`; assistant
///   `reasoning` fields are dropped (vLLM `render_message` renders no thinking
///   part).
/// - thinking + drop (release default without tools): history renders
///   byte-identically to chat mode; only the final user turn opens `<think>`
///   (vLLM `render_message`).
/// - thinking + preserve (`drop_thinking=False`): every transition opens
///   `<think>` and each assistant renders `reasoning</think>content`
///   (vLLM `render_message`).
/// - Thinking tiers prepend their effort instruction before the first
///   message content (vLLM `77434861`: low contributes nothing, high the
///   "Absolute maximum" text, max the "Beyond maximum" text); chat mode
///   never consults the table.
///
/// This still intentionally excludes tools, developer messages,
/// latest-reminder, tasks, response formats, and continuation (`wo_eos`)
/// until their richer schemas have independent byte fixtures. The subset
/// keeps one structural simplification: because roles must alternate
/// user/assistant and end with a user turn, the release lookahead transition
/// rule in `render_message` reduces to "every user turn appends the assistant
/// transition", and the final user turn is always the conversation's last
/// user index.
#[allow(dead_code)]
pub(crate) fn render_deepseek_v4_0731_messages_prompt(
    messages: &[ChatMessage],
    options: DeepSeekV4EncodeOptions,
) -> Result<String> {
    let thinking = !matches!(options.reasoning, DeepSeekV4Reasoning::None);
    if options.preserve_reasoning && !thinking {
        bail!(
            "preserve-reasoning requires reasoning low, high, or max; the release encoder renders preserved reasoning only in thinking mode"
        );
    }
    if messages.is_empty() {
        bail!("DeepSeek V4 messages contain no messages");
    }

    let mut output = String::from(DEEPSEEK_V4_BOS);
    if thinking {
        // The release encoder prepends the tier's effort prompt inside the
        // first message's render, before any role content; the low tier is
        // the empty string (vLLM `render_message` at `77434861`).
        output.push_str(options.reasoning.effort_prompt());
    }
    let mut expect_user = true;
    let mut saw_user = false;
    let last_index = messages.len() - 1;

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
        let reasoning_field = deepseek_v4_message_reasoning(index, message)?;

        match message.role.as_str() {
            "system" if index == 0 && expect_user => {
                require_no_reasoning_field(index, "system", reasoning_field)?;
                output.push_str(&message.content);
            }
            "user" if expect_user => {
                require_no_reasoning_field(index, "user", reasoning_field)?;
                output.push_str(DEEPSEEK_V4_USER);
                output.push_str(&message.content);
                // Subset-reduced release transition rule: every user turn is
                // followed by an assistant turn or generation (vLLM:336,354).
                output.push_str(DEEPSEEK_V4_ASSISTANT);
                let open_thinking = thinking && (options.preserve_reasoning || index == last_index);
                output.push_str(if open_thinking {
                    DEEPSEEK_V4_THINK_START
                } else {
                    DEEPSEEK_V4_THINK_END
                });
                expect_user = false;
                saw_user = true;
            }
            "assistant" if !expect_user => {
                if thinking && options.preserve_reasoning {
                    // `drop_thinking=False`: reasoning (or empty) closes with
                    // `</think>` before the summary (vLLM:314-316).
                    output.push_str(reasoning_field.unwrap_or(""));
                    output.push_str(DEEPSEEK_V4_THINK_END);
                }
                // chat mode and thinking+drop history intentionally drop the
                // reasoning field, matching the release encoder.
                output.push_str(&message.content);
                output.push_str(DEEPSEEK_V4_EOS);
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

fn deepseek_v4_message_reasoning(index: usize, message: &ChatMessage) -> Result<Option<&str>> {
    match (&message.reasoning, &message.reasoning_content) {
        (Some(reasoning), Some(alias)) if reasoning != alias => bail!(
            "DeepSeek V4 message {index} sets conflicting `reasoning` and `reasoning_content` fields"
        ),
        (Some(reasoning), _) => Ok(Some(reasoning.as_str())),
        (None, Some(alias)) => Ok(Some(alias.as_str())),
        (None, None) => Ok(None),
    }
}

fn require_no_reasoning_field(
    index: usize,
    role: &str,
    reasoning_field: Option<&str>,
) -> Result<()> {
    if reasoning_field.is_some() {
        bail!(
            "DeepSeek V4 message {index} ({role}) carries a reasoning field; reasoning is only representable on assistant turns"
        );
    }
    Ok(())
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
            ..Default::default()
        }
    }

    fn assistant_with_reasoning(content: &str, reasoning: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            content: content.into(),
            reasoning: Some(reasoning.into()),
            ..Default::default()
        }
    }

    fn options(
        reasoning: DeepSeekV4Reasoning,
        preserve_reasoning: bool,
    ) -> DeepSeekV4EncodeOptions {
        DeepSeekV4EncodeOptions {
            reasoning,
            preserve_reasoning,
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
    fn qwen_single_turn_generation_modes_are_byte_exact() {
        assert_eq!(
            render_qwen_single_turn_prompt("Hello", None, QwenGenerationMode::Auto),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            render_qwen_single_turn_prompt("Hello", None, QwenGenerationMode::Thinking),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            render_qwen_single_turn_prompt(
                "Hello",
                Some("Be exact."),
                QwenGenerationMode::NoThinking,
            ),
            concat!(
                "<|im_start|>system\nBe exact.<|im_end|>\n",
                "<|im_start|>user\nHello<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            )
        );
    }

    #[test]
    fn strict_messages_accept_only_the_ordinary_chat_subset() {
        for raw in [
            r#"[{"role":"user","content":"hello"}]"#,
            r#"{"messages":[{"role":"system","content":"Be exact."},{"role":"user","content":"one"},{"role":"assistant","content":"done"},{"role":"user","content":"two"}]}"#,
        ] {
            let messages = parse_strict_messages_input(raw, "test").unwrap();
            assert_eq!(messages.last().unwrap().role, "user");
            assert!(messages.iter().all(|message| message.extra.is_empty()));
        }

        let cases = [
            (
                r#"{"messages":[{"role":"user","content":"hello"}],"meta":{}}"#,
                "unknown field",
            ),
            (
                r#"[{"role":"user","content":"hello","tool_calls":[]}]"#,
                "unknown field",
            ),
            (
                r#"[{"role":"developer","content":"hello"}]"#,
                "expected \"user\"",
            ),
            (
                r#"[{"role":"user","content":"one"},{"role":"user","content":"two"}]"#,
                "expected \"assistant\"",
            ),
            (
                r#"[{"role":"user","content":"one"},{"role":"assistant","content":"done"}]"#,
                "must end with a user turn",
            ),
            (
                r#"[{"role":"user","content":"one"},{"role":"assistant","content":"<think>hidden</think>done"},{"role":"user","content":"two"}]"#,
                "does not yet represent reasoning history",
            ),
        ];
        for (raw, expected) in cases {
            let error = parse_strict_messages_input(raw, "test")
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?} in {error:?}"
            );
        }
    }

    #[test]
    fn deepseek_v4_ordinary_chat_matches_release_derived_references() {
        let chat = DeepSeekV4EncodeOptions::default();
        assert_eq!(
            render_deepseek_v4_0731_single_turn_prompt("Hello", Some("Be exact."), chat).unwrap(),
            "<｜begin▁of▁sentence｜>Be exact.<｜User｜>Hello<｜Assistant｜></think>"
        );
        assert_eq!(
            render_deepseek_v4_0731_messages_prompt(&[message("user", "Hello")], chat).unwrap(),
            "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
        );
        assert_eq!(
            render_deepseek_v4_0731_messages_prompt(
                &[message("system", "Be exact."), message("user", "Hello"),],
                chat
            )
            .unwrap(),
            "<｜begin▁of▁sentence｜>Be exact.<｜User｜>Hello<｜Assistant｜></think>"
        );
        assert_eq!(
            render_deepseek_v4_0731_messages_prompt(
                &[
                    message("system", "Be exact."),
                    message("user", "Hello"),
                    message("assistant", "Hi!"),
                    message("user", "上海 🙂"),
                ],
                chat
            )
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
    fn deepseek_v4_thinking_modes_match_release_transition_rules() {
        // Final user turn opens `<think>`; history stays byte-identical to
        // chat mode in the release drop-thinking default.
        let history = [
            message("system", "Be exact."),
            message("user", "Hello"),
            assistant_with_reasoning("Hi!", "hidden plan"),
            message("user", "上海 🙂"),
        ];
        let drop = render_deepseek_v4_0731_messages_prompt(
            &history,
            options(DeepSeekV4Reasoning::Low, false),
        )
        .unwrap();
        assert_eq!(
            drop,
            concat!(
                "<｜begin▁of▁sentence｜>Be exact.",
                "<｜User｜>Hello<｜Assistant｜></think>",
                "Hi!<｜end▁of▁sentence｜>",
                "<｜User｜>上海 🙂<｜Assistant｜><think>"
            )
        );
        let chat =
            render_deepseek_v4_0731_messages_prompt(&history, DeepSeekV4EncodeOptions::default())
                .unwrap();
        assert_eq!(
            drop.strip_suffix("<think>").unwrap(),
            chat.strip_suffix("</think>").unwrap(),
            "thinking-drop history must render byte-identically to chat mode"
        );

        // Preserve mode: every transition opens `<think>` and assistant turns
        // replay `reasoning</think>content`.
        let preserve = render_deepseek_v4_0731_messages_prompt(
            &history,
            options(DeepSeekV4Reasoning::Low, true),
        )
        .unwrap();
        assert_eq!(
            preserve,
            concat!(
                "<｜begin▁of▁sentence｜>Be exact.",
                "<｜User｜>Hello<｜Assistant｜><think>",
                "hidden plan</think>Hi!<｜end▁of▁sentence｜>",
                "<｜User｜>上海 🙂<｜Assistant｜><think>"
            )
        );

        // Preserve mode with a missing reasoning field renders an empty
        // thinking block, matching the release `reasoning or ""`.
        assert_eq!(
            render_deepseek_v4_0731_messages_prompt(
                &[
                    message("user", "one"),
                    message("assistant", "done"),
                    message("user", "two"),
                ],
                options(DeepSeekV4Reasoning::Low, true),
            )
            .unwrap(),
            concat!(
                "<｜begin▁of▁sentence｜><｜User｜>one<｜Assistant｜><think>",
                "</think>done<｜end▁of▁sentence｜>",
                "<｜User｜>two<｜Assistant｜><think>"
            )
        );

        // The reasoning_content alias is accepted and equivalent.
        let alias = ChatMessage {
            role: "assistant".into(),
            content: "Hi!".into(),
            reasoning_content: Some("hidden plan".into()),
            ..Default::default()
        };
        let aliased = render_deepseek_v4_0731_messages_prompt(
            &[message("user", "Hello"), alias, message("user", "again")],
            options(DeepSeekV4Reasoning::Low, true),
        )
        .unwrap();
        assert!(aliased.contains("<think>hidden plan</think>Hi!"));
    }

    #[test]
    fn deepseek_v4_reasoning_effort_tiers_prefix_the_first_message() {
        // Low is byte-identical to bare thinking mode: no effort bytes.
        let low = render_deepseek_v4_0731_messages_prompt(
            &[message("user", "Hello")],
            options(DeepSeekV4Reasoning::Low, false),
        )
        .unwrap();
        assert_eq!(
            low,
            "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜><think>"
        );
        // High carries the earlier two-tier encoder's max text.
        let high = render_deepseek_v4_0731_messages_prompt(
            &[message("user", "Hello")],
            options(DeepSeekV4Reasoning::High, false),
        )
        .unwrap();
        assert_eq!(
            high,
            format!(
                "<｜begin▁of▁sentence｜>{DEEPSEEK_V4_REASONING_EFFORT_HIGH}<｜User｜>Hello<｜Assistant｜><think>"
            )
        );
        assert!(
            DEEPSEEK_V4_REASONING_EFFORT_HIGH.starts_with("Reasoning Effort: Absolute maximum")
        );
        // Max carries the stronger current release text.
        let single = render_deepseek_v4_0731_messages_prompt(
            &[message("user", "Hello")],
            options(DeepSeekV4Reasoning::Max, false),
        )
        .unwrap();
        assert_eq!(
            single,
            format!(
                "<｜begin▁of▁sentence｜>{DEEPSEEK_V4_REASONING_EFFORT_MAX}<｜User｜>Hello<｜Assistant｜><think>"
            )
        );
        assert!(DEEPSEEK_V4_REASONING_EFFORT_MAX.starts_with("Reasoning Effort: Beyond maximum"));
        let with_system = render_deepseek_v4_0731_messages_prompt(
            &[message("system", "Be exact."), message("user", "Hello")],
            options(DeepSeekV4Reasoning::Max, false),
        )
        .unwrap();
        assert_eq!(
            with_system,
            format!(
                "<｜begin▁of▁sentence｜>{DEEPSEEK_V4_REASONING_EFFORT_MAX}Be exact.<｜User｜>Hello<｜Assistant｜><think>"
            )
        );
        // Chat mode never consults the effort table.
        let chat = render_deepseek_v4_0731_messages_prompt(
            &[message("user", "Hello")],
            DeepSeekV4EncodeOptions::default(),
        )
        .unwrap();
        assert!(!chat.contains("Reasoning Effort"));
    }

    #[test]
    fn deepseek_v4_reasoning_field_policy_fails_closed() {
        // Preserve without thinking is rejected.
        let error = render_deepseek_v4_0731_messages_prompt(
            &[message("user", "Hello")],
            options(DeepSeekV4Reasoning::None, true),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("requires reasoning low, high, or max"));

        // Reasoning fields on non-assistant roles are rejected.
        let mut user = message("user", "Hello");
        user.reasoning = Some("nope".into());
        let error =
            render_deepseek_v4_0731_messages_prompt(&[user], DeepSeekV4EncodeOptions::default())
                .unwrap_err()
                .to_string();
        assert!(error.contains("only representable on assistant turns"));

        // Conflicting reasoning aliases are rejected.
        let conflicted = ChatMessage {
            role: "assistant".into(),
            content: "Hi!".into(),
            reasoning: Some("a".into()),
            reasoning_content: Some("b".into()),
            ..Default::default()
        };
        let error = render_deepseek_v4_0731_messages_prompt(
            &[
                message("user", "Hello"),
                conflicted,
                message("user", "again"),
            ],
            DeepSeekV4EncodeOptions::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("conflicting"));

        // Chat mode and thinking-drop history silently drop assistant
        // reasoning, matching the release encoder bytes.
        let with_reasoning = [
            message("user", "one"),
            assistant_with_reasoning("done", "hidden"),
            message("user", "two"),
        ];
        let without_reasoning = [
            message("user", "one"),
            message("assistant", "done"),
            message("user", "two"),
        ];
        for opts in [
            DeepSeekV4EncodeOptions::default(),
            options(DeepSeekV4Reasoning::Low, false),
        ] {
            assert_eq!(
                render_deepseek_v4_0731_messages_prompt(&with_reasoning, opts).unwrap(),
                render_deepseek_v4_0731_messages_prompt(&without_reasoning, opts).unwrap(),
            );
        }
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
            assert!(
                render_deepseek_v4_0731_messages_prompt(
                    &messages,
                    DeepSeekV4EncodeOptions::default()
                )
                .is_err()
            );
        }

        let (messages, _) = parse_messages_input(json!([
            {"role": "user", "content": "hello", "reasoning": "hidden"}
        ]))
        .unwrap();
        let error =
            render_deepseek_v4_0731_messages_prompt(&messages, DeepSeekV4EncodeOptions::default())
                .unwrap_err()
                .to_string();
        assert!(error.contains("only representable on assistant turns"));

        let (messages, _) = parse_messages_input(json!([
            {"role": "user", "content": "hello", "tool_calls": []}
        ]))
        .unwrap();
        let error =
            render_deepseek_v4_0731_messages_prompt(&messages, DeepSeekV4EncodeOptions::default())
                .unwrap_err()
                .to_string();
        assert!(error.contains("unsupported fields: tool_calls"));
    }

    #[test]
    fn deepseek_v4_chat_fixtures_match_release_encoders() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek_v4_0731_chat_fixtures_v1.json"
        ))
        .expect("parse chat fixture JSON");
        let cases = fixture["cases"].as_array().expect("fixture cases");
        assert_eq!(cases.len(), 17, "fixture case census");
        for case in cases {
            let name = case["name"].as_str().expect("case name");
            let messages: Vec<ChatMessage> = serde_json::from_value(case["messages"].clone())
                .unwrap_or_else(|error| panic!("parse {name} messages: {error}"));
            let thinking_mode = case["thinking_mode"].as_str().expect("thinking mode");
            let effort = case["reasoning_effort"].as_str();
            let reasoning = match (thinking_mode, effort) {
                ("chat", None) => DeepSeekV4Reasoning::None,
                ("thinking", None | Some("low")) => DeepSeekV4Reasoning::Low,
                ("thinking", Some("high")) => DeepSeekV4Reasoning::High,
                ("thinking", Some("max")) => DeepSeekV4Reasoning::Max,
                other => panic!("unmapped fixture mode {other:?} in {name}"),
            };
            let options = DeepSeekV4EncodeOptions {
                reasoning,
                preserve_reasoning: !case["drop_thinking"].as_bool().expect("drop flag"),
            };
            let rendered = render_deepseek_v4_0731_messages_prompt(&messages, options)
                .unwrap_or_else(|error| panic!("render {name}: {error}"));
            assert_eq!(
                rendered,
                case["prompt"].as_str().expect("case prompt"),
                "fixture case {name} diverged from the release encoders"
            );
        }
    }

    #[test]
    #[ignore = "requires pinned vLLM and SGLang checkouts for fixture regeneration"]
    fn chat_fixture_regeneration_has_no_drift() {
        let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("resolve repository root");
        let status = std::process::Command::new("uv")
            .args([
                "run",
                "scripts/reference/generate_dsv4_chat_fixtures.py",
                "--check",
            ])
            .current_dir(repository_root)
            .status()
            .expect("run chat fixture drift gate");
        assert!(status.success(), "chat fixtures drifted from references");
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
        let error =
            load_deepseek_v4_0731_messages_prompt(&path, None, DeepSeekV4EncodeOptions::default())
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
        let chat = DeepSeekV4EncodeOptions::default();
        assert!(load_deepseek_v4_0731_messages_prompt(&path, Some(1), chat).is_ok());
        let error = load_deepseek_v4_0731_messages_prompt(&path, Some(2), chat)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must end with a user turn"));
        assert!(load_deepseek_v4_0731_messages_prompt(&path, Some(3), chat).is_ok());
        // The trailing assistant message now carries a representable
        // reasoning field, so the full list fails on conversation shape
        // rather than on the field itself.
        let error = load_deepseek_v4_0731_messages_prompt(&path, None, chat)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must end with a user turn"));
        std::fs::remove_file(path).unwrap();
    }
}
