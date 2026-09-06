use anyhow::{Context, Result, anyhow, bail};
use qwen_llm::gguf::GgufFile;
use qwen_llm::model_family::ModelFamily;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

use crate::model_request::{SystemSource, Turn};
use crate::open_responses::items::{QwenTemplate, ServeRequest};
use crate::open_responses::render::{
    QwenServePromptChannel, QwenServePromptSpanKind, render_qwen_serve_prompt_annotated_with,
    split_reasoning,
};

#[derive(Clone, Debug, Default, Deserialize)]
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
const QWEN38_REASONING_EFFORT_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const QWEN38_REASONING_EFFORT_LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
#[allow(dead_code)]
const QWEN4EXP_CHAT_TEMPLATE_SHA256: [u8; 32] = [
    0x12, 0x82, 0x7f, 0x24, 0xb7, 0x42, 0xea, 0x4e, 0x80, 0xcd, 0xc1, 0x2d, 0xbc, 0xf9, 0x62, 0x22,
    0x27, 0x05, 0x6b, 0x9f, 0x79, 0x72, 0x52, 0xa3, 0x14, 0x92, 0x63, 0xd4, 0xf9, 0xaa, 0xad, 0xce,
];

#[allow(dead_code)]
pub(crate) fn qwen4exp_chat_template_matches(template: &str) -> bool {
    Sha256::digest(template.as_bytes()).as_slice() == QWEN4EXP_CHAT_TEMPLATE_SHA256
}

#[allow(dead_code)]
pub(crate) fn supports_qwen4exp_prompt_protocol(family: ModelFamily, gguf: &GgufFile) -> bool {
    family == ModelFamily::Qwen4Exp
        && gguf.get_str("tokenizer.ggml.model") == Some("gpt2")
        && gguf.get_str("tokenizer.ggml.pre") == Some("qwen35")
        && gguf
            .get_str("tokenizer.chat_template")
            .is_some_and(qwen4exp_chat_template_matches)
}

#[allow(dead_code)]
pub(crate) fn supports_qwen38_release_prompt_protocol(
    family: ModelFamily,
    gguf: &GgufFile,
) -> bool {
    if family == ModelFamily::Qwen4Exp {
        return supports_qwen4exp_prompt_protocol(family, gguf);
    }
    validated_qwen38_prompt_identity(
        family,
        gguf.get_str("general.name"),
        gguf.get_str("general.base_model.0.name"),
        gguf.get_str("tokenizer.ggml.model"),
        gguf.get_str("tokenizer.ggml.pre"),
        gguf.get_u64("qwen35.context_length"),
        gguf.get_u64("qwen35.block_count"),
        gguf.get_u64("qwen35.nextn_predict_layers"),
        gguf.get_u64("qwen35.embedding_length"),
        gguf.get_u64("qwen35.feed_forward_length"),
    )
}

#[allow(dead_code)]
pub(crate) fn supports_qwen36_no_thinking_prompt_protocol(
    family: ModelFamily,
    gguf: &GgufFile,
) -> bool {
    validated_qwen36_no_thinking_identity(
        family,
        gguf.get_str("general.base_model.0.name"),
        gguf.get_str("tokenizer.ggml.model"),
        gguf.get_str("tokenizer.ggml.pre"),
    )
}

#[allow(dead_code)]
pub(crate) fn validated_qwen38_prompt_identity(
    family: ModelFamily,
    general_name: Option<&str>,
    base_model_name: Option<&str>,
    tokenizer_model: Option<&str>,
    tokenizer_pre: Option<&str>,
    context_length: Option<u64>,
    block_count: Option<u64>,
    nextn_predict_layers: Option<u64>,
    embedding_length: Option<u64>,
    feed_forward_length: Option<u64>,
) -> bool {
    let named_qwen38_27b = [general_name, base_model_name]
        .into_iter()
        .flatten()
        .any(|name| {
            let name = name.to_ascii_lowercase();
            name.contains("qwen3.8") && name.contains("27b")
        });
    family == ModelFamily::Qwen35
        && named_qwen38_27b
        && tokenizer_model == Some("gpt2")
        && tokenizer_pre == Some("qwen35")
        && context_length == Some(262_144)
        && block_count == Some(65)
        && nextn_predict_layers == Some(1)
        && embedding_length == Some(5_120)
        && feed_forward_length == Some(17_408)
}

#[allow(dead_code)]
pub(crate) fn validated_qwen36_no_thinking_identity(
    family: ModelFamily,
    base_model_name: Option<&str>,
    tokenizer_model: Option<&str>,
    tokenizer_pre: Option<&str>,
) -> bool {
    family == ModelFamily::Qwen35Moe
        && base_model_name == Some("Qwen3.6 35B A3B")
        && tokenizer_model == Some("gpt2")
        && tokenizer_pre == Some("qwen35")
}

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
pub(crate) enum Qwen38ReasoningEffort {
    Low,
    Medium,
    Xhigh,
}

impl Qwen38ReasoningEffort {
    pub(crate) fn instruction(self) -> Option<&'static str> {
        match self {
            Self::Low => Some(QWEN38_REASONING_EFFORT_LOW),
            Self::Medium => None,
            Self::Xhigh => Some(QWEN38_REASONING_EFFORT_XHIGH),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
pub(crate) enum Qwen38GenerationMode {
    Thinking(Qwen38ReasoningEffort),
    NoThinking,
}

impl Default for Qwen38GenerationMode {
    fn default() -> Self {
        Self::Thinking(Qwen38ReasoningEffort::Xhigh)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MessageRenderSpanKind {
    MessageStartMarker,
    Role,
    MessageContent,
    MessageEndMarker,
    GeneratedAssistantStartMarker,
    GeneratedAssistantRole,
    ThinkingChannelStartMarker,
    ThinkingChannelEndMarker,
    ReasoningInstructionContent,
    ContentSeparator,
}

impl MessageRenderSpanKind {
    #[allow(dead_code)]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::MessageStartMarker => "message_start_marker",
            Self::Role => "role",
            Self::MessageContent => "message_content",
            Self::MessageEndMarker => "message_end_marker",
            Self::GeneratedAssistantStartMarker => "generated_assistant_start_marker",
            Self::GeneratedAssistantRole => "generated_assistant_role",
            Self::ThinkingChannelStartMarker => "thinking_channel_start_marker",
            Self::ThinkingChannelEndMarker => "thinking_channel_end_marker",
            Self::ReasoningInstructionContent => "reasoning_instruction_content",
            Self::ContentSeparator => "content_separator",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MessageRenderChannel {
    Thinking,
}

impl MessageRenderChannel {
    #[allow(dead_code)]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Thinking => "thinking",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MessageRenderSpan {
    pub(crate) kind: MessageRenderSpanKind,
    pub(crate) message_index: Option<usize>,
    pub(crate) role: Option<String>,
    pub(crate) channel: Option<MessageRenderChannel>,
    pub(crate) byte_start: usize,
    pub(crate) byte_end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AnnotatedMessageRender {
    pub(crate) text: String,
    pub(crate) spans: Vec<MessageRenderSpan>,
}

#[derive(Default)]
struct AnnotatedMessageRenderBuilder {
    text: String,
    spans: Vec<MessageRenderSpan>,
}

impl AnnotatedMessageRenderBuilder {
    fn push(
        &mut self,
        text: &str,
        kind: MessageRenderSpanKind,
        message_index: Option<usize>,
        role: Option<&str>,
        channel: Option<MessageRenderChannel>,
    ) {
        if text.is_empty() {
            return;
        }
        let byte_start = self.text.len();
        self.text.push_str(text);
        self.spans.push(MessageRenderSpan {
            kind,
            message_index,
            role: role.map(str::to_owned),
            channel,
            byte_start,
            byte_end: self.text.len(),
        });
    }

    fn finish(self) -> AnnotatedMessageRender {
        AnnotatedMessageRender {
            text: self.text,
            spans: self.spans,
        }
    }
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
    inline_thinking: DeepSeekV4InlineThinking,
) -> Result<String> {
    let (mut messages, meta) = load_messages_input(path, max_messages)?;
    normalize_deepseek_v4_inline_thinking(&mut messages, inline_thinking)?;
    validate_deepseek_v4_0731_wrapper_metadata(&meta, options)?;
    render_deepseek_v4_0731_messages_prompt(&messages, options)
}

/// Policy for assistant-history `content` that embeds a leading inline
/// `<think>...</think>` block (or the headless `reasoning</think>visible`
/// shape produced by capturing raw DeepSeek V4 thinking-mode output, whose
/// opening tag lives in the prompt template).
///
/// `Verbatim` preserves the byte-exact 0731 release contract: content passes
/// through untouched. `Strip` and `PromoteToReasoning` are the transcript
/// normalizations behind `--messages-strip-thinking` and
/// `--messages-preserve-thinking`, which let family-portable saves (inline
/// thinking in `content`) round-trip through the release encoder's structured
/// `reasoning` field semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // The bench binary shares this module without the DS4 CLI lanes.
pub(crate) enum DeepSeekV4InlineThinking {
    Verbatim,
    Strip,
    PromoteToReasoning,
}

fn normalize_deepseek_v4_inline_thinking(
    messages: &mut [ChatMessage],
    mode: DeepSeekV4InlineThinking,
) -> Result<()> {
    if mode == DeepSeekV4InlineThinking::Verbatim {
        return Ok(());
    }
    for (index, message) in messages.iter_mut().enumerate() {
        if message.role != "assistant" {
            continue;
        }
        let Some((reasoning, visible)) = split_leading_inline_thinking(&message.content)
            .with_context(|| format!("DeepSeek V4 message {index}"))?
        else {
            continue;
        };
        match mode {
            DeepSeekV4InlineThinking::Verbatim => unreachable!(),
            DeepSeekV4InlineThinking::Strip => {
                // Dropped history tolerates cosmetic trimming (Qwen strip
                // parity); nothing downstream re-derives tokens from it.
                message.content = visible.trim().to_string();
            }
            DeepSeekV4InlineThinking::PromoteToReasoning => {
                if message.reasoning.is_some() || message.reasoning_content.is_some() {
                    bail!(
                        "DeepSeek V4 message {index} carries both inline <think> content and a structured reasoning field"
                    );
                }
                // Byte-faithful promotion: the preserve renderer re-emits
                // reasoning + `</think>` + visible verbatim, so the rendered
                // history reproduces the exact generated transcript and
                // completed-turn durable checkpoints stay strict prefixes of
                // the next turn's prompt. Trimming here would silently break
                // that token identity.
                message.reasoning = Some(reasoning);
                message.content = visible;
            }
        }
    }
    Ok(())
}

/// Split one leading inline thinking block out of assistant content.
///
/// Recognizes `<think>R</think>V` and the headless `R</think>V` transcript
/// shape. A `</think>` preceded by a later-positioned `<think>` opener is not
/// a leading block and passes through verbatim; an opened but unterminated
/// block fails closed rather than guessing at the boundary.
fn split_leading_inline_thinking(content: &str) -> Result<Option<(String, String)>> {
    let trimmed = content.trim_start();
    if let Some(rest) = trimmed.strip_prefix(DEEPSEEK_V4_THINK_START) {
        let Some((reasoning, visible)) = rest.split_once(DEEPSEEK_V4_THINK_END) else {
            bail!("assistant content opens an inline <think> block without closing it");
        };
        return Ok(Some((reasoning.to_string(), visible.to_string())));
    }
    if let Some((reasoning, visible)) = trimmed.split_once(DEEPSEEK_V4_THINK_END) {
        if reasoning.contains(DEEPSEEK_V4_THINK_START) {
            return Ok(None);
        }
        return Ok(Some((reasoning.to_string(), visible.to_string())));
    }
    Ok(None)
}

fn validate_deepseek_v4_0731_wrapper_metadata(
    meta: &serde_json::Value,
    options: DeepSeekV4EncodeOptions,
) -> Result<()> {
    const GAME_METADATA_FIELDS: &[&str] = &[
        "created_at",
        "forked_from",
        "max_tokens",
        "model",
        "preserve_thinking",
        "prompt_version",
        "reasoning",
        "seed",
        "temp",
    ];
    match meta {
        serde_json::Value::Null => Ok(()),
        serde_json::Value::Object(fields) if fields.is_empty() => Ok(()),
        serde_json::Value::Object(fields)
            if fields
                .keys()
                .all(|key| GAME_METADATA_FIELDS.contains(&key.as_str())) =>
        {
            if let Some(model) = fields.get("model")
                && !model.is_string()
            {
                bail!("DeepSeek V4 messages have malformed wrapper field: model");
            }
            if let Some(reasoning) = fields.get("reasoning")
                && !reasoning.is_string()
            {
                bail!("DeepSeek V4 messages have malformed wrapper field: reasoning");
            }
            if let Some(preserve) = fields.get("preserve_thinking") {
                let preserve = preserve.as_bool().ok_or_else(|| {
                    anyhow!("DeepSeek V4 messages have malformed wrapper field: preserve_thinking")
                })?;
                if preserve && !options.preserve_reasoning {
                    bail!(
                        "DeepSeek V4 wrapper requests preserve_thinking=true; use --messages-preserve-thinking (or --preserve-reasoning) with --reasoning low, high, or max"
                    );
                }
            }
            Ok(())
        }
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
        .map(|model| {
            let model = model.to_ascii_lowercase();
            model.contains("qwen3.6") || model.contains("qwen3.8")
        })
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
    render_qwen_messages_prompt_annotated(messages, preserve_thinking, append_generation_prompt)
        .text
}

pub(crate) fn render_qwen_messages_prompt_annotated(
    messages: &[ChatMessage],
    preserve_thinking: bool,
    append_generation_prompt: bool,
) -> AnnotatedMessageRender {
    render_qwen_messages_prompt_with_generation_annotated(
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
    render_qwen_messages_prompt_with_generation_annotated(
        messages,
        preserve_thinking,
        append_generation_prompt,
        generation_mode,
    )
    .text
}

pub(crate) fn render_qwen_messages_prompt_with_generation_annotated(
    messages: &[ChatMessage],
    preserve_thinking: bool,
    append_generation_prompt: bool,
    generation_mode: QwenGenerationMode,
) -> AnnotatedMessageRender {
    render_qwen_messages_prompt_for_template(
        messages,
        QwenTemplate::Generic,
        preserve_thinking,
        append_generation_prompt,
        generation_mode,
    )
}

/// Render ordinary chat messages through the shared Qwen renderer for a
/// resolved template. `Generic` keeps the legacy unpinned ChatML bytes;
/// pinned templates follow the released Jinja (see `open_responses::render`).
pub(crate) fn render_qwen_messages_prompt_for_template(
    messages: &[ChatMessage],
    template: QwenTemplate,
    preserve_thinking: bool,
    append_generation_prompt: bool,
    generation_mode: QwenGenerationMode,
) -> AnnotatedMessageRender {
    let mut request = ServeRequest {
        template,
        strip_history_thinking: !preserve_thinking,
        no_thinking: generation_mode == QwenGenerationMode::NoThinking,
        thinking_requested: generation_mode == QwenGenerationMode::Thinking,
        ..ServeRequest::default()
    };
    for (index, message) in messages.iter().enumerate() {
        match message.role.as_str() {
            "system" if index == 0 => {
                request.model_request.system = Some(message.content.clone());
                request.model_request.system_source = Some(SystemSource::System);
            }
            "user" => request
                .model_request
                .turns
                .push(Turn::User(message.content.clone())),
            "assistant" => {
                let (reasoning, visible) = match message
                    .reasoning_content
                    .as_deref()
                    .or(message.reasoning.as_deref())
                {
                    Some(reasoning) => (Some(reasoning.to_owned()), message.content.clone()),
                    None if preserve_thinking => {
                        let split = split_reasoning(&message.content);
                        (split.reasoning.map(str::to_owned), split.visible.to_owned())
                    }
                    None => {
                        // Legacy strip_think semantics: leading whitespace
                        // before the block is ignored and the visible tail is
                        // trimmed; content without a block is verbatim.
                        let split = split_reasoning(message.content.trim_start());
                        match split.reasoning {
                            Some(reasoning) => {
                                (Some(reasoning.to_owned()), split.visible.trim().to_owned())
                            }
                            None => (None, message.content.clone()),
                        }
                    }
                };
                request.model_request.turns.push(Turn::Assistant {
                    reasoning,
                    visible,
                    calls: Vec::new(),
                });
            }
            other => panic!(
                "ordinary Qwen chat rendering does not accept role {other:?} at message {index}"
            ),
        }
    }
    let rendered = render_qwen_serve_prompt_annotated_with(&request, append_generation_prompt);
    AnnotatedMessageRender {
        text: rendered.text,
        spans: rendered
            .spans
            .into_iter()
            .map(|span| MessageRenderSpan {
                kind: match span.kind {
                    QwenServePromptSpanKind::MessageStartMarker => {
                        MessageRenderSpanKind::MessageStartMarker
                    }
                    QwenServePromptSpanKind::Role => MessageRenderSpanKind::Role,
                    QwenServePromptSpanKind::ContentSeparator => {
                        MessageRenderSpanKind::ContentSeparator
                    }
                    QwenServePromptSpanKind::MessageContent
                    | QwenServePromptSpanKind::AssistantReasoningContent
                    | QwenServePromptSpanKind::ToolDefinitionContent
                    | QwenServePromptSpanKind::ToolCallContent
                    | QwenServePromptSpanKind::ToolResultContent => {
                        MessageRenderSpanKind::MessageContent
                    }
                    QwenServePromptSpanKind::MessageEndMarker => {
                        MessageRenderSpanKind::MessageEndMarker
                    }
                    QwenServePromptSpanKind::GeneratedAssistantStartMarker => {
                        MessageRenderSpanKind::GeneratedAssistantStartMarker
                    }
                    QwenServePromptSpanKind::GeneratedAssistantRole => {
                        MessageRenderSpanKind::GeneratedAssistantRole
                    }
                    QwenServePromptSpanKind::ThinkingChannelStartMarker => {
                        MessageRenderSpanKind::ThinkingChannelStartMarker
                    }
                    QwenServePromptSpanKind::ThinkingChannelEndMarker => {
                        MessageRenderSpanKind::ThinkingChannelEndMarker
                    }
                    QwenServePromptSpanKind::ReasoningInstructionContent => {
                        MessageRenderSpanKind::ReasoningInstructionContent
                    }
                },
                message_index: span.message_index,
                role: span.role.map(|role| role.as_str().to_owned()),
                channel: span.channel.and_then(|channel| match channel {
                    QwenServePromptChannel::Thinking => Some(MessageRenderChannel::Thinking),
                    _ => None,
                }),
                byte_start: span.byte_start,
                byte_end: span.byte_end,
            })
            .collect(),
    }
}

#[allow(dead_code)]
pub(crate) fn render_qwen38_messages_prompt_with_generation(
    messages: &[ChatMessage],
    append_generation_prompt: bool,
    generation_mode: Qwen38GenerationMode,
) -> String {
    render_qwen38_messages_prompt_with_generation_annotated(
        messages,
        append_generation_prompt,
        generation_mode,
    )
    .text
}

#[allow(dead_code)]
pub(crate) fn render_qwen38_messages_prompt_with_generation_annotated(
    messages: &[ChatMessage],
    append_generation_prompt: bool,
    generation_mode: Qwen38GenerationMode,
) -> AnnotatedMessageRender {
    let reasoning_instruction = match generation_mode {
        Qwen38GenerationMode::Thinking(effort) => effort.instruction(),
        Qwen38GenerationMode::NoThinking => None,
    };
    let mut output = AnnotatedMessageRenderBuilder::default();
    let mut first_non_system = 0;
    if let Some(system) = messages.first().filter(|message| message.role == "system") {
        let content = system.content.trim();
        if reasoning_instruction.is_some() || !content.is_empty() {
            output.push(
                "<|im_start|>",
                MessageRenderSpanKind::MessageStartMarker,
                Some(0),
                Some("system"),
                None,
            );
            output.push(
                "system",
                MessageRenderSpanKind::Role,
                Some(0),
                Some("system"),
                None,
            );
            output.push(
                "\n",
                MessageRenderSpanKind::ContentSeparator,
                Some(0),
                Some("system"),
                None,
            );
            if let Some(instruction) = reasoning_instruction {
                output.push(
                    instruction,
                    MessageRenderSpanKind::ReasoningInstructionContent,
                    Some(0),
                    Some("system"),
                    Some(MessageRenderChannel::Thinking),
                );
                if !content.is_empty() {
                    output.push(
                        "\n\n",
                        MessageRenderSpanKind::ContentSeparator,
                        Some(0),
                        Some("system"),
                        None,
                    );
                }
            }
            output.push(
                content,
                MessageRenderSpanKind::MessageContent,
                Some(0),
                Some("system"),
                None,
            );
            output.push(
                "<|im_end|>",
                MessageRenderSpanKind::MessageEndMarker,
                Some(0),
                Some("system"),
                None,
            );
            output.push(
                "\n",
                MessageRenderSpanKind::ContentSeparator,
                Some(0),
                Some("system"),
                None,
            );
        }
        first_non_system = 1;
    } else if let Some(instruction) = reasoning_instruction {
        output.push(
            "<|im_start|>",
            MessageRenderSpanKind::MessageStartMarker,
            None,
            Some("system"),
            None,
        );
        output.push(
            "system",
            MessageRenderSpanKind::Role,
            None,
            Some("system"),
            None,
        );
        output.push(
            "\n",
            MessageRenderSpanKind::ContentSeparator,
            None,
            Some("system"),
            None,
        );
        output.push(
            instruction,
            MessageRenderSpanKind::ReasoningInstructionContent,
            None,
            Some("system"),
            Some(MessageRenderChannel::Thinking),
        );
        output.push(
            "<|im_end|>",
            MessageRenderSpanKind::MessageEndMarker,
            None,
            Some("system"),
            None,
        );
        output.push(
            "\n",
            MessageRenderSpanKind::ContentSeparator,
            None,
            Some("system"),
            None,
        );
    }

    for (message_index, message) in messages.iter().enumerate().skip(first_non_system) {
        let index = Some(message_index);
        output.push(
            "<|im_start|>",
            MessageRenderSpanKind::MessageStartMarker,
            index,
            Some(&message.role),
            None,
        );
        output.push(
            &message.role,
            MessageRenderSpanKind::Role,
            index,
            Some(&message.role),
            None,
        );
        output.push(
            "\n",
            MessageRenderSpanKind::ContentSeparator,
            index,
            Some(&message.role),
            None,
        );
        if message.role == "assistant" {
            output.push(
                "<think>",
                MessageRenderSpanKind::ThinkingChannelStartMarker,
                index,
                Some("assistant"),
                Some(MessageRenderChannel::Thinking),
            );
            output.push(
                "\n\n",
                MessageRenderSpanKind::ContentSeparator,
                index,
                Some("assistant"),
                Some(MessageRenderChannel::Thinking),
            );
            output.push(
                "</think>",
                MessageRenderSpanKind::ThinkingChannelEndMarker,
                index,
                Some("assistant"),
                Some(MessageRenderChannel::Thinking),
            );
            output.push(
                "\n\n",
                MessageRenderSpanKind::ContentSeparator,
                index,
                Some("assistant"),
                None,
            );
        }
        output.push(
            message.content.trim(),
            MessageRenderSpanKind::MessageContent,
            index,
            Some(&message.role),
            None,
        );
        output.push(
            "<|im_end|>",
            MessageRenderSpanKind::MessageEndMarker,
            index,
            Some(&message.role),
            None,
        );
        output.push(
            "\n",
            MessageRenderSpanKind::ContentSeparator,
            index,
            Some(&message.role),
            None,
        );
    }
    if append_generation_prompt {
        output.push(
            "<|im_start|>",
            MessageRenderSpanKind::GeneratedAssistantStartMarker,
            None,
            Some("assistant"),
            None,
        );
        output.push(
            "assistant",
            MessageRenderSpanKind::GeneratedAssistantRole,
            None,
            Some("assistant"),
            None,
        );
        output.push(
            "\n",
            MessageRenderSpanKind::ContentSeparator,
            None,
            Some("assistant"),
            None,
        );
        match generation_mode {
            Qwen38GenerationMode::Thinking(_) => {
                output.push(
                    "<think>",
                    MessageRenderSpanKind::ThinkingChannelStartMarker,
                    None,
                    Some("assistant"),
                    Some(MessageRenderChannel::Thinking),
                );
                output.push(
                    "\n",
                    MessageRenderSpanKind::ContentSeparator,
                    None,
                    Some("assistant"),
                    Some(MessageRenderChannel::Thinking),
                );
            }
            Qwen38GenerationMode::NoThinking => {
                output.push(
                    "<think>",
                    MessageRenderSpanKind::ThinkingChannelStartMarker,
                    None,
                    Some("assistant"),
                    Some(MessageRenderChannel::Thinking),
                );
                output.push(
                    "\n\n",
                    MessageRenderSpanKind::ContentSeparator,
                    None,
                    Some("assistant"),
                    Some(MessageRenderChannel::Thinking),
                );
                output.push(
                    "</think>",
                    MessageRenderSpanKind::ThinkingChannelEndMarker,
                    None,
                    Some("assistant"),
                    Some(MessageRenderChannel::Thinking),
                );
                output.push(
                    "\n\n",
                    MessageRenderSpanKind::ContentSeparator,
                    None,
                    Some("assistant"),
                    None,
                );
            }
        }
    }
    output.finish()
}

#[allow(dead_code)]
pub(crate) fn render_qwen38_single_turn_prompt(
    user: &str,
    system: Option<&str>,
    generation_mode: Qwen38GenerationMode,
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
    render_qwen38_messages_prompt_with_generation(&messages, true, generation_mode)
}

#[allow(dead_code)]
pub(crate) fn render_qwen_single_turn_prompt(
    user: &str,
    system: Option<&str>,
    generation_mode: QwenGenerationMode,
) -> String {
    render_qwen_single_turn_prompt_for_template(
        user,
        system,
        QwenTemplate::Generic,
        generation_mode,
    )
}

pub(crate) fn render_qwen_single_turn_prompt_for_template(
    user: &str,
    system: Option<&str>,
    template: QwenTemplate,
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
    render_qwen_messages_prompt_for_template(&messages, template, false, true, generation_mode).text
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

#[allow(dead_code)]
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

    fn assert_authored_spans(render: &AnnotatedMessageRender) {
        let mut cursor = 0;
        for span in &render.spans {
            assert_eq!(span.byte_start, cursor);
            assert!(span.byte_start < span.byte_end);
            assert!(span.byte_end <= render.text.len());
            assert!(render.text.get(span.byte_start..span.byte_end).is_some());
            assert!(span.role.is_some());
            let text = &render.text[span.byte_start..span.byte_end];
            match span.kind {
                MessageRenderSpanKind::MessageStartMarker
                | MessageRenderSpanKind::GeneratedAssistantStartMarker => {
                    assert_eq!(text, "<|im_start|>");
                }
                MessageRenderSpanKind::Role => {
                    assert_eq!(text, span.role.as_deref().unwrap());
                }
                MessageRenderSpanKind::GeneratedAssistantRole => {
                    assert_eq!(text, "assistant");
                }
                MessageRenderSpanKind::MessageEndMarker => {
                    assert_eq!(text, "<|im_end|>");
                }
                MessageRenderSpanKind::ThinkingChannelStartMarker => {
                    assert_eq!(text, "<think>");
                }
                MessageRenderSpanKind::ThinkingChannelEndMarker => {
                    assert_eq!(text, "</think>");
                }
                MessageRenderSpanKind::ContentSeparator => {
                    assert!(text.bytes().all(|byte| byte == b'\n'));
                }
                MessageRenderSpanKind::MessageContent
                | MessageRenderSpanKind::ReasoningInstructionContent => {}
            }
            if matches!(
                span.kind,
                MessageRenderSpanKind::ThinkingChannelStartMarker
                    | MessageRenderSpanKind::ThinkingChannelEndMarker
                    | MessageRenderSpanKind::ReasoningInstructionContent
            ) {
                assert_eq!(span.channel, Some(MessageRenderChannel::Thinking));
            }
            cursor = span.byte_end;
        }
        assert_eq!(cursor, render.text.len());
    }

    #[test]
    fn annotated_span_labels_are_stable_snake_case() {
        let labels = [
            (
                MessageRenderSpanKind::MessageStartMarker,
                "message_start_marker",
            ),
            (MessageRenderSpanKind::Role, "role"),
            (MessageRenderSpanKind::MessageContent, "message_content"),
            (
                MessageRenderSpanKind::MessageEndMarker,
                "message_end_marker",
            ),
            (
                MessageRenderSpanKind::GeneratedAssistantStartMarker,
                "generated_assistant_start_marker",
            ),
            (
                MessageRenderSpanKind::GeneratedAssistantRole,
                "generated_assistant_role",
            ),
            (
                MessageRenderSpanKind::ThinkingChannelStartMarker,
                "thinking_channel_start_marker",
            ),
            (
                MessageRenderSpanKind::ThinkingChannelEndMarker,
                "thinking_channel_end_marker",
            ),
            (
                MessageRenderSpanKind::ReasoningInstructionContent,
                "reasoning_instruction_content",
            ),
            (MessageRenderSpanKind::ContentSeparator, "content_separator"),
        ];
        for (kind, label) in labels {
            assert_eq!(kind.as_str(), label);
        }
        assert_eq!(MessageRenderChannel::Thinking.as_str(), "thinking");
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
    fn auto_preserves_thinking_for_qwen36_and_qwen38() {
        assert!(messages_auto_preserve_thinking(
            &json!({ "model": "/models/Qwen3.6-27B-Q4_K_M.gguf" })
        ));
        assert!(messages_auto_preserve_thinking(
            &json!({ "preserve_thinking": true, "model": "anything" })
        ));
        assert!(messages_auto_preserve_thinking(
            &json!({ "model": "/models/Qwen3.8-27B-Q4_K_M.gguf" })
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
    fn qwen36_annotated_history_and_generation_modes_are_byte_exact() {
        let messages = vec![
            message("system", "Be exact."),
            message("user", "one"),
            message("assistant", "<think>hidden</think>done"),
            message("user", "two"),
        ];
        // No-thinking sessions re-render history assistant turns with the
        // preclosed block the model consumed (serve normative fixture
        // `qwen_no_thinking_preclosed_history_stable`).
        for (mode, history, suffix) in [
            (QwenGenerationMode::Auto, "", ""),
            (QwenGenerationMode::Thinking, "", "<think>\n"),
            (
                QwenGenerationMode::NoThinking,
                "<think>\n\n</think>\n\n",
                "<think>\n\n</think>\n\n",
            ),
        ] {
            let expected = format!(
                concat!(
                    "<|im_start|>system\nBe exact.<|im_end|>\n",
                    "<|im_start|>user\none<|im_end|>\n",
                    "<|im_start|>assistant\n{history}done<|im_end|>\n",
                    "<|im_start|>user\ntwo<|im_end|>\n",
                    "<|im_start|>assistant\n{suffix}",
                ),
                history = history,
                suffix = suffix,
            );
            let render =
                render_qwen_messages_prompt_with_generation_annotated(&messages, false, true, mode);
            assert_eq!(render.text, expected);
            assert_eq!(
                render.text,
                render_qwen_messages_prompt_with_generation(&messages, false, true, mode)
            );
            assert_authored_spans(&render);
            assert_eq!(
                render
                    .spans
                    .iter()
                    .filter(|span| span.kind == MessageRenderSpanKind::MessageStartMarker)
                    .map(|span| span.message_index)
                    .collect::<Vec<_>>(),
                vec![Some(0), Some(1), Some(2), Some(3)]
            );
        }
    }

    #[test]
    fn qwen38_annotated_history_and_generation_modes_are_byte_exact() {
        let messages = vec![
            message("system", " Be exact. "),
            message("user", " one "),
            message("assistant", " done "),
            message("user", " two "),
        ];
        let prefix = concat!(
            "<|im_start|>system\nBe exact.<|im_end|>\n",
            "<|im_start|>user\none<|im_end|>\n",
            "<|im_start|>assistant\n<think>\n\n</think>\n\ndone<|im_end|>\n",
            "<|im_start|>user\ntwo<|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        for (mode, suffix) in [
            (
                Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Medium),
                "<think>\n",
            ),
            (Qwen38GenerationMode::NoThinking, "<think>\n\n</think>\n\n"),
        ] {
            let render =
                render_qwen38_messages_prompt_with_generation_annotated(&messages, true, mode);
            assert_eq!(render.text, format!("{prefix}{suffix}"));
            assert_eq!(
                render.text,
                render_qwen38_messages_prompt_with_generation(&messages, true, mode)
            );
            assert_authored_spans(&render);
        }

        let instructed = render_qwen38_messages_prompt_with_generation_annotated(
            &[message("user", "hello")],
            true,
            Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Low),
        );
        assert_authored_spans(&instructed);
        let instruction = instructed
            .spans
            .iter()
            .find(|span| span.kind == MessageRenderSpanKind::ReasoningInstructionContent)
            .expect("reasoning instruction span");
        assert_eq!(instruction.message_index, None);
        assert_eq!(
            &instructed.text[instruction.byte_start..instruction.byte_end],
            QWEN38_REASONING_EFFORT_LOW
        );
    }

    #[test]
    fn qwen38_generation_modes_match_upstream_ordinary_chat_subset() {
        assert_eq!(
            render_qwen38_single_turn_prompt(" Hello ", None, Qwen38GenerationMode::default()),
            concat!(
                "<|im_start|>system\n",
                "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
                "<|im_end|>\n",
                "<|im_start|>user\nHello<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n",
            )
        );
        assert_eq!(
            render_qwen38_single_turn_prompt(
                "Hello",
                None,
                Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Low),
            ),
            concat!(
                "<|im_start|>system\n",
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
                "<|im_end|>\n",
                "<|im_start|>user\nHello<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n",
            )
        );
        assert_eq!(
            render_qwen38_single_turn_prompt(
                "Hello",
                None,
                Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Medium),
            ),
            concat!(
                "<|im_start|>user\nHello<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n",
            )
        );
        assert_eq!(
            render_qwen38_single_turn_prompt(
                "Hello",
                Some(" Be exact. "),
                Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Low),
            ),
            concat!(
                "<|im_start|>system\n",
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.\n\n",
                "Be exact.<|im_end|>\n",
                "<|im_start|>user\nHello<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n",
            )
        );
        assert_eq!(
            render_qwen38_single_turn_prompt(
                "Hello",
                Some(" Be exact. "),
                Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Medium),
            ),
            concat!(
                "<|im_start|>system\nBe exact.<|im_end|>\n",
                "<|im_start|>user\nHello<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n",
            )
        );
        assert_eq!(
            render_qwen38_single_turn_prompt(
                "Hello",
                Some("Be exact."),
                Qwen38GenerationMode::NoThinking,
            ),
            concat!(
                "<|im_start|>system\nBe exact.<|im_end|>\n",
                "<|im_start|>user\nHello<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            )
        );

        let history = vec![
            message("user", "one"),
            message("assistant", "done"),
            message("user", "two"),
        ];
        let rendered = render_qwen38_messages_prompt_with_generation(
            &history,
            true,
            Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Xhigh),
        );
        assert_eq!(
            rendered,
            concat!(
                "<|im_start|>system\n",
                "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
                "<|im_end|>\n",
                "<|im_start|>user\none<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n\n</think>\n\ndone<|im_end|>\n",
                "<|im_start|>user\ntwo<|im_end|>\n",
                "<|im_start|>assistant\n<think>\n",
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
                r#"{"messages":[{"role":"user","content":"hello"}],"tools":[]}"#,
                "unknown field",
            ),
            (
                r#"[{"role":"user","content":[{"type":"image_url","image_url":{"url":"image.jpg"}}]}]"#,
                "invalid type",
            ),
            (
                r#"[{"role":"developer","content":"hello"}]"#,
                "expected \"user\"",
            ),
            (
                r#"[{"role":"tool","content":"result"}]"#,
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
        let error = load_deepseek_v4_0731_messages_prompt(
            &path,
            None,
            DeepSeekV4EncodeOptions::default(),
            DeepSeekV4InlineThinking::Verbatim,
        )
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
        let verbatim = DeepSeekV4InlineThinking::Verbatim;
        assert!(load_deepseek_v4_0731_messages_prompt(&path, Some(1), chat, verbatim).is_ok());
        let error = load_deepseek_v4_0731_messages_prompt(&path, Some(2), chat, verbatim)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must end with a user turn"));
        assert!(load_deepseek_v4_0731_messages_prompt(&path, Some(3), chat, verbatim).is_ok());
        // The trailing assistant message now carries a representable
        // reasoning field, so the full list fails on conversation shape
        // rather than on the field itself.
        let error = load_deepseek_v4_0731_messages_prompt(&path, None, chat, verbatim)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must end with a user turn"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn deepseek_v4_loader_accepts_game_wrapper_metadata() {
        let path = std::env::temp_dir().join(format!(
            "qwen-dsv4-game-wrapper-{}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            r#"{
                "meta": {
                    "model":"/tmp/deepseek-v4.gguf",
                    "seed":42,
                    "temp":0.7,
                    "max_tokens":4096,
                    "preserve_thinking":false,
                    "created_at":"2026-08-13T00:00:00",
                    "prompt_version":"the_current",
                    "forked_from":"original"
                },
                "messages": [{"role":"user","content":"hello"}]
            }"#,
        )
        .unwrap();
        assert!(
            load_deepseek_v4_0731_messages_prompt(
                &path,
                None,
                DeepSeekV4EncodeOptions::default(),
                DeepSeekV4InlineThinking::Strip,
            )
            .is_ok()
        );

        std::fs::write(
            &path,
            r#"{
                "meta": {"model":"/tmp/deepseek-v4.gguf","preserve_thinking":true},
                "messages": [{"role":"user","content":"hello"}]
            }"#,
        )
        .unwrap();
        let error = load_deepseek_v4_0731_messages_prompt(
            &path,
            None,
            DeepSeekV4EncodeOptions::default(),
            DeepSeekV4InlineThinking::Strip,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("--messages-preserve-thinking"));
        assert!(
            load_deepseek_v4_0731_messages_prompt(
                &path,
                None,
                DeepSeekV4EncodeOptions {
                    reasoning: DeepSeekV4Reasoning::Low,
                    preserve_reasoning: true,
                },
                DeepSeekV4InlineThinking::PromoteToReasoning,
            )
            .is_ok()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn deepseek_v4_inline_thinking_strip_and_promote_normalize_history() {
        let mut messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "one <think>not assistant</think> text".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "<think>tagged plan</think>tagged reply".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "headless plan</think>headless reply".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "prose then <think>mid</think> more prose".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
            },
        ];

        let mut stripped = messages.clone();
        normalize_deepseek_v4_inline_thinking(&mut stripped, DeepSeekV4InlineThinking::Strip)
            .unwrap();
        assert_eq!(stripped[0].content, messages[0].content);
        assert_eq!(stripped[1].content, "tagged reply");
        assert_eq!(stripped[1].reasoning, None);
        assert_eq!(stripped[2].content, "headless reply");
        // A `</think>` preceded by a later `<think>` opener is not a leading
        // block; the message passes through verbatim.
        assert_eq!(stripped[3].content, messages[3].content);

        let mut promoted = messages.clone();
        normalize_deepseek_v4_inline_thinking(
            &mut promoted,
            DeepSeekV4InlineThinking::PromoteToReasoning,
        )
        .unwrap();
        assert_eq!(promoted[1].reasoning.as_deref(), Some("tagged plan"));
        assert_eq!(promoted[1].content, "tagged reply");
        assert_eq!(promoted[2].reasoning.as_deref(), Some("headless plan"));
        assert_eq!(promoted[2].content, "headless reply");
        assert_eq!(promoted[3].reasoning, None);

        // Promotion preserves raw transcript bytes so re-rendered history
        // reproduces the generated tokens; strip may trim cosmetically.
        let mut raw_transcript = vec![ChatMessage {
            role: "assistant".into(),
            content: "plan things\n</think>\n\nfinal reply\n".into(),
            reasoning: None,
            reasoning_content: None,
            extra: Default::default(),
        }];
        let mut raw_promoted = raw_transcript.clone();
        normalize_deepseek_v4_inline_thinking(
            &mut raw_promoted,
            DeepSeekV4InlineThinking::PromoteToReasoning,
        )
        .unwrap();
        assert_eq!(raw_promoted[0].reasoning.as_deref(), Some("plan things\n"));
        assert_eq!(raw_promoted[0].content, "\n\nfinal reply\n");
        normalize_deepseek_v4_inline_thinking(&mut raw_transcript, DeepSeekV4InlineThinking::Strip)
            .unwrap();
        assert_eq!(raw_transcript[0].content, "final reply");

        let mut verbatim = messages.clone();
        normalize_deepseek_v4_inline_thinking(&mut verbatim, DeepSeekV4InlineThinking::Verbatim)
            .unwrap();
        assert_eq!(verbatim[1].content, messages[1].content);
        assert_eq!(verbatim[2].content, messages[2].content);

        messages[1].reasoning = Some("structured".into());
        let error = normalize_deepseek_v4_inline_thinking(
            &mut messages,
            DeepSeekV4InlineThinking::PromoteToReasoning,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("both inline <think> content and a structured reasoning field"));

        let mut unterminated = vec![ChatMessage {
            role: "assistant".into(),
            content: "<think>never closed".into(),
            reasoning: None,
            reasoning_content: None,
            extra: Default::default(),
        }];
        let error = format!(
            "{:#}",
            normalize_deepseek_v4_inline_thinking(
                &mut unterminated,
                DeepSeekV4InlineThinking::Strip
            )
            .unwrap_err()
        );
        assert!(error.contains("DeepSeek V4 message 0"), "{error}");
        assert!(error.contains("without closing it"), "{error}");
    }

    #[test]
    fn deepseek_v4_promoted_inline_thinking_renders_release_preserve_contract() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "first".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "<think>plan</think>reply".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
            },
            ChatMessage {
                role: "user".into(),
                content: "second".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
            },
        ];
        let mut promoted = messages.clone();
        normalize_deepseek_v4_inline_thinking(
            &mut promoted,
            DeepSeekV4InlineThinking::PromoteToReasoning,
        )
        .unwrap();
        let options = DeepSeekV4EncodeOptions {
            reasoning: DeepSeekV4Reasoning::Low,
            preserve_reasoning: true,
        };
        let prompt = render_deepseek_v4_0731_messages_prompt(&promoted, options).unwrap();
        assert_eq!(
            prompt,
            format!(
                "{DEEPSEEK_V4_BOS}{DEEPSEEK_V4_USER}first{DEEPSEEK_V4_ASSISTANT}{DEEPSEEK_V4_THINK_START}plan{DEEPSEEK_V4_THINK_END}reply{DEEPSEEK_V4_EOS}{DEEPSEEK_V4_USER}second{DEEPSEEK_V4_ASSISTANT}{DEEPSEEK_V4_THINK_START}"
            )
        );

        // A raw thinking transcript with interior whitespace round-trips
        // byte-exactly through promotion + preserve rendering: the rendered
        // history is `<think>` + raw generated output.
        let mut raw = vec![
            ChatMessage {
                role: "user".into(),
                content: "first".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "plan\n</think>\n\nreply".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
            },
            ChatMessage {
                role: "user".into(),
                content: "second".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
            },
        ];
        normalize_deepseek_v4_inline_thinking(
            &mut raw,
            DeepSeekV4InlineThinking::PromoteToReasoning,
        )
        .unwrap();
        let raw_prompt = render_deepseek_v4_0731_messages_prompt(&raw, options).unwrap();
        assert_eq!(
            raw_prompt,
            format!(
                "{DEEPSEEK_V4_BOS}{DEEPSEEK_V4_USER}first{DEEPSEEK_V4_ASSISTANT}{DEEPSEEK_V4_THINK_START}plan\n{DEEPSEEK_V4_THINK_END}\n\nreply{DEEPSEEK_V4_EOS}{DEEPSEEK_V4_USER}second{DEEPSEEK_V4_ASSISTANT}{DEEPSEEK_V4_THINK_START}"
            )
        );

        let mut stripped = messages.clone();
        normalize_deepseek_v4_inline_thinking(&mut stripped, DeepSeekV4InlineThinking::Strip)
            .unwrap();
        let chat =
            render_deepseek_v4_0731_messages_prompt(&stripped, DeepSeekV4EncodeOptions::default())
                .unwrap();
        assert_eq!(
            chat,
            format!(
                "{DEEPSEEK_V4_BOS}{DEEPSEEK_V4_USER}first{DEEPSEEK_V4_ASSISTANT}{DEEPSEEK_V4_THINK_END}reply{DEEPSEEK_V4_EOS}{DEEPSEEK_V4_USER}second{DEEPSEEK_V4_ASSISTANT}{DEEPSEEK_V4_THINK_END}"
            )
        );
    }

    #[test]
    fn serve_render_fixtures_match_existing_qwen_renderer() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/serve_render_fixtures_v1.json"
        ))
        .expect("parse serve render fixture JSON");
        let cases = fixture["cases"].as_array().expect("fixture cases");
        assert_eq!(cases.len(), 9, "fixture case census");
        let mut consumed = 0usize;
        for case in cases {
            let name = case["name"].as_str().expect("case name");
            if case.get("normative_for").is_some() {
                // Frozen ahead of the serve/S2 renderers (SERVE.md gate 4,
                // fixture-before-renderer); asserted by the serve tests.
                continue;
            }
            if let Some(divergence) = case.get("documented_divergence") {
                // A negative example: the bytes a naive renderer would emit.
                // The shared renderer must NOT produce them.
                let messages: Vec<ChatMessage> =
                    serde_json::from_value(case["messages"].clone()).expect("messages");
                let rendered = render_qwen_messages_prompt_with_generation(
                    &messages,
                    false,
                    true,
                    QwenGenerationMode::NoThinking,
                );
                assert_ne!(
                    rendered,
                    case["prompt"].as_str().expect("case prompt"),
                    "{name}: {divergence}"
                );
                continue;
            }
            let messages: Vec<ChatMessage> = serde_json::from_value(case["messages"].clone())
                .unwrap_or_else(|error| panic!("parse {name} messages: {error}"));
            let preserve = match case["policy"].as_str().expect("policy") {
                "preserve" => true,
                "strip" => false,
                other => panic!("unmapped policy {other} in {name}"),
            };
            let mode = match case["generation_mode"].as_str().expect("generation mode") {
                "auto" => QwenGenerationMode::Auto,
                "no_thinking" => QwenGenerationMode::NoThinking,
                other => panic!("unmapped generation mode {other} in {name}"),
            };
            let rendered =
                render_qwen_messages_prompt_with_generation(&messages, preserve, true, mode);
            assert_eq!(
                rendered,
                case["prompt"].as_str().expect("case prompt"),
                "fixture case {name} diverged from the generic renderer"
            );
            consumed += 1;
        }
        assert_eq!(consumed, 6, "consumable case census");
    }

    #[test]
    fn serve_render_fixtures_pin_prefix_stability_topology() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/serve_render_fixtures_v1.json"
        ))
        .expect("parse serve render fixture JSON");
        let cases = fixture["cases"].as_array().expect("fixture cases");
        let prompt_of = |name: &str| -> &str {
            cases
                .iter()
                .find(|case| case["name"].as_str() == Some(name))
                .unwrap_or_else(|| panic!("missing fixture case {name}"))["prompt"]
                .as_str()
                .expect("case prompt")
        };
        let mut stability_checked = 0usize;
        for case in cases {
            let name = case["name"].as_str().expect("case name");
            let prompt = case["prompt"].as_str().expect("case prompt");
            if let Some(base) = case
                .get("assert_prefix_stable_over")
                .and_then(|value| value.as_str())
            {
                assert!(
                    prompt.starts_with(prompt_of(base)),
                    "{name}: expected {base} render to be a byte prefix"
                );
                stability_checked += 1;
            }
            if let Some(diverges) = case.get("diverges_from").and_then(|value| value.as_str()) {
                let base_prompt = prompt_of(diverges);
                assert!(
                    !prompt.starts_with(base_prompt),
                    "{name}: documented divergence unexpectedly stable against {diverges}"
                );
                let boundary = case["common_prefix_ends_after"]
                    .as_str()
                    .expect("divergence boundary");
                let common: String = prompt
                    .chars()
                    .zip(base_prompt.chars())
                    .take_while(|(left, right)| left == right)
                    .map(|(left, _)| left)
                    .collect();
                assert!(
                    common.ends_with(boundary),
                    "{name}: common prefix does not end at the documented boundary"
                );
                stability_checked += 1;
            }
        }
        assert_eq!(stability_checked, 5, "stability assertion census");
    }
}
