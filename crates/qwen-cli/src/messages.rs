use anyhow::{Context, Result, anyhow, bail, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::model_family::ModelFamily;
use serde::Deserialize;

use std::collections::BTreeMap;
use std::path::Path;

use crate::model_request::{SystemSource, ToolCall, ToolDefinition, ToolResult, Turn};
use crate::open_responses::items::{QwenTemplate, ServeRequest};
use crate::open_responses::render::{
    QwenServePromptChannel, QwenServePromptRole, QwenServePromptSpan, QwenServePromptSpanKind,
    render_qwen_serve_prompt_annotated_with, split_reasoning,
};
use crate::open_responses::tool_parse::python_json;

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
    /// Structured tool calls validated by the strict `qwen run` parser.
    #[serde(skip)]
    pub(crate) tool_calls: Vec<ToolCall>,
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

/// A request asked for something this model's contract does not provide: a
/// reasoning control that cannot bind, an input form its template cannot
/// render. `code` is stable for machine consumers; `message` is what the
/// user sees. Transports map it to their own error shape (anyhow on the CLI,
/// `ServeError` with a param).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CapabilityError {
    pub(crate) code: &'static str,
    pub(crate) message: String,
}

impl std::fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CapabilityError {}

impl CapabilityError {
    pub(crate) fn invalid_level(family: &str, levels: &[&str], got: &str) -> Self {
        Self {
            code: "reasoning_effort_invalid",
            message: format!(
                "{family} accepts reasoning effort {}; got {got:?}",
                levels.join("|")
            ),
        }
    }
}

// Shared across binaries; qwen-bench renders DeepSeek without binding controls.
#[allow(dead_code)]
impl DeepSeekV4Reasoning {
    /// Accepted `reasoning_effort` spellings and the tier each binds to —
    /// the one authority for every lane (legacy `--reasoning`, `run
    /// --reasoning-effort`, serve `reasoning.effort`). `none` is ordinary
    /// chat; the fallback when nothing is specified is also chat.
    pub(crate) const LEVELS: &'static [(&'static str, Self)] = &[
        ("none", Self::None),
        ("low", Self::Low),
        ("high", Self::High),
        ("max", Self::Max),
    ];
    pub(crate) const FALLBACK: Self = Self::None;

    pub(crate) fn level_names() -> Vec<&'static str> {
        Self::LEVELS.iter().map(|(name, _)| *name).collect()
    }

    pub(crate) fn parse(effort: Option<&str>) -> Result<Self, CapabilityError> {
        let Some(effort) = effort else {
            return Ok(Self::FALLBACK);
        };
        Self::LEVELS
            .iter()
            .find(|(name, _)| *name == effort)
            .map(|(_, tier)| *tier)
            .ok_or_else(|| {
                CapabilityError::invalid_level("DeepSeek V4", &Self::level_names(), effort)
            })
    }

    pub(crate) fn is_thinking(self) -> bool {
        !matches!(self, Self::None)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DeepSeekV4EncodeOptions {
    pub(crate) reasoning: DeepSeekV4Reasoning,
    /// Maps to the release encoder's `drop_thinking=False`. The official
    /// trigger is declared tool schemas; this explicit knob exists for
    /// no-tools workloads that want interleaved reasoning retention.
    /// Release history only.
    pub(crate) preserve_reasoning: bool,
    pub(crate) history: DeepSeekV4History,
}

/// How past assistant turns render.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum DeepSeekV4History {
    /// The release encoder: the whole transcript in the current tier. Chat
    /// renders every past turn `</think>content`; thinking renders every
    /// past turn as a thinking turn (reasoning kept or dropped per
    /// `preserve_reasoning` and declared tools).
    #[default]
    Release,
    /// Each past turn as it was generated: a turn carrying a reasoning field
    /// (even empty) renders `<think>reasoning</think>content`, a turn without
    /// one renders `</think>content`. Only the generation transition follows
    /// the current tier; the tier's effort prompt still leads the prompt.
    AsGenerated,
}

impl Default for DeepSeekV4EncodeOptions {
    fn default() -> Self {
        Self {
            reasoning: DeepSeekV4Reasoning::None,
            preserve_reasoning: false,
            history: DeepSeekV4History::Release,
        }
    }
}

const DEEPSEEK_V4_BOS: &str = "<｜begin▁of▁sentence｜>";
const DEEPSEEK_V4_EOS: &str = "<｜end▁of▁sentence｜>";
const DEEPSEEK_V4_USER: &str = "<｜User｜>";
const DEEPSEEK_V4_ASSISTANT: &str = "<｜Assistant｜>";
const DEEPSEEK_V4_THINK_START: &str = "<think>";
const DEEPSEEK_V4_THINK_END: &str = "</think>";
pub(crate) const DEEPSEEK_V4_DSML: &str = "｜DSML｜";
const DEEPSEEK_V4_TOOLS_HEADER: &str = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n";
const DEEPSEEK_V4_TOOLS_FOOTER: &str = "\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
const QWEN38_REASONING_EFFORT_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const QWEN38_REASONING_EFFORT_LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
/// Whether the model renders the Qwen3.8 contract (Qwen3.8-27B releases and
/// derivatives, Flash-Next).
#[allow(dead_code)]
pub(crate) fn supports_qwen38_release_prompt_protocol(
    family: ModelFamily,
    gguf: &GgufFile,
) -> bool {
    matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen4Exp)
        && crate::prompt_template::identify_qwen_release_for_gguf(gguf).is_ok_and(|identity| {
            matches!(
                identity.template,
                crate::prompt_template::QwenPromptTemplate::Qwen38
                    | crate::prompt_template::QwenPromptTemplate::Qwen4Next
            )
        })
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
    /// Open Responses `reasoning.effort` spelling.
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::Xhigh => "xhigh",
        }
    }

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

impl Qwen38GenerationMode {
    /// Accepted `reasoning_effort` spellings and the generation mode each
    /// binds to. `none` is the released non-thinking transition (the same
    /// mode `--no-thinking` selects); the fallback is upstream's xhigh.
    pub(crate) const LEVELS: &'static [(&'static str, Self)] = &[
        ("none", Self::NoThinking),
        ("low", Self::Thinking(Qwen38ReasoningEffort::Low)),
        ("medium", Self::Thinking(Qwen38ReasoningEffort::Medium)),
        ("xhigh", Self::Thinking(Qwen38ReasoningEffort::Xhigh)),
    ];

    pub(crate) fn level_names() -> Vec<&'static str> {
        Self::LEVELS.iter().map(|(name, _)| *name).collect()
    }

    /// Bind the request's controls. `no_thinking` and an effort level are
    /// two spellings of one decision, so both together is a conflict.
    pub(crate) fn parse(effort: Option<&str>, no_thinking: bool) -> Result<Self, CapabilityError> {
        if no_thinking && effort.is_some() {
            return Err(CapabilityError {
                code: "reasoning_conflict",
                message: "reasoning effort cannot be combined with no-thinking".into(),
            });
        }
        if no_thinking {
            return Ok(Self::NoThinking);
        }
        let Some(effort) = effort else {
            return Ok(Self::default());
        };
        Self::LEVELS
            .iter()
            .find(|(name, _)| *name == effort)
            .map(|(_, mode)| *mode)
            .ok_or_else(|| CapabilityError::invalid_level("Qwen3.8", &Self::level_names(), effort))
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
    template: QwenTemplate,
    thinking_mode: MessagesThinkingMode,
    append_generation_prompt: bool,
) -> Result<String> {
    load_messages_prompt_with_policy(
        path,
        max_messages,
        template,
        thinking_mode,
        append_generation_prompt,
    )
    .map(|(prompt, _)| prompt)
}

pub(crate) fn load_messages_prompt_with_policy(
    path: &Path,
    max_messages: Option<usize>,
    template: QwenTemplate,
    thinking_mode: MessagesThinkingMode,
    append_generation_prompt: bool,
) -> Result<(String, bool)> {
    let (messages, meta) = load_messages_input(path, max_messages)?;
    let preserve_thinking = match thinking_mode {
        MessagesThinkingMode::Preserve => true,
        MessagesThinkingMode::Strip => false,
        MessagesThinkingMode::Auto => messages_auto_preserve_thinking(&meta),
    };
    let prompt = render_qwen_messages_prompt_for_template(
        &messages,
        template,
        preserve_thinking,
        append_generation_prompt,
        QwenGenerationMode::Auto,
    )?
    .text;
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
    // Legacy flat `--messages` carries no tool declarations.
    render_deepseek_v4_0731_messages_prompt(&messages, &[], options)
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
    /// `null` (OpenAI-style assistant tool-call messages) reads as empty.
    #[serde(default)]
    content: Option<String>,
    /// Assistant reasoning history (OpenAI/SGLang field name).
    #[serde(default)]
    reasoning_content: Option<String>,
    /// OpenAI-shaped `{"type":"function","function":{"name","arguments"}}`
    /// or flat `{"name","arguments"}` calls on an assistant turn.
    #[serde(default)]
    tool_calls: Option<Vec<serde_json::Value>>,
    /// Accepted on tool results for OpenAI-shaped documents; not rendered.
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct StrictMessagesWrapper {
    messages: Vec<StrictChatMessage>,
    /// OpenAI-shaped `{"type":"function","function":{...}}` or flat
    /// `{"name",...}` tool definitions rendered into the system block.
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
}

/// A validated ordinary-chat or tool-continuation document for `qwen run`.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub(crate) struct StrictChat {
    pub(crate) messages: Vec<ChatMessage>,
    pub(crate) tools: Vec<ToolDefinition>,
}

#[allow(dead_code)]
pub(crate) fn parse_strict_messages_input(raw: &str, source: &str) -> Result<StrictChat> {
    let value: serde_json::Value = serde_json::from_str(raw)
        .map_err(|error| anyhow!("parse messages input {source}: {error}"))?;
    let (strict, tools) = match value {
        serde_json::Value::Array(_) => (
            serde_json::from_value(value)
                .map_err(|error| anyhow!("parse bare messages array from {source}: {error}"))?,
            Vec::new(),
        ),
        serde_json::Value::Object(_) => {
            let wrapper: StrictMessagesWrapper = serde_json::from_value(value)
                .map_err(|error| anyhow!("parse messages wrapper from {source}: {error}"))?;
            let tools = wrapper
                .tools
                .unwrap_or_default()
                .iter()
                .enumerate()
                .map(|(index, tool)| parse_strict_tool_definition(tool, index, source))
                .collect::<Result<Vec<_>>>()?;
            (wrapper.messages, tools)
        }
        other => bail!(
            "messages input {source} must be a message array or {{\"messages\":[...]}}, got {other}"
        ),
    };
    let messages = validate_strict_messages(strict, &tools, source)?;
    Ok(StrictChat { messages, tools })
}

fn parse_strict_tool_definition(
    tool: &serde_json::Value,
    index: usize,
    source: &str,
) -> Result<ToolDefinition> {
    let object = tool
        .as_object()
        .with_context(|| format!("tool {index} in {source} must be an object"))?;
    let function = match object.get("function") {
        Some(function) => function
            .as_object()
            .with_context(|| format!("tool {index} in {source}: function must be an object"))?,
        None => object,
    };
    if let Some(kind) = object.get("type").and_then(|kind| kind.as_str()) {
        ensure!(
            kind == "function",
            "tool {index} in {source} has unsupported type {kind:?}"
        );
    }
    let name = function
        .get("name")
        .and_then(|name| name.as_str())
        .with_context(|| format!("tool {index} in {source} is missing a string name"))?;
    ensure!(
        crate::model_request::tool_name_is_valid(name),
        "tool {index} in {source} has an invalid name {name:?}; names must match {}",
        crate::model_request::TOOL_NAME_GRAMMAR
    );
    let description = match function.get("description") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(text)) => Some(text.clone()),
        Some(_) => bail!("tool {index} in {source}: description must be a string"),
    };
    let parameters = function
        .get("parameters")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    ensure!(
        parameters.is_null() || parameters.is_object(),
        "tool {index} in {source}: parameters must be an object"
    );
    let strict = match function.get("strict") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Bool(value)) => Some(*value),
        Some(_) => bail!("tool {index} in {source}: strict must be a boolean"),
    };
    Ok(ToolDefinition {
        name: name.to_owned(),
        description,
        parameters,
        strict,
    })
}

fn parse_strict_tool_call(
    call: &serde_json::Value,
    message_index: usize,
    call_index: usize,
    tools: &[ToolDefinition],
    source: &str,
) -> Result<ToolCall> {
    let object = call.as_object().with_context(|| {
        format!("message {message_index} in {source}: tool_calls[{call_index}] must be an object")
    })?;
    if let Some(kind) = object.get("type") {
        ensure!(
            kind.as_str() == Some("function"),
            "message {message_index} in {source}: tool_calls[{call_index}] has unsupported type {kind}"
        );
    }
    let function = match object.get("function") {
        Some(function) => function.as_object().with_context(|| {
            format!("message {message_index} in {source}: tool_calls[{call_index}].function must be an object")
        })?,
        None => object,
    };
    let name = function
        .get("name")
        .and_then(|name| name.as_str())
        .with_context(|| {
            format!(
                "message {message_index} in {source}: tool_calls[{call_index}] is missing a name"
            )
        })?;
    ensure!(
        tools.iter().any(|tool| tool.name == name),
        "message {message_index} in {source}: tool_calls[{call_index}] names undeclared tool {name:?}"
    );
    let arguments = match function.get("arguments") {
        None | Some(serde_json::Value::Null) => serde_json::Map::new(),
        Some(serde_json::Value::Object(map)) => map.clone(),
        Some(serde_json::Value::String(text)) => serde_json::from_str::<serde_json::Value>(text)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .with_context(|| {
                format!("message {message_index} in {source}: tool_calls[{call_index}].arguments must be a JSON object")
            })?,
        Some(_) => bail!(
            "message {message_index} in {source}: tool_calls[{call_index}].arguments must be an object or JSON string"
        ),
    };
    let call_id = object
        .get("id")
        .or_else(|| object.get("call_id"))
        .and_then(|id| id.as_str())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("call_{message_index}_{call_index}"));
    Ok(ToolCall {
        call_id,
        name: name.to_owned(),
        arguments: serde_json::Value::Object(arguments).to_string(),
    })
}

#[allow(dead_code)]
fn validate_strict_messages(
    messages: Vec<StrictChatMessage>,
    tools: &[ToolDefinition],
    source: &str,
) -> Result<Vec<ChatMessage>> {
    if messages.is_empty() {
        bail!("messages input {source} contains no messages");
    }

    #[derive(Clone, PartialEq)]
    enum Expect {
        User,
        Assistant,
        /// Results still owed, in call order, for the preceding assistant's
        /// calls: `(call_id, name)`.
        ToolResults(Vec<(String, String)>),
    }
    let mut expect = Expect::User;
    let mut saw_user = false;
    let mut validated = Vec::with_capacity(messages.len());
    for (index, message) in messages.into_iter().enumerate() {
        let mut tool_calls = Vec::new();
        let content = message.content.unwrap_or_default();
        ensure!(
            message.role == "assistant" || message.tool_calls.is_none(),
            "message {index} in {source} has role {:?} but carries tool_calls",
            message.role
        );
        ensure!(
            message.role == "tool" || (message.tool_call_id.is_none() && message.name.is_none()),
            "message {index} in {source} has role {:?} but carries tool_call_id/name",
            message.role
        );
        match (message.role.as_str(), &expect) {
            ("system" | "developer", Expect::User) if index == 0 => {}
            ("user", Expect::User) => {
                expect = Expect::Assistant;
                saw_user = true;
            }
            ("assistant", Expect::Assistant) => {
                if content.trim_start().starts_with("<think>") || content.contains("</think>") {
                    bail!(
                        "message {index} in {source} contains inline assistant thinking; pass reasoning history as `reasoning_content` instead"
                    );
                }
                let calls = message.tool_calls.as_deref().unwrap_or_default();
                ensure!(
                    calls.is_empty() || !tools.is_empty(),
                    "message {index} in {source} carries tool_calls but the document declares no tools"
                );
                for (call_index, call) in calls.iter().enumerate() {
                    let call = parse_strict_tool_call(call, index, call_index, tools, source)?;
                    ensure!(
                        tool_calls
                            .iter()
                            .all(|existing: &ToolCall| existing.call_id != call.call_id),
                        "message {index} in {source}: duplicate tool call id {:?}",
                        call.call_id
                    );
                    tool_calls.push(call);
                }
                expect = if tool_calls.is_empty() {
                    Expect::User
                } else {
                    Expect::ToolResults(
                        tool_calls
                            .iter()
                            .map(|call| (call.call_id.clone(), call.name.clone()))
                            .collect(),
                    )
                };
            }
            ("tool", Expect::ToolResults(owed)) => {
                // Results answer the owed calls in order; a supplied id or
                // name must agree with the call it answers.
                let (call_id, name) = owed.first().cloned().expect("owed non-empty");
                if let Some(supplied) = message.tool_call_id.as_deref() {
                    ensure!(
                        supplied == call_id,
                        "message {index} in {source}: tool result answers {supplied:?} but call {call_id:?} is next in order"
                    );
                }
                if let Some(supplied) = message.name.as_deref() {
                    ensure!(
                        supplied == name,
                        "message {index} in {source}: tool result names {supplied:?} but call {call_id:?} invoked {name:?}"
                    );
                }
                let remaining: Vec<_> = owed[1..].to_vec();
                expect = if remaining.is_empty() {
                    Expect::Assistant
                } else {
                    Expect::ToolResults(remaining)
                };
            }
            (role, expect) => {
                let expected = match expect {
                    Expect::User => "user",
                    Expect::Assistant => "assistant",
                    Expect::ToolResults(_) => "tool",
                };
                bail!(
                    "message {index} in {source} has role {role:?}; expected {expected:?} (system/developer may lead; user and assistant alternate; each assistant tool call is followed by one tool result, in call order)"
                );
            }
        }
        validated.push(ChatMessage {
            role: message.role,
            content,
            reasoning_content: message.reasoning_content,
            tool_calls,
            ..Default::default()
        });
    }

    if !saw_user {
        bail!("messages input {source} requires at least one user turn");
    }
    match expect {
        Expect::Assistant => Ok(validated),
        Expect::User => bail!(
            "messages input {source} must end with a user turn or a completed tool-result round before generation"
        ),
        Expect::ToolResults(owed) => bail!(
            "messages input {source} ends with {} tool result(s) still owed for the last assistant's tool calls",
            owed.len()
        ),
    }
}

/// Strict ordinary chat only: the tool surface is rejected. Shared by every
/// consumer that renders without tool support (Lens).
#[allow(dead_code)]
pub(crate) fn parse_strict_ordinary_chat_input(
    raw: &str,
    source: &str,
) -> Result<Vec<ChatMessage>> {
    let chat = parse_strict_messages_input(raw, source)?;
    ensure!(
        chat.tools.is_empty()
            && chat
                .messages
                .iter()
                .all(|message| message.role != "tool" && message.tool_calls.is_empty()),
        "messages input {source} must be ordinary chat; tools and tool history are not supported here"
    );
    Ok(chat.messages)
}

/// Legacy unpinned-ChatML wrapper kept for probes and tests.
#[allow(dead_code)]
pub(crate) fn render_qwen_messages_prompt(
    messages: &[ChatMessage],
    preserve_thinking: bool,
    append_generation_prompt: bool,
) -> String {
    render_qwen_messages_prompt_annotated(messages, preserve_thinking, append_generation_prompt)
        .text
}

/// Legacy unpinned-ChatML wrapper kept for probes and tests.
#[allow(dead_code)]
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

/// Legacy unpinned-ChatML wrapper kept for probes and tests.
#[allow(dead_code)]
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

/// Legacy unpinned-ChatML wrapper kept for probes and tests.
#[allow(dead_code)]
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
    .expect("ordinary chat roles (system/developer, user, assistant, tool)")
}

/// Split assistant content the way the released Qwen templates do: at the
/// first `</think>`, taking the text after the last `<think>` before it as
/// reasoning and the text after the last `</think>` as content, each with
/// surrounding newlines removed as the template's `lstrip`/`rstrip` do.
pub(crate) fn template_split_think(content: &str) -> (Option<String>, String) {
    if !content.contains("</think>") {
        return (None, content.to_owned());
    }
    let before_close = content.split("</think>").next().unwrap_or("");
    let reasoning = before_close
        .trim_end_matches('\n')
        .rsplit("<think>")
        .next()
        .unwrap_or("")
        .trim_start_matches('\n')
        .to_owned();
    let visible = content
        .rsplit("</think>")
        .next()
        .unwrap_or("")
        .trim_start_matches('\n')
        .to_owned();
    (Some(reasoning), visible)
}

/// Render ordinary chat messages through the shared Qwen renderer for a
/// resolved template. `Generic` keeps the legacy unpinned ChatML bytes;
/// identified releases follow the released Jinja (see `open_responses::render`).
/// Accepts the roles the released templates accept: a leading `system` or
/// `developer`, `user`, `assistant`, and `tool` (consecutive results are
/// coalesced into one tool-response turn).
pub(crate) fn render_qwen_messages_prompt_for_template(
    messages: &[ChatMessage],
    template: QwenTemplate,
    preserve_thinking: bool,
    append_generation_prompt: bool,
    generation_mode: QwenGenerationMode,
) -> Result<AnnotatedMessageRender> {
    render_qwen_chat_for_template(
        messages,
        &[],
        template,
        preserve_thinking,
        append_generation_prompt,
        generation_mode,
        None,
    )
}

/// Render a validated chat with declared tools. `qwen38_mode` supplies the
/// Qwen3.8 effort/no-thinking controls when the template is Qwen3.8.
pub(crate) fn render_qwen_chat_for_template(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    template: QwenTemplate,
    preserve_thinking: bool,
    append_generation_prompt: bool,
    generation_mode: QwenGenerationMode,
    qwen38_mode: Option<Qwen38GenerationMode>,
) -> Result<AnnotatedMessageRender> {
    let (no_thinking, reasoning_effort) = match qwen38_mode {
        Some(Qwen38GenerationMode::NoThinking) => (true, None),
        Some(Qwen38GenerationMode::Thinking(effort)) => {
            (false, Some(effort.wire_name().to_owned()))
        }
        None => (generation_mode == QwenGenerationMode::NoThinking, None),
    };
    let mut request = ServeRequest {
        template,
        strip_history_thinking: !preserve_thinking,
        no_thinking,
        reasoning_effort,
        thinking_requested: generation_mode == QwenGenerationMode::Thinking,
        ..ServeRequest::default()
    };
    request.model_request.tools = tools.to_vec();
    for (index, message) in messages.iter().enumerate() {
        match message.role.as_str() {
            "system" | "developer" if index == 0 => {
                request.model_request.system = Some(message.content.clone());
                request.model_request.system_source = Some(if message.role == "system" {
                    SystemSource::System
                } else {
                    SystemSource::Developer
                });
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
                    None if template.verified() => template_split_think(&message.content),
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
                    calls: message.tool_calls.clone(),
                });
            }
            "tool" => {
                let result = ToolResult {
                    call_id: String::new(),
                    name: String::new(),
                    output: message.content.clone(),
                };
                match request.model_request.turns.last_mut() {
                    Some(Turn::ToolResults(results)) => results.push(result),
                    _ => request
                        .model_request
                        .turns
                        .push(Turn::ToolResults(vec![result])),
                }
            }
            other => bail!(
                "ordinary Qwen chat rendering does not accept role {other:?} at message {index} (system/developer must lead; then user, assistant, tool)"
            ),
        }
    }
    let rendered = render_qwen_serve_prompt_annotated_with(&request, append_generation_prompt);
    // The serve renderer numbers rendered blocks; span consumers (lens plans)
    // address authored input messages. Build the rendered-block -> authored
    // index table: a synthesized system block (tools or a reasoning
    // instruction without an authored system) maps to `None`, an omitted
    // empty authored system consumes no block, and consecutive tool results
    // coalesce into one block attributed to the first result.
    let authored_system = messages
        .first()
        .is_some_and(|message| matches!(message.role.as_str(), "system" | "developer"));
    let rendered_system = rendered.spans.iter().any(|span| {
        span.message_index == Some(0) && span.role == Some(QwenServePromptRole::System)
    });
    let mut block_to_authored: Vec<Option<usize>> = Vec::new();
    if rendered_system {
        block_to_authored.push(authored_system.then_some(0));
    }
    let mut previous_role: Option<&str> = None;
    for (index, message) in messages.iter().enumerate() {
        if index == 0 && authored_system {
            continue;
        }
        let coalesced = message.role == "tool" && previous_role == Some("tool");
        if !coalesced {
            block_to_authored.push(Some(index));
        }
        previous_role = Some(message.role.as_str());
    }
    let map_index = |span: &QwenServePromptSpan| -> Option<usize> {
        span.message_index
            .and_then(|block| block_to_authored.get(block).copied().flatten())
    };
    Ok(AnnotatedMessageRender {
        text: rendered.text,
        spans: rendered
            .spans
            .into_iter()
            .map(|span| MessageRenderSpan {
                message_index: map_index(&span),
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
                role: span.role.map(|role| role.as_str().to_owned()),
                channel: span.channel.and_then(|channel| match channel {
                    QwenServePromptChannel::Thinking => Some(MessageRenderChannel::Thinking),
                    _ => None,
                }),
                byte_start: span.byte_start,
                byte_end: span.byte_end,
            })
            .collect(),
    })
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
    // Qwen3.8 renders through the shared serve renderer: preclosed history,
    // trimmed content, and the effort instruction in the system turn.
    render_qwen_chat_for_template(
        messages,
        &[],
        QwenTemplate::Qwen38,
        true,
        append_generation_prompt,
        QwenGenerationMode::Auto,
        Some(generation_mode),
    )
    .expect("ordinary chat roles (system/developer, user, assistant, tool)")
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
    render_qwen_messages_prompt_for_template(&messages, template, false, true, generation_mode)
        .expect("single-turn roles are system/user")
        .text
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
    render_deepseek_v4_0731_messages_prompt(&messages, &[], options)
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
/// Tools (byte-pinned against both encoders on the `tools_*` fixture cases):
/// - Declarations attach to the first system message as `content + "\n\n" +
///   tools block`, or to an empty system message when none exists; this is
///   SGLang serving's rule and the GGUF template's, with vLLM serving
///   diverging by inserting a separate leading system (see the fixture
///   generator). Each declaration renders `function | tojson` with Python
///   separators in name/description/parameters/strict order.
/// - Assistant tool calls render after the content as a DSML block; string
///   arguments verbatim with `string="true"`, everything else as JSON with
///   `string="false"`. A tool-calling turn still ends with EOS.
/// - Tool results merge into one user turn: `<｜User｜>` then each
///   `<tool_result>…</tool_result>` joined by `"\n\n"`, and a following user
///   message joins the same turn. Tool results count as the last user turn.
/// - Any declared tool disables reasoning dropping: thinking mode preserves
///   every assistant's reasoning whatever `preserve_reasoning` says.
///
/// Still excluded: developer messages, latest-reminder, tasks, response
/// formats, and continuation (`wo_eos`).
pub(crate) fn render_deepseek_v4_0731_messages_prompt(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    options: DeepSeekV4EncodeOptions,
) -> Result<String> {
    let thinking = !matches!(options.reasoning, DeepSeekV4Reasoning::None);
    let as_generated = options.history == DeepSeekV4History::AsGenerated;
    if options.preserve_reasoning && !thinking && !as_generated {
        bail!(
            "preserve-reasoning requires reasoning low, high, or max; the release encoder renders preserved reasoning only in thinking mode"
        );
    }
    if messages.is_empty() {
        bail!("DeepSeek V4 messages contain no messages");
    }
    // vLLM `encode_messages`: any tools anywhere disable reasoning dropping.
    let preserve_reasoning = options.preserve_reasoning || !tools.is_empty();

    let mut output = String::from(DEEPSEEK_V4_BOS);
    if thinking {
        // The release encoder prepends the tier's effort prompt inside the
        // first message's render, before any role content; the low tier is
        // the empty string (vLLM `render_message` at `77434861`).
        output.push_str(options.reasoning.effort_prompt());
    }
    let mut expect_user = true;
    let mut saw_user = false;
    // Tool results answer the last user turn's position (the encoder merges
    // them into user messages before locating it).
    let last_user_like = messages
        .iter()
        .rposition(|message| matches!(message.role.as_str(), "user" | "tool"))
        .unwrap_or(0);
    let leads_with_system = messages
        .first()
        .is_some_and(|message| message.role == "system");
    if !tools.is_empty() && !leads_with_system {
        output.push_str("\n\n");
        output.push_str(&deepseek_v4_tools_block(tools));
    }
    // Whether the previous rendered turn was a user-like turn that is still
    // open (tool results and a following user message share one turn).
    let mut in_user = false;
    // Whether the transition before the next assistant turn opened `<think>`.
    let mut assistant_thinks = false;

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
                if !tools.is_empty() {
                    output.push_str("\n\n");
                    output.push_str(&deepseek_v4_tools_block(tools));
                }
            }
            "user" | "tool" if expect_user => {
                require_no_reasoning_field(index, message.role.as_str(), reasoning_field)?;
                // Only tool results extend an open user turn (the encoder
                // merges them); back-to-back plain user messages stay outside
                // the pinned subset.
                if in_user
                    && message.role == "user"
                    && index > 0
                    && messages[index - 1].role == "user"
                {
                    bail!(
                        "DeepSeek V4 message {index} has role \"user\" directly after another user turn; expected \"assistant\""
                    );
                }
                if in_user {
                    output.push_str("\n\n");
                } else {
                    output.push_str(DEEPSEEK_V4_USER);
                    in_user = true;
                }
                if message.role == "tool" {
                    output.push_str("<tool_result>");
                    output.push_str(&message.content);
                    output.push_str("</tool_result>");
                } else {
                    output.push_str(&message.content);
                }
                saw_user = true;
                // Release transition rule: a user-like turn appends the
                // assistant transition when the next message is an assistant
                // turn or the conversation ends (vLLM:336,354).
                let next_is_user_like = messages
                    .get(index + 1)
                    .is_some_and(|next| matches!(next.role.as_str(), "user" | "tool"));
                if !next_is_user_like {
                    output.push_str(DEEPSEEK_V4_ASSISTANT);
                    let open_thinking = match (as_generated, messages.get(index + 1)) {
                        // A past turn's own provenance: it thought iff it
                        // carries a reasoning field, even an empty one.
                        (true, Some(next)) => {
                            deepseek_v4_message_reasoning(index + 1, next)?.is_some()
                        }
                        (true, None) => thinking,
                        (false, _) => thinking && (preserve_reasoning || index >= last_user_like),
                    };
                    assistant_thinks = open_thinking;
                    output.push_str(if open_thinking {
                        DEEPSEEK_V4_THINK_START
                    } else {
                        DEEPSEEK_V4_THINK_END
                    });
                    expect_user = false;
                    in_user = false;
                }
            }
            "assistant" if !expect_user => {
                let renders_reasoning = if as_generated {
                    assistant_thinks
                } else {
                    thinking && preserve_reasoning
                };
                if renders_reasoning {
                    // `drop_thinking=False`: reasoning (or empty) closes with
                    // `</think>` before the summary (vLLM:314-316).
                    output.push_str(reasoning_field.unwrap_or(""));
                    output.push_str(DEEPSEEK_V4_THINK_END);
                }
                // chat mode and thinking+drop history intentionally drop the
                // reasoning field, matching the release encoder.
                output.push_str(&message.content);
                if !message.tool_calls.is_empty() {
                    output.push_str("\n\n<");
                    output.push_str(DEEPSEEK_V4_DSML);
                    output.push_str("tool_calls>\n");
                    for call in &message.tool_calls {
                        output.push_str(&deepseek_v4_tool_call_dsml(call)?);
                        output.push('\n');
                    }
                    output.push_str("</");
                    output.push_str(DEEPSEEK_V4_DSML);
                    output.push_str("tool_calls>");
                }
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

/// The release tools block: header, one `function | tojson` per declaration
/// (Python separators, name/description/parameters/strict order), footer.
fn deepseek_v4_tools_block(tools: &[ToolDefinition]) -> String {
    let mut block = String::from(DEEPSEEK_V4_TOOLS_HEADER);
    for tool in tools {
        let mut function = serde_json::Map::new();
        function.insert("name".into(), serde_json::json!(tool.name));
        if let Some(description) = tool.description.as_deref() {
            function.insert("description".into(), serde_json::json!(description));
        }
        if !tool.parameters.is_null() {
            function.insert("parameters".into(), tool.parameters.clone());
        }
        if let Some(strict) = tool.strict {
            function.insert("strict".into(), serde_json::json!(strict));
        }
        block.push_str(&python_json(&serde_json::Value::Object(function)));
        block.push('\n');
    }
    block.push_str(DEEPSEEK_V4_TOOLS_FOOTER);
    block
}

/// One `<｜DSML｜invoke>` block. Arguments are the replayed JSON object;
/// string values render verbatim with `string="true"`, everything else as
/// Python-separated JSON with `string="false"` (vLLM `encode_arguments_to_dsml`).
fn deepseek_v4_tool_call_dsml(call: &ToolCall) -> Result<String> {
    let arguments: serde_json::Value = serde_json::from_str(&call.arguments)
        .with_context(|| format!("tool call {} arguments are not JSON", call.name))?;
    let serde_json::Value::Object(arguments) = arguments else {
        bail!("tool call {} arguments must be a JSON object", call.name);
    };
    let mut dsml = format!("<{DEEPSEEK_V4_DSML}invoke name=\"{}\">\n", call.name);
    for (key, value) in &arguments {
        let (is_string, rendered) = match value {
            serde_json::Value::String(text) => ("true", text.clone()),
            other => ("false", python_json(other)),
        };
        dsml.push_str(&format!(
            "<{DEEPSEEK_V4_DSML}parameter name=\"{key}\" string=\"{is_string}\">{rendered}</{DEEPSEEK_V4_DSML}parameter>\n"
        ));
    }
    dsml.push_str(&format!("</{DEEPSEEK_V4_DSML}invoke>"));
    Ok(dsml)
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

    /// `qwen run --messages` on a pinned Qwen3.6 template preserves history,
    /// so an assistant turn without `reasoning_content` renders the empty
    /// block the released template gives it under `preserve_thinking=true`
    /// (missing reasoning is empty reasoning, as in serve).
    #[test]
    fn qwen36_cli_missing_history_reasoning_matches_jinja_preserve() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/qwen36_chat_template_oracle_v1.json"
        ))
        .unwrap();
        let expected = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["id"] == "history_preserve_missing_reasoning")
            .unwrap()["rendered"]
            .as_str()
            .unwrap()
            .to_owned();
        let rendered = render_qwen_chat_for_template(
            &[
                message("user", "One"),
                message("assistant", "Answer one"),
                message("user", "Two"),
            ],
            &[],
            QwenTemplate::Qwen36,
            true,
            true,
            QwenGenerationMode::Auto,
            None,
        )
        .unwrap();
        assert_eq!(rendered.text, expected);
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
            history: DeepSeekV4History::Release,
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
    fn template_conversion_accepts_released_roles_and_rejects_others() {
        let messages = vec![
            message("developer", " Dev "),
            message("user", "Call it"),
            message(
                "assistant",
                "<tool_call>\n<function=ping>\n</function>\n</tool_call>",
            ),
            message("tool", " pong "),
            message("tool", "pong2"),
            message("user", "Thanks"),
        ];
        let rendered = render_qwen_messages_prompt_for_template(
            &messages,
            QwenTemplate::Qwen36,
            true,
            true,
            QwenGenerationMode::Auto,
        )
        .expect("released roles render");
        assert!(
            rendered
                .text
                .starts_with("<|im_start|>system\nDev<|im_end|>\n")
        );
        assert!(rendered.text.contains(
            "<|im_start|>user\n<tool_response>\npong\n</tool_response>\n<tool_response>\npong2\n</tool_response><|im_end|>\n"
        ));
        let error = render_qwen_messages_prompt_for_template(
            &[message("user", "a"), message("system", "late")],
            QwenTemplate::Qwen36,
            true,
            true,
            QwenGenerationMode::Auto,
        )
        .expect_err("late system is rejected, not silently dropped");
        assert!(error.to_string().contains("role \"system\""));
        assert_eq!(
            template_split_think("foo</think>bar"),
            (Some("foo".into()), "bar".into())
        );
        assert_eq!(
            template_split_think("<think>\n r \n</think>\n\nv"),
            (Some(" r ".into()), "v".into())
        );
        assert_eq!(template_split_think("plain"), (None, "plain".into()));
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
    fn strict_tool_documents_link_results_to_calls() {
        let tools = r#"[{"type":"function","function":{"name":"a"}},{"type":"function","function":{"name":"b"}}]"#;
        let ok = format!(
            r#"{{"messages":[{{"role":"user","content":"go"}},{{"role":"assistant","content":null,"tool_calls":[{{"id":"c1","type":"function","function":{{"name":"a","arguments":"{{}}"}}}},{{"id":"c2","type":"function","function":{{"name":"b","arguments":{{}}}}}}]}},{{"role":"tool","tool_call_id":"c1","name":"a","content":"ra"}},{{"role":"tool","tool_call_id":"c2","content":"rb"}}],"tools":{tools}}}"#
        );
        let chat = parse_strict_messages_input(&ok, "test").expect("linked results parse");
        assert_eq!(chat.messages[1].tool_calls.len(), 2);
        assert_eq!(chat.messages[1].content, "");
        for (raw, expected) in [
            (
                format!(
                    r#"{{"messages":[{{"role":"user","content":"go"}},{{"role":"assistant","content":"","tool_calls":[{{"id":"c1","function":{{"name":"a","arguments":{{}}}}}},{{"id":"c2","function":{{"name":"b","arguments":{{}}}}}}]}},{{"role":"tool","tool_call_id":"c2","content":"rb"}},{{"role":"tool","tool_call_id":"c1","content":"ra"}}],"tools":{tools}}}"#
                ),
                "next in order",
            ),
            (
                format!(
                    r#"{{"messages":[{{"role":"user","content":"go"}},{{"role":"assistant","content":"","tool_calls":[{{"id":"c1","function":{{"name":"a","arguments":{{}}}}}}]}},{{"role":"tool","name":"b","content":"r"}}],"tools":{tools}}}"#
                ),
                "invoked \"a\"",
            ),
            (
                format!(
                    r#"{{"messages":[{{"role":"user","content":"go"}},{{"role":"assistant","content":"","tool_calls":[{{"id":"c1","function":{{"name":"a","arguments":{{}}}}}},{{"id":"c1","function":{{"name":"b","arguments":{{}}}}}}]}},{{"role":"tool","content":"r"}},{{"role":"tool","content":"r"}}],"tools":{tools}}}"#
                ),
                "duplicate tool call id",
            ),
            (
                format!(
                    r#"{{"messages":[{{"role":"user","content":"go"}},{{"role":"assistant","content":"","tool_calls":[{{"type":"custom","function":{{"name":"a","arguments":{{}}}}}}]}},{{"role":"tool","content":"r"}}],"tools":{tools}}}"#
                ),
                "unsupported type",
            ),
            (
                r#"[{"role":"user","content":"go","tool_call_id":"c1"}]"#.to_string(),
                "carries tool_call_id/name",
            ),
        ] {
            let error = parse_strict_messages_input(&raw, "test")
                .expect_err("must reject")
                .to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn span_mapping_tracks_authored_indices_through_omitted_and_coalesced_blocks() {
        // Empty authored system is omitted by the pinned template; user is
        // still authored index 1.
        let rendered = render_qwen_messages_prompt_for_template(
            &[message("system", "   "), message("user", "A")],
            QwenTemplate::Qwen36,
            true,
            true,
            QwenGenerationMode::Auto,
        )
        .unwrap();
        let user = rendered
            .spans
            .iter()
            .find(|span| span.role.as_deref() == Some("user"))
            .unwrap();
        assert_eq!(user.message_index, Some(1));
        // Coalesced tool results keep the first result's authored index and
        // the following assistant/user keep theirs.
        let messages = vec![
            message("user", "go"),
            ChatMessage {
                role: "assistant".into(),
                content: "".into(),
                tool_calls: vec![
                    ToolCall {
                        call_id: "1".into(),
                        name: "f".into(),
                        arguments: "{}".into(),
                    },
                    ToolCall {
                        call_id: "2".into(),
                        name: "f".into(),
                        arguments: "{}".into(),
                    },
                ],
                ..Default::default()
            },
            message("tool", "a"),
            message("tool", "b"),
            message("user", "next"),
        ];
        let tools = [ToolDefinition {
            name: "f".into(),
            description: None,
            parameters: serde_json::Value::Null,
            strict: None,
        }];
        let rendered = render_qwen_chat_for_template(
            &messages,
            &tools,
            QwenTemplate::Qwen36,
            true,
            true,
            QwenGenerationMode::Auto,
            None,
        )
        .unwrap();
        let user_indices: Vec<_> = rendered
            .spans
            .iter()
            .filter(|span| {
                span.role.as_deref() == Some("user")
                    && span.kind == MessageRenderSpanKind::MessageStartMarker
            })
            .map(|span| span.message_index)
            .collect();
        // synthesized tools block -> None; user 0; tool-results block -> 2; user 4
        assert_eq!(user_indices, vec![Some(0), Some(2), Some(4)]);
        let system = rendered
            .spans
            .iter()
            .find(|span| span.role.as_deref() == Some("system"))
            .unwrap();
        assert_eq!(system.message_index, None);
    }

    #[test]
    fn strict_messages_accept_only_the_ordinary_chat_subset() {
        for raw in [
            r#"[{"role":"user","content":"hello"}]"#,
            r#"{"messages":[{"role":"system","content":"Be exact."},{"role":"user","content":"one"},{"role":"assistant","content":"done"},{"role":"user","content":"two"}]}"#,
        ] {
            let messages = parse_strict_messages_input(raw, "test").unwrap().messages;
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
                "carries tool_calls",
            ),
            (
                r#"{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"ping","arguments":{}}}]},{"role":"tool","content":"pong"}]}"#,
                "declares no tools",
            ),
            (
                r#"{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"nope","arguments":{}}}]},{"role":"tool","content":"pong"}],"tools":[{"type":"function","function":{"name":"ping"}}]}"#,
                "undeclared tool",
            ),
            (
                r#"{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"ping","arguments":{}}},{"function":{"name":"ping","arguments":{}}}]},{"role":"tool","content":"pong"}],"tools":[{"type":"function","function":{"name":"ping"}}]}"#,
                "still owed",
            ),
            (
                r#"[{"role":"user","content":[{"type":"image_url","image_url":{"url":"image.jpg"}}]}]"#,
                "invalid type",
            ),
            (
                r#"[{"role":"developer","content":"hello"}]"#,
                "requires at least one user turn",
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
                "must end with a user turn or a completed tool-result round",
            ),
            (
                r#"[{"role":"user","content":"one"},{"role":"assistant","content":"<think>hidden</think>done"},{"role":"user","content":"two"}]"#,
                "inline assistant thinking",
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
            render_deepseek_v4_0731_messages_prompt(&[message("user", "Hello")], &[], chat)
                .unwrap(),
            "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
        );
        assert_eq!(
            render_deepseek_v4_0731_messages_prompt(
                &[message("system", "Be exact."), message("user", "Hello"),],
                &[],
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
                &[],
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
            &[],
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
        let chat = render_deepseek_v4_0731_messages_prompt(
            &history,
            &[],
            DeepSeekV4EncodeOptions::default(),
        )
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
            &[],
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
                &[],
                options(DeepSeekV4Reasoning::Low, true)
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
            &[],
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
            &[],
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
            &[],
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
            &[],
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
            &[],
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
            &[],
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
            &[],
            options(DeepSeekV4Reasoning::None, true),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("requires reasoning low, high, or max"));

        // Reasoning fields on non-assistant roles are rejected.
        let mut user = message("user", "Hello");
        user.reasoning = Some("nope".into());
        let error = render_deepseek_v4_0731_messages_prompt(
            &[user],
            &[],
            DeepSeekV4EncodeOptions::default(),
        )
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
            &[],
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
                render_deepseek_v4_0731_messages_prompt(&with_reasoning, &[], opts).unwrap(),
                render_deepseek_v4_0731_messages_prompt(&without_reasoning, &[], opts).unwrap(),
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
                    &[],
                    DeepSeekV4EncodeOptions::default()
                )
                .is_err()
            );
        }

        let (messages, _) = parse_messages_input(json!([
            {"role": "user", "content": "hello", "reasoning": "hidden"}
        ]))
        .unwrap();
        let error = render_deepseek_v4_0731_messages_prompt(
            &messages,
            &[],
            DeepSeekV4EncodeOptions::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("only representable on assistant turns"));

        let (messages, _) = parse_messages_input(json!([
            {"role": "user", "content": "hello", "tool_calls": []}
        ]))
        .unwrap();
        let error = render_deepseek_v4_0731_messages_prompt(
            &messages,
            &[],
            DeepSeekV4EncodeOptions::default(),
        )
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
        assert_eq!(cases.len(), 26, "fixture case census");
        let mut tool_cases = 0;
        for case in cases {
            let name = case["name"].as_str().expect("case name");
            // Tool cases are client-shaped (top-level `tools`, OpenAI tool
            // calls and `tool` results) and go through the strict `run
            // --messages` parser, the same path a user takes. Chat cases keep
            // the direct deserialization that also exercises the
            // `reasoning`/`reasoning_content` alias handling.
            let (messages, tools): (Vec<ChatMessage>, Vec<ToolDefinition>) = if case
                .get("tools")
                .is_some()
            {
                // The generator writes both `reasoning` (vLLM) and
                // `reasoning_content` (SGLang/OpenAI) so each encoder reads
                // its own; clients send the OpenAI key, which is what the
                // strict parser accepts.
                let mut client_messages = case["messages"].clone();
                for message in client_messages.as_array_mut().expect("messages array") {
                    message
                        .as_object_mut()
                        .expect("message object")
                        .remove("reasoning");
                }
                let document = serde_json::json!({
                    "messages": client_messages,
                    "tools": case["tools"],
                });
                tool_cases += 1;
                match parse_strict_messages_input(&document.to_string(), name) {
                    Ok(chat) => (chat.messages, chat.tools),
                    // The strict subset requires an assistant turn after a
                    // completed tool round; the encoders accept a user turn
                    // there (merged into the same user-like turn). Pin that
                    // case at the renderer level from the client shape.
                    Err(_) if name == "tools_chat_one_round_then_user" => {
                        let messages = client_messages
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|message| ChatMessage {
                                role: message["role"].as_str().unwrap().to_owned(),
                                content: message["content"].as_str().unwrap_or("").to_owned(),
                                reasoning_content: message["reasoning_content"]
                                    .as_str()
                                    .map(str::to_owned),
                                tool_calls: message["tool_calls"]
                                    .as_array()
                                    .map(|calls| {
                                        calls
                                            .iter()
                                            .map(|call| ToolCall {
                                                call_id: call["id"].as_str().unwrap().to_owned(),
                                                name: call["function"]["name"]
                                                    .as_str()
                                                    .unwrap()
                                                    .to_owned(),
                                                arguments: call["function"]["arguments"]
                                                    .as_str()
                                                    .unwrap()
                                                    .to_owned(),
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default(),
                                ..Default::default()
                            })
                            .collect();
                        let tools =
                            serde_json::from_value::<Vec<serde_json::Value>>(case["tools"].clone())
                                .unwrap()
                                .into_iter()
                                .map(|tool| ToolDefinition {
                                    name: tool["function"]["name"].as_str().unwrap().to_owned(),
                                    description: tool["function"]["description"]
                                        .as_str()
                                        .map(str::to_owned),
                                    parameters: tool["function"]["parameters"].clone(),
                                    strict: None,
                                })
                                .collect();
                        (messages, tools)
                    }
                    Err(error) => panic!("strict-parse {name}: {error}"),
                }
            } else {
                (
                    serde_json::from_value(case["messages"].clone())
                        .unwrap_or_else(|error| panic!("parse {name} messages: {error}")),
                    Vec::new(),
                )
            };
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
                history: DeepSeekV4History::Release,
            };
            let rendered = render_deepseek_v4_0731_messages_prompt(&messages, &tools, options)
                .unwrap_or_else(|error| panic!("render {name}: {error}"));
            assert_eq!(
                rendered,
                case["prompt"].as_str().expect("case prompt"),
                "fixture case {name} diverged from the release encoders"
            );
        }
        assert_eq!(
            tool_cases, 9,
            "every tool case went through the strict parser"
        );
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

    /// The `qwen run --messages` path (strict parser + shared renderer)
    /// reproduces the released Qwen3.6 template on every oracle case whose
    /// document shape the strict subset accepts, including both two-round
    /// tool cases and history preserved into a no-thinking generation.
    #[test]
    fn run_messages_match_qwen36_jinja_oracle() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/qwen36_chat_template_oracle_v1.json"
        ))
        .unwrap();
        let mut checked = Vec::new();
        for case in fixture["cases"].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            let input = &case["input"];
            let expected = case["rendered"].as_str().unwrap();
            let document = serde_json::json!({
                "messages": input["messages"],
                "tools": input["tools"].as_array().cloned().unwrap_or_default(),
            });
            let Ok(chat) = parse_strict_messages_input(&document.to_string(), id) else {
                continue; // shapes the strict subset rejects (two systems, ends on assistant)
            };
            let generation_mode = if input["enable_thinking"] == serde_json::json!(false) {
                QwenGenerationMode::NoThinking
            } else {
                QwenGenerationMode::Auto
            };
            let preserve = input["preserve_thinking"] == serde_json::json!(true);
            let rendered = render_qwen_chat_for_template(
                &chat.messages,
                &chat.tools,
                QwenTemplate::Qwen36,
                preserve,
                true,
                generation_mode,
                None,
            )
            .unwrap()
            .text;
            assert_eq!(rendered, expected, "{id}");
            checked.push(id);
        }
        assert!(checked.contains(&"two_rounds_strip"), "{checked:?}");
        assert!(checked.contains(&"two_rounds_then_user"), "{checked:?}");
        assert!(checked.contains(&"scalar_params"), "{checked:?}");
        assert!(checked.len() >= 15, "{checked:?}");
    }

    /// jinja2 is fetched by `uv` on first run; hermetic otherwise.
    #[test]
    #[ignore = "runs the jinja2 oracle via uv to detect fixture drift"]
    fn qwen_chat_template_oracle_fixtures_have_no_drift() {
        let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("resolve repository root");
        for (template, cases, fixture) in [
            (
                "qwen36_a3b_chat_template.jinja",
                "qwen36_chat_template_oracle_cases.json",
                "qwen36_chat_template_oracle_v1.json",
            ),
            (
                "qwen35_chat_template.jinja",
                "qwen35_chat_template_oracle_cases.json",
                "qwen35_chat_template_oracle_v1.json",
            ),
            (
                "qwen38_27b_chat_template.jinja",
                "qwen38_chat_template_oracle_cases.json",
                "qwen38_chat_template_oracle_v1.json",
            ),
        ] {
            let fixtures = "crates/qwen-cli/tests/fixtures";
            let status = std::process::Command::new("uv")
                .args([
                    "run",
                    "scripts/reference/render_qwen_chat_template.py",
                    &format!("{fixtures}/templates/{template}"),
                    &format!("{fixtures}/{cases}"),
                    "--check",
                    &format!("{fixtures}/{fixture}"),
                ])
                .current_dir(&repository_root)
                .status()
                .expect("run oracle drift gate");
            assert!(status.success(), "{fixture} drifted from {template}");
        }
        // The Unsloth-patched Qwen3.8 template (dense Q8_0 repack and
        // Flash-Next) must keep rendering the canonical fixture bytes.
        let status = std::process::Command::new("uv")
            .args([
                "run",
                "scripts/reference/render_qwen_chat_template.py",
                "crates/qwen-cli/tests/fixtures/templates/qwen38_27b_unsloth_chat_template.jinja",
                "crates/qwen-cli/tests/fixtures/qwen38_chat_template_oracle_cases.json",
            ])
            .current_dir(&repository_root)
            .output()
            .expect("render unsloth qwen38 template");
        let rendered: serde_json::Value = serde_json::from_slice(&status.stdout).unwrap();
        let canonical: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/qwen38_chat_template_oracle_v1.json"
        ))
        .unwrap();
        assert_eq!(
            rendered["cases"], canonical["cases"],
            "unsloth qwen38 template diverged"
        );
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
                    history: DeepSeekV4History::Release,
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
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "<think>tagged plan</think>tagged reply".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "headless plan</think>headless reply".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "prose then <think>mid</think> more prose".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
                tool_calls: Vec::new(),
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
            tool_calls: Vec::new(),
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
            tool_calls: Vec::new(),
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
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "<think>plan</think>reply".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "user".into(),
                content: "second".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
                tool_calls: Vec::new(),
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
            history: DeepSeekV4History::Release,
        };
        let prompt = render_deepseek_v4_0731_messages_prompt(&promoted, &[], options).unwrap();
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
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "plan\n</think>\n\nreply".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
                tool_calls: Vec::new(),
            },
            ChatMessage {
                role: "user".into(),
                content: "second".into(),
                reasoning: None,
                reasoning_content: None,
                extra: Default::default(),
                tool_calls: Vec::new(),
            },
        ];
        normalize_deepseek_v4_inline_thinking(
            &mut raw,
            DeepSeekV4InlineThinking::PromoteToReasoning,
        )
        .unwrap();
        let raw_prompt = render_deepseek_v4_0731_messages_prompt(&raw, &[], options).unwrap();
        assert_eq!(
            raw_prompt,
            format!(
                "{DEEPSEEK_V4_BOS}{DEEPSEEK_V4_USER}first{DEEPSEEK_V4_ASSISTANT}{DEEPSEEK_V4_THINK_START}plan\n{DEEPSEEK_V4_THINK_END}\n\nreply{DEEPSEEK_V4_EOS}{DEEPSEEK_V4_USER}second{DEEPSEEK_V4_ASSISTANT}{DEEPSEEK_V4_THINK_START}"
            )
        );

        let mut stripped = messages.clone();
        normalize_deepseek_v4_inline_thinking(&mut stripped, DeepSeekV4InlineThinking::Strip)
            .unwrap();
        let chat = render_deepseek_v4_0731_messages_prompt(
            &stripped,
            &[],
            DeepSeekV4EncodeOptions::default(),
        )
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

#[cfg(test)]
mod deepseek_v4_dsml_round_trip {
    use super::*;
    use crate::open_responses::tool_parse::parse_dsml_emission;

    /// Every assistant tool-call turn in the pinned fixtures, rendered by the
    /// release encoder contract, parses back to the same name/argument
    /// objects: the serve output parser and the history renderer agree.
    #[test]
    fn rendered_tool_calls_parse_back_to_their_arguments() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/deepseek_v4_0731_chat_fixtures_v1.json"
        ))
        .unwrap();
        let mut checked = 0;
        for case in fixture["cases"].as_array().unwrap() {
            for message in case["messages"].as_array().unwrap() {
                let Some(calls) = message["tool_calls"].as_array() else {
                    continue;
                };
                let tool_calls: Vec<ToolCall> = calls
                    .iter()
                    .map(|call| ToolCall {
                        call_id: call["id"].as_str().unwrap().to_owned(),
                        name: call["function"]["name"].as_str().unwrap().to_owned(),
                        arguments: call["function"]["arguments"].as_str().unwrap().to_owned(),
                    })
                    .collect();
                let mut emission = message["content"].as_str().unwrap_or("").to_owned();
                emission.push_str(&format!("\n\n<{DEEPSEEK_V4_DSML}tool_calls>\n"));
                for call in &tool_calls {
                    emission.push_str(&deepseek_v4_tool_call_dsml(call).unwrap());
                    emission.push('\n');
                }
                emission.push_str(&format!("</{DEEPSEEK_V4_DSML}tool_calls>"));
                let parsed = parse_dsml_emission(&emission);
                assert_eq!(
                    parsed.visible,
                    format!("{}\n\n", message["content"].as_str().unwrap_or(""))
                );
                assert_eq!(parsed.calls.len(), tool_calls.len());
                for (parsed, original) in parsed.calls.iter().zip(&tool_calls) {
                    assert_eq!(parsed.name, original.name);
                    let expected: serde_json::Value =
                        serde_json::from_str(&original.arguments).unwrap();
                    assert_eq!(
                        serde_json::Value::Object(parsed.arguments.clone()),
                        expected,
                        "case {}",
                        case["name"]
                    );
                }
                checked += 1;
            }
        }
        assert!(checked >= 4, "checked {checked} tool-call turns");
    }
}
