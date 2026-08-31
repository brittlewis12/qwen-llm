use crate::messages::{
    AnnotatedMessageRender, ChatMessage, MessageRenderSpanKind, Qwen38GenerationMode,
    Qwen38ReasoningEffort, QwenGenerationMode, parse_strict_messages_input,
    render_qwen_messages_prompt_with_generation_annotated,
    render_qwen38_messages_prompt_with_generation_annotated, supports_qwen4exp_prompt_protocol,
    supports_qwen36_no_thinking_prompt_protocol, supports_qwen38_release_prompt_protocol,
};
use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;
use qwen_llm::gguf::GgufFile;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::muse_glimmer::MuseGlimmerChatTemplateProfile;
use qwen_llm::muse_glimmer_prompt::{
    AnnotatedMuseGlimmerPrompt, MuseGlimmerMessage, MuseGlimmerPromptOptions,
    MuseGlimmerPromptSpanKind, MuseGlimmerReasoningStrength,
    render_muse_glimmer_atem_prompt_annotated,
};
use qwen_llm::tokenizer::{LlamaCppTokenizer, Tokenizer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum LensMessageMode {
    Auto,
    Thinking,
    NoThinking,
    Low,
    Medium,
    Xhigh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QwenMessageProtocol {
    Generic,
    Qwen36,
    Qwen38,
}

#[derive(Clone, Copy)]
pub(crate) struct LensInputSpec<'a> {
    pub(crate) prompt: Option<&'a str>,
    pub(crate) token_ids: Option<&'a [i32]>,
    pub(crate) user: Option<&'a str>,
    pub(crate) system: Option<&'a str>,
    pub(crate) messages: Option<&'a Path>,
    pub(crate) no_special_tokens: bool,
    pub(crate) message_mode: Option<LensMessageMode>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedLensInput {
    pub(crate) source: &'static str,
    pub(crate) add_special_tokens: Option<bool>,
    pub(crate) token_ids: Vec<i32>,
    pub(crate) rendering: LensInputRendering,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LensInputRendering {
    pub(crate) renderer: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) generation_mode: Option<String>,
    pub(crate) spans: Vec<LensRenderedSpan>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LensRenderedSpan {
    pub(crate) kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) message_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) channel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) label: Option<String>,
    pub(crate) byte_start: usize,
    pub(crate) byte_end: usize,
    pub(crate) token_start: Option<usize>,
    pub(crate) token_end: Option<usize>,
}

pub(crate) fn is_structural_lens_span(kind: &str) -> bool {
    matches!(
        kind,
        "bos_marker"
            | "message_start_marker"
            | "message_marker"
            | "message_end_marker"
            | "generated_assistant_start_marker"
            | "thinking_channel_start_marker"
            | "thinking_channel_end_marker"
    )
}

pub(crate) fn is_known_lens_span(kind: &str) -> bool {
    matches!(
        kind,
        "bos_marker"
            | "message_start_marker"
            | "role"
            | "recipient"
            | "tool_name"
            | "message_marker"
            | "message_content"
            | "message_end_marker"
            | "generated_assistant_start_marker"
            | "generated_assistant_role"
            | "thinking_channel_start_marker"
            | "thinking_channel_end_marker"
            | "reasoning_instruction_content"
            | "system_metadata_content"
            | "tool_definition_content"
            | "assistant_reasoning_content"
            | "tool_call_content"
            | "tool_result_content"
            | "content_separator"
    )
}

pub(crate) fn valid_lens_span_metadata(renderer: &str, span: &LensRenderedSpan) -> bool {
    if matches!(
        renderer,
        "qwen_chatml_messages_v1" | "qwen3.6_messages_v1" | "qwen3.8_messages_v1"
    ) {
        return valid_qwen_span_metadata(span);
    }
    if renderer != "muse_glimmer_atem_annotated_v1" {
        return span.tool_call_index.is_none() && span.label.is_none();
    }
    let role = span.role.as_deref();
    let channel = span.channel.as_deref();
    let call = span.tool_call_index;
    let label = span.label.as_deref();
    let indexed_tool_call = call.is_some()
        && role == Some("assistant")
        && channel == Some("tool_call")
        && span.message_index.is_some();
    let has_source_message = span.message_index.is_some() || role == Some("system");
    if call.is_some() && !indexed_tool_call {
        return false;
    }
    match span.kind.as_str() {
        "bos_marker" => {
            span.message_index.is_none()
                && call.is_none()
                && role.is_none()
                && channel.is_none()
                && label.is_none()
        }
        "recipient" => {
            role == Some("assistant")
                && span.message_index.is_some()
                && label.is_some_and(|label| !label.is_empty())
                && channel != Some("tool_result")
        }
        "tool_name" => {
            role == Some("tool")
                && span.message_index.is_some()
                && call.is_none()
                && channel == Some("tool_result")
                && label.is_some_and(|label| !label.is_empty())
        }
        "reasoning_instruction_content" => {
            role == Some("system")
                && has_source_message
                && call.is_none()
                && channel == Some("thinking")
                && label.is_none()
        }
        "assistant_reasoning_content" => {
            role == Some("assistant")
                && span.message_index.is_some()
                && call.is_none()
                && channel == Some("thinking")
                && label.is_none()
        }
        "tool_call_content" => indexed_tool_call && label.is_none(),
        "tool_result_content" => {
            role == Some("tool")
                && span.message_index.is_some()
                && call.is_none()
                && channel == Some("tool_result")
                && label.is_none()
        }
        "system_metadata_content" | "tool_definition_content" => {
            role == Some("system")
                && has_source_message
                && call.is_none()
                && channel.is_none()
                && label.is_none()
        }
        "generated_assistant_start_marker" | "generated_assistant_role" => {
            span.message_index.is_none()
                && call.is_none()
                && role == Some("assistant")
                && channel.is_none()
                && label.is_none()
        }
        "message_end_marker" => {
            role.is_some() && has_source_message && matches!(label, Some("<|eom|>" | "<|eot|>"))
        }
        "message_start_marker" | "role" | "message_marker" => {
            role.is_some() && has_source_message && label.is_none()
        }
        "message_content" => {
            matches!(role, Some("system" | "user" | "assistant"))
                && has_source_message
                && call.is_none()
                && channel != Some("tool_result")
                && label.is_none()
        }
        _ => false,
    }
}

fn valid_qwen_span_metadata(span: &LensRenderedSpan) -> bool {
    let role = span.role.as_deref();
    let channel = span.channel.as_deref();
    if span.tool_call_index.is_some()
        || span.label.is_some()
        || !matches!(role, Some("system" | "user" | "assistant"))
        || !matches!(channel, None | Some("thinking"))
    {
        return false;
    }
    let source_or_synthetic_system = span.message_index.is_some() || role == Some("system");
    match span.kind.as_str() {
        "message_start_marker" | "role" | "message_end_marker" => {
            source_or_synthetic_system && channel.is_none()
        }
        "message_content" => span.message_index.is_some() && channel.is_none(),
        "reasoning_instruction_content" => {
            source_or_synthetic_system && role == Some("system") && channel == Some("thinking")
        }
        "generated_assistant_start_marker" | "generated_assistant_role" => {
            span.message_index.is_none() && role == Some("assistant") && channel.is_none()
        }
        "thinking_channel_start_marker" | "thinking_channel_end_marker" => {
            role == Some("assistant") && channel == Some("thinking")
        }
        "content_separator" => true,
        _ => false,
    }
}

pub(crate) fn valid_lens_rendering_topology(rendering: &LensInputRendering) -> bool {
    if rendering.renderer != "muse_glimmer_atem_annotated_v1" {
        return true;
    }
    valid_muse_rendering_topology(&rendering.spans)
}

fn valid_muse_rendering_topology(spans: &[LensRenderedSpan]) -> bool {
    if spans.first().is_none_or(|span| span.kind != "bos_marker") {
        return false;
    }
    let mut index = 1;
    let mut saw_source_message = false;
    let mut saw_synthetic_system = false;
    let mut last_source_message: Option<(usize, Option<String>)> = None;
    while index < spans.len() {
        if spans[index].kind == "generated_assistant_start_marker" {
            return saw_source_message
                && spans.get(index + 1).is_some_and(|span| {
                    span.kind == "generated_assistant_role"
                        && span.message_index.is_none()
                        && span.tool_call_index.is_none()
                        && span.role.as_deref() == Some("assistant")
                        && span.channel.is_none()
                })
                && index + 2 == spans.len();
        }
        let start = &spans[index];
        if start.kind != "message_start_marker" {
            return false;
        }
        match start.message_index {
            Some(message_index) => {
                if let Some((previous_index, previous_role)) = &last_source_message
                    && (message_index < *previous_index
                        || message_index > previous_index + 1
                        || (message_index == *previous_index
                            && start.role.as_ref() != previous_role.as_ref()))
                {
                    return false;
                }
                if last_source_message.is_none() && message_index != 0 {
                    return false;
                }
                last_source_message = Some((message_index, start.role.clone()));
                saw_source_message = true;
            }
            None => {
                if saw_source_message
                    || saw_synthetic_system
                    || start.role.as_deref() != Some("system")
                {
                    return false;
                }
                saw_synthetic_system = true;
            }
        }
        index += 1;
        let Some(role) = spans.get(index) else {
            return false;
        };
        if role.kind != "role" || !same_muse_record(start, role, false) {
            return false;
        }
        index += 1;
        let mut tool_name_count = 0;
        if spans
            .get(index)
            .is_some_and(|span| matches!(span.kind.as_str(), "recipient" | "tool_name"))
        {
            if !same_muse_record(start, &spans[index], false) {
                return false;
            }
            tool_name_count += usize::from(spans[index].kind == "tool_name");
            index += 1;
        }
        let Some(marker) = spans.get(index) else {
            return false;
        };
        if marker.kind != "message_marker" || !same_muse_record(start, marker, false) {
            return false;
        }
        index += 1;
        let mut saw_tool_call_content = false;
        loop {
            let Some(span) = spans.get(index) else {
                return false;
            };
            if span.kind == "message_end_marker" {
                if !same_muse_record(start, span, false) {
                    return false;
                }
                index += 1;
                break;
            }
            if !matches!(
                span.kind.as_str(),
                "tool_name"
                    | "message_content"
                    | "reasoning_instruction_content"
                    | "system_metadata_content"
                    | "tool_definition_content"
                    | "assistant_reasoning_content"
                    | "tool_call_content"
                    | "tool_result_content"
            ) || !same_muse_record(start, span, span.kind == "reasoning_instruction_content")
            {
                return false;
            }
            tool_name_count += usize::from(span.kind == "tool_name");
            saw_tool_call_content |= span.kind == "tool_call_content";
            index += 1;
        }
        if start.tool_call_index.is_some() && !saw_tool_call_content
            || start.role.as_deref() == Some("tool") && tool_name_count < 2
        {
            return false;
        }
    }
    false
}

fn same_muse_record(
    start: &LensRenderedSpan,
    span: &LensRenderedSpan,
    allow_system_reasoning_channel: bool,
) -> bool {
    start.message_index == span.message_index
        && start.tool_call_index == span.tool_call_index
        && start.role == span.role
        && (start.channel == span.channel
            || (allow_system_reasoning_channel
                && start.role.as_deref() == Some("system")
                && start.channel.is_none()
                && span.channel.as_deref() == Some("thinking")))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResolvedMessageMode {
    Qwen36(QwenGenerationMode),
    Qwen38(Qwen38GenerationMode),
    Muse(MuseGlimmerReasoningStrength),
}

impl ResolvedMessageMode {
    const fn artifact_name(self) -> &'static str {
        match self {
            Self::Qwen36(QwenGenerationMode::Auto) => "auto",
            Self::Qwen36(QwenGenerationMode::Thinking) => "thinking",
            Self::Qwen36(QwenGenerationMode::NoThinking) => "no_thinking",
            Self::Qwen38(Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Low)) => {
                "thinking_low"
            }
            Self::Qwen38(Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Medium)) => {
                "thinking_medium"
            }
            Self::Qwen38(Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Xhigh)) => {
                "thinking_xhigh"
            }
            Self::Qwen38(Qwen38GenerationMode::NoThinking) => "no_thinking",
            Self::Muse(MuseGlimmerReasoningStrength::Low) => "reasoning_low",
            Self::Muse(MuseGlimmerReasoningStrength::Medium) => "reasoning_medium",
            Self::Muse(MuseGlimmerReasoningStrength::High) => "reasoning_high",
        }
    }
}

pub(crate) fn validate_lens_input_spec(spec: LensInputSpec<'_>) -> Result<()> {
    let input_count = usize::from(spec.prompt.is_some())
        + usize::from(spec.token_ids.is_some())
        + usize::from(spec.user.is_some())
        + usize::from(spec.messages.is_some());
    ensure!(
        input_count == 1,
        "specify exactly one of --prompt/--raw-prompt, --token-ids, --user, or --messages"
    );
    ensure!(
        spec.user.is_some() || spec.system.is_none(),
        "--system requires --user"
    );
    ensure!(
        spec.prompt.is_some() || !spec.no_special_tokens,
        "--no-special-tokens only applies to --prompt/--raw-prompt"
    );
    ensure!(
        spec.user.is_some() || spec.messages.is_some() || spec.message_mode.is_none(),
        "--message-mode only applies to --user or --messages"
    );
    ensure!(
        spec.token_ids.is_none_or(|ids| !ids.is_empty()),
        "--token-ids must not be empty"
    );
    Ok(())
}

pub(crate) fn detect_qwen_message_protocol(
    family: ModelFamily,
    gguf: &GgufFile,
) -> Result<QwenMessageProtocol> {
    if family == ModelFamily::Qwen4Exp {
        ensure!(
            supports_qwen4exp_prompt_protocol(family, gguf),
            "Flash-Next structured input requires the released qwen35 prompt protocol; use --prompt/--raw-prompt or --token-ids for exact untemplated input"
        );
        return Ok(QwenMessageProtocol::Qwen38);
    }
    if supports_qwen38_release_prompt_protocol(family, gguf) {
        return Ok(QwenMessageProtocol::Qwen38);
    }
    if [
        gguf.get_str("general.name"),
        gguf.get_str("general.base_model.0.name"),
    ]
    .into_iter()
    .flatten()
    .any(|name| name.to_ascii_lowercase().contains("qwen3.8"))
    {
        bail!(
            "model declares Qwen3.8 but does not match the validated release prompt identity; use --prompt/--raw-prompt or --token-ids for exact untemplated input"
        );
    }
    if supports_qwen36_no_thinking_prompt_protocol(family, gguf) {
        return Ok(QwenMessageProtocol::Qwen36);
    }
    Ok(QwenMessageProtocol::Generic)
}

pub(crate) fn prepare_qwen_input(
    spec: LensInputSpec<'_>,
    protocol: QwenMessageProtocol,
    tokenizer: &Tokenizer,
) -> Result<PreparedLensInput> {
    validate_lens_input_spec(spec)?;
    match (spec.prompt, spec.token_ids, spec.user, spec.messages) {
        (Some(prompt), None, None, None) => {
            let add_special_tokens = !spec.no_special_tokens;
            let token_ids = tokenizer
                .encode(prompt, add_special_tokens)
                .context("tokenize raw Lens prompt")?;
            Ok(PreparedLensInput {
                source: "prompt",
                add_special_tokens: Some(add_special_tokens),
                token_ids,
                rendering: LensInputRendering {
                    renderer: "tokenizer_text".into(),
                    generation_mode: None,
                    spans: Vec::new(),
                },
            })
        }
        (None, Some(token_ids), None, None) => {
            validate_literal_token_ids(token_ids, tokenizer.n_vocab())?;
            Ok(PreparedLensInput {
                source: "token_ids",
                add_special_tokens: None,
                token_ids: token_ids.to_vec(),
                rendering: LensInputRendering {
                    renderer: "literal_token_ids".into(),
                    generation_mode: None,
                    spans: Vec::new(),
                },
            })
        }
        (None, None, user, messages_path) => {
            let messages = acquire_structured_messages(user, spec.system, messages_path)?;
            let (rendered, renderer, mode) =
                render_qwen_structured_messages(&messages, protocol, spec.message_mode)?;
            let token_ids = tokenizer
                .encode(&rendered.text, false)
                .context("tokenize exact rendered Lens messages")?;
            let spans = align_rendered_message_spans(tokenizer, &rendered, &token_ids)?;
            Ok(PreparedLensInput {
                source: "messages",
                add_special_tokens: Some(false),
                token_ids,
                rendering: LensInputRendering {
                    renderer: renderer.into(),
                    generation_mode: Some(mode.artifact_name().into()),
                    spans,
                },
            })
        }
        _ => bail!("invalid Lens input selection"),
    }
}

pub(crate) fn prepare_qwen_model_input(
    spec: LensInputSpec<'_>,
    family: ModelFamily,
    gguf: &GgufFile,
    tokenizer: &Tokenizer,
) -> Result<PreparedLensInput> {
    let protocol = if spec.user.is_some() || spec.messages.is_some() {
        detect_qwen_message_protocol(family, gguf)?
    } else {
        QwenMessageProtocol::Generic
    };
    prepare_qwen_input(spec, protocol, tokenizer)
}

pub(crate) fn prepare_muse_input(
    spec: LensInputSpec<'_>,
    profile: MuseGlimmerChatTemplateProfile,
    tokenizer: &LlamaCppTokenizer,
    vocab_size: u32,
) -> Result<PreparedLensInput> {
    validate_lens_input_spec(spec)?;
    match (spec.prompt, spec.token_ids, spec.user, spec.messages) {
        (Some(prompt), None, None, None) => {
            let add_special_tokens = !spec.no_special_tokens;
            let token_ids = tokenizer
                .encode(prompt, add_special_tokens)
                .context("tokenize raw Muse Lens prompt")?;
            Ok(PreparedLensInput {
                source: "prompt",
                add_special_tokens: Some(add_special_tokens),
                token_ids,
                rendering: LensInputRendering {
                    renderer: "muse_tokenizer_raw_prompt".into(),
                    generation_mode: None,
                    spans: Vec::new(),
                },
            })
        }
        (None, Some(token_ids), None, None) => {
            validate_literal_token_ids(token_ids, vocab_size)?;
            Ok(PreparedLensInput {
                source: "token_ids",
                add_special_tokens: None,
                token_ids: token_ids.to_vec(),
                rendering: LensInputRendering {
                    renderer: "literal_token_ids".into(),
                    generation_mode: None,
                    spans: Vec::new(),
                },
            })
        }
        (None, None, user, messages_path) => {
            let messages = acquire_structured_messages(user, spec.system, messages_path)?;
            let (rendered, mode) =
                render_muse_structured_messages(messages, profile, spec.message_mode)?;
            let token_ids = tokenizer
                .encode(&rendered.text, false)
                .context("tokenize exact Muse Glimmer ATEM prompt")?;
            ensure!(
                token_ids
                    .iter()
                    .all(|&token| token >= 0 && (token as u32) < vocab_size),
                "rendered Muse Lens messages contain a token outside vocabulary {vocab_size}"
            );
            let spans = align_muse_rendered_spans(tokenizer, &rendered, &token_ids)?;
            Ok(PreparedLensInput {
                source: "messages",
                add_special_tokens: Some(false),
                token_ids,
                rendering: LensInputRendering {
                    renderer: "muse_glimmer_atem_annotated_v1".into(),
                    generation_mode: Some(mode.artifact_name().into()),
                    spans,
                },
            })
        }
        _ => bail!("invalid Muse Lens input selection"),
    }
}

fn render_qwen_structured_messages(
    messages: &[ChatMessage],
    protocol: QwenMessageProtocol,
    requested: Option<LensMessageMode>,
) -> Result<(AnnotatedMessageRender, &'static str, ResolvedMessageMode)> {
    let mode = resolve_qwen_message_mode(protocol, requested)?;
    let (rendered, renderer) = match mode {
        ResolvedMessageMode::Qwen36(mode) => (
            render_qwen_messages_prompt_with_generation_annotated(messages, false, true, mode),
            match protocol {
                QwenMessageProtocol::Generic => "qwen_chatml_messages_v1",
                QwenMessageProtocol::Qwen36 => "qwen3.6_messages_v1",
                QwenMessageProtocol::Qwen38 => unreachable!(),
            },
        ),
        ResolvedMessageMode::Qwen38(mode) => (
            render_qwen38_messages_prompt_with_generation_annotated(messages, true, mode),
            "qwen3.8_messages_v1",
        ),
        ResolvedMessageMode::Muse(_) => unreachable!(),
    };
    Ok((rendered, renderer, mode))
}

fn render_muse_structured_messages(
    messages: Vec<ChatMessage>,
    profile: MuseGlimmerChatTemplateProfile,
    requested: Option<LensMessageMode>,
) -> Result<(AnnotatedMuseGlimmerPrompt, ResolvedMessageMode)> {
    let messages = messages
        .into_iter()
        .enumerate()
        .map(|(index, message)| match message.role.as_str() {
            "system" => Ok(MuseGlimmerMessage::system(message.content)),
            "user" => Ok(MuseGlimmerMessage::user(message.content)),
            "assistant" => Ok(MuseGlimmerMessage::assistant(message.content)),
            role => bail!("Muse Glimmer message {index} has unsupported role {role:?}"),
        })
        .collect::<Result<Vec<_>>>()?;
    let mode = resolve_muse_message_mode(requested)?;
    let ResolvedMessageMode::Muse(reasoning_strength) = mode else {
        unreachable!()
    };
    let options = MuseGlimmerPromptOptions {
        profile,
        reasoning_strength,
        ..MuseGlimmerPromptOptions::default()
    };
    let rendered = render_muse_glimmer_atem_prompt_annotated(&messages, &options)
        .context("render Muse Glimmer ATEM Lens messages")?;
    Ok((rendered, mode))
}

fn acquire_structured_messages(
    user: Option<&str>,
    system: Option<&str>,
    messages_path: Option<&Path>,
) -> Result<Vec<ChatMessage>> {
    match (user, messages_path) {
        (Some(user), None) => {
            let user = read_stdin_sentinel(user, "user message")?;
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
                content: user,
                ..Default::default()
            });
            Ok(messages)
        }
        (None, Some(path)) => {
            let source = path.display().to_string();
            let raw = if path == Path::new("-") {
                let mut raw = String::new();
                std::io::stdin()
                    .read_to_string(&mut raw)
                    .context("read messages from stdin")?;
                raw
            } else {
                std::fs::read_to_string(path)
                    .with_context(|| format!("read messages {}", path.display()))?
            };
            parse_strict_messages_input(&raw, &source)
        }
        _ => bail!("structured Lens input requires exactly one of --user or --messages"),
    }
}

fn read_stdin_sentinel(value: &str, label: &str) -> Result<String> {
    if value != "-" {
        return Ok(value.into());
    }
    let mut value = String::new();
    std::io::stdin()
        .read_to_string(&mut value)
        .with_context(|| format!("read {label} from stdin"))?;
    Ok(value)
}

fn resolve_qwen_message_mode(
    protocol: QwenMessageProtocol,
    requested: Option<LensMessageMode>,
) -> Result<ResolvedMessageMode> {
    match protocol {
        QwenMessageProtocol::Generic => match requested {
            None | Some(LensMessageMode::Auto) => {
                Ok(ResolvedMessageMode::Qwen36(QwenGenerationMode::Auto))
            }
            Some(mode) => bail!(
                "--message-mode {} is not validated for this Qwen prompt protocol; omit it or use auto",
                mode.to_possible_value().unwrap().get_name()
            ),
        },
        QwenMessageProtocol::Qwen36 => match requested {
            None | Some(LensMessageMode::Auto) => {
                Ok(ResolvedMessageMode::Qwen36(QwenGenerationMode::Auto))
            }
            Some(LensMessageMode::Thinking) => {
                Ok(ResolvedMessageMode::Qwen36(QwenGenerationMode::Thinking))
            }
            Some(LensMessageMode::NoThinking) => {
                Ok(ResolvedMessageMode::Qwen36(QwenGenerationMode::NoThinking))
            }
            Some(mode) => bail!(
                "--message-mode {} is not supported by Qwen3.6; use auto, thinking, or no-thinking",
                mode.to_possible_value().unwrap().get_name()
            ),
        },
        QwenMessageProtocol::Qwen38 => match requested {
            None | Some(LensMessageMode::Thinking | LensMessageMode::Xhigh) => {
                Ok(ResolvedMessageMode::Qwen38(Qwen38GenerationMode::Thinking(
                    Qwen38ReasoningEffort::Xhigh,
                )))
            }
            Some(LensMessageMode::Low) => Ok(ResolvedMessageMode::Qwen38(
                Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Low),
            )),
            Some(LensMessageMode::Medium) => Ok(ResolvedMessageMode::Qwen38(
                Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Medium),
            )),
            Some(LensMessageMode::NoThinking) => Ok(ResolvedMessageMode::Qwen38(
                Qwen38GenerationMode::NoThinking,
            )),
            Some(LensMessageMode::Auto) => bail!(
                "--message-mode auto is not supported by Qwen3.8; use thinking, no-thinking, low, medium, or xhigh"
            ),
        },
    }
}

fn resolve_muse_message_mode(requested: Option<LensMessageMode>) -> Result<ResolvedMessageMode> {
    match requested {
        None | Some(LensMessageMode::Thinking | LensMessageMode::Xhigh) => Ok(
            ResolvedMessageMode::Muse(MuseGlimmerReasoningStrength::High),
        ),
        Some(LensMessageMode::Low) => {
            Ok(ResolvedMessageMode::Muse(MuseGlimmerReasoningStrength::Low))
        }
        Some(LensMessageMode::Medium) => Ok(ResolvedMessageMode::Muse(
            MuseGlimmerReasoningStrength::Medium,
        )),
        Some(LensMessageMode::Auto | LensMessageMode::NoThinking) => bail!(
            "Muse Glimmer supports message modes thinking, low, medium, or xhigh; it declares no auto or no-thinking ATEM profile"
        ),
    }
}

fn validate_literal_token_ids(token_ids: &[i32], vocab_size: u32) -> Result<()> {
    ensure!(!token_ids.is_empty(), "--token-ids must not be empty");
    ensure!(
        token_ids
            .iter()
            .all(|&token| token >= 0 && (token as u32) < vocab_size),
        "--token-ids contains an ID outside the tokenizer vocabulary"
    );
    Ok(())
}

fn align_rendered_message_spans(
    tokenizer: &Tokenizer,
    rendered: &AnnotatedMessageRender,
    full_token_ids: &[i32],
) -> Result<Vec<LensRenderedSpan>> {
    let token_pieces = full_token_ids
        .iter()
        .map(|&token_id| {
            tokenizer
                .try_decode_piece_bytes_exact(token_id)
                .with_context(|| format!("decode rendered token {token_id}"))
        })
        .collect::<Result<Vec<_>>>()?;
    map_rendered_message_spans(rendered, &token_pieces)
}

fn align_muse_rendered_spans(
    tokenizer: &LlamaCppTokenizer,
    rendered: &AnnotatedMuseGlimmerPrompt,
    full_token_ids: &[i32],
) -> Result<Vec<LensRenderedSpan>> {
    let token_pieces = full_token_ids
        .iter()
        .map(|&token_id| {
            tokenizer
                .try_decode_piece_bytes_exact(token_id)
                .with_context(|| format!("decode rendered Muse token {token_id}"))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        token_pieces.iter().all(|piece| !piece.is_empty()),
        "rendered Muse tokens contain an empty exact piece"
    );
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(rendered.text.len())
        .context("allocate Muse rendered-token validation")?;
    let mut token_boundaries = BTreeMap::from([(0usize, 0usize)]);
    for (token_index, piece) in token_pieces.iter().enumerate() {
        decoded.extend_from_slice(piece);
        token_boundaries.insert(decoded.len(), token_index + 1);
    }
    ensure!(
        decoded == rendered.text.as_bytes(),
        "decoded Muse tokens do not reproduce the ATEM renderer-authored bytes"
    );

    rendered
        .spans
        .iter()
        .map(|span| {
            ensure!(
                span.byte_start < span.byte_end
                    && span.byte_end <= rendered.text.len()
                    && rendered.text.is_char_boundary(span.byte_start)
                    && rendered.text.is_char_boundary(span.byte_end),
                "ATEM renderer produced an invalid byte span {}..{} for {} bytes",
                span.byte_start,
                span.byte_end,
                rendered.text.len()
            );
            let token_range = token_boundaries
                .get(&span.byte_start)
                .zip(token_boundaries.get(&span.byte_end))
                .filter(|(start, end)| start < end)
                .map(|(&start, &end)| (start, end));
            if is_structural_muse_span(span.kind) {
                ensure!(
                    token_range.is_some(),
                    "ATEM structural {} span {}..{} is not a nonempty exact token range",
                    span.kind.as_str(),
                    span.byte_start,
                    span.byte_end
                );
            }
            let label = matches!(
                span.kind,
                MuseGlimmerPromptSpanKind::Recipient
                    | MuseGlimmerPromptSpanKind::ToolName
                    | MuseGlimmerPromptSpanKind::MessageEndMarker
            )
            .then(|| rendered.text[span.byte_start..span.byte_end].to_owned());
            Ok(LensRenderedSpan {
                kind: span.kind.as_str().into(),
                message_index: span.message_index,
                tool_call_index: span.tool_call_index,
                role: span.role.map(|role| role.as_str().into()),
                channel: span.channel.map(|channel| channel.as_str().into()),
                label,
                byte_start: span.byte_start,
                byte_end: span.byte_end,
                token_start: token_range.map(|range| range.0),
                token_end: token_range.map(|range| range.1),
            })
        })
        .collect()
}

fn is_structural_muse_span(kind: MuseGlimmerPromptSpanKind) -> bool {
    matches!(
        kind,
        MuseGlimmerPromptSpanKind::BosMarker
            | MuseGlimmerPromptSpanKind::MessageStartMarker
            | MuseGlimmerPromptSpanKind::MessageMarker
            | MuseGlimmerPromptSpanKind::MessageEndMarker
            | MuseGlimmerPromptSpanKind::GeneratedAssistantStartMarker
    )
}

fn map_rendered_message_spans(
    rendered: &AnnotatedMessageRender,
    token_pieces: &[&[u8]],
) -> Result<Vec<LensRenderedSpan>> {
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(rendered.text.len())
        .context("allocate rendered token byte validation")?;
    let mut token_boundaries = BTreeMap::from([(0usize, 0usize)]);
    for (token_index, piece) in token_pieces.iter().enumerate() {
        decoded.extend_from_slice(piece);
        token_boundaries.insert(decoded.len(), token_index + 1);
    }
    ensure!(
        decoded == rendered.text.as_bytes(),
        "decoded rendered tokens do not reproduce the renderer-authored bytes"
    );

    for span in &rendered.spans {
        ensure!(
            span.byte_start < span.byte_end
                && span.byte_end <= rendered.text.len()
                && rendered.text.is_char_boundary(span.byte_start)
                && rendered.text.is_char_boundary(span.byte_end),
            "renderer produced an invalid byte span {}..{} for {} bytes",
            span.byte_start,
            span.byte_end,
            rendered.text.len()
        );
    }

    rendered
        .spans
        .iter()
        .map(|span| {
            let token_range = token_boundaries
                .get(&span.byte_start)
                .zip(token_boundaries.get(&span.byte_end))
                .filter(|(start, end)| start < end)
                .map(|(&start, &end)| (start, end));
            if is_structural_render_span(span.kind) {
                ensure!(
                    token_range.is_some(),
                    "renderer structural {} span {}..{} is not a nonempty exact token range",
                    span.kind.as_str(),
                    span.byte_start,
                    span.byte_end
                );
            }
            Ok(LensRenderedSpan {
                kind: span.kind.as_str().into(),
                message_index: span.message_index,
                tool_call_index: None,
                role: span.role.clone(),
                channel: span.channel.map(|channel| channel.as_str().into()),
                label: None,
                byte_start: span.byte_start,
                byte_end: span.byte_end,
                token_start: token_range.map(|range| range.0),
                token_end: token_range.map(|range| range.1),
            })
        })
        .collect()
}

fn is_structural_render_span(kind: MessageRenderSpanKind) -> bool {
    matches!(
        kind,
        MessageRenderSpanKind::MessageStartMarker
            | MessageRenderSpanKind::MessageEndMarker
            | MessageRenderSpanKind::GeneratedAssistantStartMarker
            | MessageRenderSpanKind::ThinkingChannelStartMarker
            | MessageRenderSpanKind::ThinkingChannelEndMarker
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen_llm::muse_glimmer_prompt::render_muse_glimmer_atem_prompt;

    #[test]
    fn qwen38_defaults_match_modern_run_xhigh_contract() {
        let mode = resolve_qwen_message_mode(QwenMessageProtocol::Qwen38, None).unwrap();
        assert_eq!(mode.artifact_name(), "thinking_xhigh");
        assert_eq!(
            resolve_qwen_message_mode(QwenMessageProtocol::Qwen38, Some(LensMessageMode::Thinking))
                .unwrap()
                .artifact_name(),
            "thinking_xhigh"
        );
        assert_eq!(
            resolve_qwen_message_mode(QwenMessageProtocol::Qwen38, Some(LensMessageMode::Medium))
                .unwrap()
                .artifact_name(),
            "thinking_medium"
        );
    }

    #[test]
    fn structured_renderers_are_the_normal_run_renderer_functions() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: "policy".into(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".into(),
                content: "request".into(),
                ..Default::default()
            },
        ];
        let (qwen38, renderer, mode) =
            render_qwen_structured_messages(&messages, QwenMessageProtocol::Qwen38, None).unwrap();
        assert_eq!(renderer, "qwen3.8_messages_v1");
        assert_eq!(mode.artifact_name(), "thinking_xhigh");
        assert_eq!(
            qwen38,
            render_qwen38_messages_prompt_with_generation_annotated(
                &messages,
                true,
                Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Xhigh),
            )
        );

        let (generic, renderer, mode) =
            render_qwen_structured_messages(&messages, QwenMessageProtocol::Generic, None).unwrap();
        assert_eq!(renderer, "qwen_chatml_messages_v1");
        assert_eq!(mode.artifact_name(), "auto");
        assert_eq!(
            generic,
            render_qwen_messages_prompt_with_generation_annotated(
                &messages,
                false,
                true,
                QwenGenerationMode::Auto,
            )
        );

        let (muse, mode) = render_muse_structured_messages(
            messages.clone(),
            MuseGlimmerChatTemplateProfile::UnslothLaunch,
            None,
        )
        .unwrap();
        assert_eq!(mode.artifact_name(), "reasoning_high");
        let muse_messages = vec![
            MuseGlimmerMessage::system("policy"),
            MuseGlimmerMessage::user("request"),
        ];
        assert_eq!(
            muse.text,
            render_muse_glimmer_atem_prompt(&muse_messages, &MuseGlimmerPromptOptions::default(),)
                .unwrap()
        );
        assert!(muse.spans.iter().any(|span| {
            span.kind == MuseGlimmerPromptSpanKind::MessageStartMarker
                && span.message_index == Some(0)
                && span.role.map(|role| role.as_str()) == Some("system")
        }));
        assert!(muse.spans.iter().any(|span| {
            span.kind == MuseGlimmerPromptSpanKind::MessageStartMarker
                && span.message_index == Some(1)
                && span.role.map(|role| role.as_str()) == Some("user")
        }));
        assert!(muse.spans.iter().any(|span| {
            span.kind == MuseGlimmerPromptSpanKind::GeneratedAssistantStartMarker
                && span.message_index.is_none()
        }));
    }

    #[test]
    fn rich_muse_annotations_form_complete_valid_atem_records() {
        let mut assistant = MuseGlimmerMessage::assistant("");
        assistant.reasoning_content = Some("check both".into());
        assistant.tool_calls = vec![
            qwen_llm::muse_glimmer_prompt::MuseGlimmerToolCall {
                name: "first".into(),
                arguments: serde_json::json!({"x": 1}),
            },
            qwen_llm::muse_glimmer_prompt::MuseGlimmerToolCall {
                name: "second".into(),
                arguments: serde_json::json!({"y": 2}),
            },
        ];
        let rendered = render_muse_glimmer_atem_prompt_annotated(
            &[
                MuseGlimmerMessage::user("go"),
                assistant,
                MuseGlimmerMessage::tool("second", "done"),
            ],
            &MuseGlimmerPromptOptions::default(),
        )
        .unwrap();
        let spans = rendered
            .spans
            .iter()
            .map(|span| {
                let label = matches!(
                    span.kind,
                    MuseGlimmerPromptSpanKind::Recipient
                        | MuseGlimmerPromptSpanKind::ToolName
                        | MuseGlimmerPromptSpanKind::MessageEndMarker
                )
                .then(|| rendered.text[span.byte_start..span.byte_end].to_owned());
                LensRenderedSpan {
                    kind: span.kind.as_str().into(),
                    message_index: span.message_index,
                    tool_call_index: span.tool_call_index,
                    role: span.role.map(|role| role.as_str().into()),
                    channel: span.channel.map(|channel| channel.as_str().into()),
                    label,
                    byte_start: span.byte_start,
                    byte_end: span.byte_end,
                    token_start: None,
                    token_end: None,
                }
            })
            .collect::<Vec<_>>();
        let rendering = LensInputRendering {
            renderer: "muse_glimmer_atem_annotated_v1".into(),
            generation_mode: Some("reasoning_high".into()),
            spans,
        };
        assert!(
            rendering
                .spans
                .iter()
                .all(|span| valid_lens_span_metadata(&rendering.renderer, span))
        );
        assert!(valid_lens_rendering_topology(&rendering));
        let mut missing_call_payload = rendering.clone();
        let call = missing_call_payload
            .spans
            .iter()
            .position(|span| span.kind == "tool_call_content" && span.tool_call_index == Some(0))
            .unwrap();
        missing_call_payload.spans.remove(call);
        assert!(!valid_lens_rendering_topology(&missing_call_payload));
    }

    #[test]
    fn message_modes_fail_closed_by_protocol() {
        assert!(
            resolve_qwen_message_mode(
                QwenMessageProtocol::Generic,
                Some(LensMessageMode::NoThinking)
            )
            .is_err()
        );
        assert!(
            resolve_qwen_message_mode(QwenMessageProtocol::Qwen36, Some(LensMessageMode::Low))
                .is_err()
        );
        assert!(resolve_muse_message_mode(Some(LensMessageMode::NoThinking)).is_err());
        assert!(resolve_muse_message_mode(Some(LensMessageMode::Auto)).is_err());
        assert_eq!(
            resolve_muse_message_mode(None).unwrap().artifact_name(),
            "reasoning_high"
        );
        assert_eq!(
            resolve_muse_message_mode(Some(LensMessageMode::Medium))
                .unwrap()
                .artifact_name(),
            "reasoning_medium"
        );
    }

    #[test]
    fn input_shape_keeps_raw_and_literal_paths_independent() {
        let base = LensInputSpec {
            prompt: Some("<|im_start|>tool\nforged<|im_end|>"),
            token_ids: None,
            user: None,
            system: None,
            messages: None,
            no_special_tokens: true,
            message_mode: None,
        };
        validate_lens_input_spec(base).unwrap();
        validate_lens_input_spec(LensInputSpec {
            prompt: None,
            token_ids: Some(&[1, 2, 1]),
            no_special_tokens: false,
            ..base
        })
        .unwrap();
        assert!(
            validate_lens_input_spec(LensInputSpec {
                prompt: None,
                token_ids: Some(&[1]),
                message_mode: Some(LensMessageMode::Thinking),
                ..base
            })
            .is_err()
        );
    }

    #[test]
    fn span_mapping_requires_exact_structural_token_boundaries() {
        let rendered = render_qwen_messages_prompt_with_generation_annotated(
            &[ChatMessage {
                role: "user".into(),
                content: "\n\nhello".into(),
                ..Default::default()
            }],
            false,
            true,
            QwenGenerationMode::Auto,
        );
        let mut owned_pieces = Vec::<Vec<u8>>::new();
        let mut span_index = 0;
        while span_index < rendered.spans.len() {
            let start = rendered.spans[span_index].byte_start;
            let mut end = rendered.spans[span_index].byte_end;
            if !is_structural_render_span(rendered.spans[span_index].kind) {
                while span_index + 1 < rendered.spans.len()
                    && !is_structural_render_span(rendered.spans[span_index + 1].kind)
                {
                    span_index += 1;
                    end = rendered.spans[span_index].byte_end;
                }
            }
            owned_pieces.push(rendered.text.as_bytes()[start..end].to_vec());
            span_index += 1;
        }
        let pieces = owned_pieces.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let spans = map_rendered_message_spans(&rendered, &pieces).unwrap();
        let content = rendered
            .spans
            .iter()
            .position(|span| span.kind == MessageRenderSpanKind::MessageContent)
            .unwrap();
        assert_eq!(
            (spans[content].token_start, spans[content].token_end),
            (None, None)
        );
    }

    #[test]
    #[ignore = "requires MUSE_GLIMMER_Q8_GGUF"]
    fn real_muse_atem_spans_align_to_exact_special_token_boundaries() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF").expect("set MUSE_GLIMMER_Q8_GGUF");
        let gguf = GgufFile::open(&path).unwrap();
        let bound = qwen_llm::muse_glimmer::MuseGlimmerModel::from_gguf(&gguf).unwrap();
        let tokenizer = LlamaCppTokenizer::from_gguf(&gguf, &path).unwrap();
        let prepared = prepare_muse_input(
            LensInputSpec {
                prompt: None,
                token_ids: None,
                user: Some("Inspect this boundary."),
                system: Some("Keep the channel structure exact."),
                messages: None,
                no_special_tokens: false,
                message_mode: Some(LensMessageMode::Medium),
            },
            bound.config.chat_template_profile,
            &tokenizer,
            bound.config.vocab_size,
        )
        .unwrap();
        assert_eq!(prepared.source, "messages");
        assert_eq!(prepared.add_special_tokens, Some(false));
        assert_eq!(
            prepared.rendering.renderer,
            "muse_glimmer_atem_annotated_v1"
        );
        assert_eq!(
            prepared.rendering.generation_mode.as_deref(),
            Some("reasoning_medium")
        );
        assert!(valid_lens_rendering_topology(&prepared.rendering));
        assert!(prepared.rendering.spans.iter().any(|span| {
            span.kind == "message_start_marker"
                && span.message_index == Some(1)
                && span.role.as_deref() == Some("user")
                && span.token_start.is_some()
        }));
        assert!(prepared.rendering.spans.iter().any(|span| {
            span.kind == "generated_assistant_start_marker"
                && span.token_start.is_some()
                && span.token_end.is_some()
        }));
    }
}
