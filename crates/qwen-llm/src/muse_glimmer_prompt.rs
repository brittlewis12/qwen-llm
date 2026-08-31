//! Byte-explicit Muse Glimmer ATEM prompt rendering.

use crate::muse_glimmer::MuseGlimmerChatTemplateProfile;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const MUSE_GLIMMER_BOS: &str = "<|begin_of_text|>";
pub const MUSE_GLIMMER_START: &str = "<|start|>";
pub const MUSE_GLIMMER_MESSAGE: &str = "<|message|>";
pub const MUSE_GLIMMER_EOM: &str = "<|eom|>";
pub const MUSE_GLIMMER_EOT: &str = "<|eot|>";

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MuseGlimmerReasoningStrength {
    Low,
    Medium,
    #[default]
    High,
    Xhigh,
}

impl MuseGlimmerReasoningStrength {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MuseGlimmerMessageRole {
    System,
    User,
    Assistant,
    Tool { name: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerToolCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerMessage {
    pub role: MuseGlimmerMessageRole,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub recipient: Option<String>,
    pub end_turn: Option<bool>,
    pub tool_calls: Vec<MuseGlimmerToolCall>,
}

impl MuseGlimmerMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::plain(MuseGlimmerMessageRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::plain(MuseGlimmerMessageRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::plain(MuseGlimmerMessageRole::Assistant, content)
    }

    pub fn tool(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self::plain(MuseGlimmerMessageRole::Tool { name: name.into() }, content)
    }

    fn plain(role: MuseGlimmerMessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning_content: None,
            recipient: None,
            end_turn: None,
            tool_calls: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerPromptOptions {
    pub profile: MuseGlimmerChatTemplateProfile,
    pub reasoning_strength: MuseGlimmerReasoningStrength,
    pub knowledge_cutoff: String,
    pub current_date: String,
    pub tools: Vec<MuseGlimmerToolDefinition>,
    pub tool_namespace_descriptions: BTreeMap<String, String>,
    pub add_generation_prompt: bool,
}

impl Default for MuseGlimmerPromptOptions {
    fn default() -> Self {
        Self {
            profile: MuseGlimmerChatTemplateProfile::UnslothLaunch,
            reasoning_strength: MuseGlimmerReasoningStrength::High,
            knowledge_cutoff: "2026-01-04".into(),
            current_date: "2026-08-29".into(),
            tools: Vec::new(),
            tool_namespace_descriptions: BTreeMap::new(),
            add_generation_prompt: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MuseGlimmerPromptSpanKind {
    BosMarker,
    MessageStartMarker,
    Role,
    Recipient,
    ToolName,
    MessageMarker,
    MessageContent,
    ReasoningInstructionContent,
    SystemMetadataContent,
    ToolDefinitionContent,
    AssistantReasoningContent,
    ToolCallContent,
    ToolResultContent,
    MessageEndMarker,
    GeneratedAssistantStartMarker,
    GeneratedAssistantRole,
}

impl MuseGlimmerPromptSpanKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BosMarker => "bos_marker",
            Self::MessageStartMarker => "message_start_marker",
            Self::Role => "role",
            Self::Recipient => "recipient",
            Self::ToolName => "tool_name",
            Self::MessageMarker => "message_marker",
            Self::MessageContent => "message_content",
            Self::ReasoningInstructionContent => "reasoning_instruction_content",
            Self::SystemMetadataContent => "system_metadata_content",
            Self::ToolDefinitionContent => "tool_definition_content",
            Self::AssistantReasoningContent => "assistant_reasoning_content",
            Self::ToolCallContent => "tool_call_content",
            Self::ToolResultContent => "tool_result_content",
            Self::MessageEndMarker => "message_end_marker",
            Self::GeneratedAssistantStartMarker => "generated_assistant_start_marker",
            Self::GeneratedAssistantRole => "generated_assistant_role",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MuseGlimmerPromptSpanRole {
    System,
    User,
    Assistant,
    Tool,
}

impl MuseGlimmerPromptSpanRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MuseGlimmerPromptChannel {
    Thinking,
    ToolCall,
    ToolResult,
}

impl MuseGlimmerPromptChannel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Thinking => "thinking",
            Self::ToolCall => "tool_call",
            Self::ToolResult => "tool_result",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MuseGlimmerPromptSpan {
    pub kind: MuseGlimmerPromptSpanKind,
    pub message_index: Option<usize>,
    pub tool_call_index: Option<usize>,
    pub role: Option<MuseGlimmerPromptSpanRole>,
    pub channel: Option<MuseGlimmerPromptChannel>,
    pub byte_start: usize,
    pub byte_end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnnotatedMuseGlimmerPrompt {
    pub text: String,
    pub spans: Vec<MuseGlimmerPromptSpan>,
}

#[derive(Default)]
struct AnnotatedMuseGlimmerPromptBuilder {
    text: String,
    spans: Vec<MuseGlimmerPromptSpan>,
}

impl AnnotatedMuseGlimmerPromptBuilder {
    fn push(
        &mut self,
        text: &str,
        kind: MuseGlimmerPromptSpanKind,
        message_index: Option<usize>,
        tool_call_index: Option<usize>,
        role: Option<MuseGlimmerPromptSpanRole>,
        channel: Option<MuseGlimmerPromptChannel>,
    ) {
        if text.is_empty() {
            return;
        }
        let byte_start = self.text.len();
        self.text.push_str(text);
        self.spans.push(MuseGlimmerPromptSpan {
            kind,
            message_index,
            tool_call_index,
            role,
            channel,
            byte_start,
            byte_end: self.text.len(),
        });
    }

    fn raw(&mut self, text: &str) {
        self.text.push_str(text);
    }

    fn finish(self) -> AnnotatedMuseGlimmerPrompt {
        AnnotatedMuseGlimmerPrompt {
            text: self.text,
            spans: self.spans,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MuseGlimmerPromptError {
    #[error("Muse Glimmer prompt requires at least one message")]
    EmptyMessages,
    #[error("Muse Glimmer message {index} has invalid {field}: {detail}")]
    InvalidMessage {
        index: usize,
        field: &'static str,
        detail: String,
    },
    #[error("Muse Glimmer tool {index} has invalid {field}: {detail}")]
    InvalidTool {
        index: usize,
        field: &'static str,
        detail: String,
    },
    #[error("Muse Glimmer prompt options have invalid {field}: {detail}")]
    InvalidOptions { field: &'static str, detail: String },
    #[error("failed to serialize Muse Glimmer tool JSON: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn render_muse_glimmer_single_turn(
    user: &str,
    system: Option<&str>,
    options: &MuseGlimmerPromptOptions,
) -> Result<String, MuseGlimmerPromptError> {
    let mut messages = Vec::with_capacity(usize::from(system.is_some()) + 1);
    if let Some(system) = system {
        messages.push(MuseGlimmerMessage::system(system));
    }
    messages.push(MuseGlimmerMessage::user(user));
    render_muse_glimmer_atem_prompt(&messages, options)
}

pub fn render_muse_glimmer_atem_prompt(
    messages: &[MuseGlimmerMessage],
    options: &MuseGlimmerPromptOptions,
) -> Result<String, MuseGlimmerPromptError> {
    Ok(render_muse_glimmer_atem_prompt_annotated(messages, options)?.text)
}

pub fn render_muse_glimmer_atem_prompt_annotated(
    messages: &[MuseGlimmerMessage],
    options: &MuseGlimmerPromptOptions,
) -> Result<AnnotatedMuseGlimmerPrompt, MuseGlimmerPromptError> {
    if messages.is_empty() {
        return Err(MuseGlimmerPromptError::EmptyMessages);
    }
    validate_options(options)?;
    validate_messages(messages, options)?;

    let mut output = AnnotatedMuseGlimmerPromptBuilder::default();
    output.push(
        MUSE_GLIMMER_BOS,
        MuseGlimmerPromptSpanKind::BosMarker,
        None,
        None,
        None,
        None,
    );
    if !messages
        .iter()
        .any(|message| message.role == MuseGlimmerMessageRole::System)
    {
        push_muse_message_header(
            &mut output,
            None,
            None,
            MuseGlimmerPromptSpanRole::System,
            None,
            None,
        );
        let content = format!(
            "You are a helpful AI assistant.\nKnowledge cutoff: {}.\nCurrent date: {}.\n\n",
            options.knowledge_cutoff, options.current_date
        );
        output.push(
            &content,
            MuseGlimmerPromptSpanKind::MessageContent,
            None,
            None,
            Some(MuseGlimmerPromptSpanRole::System),
            None,
        );
        push_muse_reasoning_instruction(&mut output, None, options.reasoning_strength);
        if !options.tools.is_empty() {
            output.raw("\n\n");
            push_muse_tool_definitions(&mut output, None, options)?;
        }
        output.raw("\n\n");
        push_muse_system_meta(&mut output, None, options);
        push_muse_message_end(
            &mut output,
            MUSE_GLIMMER_EOT,
            None,
            None,
            MuseGlimmerPromptSpanRole::System,
            None,
        );
    }

    for (index, message) in messages.iter().enumerate() {
        let end_token = if messages
            .get(index + 1)
            .is_some_and(|next| next.role == message.role)
        {
            MUSE_GLIMMER_EOM
        } else {
            MUSE_GLIMMER_EOT
        };
        match &message.role {
            MuseGlimmerMessageRole::System => {
                push_muse_message_header(
                    &mut output,
                    Some(index),
                    None,
                    MuseGlimmerPromptSpanRole::System,
                    None,
                    None,
                );
                let normalized = normalize_reasoning_directive(
                    &message.content,
                    options.reasoning_strength,
                    index,
                )?;
                push_muse_system_message_content(&mut output, Some(index), &normalized);
                if normalized.directive_range.is_none() {
                    output.raw("\n\n");
                    push_muse_reasoning_instruction(
                        &mut output,
                        Some(index),
                        options.reasoning_strength,
                    );
                }
                if !options.tools.is_empty() {
                    output.raw("\n\n");
                    push_muse_tool_definitions(&mut output, Some(index), options)?;
                }
                output.raw("\n\n");
                push_muse_system_meta(&mut output, Some(index), options);
                push_muse_message_end(
                    &mut output,
                    MUSE_GLIMMER_EOT,
                    Some(index),
                    None,
                    MuseGlimmerPromptSpanRole::System,
                    None,
                );
            }
            MuseGlimmerMessageRole::User => {
                push_muse_message_header(
                    &mut output,
                    Some(index),
                    None,
                    MuseGlimmerPromptSpanRole::User,
                    None,
                    None,
                );
                output.push(
                    &message.content,
                    MuseGlimmerPromptSpanKind::MessageContent,
                    Some(index),
                    None,
                    Some(MuseGlimmerPromptSpanRole::User),
                    None,
                );
                push_muse_message_end(
                    &mut output,
                    MUSE_GLIMMER_EOT,
                    Some(index),
                    None,
                    MuseGlimmerPromptSpanRole::User,
                    None,
                );
            }
            MuseGlimmerMessageRole::Tool { name } => {
                let channel = Some(MuseGlimmerPromptChannel::ToolResult);
                push_muse_message_header(
                    &mut output,
                    Some(index),
                    None,
                    MuseGlimmerPromptSpanRole::Tool,
                    Some(name),
                    channel,
                );
                output.raw("<tool_output name=\"");
                output.push(
                    name,
                    MuseGlimmerPromptSpanKind::ToolName,
                    Some(index),
                    None,
                    Some(MuseGlimmerPromptSpanRole::Tool),
                    channel,
                );
                output.raw("\">\n");
                output.push(
                    &message.content,
                    MuseGlimmerPromptSpanKind::ToolResultContent,
                    Some(index),
                    None,
                    Some(MuseGlimmerPromptSpanRole::Tool),
                    channel,
                );
                output.raw("\n</tool_output>");
                push_muse_message_end(
                    &mut output,
                    MUSE_GLIMMER_EOT,
                    Some(index),
                    None,
                    MuseGlimmerPromptSpanRole::Tool,
                    channel,
                );
            }
            MuseGlimmerMessageRole::Assistant => {
                if let Some(reasoning) = message
                    .reasoning_content
                    .as_deref()
                    .filter(|reasoning| !reasoning.is_empty())
                {
                    let channel = Some(MuseGlimmerPromptChannel::Thinking);
                    push_muse_message_header(
                        &mut output,
                        Some(index),
                        None,
                        MuseGlimmerPromptSpanRole::Assistant,
                        Some("self"),
                        channel,
                    );
                    output.push(
                        reasoning,
                        MuseGlimmerPromptSpanKind::AssistantReasoningContent,
                        Some(index),
                        None,
                        Some(MuseGlimmerPromptSpanRole::Assistant),
                        channel,
                    );
                    push_muse_message_end(
                        &mut output,
                        MUSE_GLIMMER_EOM,
                        Some(index),
                        None,
                        MuseGlimmerPromptSpanRole::Assistant,
                        channel,
                    );
                }
                if message.tool_calls.is_empty() {
                    let recipient = message.recipient.as_deref().unwrap_or("user");
                    let end_turn = message.end_turn.unwrap_or(recipient == "user");
                    let channel = if recipient == "self" {
                        Some(MuseGlimmerPromptChannel::Thinking)
                    } else if recipient == "user" {
                        None
                    } else {
                        Some(MuseGlimmerPromptChannel::ToolCall)
                    };
                    push_muse_message_header(
                        &mut output,
                        Some(index),
                        None,
                        MuseGlimmerPromptSpanRole::Assistant,
                        Some(recipient),
                        channel,
                    );
                    output.push(
                        &message.content,
                        MuseGlimmerPromptSpanKind::MessageContent,
                        Some(index),
                        None,
                        Some(MuseGlimmerPromptSpanRole::Assistant),
                        channel,
                    );
                    push_muse_message_end(
                        &mut output,
                        if end_turn {
                            MUSE_GLIMMER_EOT
                        } else {
                            MUSE_GLIMMER_EOM
                        },
                        Some(index),
                        None,
                        MuseGlimmerPromptSpanRole::Assistant,
                        channel,
                    );
                } else {
                    for (call_index, call) in message.tool_calls.iter().enumerate() {
                        let channel = Some(MuseGlimmerPromptChannel::ToolCall);
                        push_muse_message_header(
                            &mut output,
                            Some(index),
                            Some(call_index),
                            MuseGlimmerPromptSpanRole::Assistant,
                            Some(&call.name),
                            channel,
                        );
                        let mut rendered_call = String::new();
                        render_atem_call(&mut rendered_call, call)?;
                        output.push(
                            &rendered_call,
                            MuseGlimmerPromptSpanKind::ToolCallContent,
                            Some(index),
                            Some(call_index),
                            Some(MuseGlimmerPromptSpanRole::Assistant),
                            channel,
                        );
                        push_muse_message_end(
                            &mut output,
                            if call_index + 1 == message.tool_calls.len() {
                                end_token
                            } else {
                                MUSE_GLIMMER_EOM
                            },
                            Some(index),
                            Some(call_index),
                            MuseGlimmerPromptSpanRole::Assistant,
                            channel,
                        );
                    }
                }
            }
        }
    }
    if options.add_generation_prompt {
        output.push(
            MUSE_GLIMMER_START,
            MuseGlimmerPromptSpanKind::GeneratedAssistantStartMarker,
            None,
            None,
            Some(MuseGlimmerPromptSpanRole::Assistant),
            None,
        );
        output.push(
            "assistant",
            MuseGlimmerPromptSpanKind::GeneratedAssistantRole,
            None,
            None,
            Some(MuseGlimmerPromptSpanRole::Assistant),
            None,
        );
    }
    Ok(output.finish())
}

fn push_muse_message_header(
    output: &mut AnnotatedMuseGlimmerPromptBuilder,
    message_index: Option<usize>,
    tool_call_index: Option<usize>,
    role: MuseGlimmerPromptSpanRole,
    recipient: Option<&str>,
    channel: Option<MuseGlimmerPromptChannel>,
) {
    output.push(
        MUSE_GLIMMER_START,
        MuseGlimmerPromptSpanKind::MessageStartMarker,
        message_index,
        tool_call_index,
        Some(role),
        channel,
    );
    output.push(
        role.as_str(),
        MuseGlimmerPromptSpanKind::Role,
        message_index,
        tool_call_index,
        Some(role),
        channel,
    );
    if let Some(recipient) = recipient {
        output.raw(match role {
            MuseGlimmerPromptSpanRole::Assistant => " to=",
            MuseGlimmerPromptSpanRole::Tool => " ",
            MuseGlimmerPromptSpanRole::System | MuseGlimmerPromptSpanRole::User => "",
        });
        output.push(
            recipient,
            if role == MuseGlimmerPromptSpanRole::Tool {
                MuseGlimmerPromptSpanKind::ToolName
            } else {
                MuseGlimmerPromptSpanKind::Recipient
            },
            message_index,
            tool_call_index,
            Some(role),
            channel,
        );
    }
    output.push(
        MUSE_GLIMMER_MESSAGE,
        MuseGlimmerPromptSpanKind::MessageMarker,
        message_index,
        tool_call_index,
        Some(role),
        channel,
    );
}

fn push_muse_message_end(
    output: &mut AnnotatedMuseGlimmerPromptBuilder,
    marker: &str,
    message_index: Option<usize>,
    tool_call_index: Option<usize>,
    role: MuseGlimmerPromptSpanRole,
    channel: Option<MuseGlimmerPromptChannel>,
) {
    output.push(
        marker,
        MuseGlimmerPromptSpanKind::MessageEndMarker,
        message_index,
        tool_call_index,
        Some(role),
        channel,
    );
}

fn push_muse_reasoning_instruction(
    output: &mut AnnotatedMuseGlimmerPromptBuilder,
    message_index: Option<usize>,
    strength: MuseGlimmerReasoningStrength,
) {
    let text = format!("Reasoning strength: {}.", strength.as_str());
    output.push(
        &text,
        MuseGlimmerPromptSpanKind::ReasoningInstructionContent,
        message_index,
        None,
        Some(MuseGlimmerPromptSpanRole::System),
        Some(MuseGlimmerPromptChannel::Thinking),
    );
}

fn push_muse_tool_definitions(
    output: &mut AnnotatedMuseGlimmerPromptBuilder,
    message_index: Option<usize>,
    options: &MuseGlimmerPromptOptions,
) -> Result<(), MuseGlimmerPromptError> {
    let mut text = String::new();
    render_tool_definitions(&mut text, options)?;
    output.push(
        &text,
        MuseGlimmerPromptSpanKind::ToolDefinitionContent,
        message_index,
        None,
        Some(MuseGlimmerPromptSpanRole::System),
        None,
    );
    Ok(())
}

fn push_muse_system_meta(
    output: &mut AnnotatedMuseGlimmerPromptBuilder,
    message_index: Option<usize>,
    options: &MuseGlimmerPromptOptions,
) {
    let mut text = String::new();
    render_system_meta(&mut text, options);
    output.push(
        &text,
        MuseGlimmerPromptSpanKind::SystemMetadataContent,
        message_index,
        None,
        Some(MuseGlimmerPromptSpanRole::System),
        None,
    );
}

fn validate_options(options: &MuseGlimmerPromptOptions) -> Result<(), MuseGlimmerPromptError> {
    for (field, value) in [
        ("knowledge_cutoff", options.knowledge_cutoff.as_str()),
        ("current_date", options.current_date.as_str()),
    ] {
        if value.is_empty() {
            return Err(MuseGlimmerPromptError::InvalidOptions {
                field,
                detail: "must not be empty".into(),
            });
        }
    }
    let mut names = BTreeSet::new();
    for (index, tool) in options.tools.iter().enumerate() {
        if !is_atem_path(&tool.name)
            || (!tool.name.contains('.') && matches!(tool.name.as_str(), "self" | "user"))
        {
            return Err(MuseGlimmerPromptError::InvalidTool {
                index,
                field: "name",
                detail: "must be a non-reserved, dot-separated ATEM identifier".into(),
            });
        }
        if !names.insert(tool.name.as_str()) {
            return Err(MuseGlimmerPromptError::InvalidTool {
                index,
                field: "name",
                detail: "must be unique".into(),
            });
        }
        if !tool.parameters.is_object() {
            return Err(MuseGlimmerPromptError::InvalidTool {
                index,
                field: "parameters",
                detail: "must be a JSON object".into(),
            });
        }
    }
    let namespaces = tool_namespaces(&options.tools);
    for namespace in options.tool_namespace_descriptions.keys() {
        if !namespaces.contains(&namespace.as_str()) {
            return Err(MuseGlimmerPromptError::InvalidOptions {
                field: "tool_namespace_descriptions",
                detail: format!("contains unknown namespace {namespace:?}"),
            });
        }
    }
    Ok(())
}

fn validate_messages(
    messages: &[MuseGlimmerMessage],
    options: &MuseGlimmerPromptOptions,
) -> Result<(), MuseGlimmerPromptError> {
    let declared_tools = options
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect::<BTreeSet<_>>();
    let mut saw_system = false;
    let mut saw_conversation = false;
    let mut pending_tool_results = Vec::<&str>::new();
    for (index, message) in messages.iter().enumerate() {
        if matches!(message.role, MuseGlimmerMessageRole::System) {
            if saw_system || index != 0 {
                return Err(MuseGlimmerPromptError::InvalidMessage {
                    index,
                    field: "role",
                    detail: "system must appear at most once and only as the first message".into(),
                });
            }
            saw_system = true;
        }
        if !matches!(message.role, MuseGlimmerMessageRole::Assistant)
            && (message.reasoning_content.is_some()
                || message.recipient.is_some()
                || message.end_turn.is_some()
                || !message.tool_calls.is_empty())
        {
            return Err(MuseGlimmerPromptError::InvalidMessage {
                index,
                field: "assistant fields",
                detail: "are only valid on assistant messages".into(),
            });
        }
        if let MuseGlimmerMessageRole::Tool { name } = &message.role
            && !is_atem_path(name)
        {
            return Err(MuseGlimmerPromptError::InvalidMessage {
                index,
                field: "tool name",
                detail: "must be a safe ATEM path".into(),
            });
        }
        if matches!(message.role, MuseGlimmerMessageRole::Assistant)
            && !message.tool_calls.is_empty()
            && (!message.content.is_empty()
                || message.recipient.is_some()
                || message.end_turn.is_some())
        {
            return Err(MuseGlimmerPromptError::InvalidMessage {
                index,
                field: "tool_calls",
                detail: "cannot be combined with content, recipient, or end_turn".into(),
            });
        }
        if matches!(message.role, MuseGlimmerMessageRole::Assistant)
            && message.tool_calls.is_empty()
            && !matches!(
                message.recipient.as_deref(),
                None | Some("self") | Some("user")
            )
        {
            return Err(MuseGlimmerPromptError::InvalidMessage {
                index,
                field: "recipient",
                detail: "plain assistant history supports only self or user".into(),
            });
        }
        for call in &message.tool_calls {
            if !is_atem_path(&call.name)
                || !declared_tools.contains(call.name.as_str())
                || !call.arguments.is_object()
                || call
                    .arguments
                    .as_object()
                    .is_some_and(|arguments| arguments.keys().any(|name| !is_atem_segment(name)))
            {
                return Err(MuseGlimmerPromptError::InvalidMessage {
                    index,
                    field: "tool_calls",
                    detail: "each call requires a declared safe name and safe object arguments"
                        .into(),
                });
            }
        }

        let previous = index.checked_sub(1).and_then(|index| messages.get(index));
        match &message.role {
            MuseGlimmerMessageRole::System => {}
            MuseGlimmerMessageRole::User => {
                if !pending_tool_results.is_empty()
                    || (saw_conversation
                        && !matches!(
                            previous,
                            Some(MuseGlimmerMessage {
                                role: MuseGlimmerMessageRole::Assistant,
                                ..
                            }) | Some(MuseGlimmerMessage {
                                role: MuseGlimmerMessageRole::Tool { .. },
                                ..
                            })
                        ))
                {
                    return Err(invalid_history(index, "user message is out of turn"));
                }
                if let Some(previous) = previous
                    && matches!(previous.role, MuseGlimmerMessageRole::Assistant)
                    && !assistant_ends_turn(previous)
                {
                    return Err(invalid_history(
                        index,
                        "user message follows an unfinished assistant segment",
                    ));
                }
                saw_conversation = true;
            }
            MuseGlimmerMessageRole::Assistant => {
                let extends_segmented_tool_calls = !message.tool_calls.is_empty()
                    && previous.is_some_and(|previous| {
                        matches!(previous.role, MuseGlimmerMessageRole::Assistant)
                            && !previous.tool_calls.is_empty()
                    });
                let follows_valid_turn = matches!(
                    previous,
                    Some(MuseGlimmerMessage {
                        role: MuseGlimmerMessageRole::User,
                        ..
                    }) | Some(MuseGlimmerMessage {
                        role: MuseGlimmerMessageRole::Tool { .. },
                        ..
                    })
                ) || previous.is_some_and(|previous| {
                    matches!(previous.role, MuseGlimmerMessageRole::Assistant)
                        && previous.tool_calls.is_empty()
                        && !assistant_ends_turn(previous)
                }) || extends_segmented_tool_calls;
                if !saw_conversation
                    || (!pending_tool_results.is_empty() && !extends_segmented_tool_calls)
                    || !follows_valid_turn
                {
                    return Err(invalid_history(index, "assistant message is out of turn"));
                }
                pending_tool_results
                    .extend(message.tool_calls.iter().map(|call| call.name.as_str()));
            }
            MuseGlimmerMessageRole::Tool { name } => {
                if !declared_tools.contains(name.as_str()) {
                    return Err(invalid_history(
                        index,
                        "tool result names an undeclared function",
                    ));
                }
                let Some(pending_index) = pending_tool_results
                    .iter()
                    .position(|pending| *pending == name)
                else {
                    return Err(invalid_history(
                        index,
                        "tool result has no matching pending call",
                    ));
                };
                pending_tool_results.remove(pending_index);
            }
        }
    }
    if options.add_generation_prompt && !pending_tool_results.is_empty() {
        return Err(invalid_history(
            messages.len().saturating_sub(1),
            "assistant tool calls are missing results",
        ));
    }
    if options.add_generation_prompt {
        let ready = messages.last().is_some_and(|message| match message.role {
            MuseGlimmerMessageRole::User | MuseGlimmerMessageRole::Tool { .. } => true,
            MuseGlimmerMessageRole::Assistant => {
                message.tool_calls.is_empty() && !assistant_ends_turn(message)
            }
            MuseGlimmerMessageRole::System => false,
        });
        if !ready {
            return Err(invalid_history(
                messages.len().saturating_sub(1),
                "history is not awaiting an assistant continuation",
            ));
        }
    }
    Ok(())
}

fn invalid_history(index: usize, detail: impl Into<String>) -> MuseGlimmerPromptError {
    MuseGlimmerPromptError::InvalidMessage {
        index,
        field: "history",
        detail: detail.into(),
    }
}

fn assistant_ends_turn(message: &MuseGlimmerMessage) -> bool {
    let recipient = message.recipient.as_deref().unwrap_or("user");
    message.end_turn.unwrap_or(recipient == "user")
}

fn is_atem_path(value: &str) -> bool {
    !value.is_empty() && value.split('.').all(is_atem_segment)
}

fn is_atem_segment(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn render_tool_definitions(
    output: &mut String,
    options: &MuseGlimmerPromptOptions,
) -> Result<(), MuseGlimmerPromptError> {
    output.push_str("In this environment you have access to a set of tools you can use to answer the user's question.\n\n");
    output.push_str("You can invoke a function by writing a \"<atem:function_calls>\" block like the following:\n");
    output.push_str("<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\n");
    output.push_str("String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\n");
    output.push_str("Here are the functions available in JSONSchema format:\n");
    output.push_str("// Tool metadata\n");
    for namespace in tool_namespaces(&options.tools) {
        let description = options
            .tool_namespace_descriptions
            .get(namespace)
            .map(String::as_str)
            .unwrap_or("");
        output.push_str("{\"name\": ");
        output.push_str(&serde_json::to_string(namespace)?);
        output.push_str(", \"description\": ");
        output.push_str(&serde_json::to_string(description)?);
        output.push_str("}\n");
    }
    output.push_str("// Function schemas");
    for tool in &options.tools {
        output.push_str("\n{\"name\": ");
        output.push_str(&serde_json::to_string(&tool.name)?);
        output.push_str(", \"description\": ");
        output.push_str(&serde_json::to_string(&tool.description)?);
        output.push_str(", \"parameters\": ");
        output.push_str(&serde_json::to_string(&tool.parameters)?);
        output.push('}');
    }
    output.push_str("\n\nHere's an example of how to call a function in the tool set:\n");
    output.push_str("(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than `example_tool_name.example_function_name`)\n\n");
    output.push_str("to=example_tool_name.example_function_name\n\n");
    output.push_str(
        "<atem:function_calls>\n<atem:invoke name=\"example_tool_name.example_function_name\">\n",
    );
    output.push_str("<atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n");
    output.push_str("<atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\nthat can span\n\"multiple\" lines\n</atem:parameter>\n");
    output.push_str("</atem:invoke>\n</atem:function_calls>");
    Ok(())
}

fn render_system_meta(output: &mut String, options: &MuseGlimmerPromptOptions) {
    output.push_str("# Valid recipients: \"self\"");
    for recipient in tool_recipient_patterns(&options.tools) {
        output.push_str(", \"");
        output.push_str(&recipient);
        output.push('"');
    }
    output.push_str(", \"user\".");
}

fn render_atem_call(
    output: &mut String,
    call: &MuseGlimmerToolCall,
) -> Result<(), MuseGlimmerPromptError> {
    let arguments = call
        .arguments
        .as_object()
        .expect("validated ATEM arguments are objects");
    output.push_str("<atem:function_calls>\n<atem:invoke name=\"");
    output.push_str(&call.name);
    output.push_str("\">\n");
    for (name, value) in arguments {
        output.push_str("<atem:parameter name=\"");
        output.push_str(name);
        output.push_str("\">");
        match value {
            Value::String(value) => output.push_str(value),
            Value::Bool(true) => output.push_str("true"),
            Value::Bool(false) => output.push_str("false"),
            Value::Null => output.push_str("null"),
            Value::Number(value) => output.push_str(&value.to_string()),
            Value::Array(_) | Value::Object(_) => {
                output.push_str(&serde_json::to_string(value)?);
            }
        }
        output.push_str("</atem:parameter>\n");
    }
    output.push_str("</atem:invoke>\n</atem:function_calls>");
    Ok(())
}

fn tool_namespaces(tools: &[MuseGlimmerToolDefinition]) -> Vec<&str> {
    let mut namespaces = Vec::new();
    for tool in tools {
        let namespace = tool.name.split('.').next().unwrap_or(&tool.name);
        if !namespaces.contains(&namespace) {
            namespaces.push(namespace);
        }
    }
    namespaces
}

fn tool_recipient_patterns(tools: &[MuseGlimmerToolDefinition]) -> Vec<String> {
    let mut recipients = Vec::new();
    for tool in tools {
        let recipient = match tool.name.split_once('.') {
            Some((namespace, _)) => format!("{namespace}.*"),
            None => tool.name.clone(),
        };
        if !recipients.contains(&recipient) {
            recipients.push(recipient);
        }
    }
    recipients
}

struct NormalizedSystemContent {
    text: String,
    directive_range: Option<std::ops::Range<usize>>,
}

fn normalize_reasoning_directive(
    content: &str,
    strength: MuseGlimmerReasoningStrength,
    message_index: usize,
) -> Result<NormalizedSystemContent, MuseGlimmerPromptError> {
    let mut text = String::new();
    let mut directive_range = None;
    for (line_index, line) in content.split('\n').enumerate() {
        if line_index > 0 {
            text.push('\n');
        }
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        let value = ["reasoning strength:", "reasoning effort:"]
            .iter()
            .find_map(|prefix| lower.strip_prefix(prefix).map(str::trim));
        let Some(value) = value else {
            text.push_str(line);
            continue;
        };
        let value = value.strip_suffix('.').unwrap_or(value).trim();
        if !matches!(value, "low" | "medium" | "high" | "xhigh") {
            return Err(MuseGlimmerPromptError::InvalidMessage {
                index: message_index,
                field: "content",
                detail: format!("reasoning strength has unsupported value {value:?}"),
            });
        }
        if directive_range.is_some() {
            return Err(MuseGlimmerPromptError::InvalidMessage {
                index: message_index,
                field: "content",
                detail: "contains multiple reasoning-strength directives".into(),
            });
        }
        let start = text.len();
        text.push_str("Reasoning strength: ");
        text.push_str(strength.as_str());
        text.push('.');
        directive_range = Some(start..text.len());
    }
    Ok(NormalizedSystemContent {
        text,
        directive_range,
    })
}

fn push_muse_system_message_content(
    output: &mut AnnotatedMuseGlimmerPromptBuilder,
    message_index: Option<usize>,
    normalized: &NormalizedSystemContent,
) {
    let mut push_content = |text: &str, kind, channel| {
        output.push(
            text,
            kind,
            message_index,
            None,
            Some(MuseGlimmerPromptSpanRole::System),
            channel,
        );
    };
    let Some(range) = normalized.directive_range.as_ref() else {
        push_content(
            &normalized.text,
            MuseGlimmerPromptSpanKind::MessageContent,
            None,
        );
        return;
    };
    push_content(
        &normalized.text[..range.start],
        MuseGlimmerPromptSpanKind::MessageContent,
        None,
    );
    push_content(
        &normalized.text[range.clone()],
        MuseGlimmerPromptSpanKind::ReasoningInstructionContent,
        Some(MuseGlimmerPromptChannel::Thinking),
    );
    push_content(
        &normalized.text[range.end..],
        MuseGlimmerPromptSpanKind::MessageContent,
        None,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assert_valid_spans(prompt: &AnnotatedMuseGlimmerPrompt) {
        let mut previous_end = 0;
        for span in &prompt.spans {
            assert!(span.byte_start < span.byte_end);
            assert!(span.byte_start >= previous_end);
            assert!(prompt.text.is_char_boundary(span.byte_start));
            assert!(prompt.text.is_char_boundary(span.byte_end));
            previous_end = span.byte_end;
        }
    }

    #[test]
    fn renders_release_default_single_turn_bytes() {
        let messages = [MuseGlimmerMessage::user("Why is the sky blue?")];
        let annotated = render_muse_glimmer_atem_prompt_annotated(
            &messages,
            &MuseGlimmerPromptOptions::default(),
        )
        .unwrap();
        assert_valid_spans(&annotated);
        let prompt = annotated.text;
        assert_eq!(
            prompt,
            concat!(
                "<|begin_of_text|><|start|>system<|message|>",
                "You are a helpful AI assistant.\n",
                "Knowledge cutoff: 2026-01-04.\n",
                "Current date: 2026-08-29.\n\n",
                "Reasoning strength: high.\n\n",
                "# Valid recipients: \"self\", \"user\".<|eot|>",
                "<|start|>user<|message|>Why is the sky blue?<|eot|>",
                "<|start|>assistant",
            )
        );
        assert_eq!(
            render_muse_glimmer_single_turn(
                "Why is the sky blue?",
                None,
                &MuseGlimmerPromptOptions::default(),
            )
            .unwrap(),
            prompt
        );
        let bos = annotated
            .spans
            .iter()
            .find(|span| span.kind == MuseGlimmerPromptSpanKind::BosMarker)
            .unwrap();
        assert_eq!(&prompt[bos.byte_start..bos.byte_end], MUSE_GLIMMER_BOS);
        assert!(bos.message_index.is_none() && bos.role.is_none());
        let synthetic_system = annotated.spans.iter().find(|span| {
            span.kind == MuseGlimmerPromptSpanKind::MessageStartMarker
                && span.role == Some(MuseGlimmerPromptSpanRole::System)
        });
        assert!(synthetic_system.unwrap().message_index.is_none());
        let user_content = annotated.spans.iter().find(|span| {
            span.kind == MuseGlimmerPromptSpanKind::MessageContent && span.message_index == Some(0)
        });
        assert_eq!(
            user_content.map(|span| &prompt[span.byte_start..span.byte_end]),
            Some("Why is the sky blue?")
        );
    }

    #[test]
    fn selected_reasoning_strength_normalizes_and_annotates_existing_directive_once() {
        let annotated = render_muse_glimmer_atem_prompt_annotated(
            &[
                MuseGlimmerMessage::system("Be concise.\n\nReasoning Effort: low."),
                MuseGlimmerMessage::user("hello"),
            ],
            &MuseGlimmerPromptOptions {
                reasoning_strength: MuseGlimmerReasoningStrength::Xhigh,
                ..MuseGlimmerPromptOptions::default()
            },
        )
        .unwrap();
        assert_valid_spans(&annotated);
        let prompt = annotated.text;
        assert!(prompt.contains("Reasoning strength: xhigh."));
        assert!(!prompt.contains("Reasoning strength: high."));
        assert!(!prompt.contains("Reasoning Effort"));
        let directive = annotated
            .spans
            .iter()
            .find(|span| span.kind == MuseGlimmerPromptSpanKind::ReasoningInstructionContent)
            .unwrap();
        assert_eq!(directive.message_index, Some(0));
        assert_eq!(directive.channel, Some(MuseGlimmerPromptChannel::Thinking));
        assert_eq!(
            &prompt[directive.byte_start..directive.byte_end],
            "Reasoning strength: xhigh."
        );
    }

    #[test]
    fn reasoning_prose_does_not_suppress_the_typed_directive() {
        let prompt = render_muse_glimmer_single_turn(
            "hello",
            Some("Explain the reasoning strength of the evidence."),
            &MuseGlimmerPromptOptions::default(),
        )
        .unwrap();
        assert!(prompt.contains("Explain the reasoning strength of the evidence."));
        assert!(prompt.contains("Reasoning strength: high."));
    }

    #[test]
    fn rejects_invalid_duplicate_or_misplaced_reasoning_directives() {
        for system in [
            "Reasoning strength: extreme.",
            "Reasoning strength: low.\nReasoning effort: high.",
        ] {
            assert!(
                render_muse_glimmer_single_turn(
                    "hello",
                    Some(system),
                    &MuseGlimmerPromptOptions::default(),
                )
                .is_err()
            );
        }
        assert!(
            render_muse_glimmer_atem_prompt(
                &[
                    MuseGlimmerMessage::system("Reasoning strength: low."),
                    MuseGlimmerMessage::system("Reasoning strength: high."),
                    MuseGlimmerMessage::user("hello"),
                ],
                &MuseGlimmerPromptOptions::default(),
            )
            .is_err()
        );
    }

    #[test]
    fn renders_every_released_reasoning_strength_exactly() {
        for (strength, expected) in [
            (MuseGlimmerReasoningStrength::Low, "low"),
            (MuseGlimmerReasoningStrength::Medium, "medium"),
            (MuseGlimmerReasoningStrength::High, "high"),
            (MuseGlimmerReasoningStrength::Xhigh, "xhigh"),
        ] {
            let prompt = render_muse_glimmer_single_turn(
                "hello",
                None,
                &MuseGlimmerPromptOptions {
                    reasoning_strength: strength,
                    ..MuseGlimmerPromptOptions::default()
                },
            )
            .unwrap();
            assert!(prompt.contains(&format!("Reasoning strength: {expected}.")));
        }
    }

    #[test]
    fn renders_atem_tool_schema_call_and_output() {
        let mut options = MuseGlimmerPromptOptions::default();
        options.tools.push(MuseGlimmerToolDefinition {
            name: "weather.lookup".into(),
            description: "Look up weather".into(),
            parameters: json!({"type":"object","properties":{"city":{"type":"string"}}}),
        });
        options
            .tool_namespace_descriptions
            .insert("weather".into(), "Weather tools".into());
        let mut assistant = MuseGlimmerMessage::assistant("");
        assistant.reasoning_content = Some("I should check.".into());
        assistant.tool_calls.push(MuseGlimmerToolCall {
            name: "weather.lookup".into(),
            arguments: json!({"city":"New York","days":2,"metric":true}),
        });
        let messages = [
            MuseGlimmerMessage::user("Forecast?"),
            assistant,
            MuseGlimmerMessage::tool("weather.lookup", "Sunny"),
            MuseGlimmerMessage::user("Thanks"),
        ];
        let annotated = render_muse_glimmer_atem_prompt_annotated(&messages, &options).unwrap();
        assert_valid_spans(&annotated);
        let prompt = &annotated.text;
        assert!(prompt.contains("# Valid recipients: \"self\", \"weather.*\", \"user\"."));
        assert!(prompt.contains("<|start|>assistant to=self<|message|>I should check.<|eom|>"));
        assert!(prompt.contains("<atem:invoke name=\"weather.lookup\">"));
        assert!(prompt.contains("<atem:parameter name=\"city\">New York</atem:parameter>"));
        assert!(prompt.contains("<|start|>tool weather.lookup<|message|><tool_output name=\"weather.lookup\">\nSunny\n</tool_output><|eot|>"));
        let reasoning = annotated
            .spans
            .iter()
            .find(|span| span.kind == MuseGlimmerPromptSpanKind::AssistantReasoningContent)
            .unwrap();
        assert_eq!(reasoning.message_index, Some(1));
        assert_eq!(reasoning.channel, Some(MuseGlimmerPromptChannel::Thinking));
        assert_eq!(
            &prompt[reasoning.byte_start..reasoning.byte_end],
            "I should check."
        );
        let call = annotated
            .spans
            .iter()
            .find(|span| span.kind == MuseGlimmerPromptSpanKind::ToolCallContent)
            .unwrap();
        assert_eq!(
            (call.message_index, call.tool_call_index),
            (Some(1), Some(0))
        );
        assert_eq!(call.channel, Some(MuseGlimmerPromptChannel::ToolCall));
        let result = annotated
            .spans
            .iter()
            .find(|span| span.kind == MuseGlimmerPromptSpanKind::ToolResultContent)
            .unwrap();
        assert_eq!(result.message_index, Some(2));
        assert_eq!(result.channel, Some(MuseGlimmerPromptChannel::ToolResult));
        assert_eq!(&prompt[result.byte_start..result.byte_end], "Sunny");
        assert_eq!(
            render_muse_glimmer_atem_prompt(&messages, &options).unwrap(),
            *prompt
        );
    }

    #[test]
    fn renders_unqualified_tool_as_an_exact_recipient() {
        let options = MuseGlimmerPromptOptions {
            tools: vec![MuseGlimmerToolDefinition {
                name: "lookup".into(),
                description: "Lookup".into(),
                parameters: json!({"type":"object"}),
            }],
            ..MuseGlimmerPromptOptions::default()
        };
        let prompt =
            render_muse_glimmer_atem_prompt(&[MuseGlimmerMessage::user("Lookup this")], &options)
                .unwrap();
        assert!(prompt.contains("# Valid recipients: \"self\", \"lookup\", \"user\"."));
        assert!(!prompt.contains("\"lookup.*\""));
    }

    #[test]
    fn preserves_segmented_multi_tool_calls_before_results() {
        let options = MuseGlimmerPromptOptions {
            tools: ["weather.lookup", "calendar.lookup"]
                .into_iter()
                .map(|name| MuseGlimmerToolDefinition {
                    name: name.into(),
                    description: String::new(),
                    parameters: json!({"type":"object"}),
                })
                .collect(),
            ..MuseGlimmerPromptOptions::default()
        };
        let mut weather = MuseGlimmerMessage::assistant("");
        weather.reasoning_content = Some("Check both sources.".into());
        weather.tool_calls.push(MuseGlimmerToolCall {
            name: "weather.lookup".into(),
            arguments: json!({"city":"Paris"}),
        });
        let mut calendar = MuseGlimmerMessage::assistant("");
        calendar.tool_calls.push(MuseGlimmerToolCall {
            name: "calendar.lookup".into(),
            arguments: json!({"date":"2026-08-31"}),
        });
        let prompt = render_muse_glimmer_atem_prompt(
            &[
                MuseGlimmerMessage::user("Plan tomorrow"),
                weather,
                calendar,
                MuseGlimmerMessage::tool("weather.lookup", "Sunny"),
                MuseGlimmerMessage::tool("calendar.lookup", "Open"),
            ],
            &options,
        )
        .unwrap();
        assert!(prompt.contains("<atem:invoke name=\"weather.lookup\">"));
        assert!(prompt.contains("<atem:invoke name=\"calendar.lookup\">"));
        assert!(prompt.ends_with("<|start|>assistant"));
    }

    #[test]
    fn rejects_malformed_atem_names_and_incomplete_histories() {
        let mut options = MuseGlimmerPromptOptions {
            tools: vec![MuseGlimmerToolDefinition {
                name: "weather.lookup".into(),
                description: "Weather".into(),
                parameters: json!({"type":"object"}),
            }],
            ..MuseGlimmerPromptOptions::default()
        };
        for messages in [
            vec![
                MuseGlimmerMessage::user("one"),
                MuseGlimmerMessage::user("two"),
            ],
            vec![
                MuseGlimmerMessage::user("one"),
                MuseGlimmerMessage::assistant("done"),
            ],
            vec![MuseGlimmerMessage::tool("weather.lookup", "orphan")],
        ] {
            assert!(render_muse_glimmer_atem_prompt(&messages, &options).is_err());
        }

        let mut call = MuseGlimmerMessage::assistant("");
        call.tool_calls.push(MuseGlimmerToolCall {
            name: "weather.lookup".into(),
            arguments: json!({"city":"Paris"}),
        });
        assert!(
            render_muse_glimmer_atem_prompt(
                &[MuseGlimmerMessage::user("Weather?"), call.clone()],
                &options,
            )
            .is_err()
        );
        call.recipient = Some("weather.lookup".into());
        assert!(
            render_muse_glimmer_atem_prompt(
                &[
                    MuseGlimmerMessage::user("Weather?"),
                    call,
                    MuseGlimmerMessage::tool("weather.lookup", "Sunny"),
                ],
                &options,
            )
            .is_err()
        );

        options.tools[0].name = "weather\" bad".into();
        assert!(
            render_muse_glimmer_atem_prompt(&[MuseGlimmerMessage::user("Weather?")], &options,)
                .is_err()
        );
        for reserved in ["self", "user"] {
            options.tools[0].name = reserved.into();
            assert!(
                render_muse_glimmer_atem_prompt(&[MuseGlimmerMessage::user("Weather?")], &options,)
                    .is_err()
            );
        }
    }

    #[test]
    fn annotated_recipients_and_end_markers_preserve_multi_record_atem_boundaries() {
        let mut assistant = MuseGlimmerMessage::assistant("");
        assistant.reasoning_content = Some("reason".into());
        assistant.tool_calls = vec![
            MuseGlimmerToolCall {
                name: "first".into(),
                arguments: json!({"x": 1}),
            },
            MuseGlimmerToolCall {
                name: "second".into(),
                arguments: json!({"y": [1, 2]}),
            },
        ];
        let messages = [MuseGlimmerMessage::user("go"), assistant];
        let options = MuseGlimmerPromptOptions {
            tools: ["first", "second"]
                .into_iter()
                .map(|name| MuseGlimmerToolDefinition {
                    name: name.into(),
                    description: String::new(),
                    parameters: json!({"type": "object"}),
                })
                .collect(),
            add_generation_prompt: false,
            ..MuseGlimmerPromptOptions::default()
        };
        let annotated = render_muse_glimmer_atem_prompt_annotated(&messages, &options).unwrap();
        assert_valid_spans(&annotated);
        assert!(
            annotated
                .text
                .contains("# Valid recipients: \"self\", \"first\", \"second\", \"user\".")
        );
        assert!(annotated.text.ends_with(concat!(
            "<|start|>user<|message|>go<|eot|>",
            "<|start|>assistant to=self<|message|>reason<|eom|>",
            "<|start|>assistant to=first<|message|>",
            "<atem:function_calls>\n",
            "<atem:invoke name=\"first\">\n",
            "<atem:parameter name=\"x\">1</atem:parameter>\n",
            "</atem:invoke>\n",
            "</atem:function_calls><|eom|>",
            "<|start|>assistant to=second<|message|>",
            "<atem:function_calls>\n",
            "<atem:invoke name=\"second\">\n",
            "<atem:parameter name=\"y\">[1,2]</atem:parameter>\n",
            "</atem:invoke>\n",
            "</atem:function_calls><|eot|>",
        )));
        let recipients = annotated
            .spans
            .iter()
            .filter(|span| span.kind == MuseGlimmerPromptSpanKind::Recipient)
            .map(|span| &annotated.text[span.byte_start..span.byte_end])
            .collect::<Vec<_>>();
        assert_eq!(recipients, ["self", "first", "second"]);
        let assistant_ends = annotated
            .spans
            .iter()
            .filter(|span| {
                span.kind == MuseGlimmerPromptSpanKind::MessageEndMarker
                    && span.message_index == Some(1)
            })
            .map(|span| &annotated.text[span.byte_start..span.byte_end])
            .collect::<Vec<_>>();
        assert_eq!(
            assistant_ends,
            [MUSE_GLIMMER_EOM, MUSE_GLIMMER_EOM, MUSE_GLIMMER_EOT]
        );
        assert!(
            annotated
                .spans
                .iter()
                .all(|span| span.kind != MuseGlimmerPromptSpanKind::GeneratedAssistantStartMarker)
        );
    }

    #[test]
    fn rejects_non_object_atem_arguments() {
        let mut assistant = MuseGlimmerMessage::assistant("");
        assistant.tool_calls.push(MuseGlimmerToolCall {
            name: "weather.lookup".into(),
            arguments: json!("not an object"),
        });
        assert!(
            render_muse_glimmer_atem_prompt(
                &[MuseGlimmerMessage::user("Forecast?"), assistant],
                &MuseGlimmerPromptOptions {
                    tools: vec![MuseGlimmerToolDefinition {
                        name: "weather.lookup".into(),
                        description: String::new(),
                        parameters: json!({"type": "object"}),
                    }],
                    ..MuseGlimmerPromptOptions::default()
                },
            )
            .is_err()
        );
    }
}
