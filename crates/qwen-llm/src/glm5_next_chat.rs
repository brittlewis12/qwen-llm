//! GLM-5.3-Flash chat: a hand-written renderer for the upstream template's
//! text and function-tool subset, pinned byte-for-byte against the template
//! under transformers' Jinja settings
//! (`scripts/reference/generate_glm53_chat_fixtures.py`).
//!
//! The template always opens `[gMASK]<sop><|system|>Reasoning Effort: …`
//! and generates from `<|assistant|><think>`: there is no non-thinking
//! mode. Assistant history keeps its reasoning unless `clear_thinking`
//! drops it at or before the last user turn; its visible content is
//! stripped with Python's `str.strip()`. Tool definitions, calls and
//! results follow [`tools`]; shapes the template renders but this renderer
//! cannot honor (strict or deferred tools, string arguments, unmatched
//! results) are refused, not approximated.

use crate::gguf::GgufFile;
use crate::tokenizer::NativeTokenizer;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// `zai-org/GLM-5.3-Flash` revision of the pinned template and generation config.
pub const REVISION: &str = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a";
/// Upstream `chat_template.jinja`.
pub const TEMPLATE_SHA256: &str =
    "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5";
/// `tokenizer.chat_template` of the unsloth UD GGUF conversion. Renders the
/// text-only subset identically to upstream except null assistant content,
/// which it prints as `None`; this renderer follows upstream.
pub const GGUF_TEMPLATE_SHA256: &str =
    "a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724";
pub const GENERATION_CONFIG_SHA256: &str =
    "230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f";
pub const RENDERER: &str = "glm53_flash_chat_v2";
/// `generation_config.json` `eos_token_id`: `<|endoftext|>`, `<|user|>`,
/// `<|observation|>`. A turn ends by opening the next one.
pub const CHAT_STOPS: [i32; 3] = [154_820, 154_827, 154_829];
/// The released sampling defaults (`generation_config.json`).
pub const TEMPERATURE: f32 = 1.0;
pub const TOP_P: f32 = 0.95;
pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";

const PREFIX: &str = "[gMASK]<sop>";
const GENERATION_PROMPT: &str = "<|assistant|><think>";
/// Every marker the renderer writes or the output grammar reads, with the
/// single token it must encode to.
const MARKERS: [(&str, i32); 9] = [
    ("<|endoftext|>", 154_820),
    ("[gMASK]", 154_822),
    ("<sop>", 154_824),
    ("<|system|>", 154_826),
    ("<|user|>", 154_827),
    ("<|assistant|>", 154_828),
    ("<|observation|>", 154_829),
    (THINK_OPEN, 154_841),
    (THINK_CLOSE, 154_842),
];

#[derive(Debug, thiserror::Error)]
#[error("GLM-5.3 chat {code}: {message}")]
pub struct ChatError {
    code: &'static str,
    message: String,
}

impl ChatError {
    /// Stable refusal code: `glm5_next_chat_effort`, `glm5_next_chat_input`,
    /// `glm5_next_chat_tools` or `glm5_next_chat_profile_unverified`.
    pub fn code(&self) -> &'static str {
        self.code
    }

    pub(crate) fn unverified(message: impl Into<String>) -> Self {
        error("glm5_next_chat_profile_unverified", message)
    }
}

type Result<T> = std::result::Result<T, ChatError>;

fn error(code: &'static str, message: impl Into<String>) -> ChatError {
    ChatError {
        code,
        message: message.into(),
    }
}

fn input(message: impl Into<String>) -> ChatError {
    error("glm5_next_chat_input", message)
}

mod tools;
use tools::tools;
mod tool_output;
pub use tool_output::{
    OutputCall, ToolOutputEnd, ToolOutputFinish, ToolOutputStream, parse_tool_calls,
};
pub use tools::{
    ARG_KEY_CLOSE, ARG_KEY_OPEN, ARG_VALUE_CLOSE, ARG_VALUE_OPEN, OBSERVATION, TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN, TOOL_RESPONSE_CLOSE, TOOL_RESPONSE_OPEN, ToolCall, ToolDefinition, tojson,
    valid_argument_key, valid_tool_name,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    Low,
    High,
    /// The template's own fallback.
    #[default]
    Max,
}

impl Effort {
    pub const LEVELS: [&'static str; 3] = ["low", "high", "max"];

    /// `None` is the template's default, `max`. The template maps every
    /// other value to `Max` as well; this refuses them instead.
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value {
            None | Some("max") => Ok(Self::Max),
            Some("low") => Ok(Self::Low),
            Some("high") => Ok(Self::High),
            Some(other) => Err(error(
                "glm5_next_chat_effort",
                format!(
                    "reasoning effort must be low, high or max (default max); got {other:?}; there is no non-thinking mode"
                ),
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::High => "high",
            Self::Max => "max",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::High => "High",
            Self::Max => "Max",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Message {
    System(String),
    User(String),
    /// `content` is the raw message content: without `reasoning`, the
    /// renderer takes reasoning from an inline `</think>` the way the
    /// template does. `Some("")` is reasoning that was empty. `calls`
    /// follow the content.
    Assistant {
        content: String,
        reasoning: Option<String>,
        calls: Vec<ToolCall>,
    },
    /// The result of the preceding assistant turn's call `call_id`.
    Tool {
        call_id: String,
        content: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderOptions {
    pub effort: Effort,
    /// Drop assistant reasoning at or before the last user turn.
    pub clear_thinking: bool,
    /// End with `<|assistant|><think>`; generation requires a final user turn.
    pub add_generation_prompt: bool,
}

impl RenderOptions {
    pub fn generate(effort: Effort, clear_thinking: bool) -> Self {
        Self {
            effort,
            clear_thinking,
            add_generation_prompt: true,
        }
    }
}

/// A parsed `--messages` document: a bare message array, or
/// `{"messages": [...], "clear_thinking": bool, "tools": [...]}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatDocument {
    pub messages: Vec<Message>,
    pub clear_thinking: Option<bool>,
    pub tools: Vec<ToolDefinition>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMessage {
    role: String,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDocument {
    messages: Vec<WireMessage>,
    #[serde(default)]
    clear_thinking: Option<bool>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Wire {
    Array(Vec<WireMessage>),
    Object(WireDocument),
}

impl WireMessage {
    fn into_message(self, index: usize) -> Result<Message> {
        if self.tool_calls.is_some() && self.role != "assistant" {
            return Err(tools(format!(
                "message {index}: tool_calls belong only to assistant turns"
            )));
        }
        if self.tool_call_id.is_some() && self.role != "tool" {
            return Err(input(format!(
                "message {index}: tool_call_id belongs only to tool messages"
            )));
        }
        let content = match self.content {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(text)) => Some(text),
            Some(_) => {
                return Err(input(format!(
                    "message {index}: content must be a string; content parts are not supported"
                )));
            }
        };
        let text = |content: Option<String>| {
            content.ok_or_else(|| input(format!("message {index}: content must be a string")))
        };
        let no_reasoning = |reasoning: &Option<String>| match reasoning {
            Some(_) => Err(input(format!(
                "message {index}: reasoning_content belongs only to assistant turns"
            ))),
            None => Ok(()),
        };
        match self.role.as_str() {
            "system" => {
                no_reasoning(&self.reasoning_content)?;
                Ok(Message::System(text(content)?))
            }
            "user" => {
                no_reasoning(&self.reasoning_content)?;
                Ok(Message::User(text(content)?))
            }
            // The template renders null content as nothing.
            "assistant" => {
                let calls = match self.tool_calls {
                    None | Some(serde_json::Value::Null) => Vec::new(),
                    Some(serde_json::Value::Array(calls)) => calls
                        .iter()
                        .map(|call| ToolCall::from_value(call, index))
                        .collect::<Result<_>>()?,
                    Some(_) => {
                        return Err(tools(format!(
                            "message {index}: tool_calls must be an array"
                        )));
                    }
                };
                Ok(Message::Assistant {
                    content: content.unwrap_or_default(),
                    reasoning: self.reasoning_content,
                    calls,
                })
            }
            "tool" => {
                no_reasoning(&self.reasoning_content)?;
                let call_id = self.tool_call_id.ok_or_else(|| {
                    input(format!(
                        "message {index}: a tool message needs tool_call_id"
                    ))
                })?;
                // The GGUF template prints null as "None"; refuse it.
                Ok(Message::Tool {
                    call_id,
                    content: text(content)?,
                })
            }
            other => Err(input(format!(
                "message {index}: role {other:?} is not system, user, assistant or tool"
            ))),
        }
    }
}

pub fn parse_document(bytes: &[u8]) -> Result<ChatDocument> {
    let wire: Wire = serde_json::from_slice(bytes)
        .map_err(|e| input(format!("invalid GLM chat message document: {e}")))?;
    let (messages, clear_thinking, definitions) = match wire {
        Wire::Array(messages) => (messages, None, Vec::new()),
        Wire::Object(document) => {
            let definitions = match document.tools {
                None | Some(serde_json::Value::Null) => Vec::new(),
                Some(serde_json::Value::Array(definitions)) => definitions
                    .iter()
                    .map(ToolDefinition::from_value)
                    .collect::<Result<_>>()?,
                Some(_) => return Err(tools("tools must be an array of tool definitions")),
            };
            (document.messages, document.clear_thinking, definitions)
        }
    };
    let messages = messages
        .into_iter()
        .enumerate()
        .map(|(index, message)| message.into_message(index))
        .collect::<Result<_>>()?;
    Ok(ChatDocument {
        messages,
        clear_thinking,
        tools: definitions,
    })
}

/// Python's `str.isspace`, which `str.strip()` uses: Unicode `White_Space`
/// plus U+001C..U+001F (bidi class B/S).
fn python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// The template's assistant split: with no reasoning field and a `</think>`
/// in the content, reasoning is the text before the first `</think>` after
/// its last `<think>`, and content is the text after the last `</think>`.
fn assistant_parts<'a>(content: &'a str, reasoning: Option<&'a str>) -> (Option<&'a str>, &'a str) {
    if reasoning.is_some() {
        return (reasoning, content);
    }
    let (Some(first), Some(last)) = (content.find(THINK_CLOSE), content.rfind(THINK_CLOSE)) else {
        return (None, content);
    };
    let head = &content[..first];
    let reasoning = head.rsplit(THINK_OPEN).next().unwrap_or(head);
    (Some(reasoning), &content[last + THINK_CLOSE.len()..])
}

/// Render the complete prompt, `[gMASK]<sop>` included: encode it without
/// added special tokens. Authored marker-like text is rendered verbatim, as
/// upstream does. No tools.
pub fn render(messages: &[Message], options: RenderOptions) -> Result<String> {
    render_with_tools(messages, &[], options)
}

/// [`render`] with declared tools (the definitions block, calls and results).
pub fn render_with_tools(
    messages: &[Message],
    definitions: &[ToolDefinition],
    options: RenderOptions,
) -> Result<String> {
    if messages.is_empty() {
        return Err(input("the conversation is empty"));
    }
    if options.add_generation_prompt
        && !matches!(
            messages.last(),
            Some(Message::User(_) | Message::Tool { .. })
        )
    {
        return Err(input(
            "generation requires the conversation to end in a user turn or tool results",
        ));
    }
    let blocks = tools::tool_blocks(messages, definitions)?;
    let last_user = messages.iter().rposition(|m| matches!(m, Message::User(_)));
    let mut out = String::from(PREFIX);
    out.push_str("<|system|>Reasoning Effort: ");
    out.push_str(options.effort.label());
    tools::render_definitions(definitions, &mut out)?;
    for (index, message) in messages.iter().enumerate() {
        match message {
            Message::System(text) => {
                out.push_str("<|system|>");
                out.push_str(text);
            }
            Message::User(text) => {
                out.push_str("<|user|>");
                out.push_str(text);
            }
            Message::Assistant {
                content,
                reasoning,
                calls,
            } => {
                let (reasoning, content) = assistant_parts(content, reasoning.as_deref());
                let keep = !options.clear_thinking || last_user.is_none_or(|last| index > last);
                out.push_str("<|assistant|>");
                out.push_str(THINK_OPEN);
                if keep && let Some(reasoning) = reasoning {
                    out.push_str(reasoning);
                }
                out.push_str(THINK_CLOSE);
                out.push_str(content.trim_matches(python_space));
                for call in calls {
                    call.render(&mut out)?;
                }
            }
            Message::Tool { .. } => {
                // Rendered once per block, in the preceding turn's call order.
                if let Some(order) = blocks.get(&index) {
                    out.push_str(OBSERVATION);
                    for &k in order {
                        let Message::Tool { content, .. } = &messages[k] else {
                            unreachable!("tool blocks hold tool messages");
                        };
                        out.push_str(TOOL_RESPONSE_OPEN);
                        out.push_str(content);
                        out.push_str(TOOL_RESPONSE_CLOSE);
                    }
                }
            }
        }
    }
    if options.add_generation_prompt {
        out.push_str(GENERATION_PROMPT);
    }
    Ok(out)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VerifiedChatProfile {
    pub renderer: &'static str,
    pub source_revision: &'static str,
    /// The embedded template's digest: the pinned GGUF conversion's, or the
    /// upstream file verbatim.
    pub template_sha256: &'static str,
    pub template_source: &'static str,
    pub generation_config_sha256: &'static str,
    pub tokenizer_metadata_id: String,
    pub verification: &'static str,
}

/// Chat eligibility: the embedded template is one this renderer reproduces,
/// every marker encodes to its single released token, and the artifact's
/// stops are the released set. Not checkpoint content authentication; raw
/// input stays available either way.
pub fn verify_profile(
    source: &GgufFile,
    tokenizer: &NativeTokenizer,
    stops: &[i32],
) -> Result<VerifiedChatProfile> {
    let unverified = |message: String| ChatError::unverified(message);
    let template = source
        .get_str("tokenizer.chat_template")
        .ok_or_else(|| unverified("the GGUF has no tokenizer.chat_template".into()))?;
    let digest = format!("{:x}", Sha256::digest(template.as_bytes()));
    let (template_sha256, template_source) = match digest.as_str() {
        GGUF_TEMPLATE_SHA256 => (GGUF_TEMPLATE_SHA256, "unsloth_gguf"),
        TEMPLATE_SHA256 => (TEMPLATE_SHA256, "upstream"),
        _ => {
            return Err(unverified(format!(
                "embedded chat template {digest} is not a pinned GLM-5.3-Flash template; use raw input"
            )));
        }
    };
    for (marker, id) in MARKERS.into_iter().chain(tools::TOOL_MARKERS) {
        let ids = tokenizer
            .encode(marker, false)
            .map_err(|e| unverified(format!("encode {marker}: {e}")))?;
        if ids != [id] {
            return Err(unverified(format!(
                "{marker} encodes to {ids:?}, not [{id}]"
            )));
        }
    }
    if stops != CHAT_STOPS {
        return Err(unverified(format!(
            "generation stops {stops:?} are not the released {CHAT_STOPS:?}"
        )));
    }
    Ok(VerifiedChatProfile {
        renderer: RENDERER,
        source_revision: REVISION,
        template_sha256,
        template_source,
        generation_config_sha256: GENERATION_CONFIG_SHA256,
        tokenizer_metadata_id: format!(
            "{:016x}",
            crate::runtime::tokenizer_metadata_identity(source)
        ),
        verification: "embedded_template_sha256_marker_token_ids_and_stops",
    })
}

#[cfg(test)]
mod tests;
