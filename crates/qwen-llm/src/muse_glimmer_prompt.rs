//! Byte-explicit Muse Glimmer ATEM prompt rendering.

use crate::muse_glimmer::MuseGlimmerChatTemplateProfile;
use serde_json::Value;
use std::collections::BTreeMap;

pub const MUSE_GLIMMER_BOS: &str = "<|begin_of_text|>";
pub const MUSE_GLIMMER_START: &str = "<|start|>";
pub const MUSE_GLIMMER_MESSAGE: &str = "<|message|>";
pub const MUSE_GLIMMER_EOM: &str = "<|eom|>";
pub const MUSE_GLIMMER_EOT: &str = "<|eot|>";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MuseGlimmerReasoningStrength {
    Low,
    Medium,
    #[default]
    High,
}

impl MuseGlimmerReasoningStrength {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
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
    if messages.is_empty() {
        return Err(MuseGlimmerPromptError::EmptyMessages);
    }
    validate_options(options)?;
    validate_messages(messages)?;

    let mut output = String::from(MUSE_GLIMMER_BOS);
    if !messages
        .iter()
        .any(|message| message.role == MuseGlimmerMessageRole::System)
    {
        output.push_str("<|start|>system<|message|>You are a helpful AI assistant.");
        output.push_str("\nKnowledge cutoff: ");
        output.push_str(&options.knowledge_cutoff);
        output.push('.');
        output.push_str("\nCurrent date: ");
        output.push_str(&options.current_date);
        output.push_str(".\n\n");
        render_reasoning(&mut output, options.reasoning_strength);
        if !options.tools.is_empty() {
            output.push_str("\n\n");
            render_tool_definitions(&mut output, options)?;
        }
        output.push_str("\n\n");
        render_system_meta(&mut output, options);
        output.push_str(MUSE_GLIMMER_EOT);
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
                output.push_str("<|start|>system<|message|>");
                let content = match options.profile {
                    MuseGlimmerChatTemplateProfile::MetaFixed => message.content.clone(),
                    MuseGlimmerChatTemplateProfile::UnslothLaunch => {
                        normalize_reasoning_effort(&message.content)
                    }
                };
                output.push_str(&content);
                if options.profile == MuseGlimmerChatTemplateProfile::MetaFixed
                    || !content.to_ascii_lowercase().contains("reasoning strength")
                {
                    output.push_str("\n\n");
                    render_reasoning(&mut output, options.reasoning_strength);
                }
                if !options.tools.is_empty() {
                    output.push_str("\n\n");
                    render_tool_definitions(&mut output, options)?;
                }
                output.push_str("\n\n");
                render_system_meta(&mut output, options);
                output.push_str(MUSE_GLIMMER_EOT);
            }
            MuseGlimmerMessageRole::User => {
                output.push_str("<|start|>user<|message|>");
                output.push_str(&message.content);
                output.push_str(MUSE_GLIMMER_EOT);
            }
            MuseGlimmerMessageRole::Tool { name } => {
                output.push_str("<|start|>tool ");
                output.push_str(name);
                output.push_str("<|message|><tool_output name=\"");
                output.push_str(name);
                output.push_str("\">\n");
                output.push_str(&message.content);
                output.push_str("\n</tool_output><|eot|>");
            }
            MuseGlimmerMessageRole::Assistant => {
                if let Some(reasoning) = message
                    .reasoning_content
                    .as_deref()
                    .filter(|reasoning| !reasoning.is_empty())
                {
                    output.push_str("<|start|>assistant to=self<|message|>");
                    output.push_str(reasoning);
                    output.push_str(MUSE_GLIMMER_EOM);
                }
                if message.tool_calls.is_empty() {
                    let recipient = message.recipient.as_deref().unwrap_or("user");
                    let end_turn = message.end_turn.unwrap_or(recipient == "user");
                    output.push_str("<|start|>assistant to=");
                    output.push_str(recipient);
                    output.push_str(MUSE_GLIMMER_MESSAGE);
                    output.push_str(&message.content);
                    output.push_str(if end_turn {
                        MUSE_GLIMMER_EOT
                    } else {
                        MUSE_GLIMMER_EOM
                    });
                } else {
                    for (call_index, call) in message.tool_calls.iter().enumerate() {
                        output.push_str("<|start|>assistant to=");
                        output.push_str(&call.name);
                        output.push_str(MUSE_GLIMMER_MESSAGE);
                        render_atem_call(&mut output, call)?;
                        output.push_str(if call_index + 1 == message.tool_calls.len() {
                            end_token
                        } else {
                            MUSE_GLIMMER_EOM
                        });
                    }
                }
            }
        }
    }
    if options.add_generation_prompt {
        output.push_str("<|start|>assistant");
    }
    Ok(output)
}

fn validate_options(options: &MuseGlimmerPromptOptions) -> Result<(), MuseGlimmerPromptError> {
    for (field, value) in [
        ("knowledge_cutoff", options.knowledge_cutoff.as_str()),
        ("current_date", options.current_date.as_str()),
    ] {
        if value.is_empty() {
            return Err(MuseGlimmerPromptError::InvalidTool {
                index: 0,
                field,
                detail: "must not be empty".into(),
            });
        }
    }
    for (index, tool) in options.tools.iter().enumerate() {
        if tool.name.is_empty() {
            return Err(MuseGlimmerPromptError::InvalidTool {
                index,
                field: "name",
                detail: "must not be empty".into(),
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
    Ok(())
}

fn validate_messages(messages: &[MuseGlimmerMessage]) -> Result<(), MuseGlimmerPromptError> {
    for (index, message) in messages.iter().enumerate() {
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
            && name.is_empty()
        {
            return Err(MuseGlimmerPromptError::InvalidMessage {
                index,
                field: "tool name",
                detail: "must not be empty".into(),
            });
        }
        for call in &message.tool_calls {
            if call.name.is_empty() || !call.arguments.is_object() {
                return Err(MuseGlimmerPromptError::InvalidMessage {
                    index,
                    field: "tool_calls",
                    detail: "each call requires a name and object arguments".into(),
                });
            }
        }
    }
    Ok(())
}

fn render_reasoning(output: &mut String, strength: MuseGlimmerReasoningStrength) {
    output.push_str("Reasoning strength: ");
    output.push_str(strength.as_str());
    output.push('.');
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
    for namespace in tool_namespaces(&options.tools) {
        output.push_str(", \"");
        output.push_str(namespace);
        output.push_str(".*\"");
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

fn normalize_reasoning_effort(content: &str) -> String {
    content
        .replace("Reasoning effort", "Reasoning strength")
        .replace("Reasoning Effort", "Reasoning Strength")
        .replace("reasoning effort", "reasoning strength")
        .replace("REASONING EFFORT", "REASONING STRENGTH")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_release_default_single_turn_bytes() {
        let prompt = render_muse_glimmer_single_turn(
            "Why is the sky blue?",
            None,
            &MuseGlimmerPromptOptions::default(),
        )
        .unwrap();
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
    }

    #[test]
    fn unsloth_profile_normalizes_existing_reasoning_directive_once() {
        let prompt = render_muse_glimmer_single_turn(
            "hello",
            Some("Be concise.\n\nReasoning Effort: low."),
            &MuseGlimmerPromptOptions::default(),
        )
        .unwrap();
        assert!(prompt.contains("Reasoning Strength: low."));
        assert!(!prompt.contains("Reasoning strength: high."));
        assert!(!prompt.contains("Reasoning Effort"));
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
        let prompt = render_muse_glimmer_atem_prompt(&messages, &options).unwrap();
        assert!(prompt.contains("# Valid recipients: \"self\", \"weather.*\", \"user\"."));
        assert!(prompt.contains("<|start|>assistant to=self<|message|>I should check.<|eom|>"));
        assert!(prompt.contains("<atem:invoke name=\"weather.lookup\">"));
        assert!(prompt.contains("<atem:parameter name=\"city\">New York</atem:parameter>"));
        assert!(prompt.contains("<|start|>tool weather.lookup<|message|><tool_output name=\"weather.lookup\">\nSunny\n</tool_output><|eot|>"));
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
                &MuseGlimmerPromptOptions::default(),
            )
            .is_err()
        );
    }
}
