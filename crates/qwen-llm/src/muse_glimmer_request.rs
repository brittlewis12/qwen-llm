//! Shared structured request contract for Muse Glimmer ATEM prompts.

use crate::muse_glimmer::MuseGlimmerChatTemplateProfile;
use crate::muse_glimmer_prompt::{
    AnnotatedMuseGlimmerPrompt, MuseGlimmerMessage, MuseGlimmerPromptError,
    MuseGlimmerPromptOptions, MuseGlimmerReasoningStrength, MuseGlimmerToolCall,
    MuseGlimmerToolDefinition, render_muse_glimmer_atem_prompt_annotated,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerRequest {
    pub messages: Vec<MuseGlimmerMessage>,
    pub tools: Vec<MuseGlimmerToolDefinition>,
    pub tool_namespace_descriptions: BTreeMap<String, String>,
    pub reasoning_strength: Option<MuseGlimmerReasoningStrength>,
    pub current_date: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum MuseGlimmerRequestError {
    #[error("invalid Muse Glimmer request JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid Muse Glimmer request: {0}")]
    Contract(String),
    #[error(transparent)]
    Prompt(#[from] MuseGlimmerPromptError),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum WireDocument {
    Bare(Vec<WireMessage>),
    Wrapped(WireWrapper),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireWrapper {
    messages: Vec<WireMessage>,
    #[serde(default)]
    tools: Vec<WireToolDefinition>,
    #[serde(default)]
    tool_namespace_descriptions: BTreeMap<String, String>,
    reasoning_strength: Option<MuseGlimmerReasoningStrength>,
    current_date: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMessage {
    role: String,
    content: String,
    name: Option<String>,
    reasoning_content: Option<String>,
    recipient: Option<String>,
    end_turn: Option<bool>,
    #[serde(default)]
    tool_calls: Vec<WireToolCall>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireToolCall {
    name: String,
    arguments: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireToolDefinition {
    name: String,
    #[serde(default)]
    description: String,
    parameters: Value,
}

impl MuseGlimmerRequest {
    pub fn single_turn(user: impl Into<String>, system: Option<String>) -> Self {
        let mut messages = Vec::with_capacity(usize::from(system.is_some()) + 1);
        if let Some(system) = system {
            messages.push(MuseGlimmerMessage::system(system));
        }
        messages.push(MuseGlimmerMessage::user(user));
        Self {
            messages,
            tools: Vec::new(),
            tool_namespace_descriptions: BTreeMap::new(),
            reasoning_strength: None,
            current_date: None,
        }
    }

    pub fn from_json(raw: &str) -> Result<Self, MuseGlimmerRequestError> {
        let document: WireDocument = serde_json::from_str(raw)?;
        let (messages, tools, tool_namespace_descriptions, reasoning_strength, current_date) =
            match document {
                WireDocument::Bare(messages) => (messages, Vec::new(), BTreeMap::new(), None, None),
                WireDocument::Wrapped(wrapper) => (
                    wrapper.messages,
                    wrapper.tools,
                    wrapper.tool_namespace_descriptions,
                    wrapper.reasoning_strength,
                    wrapper.current_date,
                ),
            };
        let messages = messages
            .into_iter()
            .enumerate()
            .map(|(index, message)| convert_message(index, message))
            .collect::<Result<Vec<_>, _>>()?;
        let tools = tools
            .into_iter()
            .map(|tool| MuseGlimmerToolDefinition {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
            })
            .collect();
        Ok(Self {
            messages,
            tools,
            tool_namespace_descriptions,
            reasoning_strength,
            current_date,
        })
    }

    pub fn resolved_reasoning_strength(
        &self,
        requested: Option<MuseGlimmerReasoningStrength>,
    ) -> Result<MuseGlimmerReasoningStrength, MuseGlimmerRequestError> {
        if let (Some(document), Some(requested)) = (self.reasoning_strength, requested)
            && document != requested
        {
            return Err(MuseGlimmerRequestError::Contract(format!(
                "document reasoning strength {} conflicts with requested {}",
                document.as_str(),
                requested.as_str()
            )));
        }
        Ok(requested
            .or(self.reasoning_strength)
            .unwrap_or(MuseGlimmerReasoningStrength::High))
    }

    pub fn render(
        &self,
        profile: MuseGlimmerChatTemplateProfile,
        requested: Option<MuseGlimmerReasoningStrength>,
    ) -> Result<String, MuseGlimmerRequestError> {
        Ok(self.render_annotated(profile, requested)?.text)
    }

    pub fn render_annotated(
        &self,
        profile: MuseGlimmerChatTemplateProfile,
        requested: Option<MuseGlimmerReasoningStrength>,
    ) -> Result<AnnotatedMuseGlimmerPrompt, MuseGlimmerRequestError> {
        if self.current_date.is_some()
            && self.messages.first().is_some_and(|message| {
                matches!(
                    message.role,
                    crate::muse_glimmer_prompt::MuseGlimmerMessageRole::System
                )
            })
        {
            return Err(MuseGlimmerRequestError::Contract(
                "current_date applies only to the synthesized system message; include the date in explicit system content instead"
                    .into(),
            ));
        }
        let mut options = MuseGlimmerPromptOptions {
            profile,
            reasoning_strength: self.resolved_reasoning_strength(requested)?,
            tools: self.tools.clone(),
            tool_namespace_descriptions: self.tool_namespace_descriptions.clone(),
            ..MuseGlimmerPromptOptions::default()
        };
        if let Some(current_date) = &self.current_date {
            options.current_date.clone_from(current_date);
        }
        Ok(render_muse_glimmer_atem_prompt_annotated(
            &self.messages,
            &options,
        )?)
    }
}

fn convert_message(
    index: usize,
    wire: WireMessage,
) -> Result<MuseGlimmerMessage, MuseGlimmerRequestError> {
    let WireMessage {
        role,
        content,
        name,
        reasoning_content,
        recipient,
        end_turn,
        tool_calls,
    } = wire;
    let mut message = match role.as_str() {
        "system" => {
            reject_name(index, name.as_deref())?;
            MuseGlimmerMessage::system(content)
        }
        "user" => {
            reject_name(index, name.as_deref())?;
            MuseGlimmerMessage::user(content)
        }
        "assistant" => {
            reject_name(index, name.as_deref())?;
            if !content.is_empty() && !tool_calls.is_empty() {
                return Err(MuseGlimmerRequestError::Contract(format!(
                    "message {index} cannot combine visible assistant content with tool calls"
                )));
            }
            MuseGlimmerMessage::assistant(content)
        }
        "tool" => MuseGlimmerMessage::tool(
            name.ok_or_else(|| {
                MuseGlimmerRequestError::Contract(format!("tool message {index} requires name"))
            })?,
            content,
        ),
        _ => {
            return Err(MuseGlimmerRequestError::Contract(format!(
                "message {index} has unsupported role {role:?}"
            )));
        }
    };
    message.reasoning_content = reasoning_content;
    message.recipient = recipient;
    message.end_turn = end_turn;
    message.tool_calls = tool_calls
        .into_iter()
        .map(|call| MuseGlimmerToolCall {
            name: call.name,
            arguments: call.arguments,
        })
        .collect();
    Ok(message)
}

fn reject_name(index: usize, name: Option<&str>) -> Result<(), MuseGlimmerRequestError> {
    if name.is_some() {
        return Err(MuseGlimmerRequestError::Contract(format!(
            "message {index} name is valid only for role tool"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_renders_structured_atem_history_with_annotations() {
        let request = MuseGlimmerRequest::from_json(
            r#"{
                "messages":[
                    {"role":"user","content":"Weather?"},
                    {"role":"assistant","content":"","reasoning_content":"Check.","tool_calls":[{"name":"weather.lookup","arguments":{"city":"Paris"}}]},
                    {"role":"tool","name":"weather.lookup","content":"Sunny"},
                    {"role":"user","content":"Thanks"}
                ],
                "tools":[{"name":"weather.lookup","description":"Weather","parameters":{"type":"object"}}],
                "reasoning_strength":"xhigh",
                "current_date":"2026-08-31"
            }"#,
        )
        .unwrap();
        let prompt = request
            .render_annotated(MuseGlimmerChatTemplateProfile::UnslothLaunch, None)
            .unwrap();
        assert!(prompt.text.contains("Reasoning strength: xhigh."));
        assert!(prompt.text.contains("Current date: 2026-08-31."));
        assert!(
            prompt
                .text
                .contains("<atem:invoke name=\"weather.lookup\">")
        );
        assert!(
            prompt
                .text
                .contains("<tool_output name=\"weather.lookup\">\nSunny")
        );
        assert!(prompt.spans.iter().any(|span| {
            span.kind == crate::muse_glimmer_prompt::MuseGlimmerPromptSpanKind::ToolCallContent
                && span.message_index == Some(1)
                && span.tool_call_index == Some(0)
        }));
        assert!(prompt.spans.iter().any(|span| {
            span.kind == crate::muse_glimmer_prompt::MuseGlimmerPromptSpanKind::ToolResultContent
                && span.message_index == Some(2)
        }));
    }

    #[test]
    fn requested_reasoning_is_authoritative_and_conflicts_fail() {
        let request = MuseGlimmerRequest::from_json(
            r#"{"messages":[{"role":"user","content":"hello"}],"reasoning_strength":"low"}"#,
        )
        .unwrap();
        assert!(
            request
                .render(
                    MuseGlimmerChatTemplateProfile::MetaFixed,
                    Some(MuseGlimmerReasoningStrength::High),
                )
                .is_err()
        );
    }

    #[test]
    fn rejects_unknown_fields_and_lossy_tool_composition() {
        for raw in [
            r#"{"messages":[{"role":"user","content":"hello","extra":1}]}"#,
            r#"{"messages":[{"role":"assistant","content":"visible","tool_calls":[{"name":"x","arguments":{}}]}]}"#,
            r#"{"messages":[{"role":"tool","content":"result"}]}"#,
        ] {
            assert!(MuseGlimmerRequest::from_json(raw).is_err());
        }
    }

    #[test]
    fn rejects_date_overrides_that_explicit_system_content_would_hide() {
        let request = MuseGlimmerRequest::from_json(
            r#"{
                "messages":[
                    {"role":"system","content":"Be exact."},
                    {"role":"user","content":"hello"}
                ],
                "current_date":"2026-08-31"
            }"#,
        )
        .unwrap();
        assert!(
            request
                .render(MuseGlimmerChatTemplateProfile::UnslothLaunch, None)
                .is_err()
        );
    }
}
