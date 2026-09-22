//! Native IFM tool presentation, history encoding and generated-call decoding.
//! These primitives do not authorize chat, advertise frontend support or execute tools.
use super::{Result, error};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

mod conversation;
mod json;
pub use conversation::{TOOL_RENDERER, ToolChatInput, ToolConfig, decode_tool_json};
mod output;
mod parse;
mod presentation;
pub use output::{ToolOutputEnd, ToolOutputFinish, ToolOutputStream};
pub use parse::{ParsedToolBlock, parse_tool_calls};
mod types;
pub use presentation::{ToolPresentationFormat, render_tool_definitions, render_tool_system};
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallFormat {
    #[default]
    Xml,
    Json,
    XmlTyped,
}

/// A call's wire ID belongs to frontend history bookkeeping. IFM's template
/// renders name/arguments only, in their original insertion order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

/// Render the pinned template's complete call block. IFM's XML dialect emits
/// strings verbatim, not XML entities. It is not an injective encoding of arbitrary
/// strings; callers must not assume render/parse round trips for delimiter text.
pub fn render_tool_calls(
    calls: &[ToolCall],
    format: ToolCallFormat,
    definitions: &[Value],
) -> Result<String> {
    let mut out = String::from("<ifm|tool_calls>");
    for call in calls {
        out.push_str("\n<ifm|tool_call>");
        if format == ToolCallFormat::Json {
            // Upstream builds the name literally, unlike the JSON arguments.
            if call.name.contains(['"', '\\']) || call.name.chars().any(char::is_control) {
                return Err(error(
                    "tool name cannot be represented in IFM JSON call syntax",
                ));
            }
            out.push_str("{\"name\": \"");
            out.push_str(&call.name);
            out.push_str("\", \"arguments\": ");
            out.push_str(&json::encode(&Value::Object(call.arguments.clone()))?);
            out.push('}');
        } else {
            out.push_str(&call.name);
            out.push('\n');
            for (key, value) in &call.arguments {
                out.push_str("<ifm|arg_key>");
                out.push_str(key);
                out.push_str("</ifm|arg_key>\n");
                if format == ToolCallFormat::XmlTyped {
                    out.push_str("<ifm|arg_type>");
                    out.push_str(&types::argument_type(definitions, &call.name, key, value)?);
                    out.push_str("</ifm|arg_type>\n");
                }
                out.push_str("<ifm|arg_value>");
                if let Some(text) = value.as_str() {
                    out.push_str(text);
                } else {
                    out.push_str(&json::encode(value)?);
                }
                out.push_str("</ifm|arg_value>\n");
            }
        }
        out.push_str("</ifm|tool_call>");
    }
    out.push_str("\n</ifm|tool_calls>");
    Ok(out)
}

/// Preserve the template's string/list/object tool-result forms. This does not
/// authenticate a result or associate it with a pending call.
pub fn render_tool_result(content: &Value) -> Result<String> {
    let mut out = String::from("<|ifm|im_start|>tool\n");
    if let Some(text) = content.as_str() {
        out.push_str(text);
    } else if let Some(items) = content.as_array() {
        if items.is_empty() {
            return Err(error("tool message content list must not be empty"));
        }
        for (index, item) in items.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            if let Some(text) = item
                .as_str()
                .or_else(|| item.get("text").and_then(Value::as_str))
            {
                out.push_str(text);
            } else {
                out.push_str(&json::encode(item)?);
            }
        }
    } else {
        out.push_str(&json::encode(content)?);
    }
    out.push_str("<|ifm|im_end|>");
    Ok(out)
}
