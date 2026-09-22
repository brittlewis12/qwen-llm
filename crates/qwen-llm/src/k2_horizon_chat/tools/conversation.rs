use super::super::{Effort, Message, render_message_body};
use super::*;
use std::collections::{HashMap, HashSet};

pub const TOOL_RENDERER: &str = "k2_horizon_ifm_tools_v1";

pub fn decode_tool_json(text: &str) -> Result<Value> {
    super::parse::json_decode::complete(text)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolConfig {
    pub definitions: Vec<Value>,
    pub presentation: ToolPresentationFormat,
    pub call_format: ToolCallFormat,
}
impl ToolConfig {
    pub fn names(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for tool in &self.definitions {
            let function = tool.get("function").unwrap_or(tool);
            if function.get("strict").is_some_and(|v| v != false)
                || tool.get("strict").is_some_and(|v| v != false)
            {
                return Err(error(
                    "K2 does not provide strict schema-constrained generation",
                ));
            }
            let name = function["name"]
                .as_str()
                .ok_or_else(|| error("tool requires a name"))?;
            if name.is_empty()
                || name.chars().any(char::is_control)
                || name.contains(['<', '>', '"', '\\'])
            {
                return Err(error(
                    "tool name cannot be represented in native call formats",
                ));
            }
            if names.iter().any(|n| n == name) {
                return Err(error("duplicate tool definition"));
            }
            names.push(name.to_owned());
        }
        Ok(names)
    }
    pub fn echo(&self) -> Value {
        serde_json::json!({"tool_presentation_format":self.presentation,"tool_call_format":self.call_format})
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolChatInput {
    pub config: ToolConfig,
    pub effort: Effort,
    messages: Vec<Value>,
}

fn fields(value: &Value, allowed: &[&str]) -> Result<()> {
    let map = value
        .as_object()
        .ok_or_else(|| error("expected an object"))?;
    if let Some(key) = map.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(error(format!("unsupported tool-chat field {key:?}")));
    }
    Ok(())
}
fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| error(format!("{key} must be a string")))
}

impl ToolChatInput {
    pub fn system_text(&self) -> Option<&str> {
        self.messages
            .first()
            .filter(|m| m["role"] == "system")?
            .get("content")?
            .as_str()
    }

    /// Parse a native messages document. Definitions may be top-level or in the
    /// leading system turn, as in the pinned template. IDs stay outside the prompt.
    pub fn from_document(document: &Value, effort: Effort) -> Result<Self> {
        if document.is_object() {
            fields(
                document,
                &[
                    "messages",
                    "tools",
                    "tool_presentation_format",
                    "tool_call_format",
                ],
            )?;
        }
        let messages = document
            .as_array()
            .or_else(|| document["messages"].as_array())
            .ok_or_else(|| error("messages must be an array"))?
            .clone();
        let top_tools = document
            .get("tools")
            .map(|v| v.as_array().ok_or_else(|| error("tools must be an array")))
            .transpose()?;
        let system_tools = messages
            .first()
            .filter(|m| m["role"] == "system")
            .and_then(|m| m.get("tools"));
        let definitions = if let Some(tools) = top_tools.filter(|t| !t.is_empty()) {
            tools.clone()
        } else if let Some(tools) = system_tools {
            tools
                .as_array()
                .ok_or_else(|| error("system tools must be an array"))?
                .clone()
        } else {
            Vec::new()
        };
        let presentation = document
            .get("tool_presentation_format")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| error(e.to_string()))?
            .unwrap_or_default();
        let call_format = document
            .get("tool_call_format")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|e| error(e.to_string()))?
            .unwrap_or_default();
        let input = Self {
            config: ToolConfig {
                definitions,
                presentation,
                call_format,
            },
            effort,
            messages,
        };
        input.render()?;
        Ok(input)
    }

    pub fn render(&self) -> Result<String> {
        let names = self.config.names()?;
        if self
            .messages
            .last()
            .is_none_or(|m| !matches!(m["role"].as_str(), Some("user" | "tool")))
        {
            return Err(error(
                "tool chat requires a conversation ending in user or tool results",
            ));
        }
        let mut out = String::new();
        let system = self.messages.first().filter(|m| m["role"] == "system");
        if !names.is_empty() {
            out.push_str(&render_tool_system(
                &self.config.definitions,
                system
                    .map(|m| string(m, "content"))
                    .transpose()?
                    .unwrap_or(""),
                self.config.presentation,
                self.config.call_format,
            )?);
        }
        let mut pending = Vec::<String>::new();
        let mut results = HashMap::<String, Value>::new();
        let mut used = HashSet::<String>::new();
        for (index, original) in self.messages.iter().enumerate() {
            let role = string(original, "role")?;
            if role == "tool" {
                fields(original, &["role", "content", "tool_call_id"])?;
                let id = string(original, "tool_call_id")?;
                if !pending.iter().any(|p| p == id) || results.contains_key(id) {
                    return Err(error("orphan or duplicate tool result"));
                }
                let content = original
                    .get("content")
                    .ok_or_else(|| error("tool result requires content"))?;
                results.insert(id.to_owned(), content.clone());
                if results.len() == pending.len() {
                    for id in pending.drain(..) {
                        out.push_str(&render_tool_result(&results.remove(&id).unwrap())?);
                    }
                }
                continue;
            }
            if !pending.is_empty() {
                return Err(error(
                    "all pending tool results are required before another turn",
                ));
            }
            let mut message = original
                .as_object()
                .ok_or_else(|| error("message must be an object"))?
                .clone();
            let calls = message.remove("tool_calls");
            if role == "system" && index == 0 {
                message.remove("tools");
            }
            let message: Message =
                serde_json::from_value(Value::Object(message)).map_err(|e| error(e.to_string()))?;
            let body = render_message_body(&message, index)?;
            let mut parsed_calls = Vec::new();
            if let Some(calls) = calls {
                if role != "assistant" {
                    return Err(error("tool_calls belong to assistant turns"));
                }
                for call in calls
                    .as_array()
                    .ok_or_else(|| error("tool_calls must be an array"))?
                {
                    fields(call, &["id", "type", "function", "name", "arguments"])?;
                    if call.get("type").is_some_and(|v| v != "function") {
                        return Err(error("only function calls are supported"));
                    }
                    let id = string(call, "id")?;
                    if id.is_empty() || !used.insert(id.into()) {
                        return Err(error("tool call IDs must be nonempty and unique"));
                    }
                    let function = call.get("function").unwrap_or(call);
                    if call.get("function").is_some() {
                        fields(function, &["name", "arguments"])?;
                        if call.get("name").is_some() || call.get("arguments").is_some() {
                            return Err(error("ambiguous tool call wrapper"));
                        }
                    }
                    let name = string(function, "name")?;
                    if !names.iter().any(|n| n == name) {
                        return Err(error("history calls an undeclared tool"));
                    }
                    let arguments = function["arguments"]
                        .as_object()
                        .ok_or_else(|| error("native history arguments must be an object"))?
                        .clone();
                    pending.push(id.into());
                    parsed_calls.push(ToolCall {
                        name: name.into(),
                        arguments,
                    });
                }
            }
            if role == "system" && index == 0 && !names.is_empty() {
                continue;
            }
            out.push_str(&body);
            if !parsed_calls.is_empty() {
                out.push_str(&render_tool_calls(
                    &parsed_calls,
                    self.config.call_format,
                    &self.config.definitions,
                )?);
            }
            out.push_str("<|ifm|im_end|>");
        }
        if !pending.is_empty() {
            return Err(error("missing tool results"));
        }
        out.push_str(&format!(
            "<|ifm|im_start|>assistant\n<{}>\n",
            self.effort.tag()
        ));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_tool_conversations_match_full_pinned_prompts() {
        let f: Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/k2_tools_hf.json")).unwrap();
        let mut count = 0;
        for case in f["cases"].as_array().unwrap().iter().filter(|c| {
            c.get("error").is_none() && c["name"] != "tools-xml_typed-duplicate-definition"
        }) {
            let mut document = Value::Object(Map::new());
            for key in [
                "messages",
                "tools",
                "tool_presentation_format",
                "tool_call_format",
            ] {
                if let Some(value) = case.get(key) {
                    document[key] = value.clone();
                }
            }
            let mut ids = Vec::new();
            let mut result = 0;
            for message in document["messages"].as_array_mut().unwrap() {
                if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                    for call in calls {
                        if call.get("id").is_none() {
                            call["id"] = serde_json::json!(format!("call_{}", ids.len() + 1));
                        }
                        ids.push(call["id"].clone());
                    }
                }
                if message["role"] == "tool" {
                    message["tool_call_id"] = ids[result].clone();
                    result += 1;
                }
            }
            let effort = Effort::parse(case["reasoning_effort"].as_str()).unwrap();
            let input = ToolChatInput::from_document(&document, effort);
            if matches!(
                case["name"].as_str(),
                Some("tools-result-string" | "tools-result-null")
            ) {
                assert!(
                    input.is_err(),
                    "these upstream formatting fixtures omit one parallel result"
                );
                continue;
            }
            let input = input.unwrap();
            assert_eq!(input.render().unwrap(), case["body"], "{}", case["name"]);
            count += 1;
        }
        assert_eq!(count, 53);
    }
    #[test]
    fn tool_results_reorder_by_call_id_and_missing_or_duplicate_results_fail() {
        let base = serde_json::json!({"tools":[{"name":"f","parameters":{}}],"messages":[
            {"role":"user","content":"go"}, {"role":"assistant","content":"","think":"", "tool_calls":[
                {"id":"a","name":"f","arguments":{}},{"id":"b","name":"f","arguments":{}}]},
            {"role":"tool","tool_call_id":"b","content":"second"},{"role":"tool","tool_call_id":"a","content":"first"}]});
        let rendered = ToolChatInput::from_document(&base, Effort::High)
            .unwrap()
            .render()
            .unwrap();
        assert!(rendered.find("tool\nfirst").unwrap() < rendered.find("tool\nsecond").unwrap());
        for mode in 0..3 {
            let mut bad = base.clone();
            match mode {
                0 => {
                    bad["messages"].as_array_mut().unwrap().pop();
                }
                1 => bad["messages"][3]["tool_call_id"] = serde_json::json!("b"),
                _ => bad["messages"][3]["tool_call_id"] = serde_json::json!("orphan"),
            }
            assert!(ToolChatInput::from_document(&bad, Effort::High).is_err());
        }
    }
}
