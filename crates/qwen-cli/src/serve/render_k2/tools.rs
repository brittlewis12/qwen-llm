use super::super::output_partition::OutputProtocol;
use super::{
    ServeError, ServeRequest,
    chat::{self, ChatCapability},
    invalid,
};
use qwen_llm::k2_horizon_chat::{
    Effort,
    tools::{ToolChatInput, decode_tool_json},
};
use serde_json::{Value, json};

pub(crate) fn decode_request_json(body: &[u8]) -> Result<Value, ServeError> {
    let text = std::str::from_utf8(body)
        .map_err(|e| invalid("input", format!("request body is not UTF-8 JSON: {e}")))?;
    decode_tool_json(text).map_err(|e| invalid("input", format!("request body is not JSON: {e}")))
}

fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str, ServeError> {
    v[key]
        .as_str()
        .ok_or_else(|| invalid("input", format!("{key} must be a string")))
}
fn metadata(item: &Value, allowed: &[&str]) -> Result<(), ServeError> {
    chat::fields(item, allowed, "input")?;
    if item.get("type").is_some_and(|v| !v.is_string()) {
        return Err(invalid("input", "item type must be a string"));
    }
    if item.get("status").is_some_and(|v| v != "completed") {
        return Err(invalid("input", "only completed history may be replayed"));
    }
    if item.get("id").is_some_and(|v| !v.is_string()) {
        return Err(invalid("input", "item id must be a string"));
    }
    Ok(())
}

/// Shared by HTTP and CLI's Responses-shaped messages documents; verification
/// remains the caller's responsibility before invoking this wire adapter.
/// Also returns how many assistant turns arrived without a reasoning item;
/// each renders with explicit empty reasoning (missing is empty reasoning,
/// serve's rule for every family), and the caller reports the count.
pub(crate) fn input_from_responses(
    body: &Value,
    effort: Effort,
) -> Result<(ToolChatInput, usize), ServeError> {
    let tools = body
        .get("tools")
        .map(|v| {
            v.as_array()
                .ok_or_else(|| invalid("tools", "tools must be an array"))
        })
        .transpose()?
        .cloned()
        .unwrap_or_default();
    let mut definitions = Vec::new();
    for tool in tools {
        chat::fields(
            &tool,
            &["type", "name", "description", "parameters", "strict"],
            "tools",
        )?;
        if tool["type"] != "function" {
            return Err(invalid("tools", "only function tools are supported"));
        }
        if tool.get("strict").is_some_and(|v| v != false) {
            return Err(invalid(
                "tools",
                "K2 does not provide strict schema-constrained generation",
            ));
        }
        let mut function = tool.as_object().unwrap().clone();
        function.remove("type");
        definitions.push(json!({"type":"function","function":function}));
    }
    let mut messages = Vec::<Value>::new();
    if let Some(instructions) = body.get("instructions") {
        messages.push(json!({"role":"system","content":instructions.as_str().ok_or_else(|| invalid("instructions","instructions must be a string"))?}));
    }
    let mut reasoning: Option<String> = None;
    let mut assistant: Option<usize> = None;
    let mut history_reasoning_missing = 0;
    for item in body["input"]
        .as_array()
        .ok_or_else(|| invalid("input", "expected message items"))?
    {
        match item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message")
        {
            "reasoning" => {
                metadata(item, &["type", "id", "status", "summary", "content"])?;
                if item
                    .get("summary")
                    .is_some_and(|v| !v.as_array().is_some_and(Vec::is_empty))
                {
                    return Err(invalid("input", "reasoning summary must be empty"));
                }
                if reasoning.is_some() {
                    return Err(invalid("input", "orphan reasoning item"));
                }
                reasoning = Some(chat::text(&item["content"], "reasoning_text")?);
                assistant = None;
            }
            "message" => {
                metadata(item, &["type", "role", "content", "id", "status"])?;
                let role = string(item, "role")?;
                if !["system", "user", "assistant"].contains(&role) {
                    return Err(invalid("input", "unsupported message role"));
                }
                let mut message = json!({"role":role,"content":chat::text(&item["content"], if role == "assistant" {"output_text"} else {"input_text"})?});
                if role == "assistant" {
                    message["reasoning"] = json!(reasoning.take().unwrap_or_else(|| {
                        history_reasoning_missing += 1;
                        String::new()
                    }));
                    assistant = Some(messages.len());
                } else {
                    if reasoning.is_some() {
                        return Err(invalid("input", "orphan reasoning item"));
                    }
                    assistant = None;
                }
                messages.push(message);
            }
            "function_call" => {
                metadata(
                    item,
                    &["type", "id", "status", "call_id", "name", "arguments"],
                )?;
                if let Some(reasoning) = reasoning.take() {
                    assistant = Some(messages.len());
                    messages.push(json!({"role":"assistant","content":"","reasoning":reasoning}));
                } else if assistant.is_none() {
                    // A call group replayed without its reasoning item opens
                    // an assistant turn with empty reasoning; its parallel
                    // calls attach to it below.
                    history_reasoning_missing += 1;
                    assistant = Some(messages.len());
                    messages.push(json!({"role":"assistant","content":"","reasoning":""}));
                }
                let index = assistant.expect("a call group always has an assistant turn");
                let arguments = decode_tool_json(string(item, "arguments")?)
                    .map_err(|e| invalid("input", e.to_string()))?;
                if !arguments.is_object() {
                    return Err(invalid("input", "function arguments must encode an object"));
                }
                if messages[index].get("tool_calls").is_none() {
                    messages[index]["tool_calls"] = json!([]);
                }
                messages[index]["tool_calls"].as_array_mut().unwrap().push(json!({"id":string(item,"call_id")?,"type":"function","function":{"name":string(item,"name")?,"arguments":arguments}}));
            }
            "function_call_output" => {
                metadata(item, &["type", "id", "status", "call_id", "output"])?;
                if reasoning.is_some() {
                    return Err(invalid("input", "orphan reasoning item"));
                }
                let output = item
                    .get("output")
                    .filter(|v| v.is_string() || v.is_object() || v.is_array())
                    .ok_or_else(|| {
                        invalid("input", "tool output must be a string, object, or array")
                    })?;
                messages.push(
                    json!({"role":"tool","tool_call_id":string(item,"call_id")?,"content":output}),
                );
                assistant = None;
            }
            _ => return Err(invalid("input", "unsupported K2 history item")),
        }
    }
    if reasoning.is_some() {
        return Err(invalid("input", "orphan reasoning item"));
    }
    let mut document = json!({"messages":messages,"tools":definitions});
    if let Some(extension) = body.get("x_k2") {
        chat::fields(
            extension,
            &[
                "add_special_tokens",
                "tool_presentation_format",
                "tool_call_format",
            ],
            "x_k2",
        )?;
        if extension
            .get("add_special_tokens")
            .is_some_and(|v| v != true)
        {
            return Err(invalid("x_k2", "K2 chat requires native BOS"));
        }
        for key in ["tool_presentation_format", "tool_call_format"] {
            if let Some(value) = extension.get(key) {
                document[key] = value.clone();
            }
        }
    }
    ToolChatInput::from_document(&document, effort)
        .map(|input| (input, history_reasoning_missing))
        .map_err(|e| invalid("input", e.to_string()))
}

pub(super) fn parse_with_profile(
    body: &Value,
    profile: Option<&ChatCapability>,
) -> Result<ServeRequest, ServeError> {
    profile.ok_or_else(|| invalid("input", "K2 tools require a verified checkpoint profile"))?;
    chat::fields(
        body,
        &[
            "model",
            "input",
            "instructions",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "stream",
            "max_output_tokens",
            "temperature",
            "top_p",
            "store",
            "truncation",
            "x_qwen",
            "x_k2",
        ],
        "input",
    )?;
    if body.get("tool_choice").is_some_and(|v| v != "auto") {
        return Err(invalid(
            "tool_choice",
            "K2 supports only automatic tool choice",
        ));
    }
    if body.get("parallel_tool_calls").is_some_and(|v| v != true) {
        return Err(invalid(
            "parallel_tool_calls",
            "K2 native template permits multiple calls; false is not supported",
        ));
    }
    let effort = if let Some(reasoning) = body.get("reasoning") {
        chat::fields(reasoning, &["effort"], "reasoning")?;
        Effort::parse(Some(string(reasoning, "effort")?))
    } else {
        Effort::parse(None)
    }
    .map_err(|e| invalid("reasoning", e.to_string()))?;
    let (input, history_reasoning_missing) = input_from_responses(body, effort)?;
    if input.config.definitions.is_empty() {
        // The no-tools parser re-reads the input and reports its own count.
        let mut plain = body.clone();
        let map = plain.as_object_mut().unwrap();
        for key in ["tools", "tool_choice", "parallel_tool_calls"] {
            map.remove(key);
        }
        if let Some(extension) = map.get_mut("x_k2").and_then(Value::as_object_mut) {
            extension.remove("tool_presentation_format");
            extension.remove("tool_call_format");
        }
        return chat::parse_with_profile(&plain, profile);
    }
    let mut controls = body.clone();
    let map = controls.as_object_mut().unwrap();
    for key in [
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "instructions",
        "reasoning",
    ] {
        map.remove(key);
    }
    if let Some(extension) = map.get_mut("x_k2").and_then(Value::as_object_mut) {
        extension.remove("tool_presentation_format");
        extension.remove("tool_call_format");
    }
    map.insert("input".into(), json!("K2 chat controls"));
    let mut request = super::parse_request(&controls)?;
    request.k2_raw_input = None;
    request.allowed_tools = input
        .config
        .names()
        .map_err(|e| invalid("tools", e.to_string()))?;
    request.instructions = body
        .get("instructions")
        .and_then(Value::as_str)
        .map(str::to_owned);
    request.reasoning = Some(json!({"effort":effort}));
    request.k2_tools = Some(input);
    request.history_reasoning_missing = history_reasoning_missing;
    Ok(request)
}
pub(super) fn render_with_profile(
    request: &ServeRequest,
    profile: Option<&ChatCapability>,
) -> Result<String, ServeError> {
    profile.ok_or_else(|| invalid("input", "K2 tools require a verified checkpoint profile"))?;
    let chat = request.k2_tools.as_ref().unwrap();
    if request.k2_chat.is_some()
        || request.k2_raw_input.is_some()
        || request.k2_add_special_tokens != Some(true)
        || request.model_request.has_tool_surface()
        || request.no_thinking
        || request.strip_history_thinking
        || request.allowed_tools
            != chat
                .config
                .names()
                .map_err(|e| invalid("tools", e.to_string()))?
    {
        return Err(invalid(
            "input",
            "invalid K2 tool request ownership or controls",
        ));
    }
    chat.render().map_err(|e| invalid("input", e.to_string()))
}
pub(crate) fn byte_budget(max_tokens: usize, max_piece: usize) -> Result<usize, ServeError> {
    max_tokens
        .checked_mul(max_piece)
        .and_then(|n| n.checked_mul(3))
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            invalid(
                "max_output_tokens",
                "K2 decoded output byte bound overflow or zero",
            )
        })
}
pub(crate) fn output_protocol(request: &ServeRequest, max_piece: usize) -> OutputProtocol {
    if let Some(chat) = &request.k2_tools {
        if !chat.config.definitions.is_empty() {
            return OutputProtocol::K2Tools {
                effort: chat.effort,
                config: chat.config.clone(),
                max_bytes: byte_budget(request.max_output_tokens.unwrap_or(0), max_piece)
                    .unwrap_or(0),
            };
        }
        return OutputProtocol::K2Chat {
            effort: chat.effort,
        };
    }
    request
        .k2_chat
        .as_ref()
        .map_or(OutputProtocol::RawText, |chat| OutputProtocol::K2Chat {
            effort: chat.effort,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k2_wire_json_preserves_containers_and_rejects_ambiguous_duplicate_keys() {
        for body in [
            r#"{"model":"m","input":"literal <ifm|tool_calls>"}"#,
            r#"{"model":"m","input":[{"role":"user","content":"hi"}]}"#,
        ] {
            assert_eq!(
                decode_request_json(body.as_bytes()).unwrap(),
                serde_json::from_str::<Value>(body).unwrap()
            );
            let duplicate = body.replacen("{", r#"{"model":"different","#, 1);
            assert!(decode_request_json(duplicate.as_bytes()).is_err());
        }
        let value = format!(
            "{}{{\"$serde_json::private::Number\":\"7\"}}{}",
            "[".repeat(64),
            "]".repeat(64)
        );
        let mut decoded = decode_request_json(value.as_bytes()).unwrap();
        for _ in 0..64 {
            decoded = decoded.as_array_mut().unwrap().remove(0);
        }
        assert!(decoded.is_object());
        assert_eq!(decoded["$serde_json::private::Number"], "7");
        assert!(
            decode_request_json(format!("{}0{}", "[".repeat(129), "]".repeat(129)).as_bytes())
                .is_err()
        );
    }
}
