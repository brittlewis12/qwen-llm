use super::{ServeError, ServeRequest, invalid};
use qwen_llm::k2_horizon_chat::{self as k2, ChatInput, Effort, Message, VerifiedChatProfile};
use serde_json::{Map, Value, json};

pub(crate) struct ChatCapability {
    _profile: Option<VerifiedChatProfile>,
}

impl ChatCapability {
    pub(crate) fn verify(source: &qwen_llm::gguf::GgufFile) -> Result<Self, k2::ChatError> {
        Ok(Self {
            _profile: Some(k2::verify_profile_with_cancel(source, || {
                crate::shutdown::checkpoint().is_err()
            })?),
        })
    }

    #[cfg(test)]
    pub(crate) fn mock() -> Self {
        Self { _profile: None }
    }
}

pub(super) fn fields<'a>(
    value: &'a Value,
    allowed: &[&str],
    param: &'static str,
) -> Result<&'a Map<String, Value>, ServeError> {
    let map = value
        .as_object()
        .ok_or_else(|| invalid(param, "expected an object"))?;
    for (key, value) in map {
        if !allowed.contains(&key.as_str()) || value.is_null() {
            return Err(invalid(
                param,
                format!("unsupported K2 field/value {key:?}"),
            ));
        }
    }
    Ok(map)
}

fn string<'a>(map: &'a Map<String, Value>, key: &str) -> Result<&'a str, ServeError> {
    map.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("input", format!("{key} must be a string")))
}

pub(super) fn text(value: &Value, kind: &str) -> Result<String, ServeError> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_owned());
    }
    let parts = value
        .as_array()
        .ok_or_else(|| invalid("input", "content must be text or text parts"))?;
    let mut out = String::new();
    for part in parts {
        let map = fields(
            part,
            if kind == "output_text" {
                &["type", "text", "annotations"]
            } else {
                &["type", "text"]
            },
            "input",
        )?;
        empty_array_if_present(map, "annotations")?;
        if string(map, "type")? != kind {
            return Err(invalid("input", format!("expected {kind} parts")));
        }
        out.push_str(string(map, "text")?);
    }
    Ok(out)
}

fn empty_array_if_present(map: &Map<String, Value>, key: &str) -> Result<(), ServeError> {
    if let Some(value) = map.get(key) {
        if !value.as_array().is_some_and(Vec::is_empty) {
            return Err(invalid("input", format!("K2 supports only empty {key}")));
        }
    }
    Ok(())
}

pub(crate) fn parse_with_profile(
    body: &Value,
    profile: Option<&ChatCapability>,
) -> Result<ServeRequest, ServeError> {
    if body.get("input").is_some_and(Value::is_array)
        && (body.get("tools").is_some()
            || body.get("tool_choice").is_some()
            || body.get("parallel_tool_calls").is_some()
            || body
                .get("x_k2")
                .and_then(Value::as_object)
                .is_some_and(|m| m.keys().any(|k| k != "add_special_tokens"))
            || body["input"].as_array().unwrap().iter().any(|i| {
                matches!(
                    i["type"].as_str(),
                    Some("function_call" | "function_call_output")
                )
            }))
    {
        return super::tools::parse_with_profile(body, profile);
    }
    if !body.get("input").is_some_and(Value::is_array) {
        return super::parse_request(body);
    }
    profile.ok_or_else(|| {
        invalid(
            "input",
            "K2 chat requires a verified checkpoint profile; this artifact is raw-only",
        )
    })?;
    let map = fields(
        body,
        &[
            "model",
            "input",
            "instructions",
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
    let effort = if let Some(value) = map.get("reasoning") {
        let reasoning = fields(value, &["effort"], "reasoning")?;
        Effort::parse(Some(string(reasoning, "effort")?))
    } else {
        Effort::parse(None)
    }
    .map_err(|e| invalid("reasoning.effort", e.to_string()))?;
    let mut messages = Vec::new();
    let instructions = map
        .get("instructions")
        .map(|_| string(map, "instructions"))
        .transpose()?;
    if let Some(system) = instructions {
        messages.push(Message::text("system", system.into()));
    }
    let mut pending_reasoning = None;
    for item in map["input"].as_array().unwrap() {
        let kind = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        let item = fields(
            item,
            if kind == "reasoning" {
                &["type", "content", "id", "status", "summary"]
            } else {
                &["type", "role", "content", "id", "status"]
            },
            "input",
        )?;
        empty_array_if_present(item, "summary")?;
        if let Some(value) = item.get("type") {
            if value.as_str() != Some(kind) {
                return Err(invalid("input", "item type must be a string"));
            }
        }
        if let Some(value) = item.get("id") {
            if !value.is_string() {
                return Err(invalid("input", "item id must be a string"));
            }
        }
        if let Some(value) = item.get("status") {
            if value.as_str() != Some("completed") {
                return Err(invalid("input", "only completed history may be replayed"));
            }
        }
        let content = item
            .get("content")
            .ok_or_else(|| invalid("input", "content is required"))?;
        if kind == "reasoning" {
            if pending_reasoning.is_some() {
                return Err(invalid(
                    "input",
                    "reasoning must directly precede an assistant",
                ));
            }
            pending_reasoning = Some(text(content, "reasoning_text")?);
            continue;
        }
        if kind != "message" {
            return Err(invalid(
                "input",
                "only no-tools messages and reasoning items are supported",
            ));
        }
        let role = string(item, "role")?;
        let mut message = Message::text(
            role,
            text(
                content,
                if role == "assistant" {
                    "output_text"
                } else {
                    "input_text"
                },
            )?,
        );
        if role == "assistant" {
            message.reasoning = Some(pending_reasoning.take().ok_or_else(|| invalid("input", "assistant history requires an explicit preceding reasoning item (empty is valid)"))?);
        } else if pending_reasoning.is_some() {
            return Err(invalid(
                "input",
                "reasoning must directly precede an assistant",
            ));
        }
        messages.push(message);
    }
    if pending_reasoning.is_some() {
        return Err(invalid("input", "orphan reasoning item"));
    }
    k2::render(&messages, effort).map_err(|e| invalid("input", e.to_string()))?;

    // Reuse wire/sampling admission, not Qwen transcript normalization: literal
    // Qwen markers in IFM message text have no Qwen semantics.
    let mut controls = body.clone();
    let controls_map = controls.as_object_mut().unwrap();
    controls_map.remove("instructions");
    controls_map.remove("reasoning");
    controls_map.insert("input".into(), Value::String("K2 chat controls".into()));
    let mut request = super::parse_request(&controls)?;
    if request.k2_add_special_tokens == Some(false) {
        return Err(invalid(
            "x_k2.add_special_tokens",
            "K2 chat requires native automatic BOS",
        ));
    }
    request.k2_raw_input = None;
    request.instructions = instructions.map(str::to_owned);
    request.reasoning = Some(json!({"effort": effort}));
    request.k2_chat = Some(ChatInput { messages, effort });
    Ok(request)
}

pub(crate) fn render_with_profile(
    request: &ServeRequest,
    profile: Option<&ChatCapability>,
) -> Result<String, ServeError> {
    if request.k2_tools.is_some() {
        return super::tools::render_with_profile(request, profile);
    }
    let Some(chat) = &request.k2_chat else {
        return super::render(request);
    };
    profile.ok_or_else(|| invalid("input", "K2 chat requires a verified checkpoint profile"))?;
    if request.k2_raw_input.is_some()
        || request.k2_add_special_tokens != Some(true)
        || request.model_request.has_tool_surface()
        || !request.allowed_tools.is_empty()
        || request.no_thinking
        || request.strip_history_thinking
    {
        return Err(invalid("input", "invalid K2 no-tools chat request"));
    }
    k2::render(&chat.messages, chat.effort).map_err(|e| invalid("input", e.to_string()))
}

pub(crate) fn normalize_with_profile(
    request: &mut ServeRequest,
    default_max: usize,
    capacity: usize,
    profile: Option<&ChatCapability>,
) -> Result<(), ServeError> {
    render_with_profile(request, profile)?;
    super::normalize_controls(request, default_max, capacity)
}

#[cfg(test)]
pub(super) mod tests;
