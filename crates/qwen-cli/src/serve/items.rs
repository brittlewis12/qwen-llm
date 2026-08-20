//! Open Responses request parsing and item-sequence validation.
//!
//! Policy (docs/SERVE.md, k3 review defects 1-3):
//! - `input` is an item *list*, not a turn grammar: consecutive user
//!   messages are legal; `system`/`developer` only at the head; a
//!   `reasoning` item must immediately precede its assistant message; the
//!   final item must be a user message (S1).
//! - Unknown item *types* are rejected. Unknown *fields* are ignored:
//!   top-level request fields log once per name; `id`/`status` on replayed
//!   input items are accepted and ignored silently.
//! - `store:true`, `previous_response_id`, `truncation:"auto"`, `tools`,
//!   and `tool_choice` fail closed with spec error envelopes.

use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};

/// Spec error envelope. Serialized as `{"error": {...}}`; streamed inside
/// `response.failed` by the HTTP layer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ServeError {
    pub(crate) status: u16,
    pub(crate) error_type: &'static str,
    pub(crate) code: Option<&'static str>,
    pub(crate) param: Option<String>,
    pub(crate) message: String,
}

impl ServeError {
    pub(crate) fn invalid_request(param: Option<&str>, message: impl Into<String>) -> Self {
        Self {
            status: 400,
            error_type: "invalid_request",
            code: None,
            param: param.map(str::to_owned),
            message: message.into(),
        }
    }

    pub(crate) fn previous_response_not_found() -> Self {
        Self {
            status: 400,
            error_type: "invalid_request",
            code: Some("previous_response_not_found"),
            param: Some("previous_response_id".into()),
            message: "stateless serve does not store responses; resend full input".into(),
        }
    }

    pub(crate) fn model_not_found(requested: &str, loaded: &str) -> Self {
        Self {
            status: 404,
            error_type: "not_found",
            code: Some("model_not_found"),
            param: Some("model".into()),
            message: format!("model {requested:?} is not loaded; this server serves {loaded:?}"),
        }
    }

    pub(crate) fn to_json(&self) -> Value {
        serde_json::json!({
            "error": {
                "type": self.error_type,
                "code": self.code,
                "param": self.param,
                "message": self.message,
            }
        })
    }
}

/// One validated conversation turn, post item-sequence validation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Turn {
    User(String),
    Assistant {
        reasoning: Option<String>,
        visible: String,
        /// Tool calls emitted in this assistant turn, in wire order
        /// (provider_capture_v1: `function_call` items follow the
        /// assistant message / reasoning within one logical turn).
        calls: Vec<ToolCall>,
    },
    /// One or more tool results; consecutive results coalesce into a
    /// single user block per the template oracle.
    ToolResults(Vec<String>),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolCall {
    pub(crate) call_id: String,
    pub(crate) name: String,
    /// Raw `arguments` JSON string exactly as the provider replayed it.
    pub(crate) arguments: String,
}

/// One declared function tool (`tools[]` entry).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolDefinition {
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) parameters: Value,
}

/// Which family template renders this request. Resolved from the loaded
/// GGUF identity, mirroring the CLI's dispatch — serve must not render a
/// Qwen3.8 model with the generic ChatML contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum QwenTemplate {
    #[default]
    Generic,
    Qwen38,
}

/// Validated transcript plus generation controls, ready for rendering.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct ServeRequest {
    pub(crate) model: String,
    pub(crate) system: Option<String>,
    pub(crate) turns: Vec<Turn>,
    pub(crate) tools: Vec<ToolDefinition>,
    /// Executable subset (spec `tool_choice.allowed_tools`). Empty means
    /// every declared tool is executable. Enforced as a hard constraint
    /// on emitted calls; the rendered `tools` block is unchanged, which is
    /// the point — narrowing must not invalidate prompt prefixes.
    pub(crate) allowed_tools: Vec<String>,
    pub(crate) stream: bool,
    pub(crate) max_output_tokens: Option<usize>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) seed: Option<u64>,
    pub(crate) top_k: Option<usize>,
    pub(crate) min_p: Option<f32>,
    /// Spec `reasoning.effort`, passed through verbatim (the AI SDK
    /// provider forwards arbitrary strings). Families that support tiers
    /// map it; others reject or ignore per their contract.
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) no_thinking: bool,
    /// Rendering family, resolved from the loaded model at startup rather
    /// than per request.
    pub(crate) template: QwenTemplate,
    pub(crate) strip_history_thinking: bool,
    pub(crate) echo_stats: bool,
}

const KNOWN_TOP_LEVEL: &[&str] = &[
    "model",
    "input",
    "instructions",
    "max_output_tokens",
    "temperature",
    "top_p",
    "stream",
    "store",
    "previous_response_id",
    "truncation",
    "reasoning",
    "tools",
    "tool_choice",
    "x_qwen",
];

fn log_ignored_field(scope: &str, name: &str) {
    static LOGGED: OnceLock<Mutex<BTreeSet<String>>> = OnceLock::new();
    let key = format!("{scope}.{name}");
    let mut logged = LOGGED
        .get_or_init(|| Mutex::new(BTreeSet::new()))
        .lock()
        .expect("ignored-field log lock");
    if logged.insert(key) {
        tracing::info!(target: "qwen_diag", "serve: ignoring unknown {scope} field {name:?}");
    }
}

fn non_null<'a>(map: &'a serde_json::Map<String, Value>, key: &str) -> Option<&'a Value> {
    map.get(key).filter(|value| !value.is_null())
}

pub(crate) fn parse_request(body: &Value) -> Result<ServeRequest, ServeError> {
    let map = body
        .as_object()
        .ok_or_else(|| ServeError::invalid_request(None, "request body must be a JSON object"))?;

    for key in map.keys() {
        if !KNOWN_TOP_LEVEL.contains(&key.as_str()) {
            log_ignored_field("request", key);
        }
    }
    if non_null(map, "previous_response_id").is_some() {
        return Err(ServeError::previous_response_not_found());
    }
    if non_null(map, "store").and_then(Value::as_bool) == Some(true) {
        return Err(ServeError::invalid_request(
            Some("store"),
            "stateless serve supports store:false only",
        ));
    }
    if let Some(truncation) = non_null(map, "truncation")
        && truncation.as_str() != Some("disabled")
    {
        return Err(ServeError::invalid_request(
            Some("truncation"),
            "only truncation:\"disabled\" is supported; context overflow fails closed",
        ));
    }
    // The stock @ai-sdk/open-responses provider sends tool_choice:"auto"
    // unconditionally, including plain chat (provider_capture_v1, every
    // request). `allowed_tools` narrows the executable subset without
    // touching the rendered tools block (cache-preserving per spec).
    // `required`, `none`, and forced-function remain unimplemented.
    let allowed_tools = parse_tool_choice(non_null(map, "tool_choice"))?;
    let reasoning_effort = match non_null(map, "reasoning") {
        None => None,
        Some(value) => {
            let reasoning = value.as_object().ok_or_else(|| {
                ServeError::invalid_request(Some("reasoning"), "reasoning must be an object")
            })?;
            reasoning
                .get("effort")
                .filter(|effort| !effort.is_null())
                .map(|effort| {
                    effort.as_str().map(str::to_owned).ok_or_else(|| {
                        ServeError::invalid_request(
                            Some("reasoning.effort"),
                            "reasoning.effort must be a string",
                        )
                    })
                })
                .transpose()?
        }
    };
    let tools = parse_tool_definitions(non_null(map, "tools"))?;
    for name in &allowed_tools {
        if !tools.iter().any(|tool| &tool.name == name) {
            return Err(ServeError::invalid_request(
                Some("tool_choice"),
                format!("allowed_tools names {name:?}, which is not a declared tool"),
            ));
        }
    }

    let model = non_null(map, "model")
        .and_then(Value::as_str)
        .ok_or_else(|| ServeError::invalid_request(Some("model"), "model is required"))?
        .to_owned();

    let mut request = ServeRequest {
        model,
        tools,
        allowed_tools,
        reasoning_effort,
        stream: non_null(map, "stream")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        max_output_tokens: non_null(map, "max_output_tokens")
            .and_then(Value::as_u64)
            .map(|value| value as usize),
        temperature: non_null(map, "temperature")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        top_p: non_null(map, "top_p")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        ..ServeRequest::default()
    };

    if let Some(x_qwen) = non_null(map, "x_qwen") {
        let x_map = x_qwen.as_object().ok_or_else(|| {
            ServeError::invalid_request(Some("x_qwen"), "x_qwen must be an object")
        })?;
        for (key, value) in x_map {
            match key.as_str() {
                "seed" => request.seed = value.as_u64(),
                "top_k" => request.top_k = value.as_u64().map(|v| v as usize),
                "min_p" => request.min_p = value.as_f64().map(|v| v as f32),
                "no_thinking" => request.no_thinking = value.as_bool().unwrap_or(false),
                "history_thinking" => match value.as_str() {
                    Some("preserve") | None => {}
                    Some("strip") => request.strip_history_thinking = true,
                    Some(other) => {
                        return Err(ServeError::invalid_request(
                            Some("x_qwen.history_thinking"),
                            format!("unknown history_thinking {other:?}; use preserve or strip"),
                        ));
                    }
                },
                "stats" => request.echo_stats = value.as_bool().unwrap_or(false),
                other => log_ignored_field("x_qwen", other),
            }
        }
    }

    let instructions = non_null(map, "instructions")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let input = non_null(map, "input")
        .ok_or_else(|| ServeError::invalid_request(Some("input"), "input is required"))?;
    validate_input(input, instructions, &mut request)?;
    Ok(request)
}

fn validate_input(
    input: &Value,
    instructions: Option<String>,
    request: &mut ServeRequest,
) -> Result<(), ServeError> {
    request.system = instructions;
    match input {
        Value::String(text) => {
            if text.is_empty() {
                return Err(ServeError::invalid_request(Some("input"), "input is empty"));
            }
            request.turns.push(Turn::User(text.clone()));
            Ok(())
        }
        Value::Array(items) => validate_items(items, request),
        _ => Err(ServeError::invalid_request(
            Some("input"),
            "input must be a string or an item array",
        )),
    }
}

fn validate_items(items: &[Value], request: &mut ServeRequest) -> Result<(), ServeError> {
    if items.is_empty() {
        return Err(ServeError::invalid_request(Some("input"), "input is empty"));
    }
    let mut pending_reasoning: Option<String> = None;
    let mut head = true;
    for (index, item) in items.iter().enumerate() {
        let item_map = item.as_object().ok_or_else(|| {
            ServeError::invalid_request(Some("input"), format!("item {index} must be an object"))
        })?;
        // Replayed `id`/`status` accepted and ignored (review defect 3).
        let item_type = item_map
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message"); // EasyInput message shape carries no type
        match item_type {
            "message" => {
                let role = item_map
                    .get("role")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        ServeError::invalid_request(
                            Some("input"),
                            format!("item {index}: message requires a role"),
                        )
                    })?;
                let text = content_text(item_map.get("content"), role, index)?;
                match role {
                    "system" | "developer" => {
                        if !head {
                            return Err(ServeError::invalid_request(
                                Some("input"),
                                format!(
                                    "item {index}: {role} messages are only allowed at the head"
                                ),
                            ));
                        }
                        if request.system.is_some() {
                            return Err(ServeError::invalid_request(
                                Some("input"),
                                format!(
                                    "item {index}: instructions and a {role} message are mutually exclusive, and only one head system message is allowed"
                                ),
                            ));
                        }
                        request.system = Some(text);
                    }
                    "user" => {
                        if pending_reasoning.is_some() {
                            return Err(ServeError::invalid_request(
                                Some("input"),
                                format!(
                                    "item {index}: reasoning item must immediately precede its assistant message"
                                ),
                            ));
                        }
                        head = false;
                        request.turns.push(Turn::User(text));
                    }
                    "assistant" => {
                        if text.contains("<think>") {
                            return Err(ServeError::invalid_request(
                                Some("input"),
                                format!(
                                    "item {index}: assistant content must not embed <think>; reasoning travels as reasoning items"
                                ),
                            ));
                        }
                        if request.no_thinking && pending_reasoning.is_some() {
                            return Err(ServeError::invalid_request(
                                Some("input"),
                                format!(
                                    "item {index}: no_thinking sessions do not carry reasoning history"
                                ),
                            ));
                        }
                        head = false;
                        request.turns.push(Turn::Assistant {
                            reasoning: pending_reasoning.take(),
                            visible: text,
                            calls: Vec::new(),
                        });
                    }
                    other => {
                        return Err(ServeError::invalid_request(
                            Some("input"),
                            format!("item {index}: unsupported role {other:?}"),
                        ));
                    }
                }
            }
            "reasoning" => {
                if pending_reasoning.is_some() {
                    return Err(ServeError::invalid_request(
                        Some("input"),
                        format!("item {index}: consecutive reasoning items are not supported"),
                    ));
                }
                head = false;
                pending_reasoning = Some(reasoning_text(item_map, index)?);
            }
            // provider_capture_v1 tool-loop: reasoning may precede a
            // function_call rather than an assistant message, and calls
            // may follow an assistant message inside one logical turn.
            "function_call" => {
                let call = ToolCall {
                    call_id: required_str(item_map, "call_id", index)?,
                    name: required_str(item_map, "name", index)?,
                    arguments: item_map
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}")
                        .to_owned(),
                };
                head = false;
                match request.turns.last_mut() {
                    // Attach to the assistant turn opened by this same
                    // logical turn (no intervening user/tool item).
                    Some(Turn::Assistant {
                        calls,
                        reasoning: existing,
                        ..
                    }) if pending_reasoning.is_none() || existing.is_none() => {
                        if let Some(reasoning) = pending_reasoning.take()
                            && existing.is_none()
                        {
                            *existing = Some(reasoning);
                        }
                        calls.push(call);
                    }
                    _ => request.turns.push(Turn::Assistant {
                        reasoning: pending_reasoning.take(),
                        visible: String::new(),
                        calls: vec![call],
                    }),
                }
            }
            "function_call_output" => {
                if pending_reasoning.is_some() {
                    return Err(ServeError::invalid_request(
                        Some("input"),
                        format!(
                            "item {index}: reasoning item must precede an assistant message or function call"
                        ),
                    ));
                }
                let output = match item_map.get("output") {
                    Some(Value::String(text)) => text.clone(),
                    Some(value @ (Value::Array(_) | Value::Object(_))) => {
                        serde_json::to_string(value).expect("serialize tool output")
                    }
                    _ => {
                        return Err(ServeError::invalid_request(
                            Some("input"),
                            format!("item {index}: function_call_output requires output"),
                        ));
                    }
                };
                head = false;
                match request.turns.last_mut() {
                    Some(Turn::ToolResults(results)) => results.push(output),
                    _ => request.turns.push(Turn::ToolResults(vec![output])),
                }
            }
            other => {
                return Err(ServeError::invalid_request(
                    Some("input"),
                    format!("item {index}: unsupported item type {other:?}"),
                ));
            }
        }
    }
    if pending_reasoning.is_some() {
        return Err(ServeError::invalid_request(
            Some("input"),
            "trailing reasoning item has no assistant message",
        ));
    }
    match request.turns.last() {
        // A tool loop resumes generation after tool results, so
        // function_call_output is a legal terminal item (capture F-S2.3).
        Some(Turn::User(_) | Turn::ToolResults(_)) => Ok(()),
        _ => Err(ServeError::invalid_request(
            Some("input"),
            "the final input item must be a user message or function_call_output",
        )),
    }
}

fn required_str(
    map: &serde_json::Map<String, Value>,
    key: &str,
    index: usize,
) -> Result<String, ServeError> {
    map.get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            ServeError::invalid_request(Some("input"), format!("item {index}: {key} is required"))
        })
}

/// `tool_choice`: `"auto"` (default) or an `allowed_tools` object.
/// Returns the allowed subset (empty when unrestricted).
fn parse_tool_choice(tool_choice: Option<&Value>) -> Result<Vec<String>, ServeError> {
    let Some(tool_choice) = tool_choice else {
        return Ok(Vec::new());
    };
    if tool_choice.as_str() == Some("auto") {
        return Ok(Vec::new());
    }
    let map = tool_choice.as_object().ok_or_else(|| {
        ServeError::invalid_request(
            Some("tool_choice"),
            "only tool_choice:\"auto\" or an allowed_tools object is supported",
        )
    })?;
    if map.get("type").and_then(Value::as_str) != Some("allowed_tools") {
        return Err(ServeError::invalid_request(
            Some("tool_choice"),
            "only tool_choice:\"auto\" or an allowed_tools object is supported",
        ));
    }
    let entries = map.get("tools").and_then(Value::as_array).ok_or_else(|| {
        ServeError::invalid_request(Some("tool_choice"), "allowed_tools requires a tools array")
    })?;
    let mut names = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let map = entry.as_object().ok_or_else(|| {
            ServeError::invalid_request(
                Some("tool_choice"),
                format!("allowed_tools entry {index} must be an object"),
            )
        })?;
        names.push(required_str(map, "name", index)?);
    }
    if names.is_empty() {
        return Err(ServeError::invalid_request(
            Some("tool_choice"),
            "allowed_tools must name at least one tool",
        ));
    }
    Ok(names)
}

fn parse_tool_definitions(tools: Option<&Value>) -> Result<Vec<ToolDefinition>, ServeError> {
    let Some(tools) = tools else {
        return Ok(Vec::new());
    };
    let entries = tools
        .as_array()
        .ok_or_else(|| ServeError::invalid_request(Some("tools"), "tools must be an array"))?;
    let mut definitions = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let map = entry.as_object().ok_or_else(|| {
            ServeError::invalid_request(Some("tools"), format!("tool {index} must be an object"))
        })?;
        match map.get("type").and_then(Value::as_str) {
            Some("function") | None => {}
            Some(other) => {
                return Err(ServeError::invalid_request(
                    Some("tools"),
                    format!("tool {index}: hosted tool type {other:?} is not supported"),
                ));
            }
        }
        definitions.push(ToolDefinition {
            name: required_str(map, "name", index)?,
            description: map
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_owned),
            parameters: map.get("parameters").cloned().unwrap_or(Value::Null),
        });
    }
    Ok(definitions)
}

fn content_text(content: Option<&Value>, role: &str, index: usize) -> Result<String, ServeError> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                let part_map = part.as_object().ok_or_else(|| {
                    ServeError::invalid_request(
                        Some("input"),
                        format!("item {index}: content parts must be objects"),
                    )
                })?;
                match part_map.get("type").and_then(Value::as_str) {
                    Some("input_text") | Some("output_text") => {
                        text.push_str(part_map.get("text").and_then(Value::as_str).ok_or_else(
                            || {
                                ServeError::invalid_request(
                                    Some("input"),
                                    format!("item {index}: text part requires text"),
                                )
                            },
                        )?);
                    }
                    Some(other) => {
                        return Err(ServeError::invalid_request(
                            Some("input"),
                            format!(
                                "item {index}: unsupported content part type {other:?} for role {role}"
                            ),
                        ));
                    }
                    None => {
                        return Err(ServeError::invalid_request(
                            Some("input"),
                            format!("item {index}: content part requires a type"),
                        ));
                    }
                }
            }
            Ok(text)
        }
        _ => Err(ServeError::invalid_request(
            Some("input"),
            format!("item {index}: message requires string or part-array content"),
        )),
    }
}

fn reasoning_text(
    item_map: &serde_json::Map<String, Value>,
    index: usize,
) -> Result<String, ServeError> {
    match item_map.get("content") {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                let fragment = part
                    .as_object()
                    .and_then(|map| map.get("text"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        ServeError::invalid_request(
                            Some("input"),
                            format!("item {index}: reasoning parts require text"),
                        )
                    })?;
                text.push_str(fragment);
            }
            Ok(text)
        }
        Some(Value::Null) | None => Err(ServeError::invalid_request(
            Some("input"),
            format!(
                "item {index}: reasoning items require plain content in S1 (encrypted_content lands in S3)"
            ),
        )),
        _ => Err(ServeError::invalid_request(
            Some("input"),
            format!("item {index}: reasoning content must be a string or part array"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(body: Value) -> Result<ServeRequest, ServeError> {
        parse_request(&body)
    }

    #[test]
    fn string_input_is_one_user_turn() {
        let request = parse(json!({"model": "m", "input": "hello"})).unwrap();
        assert_eq!(request.turns, vec![Turn::User("hello".into())]);
        assert!(request.system.is_none());
        assert!(!request.stream);
    }

    #[test]
    fn easyinput_messages_need_no_type_and_consecutive_users_are_legal() {
        let request = parse(json!({"model": "m", "input": [
            {"role": "user", "content": "first"},
            {"role": "user", "content": [{"type": "input_text", "text": "second"}]},
        ]}))
        .unwrap();
        assert_eq!(
            request.turns,
            vec![Turn::User("first".into()), Turn::User("second".into())]
        );
    }

    #[test]
    fn reasoning_binds_to_following_assistant_message() {
        let request = parse(json!({"model": "m", "input": [
            {"role": "user", "content": "q"},
            {"type": "reasoning", "content": "\nplan\n"},
            {"type": "message", "role": "assistant", "status": "completed", "id": "msg_1",
             "content": [{"type": "output_text", "text": "a"}]},
            {"role": "user", "content": "q2"},
        ]}))
        .unwrap();
        assert_eq!(
            request.turns[1],
            Turn::Assistant {
                reasoning: Some("\nplan\n".into()),
                visible: "a".into(),
                calls: Vec::new()
            }
        );
    }

    #[test]
    fn developer_role_is_system_equivalent_and_head_only() {
        let request = parse(json!({"model": "m", "input": [
            {"role": "developer", "content": "be terse"},
            {"role": "user", "content": "q"},
        ]}))
        .unwrap();
        assert_eq!(request.system.as_deref(), Some("be terse"));

        let error = parse(json!({"model": "m", "input": [
            {"role": "user", "content": "q"},
            {"role": "system", "content": "late"},
            {"role": "user", "content": "q2"},
        ]}))
        .unwrap_err();
        assert!(error.message.contains("only allowed at the head"));
    }

    #[test]
    fn sequence_violations_fail_closed() {
        let error = parse(json!({"model": "m", "input": [
            {"role": "user", "content": "q"},
            {"type": "reasoning", "content": "r"},
            {"role": "user", "content": "q2"},
        ]}))
        .unwrap_err();
        assert!(error.message.contains("immediately precede"));

        let error = parse(json!({"model": "m", "input": [
            {"role": "user", "content": "q"},
            {"type": "reasoning", "content": "r"},
        ]}))
        .unwrap_err();
        assert!(error.message.contains("trailing reasoning"));

        let error = parse(json!({"model": "m", "input": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]}))
        .unwrap_err();
        assert!(error.message.contains("final input item"));

        let error = parse(json!({"model": "m", "input": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>x</think>a"},
            {"role": "user", "content": "q2"},
        ]}))
        .unwrap_err();
        assert!(
            error
                .message
                .contains("reasoning travels as reasoning items")
        );
    }

    #[test]
    fn stateless_and_scope_fences_fail_closed() {
        let error = parse(json!({"model": "m", "input": "q", "store": true})).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("store"));

        let error = parse(json!({"model": "m", "input": "q", "previous_response_id": "resp_1"}))
            .unwrap_err();
        assert_eq!(error.code, Some("previous_response_not_found"));

        let error = parse(json!({"model": "m", "input": "q", "truncation": "auto"})).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("truncation"));

        let error = parse(json!({"model": "m", "input": "q",
            "tools": [{"type": "web_search"}]}))
        .unwrap_err();
        assert_eq!(error.param.as_deref(), Some("tools"));

        let error =
            parse(json!({"model": "m", "input": "q", "tool_choice": "required"})).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("tool_choice"));

        let error = parse(json!({"model": "m", "input": [
            {"type": "web_search_call", "id": "ws_1"},
        ]}))
        .unwrap_err();
        assert!(error.message.contains("unsupported item type"));
    }

    #[test]
    fn stock_provider_chat_request_shape_is_accepted() {
        // Byte-shape from provider_capture_v1 chat-replay#1: the stock
        // @ai-sdk/open-responses provider replays reasoning items verbatim
        // (id + summary + reasoning_text parts), assistant messages as
        // output_text parts with ids, sends instructions for system, and
        // tool_choice:"auto" unconditionally.
        let request = parse(json!({
            "model": "m",
            "tool_choice": "auto",
            "instructions": "You are terse.",
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "Add 2 and 3."}]},
                {"type": "reasoning", "summary": [], "id": "rs_1",
                 "content": [{"type": "reasoning_text", "text": "\nplan the answer\n"}]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "It is 5."}], "id": "msg_1"},
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "Now add 10."}]},
            ],
        }))
        .expect("stock provider chat replay must parse");
        assert_eq!(request.system.as_deref(), Some("You are terse."));
        assert_eq!(
            request.turns[1],
            Turn::Assistant {
                reasoning: Some("\nplan the answer\n".into()),
                visible: "It is 5.".into(),
                calls: Vec::new()
            }
        );
    }

    #[test]
    fn unknown_fields_are_ignored_and_extensions_parse() {
        let request = parse(json!({
            "model": "m",
            "input": "q",
            "metadata": {"k": "v"},
            "reasoning": {"effort": "high"},
            "store": false,
            "stream": true,
            "max_output_tokens": 512,
            "temperature": 0.7,
            "x_qwen": {"seed": 7, "top_k": 200, "min_p": 0.05,
                        "history_thinking": "strip", "stats": true,
                        "future_knob": 1},
        }))
        .unwrap();
        assert!(request.stream);
        assert_eq!(request.max_output_tokens, Some(512));
        assert_eq!(request.seed, Some(7));
        assert_eq!(request.top_k, Some(200));
        assert!(request.strip_history_thinking);
        assert!(request.echo_stats);
    }

    #[test]
    fn stock_provider_tool_loop_replay_parses_to_turns() {
        // Byte-shape from provider_capture_v1 tool-loop#1: reasoning binds
        // to a function_call (not an assistant message), and the input
        // legally terminates with function_call_output.
        let request = parse(json!({
            "model": "m",
            "tool_choice": "auto",
            "tools": [{
                "type": "function", "name": "fs_list",
                "description": "List directory entries",
                "parameters": {"type": "object",
                               "properties": {"path": {"type": "string"}},
                               "required": ["path"]},
            }],
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "List files in /tmp."}]},
                {"type": "reasoning", "summary": [], "id": "rs_t1",
                 "content": [{"type": "reasoning_text", "text": "\nneed the listing\n"}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_abc123",
                 "name": "fs_list", "arguments": "{\"path\":\"/tmp\"}"},
                {"type": "function_call_output", "call_id": "call_abc123",
                 "output": "{\"entries\":[\"a.txt\"],\"path\":\"/tmp\"}"},
            ],
        }))
        .expect("stock provider tool-loop replay must parse");
        assert_eq!(request.tools.len(), 1);
        assert_eq!(request.tools[0].name, "fs_list");
        assert_eq!(request.turns.len(), 3);
        match &request.turns[1] {
            Turn::Assistant {
                reasoning,
                visible,
                calls,
            } => {
                assert_eq!(reasoning.as_deref(), Some("\nneed the listing\n"));
                assert!(visible.is_empty());
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].call_id, "call_abc123");
                assert_eq!(calls[0].name, "fs_list");
            }
            other => panic!("expected assistant turn with call, got {other:?}"),
        }
        assert_eq!(
            request.turns[2],
            Turn::ToolResults(vec!["{\"entries\":[\"a.txt\"],\"path\":\"/tmp\"}".into()])
        );
    }

    #[test]
    fn allowed_tools_narrows_without_changing_the_rendered_tools_block() {
        let tools = json!([
            {"type": "function", "name": "fs_list", "parameters": {"type": "object"}},
            {"type": "function", "name": "fs_write", "parameters": {"type": "object"}},
        ]);
        let unrestricted = parse(json!({
            "model": "m", "tools": tools, "tool_choice": "auto", "input": "go",
        }))
        .expect("auto parses");
        let narrowed = parse(json!({
            "model": "m", "tools": tools,
            "tool_choice": {"type": "allowed_tools", "tools": [{"name": "fs_list"}]},
            "input": "go",
        }))
        .expect("allowed_tools parses");
        assert!(unrestricted.allowed_tools.is_empty());
        assert_eq!(narrowed.allowed_tools, vec!["fs_list".to_string()]);
        // The cache-preserving property: narrowing must not perturb the
        // rendered prompt, so prior prefixes (and their checkpoints) stay
        // valid across a narrowing change.
        assert_eq!(
            crate::serve::render::render_qwen_serve_prompt(&unrestricted),
            crate::serve::render::render_qwen_serve_prompt(&narrowed),
            "allowed_tools must not change rendered bytes"
        );
    }

    #[test]
    fn allowed_tools_validation_fails_closed() {
        let tools = json!([{"type": "function", "name": "fs_list",
                            "parameters": {"type": "object"}}]);
        for (choice, fragment) in [
            (
                json!({"type": "allowed_tools", "tools": [{"name": "nope"}]}),
                "not a declared tool",
            ),
            (
                json!({"type": "allowed_tools", "tools": []}),
                "at least one tool",
            ),
            (
                json!({"type": "function", "name": "fs_list"}),
                "allowed_tools object",
            ),
            (json!("required"), "allowed_tools object"),
            (json!("none"), "allowed_tools object"),
        ] {
            let error = parse(json!({"model": "m", "tools": tools,
                                      "tool_choice": choice, "input": "go"}))
            .unwrap_err();
            assert_eq!(error.param.as_deref(), Some("tool_choice"));
            assert!(
                error.message.contains(fragment),
                "unexpected message: {}",
                error.message
            );
        }
    }

    #[test]
    fn consecutive_tool_outputs_coalesce_and_calls_attach_to_assistant_text() {
        let request = parse(json!({"model": "m", "input": [
            {"role": "user", "content": "go"},
            {"type": "message", "role": "assistant", "content": "Checking both."},
            {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"},
            {"type": "function_call", "call_id": "c2", "name": "b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "r1"},
            {"type": "function_call_output", "call_id": "c2", "output": "r2"},
        ]}))
        .expect("parallel calls and coalesced outputs must parse");
        assert_eq!(request.turns.len(), 3);
        match &request.turns[1] {
            Turn::Assistant { visible, calls, .. } => {
                assert_eq!(visible, "Checking both.");
                assert_eq!(calls.len(), 2);
            }
            other => panic!("expected assistant turn, got {other:?}"),
        }
        assert_eq!(
            request.turns[2],
            Turn::ToolResults(vec!["r1".into(), "r2".into()])
        );
    }

    #[test]
    fn no_thinking_rejects_reasoning_history() {
        let error = parse(json!({"model": "m", "input": [
            {"role": "user", "content": "q"},
            {"type": "reasoning", "content": "r"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q2"},
        ], "x_qwen": {"no_thinking": true}}))
        .unwrap_err();
        assert!(error.message.contains("no_thinking sessions"));
    }
}
