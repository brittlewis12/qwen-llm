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

use serde_json::{Value, json};
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
                "code": self.code.unwrap_or(""),
                "param": self.param.as_deref().unwrap_or(""),
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
    ToolResults(Vec<ToolResult>),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolCall {
    pub(crate) call_id: String,
    pub(crate) name: String,
    /// Raw `arguments` JSON string exactly as the provider replayed it.
    pub(crate) arguments: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolResult {
    pub(crate) call_id: String,
    pub(crate) name: String,
    pub(crate) output: String,
}

/// One declared function tool (`tools[]` entry).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolDefinition {
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) parameters: Value,
    pub(crate) strict: Option<bool>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SystemSource {
    Instructions,
    System,
    Developer,
}

impl SystemSource {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Instructions => "instructions",
            Self::System => "system",
            Self::Developer => "developer",
        }
    }
}

/// Validated transcript plus generation controls, ready for rendering.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ServeRequest {
    pub(crate) model: String,
    pub(crate) instructions: Option<String>,
    pub(crate) system: Option<String>,
    pub(crate) system_source: Option<SystemSource>,
    pub(crate) turns: Vec<Turn>,
    pub(crate) tools: Vec<ToolDefinition>,
    /// Exact executable set. Empty means no calls are executable; narrowing
    /// does not alter the rendered tools block or invalidate prompt prefixes.
    pub(crate) allowed_tools: Vec<String>,
    pub(crate) tool_choice: Value,
    pub(crate) reasoning: Option<Value>,
    pub(crate) parallel_tool_calls: bool,
    pub(crate) stream: bool,
    pub(crate) max_output_tokens: Option<usize>,
    pub(crate) temperature: Option<f32>,
    pub(crate) temperature_echo: Option<f64>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_p_echo: Option<f64>,
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

impl Default for ServeRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            instructions: None,
            system: None,
            system_source: None,
            turns: Vec::new(),
            tools: Vec::new(),
            allowed_tools: Vec::new(),
            tool_choice: Value::String("auto".into()),
            reasoning: None,
            parallel_tool_calls: true,
            stream: false,
            max_output_tokens: None,
            temperature: None,
            temperature_echo: None,
            top_p: None,
            top_p_echo: None,
            seed: None,
            top_k: None,
            min_p: None,
            reasoning_effort: None,
            no_thinking: false,
            template: QwenTemplate::default(),
            strip_history_thinking: false,
            echo_stats: false,
        }
    }
}

const KNOWN_TOP_LEVEL: &[&str] = &[
    "background",
    "conversation",
    "frequency_penalty",
    "include",
    "model",
    "input",
    "instructions",
    "max_tool_calls",
    "max_output_tokens",
    "metadata",
    "modalities",
    "temperature",
    "top_p",
    "top_logprobs",
    "presence_penalty",
    "prompt",
    "prompt_cache_key",
    "response_format",
    "safety_identifier",
    "service_tier",
    "stream",
    "stream_options",
    "store",
    "previous_response_id",
    "text",
    "truncation",
    "reasoning",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "user",
    "x_qwen",
];

const UNSUPPORTED_TOP_LEVEL: &[&str] = &[
    "background",
    "conversation",
    "frequency_penalty",
    "include",
    "max_tool_calls",
    "metadata",
    "modalities",
    "presence_penalty",
    "prompt",
    "prompt_cache_key",
    "response_format",
    "safety_identifier",
    "service_tier",
    "stream_options",
    "text",
    "top_logprobs",
    "user",
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

fn bool_field(map: &serde_json::Map<String, Value>, key: &str) -> Result<Option<bool>, ServeError> {
    non_null(map, key)
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                ServeError::invalid_request(Some(key), format!("{key} must be a boolean"))
            })
        })
        .transpose()
}

fn usize_field(
    map: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<usize>, ServeError> {
    non_null(map, key)
        .map(|value| {
            let value = value.as_u64().ok_or_else(|| {
                ServeError::invalid_request(
                    Some(key),
                    format!("{key} must be a non-negative integer"),
                )
            })?;
            usize::try_from(value)
                .map_err(|_| ServeError::invalid_request(Some(key), format!("{key} exceeds usize")))
        })
        .transpose()
}

fn f32_field(map: &serde_json::Map<String, Value>, key: &str) -> Result<Option<f32>, ServeError> {
    non_null(map, key)
        .map(|value| {
            let value = value.as_f64().ok_or_else(|| {
                ServeError::invalid_request(Some(key), format!("{key} must be a number"))
            })?;
            if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
                Err(ServeError::invalid_request(
                    Some(key),
                    format!("{key} is outside the supported numeric range"),
                ))
            } else {
                let narrowed = value as f32;
                if value != 0.0 && narrowed == 0.0 {
                    Err(ServeError::invalid_request(
                        Some(key),
                        format!("{key} underflows the supported numeric range"),
                    ))
                } else {
                    Ok(narrowed)
                }
            }
        })
        .transpose()
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
    for &key in UNSUPPORTED_TOP_LEVEL {
        if non_null(map, key).is_some() {
            return Err(ServeError::invalid_request(
                Some(key),
                format!("{key} is not supported by this Open Responses subset"),
            ));
        }
    }
    if let Some(value) = non_null(map, "previous_response_id") {
        if !value.is_string() {
            return Err(ServeError::invalid_request(
                Some("previous_response_id"),
                "previous_response_id must be a string",
            ));
        }
        return Err(ServeError::previous_response_not_found());
    }
    if bool_field(map, "store")? == Some(true) {
        return Err(ServeError::invalid_request(
            Some("store"),
            "stateless serve supports store:false only",
        ));
    }
    if let Some(truncation) = non_null(map, "truncation") {
        let truncation = truncation.as_str().ok_or_else(|| {
            ServeError::invalid_request(Some("truncation"), "truncation must be a string")
        })?;
        if truncation != "disabled" {
            return Err(ServeError::invalid_request(
                Some("truncation"),
                "only truncation:\"disabled\" is supported; context overflow fails closed",
            ));
        }
    }
    // The stock @ai-sdk/open-responses provider sends tool_choice:"auto"
    // unconditionally, including plain chat (provider_capture_v1, every
    // request). `allowed_tools` narrows the executable subset without
    // touching the rendered tools block (cache-preserving per spec).
    // `required`, `none`, and forced-function remain unimplemented.
    let choice = parse_tool_choice(non_null(map, "tool_choice"))?;
    let (reasoning_effort, reasoning) = match non_null(map, "reasoning") {
        None => (None, None),
        Some(value) => {
            let reasoning = value.as_object().ok_or_else(|| {
                ServeError::invalid_request(Some("reasoning"), "reasoning must be an object")
            })?;
            if non_null(reasoning, "summary").is_some() {
                return Err(ServeError::invalid_request(
                    Some("reasoning.summary"),
                    "reasoning.summary is not supported",
                ));
            }
            for key in reasoning.keys() {
                if key != "effort" && key != "summary" {
                    log_ignored_field("reasoning", key);
                }
            }
            let effort = non_null(reasoning, "effort")
                .map(|effort| {
                    effort.as_str().map(str::to_owned).ok_or_else(|| {
                        ServeError::invalid_request(
                            Some("reasoning.effort"),
                            "reasoning.effort must be a string",
                        )
                    })
                })
                .transpose()?;
            let echo = effort
                .as_ref()
                .map(|effort| json!({"effort": effort}))
                .or_else(|| Some(json!({})));
            (effort, echo)
        }
    };
    let tools = parse_tool_definitions(non_null(map, "tools"))?;
    let allowed_tools = choice
        .allowed
        .clone()
        .unwrap_or_else(|| tools.iter().map(|tool| tool.name.clone()).collect());
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
        tool_choice: choice.echo,
        reasoning,
        reasoning_effort,
        parallel_tool_calls: bool_field(map, "parallel_tool_calls")?.unwrap_or(true),
        stream: bool_field(map, "stream")?.unwrap_or(false),
        max_output_tokens: usize_field(map, "max_output_tokens")?,
        temperature: f32_field(map, "temperature")?,
        temperature_echo: map.get("temperature").and_then(Value::as_f64),
        top_p: f32_field(map, "top_p")?,
        top_p_echo: map.get("top_p").and_then(Value::as_f64),
        ..ServeRequest::default()
    };

    if let Some(x_qwen) = non_null(map, "x_qwen") {
        let x_map = x_qwen.as_object().ok_or_else(|| {
            ServeError::invalid_request(Some("x_qwen"), "x_qwen must be an object")
        })?;
        for (key, value) in x_map {
            if value.is_null() {
                continue;
            }
            match key.as_str() {
                "seed" => {
                    request.seed = Some(value.as_u64().ok_or_else(|| {
                        ServeError::invalid_request(
                            Some("x_qwen.seed"),
                            "x_qwen.seed must be a non-negative integer",
                        )
                    })?)
                }
                "top_k" => {
                    let raw = value.as_u64().ok_or_else(|| {
                        ServeError::invalid_request(
                            Some("x_qwen.top_k"),
                            "x_qwen.top_k must be a non-negative integer",
                        )
                    })?;
                    request.top_k = Some(usize::try_from(raw).map_err(|_| {
                        ServeError::invalid_request(
                            Some("x_qwen.top_k"),
                            "x_qwen.top_k exceeds usize",
                        )
                    })?);
                }
                "min_p" => {
                    let raw = value.as_f64().ok_or_else(|| {
                        ServeError::invalid_request(
                            Some("x_qwen.min_p"),
                            "x_qwen.min_p must be a number",
                        )
                    })?;
                    if !raw.is_finite() || raw < f32::MIN as f64 || raw > f32::MAX as f64 {
                        return Err(ServeError::invalid_request(
                            Some("x_qwen.min_p"),
                            "x_qwen.min_p is outside the supported numeric range",
                        ));
                    }
                    if !(0.0..=1.0).contains(&raw) {
                        return Err(ServeError::invalid_request(
                            Some("x_qwen.min_p"),
                            "x_qwen.min_p must be in [0, 1]",
                        ));
                    }
                    let narrowed = raw as f32;
                    if raw != 0.0 && narrowed == 0.0 {
                        return Err(ServeError::invalid_request(
                            Some("x_qwen.min_p"),
                            "x_qwen.min_p underflows the supported numeric range",
                        ));
                    }
                    request.min_p = Some(narrowed);
                }
                "no_thinking" => {
                    request.no_thinking = value.as_bool().ok_or_else(|| {
                        ServeError::invalid_request(
                            Some("x_qwen.no_thinking"),
                            "x_qwen.no_thinking must be a boolean",
                        )
                    })?
                }
                "history_thinking" => match value.as_str() {
                    Some("preserve") => {}
                    Some("strip") => request.strip_history_thinking = true,
                    Some(other) => {
                        return Err(ServeError::invalid_request(
                            Some("x_qwen.history_thinking"),
                            format!("unknown history_thinking {other:?}; use preserve or strip"),
                        ));
                    }
                    None => {
                        return Err(ServeError::invalid_request(
                            Some("x_qwen.history_thinking"),
                            "x_qwen.history_thinking must be a string",
                        ));
                    }
                },
                "stats" => {
                    request.echo_stats = value.as_bool().ok_or_else(|| {
                        ServeError::invalid_request(
                            Some("x_qwen.stats"),
                            "x_qwen.stats must be a boolean",
                        )
                    })?
                }
                other => log_ignored_field("x_qwen", other),
            }
        }
    }

    validate_generation_controls(&request)?;

    let instructions = non_null(map, "instructions")
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                ServeError::invalid_request(Some("instructions"), "instructions must be a string")
            })
        })
        .transpose()?;
    request.instructions = instructions.clone();
    let input = non_null(map, "input")
        .ok_or_else(|| ServeError::invalid_request(Some("input"), "input is required"))?;
    validate_input(input, instructions, &mut request)?;
    Ok(request)
}

fn validate_generation_controls(request: &ServeRequest) -> Result<(), ServeError> {
    if request.max_output_tokens == Some(0) {
        return Err(ServeError::invalid_request(
            Some("max_output_tokens"),
            "max_output_tokens must be >= 1",
        ));
    }
    if request.temperature_echo.is_some_and(|value| value < 0.0) {
        return Err(ServeError::invalid_request(
            Some("temperature"),
            "temperature must be >= 0",
        ));
    }
    if request
        .top_p_echo
        .is_some_and(|value| !(0.0 < value && value <= 1.0))
    {
        return Err(ServeError::invalid_request(
            Some("top_p"),
            "top_p must be in (0, 1]",
        ));
    }
    if request
        .min_p
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err(ServeError::invalid_request(
            Some("x_qwen.min_p"),
            "x_qwen.min_p must be in [0, 1]",
        ));
    }
    Ok(())
}

fn validate_input(
    input: &Value,
    instructions: Option<String>,
    request: &mut ServeRequest,
) -> Result<(), ServeError> {
    request.system = instructions;
    request.system_source = request.system.as_ref().map(|_| SystemSource::Instructions);
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
    let mut pending_calls = Vec::<(String, String)>::new();
    let mut pending_outputs = std::collections::BTreeMap::new();
    let mut seen_calls = BTreeSet::new();
    let mut seen_outputs = BTreeSet::new();
    let mut outputs_started = false;
    for (index, item) in items.iter().enumerate() {
        let item_map = item.as_object().ok_or_else(|| {
            ServeError::invalid_request(Some("input"), format!("item {index} must be an object"))
        })?;
        // Replayed `id`/`status` accepted and ignored (review defect 3).
        let item_type = match item_map.get("type") {
            Some(value) => value.as_str().ok_or_else(|| {
                ServeError::invalid_request(
                    Some("input"),
                    format!("item {index}: type must be a string"),
                )
            })?,
            None => "message",
        };
        if !pending_calls.is_empty()
            && !matches!(item_type, "function_call" | "function_call_output")
        {
            return Err(ServeError::invalid_request(
                Some("input"),
                format!(
                    "item {index}: preceding function calls require all outputs before the conversation continues"
                ),
            ));
        }
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
                        request.system_source = Some(if role == "developer" {
                            SystemSource::Developer
                        } else {
                            SystemSource::System
                        });
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
                if pending_calls.is_empty() {
                    outputs_started = false;
                }
                if outputs_started {
                    return Err(ServeError::invalid_request(
                        Some("input"),
                        format!(
                            "item {index}: function calls cannot follow outputs from the same parallel batch"
                        ),
                    ));
                }
                let call_id = required_str(item_map, "call_id", index)?;
                validate_call_id(&call_id, index)?;
                if !seen_calls.insert(call_id.clone()) {
                    return Err(ServeError::invalid_request(
                        Some("input"),
                        format!("item {index}: duplicate function call_id {call_id:?}"),
                    ));
                }
                let arguments = required_str(item_map, "arguments", index)?;
                if !matches!(
                    serde_json::from_str::<Value>(&arguments),
                    Ok(Value::Object(_))
                ) {
                    return Err(ServeError::invalid_request(
                        Some("input"),
                        format!(
                            "item {index}: function_call arguments must be a JSON object encoded as a string"
                        ),
                    ));
                }
                let name = required_str(item_map, "name", index)?;
                validate_tool_name(&name, "input", &format!("item {index}: function name"))?;
                let call = ToolCall {
                    call_id: call_id.clone(),
                    name,
                    arguments,
                };
                pending_calls.push((call_id, call.name.clone()));
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
                let call_id = required_str(item_map, "call_id", index)?;
                validate_call_id(&call_id, index)?;
                if seen_outputs.contains(&call_id) {
                    return Err(ServeError::invalid_request(
                        Some("input"),
                        format!("item {index}: duplicate output for call_id {call_id:?}"),
                    ));
                }
                if pending_calls.is_empty() {
                    return Err(ServeError::invalid_request(
                        Some("input"),
                        format!("item {index}: output references unknown call_id {call_id:?}"),
                    ));
                }
                if !pending_calls
                    .iter()
                    .any(|(pending_call_id, _)| pending_call_id == &call_id)
                {
                    return Err(ServeError::invalid_request(
                        Some("input"),
                        format!("item {index}: output references unknown call_id {call_id:?}"),
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
                pending_outputs.insert(call_id.clone(), output);
                seen_outputs.insert(call_id);
                outputs_started = true;
                if pending_outputs.len() != pending_calls.len() {
                    continue;
                }
                let results = pending_calls
                    .drain(..)
                    .map(|(call_id, name)| ToolResult {
                        output: pending_outputs
                            .remove(&call_id)
                            .expect("known output for every pending call"),
                        call_id,
                        name,
                    })
                    .collect();
                head = false;
                request.turns.push(Turn::ToolResults(results));
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
    if let Some((call_id, _)) = pending_calls.first() {
        return Err(ServeError::invalid_request(
            Some("input"),
            format!("function call {call_id:?} has no function_call_output"),
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
    required_str_for(map, key, "input", &format!("item {index}"))
}

fn required_str_for(
    map: &serde_json::Map<String, Value>,
    key: &str,
    param: &str,
    context: &str,
) -> Result<String, ServeError> {
    map.get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            ServeError::invalid_request(Some(param), format!("{context}: {key} is required"))
        })
}

fn validate_tool_name(name: &str, param: &str, context: &str) -> Result<(), ServeError> {
    if name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(ServeError::invalid_request(
            Some(param),
            format!("{context} must match [A-Za-z0-9_-]{{1,64}}"),
        ));
    }
    Ok(())
}

fn validate_call_id(call_id: &str, index: usize) -> Result<(), ServeError> {
    if call_id.len() > 64 {
        return Err(ServeError::invalid_request(
            Some("input"),
            format!("item {index}: call_id must contain at most 64 bytes"),
        ));
    }
    Ok(())
}

struct ParsedToolChoice {
    allowed: Option<Vec<String>>,
    echo: Value,
}

fn parse_tool_choice(tool_choice: Option<&Value>) -> Result<ParsedToolChoice, ServeError> {
    let Some(tool_choice) = tool_choice else {
        return Ok(ParsedToolChoice {
            allowed: None,
            echo: Value::String("auto".into()),
        });
    };
    if tool_choice.as_str() == Some("auto") {
        return Ok(ParsedToolChoice {
            allowed: None,
            echo: tool_choice.clone(),
        });
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
    match non_null(map, "mode") {
        None => {}
        Some(Value::String(mode)) if mode == "auto" => {}
        Some(Value::String(mode)) if mode == "required" => {
            return Err(ServeError::invalid_request(
                Some("tool_choice"),
                "allowed_tools mode \"required\" is not supported",
            ));
        }
        Some(Value::String(mode)) => {
            return Err(ServeError::invalid_request(
                Some("tool_choice"),
                format!("unsupported allowed_tools mode {mode:?}"),
            ));
        }
        Some(_) => {
            return Err(ServeError::invalid_request(
                Some("tool_choice"),
                "allowed_tools mode must be a string",
            ));
        }
    }
    let entries = map.get("tools").and_then(Value::as_array).ok_or_else(|| {
        ServeError::invalid_request(Some("tool_choice"), "allowed_tools requires a tools array")
    })?;
    let mut names = Vec::with_capacity(entries.len());
    let mut unique_names = BTreeSet::new();
    for (index, entry) in entries.iter().enumerate() {
        let map = entry.as_object().ok_or_else(|| {
            ServeError::invalid_request(
                Some("tool_choice"),
                format!("allowed_tools entry {index} must be an object"),
            )
        })?;
        if map.get("type").and_then(Value::as_str) != Some("function") {
            return Err(ServeError::invalid_request(
                Some("tool_choice"),
                format!("allowed_tools entry {index} requires type:\"function\""),
            ));
        }
        let name = required_str_for(
            map,
            "name",
            "tool_choice",
            &format!("allowed_tools entry {index}"),
        )?;
        validate_tool_name(
            &name,
            "tool_choice",
            &format!("allowed_tools entry {index} name"),
        )?;
        if !unique_names.insert(name.clone()) {
            return Err(ServeError::invalid_request(
                Some("tool_choice"),
                format!("allowed_tools contains duplicate name {name:?}"),
            ));
        }
        names.push(name);
    }
    if names.is_empty() {
        return Err(ServeError::invalid_request(
            Some("tool_choice"),
            "allowed_tools must name at least one tool",
        ));
    }
    let echo_tools = names
        .iter()
        .map(|name| json!({"type": "function", "name": name}))
        .collect::<Vec<_>>();
    let echo = json!({
        "type": "allowed_tools",
        "mode": "auto",
        "tools": echo_tools,
    });
    Ok(ParsedToolChoice {
        allowed: Some(names),
        echo,
    })
}

fn parse_tool_definitions(tools: Option<&Value>) -> Result<Vec<ToolDefinition>, ServeError> {
    let Some(tools) = tools else {
        return Ok(Vec::new());
    };
    let entries = tools
        .as_array()
        .ok_or_else(|| ServeError::invalid_request(Some("tools"), "tools must be an array"))?;
    let mut definitions = Vec::with_capacity(entries.len());
    let mut names = BTreeSet::new();
    for (index, entry) in entries.iter().enumerate() {
        let map = entry.as_object().ok_or_else(|| {
            ServeError::invalid_request(Some("tools"), format!("tool {index} must be an object"))
        })?;
        match non_null(map, "type") {
            Some(Value::String(kind)) if kind == "function" => {}
            None => {}
            Some(Value::String(other)) => {
                return Err(ServeError::invalid_request(
                    Some("tools"),
                    format!("tool {index}: hosted tool type {other:?} is not supported"),
                ));
            }
            Some(_) => {
                return Err(ServeError::invalid_request(
                    Some("tools"),
                    format!("tool {index}: type must be a string"),
                ));
            }
        }
        let description = non_null(map, "description")
            .map(|v| {
                v.as_str().map(str::to_owned).ok_or_else(|| {
                    ServeError::invalid_request(
                        Some("tools"),
                        format!("tool {index}: description must be a string"),
                    )
                })
            })
            .transpose()?;
        let parameters = match non_null(map, "parameters") {
            Some(value @ Value::Object(_)) => value.clone(),
            Some(_) => {
                return Err(ServeError::invalid_request(
                    Some("tools"),
                    format!("tool {index}: parameters must be an object"),
                ));
            }
            None => Value::Null,
        };
        let strict = non_null(map, "strict")
            .map(|value| {
                value.as_bool().ok_or_else(|| {
                    ServeError::invalid_request(
                        Some("tools"),
                        format!("tool {index}: strict must be a boolean"),
                    )
                })
            })
            .transpose()?;
        if strict == Some(true) {
            return Err(ServeError::invalid_request(
                Some("tools"),
                format!("tool {index}: strict:true requires unsupported schema enforcement"),
            ));
        }
        let name = required_str_for(map, "name", "tools", &format!("tool {index}"))?;
        validate_tool_name(&name, "tools", &format!("tool {index}: name"))?;
        if !names.insert(name.clone()) {
            return Err(ServeError::invalid_request(
                Some("tools"),
                format!("tools contains duplicate function name {name:?}"),
            ));
        }
        definitions.push(ToolDefinition {
            name,
            description,
            parameters,
            strict,
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
                let part = part.as_object().ok_or_else(|| {
                    ServeError::invalid_request(
                        Some("input"),
                        format!("item {index}: reasoning parts must be objects"),
                    )
                })?;
                if part.get("type").and_then(Value::as_str) != Some("reasoning_text") {
                    return Err(ServeError::invalid_request(
                        Some("input"),
                        format!("item {index}: reasoning parts require type:\"reasoning_text\""),
                    ));
                }
                let fragment = part.get("text").and_then(Value::as_str).ok_or_else(|| {
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
                "item {index}: reasoning items require plain content; encrypted_content is unsupported"
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
        assert_eq!(request.system_source, Some(SystemSource::Developer));

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
            "future_request_field": {"k": "v"},
            "reasoning": {"effort": "high", "future_reasoning_field": 1},
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
        assert_eq!(request.reasoning, Some(json!({"effort": "high"})));
    }

    #[test]
    fn unsupported_standard_controls_fail_closed() {
        for field in ["metadata", "max_tool_calls", "text", "stream_options"] {
            let mut body = json!({"model": "m", "input": "q"});
            body[field] = json!({});
            let error = parse(body).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(field));
        }
        let error = parse(json!({
            "model": "m",
            "input": "q",
            "reasoning": {"summary": "auto"}
        }))
        .unwrap_err();
        assert_eq!(error.param.as_deref(), Some("reasoning.summary"));
    }

    #[test]
    fn generation_ranges_fail_before_backend_execution() {
        for body in [
            json!({"model":"m", "input":"q", "max_output_tokens":0}),
            json!({"model":"m", "input":"q", "temperature":-0.1}),
            json!({"model":"m", "input":"q", "temperature":-1e-50}),
            json!({"model":"m", "input":"q", "temperature":1e-50}),
            json!({"model":"m", "input":"q", "top_p":0.0}),
            json!({"model":"m", "input":"q", "top_p":1e-50}),
            json!({"model":"m", "input":"q", "top_p":1.1}),
            json!({"model":"m", "input":"q", "x_qwen":{"min_p":-0.1}}),
            json!({"model":"m", "input":"q", "x_qwen":{"min_p":-1e-50}}),
            json!({"model":"m", "input":"q", "x_qwen":{"min_p":1e-50}}),
            json!({"model":"m", "input":"q", "x_qwen":{"min_p":1.1}}),
        ] {
            assert!(parse(body).is_err());
        }
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
            Turn::ToolResults(vec![ToolResult {
                call_id: "call_abc123".into(),
                name: "fs_list".into(),
                output: "{\"entries\":[\"a.txt\"],\"path\":\"/tmp\"}".into(),
            }])
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
            "tool_choice": {"type": "allowed_tools", "tools": [
                {"type": "function", "name": "fs_list"}
            ]},
            "input": "go",
        }))
        .expect("allowed_tools parses");
        assert_eq!(unrestricted.allowed_tools, vec!["fs_list", "fs_write"]);
        assert_eq!(narrowed.allowed_tools, vec!["fs_list".to_string()]);
        assert_eq!(narrowed.tool_choice["mode"], "auto");
        // The cache-preserving property: narrowing must not perturb the
        // rendered prompt, so prior prefixes (and their checkpoints) stay
        // valid across a narrowing change.
        assert_eq!(
            crate::open_responses::render::render_qwen_serve_prompt(&unrestricted),
            crate::open_responses::render::render_qwen_serve_prompt(&narrowed),
            "allowed_tools must not change rendered bytes"
        );
    }

    #[test]
    fn allowed_tools_validation_fails_closed() {
        let tools = json!([{"type": "function", "name": "fs_list",
                            "parameters": {"type": "object"}}]);
        for (choice, fragment) in [
            (
                json!({"type": "allowed_tools", "tools": [
                    {"type": "function", "name": "nope"}
                ]}),
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
            (
                json!({"type": "allowed_tools", "tools": [{"name": "fs_list"}]}),
                "type:\"function\"",
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
    fn tool_names_and_call_ids_are_bounded() {
        let long_name = "a".repeat(65);
        for name in ["bad name", "bad.name", long_name.as_str()] {
            let error = parse(json!({
                "model":"m", "input":"q", "tools":[{"name":name}]
            }))
            .unwrap_err();
            assert_eq!(error.param.as_deref(), Some("tools"));
        }
        let call_id = "c".repeat(65);
        let error = parse(json!({"model":"m", "input":[
            {"role":"user", "content":"q"},
            {"type":"function_call", "call_id":call_id, "name":"ok", "arguments":"{}"}
        ]}))
        .unwrap_err();
        assert!(error.message.contains("64 bytes"));
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
            Turn::ToolResults(vec![
                ToolResult {
                    call_id: "c1".into(),
                    name: "a".into(),
                    output: "r1".into(),
                },
                ToolResult {
                    call_id: "c2".into(),
                    name: "b".into(),
                    output: "r2".into(),
                },
            ])
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

    #[test]
    fn malformed_known_request_fields_fail_closed() {
        for (field, value) in [
            ("store", json!("false")),
            ("stream", json!(1)),
            ("max_output_tokens", json!(1.5)),
            ("temperature", json!("0.7")),
            ("top_p", json!([])),
            ("instructions", json!({})),
            ("x_qwen", json!([])),
            ("parallel_tool_calls", json!("true")),
            ("previous_response_id", json!(7)),
            ("truncation", json!(false)),
            ("reasoning", json!("high")),
            ("tools", json!({})),
            ("tool_choice", json!(7)),
        ] {
            let mut body = json!({"model": "m", "input": "q"});
            body[field] = value;
            assert!(parse(body).is_err(), "malformed {field} was accepted");
        }
        assert!(parse(json!({"model":"m", "input":"q", "temperature":1e100})).is_err());
        for extension in [
            json!({"seed":-1}),
            json!({"top_k":1.5}),
            json!({"min_p":"0.1"}),
            json!({"no_thinking":0}),
            json!({"history_thinking":false}),
            json!({"stats":"yes"}),
        ] {
            assert!(parse(json!({"model":"m", "input":"q", "x_qwen":extension})).is_err());
        }
    }

    #[test]
    fn replayed_calls_require_object_arguments_and_unique_known_linkage() {
        let base = |tail: Value| {
            json!({"model":"m", "input": [
                {"role":"user", "content":"q"},
                {"type":"function_call", "call_id":"c1", "name":"a", "arguments":"{}"},
                {"type":"function_call", "call_id":"c2", "name":"b", "arguments":"{}"},
                tail
            ]})
        };
        assert!(
            parse(base(
                json!({"type":"function_call_output", "call_id":"unknown", "output":"x"})
            ))
            .unwrap_err()
            .message
            .contains("unknown")
        );
        for arguments in [Value::Null, json!(7), json!("bad"), json!("[]")] {
            let mut body = json!({"model":"m", "input":[{"role":"user","content":"q"},{"type":"function_call","call_id":"c","name":"a"}]});
            if !arguments.is_null() {
                body["input"][1]["arguments"] = arguments;
            }
            assert!(parse(body).is_err());
        }
        let duplicate_call = json!({"model":"m", "input":[
            {"role":"user","content":"q"},
            {"type":"function_call","call_id":"c","name":"a","arguments":"{}"},
            {"type":"function_call","call_id":"c","name":"b","arguments":"{}"}
        ]});
        assert!(
            parse(duplicate_call)
                .unwrap_err()
                .message
                .contains("duplicate")
        );
        let duplicate_output = json!({"model":"m", "input":[
            {"role":"user","content":"q"},
            {"type":"function_call","call_id":"c","name":"a","arguments":"{}"},
            {"type":"function_call_output","call_id":"c","output":"one"},
            {"type":"function_call_output","call_id":"c","output":"two"}
        ]});
        assert!(
            parse(duplicate_output)
                .unwrap_err()
                .message
                .contains("duplicate")
        );
    }

    #[test]
    fn required_allowed_tools_mode_is_rejected() {
        let error = parse(json!({"model":"m", "input":"q", "tools":[{"name":"a"}],
        "tool_choice":{"type":"allowed_tools","mode":"required","tools":[
            {"type":"function","name":"a"}
        ]}}))
        .unwrap_err();
        assert!(error.message.contains("required"));
    }

    #[test]
    fn explicit_null_optional_fields_are_absent() {
        let request = parse(json!({
            "model":"m", "input":"q", "previous_response_id":null,
            "store":null, "truncation":null, "reasoning":null, "tools":null,
            "tool_choice":null, "parallel_tool_calls":null, "stream":null,
            "max_output_tokens":null, "temperature":null, "top_p":null,
            "instructions":null, "x_qwen":null
        }))
        .expect("explicit null optionals are absent");
        assert!(request.instructions.is_none());
        assert!(request.reasoning.is_none());
        assert!(request.tools.is_empty());
        assert!(request.allowed_tools.is_empty());
        assert!(!request.stream);
        assert!(request.parallel_tool_calls);
    }

    #[test]
    fn reversed_parallel_outputs_are_reordered_for_qwen_rendering() {
        let request = parse(json!({"model":"m", "input":[
            {"role":"user","content":"q"},
            {"type":"function_call","call_id":"c1","name":"a","arguments":"{}"},
            {"type":"function_call","call_id":"c2","name":"b","arguments":"{}"},
            {"type":"function_call_output","call_id":"c2","output":"r2"},
            {"type":"function_call_output","call_id":"c1","output":"r1"}
        ]}))
        .expect("parallel outputs may arrive in arbitrary order");
        assert_eq!(
            request.turns.last(),
            Some(&Turn::ToolResults(vec![
                ToolResult {
                    call_id: "c1".into(),
                    name: "a".into(),
                    output: "r1".into(),
                },
                ToolResult {
                    call_id: "c2".into(),
                    name: "b".into(),
                    output: "r2".into(),
                },
            ]))
        );
        let rendered = crate::open_responses::render::render_qwen_serve_prompt(&request);
        assert!(rendered.find("r1").unwrap() < rendered.find("r2").unwrap());

        let error = parse(json!({"model":"m", "input":[
            {"role":"user","content":"q"},
            {"type":"function_call","call_id":"c1","name":"a","arguments":"{}"},
            {"type":"function_call","call_id":"c2","name":"b","arguments":"{}"},
            {"type":"function_call_output","call_id":"c2","output":"r2"},
            {"role":"user","content":"too early"}
        ]}))
        .unwrap_err();
        assert!(error.message.contains("require all outputs"));
    }

    #[test]
    fn strict_tools_fail_closed_or_echo_false() {
        let request = parse(json!({"model":"m", "input":"q", "tools":[{
            "type":"function", "name":"a", "strict":false
        }]}))
        .expect("strict:false is supported");
        assert_eq!(request.tools[0].strict, Some(false));

        let error = parse(json!({"model":"m", "input":"q", "tools":[{
            "type":"function", "name":"a", "strict":true
        }]}))
        .unwrap_err();
        assert!(error.message.contains("unsupported schema enforcement"));
    }

    #[test]
    fn duplicate_tools_and_unknown_reasoning_parts_fail_closed() {
        let duplicate = parse(json!({
            "model":"m", "input":"q",
            "tools":[{"name":"a"},{"name":"a"}]
        }))
        .unwrap_err();
        assert!(duplicate.message.contains("duplicate"));

        let bad_reasoning = parse(json!({"model":"m", "input":[
            {"role":"user","content":"q"},
            {"type":"reasoning","content":[{"type":"mystery","text":"r"}]},
            {"role":"assistant","content":"a"},
            {"role":"user","content":"q2"}
        ]}))
        .unwrap_err();
        assert!(bad_reasoning.message.contains("reasoning_text"));
    }

    #[test]
    fn error_envelopes_always_use_string_code_and_param() {
        let generic = ServeError::invalid_request(None, "bad").to_json();
        assert_eq!(generic["error"]["code"], "");
        assert_eq!(generic["error"]["param"], "");

        let meaningful = ServeError::previous_response_not_found().to_json();
        assert_eq!(meaningful["error"]["code"], "previous_response_not_found");
        assert_eq!(meaningful["error"]["param"], "previous_response_id");
    }
}
