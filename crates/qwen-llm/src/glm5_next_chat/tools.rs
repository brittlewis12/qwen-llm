//! GLM-5.3-Flash tool definitions, calls and results, as the upstream
//! template renders them (pinned by the `tool*` fixture cases):
//!
//! - Definitions: one `<|system|>` block after the effort header,
//!   `<tools>\n{json}\n…</tools>` with each function object printed in its
//!   own key order (`strict` dropped) and values through Python `tojson`.
//! - Calls: right after the stripped content, no separators:
//!   `<tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>…</tool_call>`,
//!   a string value verbatim and any other value through `tojson`.
//! - Results: consecutive tool messages share one `<|observation|>` and
//!   render as `<tool_response>{content}</tool_response>` in call order.
//!
//! Narrower than the template on purpose: arguments must be JSON objects
//! (the template errors on a string), results must answer exactly the
//! preceding turn's calls (each once), every call must name a declared
//! function, and `strict: true` or `defer_loading` definitions are refused
//! (neither can be honored), as are `parameters` schemas that generated
//! arguments could not be typed by (unresolvable or malformed `$ref`,
//! malformed `properties` or combinators; `tool_schema::check_strict_parameters`).

use super::*;
use serde_json::{Map, Value};

pub const TOOL_CALL_OPEN: &str = "<tool_call>";
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";
pub const ARG_KEY_OPEN: &str = "<arg_key>";
pub const ARG_KEY_CLOSE: &str = "</arg_key>";
pub const ARG_VALUE_OPEN: &str = "<arg_value>";
pub const ARG_VALUE_CLOSE: &str = "</arg_value>";
pub const TOOL_RESPONSE_OPEN: &str = "<tool_response>";
pub const TOOL_RESPONSE_CLOSE: &str = "</tool_response>";
pub const OBSERVATION: &str = "<|observation|>";

pub(super) const TOOL_MARKERS: [(&str, i32); 8] = [
    (TOOL_CALL_OPEN, 154_843),
    (TOOL_CALL_CLOSE, 154_844),
    (TOOL_RESPONSE_OPEN, 154_845),
    (TOOL_RESPONSE_CLOSE, 154_846),
    (ARG_KEY_OPEN, 154_847),
    (ARG_KEY_CLOSE, 154_848),
    (ARG_VALUE_OPEN, 154_849),
    (ARG_VALUE_CLOSE, 154_850),
];

const TOOLS_HEADER: &str = "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
const TOOLS_FOOTER: &str = "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";

pub(super) fn tools(message: impl Into<String>) -> ChatError {
    error("glm5_next_chat_tools", message)
}

/// A function name the output grammar can delimit: 1-64 of
/// `[A-Za-z0-9_.-]` (the OpenAI rule plus `.`).
pub fn valid_tool_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

/// An argument key the output grammar can delimit: nonempty, no `<`/`>`.
pub fn valid_argument_key(key: &str) -> bool {
    !key.is_empty() && !key.contains(['<', '>'])
}

/// One declared function: its object exactly as the template prints it
/// (`name`, then any of `description` and `parameters`, in the given order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDefinition {
    function: Map<String, Value>,
}

impl ToolDefinition {
    /// A definition in either document form: `{"type": "function",
    /// "function": {...}}` or the bare function object.
    pub fn from_value(value: &Value) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| tools("a tool definition must be an object"))?;
        let function = match object.get("function") {
            Some(function) => {
                for key in object.keys() {
                    if !matches!(key.as_str(), "type" | "function") {
                        return Err(tools(format!("unsupported tool definition key {key:?}")));
                    }
                }
                if object.get("type").is_some_and(|t| t != "function") {
                    return Err(tools("only function tools are supported"));
                }
                function
                    .as_object()
                    .ok_or_else(|| tools("tool function must be an object"))?
            }
            None => object,
        };
        let mut kept = Map::new();
        for (key, value) in function {
            match key.as_str() {
                "name" => {
                    let name = value
                        .as_str()
                        .ok_or_else(|| tools("tool name must be a string"))?;
                    if !valid_tool_name(name) {
                        return Err(tools(format!(
                            "tool name {name:?} must be 1-64 of [A-Za-z0-9_.-]"
                        )));
                    }
                }
                "description" if !value.is_string() => {
                    return Err(tools("tool description must be a string"));
                }
                "parameters" if !value.is_object() => {
                    return Err(tools("tool parameters must be a JSON Schema object"));
                }
                // Generated arguments are typed by this schema; one that
                // cannot type them is refused now, not after generation.
                "parameters" => crate::tool_schema::check_strict_parameters(value)
                    .map_err(|e| tools(format!("tool parameters: {e}")))?,
                "description" => {}
                // The template drops `strict`; only `false` means what it shows.
                "strict" if value == &Value::Bool(false) => continue,
                "strict" => {
                    return Err(tools(
                        "strict tool schemas cannot be honored (no constrained decoding)",
                    ));
                }
                "defer_loading" => {
                    return Err(tools(
                        "deferred tools are hidden from the model but callable; not supported",
                    ));
                }
                other => {
                    return Err(tools(format!("unsupported tool function key {other:?}")));
                }
            }
            kept.insert(key.clone(), value.clone());
        }
        if !kept.contains_key("name") {
            return Err(tools("a tool definition needs a name"));
        }
        Ok(Self { function: kept })
    }

    /// A definition from its parts, printed `name`, `description`,
    /// `parameters` (the shape serve receives).
    pub fn from_parts(
        name: &str,
        description: Option<&str>,
        parameters: Option<&Value>,
    ) -> Result<Self> {
        let mut function = Map::new();
        function.insert("name".into(), Value::String(name.into()));
        if let Some(description) = description {
            function.insert("description".into(), Value::String(description.into()));
        }
        if let Some(parameters) = parameters.filter(|p| !p.is_null()) {
            function.insert("parameters".into(), parameters.clone());
        }
        Self::from_value(&Value::Object(function))
    }

    pub fn name(&self) -> &str {
        self.function["name"].as_str().unwrap()
    }

    /// The JSON Schema of the arguments, if declared.
    pub fn parameters(&self) -> Option<&Value> {
        self.function.get("parameters")
    }

    pub fn description(&self) -> Option<&str> {
        self.function.get("description").and_then(Value::as_str)
    }

    fn render(&self, out: &mut String) -> Result<()> {
        out.push('{');
        for (i, (key, value)) in self.function.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            // Keys print raw (`"{{ k }}"`); every admitted key is plain.
            out.push('"');
            out.push_str(key);
            out.push_str("\": ");
            out.push_str(&tojson(value)?);
        }
        out.push('}');
        Ok(())
    }
}

/// One assistant tool call. `arguments` keep their order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Map<String, Value>,
}

impl ToolCall {
    /// A call in either document form: `{"id", "type": "function",
    /// "function": {"name", "arguments"}}` or `{"id", "name", "arguments"}`.
    pub(super) fn from_value(value: &Value, index: usize) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| tools(format!("message {index}: a tool call must be an object")))?;
        let id = object
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| tools(format!("message {index}: a tool call needs a string id")))?;
        let (function, allowed): (&Map<String, Value>, &[&str]) = match object.get("function") {
            Some(function) => {
                if object.get("type").is_some_and(|t| t != "function") {
                    return Err(tools(format!(
                        "message {index}: only function tool calls are supported"
                    )));
                }
                let function = function.as_object().ok_or_else(|| {
                    tools(format!(
                        "message {index}: tool call function must be an object"
                    ))
                })?;
                for key in object.keys() {
                    if !matches!(key.as_str(), "id" | "type" | "function") {
                        return Err(tools(format!(
                            "message {index}: unsupported tool call key {key:?}"
                        )));
                    }
                }
                (function, &["name", "arguments"])
            }
            None => (object, &["id", "name", "arguments"]),
        };
        for key in function.keys() {
            if !allowed.contains(&key.as_str()) {
                return Err(tools(format!(
                    "message {index}: unsupported tool call key {key:?}"
                )));
            }
        }
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| tools(format!("message {index}: a tool call needs a name")))?;
        let arguments = match function.get("arguments") {
            Some(Value::Object(arguments)) => arguments.clone(),
            // The template iterates `arguments.items()`: a JSON string errors.
            Some(_) => {
                return Err(tools(format!(
                    "message {index}: tool call arguments must be a JSON object"
                )));
            }
            None => {
                return Err(tools(format!(
                    "message {index}: a tool call needs arguments"
                )));
            }
        };
        let call = Self {
            id: id.into(),
            name: name.into(),
            arguments,
        };
        call.check()?;
        Ok(call)
    }

    pub(super) fn check(&self) -> Result<()> {
        if !valid_tool_name(&self.name) {
            return Err(tools(format!(
                "tool call name {:?} must be 1-64 of [A-Za-z0-9_.-]",
                self.name
            )));
        }
        if let Some(key) = self.arguments.keys().find(|k| !valid_argument_key(k)) {
            return Err(tools(format!(
                "tool call argument key {key:?} must be nonempty without < or >"
            )));
        }
        Ok(())
    }

    /// `<tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>…</tool_call>`.
    pub fn render(&self, out: &mut String) -> Result<()> {
        out.push_str(TOOL_CALL_OPEN);
        out.push_str(&self.name);
        for (key, value) in &self.arguments {
            out.push_str(ARG_KEY_OPEN);
            out.push_str(key);
            out.push_str(ARG_KEY_CLOSE);
            out.push_str(ARG_VALUE_OPEN);
            match value {
                Value::String(text) => out.push_str(text),
                other => out.push_str(&tojson(other)?),
            }
            out.push_str(ARG_VALUE_CLOSE);
        }
        out.push_str(TOOL_CALL_CLOSE);
        Ok(())
    }
}

/// HF `tojson` (Python `json.dumps(ensure_ascii=False)`).
pub fn tojson(value: &Value) -> Result<String> {
    crate::tool_schema::python_json(value).map_err(tools)
}

/// The tools block, or nothing for no tools.
pub(super) fn render_definitions(definitions: &[ToolDefinition], out: &mut String) -> Result<()> {
    if definitions.is_empty() {
        return Ok(());
    }
    let mut names = std::collections::BTreeSet::new();
    for definition in definitions {
        if !names.insert(definition.name()) {
            return Err(tools(format!(
                "tool {:?} is declared twice",
                definition.name()
            )));
        }
    }
    out.push_str(TOOLS_HEADER);
    for definition in definitions {
        definition.render(out)?;
        out.push('\n');
    }
    out.push_str(TOOLS_FOOTER);
    Ok(())
}

/// For each tool block (consecutive tool messages), the message indices of
/// its results in the preceding turn's call order. Refuses calls without
/// declared tools, undeclared or duplicate calls, results that do not answer
/// exactly the preceding calls, and calls left unanswered before another turn.
pub(super) fn tool_blocks(
    messages: &[Message],
    definitions: &[ToolDefinition],
) -> Result<std::collections::BTreeMap<usize, Vec<usize>>> {
    let declared: std::collections::BTreeSet<&str> =
        definitions.iter().map(ToolDefinition::name).collect();
    let mut blocks = std::collections::BTreeMap::new();
    let mut index = 0;
    while index < messages.len() {
        match &messages[index] {
            Message::Assistant { calls, .. } if !calls.is_empty() => {
                let mut ids = std::collections::BTreeSet::new();
                for call in calls {
                    if declared.is_empty() {
                        return Err(tools(format!(
                            "message {index}: tool calls need declared tools"
                        )));
                    }
                    if !declared.contains(call.name.as_str()) {
                        return Err(tools(format!(
                            "message {index}: tool call {:?} names no declared tool",
                            call.name
                        )));
                    }
                    if !ids.insert(call.id.as_str()) {
                        return Err(tools(format!(
                            "message {index}: tool call id {:?} repeats",
                            call.id
                        )));
                    }
                }
                let start = index + 1;
                let mut end = start;
                while end < messages.len() && matches!(messages[end], Message::Tool { .. }) {
                    end += 1;
                }
                if end == start {
                    if start == messages.len() {
                        // A trailing call turn (no generation): results pending.
                        index = end;
                        continue;
                    }
                    return Err(tools(format!(
                        "message {index}: tool calls need their results before the next turn"
                    )));
                }
                let mut order = Vec::with_capacity(calls.len());
                for call in calls {
                    let positions: Vec<usize> = (start..end)
                        .filter(|&k| {
                            matches!(&messages[k], Message::Tool { call_id, .. } if *call_id == call.id)
                        })
                        .collect();
                    match positions.as_slice() {
                        [one] => order.push(*one),
                        [] => {
                            return Err(tools(format!(
                                "message {index}: tool call {:?} has no result",
                                call.id
                            )));
                        }
                        _ => {
                            return Err(tools(format!(
                                "tool call {:?} has more than one result",
                                call.id
                            )));
                        }
                    }
                }
                if order.len() != end - start {
                    return Err(tools(format!(
                        "message {start}: a tool result answers no call of the preceding turn"
                    )));
                }
                blocks.insert(start, order);
                index = end;
            }
            Message::Tool { .. } => {
                return Err(tools(format!(
                    "message {index}: a tool result needs a preceding tool call"
                )));
            }
            _ => index += 1,
        }
    }
    Ok(blocks)
}
