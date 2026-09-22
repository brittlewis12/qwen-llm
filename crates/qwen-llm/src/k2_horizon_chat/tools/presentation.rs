//! Pinned IFM schema presentation, not a JSON Schema argument validator.
use super::types::{compact, truthy};
use super::*;
use std::borrow::Cow;

mod classify;
mod markdown;
mod system;
mod xml;
pub use system::render_tool_system;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolPresentationFormat {
    #[default]
    Markdown,
    Xml,
    Json,
}

const STRUCTURAL: &[&str] = &[
    "type",
    "description",
    "enum",
    "default",
    "properties",
    "required",
    "items",
    "oneOf",
    "anyOf",
    "additionalProperties",
    "patternProperties",
    "returns",
];

/// Render the native tool-definition block. If the upstream pretty-printer
/// cannot represent any definition, the entire block uses its JSON fallback.
/// This neither validates call arguments nor authorizes a frontend tool request.
pub fn render_tool_definitions(
    definitions: &[Value],
    format: ToolPresentationFormat,
) -> Result<String> {
    check_input_depth(definitions)?;
    let mut fallback = false;
    for definition in definitions {
        fallback |= classify::definition(definition, format != ToolPresentationFormat::Json)?;
    }
    let mut out = String::from("<ifm|tools>");
    if format == ToolPresentationFormat::Json || fallback {
        for definition in definitions {
            out.push('\n');
            out.push_str(&json::encode(definition)?);
        }
    } else {
        for (index, definition) in definitions.iter().enumerate() {
            let function = definition.get("function").unwrap_or(definition);
            let parameters = &function["parameters"];
            let mut renderer = Renderer::new(parameters);
            match format {
                ToolPresentationFormat::Markdown => {
                    renderer.markdown_function(function, &mut out)?;
                    if index + 1 != definitions.len() {
                        out.push('\n');
                    }
                }
                ToolPresentationFormat::Xml => renderer.xml_function(function, &mut out)?,
                ToolPresentationFormat::Json => unreachable!(),
            }
        }
    }
    out.push_str("\n</ifm|tools>");
    Ok(out)
}

// Bound native recursion also for callers constructing Value directly rather
// than using the wire's bounded JSON parser. This is not a model-context limit.
const MAX_NESTING: usize = 128;
pub(super) fn check_input_depth(definitions: &[Value]) -> Result<()> {
    let mut stack: Vec<_> = definitions.iter().map(|v| (v, 1)).collect();
    while let Some((value, depth)) = stack.pop() {
        if depth > MAX_NESTING {
            return Err(error("tool schema exceeds JSON nesting safety limit (128)"));
        }
        match value {
            Value::Object(map) => stack.extend(map.values().map(|v| (v, depth + 1))),
            Value::Array(array) => stack.extend(array.iter().map(|v| (v, depth + 1))),
            _ => {}
        }
    }
    Ok(())
}

struct Renderer<'a> {
    defs: &'a Value,
    seen: String,
    depth: usize,
}

impl<'a> Renderer<'a> {
    fn new(parameters: &'a Value) -> Self {
        let defs = if parameters["$defs"].is_object() {
            &parameters["$defs"]
        } else {
            &parameters["definitions"]
        };
        Self {
            defs,
            seen: "|".into(),
            depth: 0,
        }
    }
    fn nested<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.depth >= MAX_NESTING {
            return Err(error(
                "tool schema reference expansion exceeds renderer nesting safety limit (128)",
            ));
        }
        self.depth += 1;
        let result = f(self);
        self.depth -= 1;
        result
    }
    fn resolve<'s>(&mut self, spec: &'s Value, root: bool) -> Cow<'s, Value> {
        let mut result = Cow::Borrowed(spec);
        for hop in 0..if root { 1 } else { 2 } {
            let Some(key) = result["$ref"].as_str().and_then(local_key) else {
                break;
            };
            if !root && hop == 0 && self.seen.contains(&format!("|{key}|")) {
                break;
            }
            let Some(definition) = self.defs.get(key).and_then(Value::as_object) else {
                break;
            };
            self.seen.push_str(key);
            self.seen.push('|');
            let mut merged = definition.clone();
            for (key, value) in result.as_object().unwrap() {
                if key != "$ref" {
                    merged.insert(key.clone(), value.clone());
                }
            }
            result = Cow::Owned(Value::Object(merged));
        }
        result
    }
}

fn local_key(reference: &str) -> Option<&str> {
    reference
        .strip_prefix("#/$defs/")
        .or_else(|| reference.strip_prefix("#/definitions/"))
}
fn object(value: &Value) -> Result<&Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| error("tool schema expected an object"))
}
fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| error("tool schema expected an array"))
}
fn text(value: &Value) -> Result<&str> {
    value
        .as_str()
        .ok_or_else(|| error("tool schema expected a string"))
}
fn required(spec: &Value, name: &str) -> bool {
    spec["required"]
        .as_array()
        .is_some_and(|v| v.iter().any(|v| v == name))
}
fn collapse(text: &str) -> String {
    text.split(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}
fn python_repr(value: &Value) -> Result<String> {
    Ok(match value {
        Value::String(v) => format!(
            "'{}'",
            collapse(v).replace('\\', "\\\\").replace('\'', "\\'")
        ),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        Value::Array(v) => format!(
            "[{}]",
            v.iter()
                .map(python_repr)
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        ),
        Value::Object(v) => format!(
            "{{{}}}",
            v.iter()
                .map(|(k, v)| Ok(format!(
                    "{}: {}",
                    python_repr(&Value::String(k.clone()))?,
                    python_repr(v)?
                )))
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        ),
        Value::Number(_) => json::encode(value)?,
    })
}
fn markdown_value(value: &Value) -> Result<String> {
    match value.as_str() {
        Some("") => Ok("\"\"".into()),
        Some(s) => Ok(s.into()),
        None => python_repr(value),
    }
}
fn literal(value: &Value) -> Result<String> {
    match value.as_str() {
        Some("") => Ok("\"\"".into()),
        Some(s) => Ok(format!("`{}`", s.replace('\n', "\\n"))),
        None => Ok(format!("`{}`", python_repr(value)?)),
    }
}
fn allowed(value: &Value) -> Result<String> {
    array(value)?
        .iter()
        .map(literal)
        .collect::<Result<Vec<_>>>()
        .map(|v| v.join(", "))
}
fn markdown_type(spec: &Value) -> Result<String> {
    if spec.is_boolean() {
        return python_repr(spec);
    }
    if !spec.is_object() {
        return Ok("any".into());
    }
    let name = |name: &Value| -> Result<String> {
        if name == "array" {
            Ok(format!("array of {}", markdown_type(&spec["items"])?))
        } else if truthy(name) {
            Ok(text(name)?.into())
        } else {
            Ok("any".into())
        }
    };
    if let Some(types) = spec["type"].as_array() {
        return if types.is_empty() {
            Ok("any".into())
        } else {
            types
                .iter()
                .map(name)
                .collect::<Result<Vec<_>>>()
                .map(|v| v.join(" or "))
        };
    }
    if truthy(&spec["type"]) {
        return name(&spec["type"]);
    }
    if let Some(reference) = spec["$ref"].as_str() {
        return Ok(reference.rsplit('/').next().unwrap().into());
    }
    for key in ["oneOf", "anyOf"] {
        if truthy(&spec[key]) {
            return Ok(format!(
                "{key}[{}]",
                array(&spec[key])?
                    .iter()
                    .map(markdown_type)
                    .collect::<Result<Vec<_>>>()?
                    .join(" or ")
            ));
        }
    }
    if truthy(&spec["properties"]) {
        return Ok("object".into());
    }
    if spec.get("items").is_some() {
        return Ok(format!("array of {}", markdown_type(&spec["items"])?));
    }
    Ok("any".into())
}
