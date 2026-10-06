//! Family-neutral tool-calling helpers shared by chat renderers and output
//! parsers (K2 Horizon, GLM-5.3-Flash).
//!
//! - [`python_json`]: HF chat templates' `tojson`, which is Python
//!   `json.dumps(ensure_ascii=False)`: `", "`/`": "` separators and Python's
//!   float repr (`1.0`, `1e+20`, `1.5e-07`). Compact serde JSON differs.
//! - Outer-type masks of a function's argument schema ([`argument_kinds`],
//!   [`value_kind`]): `type` (name or list), `enum`, `const`,
//!   `anyOf`/`oneOf`/`allOf` and local `$ref` (`#/$defs/`, `#/definitions/`,
//!   with the sibling overlay templates present). Ranges, patterns,
//!   required fields and nested validation remain the tool consumer's.
//! - [`decode_json_prefix`] / [`decode_json`]: JSON decoding that builds
//!   containers directly and refuses duplicate keys. serde's
//!   arbitrary-precision visitor would read a literal
//!   `"$serde_json::private::Number"` key as an internal number tag and
//!   silently collapse duplicates.

use serde_json::Value;

pub const STRING: u8 = 1;
pub const INTEGER: u8 = 2;
pub const NUMBER: u8 = 4;
pub const BOOL: u8 = 8;
pub const NULL: u8 = 16;
pub const ARRAY: u8 = 32;
pub const OBJECT: u8 = 64;
pub const ANY: u8 = 127;

/// Python `json.dumps(value, ensure_ascii=False)`. Refuses non-finite
/// floating-point values, which JSON cannot carry.
pub fn python_json(value: &Value) -> Result<String, String> {
    let mut out = String::new();
    write(value, &mut out)?;
    Ok(out)
}

fn write(value: &Value, out: &mut String) -> Result<(), String> {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, value)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(key).map_err(|e| e.to_string())?);
                out.push_str(": ");
                write(value, out)?;
            }
            out.push('}');
        }
        Value::Number(number) => {
            let text = number.to_string();
            if text.contains(['.', 'e', 'E']) {
                let value = number.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
                    "tool JSON requires finite binary64 floating-point values".to_owned()
                })?;
                let scientific = format!("{value:e}");
                let (mantissa, exponent) = scientific.split_once('e').unwrap();
                let exponent: i32 = exponent.parse().unwrap();
                if !(-4..16).contains(&exponent) {
                    out.push_str(mantissa);
                    out.push('e');
                    out.push(if exponent < 0 { '-' } else { '+' });
                    out.push_str(&format!("{:02}", exponent.abs()));
                } else {
                    let fixed = value.to_string();
                    out.push_str(&fixed);
                    if !fixed.contains('.') {
                        out.push_str(".0");
                    }
                }
            } else {
                out.push_str(&text);
            }
        }
        _ => out.push_str(&serde_json::to_string(value).map_err(|e| e.to_string())?),
    }
    Ok(())
}

// JSON Schema integer membership is mathematical, unlike a Python-style
// lexical type label. Do not round a fractional decimal through f64 to decide it.
fn integral(number: &serde_json::Number) -> bool {
    let text = number.to_string();
    let (mantissa, exponent) = text.split_once(['e', 'E']).unwrap_or((&text, "0"));
    if mantissa
        .bytes()
        .filter(u8::is_ascii_digit)
        .all(|b| b == b'0')
    {
        return true;
    }
    let Ok(exponent) = exponent.parse::<i64>() else {
        return !exponent.starts_with('-');
    };
    let fraction = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let power = i128::from(exponent) - fraction as i128;
    let zeros = mantissa
        .bytes()
        .rev()
        .filter(u8::is_ascii_digit)
        .take_while(|&b| b == b'0')
        .count();
    power >= 0 || zeros as i128 >= -power
}

/// The outer type of a JSON value as a single mask bit.
pub fn value_kind(value: &Value) -> u8 {
    match value {
        Value::String(_) => STRING,
        Value::Bool(_) => BOOL,
        Value::Null => NULL,
        Value::Array(_) => ARRAY,
        Value::Object(_) => OBJECT,
        Value::Number(number) => {
            if integral(number) {
                INTEGER
            } else {
                NUMBER
            }
        }
    }
}

/// A JSON Schema type name as a mask; unknown names admit anything.
pub fn type_name_kinds(name: &str) -> u8 {
    match name {
        "string" => STRING,
        "integer" => INTEGER,
        "number" => INTEGER | NUMBER,
        "boolean" => BOOL,
        "null" => NULL,
        "array" => ARRAY,
        "object" => OBJECT,
        _ => ANY,
    }
}

/// The schema a local `$ref` names: `#/$defs/<k>` in `$defs`,
/// `#/definitions/<k>` in `definitions` (each in its own namespace), with
/// JSON Pointer escapes (`~1`, `~0`) decoded. Deeper pointers are not
/// resolved.
pub fn ref_target<'a>(parameters: &'a Value, reference: &str) -> Option<&'a Value> {
    let (namespace, key) = if let Some(key) = reference.strip_prefix("#/$defs/") {
        ("$defs", key)
    } else {
        ("definitions", reference.strip_prefix("#/definitions/")?)
    };
    if key.contains('/') {
        return None;
    }
    let key = key.replace("~1", "/").replace("~0", "~");
    parameters.get(namespace)?.get(key.as_str())
}

/// A local `$ref` of `spec` ([`ref_target`]), merged with `spec`'s sibling
/// keys (the overlay a template presents).
pub fn resolve_ref(parameters: &Value, spec: &Value) -> Option<Value> {
    let reference = spec["$ref"].as_str()?;
    let mut merged = ref_target(parameters, reference)?.as_object()?.clone();
    for (key, value) in spec.as_object()? {
        if key != "$ref" {
            merged.insert(key.clone(), value.clone());
        }
    }
    Some(Value::Object(merged))
}

/// The outer types `spec` admits, resolving references against `root`.
pub fn schema_kinds(spec: &Value, root: &Value, active: &mut Vec<String>) -> Result<u8, String> {
    if active.len() >= 128 {
        return Err("tool argument schema exceeds reference safety limit".into());
    }
    if spec == &Value::Bool(false) {
        return Ok(0);
    }
    if !spec.is_object() {
        return Ok(ANY);
    }
    let mut mask = ANY;
    if let Some(r) = spec["$ref"].as_str()
        && !active.iter().any(|v| v == r)
        && let Some(target) = resolve_ref(root, spec)
    {
        active.push(r.into());
        // Interpret the same sibling overlay the native template presents,
        // not a different JSON Schema $ref validation dialect.
        let resolved = schema_kinds(&target, root, active)?;
        active.pop();
        return Ok(resolved);
    }
    if let Some(name) = spec["type"].as_str() {
        mask &= type_name_kinds(name);
    } else if let Some(types) = spec["type"].as_array() {
        mask &= types
            .iter()
            .fold(0, |m, v| m | v.as_str().map_or(ANY, type_name_kinds));
    }
    if let Some(values) = spec["enum"].as_array() {
        mask &= values.iter().fold(0, |m, v| m | value_kind(v));
    }
    if let Some(value) = spec.get("const") {
        mask &= value_kind(value);
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(variants) = spec[key].as_array() {
            let mut combined = if key == "allOf" { ANY } else { 0 };
            for variant in variants {
                // Nesting is guarded independently of reference depth.
                active.push(String::new());
                let child = schema_kinds(variant, root, active)?;
                active.pop();
                if key == "allOf" {
                    combined &= child;
                } else {
                    combined |= child;
                }
            }
            mask &= combined;
        }
    }
    Ok(mask)
}

/// The outer types a function's `parameters` admit for argument `key`
/// (`parameters` itself may be a `$ref`). A missing schema admits anything.
pub fn argument_kinds(parameters: &Value, key: &str) -> Result<u8, String> {
    let presented = resolve_ref(parameters, parameters).unwrap_or_else(|| parameters.clone());
    schema_kinds(&presented["properties"][key], parameters, &mut Vec::new())
}

/// [`schema_kinds`] for interpreting generated arguments, where silently
/// admitting anything would mistype a value: a `$ref` must resolve
/// ([`ref_target`]) or the schema is refused, and a `$ref` with sibling
/// keywords admits the intersection of the target's and the siblings'
/// types (JSON Schema applies both), not an overlay.
pub fn strict_schema_kinds(
    spec: &Value,
    root: &Value,
    active: &mut Vec<String>,
) -> Result<u8, String> {
    if active.len() >= 128 {
        return Err("tool argument schema exceeds reference safety limit".into());
    }
    if spec == &Value::Bool(false) {
        return Ok(0);
    }
    let Some(object) = spec.as_object() else {
        return Ok(ANY);
    };
    let mut mask = ANY;
    if let Some(reference) = object.get("$ref") {
        let reference = reference
            .as_str()
            .ok_or_else(|| "tool argument schema $ref must be a string".to_owned())?;
        let target = ref_target(root, reference).ok_or_else(|| {
            format!("tool argument schema $ref {reference:?} is not a local definition")
        })?;
        if !active.iter().any(|v| v == reference) {
            active.push(reference.into());
            mask &= strict_schema_kinds(target, root, active)?;
            active.pop();
        }
    }
    if let Some(name) = object.get("type").and_then(Value::as_str) {
        mask &= type_name_kinds(name);
    } else if let Some(types) = object.get("type").and_then(Value::as_array) {
        mask &= types
            .iter()
            .fold(0, |m, v| m | v.as_str().map_or(ANY, type_name_kinds));
    }
    if let Some(values) = object.get("enum").and_then(Value::as_array) {
        mask &= values.iter().fold(0, |m, v| m | value_kind(v));
    }
    if let Some(value) = object.get("const") {
        mask &= value_kind(value);
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(variants) = object.get(key).and_then(Value::as_array) {
            let mut combined = if key == "allOf" { ANY } else { 0 };
            for variant in variants {
                active.push(String::new());
                let child = strict_schema_kinds(variant, root, active)?;
                active.pop();
                if key == "allOf" {
                    combined &= child;
                } else {
                    combined |= child;
                }
            }
            mask &= combined;
        }
    }
    Ok(mask)
}

/// The outer types a function's `parameters` admit for argument `key` under
/// [`strict_schema_kinds`] (`parameters` may itself be a `$ref`). A missing
/// schema admits anything.
pub fn strict_argument_kinds(parameters: &Value, key: &str) -> Result<u8, String> {
    let mut active = Vec::new();
    let resolved;
    let parameters_schema = match parameters.get("$ref").and_then(Value::as_str) {
        Some(reference) => {
            resolved = ref_target(parameters, reference).ok_or_else(|| {
                format!("tool parameters $ref {reference:?} is not a local definition")
            })?;
            resolved
        }
        None => parameters,
    };
    match parameters_schema.get("properties").and_then(|p| p.get(key)) {
        Some(schema) => strict_schema_kinds(schema, parameters, &mut active),
        None => Ok(ANY),
    }
}

/// Why [`decode_json_prefix`] stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JsonDecodeError {
    /// The text ends inside a value.
    Incomplete,
    Malformed(String),
}

/// One JSON value at the start of `text` and the bytes it used.
pub fn decode_json_prefix(text: &str) -> Result<(Value, usize), JsonDecodeError> {
    let mut parser = JsonParser { text, at: 0 };
    let value = parser.value(0)?;
    Ok((value, parser.at))
}

/// Exactly one JSON value (surrounding JSON whitespace allowed).
pub fn decode_json(text: &str) -> Result<Value, String> {
    match decode_json_prefix(text) {
        Ok((value, offset)) if text[offset..].trim_matches(json_space).is_empty() => Ok(value),
        Ok(_) => Err("trailing tool JSON content".into()),
        Err(JsonDecodeError::Incomplete) => Err("incomplete tool JSON value".into()),
        Err(JsonDecodeError::Malformed(e)) => Err(e),
    }
}

fn json_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}

fn malformed(message: impl Into<String>) -> JsonDecodeError {
    JsonDecodeError::Malformed(message.into())
}

struct JsonParser<'a> {
    text: &'a str,
    at: usize,
}

impl JsonParser<'_> {
    fn whitespace(&mut self) {
        while self
            .text
            .as_bytes()
            .get(self.at)
            .is_some_and(|&c| matches!(c, b' ' | b'\t' | b'\r' | b'\n'))
        {
            self.at += 1;
        }
    }
    fn take(&mut self, byte: u8) -> bool {
        self.whitespace();
        if self.text.as_bytes().get(self.at) == Some(&byte) {
            self.at += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, byte: u8) -> Result<(), JsonDecodeError> {
        if self.take(byte) {
            Ok(())
        } else if self.at == self.text.len() {
            Err(JsonDecodeError::Incomplete)
        } else {
            Err(malformed("invalid tool JSON container syntax"))
        }
    }
    fn string(&mut self) -> Result<String, JsonDecodeError> {
        self.whitespace();
        let mut stream =
            serde_json::Deserializer::from_str(&self.text[self.at..]).into_iter::<String>();
        match stream.next() {
            Some(Ok(value)) => {
                self.at += stream.byte_offset();
                Ok(value)
            }
            Some(Err(e)) if e.is_eof() => Err(JsonDecodeError::Incomplete),
            Some(Err(e)) => Err(malformed(format!("invalid tool JSON string: {e}"))),
            None => Err(JsonDecodeError::Incomplete),
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value, JsonDecodeError> {
        if depth >= 128 {
            return Err(malformed("tool JSON exceeds nesting safety limit"));
        }
        self.whitespace();
        match self.text.as_bytes().get(self.at).copied() {
            None => Err(JsonDecodeError::Incomplete),
            Some(b'"') => self.string().map(Value::String),
            Some(b'{') => {
                self.at += 1;
                let mut map = serde_json::Map::new();
                if self.take(b'}') {
                    return Ok(Value::Object(map));
                }
                loop {
                    let key = self.string()?;
                    self.expect(b':')?;
                    if map.contains_key(&key) {
                        return Err(malformed("duplicate tool JSON key"));
                    }
                    map.insert(key, self.value(depth + 1)?);
                    if self.take(b'}') {
                        return Ok(Value::Object(map));
                    }
                    self.expect(b',')?;
                }
            }
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                if self.take(b']') {
                    return Ok(Value::Array(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    if self.take(b']') {
                        return Ok(Value::Array(items));
                    }
                    self.expect(b',')?;
                }
            }
            Some(first @ (b't' | b'f' | b'n')) => {
                let (literal, value) = match first {
                    b't' => ("true", Value::Bool(true)),
                    b'f' => ("false", Value::Bool(false)),
                    _ => ("null", Value::Null),
                };
                let rest = &self.text[self.at..];
                if rest.starts_with(literal) {
                    self.at += literal.len();
                    Ok(value)
                } else if literal.starts_with(rest) {
                    Err(JsonDecodeError::Incomplete)
                } else {
                    Err(malformed("invalid tool JSON literal"))
                }
            }
            Some(b'-' | b'0'..=b'9') => {
                let start = self.at;
                while self
                    .text
                    .as_bytes()
                    .get(self.at)
                    .is_some_and(|c| matches!(c, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'))
                {
                    self.at += 1;
                }
                match serde_json::from_str::<serde_json::Number>(&self.text[start..self.at]) {
                    Ok(number) => Ok(Value::Number(number)),
                    Err(e) if e.is_eof() && self.at == self.text.len() => {
                        Err(JsonDecodeError::Incomplete)
                    }
                    Err(e) => Err(malformed(format!("invalid tool JSON number: {e}"))),
                }
            }
            _ => Err(malformed("invalid tool JSON value")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn refs_resolve_in_their_own_namespace_with_pointer_escapes() {
        let parameters = json!({
            "$defs": {"T": {"type": "integer"}, "a/b": {"type": "boolean"}},
            "definitions": {"T": {"type": "string"}},
            "properties": {
                "d": {"$ref": "#/definitions/T"},
                "s": {"$ref": "#/$defs/T"},
                "e": {"$ref": "#/$defs/a~1b"},
                "deep": {"$ref": "#/$defs/T/properties/x"},
                "remote": {"$ref": "https://example.com/t.json"},
                "sib": {"$ref": "#/$defs/T", "type": ["integer", "string"]}
            }
        });
        assert_eq!(strict_argument_kinds(&parameters, "d"), Ok(STRING));
        assert_eq!(strict_argument_kinds(&parameters, "s"), Ok(INTEGER));
        assert_eq!(strict_argument_kinds(&parameters, "e"), Ok(BOOL));
        assert!(strict_argument_kinds(&parameters, "deep").is_err());
        assert!(strict_argument_kinds(&parameters, "remote").is_err());
        // Siblings intersect with the target (integer), not override it.
        assert_eq!(strict_argument_kinds(&parameters, "sib"), Ok(INTEGER));
        assert_eq!(strict_argument_kinds(&parameters, "missing"), Ok(ANY));
        // The shared resolver now also picks the right namespace.
        assert_eq!(
            resolve_ref(&parameters, &json!({"$ref": "#/definitions/T"})),
            Some(json!({"type": "string"}))
        );
    }

    #[test]
    fn the_decoder_keeps_containers_and_refuses_duplicates() {
        assert_eq!(
            decode_json(r#"{"x":{"$serde_json::private::Number":"7"}}"#).unwrap(),
            {
                let mut inner = serde_json::Map::new();
                inner.insert(
                    "$serde_json::private::Number".into(),
                    Value::String("7".into()),
                );
                let mut outer = serde_json::Map::new();
                outer.insert("x".into(), Value::Object(inner));
                Value::Object(outer)
            }
        );
        assert!(decode_json(r#"{"a":1,"a":2}"#).is_err());
        assert!(decode_json(r#"{"o":{"a":1,"a":2}}"#).is_err());
        assert_eq!(
            decode_json_prefix("[1, 2"),
            Err(JsonDecodeError::Incomplete)
        );
    }
}
