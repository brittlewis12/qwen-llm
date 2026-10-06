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

/// A local `$ref` (`#/$defs/<k>` or `#/definitions/<k>`) of `spec`, merged
/// with `spec`'s sibling keys.
pub fn resolve_ref(parameters: &Value, spec: &Value) -> Option<Value> {
    let reference = spec["$ref"].as_str()?;
    let key = reference
        .strip_prefix("#/$defs/")
        .or_else(|| reference.strip_prefix("#/definitions/"))?;
    let defs = if parameters["$defs"].is_object() {
        &parameters["$defs"]
    } else {
        &parameters["definitions"]
    };
    let mut merged = defs.get(key)?.as_object()?.clone();
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
