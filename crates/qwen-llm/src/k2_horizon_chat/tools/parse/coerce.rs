//! Interpret outer value types only. Ranges, regexes, required fields and nested
//! object validation remain the tool consumer's responsibility.
use super::*;

const STRING: u8 = 1;
const INTEGER: u8 = 2;
const NUMBER: u8 = 4;
const BOOL: u8 = 8;
const NULL: u8 = 16;
const ARRAY: u8 = 32;
const OBJECT: u8 = 64;
const ANY: u8 = 127;

// JSON Schema integer membership is mathematical, unlike IFM's Python-style
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

fn kind(value: &Value) -> u8 {
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
fn named(name: &str) -> u8 {
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
fn kinds(spec: &Value, root: &Value, active: &mut Vec<String>) -> Result<u8> {
    if active.len() >= 128 {
        return Err(error("tool argument schema exceeds reference safety limit"));
    }
    if spec == &Value::Bool(false) {
        return Ok(0);
    }
    if !spec.is_object() {
        return Ok(ANY);
    }
    let mut mask = ANY;
    if let Some(r) = spec["$ref"].as_str() {
        if !active.iter().any(|v| v == r)
            && let Some(target) = types::resolve_argument_ref(root, spec)
        {
            active.push(r.into());
            // Interpret the same sibling overlay the native template presents,
            // not a different JSON Schema $ref validation dialect.
            let resolved = kinds(&target, root, active)?;
            active.pop();
            return Ok(resolved);
        }
    }
    if let Some(name) = spec["type"].as_str() {
        mask &= named(name);
    } else if let Some(types) = spec["type"].as_array() {
        mask &= types
            .iter()
            .fold(0, |m, v| m | v.as_str().map_or(ANY, named));
    }
    if let Some(values) = spec["enum"].as_array() {
        mask &= values.iter().fold(0, |m, v| m | kind(v));
    }
    if let Some(value) = spec.get("const") {
        mask &= kind(value);
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(variants) = spec[key].as_array() {
            let mut combined = if key == "allOf" { ANY } else { 0 };
            for variant in variants {
                // Nesting is guarded independently of reference depth.
                active.push(String::new());
                let child = kinds(variant, root, active)?;
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

fn argument_mask(function: &Value, key: &str, label: Option<&str>) -> Result<u8> {
    let parameters = &function["parameters"];
    let presented =
        types::resolve_argument_ref(parameters, parameters).unwrap_or_else(|| parameters.clone());
    let schema = &presented["properties"][key];
    let mut mask = kinds(schema, parameters, &mut Vec::new())?;
    if let Some(label) = label {
        let base = label.split('[').next().unwrap();
        if !label.contains('|') {
            mask &= named(base);
        }
    }
    Ok(mask)
}

pub(super) fn structured(function: &Value, key: &str, label: Option<&str>) -> Result<bool> {
    let mask = argument_mask(function, key, label)?;
    Ok(mask & STRING == 0 && mask & (ARRAY | OBJECT) != 0)
}

pub(super) fn argument(
    function: &Value,
    key: &str,
    raw: &str,
    label: Option<&str>,
) -> Result<Value> {
    let mask = argument_mask(function, key, label)?;
    let string = (mask & STRING != 0).then(|| Value::String(raw.into()));
    let parsed = json_decode::complete(raw)
        .ok()
        .filter(|v| !v.is_string() && kind(v) & mask != 0);
    match (string, parsed) {
        (Some(_), Some(_)) => Err(error(format!(
            "ambiguous XML argument {key:?}; use JSON call format or an unambiguous typed schema"
        ))),
        (Some(value), None) => Ok(value),
        (None, Some(value)) => {
            json::encode(&value)?;
            Ok(value)
        }
        _ => Err(error(format!(
            "XML argument {key:?} does not encode its declared outer type"
        ))),
    }
}
