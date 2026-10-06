//! Interpret outer value types only. Ranges, regexes, required fields and nested
//! object validation remain the tool consumer's responsibility.
use super::*;

use crate::tool_schema::{ARRAY, OBJECT, STRING, schema_kinds, type_name_kinds, value_kind};

fn kind(value: &Value) -> u8 {
    value_kind(value)
}

fn named(name: &str) -> u8 {
    type_name_kinds(name)
}

fn kinds(spec: &Value, root: &Value, active: &mut Vec<String>) -> Result<u8> {
    schema_kinds(spec, root, active).map_err(error)
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
