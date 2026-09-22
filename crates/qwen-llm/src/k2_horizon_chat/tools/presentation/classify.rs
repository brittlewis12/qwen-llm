use super::*;

const ROOT_KEYS: &[&str] = &[
    "type",
    "description",
    "enum",
    "default",
    "properties",
    "required",
    "optional",
    "title",
    "items",
    "oneOf",
    "anyOf",
    "additionalProperties",
    "patternProperties",
    "returns",
    "examples",
    "$defs",
    "definitions",
    "$ref",
];
const FUNCTION_KEYS: &[&str] = &[
    "name",
    "description",
    "parameters",
    "returns",
    "response",
    "type",
    "function",
];

fn container(value: &Value) -> bool {
    value.is_array() || value.is_object()
}
fn contains_mapping(value: &Value) -> bool {
    value.is_object()
        || value
            .as_array()
            .is_some_and(|v| v.iter().any(contains_mapping))
}

/// True means fallback, not invalid input. Do not short-circuit later validation
/// once a tool has requested fallback: later malformed definitions must still fail.
pub(super) fn definition(tool: &Value, classify: bool) -> Result<bool> {
    let function = tool.get("function").unwrap_or(tool);
    let map = object(function)?;
    text(&function["name"])?;
    let parameters = function
        .get("parameters")
        .filter(|v| !v.is_null())
        .ok_or_else(|| error("tool definition requires parameters (not arguments)"))?;
    if parameters.is_string() {
        return Err(error("tool parameters must not be a JSON string"));
    }
    let mut fallback = schema(parameters, false, classify, false)?;
    if classify {
        fallback |= match parameters.as_object() {
            None => true,
            Some(map) => map
                .iter()
                .any(|(key, value)| !ROOT_KEYS.contains(&key.as_str()) && container(value)),
        };
    }
    if function["returns"].is_object() {
        fallback |= schema(&function["returns"], false, classify, false)?;
    }
    if classify && function.get("returns").is_none() && function["response"].is_object() {
        fallback |= schema(&function["response"], true, true, false)?;
    }
    if classify {
        fallback |= map
            .iter()
            .any(|(key, value)| !FUNCTION_KEYS.contains(&key.as_str()) && container(value));
    }
    Ok(fallback)
}

fn schema(spec: &Value, lenient: bool, classify: bool, in_variant: bool) -> Result<bool> {
    let Some(map) = spec.as_object() else {
        return Ok(false);
    };
    if !lenient && let Some(required) = map.get("required") {
        let required = array(required)?;
        if !required.is_empty() && !truthy(&spec["properties"]) && !in_variant {
            return Err(error("required fields have no properties object"));
        }
        if truthy(&spec["properties"]) {
            let properties = object(&spec["properties"])?;
            for name in required {
                if !properties.contains_key(text(name)?) {
                    return Err(error("required field is absent from properties"));
                }
            }
        }
    }
    let mut fallback = false;
    if classify {
        for (key, value) in map {
            match key.as_str() {
                "$ref" => fallback |= !value.as_str().is_some_and(|v| local_key(v).is_some()),
                "$defs" | "definitions" => {
                    if let Some(defs) = value.as_object() {
                        for spec in defs.values() {
                            fallback |= schema(spec, true, true, false)?;
                        }
                    } else {
                        fallback = true;
                    }
                }
                "type" => fallback |= value.is_object(),
                "enum" | "oneOf" | "anyOf" => fallback |= !value.is_array(),
                "required" => fallback |= truthy(value) && !truthy(&spec["properties"]),
                "items"
                | "description"
                | "default"
                | "title"
                | "examples"
                | "properties"
                | "patternProperties"
                | "additionalProperties"
                | "returns" => {}
                _ => {
                    fallback |= if let Some(map) = value.as_object() {
                        map.values().any(contains_mapping)
                    } else {
                        value.is_array() && contains_mapping(value)
                    };
                }
            }
        }
    }
    if truthy(&spec["properties"]) {
        for child in object(&spec["properties"])?.values() {
            fallback |= schema(child, lenient, classify, false)?;
        }
    }
    if let Some(items) = map.get("items") {
        fallback |= schema(items, lenient, classify, false)?;
    }
    for key in ["oneOf", "anyOf"] {
        if truthy(&spec[key]) {
            match &spec[key] {
                Value::Array(variants) => {
                    for variant in variants {
                        fallback |= schema(variant, lenient, classify, true)?;
                    }
                }
                Value::String(_) | Value::Object(_) => {}
                _ => return Err(error("schema combinator is not iterable")),
            }
        }
    }
    for key in ["additionalProperties", "returns"] {
        if spec[key].is_object() {
            fallback |= schema(&spec[key], lenient, classify, false)?;
        }
    }
    if let Some(patterns) = spec["patternProperties"].as_object() {
        for spec in patterns.values() {
            fallback |= schema(spec, lenient, classify, false)?;
        }
    }
    Ok(fallback)
}
