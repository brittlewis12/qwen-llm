use super::*;

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(v) => *v,
        Value::String(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
        Value::Number(v) => v.as_f64() != Some(0.),
    }
}

fn compact_name(name: &Value, spec: &Value) -> Result<String> {
    if name == "array" {
        Ok(format!("array[{}]", compact(&spec["items"])?))
    } else if !truthy(name) {
        Ok("any".into())
    } else {
        name.as_str()
            .map(str::to_owned)
            .ok_or_else(|| error("tool schema type must be a string"))
    }
}

fn compact(spec: &Value) -> Result<String> {
    if !spec.is_object() {
        return Ok("any".into());
    }
    if let Some(types) = spec["type"].as_array() {
        if types.is_empty() {
            return Ok("any".into());
        }
        return types
            .iter()
            .map(|name| compact_name(name, spec))
            .collect::<Result<Vec<_>>>()
            .map(|t| t.join("|"));
    }
    if truthy(&spec["type"]) {
        return compact_name(&spec["type"], spec);
    }
    if let Some(reference) = spec["$ref"].as_str() {
        return Ok(reference.rsplit('/').next().unwrap().into());
    }
    for key in ["oneOf", "anyOf"] {
        if truthy(&spec[key]) {
            let variants = spec[key]
                .as_array()
                .ok_or_else(|| error("tool schema combinator must be an array"))?;
            return Ok(format!(
                "{key}[{}]",
                variants
                    .iter()
                    .map(compact)
                    .collect::<Result<Vec<_>>>()?
                    .join("|")
            ));
        }
    }
    if truthy(&spec["properties"]) {
        return Ok("object".into());
    }
    if spec.get("items").is_some() {
        return Ok(format!("array[{}]", compact(&spec["items"])?));
    }
    Ok("any".into())
}

fn has_combinator(spec: &Value) -> bool {
    if truthy(&spec["oneOf"])
        || truthy(&spec["anyOf"])
        || spec["type"].as_array().is_some_and(|types| types.len() > 1)
    {
        true
    } else if spec["type"] == "array" && spec.get("items").is_some() {
        has_combinator(&spec["items"])
    } else {
        spec["properties"]
            .as_object()
            .is_some_and(|properties| properties.values().any(has_combinator))
    }
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
        Value::Number(number) => {
            if number.to_string().contains(['.', 'e', 'E']) {
                "number"
            } else {
                "integer"
            }
        }
    }
}

pub(super) fn argument_type(
    definitions: &[Value],
    name: &str,
    argument: &str,
    value: &Value,
) -> Result<String> {
    let mut found = "any".to_owned();
    for definition in definitions {
        let function = definition.get("function").unwrap_or(definition);
        if function["name"] != name {
            continue;
        }
        let parameters = &function["parameters"];
        let Some(spec) = parameters["properties"].get(argument) else {
            continue;
        };
        let mut spec = spec.clone();
        if let Some(reference) = spec["$ref"].as_str() {
            let key = reference
                .strip_prefix("#/$defs/")
                .or_else(|| reference.strip_prefix("#/definitions/"));
            let defs = if parameters["$defs"].is_object() {
                &parameters["$defs"]
            } else {
                &parameters["definitions"]
            };
            if let Some(definition) = key.and_then(|key| defs.get(key)).and_then(Value::as_object) {
                let mut merged = definition.clone();
                for (key, value) in spec.as_object().unwrap() {
                    if key != "$ref" {
                        merged.insert(key.clone(), value.clone());
                    }
                }
                spec = Value::Object(merged);
            }
        }
        found = if has_combinator(&spec) {
            value_type(value).into()
        } else {
            compact(&spec)?
        };
    }
    Ok(found)
}
