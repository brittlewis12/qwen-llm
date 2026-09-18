//! Completion-style raw subset of /v1/responses, never a chat template adapter.
use super::items::{ServeError, ServeRequest};
use qwen_llm::sampling::SamplingConfig;
use serde_json::Value;

fn invalid(param: &'static str, message: impl Into<String>) -> ServeError {
    ServeError::invalid_request(Some(param), message)
}

pub(crate) fn parse_request(body: &Value) -> Result<ServeRequest, ServeError> {
    let map = body
        .as_object()
        .ok_or_else(|| invalid("input", "K2 request must be an object"))?;
    for (key, value) in map {
        if ![
            "model",
            "input",
            "stream",
            "max_output_tokens",
            "temperature",
            "top_p",
            "store",
            "truncation",
            "x_qwen",
            "x_k2",
        ]
        .contains(&key.as_str())
        {
            return Err(invalid(
                "input",
                format!("K2 raw serving does not support field {key:?}"),
            ));
        }
        if value.is_null() {
            return Err(invalid(
                "input",
                format!("K2 raw field {key:?} cannot be null"),
            ));
        }
    }
    let raw = map
        .get("input")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| {
            invalid(
                "input",
                "K2 requires a nonempty raw input string, not message/items or token-ID arrays",
            )
        })?;
    if let Some(extension) = map.get("x_qwen") {
        let values = extension
            .as_object()
            .ok_or_else(|| invalid("x_qwen", "x_qwen must be an object"))?;
        for (key, value) in values {
            if !["seed", "top_k", "min_p", "stats"].contains(&key.as_str()) || value.is_null() {
                return Err(invalid(
                    "x_qwen",
                    format!("unsupported K2 x_qwen field/value {key:?}"),
                ));
            }
        }
    }
    let mut add_special_tokens = true;
    if let Some(extension) = map.get("x_k2") {
        let values = extension
            .as_object()
            .ok_or_else(|| invalid("x_k2", "x_k2 must be an object"))?;
        for (key, value) in values {
            if key != "add_special_tokens" {
                return Err(invalid("x_k2", format!("unsupported K2 extension {key:?}")));
            }
            add_special_tokens = value.as_bool().ok_or_else(|| {
                invalid(
                    "x_k2.add_special_tokens",
                    "add_special_tokens must be boolean",
                )
            })?;
        }
    }
    let mut common = body.clone();
    common.as_object_mut().unwrap().remove("x_k2");
    let mut request = super::items::parse_request(&common)?;
    request.k2_raw_input = Some(raw.to_owned());
    request.k2_add_special_tokens = Some(add_special_tokens);
    Ok(request)
}

pub(crate) fn normalize(
    request: &mut ServeRequest,
    default_max: usize,
    capacity: usize,
) -> Result<(), ServeError> {
    render(request)?;
    let maximum = *request.max_output_tokens.get_or_insert(default_max);
    if maximum == 0 || maximum > capacity {
        return Err(invalid(
            "max_output_tokens",
            format!("K2 max_output_tokens must be in 1..={capacity}"),
        ));
    }
    if request.temperature.is_none() {
        request.temperature = Some(0.0);
        request.temperature_echo = Some(0.0);
    }
    if request.top_p.is_none() {
        request.top_p = Some(1.0);
        request.top_p_echo = Some(1.0);
    }
    request.top_k.get_or_insert(0);
    request.min_p.get_or_insert(0.0);
    request.seed.get_or_insert(0);
    request.parallel_tool_calls = false;
    request.tool_choice = Value::String("none".into());
    sampling(request)
        .validate()
        .map_err(|e| invalid("temperature", format!("K2 sampling: {e}")))?;
    Ok(())
}

pub(crate) fn sampling(request: &ServeRequest) -> SamplingConfig {
    SamplingConfig {
        temperature: request.temperature.unwrap_or(0.0),
        top_k: request.top_k.unwrap_or(0),
        top_p: request.top_p.unwrap_or(1.0),
        min_p: request.min_p.unwrap_or(0.0),
        seed: request.seed.unwrap_or(0),
    }
}

pub(crate) fn render(request: &ServeRequest) -> Result<String, ServeError> {
    if request.instructions.is_some()
        || request.model_request.system.is_some()
        || request.model_request.has_tool_surface()
        || request.reasoning.is_some()
        || request.no_thinking
        || request.thinking_requested
        || request.strip_history_thinking
    {
        return Err(invalid(
            "input",
            "K2 raw serving does not render instructions, chat, tools, or reasoning controls",
        ));
    }
    request
        .k2_raw_input
        .clone()
        .filter(|text| !text.is_empty())
        .ok_or_else(|| {
            invalid(
                "input",
                "K2 input must originate from the raw string request parser",
            )
        })
}

#[cfg(test)]
mod tests;
