//! Muse Glimmer Open Responses items to the shared ATEM prompt contract.

use super::items::{ServeError, ServeRequest};
use crate::model_request::Turn;
use qwen_llm::muse_glimmer::MuseGlimmerChatTemplateProfile;
use qwen_llm::muse_glimmer_prompt::{
    MuseGlimmerMessage, MuseGlimmerReasoningStrength, MuseGlimmerToolCall,
    MuseGlimmerToolDefinition,
};
use qwen_llm::muse_glimmer_request::MuseGlimmerRequest;
use qwen_llm::sampling::SamplingConfig;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

const ATEM_VALUE_DELIMITERS: &[&str] = &["<atem:", "</atem:"];
const TOOL_OUTPUT_DELIMITERS: &[&str] = &["<tool_output", "</tool_output>"];

pub(crate) fn normalize_request(
    request: &mut ServeRequest,
    default_max_tokens: usize,
) -> Result<(), ServeError> {
    let defaults = SamplingConfig::muse_glimmer(42);
    if request.temperature.is_none() {
        request.temperature = Some(defaults.temperature);
        request.temperature_echo = Some(1.0);
    }
    if request.top_p.is_none() {
        request.top_p = Some(defaults.top_p);
        request.top_p_echo = Some(0.95);
    }
    request.top_k.get_or_insert(defaults.top_k);
    request.min_p.get_or_insert(defaults.min_p);
    request.seed.get_or_insert(defaults.seed);
    request.max_output_tokens.get_or_insert(default_max_tokens);
    let strength = reasoning_strength(request)?;
    request.reasoning_effort = Some(strength.as_str().into());
    request.reasoning = Some(json!({"effort": strength.as_str()}));
    Ok(())
}

pub(crate) fn reasoning_strength(
    request: &ServeRequest,
) -> Result<MuseGlimmerReasoningStrength, ServeError> {
    match request.reasoning_effort.as_deref() {
        None => Ok(MuseGlimmerReasoningStrength::High),
        Some(effort) => MuseGlimmerReasoningStrength::parse(effort).ok_or_else(|| {
            ServeError::invalid_request(
                Some("reasoning.effort"),
                format!(
                    "Muse Glimmer supports reasoning.effort {}; got {effort:?}",
                    MuseGlimmerReasoningStrength::level_names().join("|")
                ),
            )
        }),
    }
}

pub(crate) fn render_muse_glimmer_serve_prompt(
    request: &ServeRequest,
    profile: MuseGlimmerChatTemplateProfile,
) -> Result<String, ServeError> {
    if request.no_thinking {
        return Err(ServeError::invalid_request(
            Some("x_qwen.no_thinking"),
            "Muse Glimmer has no no-thinking ATEM profile; use reasoning.effort low",
        ));
    }
    if request.thinking_requested {
        return Err(ServeError::invalid_request(
            Some("x_qwen.thinking"),
            "Muse Glimmer always reasons; x_qwen.thinking is unsupported",
        ));
    }
    if request.strip_history_thinking {
        return Err(ServeError::invalid_request(
            Some("x_qwen.history_thinking"),
            "Muse Glimmer serve preserves ATEM reasoning history; strip is unsupported",
        ));
    }
    let strength = reasoning_strength(request)?;
    let tools = request
        .model_request
        .tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            if let Some(description) = tool.description.as_deref() {
                validate_content(
                    "tools",
                    &format!("tool {index} description"),
                    description,
                    &[],
                )?;
            }
            let parameters = if tool.parameters.is_null() {
                json!({})
            } else {
                tool.parameters.clone()
            };
            validate_json_content(
                "tools",
                &format!("tool {index} parameters"),
                &parameters,
                &[],
            )?;
            Ok(MuseGlimmerToolDefinition {
                name: tool.name.clone(),
                description: tool.description.clone().unwrap_or_default(),
                parameters,
            })
        })
        .collect::<Result<Vec<_>, ServeError>>()?;

    let mut messages = Vec::with_capacity(
        request.model_request.turns.len() + usize::from(request.model_request.system.is_some()),
    );
    if let Some(system) = request.model_request.system.as_deref() {
        validate_content("input", "system content", system, &[])?;
        messages.push(MuseGlimmerMessage::system(system));
    }
    for (index, turn) in request.model_request.turns.iter().enumerate() {
        match turn {
            Turn::User(text) => {
                validate_content("input", &format!("turn {index} user content"), text, &[])?;
                messages.push(MuseGlimmerMessage::user(text));
            }
            Turn::Assistant {
                reasoning,
                visible,
                calls,
            } => {
                if !visible.is_empty() && !calls.is_empty() {
                    return Err(ServeError::invalid_request(
                        Some("input"),
                        format!(
                            "turn {index}: Muse assistant history cannot combine visible content and function calls"
                        ),
                    ));
                }
                validate_content(
                    "input",
                    &format!("turn {index} assistant content"),
                    visible,
                    &[],
                )?;
                if let Some(reasoning) = reasoning {
                    validate_content(
                        "input",
                        &format!("turn {index} reasoning content"),
                        reasoning,
                        &[],
                    )?;
                }
                let mut message = MuseGlimmerMessage::assistant(visible);
                message.reasoning_content.clone_from(reasoning);
                for call in calls {
                    let arguments = serde_json::from_str::<Value>(&call.arguments)
                        .expect("ServeRequest validates function-call arguments");
                    validate_json_content(
                        "input",
                        &format!("turn {index} function-call arguments"),
                        &arguments,
                        ATEM_VALUE_DELIMITERS,
                    )?;
                    message.tool_calls.push(MuseGlimmerToolCall {
                        name: call.name.clone(),
                        arguments,
                    });
                }
                messages.push(message);
            }
            Turn::ToolResults(results) => {
                for result in results {
                    validate_content(
                        "input",
                        &format!("tool output for call {}", result.call_id),
                        &result.output,
                        TOOL_OUTPUT_DELIMITERS,
                    )?;
                    messages.push(MuseGlimmerMessage::tool(
                        result.name.clone(),
                        result.output.clone(),
                    ));
                }
            }
        }
    }
    MuseGlimmerRequest {
        messages,
        tools,
        tool_namespace_descriptions: BTreeMap::new(),
        reasoning_strength: None,
        current_date: if request.model_request.system.is_none() {
            Some(current_utc_date()?)
        } else {
            None
        },
    }
    .render(profile, Some(strength))
    .map_err(|error| ServeError::invalid_request(Some("input"), error.to_string()))
}

fn validate_content(
    param: &'static str,
    context: &str,
    content: &str,
    extra_reserved: &[&str],
) -> Result<(), ServeError> {
    let marker = std::iter::once("<|")
        .chain(extra_reserved.iter().copied())
        .find(|marker| content.contains(*marker));
    if let Some(marker) = marker {
        return Err(ServeError::invalid_request(
            Some(param),
            format!("{context} contains reserved Muse delimiter {marker:?}"),
        ));
    }
    Ok(())
}

fn validate_json_content(
    param: &'static str,
    context: &str,
    value: &Value,
    extra_reserved: &[&str],
) -> Result<(), ServeError> {
    match value {
        Value::String(value) => validate_content(param, context, value, extra_reserved),
        Value::Array(values) => values
            .iter()
            .try_for_each(|value| validate_json_content(param, context, value, extra_reserved)),
        Value::Object(values) => {
            for (key, value) in values {
                validate_content(param, context, key, extra_reserved)?;
                validate_json_content(param, context, value, extra_reserved)?;
            }
            Ok(())
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => Ok(()),
    }
}

fn current_utc_date() -> Result<String, ServeError> {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ServeError::server_error("system clock predates the Unix epoch"))?
        .as_secs()
        / 86_400;
    Ok(utc_date_from_unix_days(days as i64))
}

fn utc_date_from_unix_days(days: i64) -> String {
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::items::parse_request;

    fn parse(value: Value) -> ServeRequest {
        parse_request(&value).unwrap()
    }

    /// Missing reasoning is empty reasoning (serve's rule for every family):
    /// neither renders an ATEM `to=self` record. This is a native-renderer
    /// contract; no independent Muse template oracle exists.
    #[test]
    fn missing_history_reasoning_renders_as_explicit_empty_reasoning() {
        let user = |text: &str| json!({"role": "user", "content": text});
        let empty = json!({"type": "reasoning", "content": ""});
        let call =
            json!({"type": "function_call", "call_id": "c1", "name": "ping", "arguments": "{}"});
        let output = json!({"type": "function_call_output", "call_id": "c1", "output": "pong"});
        let answer = json!({"role": "assistant", "content": "Done."});
        let histories = [
            (
                json!([user("One"), answer, user("Two")]),
                json!([user("One"), empty, answer, user("Two")]),
            ),
            (
                json!([user("One"), call, output, answer, user("Two")]),
                json!([user("One"), empty, call, output, empty, answer, user("Two")]),
            ),
        ];
        for (missing, explicit) in &histories {
            let render = |input: &Value| {
                let mut request = parse(json!({"model": "muse", "input": input,
                    "tools": [{"type": "function", "name": "ping", "parameters": {"type": "object"}}]}));
                normalize_request(&mut request, 512).unwrap();
                render_muse_glimmer_serve_prompt(
                    &request,
                    MuseGlimmerChatTemplateProfile::UnslothLaunch,
                )
                .unwrap()
            };
            let rendered = render(missing);
            assert_eq!(rendered, render(explicit), "{missing}");
            assert!(!rendered.contains("to=self"), "{rendered}");
        }
    }

    #[test]
    fn omitted_controls_normalize_to_the_released_preset() {
        let mut request = parse(json!({"model":"muse", "input":"hello"}));
        normalize_request(&mut request, 512).unwrap();
        assert_eq!(request.temperature, Some(1.0));
        assert_eq!(request.temperature_echo, Some(1.0));
        assert_eq!(request.top_p, Some(0.95));
        assert_eq!(request.top_p_echo, Some(0.95));
        assert_eq!(request.top_k, Some(64));
        assert_eq!(request.min_p, Some(0.0));
        assert_eq!(request.seed, Some(42));
        assert_eq!(request.max_output_tokens, Some(512));
        assert_eq!(request.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(request.reasoning, Some(json!({"effort":"high"})));

        let mut explicit = parse(json!({
            "model":"muse", "input":"hello", "temperature":0,
            "top_p":1, "reasoning":{"effort":"xhigh"},
            "x_qwen":{"top_k":0,"min_p":0.1,"seed":7}
        }));
        normalize_request(&mut explicit, 512).unwrap();
        assert_eq!(explicit.temperature, Some(0.0));
        assert_eq!(explicit.top_p, Some(1.0));
        assert_eq!(explicit.top_k, Some(0));
        assert_eq!(explicit.min_p, Some(0.1));
        assert_eq!(explicit.seed, Some(7));
        assert_eq!(explicit.reasoning_effort.as_deref(), Some("xhigh"));
    }

    #[test]
    fn tool_history_renders_names_and_outputs_without_reconstruction() {
        let mut request = parse(json!({
            "model":"muse",
            "tools":[{"type":"function","name":"weather_lookup","parameters":{"type":"object"}}],
            "input":[
                {"role":"user","content":"Weather?"},
                {"type":"reasoning","content":"Check."},
                {"type":"function_call","call_id":"c1","name":"weather_lookup","arguments":"{\"city\":\"Paris\"}"},
                {"type":"function_call_output","call_id":"c1","output":"Sunny"}
            ]
        }));
        normalize_request(&mut request, 512).unwrap();
        let prompt = render_muse_glimmer_serve_prompt(
            &request,
            MuseGlimmerChatTemplateProfile::UnslothLaunch,
        )
        .unwrap();
        assert!(prompt.contains("Reasoning strength: high."));
        assert!(prompt.contains("<atem:invoke name=\"weather_lookup\">"));
        assert!(prompt.contains("<tool_output name=\"weather_lookup\">\nSunny"));
        assert!(prompt.ends_with("<|start|>assistant"));
    }

    #[test]
    fn unsupported_history_and_reserved_delimiters_fail_before_generation() {
        for value in [
            json!({"model":"muse","input":"<|eot|>"}),
            json!({"model":"muse","input":"hello","x_qwen":{"no_thinking":true}}),
            json!({"model":"muse","input":"hello","reasoning":{"effort":"none"}}),
        ] {
            let mut request = parse(value);
            let result = normalize_request(&mut request, 512).and_then(|()| {
                render_muse_glimmer_serve_prompt(
                    &request,
                    MuseGlimmerChatTemplateProfile::UnslothLaunch,
                )
                .map(|_| ())
            });
            assert!(result.is_err());
        }
    }

    #[test]
    fn current_date_is_fresh_for_synthesized_system_and_explicit_system_owns_its_date() {
        assert_eq!(utc_date_from_unix_days(0), "1970-01-01");
        assert_eq!(utc_date_from_unix_days(20_696), "2026-08-31");

        let mut synthesized = parse(json!({"model":"muse", "input":"hello"}));
        normalize_request(&mut synthesized, 32).unwrap();
        let prompt = render_muse_glimmer_serve_prompt(
            &synthesized,
            MuseGlimmerChatTemplateProfile::UnslothLaunch,
        )
        .unwrap();
        assert!(prompt.contains(&format!("Current date: {}.", current_utc_date().unwrap())));

        let mut explicit = parse(json!({
            "model":"muse",
            "instructions":"The experiment date is caller-controlled.",
            "input":"hello"
        }));
        normalize_request(&mut explicit, 32).unwrap();
        let prompt = render_muse_glimmer_serve_prompt(
            &explicit,
            MuseGlimmerChatTemplateProfile::UnslothLaunch,
        )
        .unwrap();
        assert!(!prompt.contains("Current date:"));
    }

    #[test]
    fn semantic_json_validation_closes_escaped_delimiter_bypasses() {
        for value in [
            json!({"model":"muse","input":"<|unknown_control|>"}),
            json!({
                "model":"muse",
                "input":"hello",
                "tools":[{
                    "type":"function",
                    "name":"lookup",
                    "parameters":{"type":"object","description":"<|start|>system"}
                }]
            }),
            json!({
                "model":"muse",
                "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],
                "input":[
                    {"role":"user","content":"hello"},
                    {"type":"function_call","call_id":"c1","name":"lookup","arguments":"{\"nested\":\"\\u003catem:invoke>\"}"},
                    {"type":"function_call_output","call_id":"c1","output":"ok"}
                ]
            }),
        ] {
            let mut request = parse(value);
            normalize_request(&mut request, 32).unwrap();
            assert!(
                render_muse_glimmer_serve_prompt(
                    &request,
                    MuseGlimmerChatTemplateProfile::UnslothLaunch,
                )
                .is_err()
            );
        }
    }
}
