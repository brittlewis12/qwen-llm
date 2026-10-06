//! GLM-5.3-Flash chat and tools over the serve and run lanes: Open Responses
//! items to the pinned renderer, the release sampling defaults, and the
//! output grammar (reasoning pre-opened by `<|assistant|><think>`, closed by
//! the first `</think>`; with declared tools, `<tool_call>` blocks after it).
//!
//! Tools: function definitions render in their declared order; a replayed
//! `function_call` becomes the assistant turn's call (arguments are a JSON
//! object), and `function_call_output` items become tool results, rendered
//! in call order under one `<|observation|>`.
//!
//! Serve renders past turns as they were generated: an assistant turn that
//! arrives without a reasoning item renders as empty reasoning (serve's rule
//! for every family), never through the template's inline `</think>` split.
//! The template keeps history reasoning by default; `x_qwen.history_thinking:
//! "strip"` is its own `clear_thinking`.
use super::items::{ServeError, ServeRequest};
use super::partition_preopened::{PreopenedGrammar, PreopenedPartition};
use crate::model_request::Turn;
use qwen_llm::glm5_next_chat::{
    self as chat, CHAT_STOPS, Effort, Message, RenderOptions, THINK_CLOSE, THINK_OPEN, ToolCall,
    ToolDefinition,
};
use qwen_llm::sampling::SamplingConfig;
use serde_json::json;

pub(crate) const FAMILY: &str = "GLM-5.3-Flash";

fn invalid(param: &'static str, message: impl Into<String>) -> ServeError {
    ServeError::invalid_request(Some(param), message)
}

pub(crate) fn partition() -> PreopenedPartition {
    PreopenedPartition::new(PreopenedGrammar {
        family: FAMILY,
        open: THINK_OPEN.into(),
        closes: &[THINK_CLOSE],
        stops: &CHAT_STOPS,
    })
}

pub(crate) fn effort(request: &ServeRequest) -> Result<Effort, ServeError> {
    Effort::parse(request.reasoning_effort.as_deref())
        .map_err(|error| invalid("reasoning.effort", error.to_string()))
}

pub(crate) fn sampling(request: &ServeRequest) -> SamplingConfig {
    let defaults = SamplingConfig::glm5_next(42);
    SamplingConfig {
        temperature: request.temperature.unwrap_or(defaults.temperature),
        top_k: request.top_k.unwrap_or(defaults.top_k),
        top_p: request.top_p.unwrap_or(defaults.top_p),
        min_p: request.min_p.unwrap_or(defaults.min_p),
        seed: request.seed.unwrap_or(defaults.seed),
    }
}

/// Release sampling defaults, the output budget, and the normalized effort
/// echo; the rendering checks run here too, so a refusal precedes any
/// response bytes.
pub(crate) fn normalize_request(
    request: &mut ServeRequest,
    default_max_tokens: usize,
    capacity: usize,
    max_piece_bytes: usize,
) -> Result<(), ServeError> {
    let defaults = SamplingConfig::glm5_next(42);
    if request.temperature.is_none() {
        request.temperature = Some(defaults.temperature);
        request.temperature_echo = Some(f64::from(defaults.temperature));
    }
    if request.top_p.is_none() {
        request.top_p = Some(defaults.top_p);
        request.top_p_echo = Some(0.95);
    }
    request.top_k.get_or_insert(defaults.top_k);
    request.min_p.get_or_insert(defaults.min_p);
    request.seed.get_or_insert(defaults.seed);
    let maximum = *request.max_output_tokens.get_or_insert(default_max_tokens);
    if maximum == 0 || maximum > capacity {
        return Err(invalid(
            "max_output_tokens",
            format!("{FAMILY} max_output_tokens must be in 1..={capacity}"),
        ));
    }
    sampling(request)
        .validate()
        .map_err(|error| invalid("temperature", format!("{FAMILY} sampling: {error}")))?;
    let effort = effort(request)?;
    request.reasoning_effort = Some(effort.as_str().into());
    request.reasoning = Some(json!({"effort": effort.as_str()}));
    if !request.model_request.tools.is_empty() {
        tool_byte_budget(maximum, max_piece_bytes)?;
    }
    render(request).map(drop)
}

/// The buffered tool block's bound: every output token at its longest
/// decoded piece, times 3 for UTF-8 replacement expansion.
pub(crate) fn tool_byte_budget(
    max_tokens: usize,
    max_piece_bytes: usize,
) -> Result<usize, ServeError> {
    max_tokens
        .checked_mul(max_piece_bytes)
        .and_then(|n| n.checked_mul(3))
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            invalid(
                "max_output_tokens",
                format!("{FAMILY} decoded output byte bound overflow or zero"),
            )
        })
}

/// Text chat, or the tools grammar when the request declares functions.
pub(crate) fn output_protocol(
    request: &ServeRequest,
    max_piece_bytes: usize,
) -> crate::serve::output_partition::OutputProtocol {
    use crate::serve::output_partition::OutputProtocol;
    match definitions(request) {
        Ok(definitions) if !definitions.is_empty() => OutputProtocol::Glm5NextTools {
            definitions,
            // Normalization admitted this bound before any response bytes.
            max_bytes: tool_byte_budget(request.max_output_tokens.unwrap_or(1), max_piece_bytes)
                .unwrap_or(usize::MAX),
        },
        _ => OutputProtocol::Glm5NextChat,
    }
}

/// GLM's template prints every key of a function object except `strict`
/// and `defer_loading`, and hides deferred functions. The shared parser keeps
/// only name, description, parameters and strict, so refuse anything it
/// would drop before parsing: `defer_loading: true`, and any other key.
pub(crate) fn check_raw_tool_definitions(body: &serde_json::Value) -> Result<(), ServeError> {
    let Some(tools) = body.get("tools").and_then(serde_json::Value::as_array) else {
        return Ok(());
    };
    for (index, tool) in tools.iter().enumerate() {
        let Some(map) = tool.as_object() else {
            continue;
        };
        for (key, value) in map {
            match key.as_str() {
                "type" | "name" | "description" | "parameters" | "strict" => {}
                "defer_loading"
                    if matches!(
                        value,
                        serde_json::Value::Null | serde_json::Value::Bool(false)
                    ) => {}
                "defer_loading" => {
                    return Err(invalid(
                        "tools",
                        format!(
                            "tool {index}: {FAMILY} cannot honor defer_loading (deferred functions are hidden from the model but callable)"
                        ),
                    ));
                }
                other => {
                    return Err(invalid(
                        "tools",
                        format!(
                            "tool {index}: {FAMILY} renders function objects verbatim and does not support key {other:?}"
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// The declared functions, in request order.
fn definitions(request: &ServeRequest) -> Result<Vec<ToolDefinition>, ServeError> {
    request
        .model_request
        .tools
        .iter()
        .map(|tool| {
            if tool.strict == Some(true) {
                return Err(invalid(
                    "tools",
                    format!("{FAMILY} cannot honor strict tool schemas"),
                ));
            }
            ToolDefinition::from_parts(
                &tool.name,
                tool.description.as_deref(),
                Some(&tool.parameters),
            )
            .map_err(|error| invalid("tools", error.to_string()))
        })
        .collect()
}

/// The conversation the renderer sees: the system message, then each turn.
fn messages(request: &ServeRequest) -> Result<Vec<Message>, ServeError> {
    let model = &request.model_request;
    let mut messages = Vec::with_capacity(model.turns.len() + 1);
    if let Some(system) = &model.system {
        messages.push(Message::System(system.clone()));
    }
    for turn in &model.turns {
        match turn {
            Turn::User(text) => messages.push(Message::User(text.clone())),
            Turn::Assistant {
                reasoning,
                visible,
                calls,
            } => {
                let calls = calls
                    .iter()
                    .map(|call| {
                        // Lossless: containers kept, duplicate keys refused.
                        let arguments = qwen_llm::tool_schema::decode_json(&call.arguments)
                            .ok()
                            .and_then(|value| value.as_object().cloned())
                            .ok_or_else(|| {
                                invalid(
                                    "input",
                                    format!(
                                        "{FAMILY} function_call {:?} arguments must be a JSON object without duplicate keys",
                                        call.call_id
                                    ),
                                )
                            })?;
                        Ok(ToolCall {
                            id: call.call_id.clone(),
                            name: call.name.clone(),
                            arguments,
                        })
                    })
                    .collect::<Result<Vec<_>, ServeError>>()?;
                messages.push(Message::Assistant {
                    content: visible.clone(),
                    reasoning: Some(reasoning.clone().unwrap_or_default()),
                    calls,
                });
            }
            Turn::ToolResults(results) => {
                for result in results {
                    messages.push(Message::Tool {
                        call_id: result.call_id.clone(),
                        content: result.output.clone(),
                    });
                }
            }
        }
    }
    Ok(messages)
}

pub(crate) fn render(request: &ServeRequest) -> Result<String, ServeError> {
    if request.no_thinking {
        return Err(invalid(
            "x_qwen.no_thinking",
            format!(
                "{FAMILY} always reasons (its template opens <think>); use reasoning.effort low"
            ),
        ));
    }
    let definitions = definitions(request)?;
    let options = RenderOptions::generate(effort(request)?, request.strip_history_thinking);
    chat::render_with_tools(&messages(request)?, &definitions, options).map_err(|error| {
        let param = if error.code() == "glm5_next_chat_tools" {
            "tools"
        } else {
            "input"
        };
        invalid(param, error.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::output_partition::GenerationEnd;
    use crate::serve::partition::PartitionEvent;

    fn collect(events: &[PartitionEvent]) -> (String, String, usize) {
        let (mut r, mut v, mut close) = (String::new(), String::new(), 0);
        for event in events {
            match event {
                PartitionEvent::Reasoning(t) => r.push_str(t),
                PartitionEvent::Visible(t) => v.push_str(t),
                PartitionEvent::ReasoningClosed => close += 1,
                PartitionEvent::FunctionCall(_) => panic!("no-tools grammar parsed a call"),
            }
        }
        (r, v, close)
    }

    use crate::serve::request_profile::RequestProfile;
    use serde_json::{Value, json};

    const PROFILE: RequestProfile = RequestProfile::Glm5Next {
        default_max_tokens: 8,
        capacity: 64,
        max_piece_bytes: 64,
    };

    fn bound(body: Value) -> Result<(ServeRequest, String), ServeError> {
        let mut request = PROFILE.parse(&body)?;
        PROFILE.normalize(&mut request)?;
        let prompt = PROFILE.render(&request)?;
        Ok((request, prompt))
    }

    const HEAD: &str = "[gMASK]<sop><|system|>Reasoning Effort: ";

    /// Open Responses tools and tool history render exactly as the native
    /// document does (the pinned `tool-call-content-and-reasoning` fixture
    /// case): definitions in order, the replayed call with its JSON-object
    /// arguments, and the output under `<|observation|>`. The output grammar
    /// switches to tools.
    #[test]
    fn tools_and_tool_history_render_as_the_pinned_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../qwen-llm/tests/fixtures/glm53_chat_hf.json"
        ))
        .unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "tool-call-content-and-reasoning")
            .unwrap();
        let function = &case["tools"][0]["function"];
        let body = json!({"model":"m","tools":[{"type":"function","name":function["name"],
            "description":function["description"],"parameters":function["parameters"]}],
            "input":[
                {"role":"user","content":"What's the weather in Paris?"},
                {"type":"reasoning","content":"need the weather"},
                {"type":"message","role":"assistant","content":" Let me check. \n"},
                {"type":"function_call","call_id":"c1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"},
                {"type":"function_call_output","call_id":"c1","output":"18C, clear"}]});
        let (request, prompt) = bound(body).unwrap();
        assert_eq!(prompt, case["rendered"].as_str().unwrap());
        let crate::serve::output_partition::OutputProtocol::Glm5NextTools {
            definitions,
            max_bytes,
        } = PROFILE.output(&request)
        else {
            panic!("declared tools select the tools grammar");
        };
        assert_eq!(definitions.len(), 1);
        assert_eq!(max_bytes, 8 * 64 * 3);
        // Duplicate keys are refused, not collapsed.
        let error = bound(json!({"model":"m","tools":[{"type":"function","name":"get_weather"}],
            "input":[{"role":"user","content":"q"},
                {"type":"function_call","call_id":"c1","name":"get_weather","arguments":"{\"a\":1,\"a\":2}"},
                {"type":"function_call_output","call_id":"c1","output":"r"}]}))
        .unwrap_err();
        assert_eq!(error.status, 400, "{error:?}");
        // Arguments that are not a JSON object are refused before execution.
        let error = bound(
            json!({"model":"m","tools":[{"type":"function","name":"get_weather"}],
            "input":[{"role":"user","content":"q"},
                {"type":"function_call","call_id":"c1","name":"get_weather","arguments":"[1]"},
                {"type":"function_call_output","call_id":"c1","output":"r"}]}),
        )
        .unwrap_err();
        assert_eq!(error.status, 400, "{error:?}");
    }

    /// Live-session reuse after a tool call is an exact prefix property: a
    /// call the model wrote in `tojson`'s canonical form re-renders as the
    /// same bytes, so the next prompt extends prompt + generated text; a
    /// noncanonical but valid call (`["a","b"]`) is accepted, re-renders
    /// canonically, and the next prompt does not extend it, so serve's exact
    /// prefix check declines reuse and prefills fresh rather than diverge.
    #[test]
    fn tool_history_extends_only_canonical_generated_calls() {
        use crate::serve::output_partition::{GenerationEnd, OutputPartition};
        let tools = json!([{"type":"function","name":"tag",
            "parameters":{"type":"object","properties":{"tags":{"type":"array"}}}}]);
        let (request, prompt) =
            bound(json!({"model":"m","input":"q","tools":tools,"reasoning":{"effort":"low"}}))
                .unwrap();
        for (generated_value, extends) in [(r#"["a", "b"]"#, true), (r#"["a","b"]"#, false)] {
            let generated = format!(
                "</think><tool_call>tag<arg_key>tags</arg_key><arg_value>{generated_value}</arg_value></tool_call>"
            );
            let mut partition = OutputPartition::new(PROFILE.output(&request));
            let mut events = Vec::new();
            partition.push(generated.as_bytes(), &mut events);
            partition
                .finish(GenerationEnd::StopToken(154_829), &mut events)
                .unwrap();
            let call = events
                .iter()
                .find_map(|e| match e {
                    PartitionEvent::FunctionCall(call) => Some(call.clone()),
                    _ => None,
                })
                .expect("a parsed call");
            assert_eq!(
                serde_json::Value::Object(call.arguments.clone()),
                json!({"tags":["a","b"]})
            );
            let arguments = serde_json::to_string(&call.arguments).unwrap();
            let (_, next) = bound(
                json!({"model":"m","tools":tools,"reasoning":{"effort":"low"},
                "input":[{"role":"user","content":"q"},
                    {"type":"reasoning","content":""},
                    {"type":"function_call","call_id":"c1","name":"tag","arguments":arguments},
                    {"type":"function_call_output","call_id":"c1","output":"ok"}]}),
            )
            .unwrap();
            assert_eq!(
                next.starts_with(&format!("{prompt}{generated}")),
                extends,
                "{generated_value}: next prompt {next:?}"
            );
        }
    }

    /// Generated output with a call: reasoning, visible text, then the call
    /// as a FunctionCall event with schema-typed arguments; a tool marker
    /// inside reasoning stays reasoning.
    #[test]
    fn generated_tool_calls_become_function_call_events() {
        use crate::serve::output_partition::{GenerationEnd, OutputPartition};
        let (request, _) = bound(json!({"model":"m","input":"q","tools":[{"type":"function",
            "name":"get_weather","parameters":{"type":"object","properties":{
                "city":{"type":"string"},"days":{"type":"integer"}}}}]}))
        .unwrap();
        let output = "plan <tool_call>not yet</think>Checking.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
        let mut partition = OutputPartition::new(PROFILE.output(&request));
        let mut events = Vec::new();
        for byte in output.as_bytes().chunks(3) {
            partition.push(byte, &mut events);
        }
        partition
            .finish(GenerationEnd::StopToken(154_829), &mut events)
            .unwrap();
        let mut reasoning = String::new();
        let mut visible = String::new();
        let mut calls = Vec::new();
        for event in events {
            match event {
                PartitionEvent::Reasoning(text) => reasoning.push_str(&text),
                PartitionEvent::Visible(text) => visible.push_str(&text),
                PartitionEvent::FunctionCall(call) => calls.push(call),
                PartitionEvent::ReasoningClosed => {}
            }
        }
        assert_eq!(reasoning, "plan <tool_call>not yet");
        assert_eq!(visible, "Checking.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            serde_json::Value::Object(calls[0].arguments.clone()),
            json!({"city":"Paris","days":3})
        );
    }

    #[test]
    fn requests_render_through_the_pinned_template_with_release_defaults() {
        let (request, prompt) = bound(json!({"model":"m","input":"Hello"})).unwrap();
        assert_eq!(
            prompt,
            format!("{HEAD}Max<|user|>Hello<|assistant|><think>")
        );
        assert_eq!(request.reasoning, Some(json!({"effort":"max"})));
        assert_eq!(
            sampling(&request),
            SamplingConfig::glm5_next(42),
            "release defaults"
        );
        assert_eq!(request.max_output_tokens, Some(8));
        for (effort, label) in [("low", "Low"), ("high", "High"), ("max", "Max")] {
            let (request, prompt) = bound(
                json!({"model":"m","input":"q","instructions":"S","reasoning":{"effort":effort}}),
            )
            .unwrap();
            assert_eq!(
                prompt,
                format!("{HEAD}{label}<|system|>S<|user|>q<|assistant|><think>")
            );
            assert_eq!(request.reasoning_effort.as_deref(), Some(effort));
        }
        let history = |extra: Value| {
            let mut body = json!({"model":"m","input":[
                {"role":"user","content":"one"},
                {"type":"reasoning","content":"r1"},
                {"role":"assistant","content":"a1"},
                {"role":"user","content":"two"},
                {"role":"assistant","content":"tag </think> stays visible"},
                {"role":"user","content":"three"}]});
            body["x_qwen"] = extra;
            bound(body).unwrap()
        };
        // Missing reasoning renders as empty, never through the inline split.
        let (request, prompt) = history(json!({}));
        assert_eq!(
            prompt,
            format!(
                "{HEAD}Max<|user|>one<|assistant|><think>r1</think>a1<|user|>two<|assistant|><think></think>tag </think> stays visible<|user|>three<|assistant|><think>"
            )
        );
        assert_eq!(request.history_reasoning_missing, 1);
        assert_eq!(history(json!({"history_thinking":"preserve"})).1, prompt);
        // strip is the template's clear_thinking.
        assert_eq!(
            history(json!({"history_thinking":"strip"})).1,
            format!(
                "{HEAD}Max<|user|>one<|assistant|><think></think>a1<|user|>two<|assistant|><think></think>tag </think> stays visible<|user|>three<|assistant|><think>"
            )
        );
        // Explicit sampling overrides and a no-op thinking flag.
        let (request, _) = bound(json!({"model":"m","input":"q","temperature":0,"top_p":1,
            "max_output_tokens":64,"x_qwen":{"top_k":5,"seed":7,"thinking":true,"no_thinking":false}}))
        .unwrap();
        assert_eq!(
            sampling(&request),
            SamplingConfig {
                temperature: 0.0,
                top_k: 5,
                top_p: 1.0,
                min_p: 0.0,
                seed: 7
            }
        );
    }

    #[test]
    fn unsupported_controls_are_refused_before_execution_with_their_parameter() {
        for (body, param) in [
            (
                json!({"model":"m","input":"q","reasoning":{"effort":"medium"}}),
                "reasoning.effort",
            ),
            (
                json!({"model":"m","input":"q","reasoning":{"effort":"none"}}),
                "reasoning.effort",
            ),
            (
                json!({"model":"m","input":"q","x_qwen":{"no_thinking":true}}),
                "x_qwen.no_thinking",
            ),
            (
                json!({"model":"m","input":"q","max_output_tokens":65}),
                "max_output_tokens",
            ),
            (
                json!({"model":"m","input":"q","tools":[{"type":"function","name":"f","strict":true,"parameters":{}}]}),
                "tools",
            ),
            (
                json!({"model":"m","input":"q","tools":[{"type":"function","name":"f","defer_loading":true}]}),
                "tools",
            ),
            (
                json!({"model":"m","input":"q","tools":[{"type":"function","name":"f","examples":[]}]}),
                "tools",
            ),
            // A schema that could not type generated arguments.
            (
                json!({"model":"m","input":"q","tools":[{"type":"function","name":"f",
                    "parameters":{"$ref":"#/$defs/A","$defs":{"A":{"$ref":"#/$defs/Gone"}}}}]}),
                "tools",
            ),
            (
                json!({"model":"m","input":[{"role":"user","content":"q"},
                    {"type":"function_call","call_id":"c","name":"f","arguments":"{}"},
                    {"type":"function_call_output","call_id":"c","output":"r"}]}),
                "tools",
            ),
        ] {
            let error = bound(body.clone()).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(param), "{body}: {error:?}");
            assert_eq!(error.status, 400);
        }
        // A conversation must end in a user turn.
        let error = bound(json!({"model":"m","input":[{"role":"user","content":"q"},
            {"role":"assistant","content":"a"}]}))
        .unwrap_err();
        assert_eq!(error.status, 400, "{error:?}");
        assert_eq!(PROFILE.template_style_default(), None);
        assert_eq!(
            PROFILE.output(&bound(json!({"model":"m","input":"q"})).unwrap().0),
            crate::serve::output_partition::OutputProtocol::Glm5NextChat
        );
    }

    #[test]
    fn every_byte_split_closes_once_and_keeps_later_tags_visible() {
        for opener in ["", "<think>"] {
            for reason in ["", "plan \u{1f389} <|user|> <think>nested"] {
                let visible = "answer \u{2192}</think><think>x</think><tool_call>t</tool_call>";
                let all = format!("{opener}{reason}</think>{visible}");
                for split in 0..=all.len() {
                    for stop in CHAT_STOPS {
                        let mut p = partition();
                        let mut events = Vec::new();
                        p.push(&all.as_bytes()[..split], &mut events);
                        p.push(&all.as_bytes()[split..], &mut events);
                        assert!(p.closed());
                        p.finish(GenerationEnd::StopToken(stop), &mut events)
                            .unwrap();
                        assert_eq!(collect(&events), (reason.into(), visible.into(), 1));
                        assert!(matches!(events.first(), Some(PartitionEvent::Reasoning(_))));
                    }
                }
            }
        }
    }

    #[test]
    fn truncated_reasoning_is_incomplete_and_bad_stops_fail() {
        for text in ["", "p</thi", "<thi", "p</think_x>", "<think>"] {
            for end in [
                GenerationEnd::TokenLimit,
                GenerationEnd::StopToken(154_820),
                GenerationEnd::StopToken(154_827),
            ] {
                let mut p = partition();
                let mut events = Vec::new();
                for byte in text.as_bytes() {
                    p.push(&[*byte], &mut events);
                }
                let result = p.finish(end, &mut events);
                assert_eq!(result.is_ok(), end.is_token_limit(), "{text:?} {end:?}");
                if let Err(error) = result {
                    assert!(error.message.contains("GLM-5.3-Flash"), "{}", error.message);
                }
                assert!(collect(&events).1.is_empty());
                assert_eq!(collect(&events).2, 0);
            }
        }
        let mut p = partition();
        let mut events = Vec::new();
        p.push(b"a</think>x", &mut events);
        // <|assistant|> is not a released stop.
        assert!(
            p.finish(GenerationEnd::StopToken(154_828), &mut events)
                .is_err()
        );
    }
}
