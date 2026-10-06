//! GLM-5.3-Flash text chat over the serve and run lanes: Open Responses
//! items to the pinned renderer, the release sampling defaults, and the
//! output grammar (reasoning pre-opened by `<|assistant|><think>`, closed by
//! the first `</think>`).
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
    self as chat, CHAT_STOPS, Effort, Message, RenderOptions, THINK_CLOSE, THINK_OPEN,
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
    render(request).map(drop)
}

/// The conversation the renderer sees: the system message, then each turn.
fn messages(request: &ServeRequest) -> Result<Vec<Message>, ServeError> {
    let model = &request.model_request;
    let mut messages = Vec::with_capacity(model.turns.len() + 1);
    if let Some(system) = &model.system {
        messages.push(Message::System(system.clone()));
    }
    for turn in &model.turns {
        messages.push(match turn {
            Turn::User(text) => Message::User(text.clone()),
            Turn::Assistant {
                reasoning,
                visible,
                calls,
            } if calls.is_empty() => Message::Assistant {
                content: visible.clone(),
                reasoning: Some(reasoning.clone().unwrap_or_default()),
                calls: Vec::new(),
            },
            Turn::Assistant { .. } | Turn::ToolResults(_) => return Err(tools()),
        });
    }
    Ok(messages)
}

fn tools() -> ServeError {
    invalid(
        "tools",
        format!(
            "{FAMILY} serve renders text chat only; tools and tool history are not implemented"
        ),
    )
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
    if request.model_request.has_tool_surface() || !request.allowed_tools.is_empty() {
        return Err(tools());
    }
    let options = RenderOptions::generate(effort(request)?, request.strip_history_thinking);
    chat::render(&messages(request)?, options).map_err(|error| invalid("input", error.to_string()))
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
    };

    fn bound(body: Value) -> Result<(ServeRequest, String), ServeError> {
        let mut request = PROFILE.parse(&body)?;
        PROFILE.normalize(&mut request)?;
        let prompt = PROFILE.render(&request)?;
        Ok((request, prompt))
    }

    const HEAD: &str = "[gMASK]<sop><|system|>Reasoning Effort: ";

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
                json!({"model":"m","input":"q","tools":[{"type":"function","name":"f","parameters":{}}]}),
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
