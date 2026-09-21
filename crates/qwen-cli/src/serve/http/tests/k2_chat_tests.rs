use super::*;
use crate::serve::render_k2;
use qwen_llm::k2_horizon_chat::Effort;

struct K2Mock {
    profile: Option<render_k2::ChatCapability>,
    pieces: Vec<String>,
    end: GenerationEnd,
}
impl GenerationBackend for K2Mock {
    fn model_id(&self) -> &str {
        "k2-test"
    }
    fn parse_request(&self, body: &Value) -> Result<ServeRequest, ServeError> {
        render_k2::parse_with_profile(body, self.profile.as_ref())
    }
    fn normalize_request(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
        render_k2::normalize_with_profile(request, 8, 128, self.profile.as_ref())
    }
    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        render_k2::render_with_profile(request, self.profile.as_ref())
    }
    fn output_protocol(&self, request: &ServeRequest) -> OutputProtocol {
        match (&self.profile, &request.k2_chat) {
            (Some(_), Some(chat)) => OutputProtocol::K2Chat {
                effort: chat.effort,
            },
            _ => OutputProtocol::RawText,
        }
    }
    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        assert_eq!(prompt, self.render_prompt(request)?);
        for piece in &self.pieces {
            sink.piece(piece.as_bytes())
                .map_err(BackendFailure::Aborted)?;
        }
        Ok(GenerationOutcome {
            end: self.end,
            usage: Usage {
                input_tokens: 7,
                output_tokens: 3,
                cached_tokens: 0,
            },
            stats: None,
        })
    }
}

#[test]
fn k2_chat_json_sse_partition_and_empty_reasoning_replay() {
    for effort in [Effort::High, Effort::Medium, Effort::Low] {
        for reasoning in ["", "plan<think>literal"] {
            for streaming in [false, true] {
                let text = format!(
                    "<{}>{reasoning}</{}>answer<ifm|tool_calls>literal",
                    effort.tag(),
                    effort.tag()
                );
                let backend = K2Mock {
                    profile: Some(render_k2::mock_profile()),
                    pieces: text.chars().map(|c| c.to_string()).collect(),
                    end: GenerationEnd::StopToken(250019),
                };
                let body = json!({"model":"k2-test","input":[{"role":"user","content":"hi"}],"reasoning":{"effort":effort},"stream":streaming});
                let response = roundtrip(backend, &post("/v1/responses", &body.to_string()));
                assert!(response.starts_with("HTTP/1.1 200"), "{response}");
                let envelope = if streaming {
                    sse_payload(&response, "response.completed")["response"].clone()
                } else {
                    serde_json::from_str::<Value>(body_of(&response)).unwrap()
                };
                let output = envelope["output"].as_array().unwrap();
                assert_eq!(output.len(), 2);
                assert_eq!(output[0]["type"], "reasoning");
                assert_eq!(output[0]["content"][0]["text"], reasoning);
                assert_eq!(
                    output[1]["content"][0]["text"],
                    "answer<ifm|tool_calls>literal"
                );
                assert_eq!(envelope["reasoning"], json!({"effort":effort}));
                assert_eq!(envelope["tools"], json!([]));
                assert_eq!(envelope["tool_choice"], "none");
                let mut replay = output.clone();
                replay.push(json!({"role":"user","content":"next"}));
                render_k2::parse_with_profile(
                    &json!({"model":"k2-test","input":replay}),
                    Some(&render_k2::mock_profile()),
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn k2_chat_budget_and_malformed_stop_have_distinct_terminal_states() {
    for streaming in [false, true] {
        for end in [
            GenerationEnd::TokenLimit,
            GenerationEnd::StopToken(1),
            GenerationEnd::StopToken(250019),
        ] {
            let backend = K2Mock {
                profile: Some(render_k2::mock_profile()),
                pieces: vec!["plan</ifm|thi".into()],
                end,
            };
            let body = json!({"model":"k2-test","input":[{"role":"user","content":"hi"}],"stream":streaming});
            let response = roundtrip(backend, &post("/v1/responses", &body.to_string()));
            if !end.is_token_limit() && !streaming {
                assert!(response.starts_with("HTTP/1.1 500"));
                assert!(response.contains("stop before reasoning close"));
                continue;
            }
            let envelope = if streaming {
                sse_payload(
                    &response,
                    if end.is_token_limit() {
                        "response.incomplete"
                    } else {
                        "response.failed"
                    },
                )["response"]
                    .clone()
            } else {
                serde_json::from_str::<Value>(body_of(&response)).unwrap()
            };
            assert_eq!(
                envelope["status"],
                if end.is_token_limit() {
                    "incomplete"
                } else {
                    "failed"
                }
            );
            assert_eq!(envelope["output"].as_array().unwrap().len(), 1);
            assert_eq!(envelope["output"][0]["type"], "reasoning");
            assert_eq!(envelope["output"][0]["status"], "incomplete");
        }
    }
}

#[test]
fn k2_chat_missing_profile_refuses_before_generation() {
    let backend = K2Mock {
        profile: None,
        pieces: vec!["must not generate".into()],
        end: GenerationEnd::StopToken(1),
    };
    let response = roundtrip(
        backend,
        &post(
            "/v1/responses",
            &json!({"model":"k2-test","input":[{"role":"user","content":"hi"}]}).to_string(),
        ),
    );
    assert!(response.starts_with("HTTP/1.1 400"));
    assert!(response.contains("verified checkpoint profile"));
    assert!(!response.contains("must not generate"));
}
