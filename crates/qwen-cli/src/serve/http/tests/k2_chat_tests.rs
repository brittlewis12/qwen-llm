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
    fn decode_request_json(&self, body: &[u8]) -> Result<Value, ServeError> {
        render_k2::tools::decode_request_json(body)
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
        if self.profile.is_some() && request.k2_tools.is_some() {
            return render_k2::tools::output_protocol(request, 128);
        }
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
fn k2_tools_json_sse_roundtrip_ids_modes_and_reordered_results() {
    use qwen_llm::k2_horizon_chat::tools::{
        ToolCall, ToolCallFormat, decode_tool_json, render_tool_calls,
    };
    for format in [
        ToolCallFormat::Xml,
        ToolCallFormat::Json,
        ToolCallFormat::XmlTyped,
    ] {
        for streaming in [false, true] {
            let defs = vec![
                json!({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"x":{"type":"integer"}}}}}),
            ];
            let calls = vec![
                ToolCall {
                    name: "lookup".into(),
                    arguments: json!({"x":1}).as_object().unwrap().clone(),
                },
                ToolCall {
                    name: "lookup".into(),
                    arguments: json!({"x":2}).as_object().unwrap().clone(),
                },
            ];
            let block = render_tool_calls(&calls, format, &defs).unwrap();
            let text = format!("plan</ifm|think>Checking. {block}");
            let backend = K2Mock {
                profile: Some(render_k2::mock_profile()),
                pieces: text.chars().map(|c| c.to_string()).collect(),
                end: GenerationEnd::StopToken(250019),
            };
            let body = json!({"model":"k2-test","input":[{"role":"user","content":"go"}],"tools":[{"type":"function","name":"lookup","parameters":defs[0]["function"]["parameters"]}],"x_k2":{"tool_call_format":format},"stream":streaming});
            let response = roundtrip(backend, &post("/v1/responses", &body.to_string()));
            assert!(response.starts_with("HTTP/1.1 200"), "{response}");
            let envelope = if streaming {
                sse_payload(&response, "response.completed")["response"].clone()
            } else {
                serde_json::from_str::<Value>(body_of(&response)).unwrap()
            };
            assert_eq!(envelope["x_k2"]["tool_call_format"], json!(format));
            assert_eq!(envelope["x_k2"]["tool_presentation_format"], "markdown");
            assert_eq!(envelope["tool_choice"], "auto");
            assert_eq!(envelope["parallel_tool_calls"], true);
            let items = envelope["output"].as_array().unwrap();
            let function_calls: Vec<_> = items
                .iter()
                .filter(|i| i["type"] == "function_call")
                .collect();
            assert_eq!(function_calls.len(), 2);
            assert_ne!(function_calls[0]["call_id"], function_calls[1]["call_id"]);
            for (index, item) in function_calls.iter().enumerate() {
                assert_eq!(item["status"], "completed");
                assert_eq!(
                    decode_tool_json(item["arguments"].as_str().unwrap()).unwrap(),
                    json!({"x":index+1})
                );
                if streaming {
                    let events: Vec<Value> = body_of(&response)
                        .split("\n\n")
                        .filter_map(|block| block.lines().find_map(|l| l.strip_prefix("data: ")))
                        .filter(|data| *data != "[DONE]")
                        .map(|data| serde_json::from_str(data).unwrap())
                        .collect();
                    for kind in ["response.output_item.added", "response.output_item.done"] {
                        let event = events
                            .iter()
                            .find(|e| e["type"] == kind && e["item"]["id"] == item["id"])
                            .unwrap();
                        assert_eq!(event["item"]["call_id"], item["call_id"]);
                    }
                }
            }
            let mut replay = body.clone();
            replay["input"]
                .as_array_mut()
                .unwrap()
                .extend(items.iter().cloned());
            for (index, content) in [(1, "second"), (0, "first")] {
                replay["input"].as_array_mut().unwrap().push(json!({"type":"function_call_output","call_id":function_calls[index]["call_id"],"output":content}));
            }
            let profile = render_k2::mock_profile();
            let request = render_k2::parse_with_profile(&replay, Some(&profile)).unwrap();
            let prompt = render_k2::render_with_profile(&request, Some(&profile)).unwrap();
            assert!(prompt.find("tool\nfirst").unwrap() < prompt.find("tool\nsecond").unwrap());
            let backend = K2Mock {
                profile: Some(profile),
                pieces: vec!["done</ifm|think>Final answer".into()],
                end: GenerationEnd::StopToken(1),
            };
            let result = roundtrip(backend, &post("/v1/responses", &replay.to_string()));
            assert!(result.starts_with("HTTP/1.1 200"), "{result}");
            let envelope = if streaming {
                sse_payload(&result, "response.completed")["response"].clone()
            } else {
                serde_json::from_str::<Value>(body_of(&result)).unwrap()
            };
            assert_eq!(envelope["output"][1]["content"][0]["text"], "Final answer");
        }
    }
}

/// Missing reasoning is empty reasoning on K2 too (serve's rule for every
/// family): history replayed without reasoning items renders the same native
/// bytes as history replaying explicit empty ones, on the chat and tools
/// paths, and the adapter reports how many turns it filled.
#[test]
fn k2_missing_history_reasoning_renders_as_explicit_empty_reasoning() {
    let user = |text: &str| json!({"role":"user","content":text});
    let assistant = |text: &str| json!({"role":"assistant","content":text});
    let empty = json!({"type":"reasoning","content":""});
    let call =
        |id: &str| json!({"type":"function_call","call_id":id,"name":"lookup","arguments":"{}"});
    let output = |id: &str| json!({"type":"function_call_output","call_id":id,"output":"ok"});
    let tools = json!([{"type":"function","name":"lookup","parameters":{"type":"object"}}]);
    let no_tools = json!([]);
    let cases = [
        // Empty `tools` falls back to the chat parser, which alone counts.
        (
            Some(&no_tools),
            json!([user("a"), assistant("x"), user("b")]),
            json!([user("a"), empty, assistant("x"), user("b")]),
            1,
        ),
        // (tools, missing history, explicit-empty history, missing count)
        (
            None,
            json!([user("a"), assistant("x"), user("b")]),
            json!([user("a"), empty, assistant("x"), user("b")]),
            1,
        ),
        // History may open with an assistant turn (the upstream
        // `history-missing` shape, which the native template refuses).
        (
            None,
            json!([assistant("x"), user("b")]),
            json!([empty, assistant("x"), user("b")]),
            1,
        ),
        // Parallel call group, then a final answer, both without reasoning.
        (
            Some(&tools),
            json!([
                user("a"),
                call("c1"),
                call("c2"),
                output("c2"),
                output("c1"),
                assistant("done"),
                user("b")
            ]),
            json!([
                user("a"),
                empty,
                call("c1"),
                call("c2"),
                output("c2"),
                output("c1"),
                empty,
                assistant("done"),
                user("b")
            ]),
            2,
        ),
        // A message and its calls are one turn.
        (
            Some(&tools),
            json!([user("a"), assistant("Checking."), call("c1"), output("c1")]),
            json!([
                user("a"),
                empty,
                assistant("Checking."),
                call("c1"),
                output("c1")
            ]),
            1,
        ),
    ];
    let profile = render_k2::mock_profile();
    for effort in ["high", "medium", "low"] {
        for (tools, missing, explicit, count) in &cases {
            let parse = |input: &Value| {
                let mut body =
                    json!({"model":"k2-test","input":input,"reasoning":{"effort":effort}});
                if let Some(tools) = tools {
                    body["tools"] = (*tools).clone();
                }
                let request = render_k2::parse_with_profile(&body, Some(&profile)).unwrap();
                let prompt = render_k2::render_with_profile(&request, Some(&profile)).unwrap();
                (prompt, request.history_reasoning_missing)
            };
            let (missing_prompt, missing_count) = parse(missing);
            let (explicit_prompt, explicit_count) = parse(explicit);
            assert_eq!(missing_prompt, explicit_prompt, "{effort} {missing}");
            assert_eq!(
                (missing_count, explicit_count),
                (*count, 0),
                "{effort} {missing}"
            );
        }
    }
}

/// The per-request diagnostic fires when a reasoning request's history was
/// missing reasoning items, and not for explicit empty reasoning or for a
/// no-thinking generation (where absent reasoning changes nothing).
#[test]
fn history_reasoning_diagnostic_fires_only_for_filled_reasoning_history() {
    use crate::serve::output_partition::ToolGrammar;
    let profile = render_k2::mock_profile();
    let user = |text: &str| json!({"role":"user","content":text});
    let call = json!({"type":"function_call","call_id":"c1","name":"lookup","arguments":"{}"});
    let output = json!({"type":"function_call_output","call_id":"c1","output":"ok"});
    let k2 = |input: Value| {
        let body = json!({"model":"k2-test","input":input,
            "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]});
        let request = render_k2::parse_with_profile(&body, Some(&profile)).unwrap();
        let protocol = render_k2::tools::output_protocol(&request, 128);
        assert!(matches!(protocol, OutputProtocol::K2Tools { .. }));
        history_reasoning_diagnostic(&request, &protocol)
    };
    let line = k2(json!([user("a"), call, output])).unwrap();
    assert!(
        line.starts_with("serve: history_reasoning_missing=1 "),
        "{line}"
    );
    assert_eq!(
        k2(json!([user("a"), {"type":"reasoning","content":""}, call, output])),
        None
    );

    let qwen = crate::open_responses::items::parse_request(&json!({"model":"m","input":[
        user("q"), {"role":"assistant","content":"a"}, user("q2")
    ]}))
    .unwrap();
    assert_eq!(qwen.history_reasoning_missing, 1);
    let protocol = |preopened_reasoning| OutputProtocol::Qwen {
        preopened_reasoning,
        parse_tools: true,
        tool_grammar: ToolGrammar::QwenXml,
    };
    assert!(history_reasoning_diagnostic(&qwen, &protocol(true)).is_some());
    assert_eq!(history_reasoning_diagnostic(&qwen, &protocol(false)), None);
}

#[test]
fn k2_tool_replay_preserves_typed_outputs_and_lossless_arguments_on_wire() {
    use qwen_llm::k2_horizon_chat::tools::{decode_tool_json, render_tool_result};
    let values = [
        r#"{"nested":[9007199254740993,1.0000000000000001]}"#,
        r#"[1,{"ok":true}]"#,
        r#"{"$serde_json::private::Number":"7"}"#,
    ];
    for streaming in [false, true] {
        for text in values {
            let output = decode_tool_json(text).unwrap();
            let arguments = r#"{"x":{"$serde_json::private::Number":"7"},"n":9007199254740993}"#;
            let body = json!({"model":"k2-test","stream":streaming,"tools":[{"type":"function","name":"f","parameters":{}}],"x_k2":{"tool_call_format":"json"},"input":[
                {"role":"user","content":"go"},
                {"type":"reasoning","content":"plan"},
                {"type":"function_call","call_id":"call_a","name":"f","arguments":arguments},
                {"type":"function_call_output","call_id":"call_a","output":output}
            ]});
            // Exercise the actual outer JSON decoder, not only a prebuilt Value.
            let wire_body = body.to_string();
            let decoded = render_k2::tools::decode_request_json(wire_body.as_bytes()).unwrap();
            assert_eq!(decoded["input"][3]["output"], output);
            let profile = render_k2::mock_profile();
            let request = render_k2::parse_with_profile(&decoded, Some(&profile)).unwrap();
            let prompt = render_k2::render_with_profile(&request, Some(&profile)).unwrap();
            assert!(prompt.contains(&render_tool_result(&output).unwrap()));
            assert!(prompt.contains(r#""x": {"$serde_json::private::Number": "7"}"#));
            assert!(prompt.contains("9007199254740993"));
            let result = roundtrip(
                K2Mock {
                    profile: Some(profile),
                    pieces: vec!["p</ifm|think>ok".into()],
                    end: GenerationEnd::StopToken(1),
                },
                &post("/v1/responses", &wire_body),
            );
            assert!(result.starts_with("HTTP/1.1 200"), "{result}");
            if streaming {
                assert!(result.contains("event: response.completed"));
            }
        }
    }
}

#[test]
fn k2_empty_tools_and_format_extensions_preserve_plain_chat_protocol() {
    for extra in [
        json!({"tools":[]}),
        json!({"x_k2":{"tool_call_format":"json"}}),
        json!({"tools":[],"x_k2":{"tool_presentation_format":"xml","tool_call_format":"xml_typed"}}),
    ] {
        for streaming in [false, true] {
            let mut body = json!({"model":"k2-test","stream":streaming,"input":[{"role":"user","content":"hi"}]});
            let profile = render_k2::mock_profile();
            let base = render_k2::parse_with_profile(&body, Some(&profile)).unwrap();
            let prompt = render_k2::render_with_profile(&base, Some(&profile)).unwrap();
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let request = render_k2::parse_with_profile(&body, Some(&profile)).unwrap();
            assert!(request.k2_tools.is_none());
            assert_eq!(
                render_k2::render_with_profile(&request, Some(&profile)).unwrap(),
                prompt
            );
            let result = roundtrip(
                K2Mock {
                    profile: Some(profile),
                    pieces: vec!["p</ifm|think><ifm|tool_calls>literal".into()],
                    end: GenerationEnd::StopToken(1),
                },
                &post("/v1/responses", &body.to_string()),
            );
            let envelope = if streaming {
                sse_payload(&result, "response.completed")["response"].clone()
            } else {
                serde_json::from_str::<Value>(body_of(&result)).unwrap()
            };
            assert!(envelope.get("x_k2").is_none());
            assert_eq!(envelope["tools"], json!([]));
            assert_eq!(
                envelope["output"][1]["content"][0]["text"],
                "<ifm|tool_calls>literal"
            );
        }
    }
}

#[test]
fn k2_tools_refuse_false_controls_and_incomplete_calls_never_publish() {
    let body = json!({"model":"k2-test","input":[{"role":"user","content":"go"}],"tools":[{"type":"function","name":"f","parameters":{}}]});
    let profile = render_k2::mock_profile();
    for (key, value) in [
        ("tool_choice", json!("required")),
        ("tool_choice", json!("none")),
        ("parallel_tool_calls", json!(false)),
        ("x_k2", json!({"tool_format":"json"})),
    ] {
        let mut bad = body.clone();
        bad[key] = value;
        assert!(render_k2::parse_with_profile(&bad, Some(&profile)).is_err());
    }
    assert!(render_k2::tools::byte_budget(usize::MAX, 2).is_err());
    assert!(render_k2::parse_with_profile(&body, None).is_err());
    for bad in [
        json!({"model":"k2-test","input":"raw","tools":[]}),
        json!({"model":"k2-test","input":[{"type":1,"role":"user","content":"go"}],"tools":[]}),
        json!({"model":"k2-test","input":[{"role":"user","content":"go"}],"tools":[{"type":"function","name":"f","parameters":{},"strict":true}]}),
    ] {
        assert!(render_k2::parse_with_profile(&bad, Some(&profile)).is_err());
    }
    for streaming in [false, true] {
        let mut body = body.clone();
        body["stream"] = json!(streaming);
        let backend = K2Mock {
            profile: Some(render_k2::mock_profile()),
            pieces: vec!["p</ifm|think><ifm|tool_calls><ifm|tool_call>f\n".into()],
            end: GenerationEnd::TokenLimit,
        };
        let result = roundtrip(backend, &post("/v1/responses", &body.to_string()));
        let envelope = if streaming {
            sse_payload(&result, "response.incomplete")["response"].clone()
        } else {
            serde_json::from_str::<Value>(body_of(&result)).unwrap()
        };
        assert_eq!(envelope["status"], "incomplete");
        assert!(
            envelope["output"]
                .as_array()
                .unwrap()
                .iter()
                .all(|i| i["type"] != "function_call")
        );
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
