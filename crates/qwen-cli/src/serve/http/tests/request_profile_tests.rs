use super::*;
use crate::serve::items::QwenTemplate;
use crate::serve::request_profile::RequestProfile;
use qwen_llm::muse_glimmer::MuseGlimmerChatTemplateProfile;
use std::sync::Arc;

#[derive(Clone)]
struct ProfileBackend {
    profile: RequestProfile,
    expected_prompt: Option<String>,
    output: String,
    end: GenerationEnd,
}

fn roundtrip(mut backend: ProfileBackend, request: &str) -> String {
    let direct = super::roundtrip(backend.clone(), request);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut activity = crate::serve::owner_activity::OwnerActivity::default();
        let guard = activity.admission().try_admit().unwrap();
        crate::serve::transport::handle_connection(stream, &mut backend, None, guard, || Ok(()))
            .unwrap();
        let mut completions = 0;
        activity.drain_finished(|| completions += 1);
        assert_eq!(completions, 1);
        assert!(activity.is_settled());
    });
    let mut client = TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    client.write_all(request.as_bytes()).unwrap();
    let mut bridged = String::new();
    client.read_to_string(&mut bridged).unwrap();
    server.join().unwrap();
    assert_eq!(regex_lite_replace(&direct), regex_lite_replace(&bridged));
    bridged
}

impl GenerationBackend for ProfileBackend {
    fn model_id(&self) -> &str {
        "test"
    }
    fn request_profile(&self) -> RequestProfile {
        self.profile.clone()
    }
    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        if let Some(expected) = &self.expected_prompt {
            assert_eq!(prompt, expected);
        }
        match &self.profile {
            RequestProfile::OrdinaryQwen { .. } => {
                assert_eq!(request.template, QwenTemplate::Generic)
            }
            RequestProfile::FlashNext { .. } => assert_eq!(request.template, QwenTemplate::Qwen38),
            RequestProfile::Muse {
                default_max_tokens, ..
            } => {
                assert_eq!(request.max_output_tokens, Some(*default_max_tokens));
                assert_eq!(request.top_k, Some(64));
                assert_eq!(request.reasoning_effort.as_deref(), Some("high"));
            }
            RequestProfile::K2 {
                default_max_tokens, ..
            } => assert_eq!(request.max_output_tokens, Some(*default_max_tokens)),
            RequestProfile::Glm5Next {
                default_max_tokens, ..
            } => {
                assert_eq!(request.max_output_tokens, Some(*default_max_tokens));
                assert_eq!((request.top_k, request.min_p), (Some(0), Some(0.0)));
                assert_eq!(request.seed, Some(42));
            }
            _ => {}
        }
        sink.tick().map_err(BackendFailure::Aborted)?;
        for byte in self.output.as_bytes() {
            sink.piece(&[*byte]).map_err(BackendFailure::Aborted)?;
        }
        Ok(GenerationOutcome {
            end: self.end,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 2,
                cached_tokens: 0,
            },
            stats: None,
        })
    }
}

fn envelope(response: &str, streaming: bool) -> Value {
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    if streaming {
        sse_payload(response, "response.completed")["response"].clone()
    } else {
        serde_json::from_str(body_of(response)).unwrap()
    }
}

fn k2(chat: bool, max_piece_bytes: usize) -> RequestProfile {
    RequestProfile::K2 {
        chat: chat.then(|| Arc::new(crate::serve::render_k2::mock_profile())),
        default_max_tokens: 8,
        capacity: 128,
        max_piece_bytes,
    }
}

#[test]
fn family_profiles_drive_real_http_json_and_sse_without_hook_overrides() {
    let qwen_prompt =
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let ds: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/deepseek_v4_0731_chat_fixtures_v1.json"
    ))
    .unwrap();
    let ds_prompt = ds["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["name"] == "chat_single_user")
        .unwrap()["prompt"]
        .as_str()
        .unwrap();
    let mut cases = Vec::new();
    for template in [
        QwenTemplate::Qwen35,
        QwenTemplate::Qwen36,
        QwenTemplate::Qwen38,
    ] {
        cases.push((
            RequestProfile::OrdinaryQwen {
                template,
                no_thinking_supported: true,
                style: TemplateStyle::House,
            },
            json!({"model":"test","input":"Hello","x_qwen":{"no_thinking":true}}),
            qwen_prompt.to_string(),
            "answer".to_string(),
            GenerationEnd::StopToken(0),
            false,
        ));
    }
    cases.push((
        RequestProfile::FlashNext {
            style: TemplateStyle::House,
        },
        json!({"model":"test","input":"Hello","x_qwen":{"no_thinking":true}}),
        qwen_prompt.into(),
        "answer".into(),
        GenerationEnd::StopToken(0),
        false,
    ));
    for style in [TemplateStyle::House, TemplateStyle::Upstream] {
        cases.push((
            RequestProfile::DeepSeekV4 { style },
            json!({"model":"test","input":"Hello"}),
            ds_prompt.into(),
            "answer".into(),
            GenerationEnd::StopToken(0),
            false,
        ));
    }
    cases.push((RequestProfile::Muse { template: MuseGlimmerChatTemplateProfile::UnslothLaunch, default_max_tokens: 8, eos_token_id: 1, eot_token_id: 2 }, json!({"model":"test","input":"Hello","instructions":"System"}), "<|begin_of_text|><|start|>system<|message|>System\n\nReasoning strength: high.\n\n# Valid recipients: \"self\", \"user\".<|eot|><|start|>user<|message|>Hello<|eot|><|start|>assistant".into(), " to=self<|message|>plan<|eom|><|start|>assistant to=user<|message|>answer".into(), GenerationEnd::StopToken(2), true));
    cases.push((
        k2(false, 128),
        json!({"model":"test","input":"raw bytes"}),
        "raw bytes".into(),
        "<think>literal</think>".into(),
        GenerationEnd::StopToken(1),
        false,
    ));
    cases.push((k2(true, 128), json!({"model":"test","input":[{"role":"user","content":"Hello"}],"reasoning":{"effort":"low"}}), "<|ifm|im_start|>user\nHello<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think_faster>\n".into(), "plan</ifm|think_faster>answer".into(), GenerationEnd::StopToken(1), true));
    let glm = RequestProfile::Glm5Next {
        default_max_tokens: 8,
        capacity: 128,
        max_piece_bytes: 64,
    };
    cases.push((
        glm.clone(),
        json!({"model":"test","input":"Hello","instructions":"System"}),
        "[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>System<|user|>Hello<|assistant|><think>".into(),
        "plan</think>answer".into(),
        GenerationEnd::StopToken(154_827),
        true,
    ));
    cases.push((
        glm,
        json!({"model":"test","input":[{"role":"user","content":"one"},{"type":"reasoning","content":"r1"},{"role":"assistant","content":"a1"},{"role":"user","content":"two"}],"reasoning":{"effort":"low"}}),
        "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>one<|assistant|><think>r1</think>a1<|user|>two<|assistant|><think>".into(),
        "plan</think>answer".into(),
        GenerationEnd::StopToken(154_820),
        true,
    ));
    for (profile, body, prompt, output, end, reasoning) in cases {
        for streaming in [false, true] {
            let mut body = body.clone();
            body["stream"] = streaming.into();
            let result = roundtrip(
                ProfileBackend {
                    profile: profile.clone(),
                    expected_prompt: Some(prompt.clone()),
                    output: output.clone(),
                    end,
                },
                &post("/v1/responses", &body.to_string()),
            );
            let reply = envelope(&result, streaming);
            let items = reply["output"].as_array().unwrap();
            assert_eq!(items.len(), if reasoning { 2 } else { 1 });
            if reasoning {
                assert_eq!(items[0]["type"], "reasoning");
                assert_eq!(items[0]["content"][0]["text"], "plan");
            }
            assert_eq!(
                items.last().unwrap()["content"][0]["text"],
                if matches!(profile, RequestProfile::K2 { chat: None, .. }) {
                    "<think>literal</think>"
                } else {
                    "answer"
                }
            );
            if matches!(profile, RequestProfile::Muse { .. }) {
                assert_eq!(reply["temperature"], 1.0);
                assert_eq!(reply["top_p"], 0.95);
                assert_eq!(reply["max_output_tokens"], 8);
                assert_eq!(reply["reasoning"], json!({"effort":"high"}));
            }
            if matches!(profile, RequestProfile::Glm5Next { .. }) {
                assert_eq!(reply["temperature"], 1.0);
                assert_eq!(reply["top_p"], 0.95);
                assert_eq!(reply["max_output_tokens"], 8);
                let effort = body["reasoning"]["effort"].as_str().unwrap_or("max");
                assert_eq!(reply["reasoning"], json!({"effort": effort}));
            }
        }
    }
}

#[test]
fn profiles_preserve_style_and_history_normalization_boundaries() {
    let body = json!({"model":"test","input":[{"role":"user","content":"one"},{"role":"assistant","content":"answer"},{"role":"user","content":"two"}]});
    for style in [TemplateStyle::House, TemplateStyle::Upstream] {
        let profile = RequestProfile::DeepSeekV4 { style };
        let mut request = profile.parse(&body).unwrap();
        assert_eq!(request.history_reasoning_missing, 1);
        request.template_style = profile.template_style_default();
        profile.normalize(&mut request).unwrap();
        assert_eq!(
            request.history_reasoning_missing,
            usize::from(style == TemplateStyle::Upstream)
        );
    }
    for profile in [
        RequestProfile::OrdinaryQwen {
            template: QwenTemplate::Qwen36,
            no_thinking_supported: true,
            style: TemplateStyle::Upstream,
        },
        RequestProfile::FlashNext {
            style: TemplateStyle::Upstream,
        },
    ] {
        assert_eq!(
            profile.template_style_default(),
            Some(TemplateStyle::Upstream)
        );
        let mut request = profile.parse(&body).unwrap();
        request.template_style = Some(TemplateStyle::Upstream);
        profile.normalize(&mut request).unwrap();
        let original = request.template;
        profile.render(&request).unwrap();
        assert_eq!(
            request.template, original,
            "rendering must not mutate echoed fields"
        );
    }
}

#[test]
fn profile_delegates_keep_k2_tool_lexical_values_and_output_events() {
    use qwen_llm::k2_horizon_chat::tools::{
        ToolCall, ToolCallFormat, decode_tool_json, render_tool_calls,
    };
    let arguments = decode_tool_json(r#"{"n":9007199254740993}"#).unwrap();
    let definitions = vec![
        json!({"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}),
    ];
    let block = render_tool_calls(
        &[ToolCall {
            name: "lookup".into(),
            arguments: arguments.as_object().unwrap().clone(),
        }],
        ToolCallFormat::Json,
        &definitions,
    )
    .unwrap();
    for streaming in [false, true] {
        let body = json!({"model":"test","stream":streaming,"input":[{"role":"user","content":"go"}],"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"x_k2":{"tool_call_format":"json"}});
        let result = roundtrip(
            ProfileBackend {
                profile: k2(true, 128),
                expected_prompt: None,
                output: format!("plan</ifm|think>{block}"),
                end: GenerationEnd::StopToken(250019),
            },
            &post("/v1/responses", &body.to_string()),
        );
        let reply = envelope(&result, streaming);
        assert_eq!(reply["x_k2"]["tool_call_format"], "json");
        let call = reply["output"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "function_call")
            .unwrap();
        assert_eq!(call["name"], "lookup");
        assert_eq!(
            decode_tool_json(call["arguments"].as_str().unwrap()).unwrap(),
            arguments
        );
        let replay = format!(
            r#"{{"model":"test","input":[{{"role":"user","content":"go"}},{{"type":"function_call","call_id":"c","name":"lookup","arguments":"{{}}"}},{{"type":"function_call_output","call_id":"c","output":{{"$serde_json::private::Number":"7","n":9007199254740993}}}}],"tools":[{{"type":"function","name":"lookup","parameters":{{}}}}]}}"#
        );
        let profile = k2(true, 128);
        let decoded = profile.decode(replay.as_bytes()).unwrap();
        assert!(decoded["input"][2]["output"].is_object());
        let request = profile.parse(&decoded).unwrap();
        let prompt = profile.render(&request).unwrap();
        assert!(prompt.contains("9007199254740993"));
        assert!(prompt.contains(r#""$serde_json::private::Number": "7""#));
    }
    for chat in [false, true] {
        assert!(
            k2(chat, 128)
                .decode(br#"{"model":"test","input":"a","input":"b"}"#)
                .is_err()
        );
    }
    let profile = k2(true, usize::MAX);
    let mut request = profile.parse(&json!({"model":"test","input":[{"role":"user","content":"go"}],"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]})).unwrap();
    assert!(
        profile.normalize(&mut request).is_err(),
        "K2 tool-byte budget must precede execution"
    );
}

#[test]
fn style_overrides_are_applied_or_refused_before_execution() {
    for (deployment, requested) in [
        (TemplateStyle::House, TemplateStyle::Upstream),
        (TemplateStyle::Upstream, TemplateStyle::House),
    ] {
        let body = json!({"model":"test","input":[{"role":"user","content":"one"},{"type":"reasoning","content":"plan"},{"role":"assistant","content":"answer"},{"role":"user","content":"two"}],"x_qwen":{"no_thinking":true,"template_style":requested.as_str()}});
        let history = if requested == TemplateStyle::House {
            "<think>\nplan\n</think>\n\nanswer"
        } else {
            "answer"
        };
        let prompt = format!(
            "<|im_start|>user\none<|im_end|>\n<|im_start|>assistant\n{history}<|im_end|>\n<|im_start|>user\ntwo<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let result = roundtrip(
            ProfileBackend {
                profile: RequestProfile::OrdinaryQwen {
                    template: QwenTemplate::Qwen36,
                    no_thinking_supported: true,
                    style: deployment,
                },
                expected_prompt: Some(prompt),
                output: "answer".into(),
                end: GenerationEnd::StopToken(0),
            },
            &post("/v1/responses", &body.to_string()),
        );
        let reply = envelope(&result, false);
        assert_eq!(reply["output"][0]["content"][0]["text"], "answer");
    }
    for profile in [
        k2(false, 128),
        RequestProfile::Muse {
            template: MuseGlimmerChatTemplateProfile::UnslothLaunch,
            default_max_tokens: 8,
            eos_token_id: 1,
            eot_token_id: 2,
        },
        RequestProfile::Glm5Next {
            default_max_tokens: 8,
            capacity: 128,
            max_piece_bytes: 64,
        },
    ] {
        let body = json!({"model":"test","input":"Hello","x_qwen":{"template_style":"upstream"}});
        let result = roundtrip(
            ProfileBackend {
                profile,
                expected_prompt: Some("must not generate".into()),
                output: String::new(),
                end: GenerationEnd::TokenLimit,
            },
            &post("/v1/responses", &body.to_string()),
        );
        assert!(result.starts_with("HTTP/1.1 400"), "{result}");
    }
}
