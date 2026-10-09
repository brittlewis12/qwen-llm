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
            RequestProfile::FlashNext { .. } => {
                assert_eq!(request.template, QwenTemplate::Qwen38);
                // Normalization resolved every sampling field before generation.
                assert!(request.temperature.is_some() && request.top_p.is_some());
                assert!(
                    request.top_k.is_some() && request.min_p.is_some() && request.seed.is_some()
                );
            }
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
                assert!(request.seed.is_some(), "a fresh seed is drawn");
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
    // Served Qwen models carry their release's sampling decision; 0.6 is the
    // Qwen3.5-27B/122B-A10B case.
    for (template, temperature) in [
        (QwenTemplate::Qwen35, 0.6),
        (QwenTemplate::Qwen36, 1.0),
        (QwenTemplate::Qwen38, 1.0),
    ] {
        cases.push((
            RequestProfile::OrdinaryQwen {
                template,
                no_thinking_supported: true,
                style: TemplateStyle::House,
                sampling: Some(qwen_llm::sampling::SamplingConfig::qwen3_release(
                    temperature,
                    42,
                )),
                limits: crate::serve::request_profile::OutputLimits::TEST,
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
            limits: crate::serve::request_profile::OutputLimits::TEST,
        },
        json!({"model":"test","input":"Hello","x_qwen":{"no_thinking":true}}),
        qwen_prompt.into(),
        "answer".into(),
        GenerationEnd::StopToken(0),
        false,
    ));
    for style in [TemplateStyle::House, TemplateStyle::Upstream] {
        cases.push((
            RequestProfile::DeepSeekV4 {
                style,
                sampling: Some(qwen_llm::sampling::SamplingConfig::deepseek_v4_0731(42)),
                limits: crate::serve::request_profile::OutputLimits::TEST,
            },
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
            // Release sampling even with no_thinking: one preset at every effort.
            if matches!(profile, RequestProfile::FlashNext { .. }) {
                assert_eq!(reply["temperature"], 1.0);
                assert_eq!(reply["top_p"], 0.95);
            }
            // Omitted sampling echoes the release decision as sampled.
            if let RequestProfile::OrdinaryQwen {
                sampling: Some(release),
                ..
            }
            | RequestProfile::DeepSeekV4 {
                sampling: Some(release),
                ..
            } = &profile
            {
                let decimal = crate::release_sampling::decimal;
                assert_eq!(reply["temperature"], decimal(release.temperature));
                assert_eq!(reply["top_p"], decimal(release.top_p));
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

/// Served models sample with their release's defaults when a request omits
/// sampling (Flash-Next at every effort, no_thinking included), echo what was
/// sampled, and keep every explicit field on its own, greedy included. A
/// profile without a release decision keeps the legacy greedy fallbacks.
#[test]
fn served_families_default_to_release_sampling_and_keep_explicit_fields() {
    use crate::serve::backend::release_request_sampler;
    use crate::serve::request_profile::{flash_next_release, sampling_with_defaults};
    use qwen_llm::sampling::SamplingConfig;
    let qwen35_27b = SamplingConfig::qwen3_release(0.6, 42);
    let ds4 = SamplingConfig::deepseek_v4_0731(42);
    let profiles = [
        (
            RequestProfile::FlashNext {
                style: TemplateStyle::House,
                limits: crate::serve::request_profile::OutputLimits::TEST,
            },
            SamplingConfig::qwen38_flash_next(42),
        ),
        (
            RequestProfile::OrdinaryQwen {
                template: QwenTemplate::Qwen35,
                no_thinking_supported: true,
                style: TemplateStyle::House,
                sampling: Some(qwen35_27b),
                limits: crate::serve::request_profile::OutputLimits::TEST,
            },
            qwen35_27b,
        ),
        (
            RequestProfile::DeepSeekV4 {
                style: TemplateStyle::House,
                sampling: Some(ds4),
                limits: crate::serve::request_profile::OutputLimits::TEST,
            },
            ds4,
        ),
    ];
    assert_eq!(flash_next_release(), SamplingConfig::qwen38_flash_next(42));
    for (profile, release) in &profiles {
        for body in [
            json!({"model":"test","input":"Hello"}),
            json!({"model":"test","input":"Hello","x_qwen":{"no_thinking":true}}),
            json!({"model":"test","input":"Hello","reasoning":{"effort":"low"}}),
        ] {
            if matches!(profile, RequestProfile::DeepSeekV4 { .. }) && body["x_qwen"].is_object() {
                continue; // DS4 selects tiers through reasoning.effort only.
            }
            let mut request = profile.parse(&body).unwrap();
            if profile.normalize(&mut request).is_err() {
                continue; // A family that refuses this effort is not under test here.
            }
            // An omitted seed is a fresh draw per request, not a constant.
            let mut again = profile.parse(&body).unwrap();
            profile.normalize(&mut again).unwrap();
            let seed = request.seed.expect("normalization draws a seed");
            assert_ne!(Some(seed), again.seed, "{body}");
            let release = &SamplingConfig { seed, ..*release };
            assert_eq!(
                sampling_with_defaults(&request, *release),
                *release,
                "{body}"
            );
            let decimal = crate::release_sampling::decimal;
            assert_eq!(
                (request.temperature_echo, request.top_p_echo),
                (
                    Some(decimal(release.temperature)),
                    Some(decimal(release.top_p))
                ),
                "{body}"
            );
            assert_eq!(
                release_request_sampler(&request, Some(*release))
                    .unwrap()
                    .config(),
                *release
            );
        }

        let mut request = profile
            .parse(
                &json!({"model":"test","input":"Hello","temperature":0.0,"top_p":0.8,
                "x_qwen":{"top_k":5,"min_p":0.1,"seed":7}}),
            )
            .unwrap();
        profile.normalize(&mut request).unwrap();
        let explicit = SamplingConfig {
            temperature: 0.0,
            top_k: 5,
            top_p: 0.8,
            min_p: 0.1,
            seed: 7,
        };
        assert_eq!(sampling_with_defaults(&request, *release), explicit);
        assert_eq!(
            (request.temperature_echo, request.top_p_echo),
            (Some(0.0), Some(0.8))
        );
    }

    let undecided = RequestProfile::OrdinaryQwen {
        template: QwenTemplate::Qwen38,
        no_thinking_supported: true,
        style: TemplateStyle::House,
        sampling: None,
        limits: crate::serve::request_profile::OutputLimits::TEST,
    };
    let mut request = undecided
        .parse(&json!({"model":"test","input":"Hello"}))
        .unwrap();
    undecided.normalize(&mut request).unwrap();
    assert_eq!(request.temperature, None);
    assert_eq!(
        release_request_sampler(&request, None)
            .unwrap()
            .config()
            .temperature,
        0.0
    );
}

/// K2 Horizon serve, raw and chat, defaults to the card's 1.0 / top-p 0.95.
#[test]
fn k2_serve_defaults_to_release_sampling() {
    for chat in [false, true] {
        let profile = k2(chat, 128);
        let body = if chat {
            json!({"model":"test","input":[{"role":"user","content":"hi"}]})
        } else {
            json!({"model":"test","input":"raw"})
        };
        let mut request = profile.parse(&body).unwrap();
        profile.normalize(&mut request).unwrap();
        let seed = request.seed.expect("normalization draws a seed");
        assert_eq!(
            crate::serve::render_k2::sampling(&request),
            qwen_llm::sampling::SamplingConfig::k2_horizon(seed)
        );
        assert_eq!(
            (request.temperature_echo, request.top_p_echo),
            (Some(1.0), Some(0.95))
        );
    }
}

#[test]
fn profiles_preserve_style_and_history_normalization_boundaries() {
    let body = json!({"model":"test","input":[{"role":"user","content":"one"},{"role":"assistant","content":"answer"},{"role":"user","content":"two"}]});
    for style in [TemplateStyle::House, TemplateStyle::Upstream] {
        let profile = RequestProfile::DeepSeekV4 {
            style,
            sampling: None,
            limits: crate::serve::request_profile::OutputLimits::TEST,
        };
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
            sampling: None,
            limits: crate::serve::request_profile::OutputLimits::TEST,
        },
        RequestProfile::FlashNext {
            style: TemplateStyle::Upstream,
            limits: crate::serve::request_profile::OutputLimits::TEST,
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
                    sampling: None,
                    limits: crate::serve::request_profile::OutputLimits::TEST,
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

/// A tool-block failure in the output partition happens on the HTTP worker
/// after the backend succeeded. It must still reach the owner as a
/// server-side failure (idle residency closes and never renews), streaming
/// or not: directly, and across the owner bridge with the completion.
#[test]
fn partition_failures_after_generation_reach_the_owner() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[derive(Clone)]
    struct Counting {
        inner: ProfileBackend,
        failures: Arc<AtomicUsize>,
    }
    impl GenerationBackend for Counting {
        fn model_id(&self) -> &str {
            "test"
        }
        fn request_profile(&self) -> RequestProfile {
            self.inner.profile.clone()
        }
        fn generate(
            &mut self,
            request: &ServeRequest,
            prompt: &str,
            sink: &mut dyn GenerationSink,
        ) -> Result<GenerationOutcome, BackendFailure> {
            self.inner.generate(request, prompt, sink)
        }
        fn request_failed_on_server(&mut self) {
            self.failures.fetch_add(1, Ordering::SeqCst);
        }
    }
    let backend = |failures: &Arc<AtomicUsize>| Counting {
        inner: ProfileBackend {
            profile: RequestProfile::Glm5Next {
                default_max_tokens: 8,
                capacity: 128,
                max_piece_bytes: 64,
            },
            expected_prompt: None,
            // A call to an undeclared function: the partition refuses it
            // (500) only when the turn finishes.
            output: "plan</think><tool_call>undeclared</tool_call>".into(),
            end: GenerationEnd::StopToken(154_829),
        },
        failures: Arc::clone(failures),
    };
    for streaming in [false, true] {
        let body = json!({"model":"test","input":"Weather?","stream":streaming,
            "tools":[{"type":"function","name":"get_weather","parameters":{"type":"object",
            "properties":{"city":{"type":"string"}}}}]});
        let request = post("/v1/responses", &body.to_string());

        let failures = Arc::new(AtomicUsize::new(0));
        let direct = super::roundtrip(backend(&failures), &request);
        assert!(direct.contains("names no declared tool"), "{direct}");
        assert_eq!(
            failures.load(Ordering::SeqCst),
            1,
            "direct, stream={streaming}"
        );

        let failures = Arc::new(AtomicUsize::new(0));
        let mut bridged = backend(&failures);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut activity = crate::serve::owner_activity::OwnerActivity::default();
            let guard = activity.admission().try_admit().unwrap();
            crate::serve::transport::handle_connection(
                stream,
                &mut bridged,
                None,
                guard,
                || Ok(()),
            )
            .unwrap();
            let mut completions = Vec::new();
            activity.drain_finished_with_failures(|failed| completions.push(failed));
            completions
        });
        let mut client = TcpStream::connect(address).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client.write_all(request.as_bytes()).unwrap();
        let mut reply = String::new();
        client.read_to_string(&mut reply).unwrap();
        assert!(reply.contains("names no declared tool"), "{reply}");
        // The owner's backend saw a successful generation; the failure
        // arrives with the completion, for the owner to report.
        assert_eq!(
            server.join().unwrap(),
            vec![true],
            "bridged, stream={streaming}"
        );
        assert_eq!(failures.load(Ordering::SeqCst), 0);
    }
}

/// `x_qwen.prefill_lineage` is GLM-5.3-Flash's alone: GLM binds it; every
/// other family refuses it explicitly (400, named parameter); a value other
/// than "fast" or "exact" is refused at parse.
#[test]
fn prefill_lineage_is_glm_only_and_refused_elsewhere() {
    use crate::serve::items::PrefillLineage;
    let body = |lineage: Value| json!({"model":"test","input":"Hello","x_qwen":{"prefill_lineage": lineage}});
    let glm = RequestProfile::Glm5Next {
        default_max_tokens: 8,
        capacity: 128,
        max_piece_bytes: 64,
    };
    for (value, expected) in [
        ("exact", PrefillLineage::Exact),
        ("fast", PrefillLineage::Fast),
    ] {
        let mut request = glm.parse(&body(json!(value))).unwrap();
        glm.normalize(&mut request).unwrap();
        assert_eq!(request.prefill_lineage, Some(expected));
    }
    for bad in [json!("serial"), json!(true), json!(1)] {
        let error = glm.parse(&body(bad.clone())).unwrap_err();
        assert_eq!(
            (error.status, error.param.as_deref()),
            (400, Some("x_qwen.prefill_lineage")),
            "{bad}"
        );
    }
    let others = [
        RequestProfile::OrdinaryQwen {
            template: QwenTemplate::Qwen38,
            no_thinking_supported: true,
            style: TemplateStyle::House,
            sampling: None,
            limits: crate::serve::request_profile::OutputLimits::TEST,
        },
        RequestProfile::FlashNext {
            style: TemplateStyle::House,
            limits: crate::serve::request_profile::OutputLimits::TEST,
        },
        RequestProfile::DeepSeekV4 {
            style: TemplateStyle::House,
            sampling: None,
            limits: crate::serve::request_profile::OutputLimits::TEST,
        },
        RequestProfile::Muse {
            template: MuseGlimmerChatTemplateProfile::UnslothLaunch,
            default_max_tokens: 8,
            eos_token_id: 1,
            eot_token_id: 2,
        },
        k2(false, 128),
    ];
    // K2's own parser already refuses any unknown x_qwen field; the others
    // parse it and refuse it when the family binds the request.
    for profile in others {
        let error = profile
            .parse(&body(json!("exact")))
            .and_then(|mut request| profile.normalize(&mut request))
            .unwrap_err();
        assert_eq!(error.status, 400, "{}", error.message);
        assert!(
            error
                .param
                .as_deref()
                .is_some_and(|p| p.starts_with("x_qwen")),
            "{error:?}"
        );
        assert!(
            error.message.contains("prefill_lineage"),
            "{}",
            error.message
        );
    }
}
