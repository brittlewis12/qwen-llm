use super::*;
use qwen_llm::metal::MetalContext;
use serde_json::json;
use std::io;

#[test]
#[ignore = "K2_GGUF production lease/API validation; real tool call-result-final HTTP JSON/SSE round trip"]
fn gpu_k2_tools_roundtrip_all_formats_json_sse() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let path = std::env::var("K2_GGUF").unwrap();
    let source = GgufFile::open(&path).unwrap();
    let invocation = crate::cli::ServeInvocation {
        model: path.into(),
        addr: "127.0.0.1:0".into(),
        max_tokens: Some(512),
        max_context_tokens: Some(2048),
        snapshot_cache_mib: Some(0),
        snapshot_policy: Default::default(),
        drafter: None,
        trace_sse: None,
    };
    let prepared = Prepared::new(&source, &invocation).unwrap();
    assert!(prepared.chat_profile.is_some());
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load(&ctx, &source, 2048).unwrap();
    let mut backend = K2Backend::new(&model, prepared, "k2-tools".into());
    let mut evidence = Vec::new();
    fn envelope(response: &str, stream: bool) -> serde_json::Value {
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let body = response.split_once("\r\n\r\n").unwrap().1;
        if stream {
            let event = body
                .split("\n\n")
                .find(|e| e.starts_with("event: response.completed\n"))
                .unwrap_or_else(|| panic!("{response}"));
            let data = event
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .unwrap();
            serde_json::from_str::<serde_json::Value>(data).unwrap()["response"].clone()
        } else {
            serde_json::from_str(body).unwrap()
        }
    }
    for format in ["xml", "json", "xml_typed"] {
        for stream in [false, true] {
            let mut body = json!({"model":"k2-tools","input":[{"role":"user","content":"Call lookup_code with key orbital to obtain its code. Do not guess the code. After the tool result arrives, repeat that code as your final answer."}],"tools":[{"type":"function","name":"lookup_code","description":"Retrieve the code for a key.","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}],"reasoning":{"effort":"low"},"max_output_tokens":512,"stream":stream});
            if format != "xml" {
                body["x_k2"] = json!({"tool_call_format":format});
            }
            let (_, prompt) = request(&backend, body.clone());
            let prompt_ids = backend.prepared.tokenizer.encode(&prompt, true).unwrap();
            eprintln!("K2 tools HTTP format={format} stream={stream} phase=call");
            let wire = wire_request(&mut backend, body.clone());
            if !wire.starts_with("HTTP/1.1 200") {
                let (req, prompt) = request(&backend, body.clone());
                let mut sink = Sink::default();
                let outcome = backend.generate(&req, &prompt, &mut sink).unwrap();
                eprintln!(
                    "K2 failed wire diagnostic end={:?} bytes={:?}",
                    outcome.end,
                    String::from_utf8_lossy(&sink.bytes)
                );
            }
            let first = envelope(&wire, stream);
            assert_eq!(first["status"], "completed");
            assert_eq!(first["x_k2"]["tool_call_format"], format);
            let calls: Vec<_> = first["output"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|i| i["type"] == "function_call")
                .collect();
            assert_eq!(calls.len(), 1, "{first}");
            assert_eq!(calls[0]["name"], "lookup_code");
            assert_eq!(
                qwen_llm::k2_horizon_chat::tools::decode_tool_json(
                    calls[0]["arguments"].as_str().unwrap()
                )
                .unwrap(),
                json!({"key":"orbital"})
            );
            body["input"]
                .as_array_mut()
                .unwrap()
                .extend(first["output"].as_array().unwrap().iter().cloned());
            body["input"].as_array_mut().unwrap().push(json!({"type":"function_call_output","call_id":calls[0]["call_id"],"output":"copper-731"}));
            eprintln!("K2 tools HTTP format={format} stream={stream} phase=result");
            let final_response = envelope(&wire_request(&mut backend, body.clone()), stream);
            assert_eq!(final_response["status"], "completed");
            assert!(
                final_response["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|i| i["type"] != "function_call")
            );
            let text = final_response["output"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|i| i["type"] == "message")
                .flat_map(|i| i["content"].as_array().unwrap())
                .map(|p| p["text"].as_str().unwrap())
                .collect::<String>();
            assert!(text.contains("copper-731"), "{final_response}");
            evidence.push(json!({"format":format,"stream":stream,"prompt_token_ids_sha256_i32le":qwen_llm::tokenizer::token_ids_sha256_i32le(&prompt_ids),"first":first,"final":final_response}));
        }
    }
    let body = json!({"model":"k2-tools","input":[{"role":"user","content":"Call lookup_code with key orbital."}],"tools":[{"type":"function","name":"lookup_code","parameters":{"properties":{"key":{"type":"string"}}}}],"reasoning":{"effort":"low"}});
    let (req, prompt) = request(&backend, body);
    for mut sink in [
        Sink {
            abort_tick: Some(3),
            ..Sink::default()
        },
        Sink {
            abort_piece: true,
            ..Sink::default()
        },
    ] {
        assert!(matches!(
            backend.generate(&req, &prompt, &mut sink),
            Err(BackendFailure::Aborted(_))
        ));
        assert!(backend.history.is_empty());
    }
    let output = std::env::var("K2_TOOLS_HTTP_EVIDENCE").unwrap();
    std::fs::write(output,serde_json::to_vec_pretty(&json!({"status":"passed","caller_supplied_tool_result":true,"engine_executes_tools":false,"cases":evidence})).unwrap()).unwrap();
}

#[test]
#[ignore = "K2_GGUF leased production Metal chat/JSON/SSE correctness; ephemeral owned loopback only"]
fn gpu_verified_k2_chat_http_matches_raw_and_releases_sessions() {
    use super::super::partition::PartitionEvent;
    use qwen_llm::k2_horizon_chat::Effort;
    let path = std::env::var("K2_GGUF").expect("K2_GGUF");
    let source = GgufFile::open(&path).unwrap();
    let invocation = crate::cli::ServeInvocation {
        model: path.into(),
        addr: "127.0.0.1:0".into(),
        max_tokens: Some(8),
        max_context_tokens: Some(384),
        snapshot_cache_mib: Some(0),
        snapshot_policy: Default::default(),
        drafter: None,
        trace_sse: None,
    };
    let prepared = Prepared::new(&source, &invocation).unwrap();
    assert!(prepared.chat_profile.is_some());
    // Non-test dependency owns the production lease and wired-memory gate.
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load(&ctx, &source, 384).unwrap();
    let mut backend = K2Backend::new(&model, prepared, "k2-chat".into());
    let mut evidence = Vec::new();
    for (effort, budget) in [
        (Effort::High, 8),
        (Effort::Medium, 8),
        (Effort::Low, 8),
        (Effort::Low, 128),
    ] {
        let body = json!({"model":"k2-chat","input":[{"role":"user","content":"What is 2+2? Answer briefly."}], "reasoning":{"effort":effort}, "max_output_tokens":budget});
        let (req, prompt) = request(&backend, body.clone());
        assert_eq!(
            backend.output_protocol(&req),
            OutputProtocol::K2Chat { effort }
        );
        let mut sink = Sink::default();
        let outcome = backend.generate(&req, &prompt, &mut sink).unwrap();
        if budget == 8 {
            // Raw deliberately does not stop on im_end. Compare the emitted
            // prefix only, excluding chat's counted-but-unemitted terminal ID.
            let raw_budget =
                outcome.usage.output_tokens - usize::from(!outcome.end.is_token_limit());
            assert!(raw_budget > 0);
            let (raw, raw_prompt) = request(
                &backend,
                json!({"model":"k2-chat","input":prompt,"max_output_tokens":raw_budget}),
            );
            let mut raw_sink = Sink::default();
            let raw_outcome = backend.generate(&raw, &raw_prompt, &mut raw_sink).unwrap();
            assert_eq!(raw_sink.bytes, sink.bytes);
            assert!(raw_outcome.end.is_token_limit());
            assert_eq!(raw_outcome.usage.output_tokens, raw_budget);
            assert_eq!(raw_outcome.usage.input_tokens, outcome.usage.input_tokens);
        }
        let mut partition = super::super::partition_k2::K2Partition::new(effort);
        let mut events = Vec::new();
        partition.push(&sink.bytes, &mut events);
        partition.finish(outcome.end, &mut events).unwrap();
        let mut reasoning = String::new();
        let mut visible = String::new();
        for event in events {
            match event {
                PartitionEvent::Reasoning(text) => reasoning.push_str(&text),
                PartitionEvent::Visible(text) => visible.push_str(&text),
                PartitionEvent::ReasoningClosed => {}
                _ => panic!("unexpected tools"),
            }
        }
        for stream in [false, true] {
            let mut body = body.clone();
            body["stream"] = json!(stream);
            let response = wire_request(&mut backend, body);
            assert!(response.starts_with("HTTP/1.1 200"), "{response}");
            let body = response.split_once("\r\n\r\n").unwrap().1;
            let envelope = if stream {
                let block = body
                    .split("\n\n")
                    .find(|b| {
                        b.starts_with("event: response.incomplete\n")
                            || b.starts_with("event: response.completed\n")
                    })
                    .unwrap();
                let data = block
                    .lines()
                    .find_map(|l| l.strip_prefix("data: "))
                    .unwrap();
                serde_json::from_str::<serde_json::Value>(data).unwrap()["response"].clone()
            } else {
                serde_json::from_str::<serde_json::Value>(body).unwrap()
            };
            assert_eq!(envelope["output"][0]["content"][0]["text"], reasoning);
            if visible.is_empty() {
                assert_eq!(envelope["output"].as_array().unwrap().len(), 1);
            } else {
                assert_eq!(envelope["output"][1]["content"][0]["text"], visible);
            }
            assert_eq!(
                envelope["usage"]["input_tokens"],
                outcome.usage.input_tokens
            );
            assert_eq!(
                envelope["usage"]["output_tokens"],
                outcome.usage.output_tokens
            );
            assert_eq!(envelope["reasoning"], json!({"effort":effort}));
            evidence
                .push(json!({"effort":effort,"budget":budget,"stream":stream,"response":envelope}));
        }
        let mut aborted = Sink {
            abort_tick: Some(2),
            ..Sink::default()
        };
        assert!(matches!(
            backend.generate(&req, &prompt, &mut aborted),
            Err(BackendFailure::Aborted(_))
        ));
        assert!(backend.history.is_empty());
    }
    if let Ok(path) = std::env::var("K2_CHAT_HTTP_EVIDENCE") {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&json!({"status":"passed","cases":evidence})).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn startup_limits_use_declared_context_and_explicit_residency_no_drafter() {
    assert_eq!(limits(8192, Some(32), Some(8), false).unwrap(), (32, 8));
    assert_eq!(
        limits(8192, Some(256), Some(256), false).unwrap(),
        (256, 256)
    );
    for size in [257, 1024, 7169, 8192, 524288] {
        assert_eq!(
            limits(524288, Some(size), Some(size), false).unwrap(),
            (size, size)
        );
    }
    for (context, capacity, maximum, drafter) in [
        (8192, None, Some(8), false),
        (8192, Some(32), None, false),
        (8192, Some(0), Some(1), false),
        (8192, Some(8193), Some(1), false),
        (8192, Some(32), Some(0), false),
        (8192, Some(32), Some(33), false),
        (1, Some(2), Some(1), false),
        (8192, Some(32), Some(8), true),
    ] {
        assert!(limits(context, capacity, maximum, drafter).is_err());
    }
}

#[derive(Default)]
struct Sink {
    bytes: Vec<u8>,
    ticks: usize,
    abort_tick: Option<usize>,
    abort_piece: bool,
}

#[test]
#[ignore = "CPU/header-only K2_GGUF startup rejection; never binds a socket or initializes Metal"]
fn cpu_downloaded_startup_rejects_options_before_listener_or_metal() {
    let path = std::env::var("K2_GGUF").expect("K2_GGUF");
    for (capacity, maximum, drafter, expected) in [
        (Some(usize::MAX), Some(8), None, "capacity must fit"),
        (None, Some(8), None, "explicit --max-context-tokens"),
        (Some(32), None, None, "explicit --max-tokens"),
        (
            Some(32),
            Some(8),
            Some("nonexistent-drafter.gguf"),
            "does not support a drafter",
        ),
    ] {
        let invocation = crate::cli::ServeInvocation {
            model: path.clone().into(),
            addr: "invalid-listen-address".into(),
            max_tokens: maximum,
            max_context_tokens: capacity,
            // Ignored by K2; a nonzero budget must not affect startup checks.
            snapshot_cache_mib: Some(4096),
            snapshot_policy: Default::default(),
            drafter: drafter.map(Into::into),
            trace_sse: None,
        };
        let error = super::super::run_serve(invocation).unwrap_err();
        assert!(error.to_string().contains(expected), "{error:#}");
    }
}

#[test]
fn canonical_k2_generation_counts_eos_without_emitting_or_forwarding_it() {
    let mut sampler = Sampler::new(qwen_llm::sampling::SamplingConfig::default()).unwrap();
    let first = crate::generate_serial(
        vec![0.0, 10.0, -10.0],
        8,
        &[1],
        &mut sampler,
        |_| panic!("EOS emitted"),
        |_| panic!("EOS forwarded"),
    )
    .unwrap();
    assert_eq!(first.tokens, [1]);
    assert_eq!(first.transitions, 0);
    assert!(matches!(first.stop_reason, crate::StopReason::Eos));
    let mut emitted = Vec::new();
    let mut forwarded = Vec::new();
    let mut sampler = Sampler::new(qwen_llm::sampling::SamplingConfig::default()).unwrap();
    let late = crate::generate_serial(
        vec![0.0, -10.0, 10.0],
        8,
        &[1],
        &mut sampler,
        |id| {
            emitted.push(id);
            Ok(())
        },
        |id| {
            forwarded.push(id);
            Ok(vec![0.0, 10.0, -10.0])
        },
    )
    .unwrap();
    assert_eq!(late.tokens, [2, 1]);
    assert_eq!(emitted, [2]);
    assert_eq!(forwarded, [2]);
    assert_eq!(late.transitions, 1);
    assert_eq!(
        super::super::outcome::generation_end(&late).1,
        super::super::output_partition::GenerationEnd::StopToken(1)
    );
}
impl GenerationSink for Sink {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        if self.abort_piece {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "test disconnect"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        self.ticks += 1;
        if self.abort_tick == Some(self.ticks) {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "test disconnect"));
        }
        Ok(())
    }
}

fn request(backend: &K2Backend<'_, '_>, body: serde_json::Value) -> (ServeRequest, String) {
    let mut request = backend.parse_request(&body).unwrap();
    backend.normalize_request(&mut request).unwrap();
    let prompt = backend.render_prompt(&request).unwrap();
    (request, prompt)
}

fn wire_request(backend: &mut K2Backend<'_, '_>, body: serde_json::Value) -> String {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::time::Duration;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let client = std::thread::spawn(move || {
        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(120)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        let body = body.to_string();
        write!(stream, "POST /v1/responses HTTP/1.1\r\nhost: localhost\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}", body.len()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    });
    let (stream, _) = listener.accept().unwrap();
    // The Metal backend stays on this thread. Only the socket client is spawned.
    super::super::http::handle_connection(&stream, backend, None).unwrap();
    drop(stream);
    client.join().unwrap()
}

#[test]
#[ignore = "K2_GGUF GPU correctness; CLI links production library lease/memory gate, no server process"]
fn gpu_borrowed_backend_matches_raw_run_and_discards_aborted_requests() {
    let path = std::env::var("K2_GGUF").expect("K2_GGUF");
    let source = GgufFile::open(&path).unwrap();
    let invocation = crate::cli::ServeInvocation {
        model: path.into(),
        addr: "127.0.0.1:0".into(),
        max_tokens: Some(8),
        max_context_tokens: Some(32),
        snapshot_cache_mib: Some(0),
        snapshot_policy: Default::default(),
        drafter: None,
        trace_sse: None,
    };
    let prepared = Prepared::new(&source, &invocation).unwrap();
    // qwen-llm is a non-test dependency in this CLI test: new() acquires the
    // production exclusive lease and real wired gate. Do not take a second lease.
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load_unqualified(&ctx, &source, 32).unwrap();
    let mut backend = K2Backend::new(&model, prepared, "k2-test".into());
    let text = "The capital of France is";
    let (req, prompt) = request(&backend, json!({"model":"k2-test","input":text}));
    assert_eq!(backend.output_protocol(&req), OutputProtocol::RawText);
    let mut first = Sink::default();
    let result = backend.generate(&req, &prompt, &mut first).unwrap();
    assert_eq!(first.bytes, b" Paris. The capital of Germany is Berlin");
    assert_eq!(result.usage.input_tokens, 6);
    assert_eq!(result.usage.output_tokens, 8);
    assert_eq!(result.usage.cached_tokens, 0);
    assert_eq!(
        result.end,
        super::super::output_partition::GenerationEnd::TokenLimit
    );
    // Prompt plus the 7 forwarded outputs (the 8th is sampled, never forwarded).
    assert_eq!(backend.history.len(), 13);

    // An abort at the admission tick precedes any session movement and keeps
    // history; later aborts clear it, so the retry recomputes from zero.
    for (mut sink, cached) in [
        (
            Sink {
                abort_tick: Some(1),
                ..Sink::default()
            },
            5,
        ),
        (
            Sink {
                abort_tick: Some(4),
                ..Sink::default()
            },
            0,
        ),
        (
            Sink {
                abort_piece: true,
                ..Sink::default()
            },
            0,
        ),
        (
            Sink {
                abort_tick: Some(10),
                ..Sink::default()
            },
            0,
        ),
    ] {
        assert!(matches!(
            backend.generate(&req, &prompt, &mut sink),
            Err(BackendFailure::Aborted(_))
        ));
        assert_eq!(backend.history.is_empty(), cached == 0);
        let mut retry = Sink::default();
        let result = backend.generate(&req, &prompt, &mut retry).unwrap();
        assert_eq!(retry.bytes, first.bytes);
        assert_eq!(result.usage.cached_tokens, cached);
    }
    // Warm repeat: the longest common prefix, capped to recompute the last row.
    let mut warm = Sink::default();
    let result = backend.generate(&req, &prompt, &mut warm).unwrap();
    assert_eq!(warm.bytes, first.bytes);
    assert_eq!(result.usage.cached_tokens, 5);
    // Continuation: the whole previous exchange is reused, only the new turn runs.
    let (followup, followup_prompt) = request(
        &backend,
        json!({"model":"k2-test","input":format!("{text}{}.", String::from_utf8(first.bytes.clone()).unwrap()),"max_output_tokens":1}),
    );
    let result = backend
        .generate(&followup, &followup_prompt, &mut Sink::default())
        .unwrap();
    assert!(result.usage.cached_tokens >= 13, "{:?}", result.usage);
    backend.prefix_reuse = false;
    let result = backend
        .generate(&req, &prompt, &mut Sink::default())
        .unwrap();
    assert_eq!(result.usage.cached_tokens, 0);
    backend.prefix_reuse = true;
    let (explicit, prompt) = request(
        &backend,
        json!({"model":"k2-test","input":format!("<|ifm|begin_of_text|>{text}"),
        "x_k2":{"add_special_tokens":false}, "max_output_tokens":1}),
    );
    let mut sink = Sink::default();
    let result = backend.generate(&explicit, &prompt, &mut sink).unwrap();
    assert_eq!(result.usage.input_tokens, 6);
    assert_eq!(result.usage.output_tokens, 1);
    assert_eq!(sink.bytes, b" Paris");

    let (automatic, prompt) = request(
        &backend,
        json!({"model":"k2-test","input":format!("<|ifm|begin_of_text|>{text}"), "max_output_tokens":1}),
    );
    let result = backend
        .generate(&automatic, &prompt, &mut Sink::default())
        .unwrap();
    assert_eq!(
        result.usage.input_tokens, 7,
        "automatic BOS never deduplicates authored IDs"
    );

    for stream in [false, true] {
        let response = wire_request(
            &mut backend,
            json!({"model":"k2-test","input":text,"stream":stream,
            "max_output_tokens":8,"x_qwen":{"stats":stream}}),
        );
        assert!(response.starts_with("HTTP/1.1 200"));
        let body = response.split_once("\r\n\r\n").unwrap().1;
        let envelope = if stream {
            let block = body
                .split("\n\n")
                .find(|block| block.starts_with("event: response.incomplete\n"))
                .unwrap();
            let data = block
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap();
            serde_json::from_str::<serde_json::Value>(data).unwrap()["response"].clone()
        } else {
            serde_json::from_str::<serde_json::Value>(body).unwrap()
        };
        assert_eq!(
            envelope["output"][0]["content"][0]["text"],
            " Paris. The capital of Germany is Berlin"
        );
        assert_eq!(envelope["usage"]["input_tokens"], 6);
        assert_eq!(envelope["usage"]["output_tokens"], 8);
        if stream {
            // Follows the identical nonstream request.
            assert_eq!(envelope["x_qwen"]["matched_tokens"], 5);
        } else {
            assert!(envelope.get("x_qwen").is_none());
        }
    }

    let (oversized, prompt) = request(
        &backend,
        json!({"model":"k2-test","input":text,"max_output_tokens":32}),
    );
    let mut sink = Sink::default();
    assert!(matches!(
        backend.generate(&oversized, &prompt, &mut sink),
        Err(BackendFailure::Serve(_))
    ));
    assert_eq!(sink.ticks, 0);
    // Capacity rejection precedes the session and keeps its history.
    assert_eq!(backend.history.len(), 13);
}

#[test]
#[ignore = "K2_GGUF and K2_BOUNDARY_EVIDENCE; production CLI lease/gate; ephemeral JSON/SSE boundary correctness"]
fn gpu_context_json_sse_match_run_bench_and_reject_capacity_plus_one() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let path = std::env::var("K2_GGUF").expect("K2_GGUF");
    let evidence = std::path::PathBuf::from(
        std::env::var("K2_BOUNDARY_EVIDENCE").expect("K2_BOUNDARY_EVIDENCE"),
    );
    let source = GgufFile::open(&path).unwrap();
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(evidence.join("summary.json")).unwrap()).unwrap();
    let capacity = summary["capacity"].as_u64().unwrap() as usize;
    assert_eq!(summary["status"], "passed");
    assert!(capacity >= 258);
    let invocation = crate::cli::ServeInvocation {
        model: path.into(),
        addr: "127.0.0.1:0".into(),
        max_tokens: Some(1),
        max_context_tokens: Some(capacity),
        snapshot_cache_mib: Some(0),
        snapshot_policy: Default::default(),
        drafter: None,
        trace_sse: None,
    };
    let prepared = Prepared::new(&source, &invocation).unwrap();
    let mut cases = Vec::new();
    for (name, words, sampled, count) in [
        ("boundary", capacity - 1, 1, capacity),
        ("transition", capacity - 2, 2, capacity - 1),
        ("response", capacity - 257, 257, capacity - 256),
    ] {
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(evidence.join(format!("bench-{name}.stdout"))).unwrap(),
        )
        .unwrap();
        let text = vec!["a"; words].join(" ");
        let ids = prepared.tokenizer.encode(&text, true).unwrap();
        assert_eq!(ids.len(), count);
        assert_eq!(json!(ids), reference["request"]["prompt_token_ids"]);
        assert_eq!(reference["request"]["capacity"], capacity);
        assert_eq!(reference["samples"][0]["committed_positions"], capacity);
        let hex = reference["samples"][0]["outcome"]["emitted_bytes_hex"]
            .as_str()
            .unwrap();
        assert!(hex.is_ascii() && hex.len() % 2 == 0);
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect::<Vec<_>>();
        cases.push((text, sampled, count, bytes));
    }
    // This binary links the production library: its context owns the lease.
    let ctx = MetalContext::new().unwrap();
    let model = K2LoadedModel::load(&ctx, &source, u32::try_from(capacity).unwrap()).unwrap();
    let mut backend = K2Backend::new(&model, prepared, "k2-boundary".into());
    // Fresh-prefill parity with run/bench; the capacity-sized session is still
    // kept and rewound to zero between requests.
    backend.prefix_reuse = false;
    for (text, sampled, count, expected) in &cases {
        let (req, prompt) = request(
            &backend,
            json!({"model":"k2-boundary","input":text,"max_output_tokens":sampled}),
        );
        let mut sink = Sink::default();
        let outcome = backend.generate(&req, &prompt, &mut sink).unwrap();
        assert_eq!(&sink.bytes, expected);
        assert_eq!(outcome.usage.input_tokens, *count);
        assert_eq!(outcome.usage.output_tokens, *sampled);
        assert_eq!(outcome.usage.cached_tokens, 0);
        for stream in [false, true] {
            let mut body =
                json!({"model":"k2-boundary","input":text,"stream":stream,"x_qwen":{"stats":true}});
            // Exercise the startup default at the configured request boundary.
            if *sampled != 1 {
                body["max_output_tokens"] = json!(sampled);
            }
            let response = wire_request(&mut backend, body);
            assert!(response.starts_with("HTTP/1.1 200"), "{response}");
            let body = response.split_once("\r\n\r\n").unwrap().1;
            let envelope = if stream {
                let block = body
                    .split("\n\n")
                    .find(|b| {
                        b.starts_with("event: response.incomplete\n")
                            || b.starts_with("event: response.completed\n")
                    })
                    .unwrap();
                let data = block
                    .lines()
                    .find_map(|l| l.strip_prefix("data: "))
                    .unwrap();
                serde_json::from_str::<serde_json::Value>(data).unwrap()["response"].clone()
            } else {
                serde_json::from_str::<serde_json::Value>(body).unwrap()
            };
            assert_eq!(
                envelope["output"][0]["content"][0]["text"],
                String::from_utf8(expected.clone()).unwrap()
            );
            assert_eq!(envelope["usage"]["input_tokens"], json!(count));
            assert_eq!(envelope["usage"]["output_tokens"], json!(sampled));
            assert_eq!(envelope["x_qwen"]["matched_tokens"], 0);
        }
    }
    let text = &cases[0].0;
    let (oversized, prompt) = request(
        &backend,
        json!({"model":"k2-boundary","input":text,"max_output_tokens":2}),
    );
    let mut sink = Sink::default();
    assert!(matches!(
        backend.generate(&oversized, &prompt, &mut sink),
        Err(BackendFailure::Serve(_))
    ));
    assert_eq!(sink.ticks, 0);
    for stream in [false, true] {
        let response = wire_request(
            &mut backend,
            json!({"model":"k2-boundary","input":text,"max_output_tokens":2,"stream":stream}),
        );
        if stream {
            // The shared transport starts SSE before backend tokenization, so
            // a pre-session budget error is a failure event after HTTP 200.
            assert!(response.starts_with("HTTP/1.1 200"), "{response}");
            let body = response.split_once("\r\n\r\n").unwrap().1;
            let block = body
                .split("\n\n")
                .find(|b| b.starts_with("event: response.failed\n"))
                .unwrap();
            let data = block
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .unwrap();
            let failed: serde_json::Value = serde_json::from_str(data).unwrap();
            assert_eq!(failed["response"]["error"]["type"], "invalid_request");
            assert_eq!(failed["response"]["error"]["param"], "max_output_tokens");
            assert!(
                failed["response"]["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains(&format!("{} K2 forwards", capacity + 1))
            );
            assert_eq!(failed["response"]["output"], json!([]));
            assert!(!body.contains("event: response.output_text.delta"));
        } else {
            assert!(response.starts_with("HTTP/1.1 400"), "{response}");
        }
    }
    let (req, prompt) = request(&backend, json!({"model":"k2-boundary","input":text}));
    let mut aborted = Sink {
        abort_tick: Some(250),
        ..Sink::default()
    };
    assert!(matches!(
        backend.generate(&req, &prompt, &mut aborted),
        Err(BackendFailure::Aborted(_))
    ));
    let mut fresh = Sink::default();
    assert_eq!(
        backend
            .generate(&req, &prompt, &mut fresh)
            .unwrap()
            .usage
            .cached_tokens,
        0
    );
    assert_eq!(fresh.bytes, cases[0].3);
    std::fs::write(evidence.join("serve-context-result.json"),serde_json::to_vec_pretty(&json!({
        "status":"passed","capacity":capacity,"json_sse_run_bench_parity":true,"default_output_boundary":true,
        "reject_capacity_plus_one_before_session":true,"fresh_after_late_prefill_abort":true,"response_budget_257":true,"performance_claim":false,
    })).unwrap()).unwrap();
}

#[test]
fn prefill_span_is_a_whole_multiple_of_the_physical_chunk() {
    // (chunk_tokens, span): serial, Q8 lcpp, short general appends, general.
    for (chunk, span) in [
        (0, 64),
        (1, 64),
        (32, 64),
        (17, 68),
        (64, 64),
        (100, 100),
        (256, 256),
    ] {
        assert_eq!(prefill_span(chunk), span, "chunk={chunk}");
        if chunk > 0 {
            assert_eq!(prefill_span(chunk) % chunk, 0);
            assert!(prefill_span(chunk) >= PREFILL_SPAN);
        }
    }
}
