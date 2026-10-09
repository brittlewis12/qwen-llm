//! End-to-end peak memory of a GLM-5.3 tool block through the real HTTP
//! handler (`handle_connection`): non-streaming, streaming, and streaming
//! with an SSE trace, measured by the test allocator (`crate::test_alloc`;
//! frees on the trace writer thread count). The peak above a run without a
//! tool block must stay within the admission model,
//! `tool_block_peak_bytes(block)`.

use super::*;
use crate::serve::request_profile::RequestProfile;
use qwen_llm::glm5_next_chat::{TOOL_BLOCK_PEAK_FACTOR, tool_block_peak_bytes};

/// Emits `output` in `piece`-byte pieces (ASCII, so every split is a char
/// boundary), as a backend streams decoded tokens.
#[derive(Clone)]
struct ToolOutput {
    output: String,
    piece: usize,
    profile: RequestProfile,
}

fn glm_profile() -> RequestProfile {
    RequestProfile::Glm5Next {
        default_max_tokens: 64,
        capacity: 1 << 20,
        max_piece_bytes: 512,
    }
}

impl GenerationBackend for ToolOutput {
    fn model_id(&self) -> &str {
        "test"
    }
    fn request_profile(&self) -> RequestProfile {
        self.profile.clone()
    }
    fn generate(
        &mut self,
        _request: &ServeRequest,
        _prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        for piece in self.output.as_bytes().chunks(self.piece) {
            sink.piece(piece).map_err(BackendFailure::Aborted)?;
        }
        Ok(GenerationOutcome {
            end: GenerationEnd::StopToken(154_829),
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
            },
            stats: None,
        })
    }
}

/// Serve one request with `backend`, measuring the handler thread; returns
/// the response and the peak.
fn served(backend: ToolOutput, request: String, trace: Option<&Path>) -> (String, usize) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let trace = trace.map(|path| TraceLog::open(path).unwrap());
    let subscriber = trace.as_ref().map(TraceLog::subscriber);
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut backend = backend;
        let ((), peak) = crate::test_alloc::measure(|| {
            handle_connection(&stream, &mut backend, subscriber).unwrap();
        });
        peak
    });
    let mut client = TcpStream::connect(address).unwrap();
    client.write_all(request.as_bytes()).unwrap();
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    let peak = server.join().unwrap();
    drop(trace);
    (response, peak)
}

#[test]
fn tool_block_publication_stays_within_the_admission_model() {
    let body = |streaming: bool, schema: Value| {
        json!({"model": "test", "input": "Go.", "stream": streaming,
            "max_output_tokens": 200_000, "reasoning": {"effort": "low"},
            "tools": [{"type": "function", "name": "f", "parameters": {"type": "object",
                "properties": {"a": schema}}}]})
        .to_string()
    };
    let call = |value: &str| {
        format!("<tool_call>f<arg_key>a</arg_key><arg_value>{value}</arg_value></tool_call>")
    };
    let inner = format!("{}1{}", "[".repeat(126), "]".repeat(126));
    let shapes: Vec<(&str, Value, String)> = vec![
        (
            "array of deep arrays",
            json!({"type": "array"}),
            call(&format!("[{}]", vec![inner.as_str(); 300].join(","))),
        ),
        (
            "one-element arrays",
            json!({"type": "array"}),
            call(&format!("[{}]", vec!["[1]"; 20_000].join(","))),
        ),
        (
            "control escapes",
            json!({"type": "string"}),
            call(&"\u{1}".repeat(60_000)),
        ),
        (
            "short strings",
            json!({"type": "array"}),
            call(&format!("[{}]", vec!["\"a\""; 20_000].join(","))),
        ),
    ];
    let trace_dir = std::env::temp_dir().join(format!("qwen-tool-peak-{}", std::process::id()));
    std::fs::create_dir_all(&trace_dir).unwrap();
    let mut worst = 0.0f64;
    // One-byte pieces are the worst case for non-streaming collection (one
    // allocation per piece); four bytes is closer to typical tokens.
    for (mode, streaming, traced, piece) in [
        ("non-stream/1", false, false, 1),
        ("non-stream/4", false, false, 4),
        ("stream/1", true, false, 1),
        ("stream+trace/1", true, true, 1),
    ] {
        let trace_path = trace_dir.join(format!("{}.jsonl", mode.replace('/', "-")));
        let trace = traced.then_some(trace_path.as_path());
        // Fixed costs of the same request with no tool block.
        let (reply, baseline) = served(
            ToolOutput {
                output: "plan</think>ok".into(),
                piece,
                profile: glm_profile(),
            },
            post("/v1/responses", &body(streaming, json!({"type": "array"}))),
            trace,
        );
        assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
        for (label, schema, block) in &shapes {
            let (reply, peak) = served(
                ToolOutput {
                    output: format!("plan</think>{block}"),
                    piece,
                    profile: glm_profile(),
                },
                post("/v1/responses", &body(streaming, schema.clone())),
                trace,
            );
            assert!(
                reply.starts_with("HTTP/1.1 200"),
                "{mode} {label}: {reply:.300}"
            );
            assert!(reply.contains("function_call"), "{mode} {label}");
            let above = peak.saturating_sub(baseline);
            let ratio = above as f64 / block.len() as f64;
            worst = worst.max(ratio);
            eprintln!(
                "[tool-block-e2e] {mode} {label}: block={} peak_above_baseline={above} ratio={ratio:.1}",
                block.len()
            );
            assert!(
                above <= tool_block_peak_bytes(block.len()),
                "{mode} {label}: {above} bytes above baseline for a {}-byte block exceeds {TOOL_BLOCK_PEAK_FACTOR}x",
                block.len()
            );
        }
    }
    eprintln!("[tool-block-e2e] worst ratio {worst:.1} (model {TOOL_BLOCK_PEAK_FACTOR})");
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// Map #14 measurement: plain output in one-byte pieces, the worst piece
/// size for per-piece retention; the peak above a minimal response per
/// output byte. Measured 2026-10-08: non-streaming 246.5 with one
/// allocation and one retained event per piece, 16.3 with contiguous
/// collection and events consumed as replayed; streaming 5.2. The bounds
/// are regression alarms at about 1.5x the measured values.
#[test]
fn nonstream_plain_output_peak_per_byte() {
    let request = |streaming: bool| {
        post(
            "/v1/responses",
            &json!({"model": "test", "input": "Go.", "stream": streaming,
                "max_output_tokens": 400_000, "reasoning": {"effort": "low"}})
            .to_string(),
        )
    };
    let text = "a".repeat(200_000);
    for streaming in [false, true] {
        let (_, base) = served(
            ToolOutput {
                output: "plan</think>ok".into(),
                piece: 1,
                profile: glm_profile(),
            },
            request(streaming),
            None,
        );
        let (response, peak) = served(
            ToolOutput {
                output: format!("plan</think>{text}"),
                piece: 1,
                profile: glm_profile(),
            },
            request(streaming),
            None,
        );
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let ratio = (peak - base) as f64 / text.len() as f64;
        eprintln!(
            "[nonstream-plain] stream={streaming} bytes={} peak_above_base={} ratio={ratio:.1}",
            text.len(),
            peak - base
        );
        let bound = if streaming { 8.0 } else { 24.0 };
        assert!(
            ratio <= bound,
            "stream={streaming}: {ratio:.1} bytes per output byte > {bound}"
        );
    }
}

/// Map #14 packet 2 measurement (before their admission): Qwen XML and
/// DeepSeek V4 DSML tool blocks through the real handler, the same shapes
/// as GLM's, in one-byte pieces. Measured 2026-10-08: worst 147.5 bytes per
/// block byte (deep arrays, streaming), the same as GLM's (the shared
/// argument parse and publication dominate). Held to GLM's 256 model, the
/// factor the admission adopts; four shapes are evidence, not a proof.
#[test]
fn qwen_and_ds4_tool_block_peaks() {
    let body = |streaming: bool| {
        json!({"model": "test", "input": "Go.", "stream": streaming,
            "max_output_tokens": 200_000, "reasoning": {"effort": "high"},
            "tools": [{"type": "function", "name": "f", "parameters": {"type": "object",
                "properties": {"a": {}}}}]})
        .to_string()
    };
    let inner = format!("{}1{}", "[".repeat(126), "]".repeat(126));
    let values: Vec<(&str, String)> = vec![
        (
            "array of deep arrays",
            format!("[{}]", vec![inner.as_str(); 300].join(",")),
        ),
        (
            "one-element arrays",
            format!("[{}]", vec!["[1]"; 20_000].join(",")),
        ),
        ("control escapes", "\u{1}".repeat(60_000)),
        (
            "short strings",
            format!("[{}]", vec!["\"a\""; 20_000].join(",")),
        ),
    ];
    let qwen = |value: &str| {
        format!(
            "<tool_call>\n<function=f>\n<parameter=a>\n{value}\n</parameter>\n</function>\n</tool_call>"
        )
    };
    let dsml = |value: &str| {
        format!(
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"a\" string=\"false\">{value}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        )
    };
    /// (label, profile, reasoning prefix, call renderer)
    type Family<'a> = (&'a str, RequestProfile, &'a str, &'a dyn Fn(&str) -> String);
    let families: [Family; 2] = [
        (
            "qwen-xml",
            RequestProfile::UnboundQwen,
            "<think>plan</think>",
            &qwen,
        ),
        (
            "ds4-dsml",
            RequestProfile::DeepSeekV4 {
                style: TemplateStyle::House,
                sampling: None,
                limits: crate::serve::request_profile::OutputLimits::TEST,
            },
            "plan</think>",
            &dsml,
        ),
    ];
    let mut worst = 0.0f64;
    for (family, profile, reasoning, call) in &families {
        for (mode, streaming, piece) in [("non-stream/1", false, 1), ("stream/1", true, 1)] {
            let (reply, baseline) = served(
                ToolOutput {
                    output: format!("{reasoning}ok"),
                    piece,
                    profile: profile.clone(),
                },
                post("/v1/responses", &body(streaming)),
                None,
            );
            assert!(reply.starts_with("HTTP/1.1 200"), "{reply:.300}");
            for (label, value) in &values {
                let block = call(value);
                let (reply, peak) = served(
                    ToolOutput {
                        output: format!("{reasoning}{block}"),
                        piece,
                        profile: profile.clone(),
                    },
                    post("/v1/responses", &body(streaming)),
                    None,
                );
                assert!(
                    reply.starts_with("HTTP/1.1 200"),
                    "{family} {mode} {label}: {reply:.300}"
                );
                let called = reply.contains("function_call");
                let above = peak.saturating_sub(baseline);
                let ratio = above as f64 / block.len() as f64;
                worst = worst.max(ratio);
                eprintln!(
                    "[tool-block-family] {family} {mode} {label}: block={} parsed_call={called} peak_above_baseline={above} ratio={ratio:.1}",
                    block.len()
                );
                assert!(called, "{family} {mode} {label}: the call did not parse");
                assert!(
                    above <= tool_block_peak_bytes(block.len()),
                    "{family} {mode} {label}: {above} bytes above baseline exceeds {TOOL_BLOCK_PEAK_FACTOR}x"
                );
            }
        }
    }
    eprintln!("[tool-block-family] worst ratio {worst:.1}");
}
