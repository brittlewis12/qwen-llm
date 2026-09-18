use super::*;
use qwen_llm::metal::MetalContext;
use serde_json::json;
use std::io;

#[test]
fn startup_limits_require_explicit_short_capacity_no_cache_or_drafter() {
    assert_eq!(limits(8192, Some(32), Some(8), 0, false).unwrap(), (32, 8));
    for (context, capacity, maximum, snapshots, drafter) in [
        (8192, None, Some(8), 0, false),
        (8192, Some(32), None, 0, false),
        (8192, Some(0), Some(1), 0, false),
        (8192, Some(33), Some(1), 0, false),
        (8192, Some(32), Some(0), 0, false),
        (8192, Some(32), Some(33), 0, false),
        (1, Some(2), Some(1), 0, false),
        (8192, Some(32), Some(8), 1, false),
        (8192, Some(32), Some(8), 0, true),
    ] {
        assert!(limits(context, capacity, maximum, snapshots, drafter).is_err());
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
    for (capacity, maximum, snapshots, drafter, expected) in [
        (Some(33), Some(8), 0, None, "capacity must fit"),
        (None, Some(8), 0, None, "explicit --max-context-tokens"),
        (Some(32), None, 0, None, "explicit --max-tokens"),
        (Some(32), Some(8), 1, None, "--snapshot-cache-mib 0"),
        (
            Some(32),
            Some(8),
            0,
            Some("nonexistent-drafter.gguf"),
            "does not support a drafter",
        ),
    ] {
        let invocation = crate::cli::ServeInvocation {
            model: path.clone().into(),
            addr: "invalid-listen-address".into(),
            max_tokens: maximum,
            max_context_tokens: capacity,
            snapshot_cache_mib: snapshots,
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
            .set_read_timeout(Some(Duration::from_secs(30)))
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
        snapshot_cache_mib: 0,
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
    drop(model.create_session(0).unwrap());

    for mut sink in [
        Sink {
            abort_tick: Some(1),
            ..Sink::default()
        },
        Sink {
            abort_tick: Some(4),
            ..Sink::default()
        },
        Sink {
            abort_piece: true,
            ..Sink::default()
        },
        Sink {
            abort_tick: Some(10),
            ..Sink::default()
        },
    ] {
        assert!(matches!(
            backend.generate(&req, &prompt, &mut sink),
            Err(BackendFailure::Aborted(_))
        ));
        drop(model.create_session(0).unwrap());
        let mut fresh = Sink::default();
        let result = backend.generate(&req, &prompt, &mut fresh).unwrap();
        assert_eq!(fresh.bytes, first.bytes);
        assert_eq!(result.usage.cached_tokens, 0);
    }
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
            assert_eq!(envelope["x_qwen"]["matched_tokens"], 0);
        } else {
            assert!(envelope.get("x_qwen").is_none());
        }
        drop(model.create_session(0).unwrap());
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
    drop(model.create_session(0).unwrap());
}
