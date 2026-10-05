use super::*;
use crate::serve::items::TemplateStyle;
use crate::serve::output_partition::{GenerationEnd, OutputPartition, OutputProtocol};
use crate::serve::partition::PartitionEvent;
use serde_json::{Value, json};

fn invocation(
    path: &str,
    capacity: Option<usize>,
    maximum: Option<usize>,
) -> crate::cli::ServeInvocation {
    crate::cli::ServeInvocation {
        lens_data_dir: None,
        lens_config: None,
        web_root: None,
        lens_allowed_origin: Vec::new(),
        model: path.into(),
        addr: "invalid-listen-address".into(),
        max_tokens: maximum,
        max_context_tokens: capacity,
        snapshot_cache_mib: None,
        snapshot_policy: Default::default(),
        durable: crate::serve::durable::DurableSnapshotConfig::off(),
        drafter: None,
        trace_sse: None,
        template_style: Default::default(),
    }
}

fn gguf_path() -> String {
    std::env::var("GLM53_GGUF").expect("GLM53_GGUF (GLM-5.3-Flash shard 1)")
}

#[test]
#[ignore = "CPU/header-only GLM53_GGUF startup refusals; never binds a socket or initializes Metal"]
fn cpu_startup_refuses_bad_limits_before_listener_or_metal() {
    let path = gguf_path();
    for (capacity, maximum, change, expected) in [
        (None, Some(8), None, "requires --max-context-tokens"),
        (Some(64), None, None, "requires explicit --max-tokens"),
        (Some(64), Some(65), None, "exceeds --max-context-tokens"),
        (Some(4 << 20), Some(8), None, "exceeds model context"),
        (Some(0), Some(0), None, "positive --max-context-tokens"),
        (
            Some(64),
            Some(8),
            Some("drafter"),
            "does not support a drafter",
        ),
        (
            Some(64),
            Some(8),
            Some("upstream"),
            "--template-style upstream",
        ),
    ] {
        let mut invocation = invocation(&path, capacity, maximum);
        match change {
            Some("drafter") => invocation.drafter = Some("nonexistent-drafter.gguf".into()),
            Some("upstream") => invocation.template_style = TemplateStyle::Upstream,
            _ => {}
        }
        let error = crate::serve::run_serve(invocation).unwrap_err();
        assert!(format!("{error:#}").contains(expected), "{error:#}");
    }
}

#[derive(Default)]
struct Sink {
    bytes: Vec<u8>,
    ticks: usize,
    abort_tick: Option<usize>,
    abort_piece: bool,
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

fn request(backend: &Glm5NextBackend<'_, '_>, body: Value) -> (ServeRequest, String) {
    let mut request = backend.parse_request(&body).unwrap();
    backend.normalize_request(&mut request).unwrap();
    let prompt = backend.render_prompt(&request).unwrap();
    (request, prompt)
}

/// (reasoning, answer) through the serve output grammar.
fn split(bytes: &[u8], end: GenerationEnd) -> (String, String) {
    let mut partition = OutputPartition::new(OutputProtocol::Glm5NextChat);
    let mut events = Vec::new();
    partition.push(bytes, &mut events);
    partition.finish(end, &mut events).unwrap();
    let (mut reasoning, mut answer) = (String::new(), String::new());
    for event in events {
        match event {
            PartitionEvent::Reasoning(text) => reasoning.push_str(&text),
            PartitionEvent::Visible(text) => answer.push_str(&text),
            PartitionEvent::ReasoningClosed => {}
            PartitionEvent::FunctionCall(_) => panic!("no-tools grammar parsed a call"),
        }
    }
    (reasoning, answer)
}

fn wire(backend: &mut Glm5NextBackend<'_, '_>, body: Value) -> String {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::time::Duration;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let client = std::thread::spawn(move || {
        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(600)))
            .unwrap();
        let body = body.to_string();
        write!(stream, "POST /v1/responses HTTP/1.1\r\nhost: localhost\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}", body.len()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    });
    let (stream, _) = listener.accept().unwrap();
    // The Metal backend stays on this thread; only the socket client is spawned.
    crate::serve::http::handle_connection(&stream, backend, None).unwrap();
    drop(stream);
    client.join().unwrap()
}

/// The run lane's cold path for the same prompt: a fresh session, packed
/// prefill, the shared serial loop.
fn reference_bytes(
    backend: &Glm5NextBackend<'_, '_>,
    request: &ServeRequest,
    prompt: &str,
) -> Vec<u8> {
    let tokenizer = backend.prepared.artifact.tokenizer();
    let tokens: Vec<u32> = tokenizer
        .encode(prompt, false)
        .unwrap()
        .into_iter()
        .map(|id| id as u32)
        .collect();
    let mut session = backend.fresh_session().unwrap();
    let logits = session.prefill_packed(backend.ctx, &tokens).unwrap();
    let mut sampler = Sampler::new(render::sampling(request)).unwrap();
    let mut bytes = Vec::new();
    crate::generate_serial(
        logits,
        request.max_output_tokens.unwrap(),
        &CHAT_STOPS,
        &mut sampler,
        |token| {
            bytes.extend_from_slice(tokenizer.try_decode_piece_bytes_exact(token)?);
            Ok(())
        },
        |token| Ok(session.forward(backend.ctx, token as u32)?),
    )
    .unwrap();
    bytes
}

const QUESTION: &str = "What is 6 times 7? Reply with the number only.";

#[test]
#[ignore = "GLM53_GGUF GPU serve correctness under MTL_DEBUG_LAYER=1; production CLI lease, no server process"]
fn gpu_live_session_extends_resumes_and_resets() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let path = gguf_path();
    let source = GgufFile::open(&path).unwrap();
    let prepared = Prepared::new(&source, &invocation(&path, Some(1024), Some(64))).unwrap();
    assert_eq!(prepared.prefill_rows, 512);
    // A CLI test links the production library: this takes the real lease.
    let ctx = MetalContext::new().unwrap();
    let weights = load(&ctx, &source, &prepared).unwrap();
    let mut backend = Glm5NextBackend::new(&ctx, &weights, prepared, "glm".into());
    eprintln!("warm_up_ms={:.1}", backend.warm_up().unwrap());
    let greedy = |input: Value, effort: &str| {
        json!({"model":"glm","input":input,"reasoning":{"effort":effort},
            "temperature":0,"max_output_tokens":64})
    };

    // Cold request: the run lane's bytes, history = prompt + forwarded.
    let (first_req, first_prompt) = request(&backend, greedy(json!(QUESTION), "low"));
    assert_eq!(
        backend.output_protocol(&first_req),
        OutputProtocol::Glm5NextChat
    );
    let mut first = Sink::default();
    let started = Instant::now();
    let outcome = backend
        .generate(&first_req, &first_prompt, &mut first)
        .unwrap();
    eprintln!(
        "first request ms={:.1} usage={:?}",
        started.elapsed().as_secs_f64() * 1e3,
        outcome.usage
    );
    assert_eq!(outcome.usage.cached_tokens, 0);
    assert_eq!(
        first.bytes,
        reference_bytes(&backend, &first_req, &first_prompt)
    );
    assert_eq!(
        backend.history.len(),
        outcome.usage.input_tokens + outcome.usage.output_tokens - 1
    );
    let (reasoning, answer) = split(&first.bytes, outcome.end);
    assert!(answer.contains("42"), "{reasoning:?} / {answer:?}");

    // Replaying the exchange plus a new turn extends the live session.
    let replay = |next: &str| {
        json!([
            {"role":"user","content":QUESTION},
            {"type":"reasoning","content":reasoning},
            {"role":"assistant","content":answer},
            {"role":"user","content":next}])
    };
    let (second_req, second_prompt) = request(&backend, greedy(replay("And 6 times 8?"), "low"));
    assert!(second_prompt.starts_with(&first_prompt));
    let history = backend.history.len();
    let mut second = Sink::default();
    let outcome = backend
        .generate(&second_req, &second_prompt, &mut second)
        .unwrap();
    assert_eq!(
        outcome.usage.cached_tokens, history,
        "replayed history must re-tokenize to the consumed tokens"
    );
    let (_, second_answer) = split(&second.bytes, outcome.end);
    assert!(second_answer.contains("48"), "{second_answer:?}");

    // A cold run of the same prompt: semantically the same under Fast...
    backend.prefix_reuse = false;
    let mut cold = Sink::default();
    let outcome = backend
        .generate(&second_req, &second_prompt, &mut cold)
        .unwrap();
    assert_eq!(outcome.usage.cached_tokens, 0);
    assert!(split(&cold.bytes, outcome.end).1.contains("48"));
    eprintln!("fast warm==cold bytes: {}", cold.bytes == second.bytes);
    backend.prefix_reuse = true;
    // ...and bitwise under Exact, whose prefill matches serial decode.
    backend.lineage = PackedLineage::Exact;
    backend.session = None;
    backend.history.clear();
    let mut exact_first = Sink::default();
    let outcome = backend
        .generate(&first_req, &first_prompt, &mut exact_first)
        .unwrap();
    let (r, a) = split(&exact_first.bytes, outcome.end);
    let exact_replay = json!([
        {"role":"user","content":QUESTION},
        {"type":"reasoning","content":r},
        {"role":"assistant","content":a},
        {"role":"user","content":"And 6 times 8?"}]);
    let (exact_req, exact_prompt) = request(&backend, greedy(exact_replay, "low"));
    let history = backend.history.len();
    let mut warm = Sink::default();
    let outcome = backend
        .generate(&exact_req, &exact_prompt, &mut warm)
        .unwrap();
    assert_eq!(outcome.usage.cached_tokens, history);
    backend.prefix_reuse = false;
    let mut cold = Sink::default();
    backend
        .generate(&exact_req, &exact_prompt, &mut cold)
        .unwrap();
    assert_eq!(warm.bytes, cold.bytes, "Exact: warm continuation == cold");
    backend.prefix_reuse = true;
    backend.lineage = PackedLineage::Fast;
    backend.session = None;
    backend.history.clear();

    // A different effort rewrites the prefix: a fresh session.
    backend
        .generate(&first_req, &first_prompt, &mut Sink::default())
        .unwrap();
    let (high_req, high_prompt) = request(&backend, greedy(replay("And 6 times 8?"), "high"));
    let outcome = backend
        .generate(&high_req, &high_prompt, &mut Sink::default())
        .unwrap();
    assert_eq!(outcome.usage.cached_tokens, 0);

    // A two-chunk prompt cancelled between chunks keeps the committed chunk;
    // the retry resumes from it with the cold run's bytes.
    let long = format!(
        "{} Summarize the text above in five words.",
        "The quick brown fox jumps over the lazy dog. ".repeat(70)
    );
    let (long_req, long_prompt) = request(&backend, greedy(json!(long), "low"));
    backend.prefix_reuse = false;
    let mut long_cold = Sink::default();
    let cold_outcome = backend
        .generate(&long_req, &long_prompt, &mut long_cold)
        .unwrap();
    assert!(
        cold_outcome.usage.input_tokens > 512,
        "{:?}",
        cold_outcome.usage
    );
    backend.prefix_reuse = true;
    backend.session = None;
    backend.history.clear();
    // Ticks: admission, chunk 1, chunk 2.
    let mut cancelled = Sink {
        abort_tick: Some(3),
        ..Sink::default()
    };
    assert!(matches!(
        backend.generate(&long_req, &long_prompt, &mut cancelled),
        Err(BackendFailure::Aborted(_))
    ));
    assert_eq!(backend.history.len(), 512);
    let mut resumed = Sink::default();
    let outcome = backend
        .generate(&long_req, &long_prompt, &mut resumed)
        .unwrap();
    assert_eq!(outcome.usage.cached_tokens, 512);
    assert_eq!(resumed.bytes, long_cold.bytes);

    // An abort during decode clears the session; the retry is cold and equal.
    let mut dropped = Sink {
        abort_piece: true,
        ..Sink::default()
    };
    assert!(matches!(
        backend.generate(&first_req, &first_prompt, &mut dropped),
        Err(BackendFailure::Aborted(_))
    ));
    assert!(backend.history.is_empty() && backend.session.is_none());
    let mut retry = Sink::default();
    let outcome = backend
        .generate(&first_req, &first_prompt, &mut retry)
        .unwrap();
    assert_eq!(outcome.usage.cached_tokens, 0);
    assert_eq!(retry.bytes, first.bytes);

    // Over capacity is refused before the session: no ticks, history kept.
    let history = backend.history.clone();
    let (mut oversized, prompt) = request(&backend, greedy(replay("Next?"), "low"));
    oversized.max_output_tokens = Some(1024);
    let mut sink = Sink::default();
    assert!(matches!(
        backend.generate(&oversized, &prompt, &mut sink),
        Err(BackendFailure::Serve(_))
    ));
    assert_eq!(sink.ticks, 0);
    assert_eq!(backend.history, history);

    // JSON and SSE carry the reasoning item and the answer.
    for stream in [false, true] {
        let mut body = greedy(json!(QUESTION), "low");
        body["stream"] = stream.into();
        let response = wire(&mut backend, body);
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let body = response.split_once("\r\n\r\n").unwrap().1;
        let envelope = if stream {
            let block = body
                .split("\n\n")
                .find(|block| block.starts_with("event: response.completed\n"))
                .unwrap_or_else(|| panic!("{body}"));
            let data = block
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap();
            serde_json::from_str::<Value>(data).unwrap()["response"].clone()
        } else {
            serde_json::from_str::<Value>(body).unwrap()
        };
        let output = envelope["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["text"], reasoning);
        assert_eq!(output[1]["content"][0]["text"], answer);
        assert_eq!(envelope["reasoning"], json!({"effort":"low"}));
    }
}
