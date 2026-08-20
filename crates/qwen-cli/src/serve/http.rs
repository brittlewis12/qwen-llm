//! Serial HTTP/1.1 transport and request handler.
//!
//! Scope per docs/SERVE.md: loopback threat model, `Content-Length` JSON
//! request bodies only, `Connection: close` per response, one request in
//! flight (the accept loop lives with the subcommand wiring). Everything
//! model-shaped hides behind [`GenerationBackend`] so this layer tests
//! against a mock over real loopback sockets.

use super::events::{EventWrite, ResponseStream, ServeStats, SseWriter, StopReason, Usage};
use super::items::{ServeError, ServeRequest, parse_request};
use super::partition::StreamPartition;
use super::render::render_qwen_serve_prompt;
use serde_json::{Value, json};
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Streaming sink handed to the backend. `piece` delivers generated text;
/// `tick` is called between prefill chunks so the transport can heartbeat
/// and detect disconnects (cancellation = the returned error).
pub(crate) trait GenerationSink {
    fn piece(&mut self, text: &str) -> io::Result<()>;
    fn tick(&mut self) -> io::Result<()>;
}

#[derive(Debug, Clone)]
pub(crate) struct GenerationOutcome {
    pub(crate) stop_reason: StopReason,
    pub(crate) usage: Usage,
    pub(crate) stats: Option<ServeStats>,
}

pub(crate) trait GenerationBackend {
    fn model_id(&self) -> &str;
    /// True when the rendered prompt leaves `<think>` open, so generated
    /// bytes arrive headless (DeepSeek V4 thinking tiers). Default false.
    fn preopens_reasoning(&self, _request: &ServeRequest) -> bool {
        false
    }
    /// Family-specific prompt rendering. Defaults to the Qwen ChatML path.
    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        Ok(render_qwen_serve_prompt(request))
    }
    /// Render is already done; `prompt` is the exact model input. The
    /// backend streams raw generated text into `sink` and returns the
    /// outcome, or a spec error (e.g., context overflow → invalid_request
    /// per S0 F3).
    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure>;
}

pub(crate) struct TraceLog {
    file: BufWriter<File>,
}

impl TraceLog {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file: BufWriter::new(file),
        })
    }

    fn line(&mut self, value: Value) -> io::Result<()> {
        serde_json::to_writer(&mut self.file, &value)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        self.file.write_all(b"\n")?;
        self.file.flush()
    }
}

struct TraceSseWriter<'a, 'b> {
    inner: SseWriter<&'a TcpStream>,
    trace: Option<&'b mut TraceLog>,
}

impl<'a, 'b> TraceSseWriter<'a, 'b> {
    fn new(stream: &'a TcpStream, trace: Option<&'b mut TraceLog>) -> Self {
        Self {
            inner: SseWriter(stream),
            trace,
        }
    }

    fn heartbeat(&mut self) -> io::Result<()> {
        self.inner.heartbeat()?;
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.line(json!({"kind": "heartbeat"}))?;
        }
        Ok(())
    }

    fn done(&mut self) -> io::Result<()> {
        self.inner.done()?;
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.line(json!({"kind": "done"}))?;
        }
        Ok(())
    }
}

impl EventWrite for TraceSseWriter<'_, '_> {
    fn event(&mut self, event_type: &str, payload: Value) -> io::Result<()> {
        self.inner.event(event_type, payload.clone())?;
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.line(json!({
                "kind": "event",
                "event": event_type,
                "data": payload,
            }))?;
        }
        Ok(())
    }

    fn comment(&mut self) -> io::Result<()> {
        self.heartbeat()
    }
}

/// Backend failures split transport aborts (client gone; nothing left to
/// write) from spec errors (write an envelope / `response.failed`).
#[derive(Debug)]
pub(crate) enum BackendFailure {
    Serve(ServeError),
    Aborted(io::Error),
}

impl From<ServeError> for BackendFailure {
    fn from(error: ServeError) -> Self {
        Self::Serve(error)
    }
}

#[derive(Debug)]
pub(crate) struct HttpRequest {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) body: Vec<u8>,
}

fn transport_error(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_owned())
}

/// Read one request. `Ok(None)` on clean EOF before a request line.
pub(crate) fn read_http_request(
    reader: &mut BufReader<&TcpStream>,
) -> io::Result<Option<HttpRequest>> {
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(None);
    }
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| transport_error("empty request line"))?
        .to_owned();
    let path = parts
        .next()
        .ok_or_else(|| transport_error("request line missing path"))?
        .to_owned();

    let mut content_length: Option<usize> = None;
    let mut header_bytes = request_line.len();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(transport_error("connection closed inside headers"));
        }
        header_bytes += line.len();
        if header_bytes > MAX_HEADER_BYTES {
            return Err(transport_error("headers exceed limit"));
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            let name = name.trim().to_ascii_lowercase();
            let value = value.trim();
            if name == "content-length" {
                content_length = Some(
                    value
                        .parse()
                        .map_err(|_| transport_error("content-length is not a number"))?,
                );
            } else if name == "transfer-encoding" {
                return Err(transport_error(
                    "chunked request bodies are not supported; send content-length",
                ));
            }
        }
    }

    let mut body = Vec::new();
    if method == "POST" {
        let length =
            content_length.ok_or_else(|| transport_error("POST requires content-length"))?;
        if length > MAX_BODY_BYTES {
            return Err(transport_error("request body exceeds limit"));
        }
        body.resize(length, 0);
        reader.read_exact(&mut body)?;
    }
    Ok(Some(HttpRequest { method, path, body }))
}

fn write_json_response(stream: &mut &TcpStream, status: u16, body: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(body).expect("serialize response body");
    write!(
        stream,
        "HTTP/1.1 {status} {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        status_text(status),
        body.len(),
    )?;
    stream.write_all(&body)?;
    stream.flush()
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "",
    }
}

fn write_serve_error(stream: &mut &TcpStream, error: &ServeError) -> io::Result<()> {
    write_json_response(stream, error.status, &error.to_json())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn next_response_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    format!(
        "resp_{:08x}{:04x}",
        now_unix(),
        COUNTER.fetch_add(1, Ordering::Relaxed) & 0xffff
    )
}

struct CollectSink {
    pieces: Vec<String>,
}

impl GenerationSink for CollectSink {
    fn piece(&mut self, text: &str) -> io::Result<()> {
        self.pieces.push(text.to_owned());
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct StreamingSink<'a, 'b, W: EventWrite> {
    stream: &'a mut ResponseStream<'b, W>,
    partition: StreamPartition,
}

impl<W: EventWrite> GenerationSink for StreamingSink<'_, '_, W> {
    fn piece(&mut self, text: &str) -> io::Result<()> {
        let mut events = Vec::new();
        self.partition.push(text, &mut events);
        for event in &events {
            self.stream.on_partition(event)?;
        }
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        self.stream.heartbeat_if_idle()
    }
}

/// Handle one connection: read one request, dispatch, respond, close.
/// Returns Ok(()) even for request-level errors (they were answered);
/// Err means the connection is unusable (disconnect/cancellation).
pub(crate) fn handle_connection(
    stream: &TcpStream,
    backend: &mut dyn GenerationBackend,
    trace: Option<&mut TraceLog>,
) -> io::Result<()> {
    let mut reader = BufReader::new(stream);
    let mut writer = stream;
    let request = match read_http_request(&mut reader) {
        Ok(Some(request)) => request,
        Ok(None) => return Ok(()),
        Err(error) => {
            let envelope = ServeError::invalid_request(None, error.to_string());
            let _ = write_serve_error(&mut writer, &envelope);
            return Ok(());
        }
    };

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/v1/models") => write_json_response(
            &mut writer,
            200,
            &json!({
                "object": "list",
                "data": [{"id": backend.model_id(), "object": "model", "owned_by": "local"}],
            }),
        ),
        ("POST", "/v1/responses") => handle_responses(&request.body, stream, backend, trace),
        ("GET", _) | ("POST", _) => write_serve_error(
            &mut writer,
            &ServeError {
                status: 404,
                error_type: "not_found",
                code: None,
                param: None,
                message: format!("unknown path {}", request.path),
            },
        ),
        (method, _) => write_serve_error(
            &mut writer,
            &ServeError {
                status: 405,
                error_type: "invalid_request",
                code: None,
                param: None,
                message: format!("unsupported method {method}"),
            },
        ),
    }
}

fn handle_responses(
    body: &[u8],
    stream: &TcpStream,
    backend: &mut dyn GenerationBackend,
    mut trace: Option<&mut TraceLog>,
) -> io::Result<()> {
    let mut writer = stream;
    let parsed: Value = match serde_json::from_slice::<Value>(body) {
        Ok(parsed) => {
            if let Some(trace) = trace.as_deref_mut() {
                trace.line(json!({"kind": "request", "body": parsed.clone()}))?;
            }
            parsed
        }
        Err(error) => {
            if let Some(trace) = trace.as_deref_mut() {
                trace.line(json!({
                    "kind": "request",
                    "body": String::from_utf8_lossy(body),
                }))?;
            }
            return write_serve_error(
                &mut writer,
                &ServeError::invalid_request(None, format!("request body is not JSON: {error}")),
            );
        }
    };
    let request = match parse_request(&parsed) {
        Ok(request) => request,
        Err(error) => return write_serve_error(&mut writer, &error),
    };
    if request.model != backend.model_id() {
        return write_serve_error(
            &mut writer,
            &ServeError::model_not_found(&request.model, backend.model_id()),
        );
    }
    let prompt = match backend.render_prompt(&request) {
        Ok(prompt) => prompt,
        Err(error) => return write_serve_error(&mut writer, &error),
    };
    let response_id = next_response_id();
    let created_at = now_unix();

    // Resolved before the mutable generate borrow.
    let preopened = backend.preopens_reasoning(&request);
    let partition_mode = move || {
        if preopened {
            StreamPartition::with_preopened_reasoning()
        } else {
            StreamPartition::new()
        }
    };
    if !request.stream {
        let mut sink = CollectSink { pieces: Vec::new() };
        match backend.generate(&request, &prompt, &mut sink) {
            Ok(outcome) => {
                let mut partition = partition_mode();
                let mut partition_events = Vec::new();
                for piece in &sink.pieces {
                    partition.push(piece, &mut partition_events);
                }
                partition.finish(&mut partition_events);
                let envelope = super::events::build_response_object(
                    &request,
                    response_id,
                    created_at,
                    &partition_events,
                    outcome.stop_reason,
                    outcome.usage,
                    outcome.stats.as_ref().filter(|_| request.echo_stats),
                )?;
                write_json_response(&mut writer, 200, &envelope)
            }
            Err(BackendFailure::Serve(error)) => write_serve_error(&mut writer, &error),
            Err(BackendFailure::Aborted(error)) => Err(error),
        }
    } else {
        write!(
            writer,
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-store\r\nconnection: close\r\n\r\n"
        )?;
        writer.flush()?;
        let mut sse = TraceSseWriter::new(stream, trace);
        // Heartbeat immediately after admission (SERVE.md gate 3), then on
        // ticks between prefill chunks.
        sse.heartbeat()?;
        let mut response = ResponseStream::begin(
            &mut sse,
            response_id,
            request.model.clone(),
            created_at,
            super::events::envelope_echo(&request),
        )?;
        response.set_allowed_tools(request.allowed_tools.clone());
        let mut sink = StreamingSink {
            stream: &mut response,
            partition: partition_mode(),
        };
        let outcome = backend.generate(&request, &prompt, &mut sink);
        let StreamingSink { partition, .. } = sink;
        let mut events = Vec::new();
        partition.finish(&mut events);
        for event in &events {
            response.on_partition(event)?;
        }
        match outcome {
            Ok(outcome) => {
                response.finish(
                    outcome.stop_reason,
                    outcome.usage,
                    outcome.stats.as_ref().filter(|_| request.echo_stats),
                )?;
                sse.done()
            }
            Err(BackendFailure::Serve(error)) => {
                response.fail(&error)?;
                sse.done()
            }
            Err(BackendFailure::Aborted(error)) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    struct MockBackend {
        model: String,
        pieces: Vec<String>,
        stop_reason: StopReason,
        fail_with: Option<ServeError>,
    }

    impl MockBackend {
        fn new(pieces: &[&str], stop_reason: StopReason) -> Self {
            Self {
                model: "qwen-test".into(),
                pieces: pieces.iter().map(|s| s.to_string()).collect(),
                stop_reason,
                fail_with: None,
            }
        }
    }

    impl GenerationBackend for MockBackend {
        fn model_id(&self) -> &str {
            &self.model
        }
        fn generate(
            &mut self,
            _request: &ServeRequest,
            _prompt: &str,
            sink: &mut dyn GenerationSink,
        ) -> Result<GenerationOutcome, BackendFailure> {
            if let Some(error) = self.fail_with.clone() {
                return Err(error.into());
            }
            sink.tick().map_err(BackendFailure::Aborted)?;
            for piece in &self.pieces {
                sink.piece(piece).map_err(BackendFailure::Aborted)?;
            }
            Ok(GenerationOutcome {
                stop_reason: self.stop_reason,
                usage: Usage {
                    input_tokens: 7,
                    output_tokens: 3,
                    cached_tokens: 0,
                },
                stats: Some(ServeStats {
                    matched_tokens: 5,
                    restore_ms: 1.25,
                    prompt_tokens: 7,
                }),
            })
        }
    }

    fn roundtrip(mut backend: MockBackend, request: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(&stream, &mut backend, None).unwrap();
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        server.join().unwrap();
        response
    }

    fn post(path: &str, body: &str) -> String {
        format!(
            "POST {path} HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
            body.len(),
        )
    }

    fn body_of(response: &str) -> &str {
        response.split("\r\n\r\n").nth(1).unwrap()
    }

    #[test]
    fn trace_log_writes_one_json_object_per_line() {
        let path = std::env::temp_dir().join(format!(
            "serve_trace_{}_{}.jsonl",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_file(&path);
        {
            let mut trace = TraceLog::open(&path).expect("open trace");
            trace
                .line(json!({"kind": "request", "body": {"model": "m"}}))
                .unwrap();
            trace
                .line(json!({"kind": "event", "event": "response.created"}))
                .unwrap();
            trace.line(json!({"kind": "done"})).unwrap();
        }
        let contents = std::fs::read_to_string(&path).expect("read trace");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 3, "one object per line");
        for line in &lines {
            serde_json::from_str::<Value>(line).expect("each line is standalone JSON");
        }
        assert_eq!(
            serde_json::from_str::<Value>(lines[0]).unwrap()["body"]["model"],
            "m"
        );
        // Appends rather than truncating, so a restart keeps history.
        {
            let mut trace = TraceLog::open(&path).expect("reopen trace");
            trace.line(json!({"kind": "heartbeat"})).unwrap();
        }
        assert_eq!(
            std::fs::read_to_string(&path).unwrap().lines().count(),
            4,
            "reopen must append"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tracing_does_not_alter_the_wire_bytes() {
        // The trace tees; a traced stream must be byte-identical to an
        // untraced one, or debugging output would change what clients see.
        let request = post(
            "/v1/responses",
            r#"{"model":"qwen-test","input":"hi","stream":true}"#,
        );
        let untraced = roundtrip(
            MockBackend::new(&["<think>\np\n</think>\n\nanswer"], StopReason::Eos),
            &request,
        );
        let path =
            std::env::temp_dir().join(format!("serve_trace_wire_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let traced = roundtrip_traced(
            MockBackend::new(&["<think>\np\n</think>\n\nanswer"], StopReason::Eos),
            &request,
            &path,
        );
        let strip_ids = |text: &str| regex_lite_replace(text);
        assert_eq!(
            strip_ids(&untraced),
            strip_ids(&traced),
            "trace changed the response bytes"
        );
        let traced_lines = std::fs::read_to_string(&path).unwrap();
        assert!(traced_lines.lines().count() > 3, "trace captured events");
        assert!(traced_lines.contains("\"kind\":\"request\""));
        assert!(traced_lines.contains("response.output_text.delta"));
        assert!(traced_lines.contains("\"kind\":\"done\""));
        let _ = std::fs::remove_file(&path);
    }

    /// Response ids and timestamps differ per request; blank them so the
    /// comparison is about framing, not identity.
    fn regex_lite_replace(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(index) = rest.find("resp_") {
            out.push_str(&rest[..index]);
            out.push_str("resp_X");
            rest = &rest[index + 5..];
            let skip = rest
                .find(|c: char| !c.is_ascii_hexdigit())
                .unwrap_or(rest.len());
            rest = &rest[skip..];
        }
        out.push_str(rest);
        let mut deadline = String::with_capacity(out.len());
        let mut rest = out.as_str();
        while let Some(index) = rest.find("\"created_at\":") {
            deadline.push_str(&rest[..index]);
            deadline.push_str("\"created_at\":0");
            rest = &rest[index + 13..];
            let skip = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            rest = &rest[skip..];
        }
        deadline.push_str(rest);
        deadline
    }

    fn roundtrip_traced(mut backend: MockBackend, request: &str, trace_path: &Path) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let trace_path = trace_path.to_path_buf();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut trace = TraceLog::open(&trace_path).expect("open trace");
            handle_connection(&stream, &mut backend, Some(&mut trace)).unwrap();
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        server.join().unwrap();
        response
    }

    #[test]
    fn models_endpoint_lists_the_loaded_model() {
        let response = roundtrip(
            MockBackend::new(&[], StopReason::Eos),
            "GET /v1/models HTTP/1.1\r\nhost: x\r\n\r\n",
        );
        assert!(response.starts_with("HTTP/1.1 200"));
        let parsed: Value = serde_json::from_str(body_of(&response)).unwrap();
        assert_eq!(parsed["data"][0]["id"], "qwen-test");
    }

    #[test]
    fn non_stream_response_returns_final_envelope() {
        let response = roundtrip(
            MockBackend::new(&["<think>\np\n</think>\n\nanswer"], StopReason::Eos),
            &post(
                "/v1/responses",
                r#"{"model":"qwen-test","input":"hi","x_qwen":{"stats":true}}"#,
            ),
        );
        assert!(response.starts_with("HTTP/1.1 200"));
        let parsed: Value = serde_json::from_str(body_of(&response)).unwrap();
        assert_eq!(parsed["status"], "completed");
        assert_eq!(parsed["output"][0]["type"], "reasoning");
        assert_eq!(parsed["output"][1]["content"][0]["text"], "\n\nanswer");
        assert_eq!(parsed["usage"]["total_tokens"], 10);
        assert_eq!(parsed["x_qwen"]["matched_tokens"], 5);
    }

    #[test]
    fn stream_response_emits_sse_with_heartbeat_and_done() {
        let response = roundtrip(
            MockBackend::new(&["<think>\np\n</think>\n\nanswer"], StopReason::Eos),
            &post(
                "/v1/responses",
                r#"{"model":"qwen-test","input":"hi","stream":true}"#,
            ),
        );
        assert!(response.contains("content-type: text/event-stream"));
        let payload = body_of(&response);
        assert!(
            payload.starts_with(": ping\n\n"),
            "admission heartbeat first"
        );
        assert!(payload.contains("event: response.created\n"));
        assert!(payload.contains("event: response.reasoning.delta\n"));
        assert!(payload.contains("event: response.output_text.delta\n"));
        assert!(payload.contains("event: response.completed\n"));
        assert!(payload.ends_with("data: [DONE]\n\n"));
        let created_index = payload.find("response.created").unwrap();
        let completed_index = payload.find("response.completed").unwrap();
        assert!(created_index < completed_index);
    }

    #[test]
    fn stream_trace_records_request_events_and_done() {
        let path = std::env::temp_dir().join(format!("qwen-trace-{}.jsonl", next_response_id()));
        let trace_path = path.clone();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut backend = MockBackend::new(&["answer"], StopReason::Eos);
            let mut trace = TraceLog::open(&trace_path).unwrap();
            handle_connection(&stream, &mut backend, Some(&mut trace)).unwrap();
        });
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .write_all(
                post(
                    "/v1/responses",
                    r#"{"model":"qwen-test","input":"hi","stream":true}"#,
                )
                .as_bytes(),
            )
            .unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        server.join().unwrap();

        let lines = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lines[0]["kind"], "request");
        assert_eq!(lines[0]["body"]["stream"], true);
        assert!(
            lines
                .iter()
                .any(|line| { line["kind"] == "event" && line["event"] == "response.created" })
        );
        assert_eq!(lines.last().unwrap()["kind"], "done");
        assert!(response.ends_with("data: [DONE]\n\n"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn spec_errors_map_to_envelopes() {
        for (body, expect_status, expect_fragment) in [
            (r#"{"model":"qwen-test"}"#, "400", "input is required"),
            (r#"{"model":"other","input":"q"}"#, "404", "model_not_found"),
            (
                r#"{"model":"qwen-test","input":"q","store":true}"#,
                "400",
                "store",
            ),
            ("not json", "400", "not JSON"),
        ] {
            let response = roundtrip(
                MockBackend::new(&[], StopReason::Eos),
                &post("/v1/responses", body),
            );
            assert!(
                response.starts_with(&format!("HTTP/1.1 {expect_status}")),
                "{body} → {response}"
            );
            assert!(body_of(&response).contains(expect_fragment), "{response}");
        }

        let response = roundtrip(
            MockBackend::new(&[], StopReason::Eos),
            "GET /v1/nope HTTP/1.1\r\nhost: x\r\n\r\n",
        );
        assert!(response.starts_with("HTTP/1.1 404"));
    }

    #[test]
    fn mid_stream_backend_error_becomes_response_failed() {
        let mut backend = MockBackend::new(&[], StopReason::Eos);
        backend.fail_with = Some(ServeError::invalid_request(
            Some("input"),
            "prompt 40000 exceeds max context 32768",
        ));
        let response = roundtrip(
            backend,
            &post(
                "/v1/responses",
                r#"{"model":"qwen-test","input":"q","stream":true}"#,
            ),
        );
        assert!(response.contains("event: response.failed\n"));
        assert!(response.contains("exceeds max context"));
        assert!(body_of(&response).ends_with("data: [DONE]\n\n"));
    }

    #[test]
    fn token_limit_streams_incomplete_terminal() {
        let response = roundtrip(
            MockBackend::new(&["<think>\ntrunc"], StopReason::TokenLimit),
            &post(
                "/v1/responses",
                r#"{"model":"qwen-test","input":"q","stream":true}"#,
            ),
        );
        assert!(response.contains("event: response.incomplete\n"));
        assert!(response.contains("max_output_tokens"));
    }

    #[test]
    fn post_without_content_length_is_answered_with_envelope() {
        let response = roundtrip(
            MockBackend::new(&[], StopReason::Eos),
            "POST /v1/responses HTTP/1.1\r\nhost: x\r\n\r\n",
        );
        assert!(response.starts_with("HTTP/1.1 400"));
        assert!(response.contains("content-length"));
    }
}
