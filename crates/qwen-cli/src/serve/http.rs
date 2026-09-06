//! Serial HTTP/1.1 transport and request handler.
//!
//! Scope per docs/SERVE.md: loopback threat model, `Content-Length` JSON
//! request bodies only, `Connection: close` per response, one request in
//! flight (the accept loop lives with the subcommand wiring). Everything
//! model-shaped hides behind [`GenerationBackend`] so this layer tests
//! against a mock over real loopback sockets.

use super::events::{EventWrite, ResponseStream, ServeStats, SseWriter, StopReason, Usage};
use super::items::{ServeError, ServeRequest, parse_request};
use super::output_partition::{GenerationEnd, OutputPartition, OutputProtocol};
use super::render::render_qwen_serve_prompt;
use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::net::{Shutdown, TcpStream};
use std::path::Path;
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_REQUEST_LINE_BYTES: usize = 8 * 1024;
const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
const SOCKET_READ_TIMEOUT: Duration = Duration::from_secs(35);
const SOCKET_WRITE_TIMEOUT: Duration = Duration::from_secs(30);
const REQUEST_READ_DEADLINE: Duration = Duration::from_secs(30);
const TRACE_QUEUE_CAPACITY: usize = 8;
const TRACE_SHUTDOWN_GRACE: Duration = Duration::from_millis(250);
const TRACE_SHUTDOWN_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Streaming sink handed to the backend. `piece` delivers generated text;
/// `tick` is called between prefill chunks so the transport can heartbeat
/// and detect disconnects (cancellation = the returned error).
pub(crate) trait GenerationSink {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()>;
    fn tick(&mut self) -> io::Result<()>;
}

#[derive(Debug, Clone)]
pub(crate) struct GenerationOutcome {
    pub(crate) end: GenerationEnd,
    pub(crate) usage: Usage,
    pub(crate) stats: Option<ServeStats>,
}

pub(crate) trait GenerationBackend {
    fn model_id(&self) -> &str;
    /// Resolve family defaults before prompt rendering and response echoes.
    fn normalize_request(&self, _request: &mut ServeRequest) -> Result<(), ServeError> {
        Ok(())
    }
    /// Family-owned grammar for exact generated token bytes.
    fn output_protocol(&self, _request: &ServeRequest) -> OutputProtocol {
        OutputProtocol::Qwen {
            preopened_reasoning: false,
            parse_tools: true,
        }
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
    sender: Option<SyncSender<Value>>,
    worker: Option<JoinHandle<()>>,
}

impl TraceLog {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "trace path is not a regular file",
            ));
        }
        if metadata.uid() != unsafe { libc::geteuid() } {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "trace file must be owned by the current user",
            ));
        }
        if metadata.permissions().mode() & 0o077 != 0 {
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        }
        let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFL, flags & !libc::O_NONBLOCK) } < 0 {
            return Err(io::Error::last_os_error());
        }
        let (sender, receiver) = sync_channel::<Value>(TRACE_QUEUE_CAPACITY);
        let worker = std::thread::Builder::new()
            .name("qwen-sse-trace".into())
            .spawn(move || {
                let mut file = BufWriter::new(file);
                for value in receiver {
                    let result = (|| {
                        serde_json::to_writer(&mut file, &value)
                            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                        file.write_all(b"\n")?;
                        file.flush()
                    })();
                    if let Err(error) = result {
                        tracing::warn!(target: "qwen_diag", "serve: disabling SSE trace after write failure: {error}");
                        break;
                    }
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
        })
    }

    fn is_enabled(&self) -> bool {
        self.sender.is_some()
    }

    fn line(&mut self, make_value: impl FnOnce() -> Value) {
        let Some(sender) = self.sender.as_ref() else {
            return;
        };
        match sender.try_send(make_value()) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                tracing::warn!(target: "qwen_diag", "serve: disabling SSE trace because its bounded queue is full");
                self.sender = None;
            }
            Err(TrySendError::Disconnected(_)) => {
                tracing::warn!(target: "qwen_diag", "serve: disabling SSE trace because its writer stopped");
                self.sender = None;
            }
        }
    }
}

fn join_trace_worker_with_grace(
    worker: JoinHandle<()>,
    grace: Duration,
) -> Option<std::thread::Result<()>> {
    let started = Instant::now();
    while !worker.is_finished() {
        let remaining = grace.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return None;
        }
        std::thread::sleep(remaining.min(TRACE_SHUTDOWN_POLL_INTERVAL));
    }
    Some(worker.join())
}

impl Drop for TraceLog {
    fn drop(&mut self) {
        self.sender = None;
        let Some(worker) = self.worker.take() else {
            return;
        };
        match join_trace_worker_with_grace(worker, TRACE_SHUTDOWN_GRACE) {
            Some(Ok(())) => {}
            Some(Err(_)) => {
                tracing::warn!(target: "qwen_diag", "serve: SSE trace writer panicked");
            }
            None => {
                tracing::warn!(target: "qwen_diag", "serve: SSE trace writer did not stop within {} ms; detaching so shutdown can continue (queued trace events may be lost)", TRACE_SHUTDOWN_GRACE.as_millis());
            }
        }
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
            trace.line(|| json!({"kind": "heartbeat"}));
        }
        Ok(())
    }

    fn done(&mut self) -> io::Result<()> {
        self.inner.done()?;
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.line(|| json!({"kind": "done"}));
        }
        Ok(())
    }
}

impl EventWrite for TraceSseWriter<'_, '_> {
    fn event(&mut self, event_type: &str, payload: Value) -> io::Result<()> {
        let trace_enabled = self.trace.as_deref().is_some_and(TraceLog::is_enabled);
        if trace_enabled {
            self.inner.event(event_type, payload.clone())?;
        } else {
            return self.inner.event(event_type, payload);
        }
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.line(|| {
                json!({
                    "kind": "event",
                    "event": event_type,
                    "data": payload,
                })
            });
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

fn is_http_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
}

fn valid_host_authority(value: &[u8]) -> bool {
    let Ok(authority) = std::str::from_utf8(value) else {
        return false;
    };
    if authority.is_empty() {
        return false;
    }
    if let Some(rest) = authority.strip_prefix('[') {
        let Some((host, suffix)) = rest.split_once(']') else {
            return false;
        };
        if host.parse::<std::net::Ipv6Addr>().is_err() {
            return false;
        }
        return suffix.is_empty() || suffix.strip_prefix(':').is_some_and(valid_host_port);
    }
    if authority.contains('[') || authority.contains(']') || authority.matches(':').count() > 1 {
        return false;
    }
    let (host, port) = authority
        .rsplit_once(':')
        .map_or((authority, None), |(host, port)| (host, Some(port)));
    if !valid_host_name(host) {
        return false;
    }
    port.is_none_or(valid_host_port)
}

fn valid_host_name(host: &str) -> bool {
    if host.parse::<std::net::Ipv4Addr>().is_ok() {
        return true;
    }
    let host = host.strip_suffix('.').unwrap_or(host);
    !host.is_empty()
        && host.len() <= 253
        && host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
                && label
                    .as_bytes()
                    .first()
                    .is_some_and(u8::is_ascii_alphanumeric)
                && label
                    .as_bytes()
                    .last()
                    .is_some_and(u8::is_ascii_alphanumeric)
        })
}

fn valid_host_port(port: &str) -> bool {
    !port.is_empty()
        && port.bytes().all(|byte| byte.is_ascii_digit())
        && port.parse::<u16>().is_ok()
}

fn read_bounded_line<R: BufRead>(reader: &mut R, limit: usize) -> io::Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Ok(Some(line))
            };
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if take > limit.saturating_sub(line.len()) {
            return Err(transport_error("HTTP line exceeds limit"));
        }
        line.extend_from_slice(&available[..take]);
        reader.consume(take);
        if line.last() == Some(&b'\n') {
            return Ok(Some(line));
        }
    }
}

/// Read one request. `Ok(None)` on clean EOF before a request line.
pub(crate) fn read_http_request<R: BufRead>(reader: &mut R) -> io::Result<Option<HttpRequest>> {
    let request_line_bytes = match read_bounded_line(reader, MAX_REQUEST_LINE_BYTES)? {
        Some(line) => line,
        None => return Ok(None),
    };
    let request_line = std::str::from_utf8(&request_line_bytes)
        .map_err(|_| transport_error("request line is not UTF-8"))?;
    let request_line = request_line
        .strip_suffix("\r\n")
        .ok_or_else(|| transport_error("request line must end with CRLF"))?;
    let mut parts = request_line.split(' ');
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();
    let version = parts.next().unwrap_or_default();
    if method.is_empty() || path.is_empty() || parts.next().is_some() {
        return Err(transport_error("malformed HTTP request line"));
    }
    if !method.bytes().all(is_http_token_byte)
        || path.bytes().any(|byte| byte <= b' ' || byte == 0x7f)
    {
        return Err(transport_error("invalid HTTP request line"));
    }
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1") {
        return Err(transport_error("unsupported HTTP version"));
    }
    let method = method.to_owned();
    let path = path.to_owned();

    let mut content_length: Option<usize> = None;
    let mut host_count = 0_usize;
    let mut header_bytes = request_line_bytes.len();
    loop {
        let line = read_bounded_line(reader, MAX_HEADER_BYTES.saturating_sub(header_bytes))?
            .ok_or_else(|| transport_error("connection closed inside headers"))?;
        header_bytes += line.len();
        let line = line
            .strip_suffix(b"\r\n")
            .ok_or_else(|| transport_error("header line must end with CRLF"))?;
        if line.is_empty() {
            break;
        }
        let separator = line
            .iter()
            .position(|byte| *byte == b':')
            .ok_or_else(|| transport_error("malformed HTTP header"))?;
        let (name, value) = (&line[..separator], &line[separator + 1..]);
        if name.is_empty() || !name.iter().copied().all(is_http_token_byte) {
            return Err(transport_error("invalid HTTP header name"));
        }
        if value
            .iter()
            .copied()
            .any(|byte| (byte < b' ' && byte != b'\t') || byte == 0x7f)
        {
            return Err(transport_error("invalid HTTP header value"));
        }
        let value = value.strip_prefix(b" ").unwrap_or(value);
        let value = value
            .iter()
            .copied()
            .skip_while(|byte| matches!(byte, b' ' | b'\t'))
            .collect::<Vec<_>>();
        let value = value
            .iter()
            .rposition(|byte| !matches!(byte, b' ' | b'\t'))
            .map_or(&[][..], |last| &value[..=last]);
        if name.eq_ignore_ascii_case(b"content-length") {
            if content_length.is_some() {
                return Err(transport_error("duplicate content-length"));
            }
            if value.is_empty() || !value.iter().all(u8::is_ascii_digit) {
                return Err(transport_error("content-length is not a number"));
            }
            content_length = Some(
                std::str::from_utf8(value)
                    .expect("ASCII content-length")
                    .parse()
                    .map_err(|_| transport_error("content-length is not a number"))?,
            );
        } else if name.eq_ignore_ascii_case(b"transfer-encoding") {
            if !value.is_ascii() {
                return Err(transport_error("transfer-encoding must be ASCII"));
            }
            return Err(transport_error(
                "transfer-encoding request bodies are not supported; send content-length",
            ));
        } else if name.eq_ignore_ascii_case(b"host") {
            host_count += 1;
            if host_count > 1 {
                return Err(transport_error("duplicate host header"));
            }
            if !valid_host_authority(value) {
                return Err(transport_error("host is not a valid authority"));
            }
        }
    }
    if version == "HTTP/1.1" && host_count != 1 {
        return Err(transport_error("HTTP/1.1 requires exactly one host header"));
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

pub(crate) fn configure_stream(stream: &TcpStream) -> io::Result<()> {
    stream.set_read_timeout(Some(SOCKET_READ_TIMEOUT))?;
    stream.set_write_timeout(Some(SOCKET_WRITE_TIMEOUT))
}

fn read_http_request_with_deadline(
    stream: &TcpStream,
    deadline: Duration,
) -> io::Result<Option<HttpRequest>> {
    let watchdog_stream = stream.try_clone()?;
    let (cancel_sender, cancel_receiver) = sync_channel::<()>(0);
    let timed_out = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watchdog_timed_out = std::sync::Arc::clone(&timed_out);
    let watchdog = std::thread::Builder::new()
        .name("qwen-http-read-deadline".into())
        .spawn(move || {
            if matches!(
                cancel_receiver.recv_timeout(deadline),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ) {
                watchdog_timed_out.store(true, std::sync::atomic::Ordering::Release);
                let _ = watchdog_stream.shutdown(Shutdown::Read);
            }
        })?;
    let result = read_http_request(&mut BufReader::new(stream));
    let _ = cancel_sender.send(());
    watchdog
        .join()
        .map_err(|_| io::Error::other("HTTP read-deadline watchdog panicked"))?;
    if is_request_read_timeout(
        timed_out.load(std::sync::atomic::Ordering::Acquire),
        result.as_ref().err(),
    ) {
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "HTTP request read deadline exceeded",
        ))
    } else {
        result
    }
}

fn is_request_read_timeout(watchdog_fired: bool, error: Option<&io::Error>) -> bool {
    watchdog_fired
        || error.is_some_and(|error| {
            matches!(
                error.kind(),
                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
            )
        })
}

pub(crate) fn write_busy_response(mut stream: &TcpStream) -> io::Result<()> {
    let body = br#"{"error":{"type":"server_busy","code":"server_busy","param":"","message":"server is processing another request"}}"#;
    write!(
        stream,
        "HTTP/1.1 503 Service Unavailable\r\ncontent-type: application/json\r\ncontent-length: {}\r\nretry-after: 1\r\nconnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        408 => "Request Timeout",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
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

fn probe_peer(stream: &TcpStream) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let mut byte = 0_u8;
    loop {
        // MSG_DONTWAIT is scoped to this recv and does not change socket flags.
        let received = unsafe {
            libc::recv(
                stream.as_raw_fd(),
                (&mut byte as *mut u8).cast(),
                1,
                libc::MSG_PEEK | libc::MSG_DONTWAIT,
            )
        };
        if received > 0 {
            return Ok(());
        }
        if received == 0 {
            // A receive-side FIN may be a valid request-side half-close from
            // a client that is still waiting for its response. Before the
            // first non-stream response write, graceful full close and this
            // half-close are indistinguishable, so cancellation is best effort.
            return Ok(());
        }
        let error = io::Error::last_os_error();
        match error.kind() {
            io::ErrorKind::WouldBlock => return Ok(()),
            io::ErrorKind::Interrupted => continue,
            _ => return Err(error),
        }
    }
}

fn shutdown_checkpoint() -> io::Result<()> {
    crate::shutdown::checkpoint()
        .map_err(|error| io::Error::new(io::ErrorKind::Interrupted, error.to_string()))
}

struct CollectSink<'a> {
    pieces: Vec<Vec<u8>>,
    stream: &'a TcpStream,
}

impl GenerationSink for CollectSink<'_> {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        shutdown_checkpoint()?;
        probe_peer(self.stream)?;
        self.pieces.push(bytes.to_owned());
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        shutdown_checkpoint()?;
        probe_peer(self.stream)
    }
}

struct StreamingSink<'a, 'b, W: EventWrite> {
    stream: &'a mut ResponseStream<'b, W>,
    partition: OutputPartition,
}

impl<W: EventWrite> GenerationSink for StreamingSink<'_, '_, W> {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        shutdown_checkpoint()?;
        let mut events = Vec::new();
        self.partition.push(bytes, &mut events);
        for event in &events {
            self.stream.on_partition(event)?;
        }
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        shutdown_checkpoint()?;
        self.stream.heartbeat_if_idle()
    }
}

/// Handle one connection: read one request, dispatch, respond, close.
/// Returns Ok(()) even for request-level errors (they were answered);
/// Err means the connection is unusable (disconnect/cancellation).
#[cfg(test)]
fn handle_connection(
    stream: &TcpStream,
    backend: &mut dyn GenerationBackend,
    trace: Option<&mut TraceLog>,
) -> io::Result<()> {
    handle_connection_with_completion(stream, backend, trace, || {})
}

pub(crate) fn handle_connection_with_completion(
    stream: &TcpStream,
    backend: &mut dyn GenerationBackend,
    trace: Option<&mut TraceLog>,
    mut on_completion: impl FnMut(),
) -> io::Result<()> {
    configure_stream(stream)?;
    let mut writer = stream;
    let request = match read_http_request_with_deadline(stream, REQUEST_READ_DEADLINE) {
        Ok(Some(request)) => request,
        Ok(None) => return Ok(()),
        Err(error) => {
            let mut envelope = ServeError::invalid_request(None, error.to_string());
            if error.kind() == io::ErrorKind::TimedOut {
                envelope.status = 408;
                envelope.error_type = "request_timeout";
            }
            let _ = write_serve_error(&mut writer, &envelope);
            return Ok(());
        }
    };

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/v1/models") => {
            on_completion();
            write_json_response(
                &mut writer,
                200,
                &json!({
                    "object": "list",
                    "data": [{"id": backend.model_id(), "object": "model", "owned_by": "local"}],
                }),
            )
        }
        ("POST", "/v1/responses") => {
            handle_responses(&request.body, stream, backend, trace, &mut on_completion)
        }
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
    on_completion: &mut dyn FnMut(),
) -> io::Result<()> {
    let mut writer = stream;
    let parsed: Value = match serde_json::from_slice::<Value>(body) {
        Ok(parsed) => {
            if let Some(trace) = trace.as_deref_mut()
                && trace.is_enabled()
            {
                trace.line(|| json!({"kind": "request", "body": parsed.clone()}));
            }
            parsed
        }
        Err(error) => {
            if let Some(trace) = trace.as_deref_mut()
                && trace.is_enabled()
            {
                trace.line(|| {
                    json!({
                        "kind": "request",
                        "body": String::from_utf8_lossy(body),
                    })
                });
            }
            return write_serve_error(
                &mut writer,
                &ServeError::invalid_request(None, format!("request body is not JSON: {error}")),
            );
        }
    };
    let mut request = match parse_request(&parsed) {
        Ok(request) => request,
        Err(error) => return write_serve_error(&mut writer, &error),
    };
    if request.model != backend.model_id() {
        return write_serve_error(
            &mut writer,
            &ServeError::model_not_found(&request.model, backend.model_id()),
        );
    }
    if let Err(error) = backend.normalize_request(&mut request) {
        return write_serve_error(&mut writer, &error);
    }
    let prompt = match backend.render_prompt(&request) {
        Ok(prompt) => prompt,
        Err(error) => return write_serve_error(&mut writer, &error),
    };
    let response_id = next_response_id();
    let created_at = now_unix();

    // Resolved before the mutable generate borrow.
    let output_protocol = backend.output_protocol(&request);
    if !request.stream {
        let mut sink = CollectSink {
            pieces: Vec::new(),
            stream,
        };
        match backend.generate(&request, &prompt, &mut sink) {
            Ok(outcome) => {
                let mut partition = OutputPartition::new(output_protocol);
                let mut partition_events = Vec::new();
                for piece in &sink.pieces {
                    partition.push(piece, &mut partition_events);
                }
                if let Err(error) = partition.finish(outcome.end, &mut partition_events) {
                    return write_serve_error(&mut writer, &error);
                }
                let envelope = super::events::build_response_object(
                    &request,
                    response_id,
                    created_at,
                    &partition_events,
                    response_stop_reason(outcome.end),
                    outcome.usage,
                    outcome.stats.as_ref().filter(|_| request.echo_stats),
                )?;
                on_completion();
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
            partition: OutputPartition::new(output_protocol),
        };
        let outcome = backend.generate(&request, &prompt, &mut sink);
        let StreamingSink { partition, .. } = sink;
        match outcome {
            Ok(outcome) => {
                let mut events = Vec::new();
                if let Err(error) = partition.finish(outcome.end, &mut events) {
                    response.fail(&error)?;
                    return sse.done();
                }
                for event in &events {
                    response.on_partition(event)?;
                }
                on_completion();
                response.finish(
                    response_stop_reason(outcome.end),
                    outcome.usage,
                    outcome.stats.as_ref().filter(|_| request.echo_stats),
                )?;
                sse.done()
            }
            Err(BackendFailure::Serve(error)) => {
                let mut events = Vec::new();
                partition.abort(&mut events);
                for event in &events {
                    response.on_partition(event)?;
                }
                response.fail(&error)?;
                sse.done()
            }
            Err(BackendFailure::Aborted(error)) => Err(error),
        }
    }
}

fn response_stop_reason(end: GenerationEnd) -> StopReason {
    match end {
        GenerationEnd::StopToken(_) => StopReason::Eos,
        GenerationEnd::TokenLimit => StopReason::TokenLimit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;

    struct MockBackend {
        model: String,
        pieces: Vec<String>,
        end: GenerationEnd,
        protocol: OutputProtocol,
        fail_with: Option<ServeError>,
    }

    impl MockBackend {
        fn new(pieces: &[&str], stop_reason: StopReason) -> Self {
            Self {
                model: "qwen-test".into(),
                pieces: pieces.iter().map(|s| s.to_string()).collect(),
                end: match stop_reason {
                    StopReason::Eos => GenerationEnd::StopToken(0),
                    StopReason::TokenLimit => GenerationEnd::TokenLimit,
                },
                protocol: OutputProtocol::Qwen {
                    preopened_reasoning: false,
                    parse_tools: true,
                },
                fail_with: None,
            }
        }

        fn muse(pieces: &[&str], end: GenerationEnd) -> Self {
            let mut backend = Self::new(pieces, StopReason::Eos);
            backend.end = end;
            backend.protocol = OutputProtocol::MuseAtem {
                eos_token_id: 1,
                eot_token_id: 2,
                declared_tools: vec!["weather_lookup".into()],
            };
            backend
        }
    }

    impl GenerationBackend for MockBackend {
        fn model_id(&self) -> &str {
            &self.model
        }
        fn output_protocol(&self, _request: &ServeRequest) -> OutputProtocol {
            self.protocol.clone()
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

    fn sse_payload(response: &str, event_type: &str) -> Value {
        let block = body_of(response)
            .split("\n\n")
            .find(|block| block.starts_with(&format!("event: {event_type}\n")))
            .unwrap_or_else(|| panic!("missing SSE event {event_type}"));
        let data = block
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("SSE event has data");
        serde_json::from_str(data).expect("SSE event data is JSON")
    }

    #[test]
    fn muse_non_stream_partitions_reasoning_visible_and_calls() {
        let backend = MockBackend::muse(
            &[
                " to=self<|message|>check<|eom|><|start|>assistant ",
                "to=weather_lookup<|message|><atem:function_calls>\n",
                "<atem:invoke name=\"weather_lookup\">\n<atem:parameter name=\"city\">Paris</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
            ],
            GenerationEnd::StopToken(2),
        );
        let body = json!({
            "model":"qwen-test",
            "input":"weather",
            "tools":[{"type":"function","name":"weather_lookup","parameters":{"type":"object"}}]
        });
        let response = roundtrip(backend, &post("/v1/responses", &body.to_string()));
        assert!(response.starts_with("HTTP/1.1 200"));
        let envelope: Value = serde_json::from_str(body_of(&response)).unwrap();
        let output = envelope["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["text"], "check");
        assert_eq!(output[1]["type"], "function_call");
        assert_eq!(output[1]["name"], "weather_lookup");
        assert_eq!(output[1]["arguments"], "{\"city\":\"Paris\"}");
        assert!(!body_of(&response).contains("<atem:"));
        assert!(!body_of(&response).contains("<|message|>"));
    }

    #[test]
    fn muse_malformed_completion_fails_and_truncated_header_stays_hidden() {
        let malformed = MockBackend::muse(&["raw answer"], GenerationEnd::StopToken(2));
        let body = json!({"model":"qwen-test","input":"hi"}).to_string();
        let response = roundtrip(malformed, &post("/v1/responses", &body));
        assert!(response.starts_with("HTTP/1.1 500"));
        assert!(body_of(&response).contains("invalid Muse ATEM model output"));

        let truncated = MockBackend::muse(&[" to=self<|mess"], GenerationEnd::TokenLimit);
        let response = roundtrip(truncated, &post("/v1/responses", &body));
        assert!(response.starts_with("HTTP/1.1 200"));
        let envelope: Value = serde_json::from_str(body_of(&response)).unwrap();
        assert_eq!(envelope["status"], "incomplete");
        assert!(envelope["output"].as_array().unwrap().is_empty());
        assert!(!body_of(&response).contains("<|mess"));
    }

    #[test]
    fn muse_stream_protocol_failure_emits_failed_terminal_and_done() {
        let backend = MockBackend::muse(
            &[concat!(
                " to=self<|message|>finished plan<|eom|>",
                "<|start|>assistant to=user<|message|>safe<|bad|>"
            )],
            GenerationEnd::StopToken(2),
        );
        let body = json!({"model":"qwen-test","input":"hi","stream":true}).to_string();
        let response = roundtrip(backend, &post("/v1/responses", &body));
        assert!(response.starts_with("HTTP/1.1 200"));
        assert!(response.contains("event: response.failed"));
        assert!(response.contains("data: [DONE]"));
        assert!(!response.contains("<|bad|>"));
        let failed = sse_payload(&response, "response.failed");
        let output = failed["response"]["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["status"], "completed");
        assert_eq!(output[0]["content"][0]["text"], "finished plan");
        assert_eq!(output[1]["type"], "message");
        assert_eq!(output[1]["status"], "incomplete");
        assert_eq!(output[1]["content"][0]["text"], "safe");
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
            trace.line(|| json!({"kind": "request", "body": {"model": "m"}}));
            trace.line(|| json!({"kind": "event", "event": "response.created"}));
            trace.line(|| json!({"kind": "done"}));
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
            trace.line(|| json!({"kind": "heartbeat"}));
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
    fn invalid_generation_controls_fail_before_stream_headers() {
        for body in [
            r#"{"model":"qwen-test","input":"hi","stream":true,"top_p":0}"#,
            r#"{"model":"qwen-test","input":"hi","stream":true,"temperature":-1}"#,
            r#"{"model":"qwen-test","input":"hi","stream":true,"temperature":-1e-50}"#,
            r#"{"model":"qwen-test","input":"hi","stream":false,"max_output_tokens":0}"#,
        ] {
            let response = roundtrip(
                MockBackend::new(&["unused"], StopReason::Eos),
                &post("/v1/responses", body),
            );
            assert!(response.starts_with("HTTP/1.1 400"), "{response}");
            assert!(!response.contains("response.created"));
        }
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

    #[test]
    fn partial_lines_are_parsed_and_limits_are_enforced() {
        struct Chunks(std::io::Cursor<Vec<u8>>);
        impl Read for Chunks {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                let length = buf.len().min(2);
                self.0.read(&mut buf[..length])
            }
        }

        let bytes = b"GET /v1/models HTTP/1.1\r\nhost: x\r\n\r\n".to_vec();
        let mut reader = BufReader::with_capacity(3, Chunks(std::io::Cursor::new(bytes)));
        assert_eq!(
            read_http_request(&mut reader).unwrap().unwrap().path,
            "/v1/models"
        );

        let mut line = vec![b'x'; MAX_REQUEST_LINE_BYTES + 1];
        line.push(b'\n');
        let mut reader = BufReader::with_capacity(3, Chunks(std::io::Cursor::new(line)));
        assert!(read_http_request(&mut reader).is_err());

        let mut headers = b"GET / HTTP/1.1\r\nx: ".to_vec();
        headers.extend(std::iter::repeat_n(b'x', MAX_HEADER_BYTES));
        headers.extend_from_slice(b"\r\n\r\n");
        let mut reader = BufReader::with_capacity(3, Chunks(std::io::Cursor::new(headers)));
        assert!(read_http_request(&mut reader).is_err());
    }

    #[test]
    fn malformed_http_and_duplicate_content_length_are_rejected() {
        for request in [
            "GET /v1/models\r\nhost: x\r\n\r\n",
            "GET /v1/models HTTP/2\r\nhost: x\r\n\r\n",
            "GET  /v1/models HTTP/1.1\r\nhost: x\r\n\r\n",
            "GET /v1/models HTTP/1.1\r\nbroken\r\n\r\n",
            "GET /v1/models HTTP/1.1\nhost: x\n\n",
            "GET /v1/models HTTP/1.1\r\n\r\n",
            "GET /v1/models HTTP/1.1\r\nhost:\r\n\r\n",
            "GET /v1/models HTTP/1.1\r\nhost: local host\r\n\r\n",
            "GET /v1/models HTTP/1.1\r\nhost: localhost,evil\r\n\r\n",
            "GET /v1/models HTTP/1.1\r\nhost: [::1\r\n\r\n",
            "GET /v1/models HTTP/1.1\r\nhost: ::1\r\n\r\n",
            "GET /v1/models HTTP/1.1\r\nhost: localhost:99999\r\n\r\n",
            "GET /v1/models HTTP/1.1\r\nhost: bad..host\r\n\r\n",
            "GET /v1/models HTTP/1.1\r\nhost: -localhost\r\n\r\n",
            "GET /v1/models HTTP/1.1\r\nhost: x\r\nhost: y\r\n\r\n",
            "POST /v1/responses HTTP/1.1\r\nhost: x\r\ncontent-length: 0\r\ncontent-length: 0\r\n\r\n",
        ] {
            let mut reader = BufReader::new(request.as_bytes());
            assert!(
                read_http_request(&mut reader).is_err(),
                "unexpectedly accepted {request:?}"
            );
        }

        for request in [
            "GET /v1/models HTTP/1.1\r\nhost: localhost:8737\r\n\r\n",
            "GET /v1/models HTTP/1.1\r\nhost: 127.0.0.1\r\n\r\n",
            "GET /v1/models HTTP/1.1\r\nhost: [::1]:8737\r\n\r\n",
        ] {
            let mut reader = BufReader::new(request.as_bytes());
            assert!(read_http_request(&mut reader).unwrap().is_some());
        }
    }

    #[test]
    fn header_values_allow_obs_text_but_recognized_values_require_ascii() {
        let mut request = b"GET /v1/models HTTP/1.1\r\nhost: x\r\nx-opaque: ".to_vec();
        request.push(0xff);
        request.extend_from_slice(b"\r\n\r\n");
        let mut reader = BufReader::new(request.as_slice());
        assert!(read_http_request(&mut reader).unwrap().is_some());

        let mut request = b"GET /v1/models HTTP/1.1\r\nhost: ".to_vec();
        request.push(0xff);
        request.extend_from_slice(b"\r\n\r\n");
        let mut reader = BufReader::new(request.as_slice());
        assert!(read_http_request(&mut reader).is_err());
    }

    #[test]
    fn socket_timeout_cannot_preempt_absolute_watchdog_mapping() {
        assert!(SOCKET_READ_TIMEOUT > REQUEST_READ_DEADLINE);
        let would_block = io::Error::new(io::ErrorKind::WouldBlock, "socket timeout");
        let timed_out = io::Error::new(io::ErrorKind::TimedOut, "socket timeout");
        assert!(is_request_read_timeout(false, Some(&would_block)));
        assert!(is_request_read_timeout(false, Some(&timed_out)));
        assert!(is_request_read_timeout(true, None));
    }

    #[test]
    fn absolute_read_deadline_stops_a_trickling_request() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let started = std::time::Instant::now();
            let error = read_http_request_with_deadline(&stream, Duration::from_millis(40))
                .expect_err("partial request must time out");
            assert_eq!(error.kind(), io::ErrorKind::TimedOut);
            assert!(started.elapsed() < Duration::from_secs(1));
        });
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(b"G").unwrap();
        server.join().unwrap();
    }

    #[test]
    fn non_stream_request_half_close_still_receives_response() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut backend = MockBackend::new(&["answer"], StopReason::Eos);
            handle_connection(&stream, &mut backend, None).unwrap();
        });
        let mut client = TcpStream::connect(address).unwrap();
        client
            .write_all(post("/v1/responses", r#"{"model":"qwen-test","input":"hi"}"#).as_bytes())
            .unwrap();
        client.shutdown(Shutdown::Write).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        server.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 200"));
        assert!(body_of(&response).contains("answer"));
    }

    #[test]
    fn busy_response_has_retry_after() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            configure_stream(&stream).unwrap();
            write_busy_response(&stream).unwrap();
        });
        let mut client = TcpStream::connect(address).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        server.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 503 Service Unavailable"));
        assert!(response.contains("\r\nretry-after: 1\r\n"));
        let body: Value = serde_json::from_str(body_of(&response)).unwrap();
        assert_eq!(body["error"]["code"], "server_busy");
        assert_eq!(body["error"]["param"], "");
    }

    #[test]
    fn full_trace_queue_disables_without_future_value_construction() {
        let (sender, receiver) = sync_channel(1);
        sender.send(json!({"kind": "queued"})).unwrap();
        let mut trace = TraceLog {
            sender: Some(sender),
            worker: None,
        };
        trace.line(|| json!({"kind": "full"}));
        assert!(!trace.is_enabled());
        let constructed = std::cell::Cell::new(false);
        trace.line(|| {
            constructed.set(true);
            json!({"kind": "ignored"})
        });
        assert!(!constructed.get());
        drop(receiver);
    }

    #[test]
    fn stalled_trace_worker_is_detached_instead_of_blocking_shutdown() {
        let (release_sender, release_receiver) = std::sync::mpsc::channel();
        let (done_sender, done_receiver) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            release_receiver.recv().unwrap();
            done_sender.send(()).unwrap();
        });
        assert!(join_trace_worker_with_grace(worker, Duration::ZERO).is_none());
        release_sender.send(()).unwrap();
        done_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("detached worker remains able to finish");
    }

    #[test]
    fn trace_files_are_private_and_symlinks_are_rejected() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let path = std::env::temp_dir().join(format!("qwen-trace-private-{}", next_response_id()));
        let link = path.with_extension("link");
        drop(TraceLog::open(&path).unwrap());
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        drop(TraceLog::open(&path).expect("owned trace permissions are tightened"));
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        symlink(&path, &link).unwrap();
        assert!(TraceLog::open(&link).is_err());
        std::fs::remove_file(link).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn trace_fifo_is_rejected_without_waiting_for_a_reader() {
        use std::os::unix::ffi::OsStrExt;

        let path = std::env::temp_dir().join(format!("qwen-trace-fifo-{}", next_response_id()));
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
        let started = std::time::Instant::now();
        assert!(TraceLog::open(&path).is_err());
        assert!(started.elapsed() < Duration::from_secs(1));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn disconnect_probe_preserves_flags_and_treats_fin_as_inconclusive() {
        use std::os::fd::AsRawFd;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let before = unsafe { libc::fcntl(server.as_raw_fd(), libc::F_GETFL) };
        probe_peer(&server).unwrap();
        let after = unsafe { libc::fcntl(server.as_raw_fd(), libc::F_GETFL) };
        assert_eq!(before, after);
        assert_eq!(after & libc::O_NONBLOCK, 0);
        client.shutdown(Shutdown::Write).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        probe_peer(&server).expect("request-side FIN is not proof the reader disconnected");
    }
}
