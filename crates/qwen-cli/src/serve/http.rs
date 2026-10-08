//! Serial HTTP/1.1 transport and request handler.
//!
//! Scope per docs/SERVE.md: loopback threat model, `Content-Length` JSON
//! request bodies only, `Connection: close` per response, one request in
//! flight (the accept loop lives with the subcommand wiring). Everything
//! model-shaped hides behind [`GenerationBackend`] so this layer tests
//! against a mock over real loopback sockets.

use super::events::{EventWrite, ResponseStream, ServeStats, SseWriter, StopReason, Usage};
#[cfg(test)]
use super::items::parse_request;
use super::items::{ServeError, ServeRequest, TemplateStyle};
#[cfg(test)]
use super::output_partition::ToolGrammar;
use super::output_partition::{GenerationEnd, OutputPartition, OutputProtocol};
#[cfg(test)]
use super::render::render_qwen_serve_prompt;
pub(crate) use super::trace::TraceLog;
use super::trace::TraceSubscriber;
use serde_json::{Value, json};
use std::io::{self, BufRead, BufReader, Write};
use std::net::{Shutdown, TcpStream};
#[cfg(test)]
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::sync_channel;
#[cfg(test)]
use std::time::Instant;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const MAX_REQUEST_LINE_BYTES: usize = 8 * 1024;
const MAX_HEADER_BYTES: usize = 16 * 1024;
pub(super) const READ_WATCHDOG_STACK_BYTES: usize = 2 * 1024 * 1024;
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
const SOCKET_READ_TIMEOUT: Duration = Duration::from_secs(35);
const SOCKET_WRITE_TIMEOUT: Duration = Duration::from_secs(30);
const REQUEST_READ_DEADLINE: Duration = Duration::from_secs(30);

/// Streaming sink handed to the backend. `piece` delivers generated text;
/// `tick` is called between prefill chunks so the transport can heartbeat
/// and detect disconnects (cancellation = the returned error).
pub(crate) trait GenerationSink {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()>;
    /// Transport fragments may change streaming delta boundaries. A collector
    /// can use the first fragment's original length to keep its allocation shape.
    fn piece_fragment(&mut self, bytes: &[u8], _new_piece_bytes: Option<usize>) -> io::Result<()> {
        self.piece(bytes)
    }
    fn tick(&mut self) -> io::Result<()>;
    /// Additional future CPU buffer allowance, beyond resident request storage.
    fn transport_reserve_bytes(&self) -> u64 {
        0
    }
}

/// Process headroom read before each admitted output growth step.
pub(crate) type Headroom = fn() -> Option<u64>;

pub(crate) struct PreparedResponse {
    pub(crate) request: ServeRequest,
    pub(crate) prompt: String,
    /// Boundaries the family renderer recorded in `prompt`, if any.
    pub(crate) boundaries: Option<PromptBoundaries>,
}

/// Byte offsets into a rendered prompt that its renderer recorded while
/// appending (never found by scanning the text). A backend maps them to
/// token positions by tokenizing the prefix and verifying it against the
/// prompt's tokens; families without boundaries render text only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PromptBoundaries {
    /// End of the prefix shared by requests with the same instructions and
    /// tools (before the first conversation turn).
    pub(crate) shared_prefix_end: Option<usize>,
    /// Start of the generation header, after the whole transcript.
    pub(crate) generation_header_start: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct GenerationOutcome {
    pub(crate) end: GenerationEnd,
    pub(crate) usage: Usage,
    pub(crate) stats: Option<ServeStats>,
}

pub(crate) trait GenerationBackend {
    fn model_id(&self) -> &str;
    fn set_control_memory_reserve(&mut self, _bytes: u64) {}
    fn native_profile(&self) -> anyhow::Result<Option<Arc<super::native::Profile>>> {
        Ok(None)
    }
    fn generate_native(
        &mut self,
        prepared: &super::native::Prepared,
        _sink: &super::native::Sink,
    ) -> super::native::Outcome {
        super::native::Outcome::failed(
            prepared.counters(),
            "unsupported_capability",
            "The resident backend has no native diagnostic executor.",
        )
    }
    fn request_profile(&self) -> super::request_profile::RequestProfile {
        super::request_profile::RequestProfile::UnboundQwen
    }
    /// Preserve family-owned JSON values before generic Value deserialization.
    fn decode_request_json(&self, body: &[u8]) -> Result<Value, ServeError> {
        self.request_profile().decode(body)
    }
    /// Family-specific wire admission before transcript normalization loses origin.
    fn parse_request(&self, body: &Value) -> Result<ServeRequest, ServeError> {
        self.request_profile().parse(body)
    }
    /// Resolve family defaults before prompt rendering and response echoes.
    fn normalize_request(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
        self.request_profile().normalize(request)
    }
    /// The deployment's template style (`--template-style`), or `None` for
    /// families that define no house departures from their release format
    /// (the per-request override is then refused rather than ignored).
    fn template_style_default(&self) -> Option<TemplateStyle> {
        self.request_profile().template_style_default()
    }
    /// Family-owned grammar for exact generated token bytes.
    fn output_protocol(&self, request: &ServeRequest) -> OutputProtocol {
        self.request_profile().output(request)
    }
    /// Family-specific prompt rendering. Defaults to the Qwen ChatML path.
    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        self.request_profile().render(request)
    }
    /// [`Self::render_prompt`] plus the boundaries the renderer recorded;
    /// families without boundaries keep the text-only default.
    fn render_prepared(
        &self,
        request: &ServeRequest,
    ) -> Result<(String, Option<PromptBoundaries>), ServeError> {
        self.render_prompt(request).map(|prompt| (prompt, None))
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
    fn generate_prepared(
        &mut self,
        prepared: Arc<PreparedResponse>,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        self.generate(&prepared.request, &prepared.prompt, sink)
    }
    /// Called from the serial loop while no request is admitted (snapshot
    /// cache expiry). Must be cheap.
    fn idle(&mut self) {}
    /// Called once when the serial loop stops (including on a termination
    /// signal), before the backend is torn down. May block for a bounded
    /// time (durable-snapshot flush).
    fn shutdown(&mut self) {}
    /// Called by the serial loop after every admitted connection, whatever
    /// its outcome (idle-publication debounce).
    fn request_finished(&mut self) {}
    /// Called by the serial loop when a generation ended in a server-side
    /// failure (status 5xx: GPU or command-buffer faults, memory admission,
    /// internal validation), before its connection finishes.
    fn request_failed_on_server(&mut self) {}
    /// Headroom the HTTP side admits output buffers against (tests inject).
    fn output_headroom(&self) -> Headroom {
        qwen_llm::metal::MetalContext::process_limit_bytes_remaining
    }
}

struct TraceSseWriter<'a, 'b> {
    inner: SseWriter<&'a TcpStream>,
    trace: Option<&'b TraceSubscriber>,
}

impl<'a, 'b> TraceSseWriter<'a, 'b> {
    fn new(stream: &'a TcpStream, trace: Option<&'b TraceSubscriber>) -> Self {
        Self {
            inner: SseWriter(stream),
            trace,
        }
    }

    fn heartbeat(&mut self) -> io::Result<()> {
        self.inner.heartbeat()?;
        if let Some(trace) = self.trace {
            trace.line(|| json!({"kind": "heartbeat"}));
        }
        Ok(())
    }

    fn done(&mut self) -> io::Result<()> {
        self.inner.done()?;
        if let Some(trace) = self.trace {
            trace.line(|| json!({"kind": "done"}));
        }
        Ok(())
    }
}

impl EventWrite for TraceSseWriter<'_, '_> {
    fn event(&mut self, event_type: &str, payload: Value) -> io::Result<()> {
        let trace_enabled = self.trace.is_some_and(TraceSubscriber::is_enabled);
        if trace_enabled {
            self.inner.event(event_type, payload.clone())?;
        } else {
            return self.inner.event(event_type, payload);
        }
        if let Some(trace) = self.trace {
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
    pub(crate) host: Option<String>,
    pub(crate) origin: Option<String>,
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
    read_http_request_admitted(reader, |_, _| Ok(()))
}

fn read_http_request_admitted<R: BufRead>(
    reader: &mut R,
    mut admit: impl FnMut(&HttpRequest, usize) -> io::Result<()>,
) -> io::Result<Option<HttpRequest>> {
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
    let mut host = None;
    let mut origin = None;
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
            host = Some(String::from_utf8_lossy(value).into_owned());
        } else if name.eq_ignore_ascii_case(b"origin") {
            if origin.is_some() {
                return Err(transport_error("duplicate origin header"));
            }
            origin = Some(String::from_utf8_lossy(value).into_owned());
        }
    }
    if version == "HTTP/1.1" && host_count != 1 {
        return Err(transport_error("HTTP/1.1 requires exactly one host header"));
    }

    let length = if method == "POST" {
        content_length.ok_or_else(|| transport_error("POST requires content-length"))?
    } else {
        content_length.unwrap_or(0)
    };
    let mut request = HttpRequest {
        method,
        path,
        host,
        origin,
        body: Vec::new(),
    };
    admit(&request, length)?;
    if request.method == "POST" {
        if length > MAX_BODY_BYTES {
            return Err(transport_error("request body exceeds limit"));
        }
        request.body.resize(length, 0);
        reader.read_exact(&mut request.body)?;
    }
    Ok(Some(request))
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
    // BSD accept can inherit the listener's nonblocking flag. Request reads
    // must wait for arriving bytes under the existing deadline, not return EAGAIN.
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(SOCKET_READ_TIMEOUT))?;
    stream.set_write_timeout(Some(SOCKET_WRITE_TIMEOUT))
}

fn read_http_request_with_deadline(
    stream: &TcpStream,
    deadline: Duration,
) -> io::Result<Option<HttpRequest>> {
    read_http_request_with_admission(stream, deadline, |_, _| Ok(()))
}

pub(super) fn read_http_request_with_admission(
    stream: &TcpStream,
    deadline: Duration,
    admit: impl FnMut(&HttpRequest, usize) -> io::Result<()>,
) -> io::Result<Option<HttpRequest>> {
    let watchdog_stream = stream.try_clone()?;
    let (cancel_sender, cancel_receiver) = sync_channel::<()>(0);
    let timed_out = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watchdog_timed_out = std::sync::Arc::clone(&timed_out);
    let watchdog = std::thread::Builder::new()
        .stack_size(READ_WATCHDOG_STACK_BYTES)
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
    let result = read_http_request_admitted(&mut BufReader::new(stream), admit);
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

/// A reasoning request whose history arrived without reasoning items says
/// so (never silently): that reasoning cannot be restored, and each family
/// renders such a turn in its template's form for absent reasoning. Absent
/// reasoning in a no-thinking generation changes nothing and is not logged;
/// families whose absent reasoning is provenance rather than loss (DS4 house
/// style: a chat turn) clear the count when normalizing.
fn history_reasoning_diagnostic(
    request: &ServeRequest,
    protocol: &OutputProtocol,
) -> Option<String> {
    (request.history_reasoning_missing > 0 && protocol.reasons()).then(|| {
        format!(
            "serve: history_reasoning_missing={} (assistant turns replayed without a reasoning item; their reasoning cannot be restored)",
            request.history_reasoning_missing
        )
    })
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

/// Non-streaming output, collected until generation ends: every original
/// piece's bytes in one contiguous buffer plus each piece's end offset
/// (map #14: was one allocation per piece). Both buffers admit each growth
/// step against fresh process headroom before allocating
/// ([`super::output_memory`]); a refusal stops generation and is answered
/// with its typed error. A piece arriving in transport fragments reserves
/// its whole length first, so its fragments never reallocate.
struct CollectSink<'a> {
    bytes: Vec<u8>,
    ends: Vec<usize>,
    fragment_remaining: usize,
    stream: &'a TcpStream,
    headroom: Headroom,
}

impl<'a> CollectSink<'a> {
    fn new(stream: &'a TcpStream, headroom: Headroom) -> Self {
        Self {
            bytes: Vec::new(),
            ends: Vec::new(),
            fragment_remaining: 0,
            stream,
            headroom,
        }
    }

    /// Admit, then reserve, room for `additional` bytes and one piece end.
    /// Overflowing sizes are typed 500 `memory_size_overflow`; an admitted
    /// allocation that fails is a typed server error.
    fn reserve_piece(&mut self, additional: usize) -> io::Result<()> {
        use super::output_memory::{
            OUTPUT_STEP_BYTES, admit_growth, allocation_failure, grown_capacity, refusal_error,
            size_overflow,
        };
        const WHAT: &str = "non-streaming output collection";
        let overflow = || refusal_error(size_overflow(WHAT));
        let word = size_of::<usize>();
        let (byte_cap, end_cap) = (self.bytes.capacity(), self.ends.capacity());
        let needed = self
            .bytes
            .len()
            .checked_add(additional)
            .ok_or_else(overflow)?;
        let new_bytes = if needed > byte_cap {
            grown_capacity(byte_cap, needed, OUTPUT_STEP_BYTES).ok_or_else(overflow)?
        } else {
            byte_cap
        };
        let new_ends = if self.ends.len() == end_cap {
            let one_more = end_cap.checked_add(1).ok_or_else(overflow)?;
            grown_capacity(end_cap, one_more, OUTPUT_STEP_BYTES / word).ok_or_else(overflow)?
        } else {
            end_cap
        };
        if (new_bytes, new_ends) == (byte_cap, end_cap) {
            return Ok(());
        }
        // Byte sizes, checked; a vector cannot exceed isize::MAX bytes.
        let bytes_of = |elements: usize, size: usize| {
            elements
                .checked_mul(size)
                .filter(|&bytes| bytes <= isize::MAX as usize)
                .map(|bytes| bytes as u64)
                .ok_or_else(overflow)
        };
        let (old_b, new_b) = (bytes_of(byte_cap, 1)?, bytes_of(new_bytes, 1)?);
        let (old_e, new_e) = (bytes_of(end_cap, word)?, bytes_of(new_ends, word)?);
        // While a buffer grows, its old allocation is live during the copy.
        let grown = |new: u64, old: u64| {
            if new > old {
                new.checked_add(old).ok_or_else(overflow)
            } else {
                Ok(old)
            }
        };
        let held = old_b.checked_add(old_e).ok_or_else(overflow)?;
        let peak = grown(new_b, old_b)?
            .checked_add(grown(new_e, old_e)?)
            .ok_or_else(overflow)?;
        admit_growth(peak, held, (self.headroom)()).map_err(refusal_error)?;
        self.bytes
            .try_reserve_exact(new_bytes - self.bytes.len())
            .map_err(|e| refusal_error(allocation_failure(WHAT, e)))?;
        self.ends
            .try_reserve_exact(new_ends - self.ends.len())
            .map_err(|e| refusal_error(allocation_failure(WHAT, e)))?;
        Ok(())
    }

    /// The original pieces, in order (every piece must be complete).
    fn pieces(&self) -> impl Iterator<Item = &[u8]> {
        let starts = std::iter::once(0).chain(self.ends.iter().copied());
        starts
            .zip(self.ends.iter().copied())
            .map(|(start, end)| &self.bytes[start..end])
    }
}

impl GenerationSink for CollectSink<'_> {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        shutdown_checkpoint()?;
        probe_peer(self.stream)?;
        if self.fragment_remaining != 0 {
            return Err(transport_error("previous output piece is incomplete"));
        }
        self.reserve_piece(bytes.len())?;
        self.bytes.extend_from_slice(bytes);
        self.ends.push(self.bytes.len());
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        shutdown_checkpoint()?;
        probe_peer(self.stream)
    }

    fn piece_fragment(&mut self, bytes: &[u8], new_piece_bytes: Option<usize>) -> io::Result<()> {
        self.tick()?;
        if let Some(length) = new_piece_bytes {
            if self.fragment_remaining != 0 {
                return Err(transport_error("previous output piece is incomplete"));
            }
            self.reserve_piece(length)?;
            self.ends.push(self.bytes.len() + length);
            self.fragment_remaining = length;
        }
        if self.ends.is_empty() {
            return Err(transport_error("output fragment has no original piece"));
        }
        let remaining = self
            .fragment_remaining
            .checked_sub(bytes.len())
            .ok_or_else(|| transport_error("output fragment exceeds original piece"))?;
        self.bytes.extend_from_slice(bytes);
        self.fragment_remaining = remaining;
        Ok(())
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
        // A stored refusal already decides the turn: stop generating.
        if let Some(failure) = self.partition.failure() {
            return Err(super::output_memory::refusal_error(failure.clone()));
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
pub(crate) fn handle_connection(
    stream: &TcpStream,
    backend: &mut dyn GenerationBackend,
    trace: Option<TraceSubscriber>,
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

    handle_request(stream, backend, trace, request)
}

pub(super) fn handle_request(
    stream: &TcpStream,
    backend: &mut dyn GenerationBackend,
    trace: Option<TraceSubscriber>,
    request: HttpRequest,
) -> io::Result<()> {
    let mut writer = stream;
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/v1/models") => write_json_response(
            &mut writer,
            200,
            &json!({
                "object": "list",
                "data": [{"id": backend.model_id(), "object": "model", "owned_by": "local"}],
            }),
        ),
        ("POST", "/v1/responses") => {
            handle_responses(&request.body, stream, backend, trace.as_ref())
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
    trace: Option<&TraceSubscriber>,
) -> io::Result<()> {
    let mut writer = stream;
    let parsed = match backend.decode_request_json(body) {
        Ok(parsed) => {
            if let Some(trace) = trace
                && trace.is_enabled()
            {
                trace.line(|| json!({"kind": "request", "body": parsed.clone()}));
            }
            parsed
        }
        Err(error) => {
            if let Some(trace) = trace
                && trace.is_enabled()
            {
                trace.line(|| {
                    json!({
                        "kind": "request",
                        "body": String::from_utf8_lossy(body),
                    })
                });
            }
            return write_serve_error(&mut writer, &error);
        }
    };
    let mut request = match backend.parse_request(&parsed) {
        Ok(request) => request,
        Err(error) => return write_serve_error(&mut writer, &error),
    };
    if request.model != backend.model_id() {
        return write_serve_error(
            &mut writer,
            &ServeError::model_not_found(&request.model, backend.model_id()),
        );
    }
    match (backend.template_style_default(), request.template_style) {
        (Some(default), None) => request.template_style = Some(default),
        (None, Some(_)) => {
            return write_serve_error(
                &mut writer,
                &ServeError::invalid_request(
                    Some("x_qwen.template_style"),
                    "x_qwen.template_style is defined for identified Qwen releases and DeepSeek V4 only",
                ),
            );
        }
        _ => {}
    }
    if let Err(error) = backend.normalize_request(&mut request) {
        return write_serve_error(&mut writer, &error);
    }
    let (prompt, boundaries) = match backend.render_prepared(&request) {
        Ok(rendered) => rendered,
        Err(error) => return write_serve_error(&mut writer, &error),
    };
    let response_id = next_response_id();
    let created_at = now_unix();

    // Resolved before the mutable generate borrow.
    let output_protocol = backend.output_protocol(&request);
    // Said per request, before generation can fail.
    if let Some(line) = history_reasoning_diagnostic(&request, &output_protocol) {
        eprintln!("{line}");
    }
    let prepared = Arc::new(PreparedResponse {
        request,
        prompt,
        boundaries,
    });
    let request = &prepared.request;
    if !request.stream {
        let mut sink = CollectSink::new(stream, backend.output_headroom());
        match backend.generate_prepared(Arc::clone(&prepared), &mut sink) {
            Ok(outcome) => {
                if sink.fragment_remaining != 0 {
                    let error = ServeError::server_error("an output piece ended incomplete");
                    note_server_failure(backend, &error);
                    return write_serve_error(&mut writer, &error);
                }
                // Each piece's events go straight into the response object
                // (no second retained copy of the output as events).
                let mut discard = super::events::DiscardEvents;
                let mut response = ResponseStream::begin(
                    &mut discard,
                    response_id,
                    request.model.clone(),
                    created_at,
                    super::events::envelope_echo(request),
                )?;
                response.set_allowed_tools(request.allowed_tools.clone());
                let mut partition =
                    OutputPartition::with_headroom(output_protocol, backend.output_headroom());
                let mut events = Vec::new();
                for piece in sink.pieces() {
                    partition.push(piece, &mut events);
                    for event in events.drain(..) {
                        response.on_partition(&event)?;
                    }
                }
                drop(sink);
                if let Err(error) = partition.finish(outcome.end, &mut events) {
                    note_server_failure(backend, &error);
                    return write_serve_error(&mut writer, &error);
                }
                for event in events.drain(..) {
                    response.on_partition(&event)?;
                }
                let envelope = response.finish(
                    response_stop_reason(outcome.end),
                    outcome.usage,
                    ServeStats::echo_for(outcome.stats.as_ref(), request).as_ref(),
                )?;
                write_json_response(&mut writer, 200, &envelope)
            }
            Err(BackendFailure::Serve(error)) => write_serve_error(&mut writer, &error),
            Err(BackendFailure::Aborted(error)) => match super::output_memory::refusal_in(&error) {
                // Refused on this side: the backend saw only a stopped
                // sink, so the server failure is reported here.
                Some(refusal) => {
                    note_server_failure(backend, refusal);
                    write_serve_error(&mut writer, refusal)
                }
                None => Err(error),
            },
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
            partition: OutputPartition::with_headroom(output_protocol, backend.output_headroom()),
        };
        let outcome = backend.generate_prepared(Arc::clone(&prepared), &mut sink);
        let StreamingSink { partition, .. } = sink;
        match outcome {
            Ok(outcome) => {
                let mut events = Vec::new();
                if let Err(error) = partition.finish(outcome.end, &mut events) {
                    note_server_failure(backend, &error);
                    response.fail(&error)?;
                    return sse.done();
                }
                for event in &events {
                    response.on_partition(event)?;
                }
                response.finish(
                    response_stop_reason(outcome.end),
                    outcome.usage,
                    ServeStats::echo_for(outcome.stats.as_ref(), &request).as_ref(),
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
            Err(BackendFailure::Aborted(error)) => match super::output_memory::refusal_in(&error) {
                // Refused on this side after the headers: the typed failure
                // event, reported here (the backend saw a stopped sink).
                // No raw-text fallback: the held output is dropped.
                Some(refusal) => {
                    let refusal = refusal.clone();
                    note_server_failure(backend, &refusal);
                    drop(partition);
                    response.fail(&refusal)?;
                    sse.done()
                }
                None => Err(error),
            },
        }
    }
}

/// A failure after the backend succeeded (the output partition) is still a
/// server-side failure: report it so idle residency does not renew.
fn note_server_failure(backend: &mut dyn GenerationBackend, error: &ServeError) {
    if error.status >= 500 {
        backend.request_failed_on_server();
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
        headroom: Option<Headroom>,
        server_failures: Arc<std::sync::atomic::AtomicUsize>,
        /// After its pieces, end as a sink-raised typed refusal would.
        refuse_after: Option<ServeError>,
        /// Pieces offered to the sink (including a refused one).
        attempts: Arc<std::sync::atomic::AtomicUsize>,
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
                    tool_grammar: ToolGrammar::QwenXml,
                },
                fail_with: None,
                headroom: None,
                server_failures: Arc::default(),
                refuse_after: None,
                attempts: Arc::default(),
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
        fn parse_request(&self, body: &Value) -> Result<ServeRequest, ServeError> {
            if self.protocol == OutputProtocol::RawText {
                super::super::render_k2::parse_request(body)
            } else {
                parse_request(body)
            }
        }
        fn normalize_request(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
            if self.protocol == OutputProtocol::RawText {
                super::super::render_k2::normalize(request, 8, 32)
            } else {
                Ok(())
            }
        }
        fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
            if self.protocol == OutputProtocol::RawText {
                super::super::render_k2::render(request)
            } else {
                Ok(render_qwen_serve_prompt(request))
            }
        }
        fn output_protocol(&self, _request: &ServeRequest) -> OutputProtocol {
            self.protocol.clone()
        }
        fn output_headroom(&self) -> Headroom {
            self.headroom
                .unwrap_or(qwen_llm::metal::MetalContext::process_limit_bytes_remaining)
        }
        fn request_failed_on_server(&mut self) {
            self.server_failures
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
            if self.protocol == OutputProtocol::RawText {
                assert_eq!(Some(_prompt), _request.k2_raw_input.as_deref());
            }
            sink.tick().map_err(BackendFailure::Aborted)?;
            for piece in &self.pieces {
                self.attempts
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                sink.piece(piece.as_bytes())
                    .map_err(BackendFailure::Aborted)?;
            }
            if let Some(error) = self.refuse_after.clone() {
                return Err(BackendFailure::Aborted(
                    super::super::output_memory::refusal_error(error),
                ));
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
                    seed: None,
                }),
            })
        }
    }

    #[test]
    fn accepted_nonblocking_socket_waits_for_request_bytes() {
        use std::os::fd::AsRawFd;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let (stream, _) = loop {
            match listener.accept() {
                Ok(connection) => break connection,
                Err(error)
                    if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
                {
                    super::super::wait_for_connection(&listener).unwrap();
                }
                Err(error) => panic!("accept socket-mode test client: {error}"),
            }
        };
        let inherited = unsafe { libc::fcntl(stream.as_raw_fd(), libc::F_GETFL) };
        assert!(inherited >= 0);
        eprintln!(
            "accepted inherited O_NONBLOCK={}",
            inherited & libc::O_NONBLOCK != 0
        );
        stream.set_nonblocking(true).unwrap();
        configure_stream(&stream).unwrap();
        let configured = unsafe { libc::fcntl(stream.as_raw_fd(), libc::F_GETFL) };
        assert!(configured >= 0);
        assert_eq!(configured & libc::O_NONBLOCK, 0);
        let (entered, entry) = std::sync::mpsc::channel();
        let server = std::thread::spawn(move || {
            entered.send(()).unwrap();
            read_http_request_with_deadline(&stream, Duration::from_secs(2))
        });
        entry.recv().unwrap();
        std::thread::sleep(Duration::from_millis(10));
        client
            .write_all(b"GET /v1/models HTTP/1.1\r\nhost: localhost\r\n\r\n")
            .unwrap();
        let request = server.join().unwrap().unwrap().unwrap();
        assert_eq!(request.path, "/v1/models");
    }

    fn roundtrip(mut backend: impl GenerationBackend + Send + 'static, request: &str) -> String {
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

    /// One original piece's transport fragments reassemble without
    /// reallocating (its whole length is reserved first), pieces keep their
    /// boundaries in one contiguous buffer, and malformed fragments refuse.
    #[test]
    fn nonstream_fragments_reassemble_each_original_piece_without_reallocation() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (stream, _) = listener.accept().unwrap();
        let mut sink = CollectSink::new(&stream, || Some(0));
        assert!(sink.piece_fragment(b"x", None).is_err(), "no piece yet");
        sink.piece_fragment(b"abc", Some(7)).unwrap();
        let pointer = sink.bytes.as_ptr();
        sink.piece_fragment(b"defg", None).unwrap();
        assert_eq!(sink.bytes.as_ptr(), pointer);
        assert_eq!(sink.fragment_remaining, 0);
        assert!(
            sink.piece(b"!").is_ok(),
            "a whole piece after a completed one"
        );
        sink.piece_fragment(b"next", Some(5)).unwrap();
        assert!(
            sink.piece(b"?").is_err(),
            "a whole piece inside a fragmented one"
        );
        assert!(sink.piece_fragment(b"!!", None).is_err(), "overflow");
        sink.piece_fragment(b"!", None).unwrap();
        let pieces: Vec<&[u8]> = sink.pieces().collect();
        assert_eq!(pieces, [&b"abcdefg"[..], b"!", b"next!"]);
    }

    /// Each growth step admits its whole outstanding peak against fresh
    /// headroom before allocating; a refusal leaves the buffers unchanged
    /// and carries the typed error.
    #[test]
    fn nonstream_collection_admits_each_growth_step_before_allocating() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static HEADROOM: AtomicU64 = AtomicU64::new(u64::MAX);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (stream, _) = listener.accept().unwrap();
        let mut sink = CollectSink::new(&stream, || Some(HEADROOM.load(Ordering::SeqCst)));
        let step = super::super::output_memory::OUTPUT_STEP_BYTES;
        // The first step: 64 KiB of bytes plus 64 KiB of piece ends.
        HEADROOM.store(2 * step as u64 - 1, Ordering::SeqCst);
        let refused = sink.piece(b"a").unwrap_err();
        let error = super::super::output_memory::refusal_in(&refused).unwrap();
        assert_eq!(
            (error.status, error.code),
            (503, Some("memory_admission_denied"))
        );
        assert_eq!((sink.bytes.capacity(), sink.ends.capacity()), (0, 0));
        HEADROOM.store(2 * step as u64, Ordering::SeqCst);
        sink.piece(b"a").unwrap();
        assert_eq!(sink.bytes.capacity(), step);
        // Within capacity no admission is needed, whatever the headroom.
        HEADROOM.store(1, Ordering::SeqCst);
        sink.piece(&vec![b'b'; step - 1]).unwrap();
        // The next step doubles the bytes: peak = old + new, held = old, so
        // the outstanding is the new allocation alone.
        let piece = *b"c";
        HEADROOM.store(2 * step as u64 - 1, Ordering::SeqCst);
        assert!(sink.piece(&piece).is_err());
        assert_eq!(sink.bytes.len(), step);
        HEADROOM.store(2 * step as u64, Ordering::SeqCst);
        sink.piece(&piece).unwrap();
        assert_eq!(sink.bytes.capacity(), 2 * step);
        // An unreadable signal fails closed as telemetry (500).
        let mut fresh = CollectSink::new(&stream, || None);
        let error = fresh.piece(b"x").unwrap_err();
        assert_eq!(
            super::super::output_memory::refusal_in(&error).map(|e| e.status),
            Some(500)
        );
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
    fn k2_raw_nonstream_and_sse_never_parse_reasoning_or_tools() {
        let pieces = [
            "<thi",
            "nk>literal</think>",
            "<tool_call>{\"name\":\"x\"}</tool_call>",
            "<|ifm|end_of_text|>",
        ];
        for streaming in [false, true] {
            let mut backend = MockBackend::new(&pieces, StopReason::Eos);
            backend.protocol = OutputProtocol::RawText;
            backend.end = GenerationEnd::StopToken(1);
            let body = json!({"model":"qwen-test","input":"exact raw input","stream":streaming,"x_k2":{"add_special_tokens":false}});
            let response = roundtrip(backend, &post("/v1/responses", &body.to_string()));
            assert!(response.starts_with("HTTP/1.1 200"));
            let envelope = if streaming {
                sse_payload(&response, "response.completed")["response"].clone()
            } else {
                serde_json::from_str::<Value>(body_of(&response)).unwrap()
            };
            let output = envelope["output"].as_array().unwrap();
            assert_eq!(output.len(), 1);
            assert_eq!(output[0]["type"], "message");
            assert_eq!(output[0]["content"][0]["text"], pieces.concat());
            assert!(envelope["reasoning"].is_null());
            assert_eq!(envelope["tools"], json!([]));
            assert_eq!(envelope["parallel_tool_calls"], false);
            assert_eq!(envelope["tool_choice"], "none");
            assert!(envelope.get("x_qwen").is_none());
        }
    }

    #[test]
    fn k2_family_parser_refuses_chat_before_backend_generation() {
        let mut backend = MockBackend::new(&[], StopReason::Eos);
        backend.protocol = OutputProtocol::RawText;
        backend.fail_with = Some(ServeError::server_error("must not generate"));
        let body = json!({"model":"qwen-test","input":[{"role":"user","content":"chat"}]});
        let response = roundtrip(backend, &post("/v1/responses", &body.to_string()));
        assert!(response.starts_with("HTTP/1.1 400"));
        assert!(body_of(&response).contains("raw input string"));
        assert!(!body_of(&response).contains("must not generate"));
    }

    #[path = "k2_chat_tests.rs"]
    mod k2_chat;
    #[path = "request_profile_tests.rs"]
    mod request_profiles;
    #[path = "tool_block_memory_tests.rs"]
    mod tool_block_memory;

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
            let log = TraceLog::open(&path).expect("open trace");
            let trace = log.subscriber();
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
            let log = TraceLog::open(&path).expect("reopen trace");
            let trace = log.subscriber();
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
        for field in ["\"created_at\":", "\"completed_at\":"] {
            let mut normalized = String::with_capacity(out.len());
            let mut rest = out.as_str();
            while let Some(index) = rest.find(field) {
                normalized.push_str(&rest[..index + field.len()]);
                rest = &rest[index + field.len()..];
                let skip = rest
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(rest.len());
                if skip > 0 {
                    normalized.push('0');
                }
                rest = &rest[skip..];
            }
            normalized.push_str(rest);
            out = normalized;
        }
        out
    }

    fn roundtrip_traced(mut backend: MockBackend, request: &str, trace_path: &Path) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let trace_path = trace_path.to_path_buf();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let trace = TraceLog::open(&trace_path).expect("open trace");
            handle_connection(&stream, &mut backend, Some(trace.subscriber())).unwrap();
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

    /// A family that defines no template styles refuses the per-request
    /// override rather than ignoring it.
    #[test]
    fn template_style_is_refused_where_undefined() {
        let response = roundtrip(
            MockBackend::new(&["answer"], StopReason::Eos),
            &post(
                "/v1/responses",
                r#"{"model":"qwen-test","input":"hi","x_qwen":{"template_style":"upstream"}}"#,
            ),
        );
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");
        assert!(body_of(&response).contains("x_qwen.template_style"));
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
        // The stats echo carries the request's effective seed for replay.
        let seeded = roundtrip(
            MockBackend::new(&["<think>\np\n</think>\n\nanswer"], StopReason::Eos),
            &post(
                "/v1/responses",
                r#"{"model":"qwen-test","input":"hi","x_qwen":{"stats":true,"seed":7}}"#,
            ),
        );
        let parsed: Value = serde_json::from_str(body_of(&seeded)).unwrap();
        assert_eq!(parsed["x_qwen"]["seed"], 7);
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
            let trace = TraceLog::open(&trace_path).unwrap();
            handle_connection(&stream, &mut backend, Some(trace.subscriber())).unwrap();
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
            lines[0]["trace_request_id"]
                .as_str()
                .unwrap()
                .starts_with("trace_")
        );
        assert!(
            lines
                .iter()
                .all(|line| line["trace_request_id"] == lines[0]["trace_request_id"])
        );
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
    fn trace_correlates_each_request_even_when_json_decoding_fails() {
        let path =
            std::env::temp_dir().join(format!("qwen-trace-invalid-{}.jsonl", next_response_id()));
        let trace_path = path.clone();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let trace = TraceLog::open(&trace_path).unwrap();
            let mut backend = MockBackend::new(&["answer"], StopReason::Eos);
            for _ in 0..2 {
                let (stream, _) = listener.accept().unwrap();
                handle_connection(&stream, &mut backend, Some(trace.subscriber())).unwrap();
            }
        });
        for (body, status) in [
            (r#"{"model":"qwen-test","input":"hi"}"#, "200"),
            ("not JSON", "400"),
        ] {
            let mut client = TcpStream::connect(address).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            client
                .write_all(post("/v1/responses", body).as_bytes())
                .unwrap();
            let mut response = String::new();
            client.read_to_string(&mut response).unwrap();
            assert!(
                response.starts_with(&format!("HTTP/1.1 {status}")),
                "{response}"
            );
        }
        server.join().unwrap();
        let rows = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["body"]["input"], "hi");
        assert_eq!(rows[1]["body"], "not JSON");
        assert_ne!(rows[0]["trace_request_id"], rows[1]["trace_request_id"]);
        for row in rows {
            assert_eq!(row["kind"], "request");
            assert!(
                row["trace_request_id"]
                    .as_str()
                    .unwrap()
                    .starts_with("trace_")
            );
        }
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

    /// A typed memory refusal from a session (lane audit B3): pressure is a
    /// 503 `memory_admission_denied` envelope; missing telemetry is a 500 with
    /// its own code. After stream headers, both are `response.failed` with
    /// the same code (the HTTP status is already 200).
    #[test]
    fn typed_memory_refusals_reach_json_and_stream_with_their_codes() {
        use qwen_llm::metal::{
            MemoryAdmissionDenied, MetalMemoryAdmissionReason as R, MetalMemorySignals,
        };
        let refusal = |reason| {
            super::super::transport_memory::memory_refusal(
                "GLM-5.3-Flash session",
                &MemoryAdmissionDenied {
                    reason,
                    required_bytes: Some(1 << 33),
                    signals: MetalMemorySignals {
                        recommended_max_bytes: 1 << 34,
                        current_allocated_bytes: 1 << 34,
                        process_limit_remaining_bytes: Some(1 << 30),
                    },
                    working_set_headroom_bytes: Some(0),
                },
            )
        };
        for (reason, status, code) in [
            (R::WorkingSetInsufficient, "503", "memory_admission_denied"),
            (
                R::ProcessSignalUnavailable,
                "500",
                "memory_signal_unavailable",
            ),
        ] {
            let mut backend = MockBackend::new(&[], StopReason::Eos);
            backend.fail_with = Some(refusal(reason));
            let response = roundtrip(
                backend,
                &post("/v1/responses", r#"{"model":"qwen-test","input":"q"}"#),
            );
            assert!(
                response.starts_with(&format!("HTTP/1.1 {status}")),
                "{response}"
            );
            let envelope: Value = serde_json::from_str(body_of(&response)).unwrap();
            assert_eq!(envelope["error"]["code"], code, "{envelope}");

            let mut backend = MockBackend::new(&[], StopReason::Eos);
            backend.fail_with = Some(refusal(reason));
            let response = roundtrip(
                backend,
                &post(
                    "/v1/responses",
                    r#"{"model":"qwen-test","input":"q","stream":true}"#,
                ),
            );
            assert!(response.starts_with("HTTP/1.1 200"), "{response}");
            let failed = sse_payload(&response, "response.failed");
            assert_eq!(failed["response"]["error"]["code"], code, "{failed}");
            assert!(body_of(&response).ends_with("data: [DONE]\n\n"));
        }
    }

    /// A non-streaming response whose collected output is refused admission
    /// answers with the typed refusal and reports a server failure once; a
    /// streaming response does not collect raw output, and an admitted
    /// non-streaming one is unchanged.
    #[test]
    fn nonstream_output_refusals_answer_typed_and_report_the_failure() {
        let request = post("/v1/responses", r#"{"model":"qwen-test","input":"q"}"#);
        let refused = |headroom: Headroom| {
            let mut backend = MockBackend::new(&["hello ", "world"], StopReason::Eos);
            backend.headroom = Some(headroom);
            let failures = Arc::clone(&backend.server_failures);
            (roundtrip(backend, &request), failures)
        };
        for (headroom, status, code) in [
            ((|| Some(1)) as Headroom, "503", "memory_admission_denied"),
            (|| None, "500", "memory_signal_unavailable"),
        ] {
            let (response, failures) = refused(headroom);
            assert!(
                response.starts_with(&format!("HTTP/1.1 {status}")),
                "{response}"
            );
            let envelope: Value = serde_json::from_str(body_of(&response)).unwrap();
            assert_eq!(envelope["error"]["code"], code, "{envelope}");
            assert_eq!(failures.load(std::sync::atomic::Ordering::SeqCst), 1);
        }
        let (response, failures) = refused(|| Some(u64::MAX));
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let envelope: Value = serde_json::from_str(body_of(&response)).unwrap();
        assert_eq!(envelope["output"][0]["content"][0]["text"], "hello world");
        assert_eq!(failures.load(std::sync::atomic::Ordering::SeqCst), 0);
        let mut backend = MockBackend::new(&["hello ", "world"], StopReason::Eos);
        backend.headroom = Some(|| Some(1));
        let response = roundtrip(
            backend,
            &post(
                "/v1/responses",
                r#"{"model":"qwen-test","input":"q","stream":true}"#,
            ),
        );
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(body_of(&response).contains("response.completed"));

        // A typed refusal after the SSE headers: response.failed with its
        // code, [DONE], no success terminal, one server-failure report.
        let mut backend = MockBackend::new(&["hello "], StopReason::Eos);
        backend.refuse_after = Some(
            super::super::transport_memory::admit_resident_transport(10, Some(1)).unwrap_err(),
        );
        let failures = Arc::clone(&backend.server_failures);
        let response = roundtrip(
            backend,
            &post(
                "/v1/responses",
                r#"{"model":"qwen-test","input":"q","stream":true}"#,
            ),
        );
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let failed = sse_payload(&response, "response.failed");
        assert_eq!(
            failed["response"]["error"]["code"], "memory_admission_denied",
            "{failed}"
        );
        assert!(!body_of(&response).contains("response.completed"));
        assert!(body_of(&response).ends_with("data: [DONE]\n\n"));
        assert_eq!(failures.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    /// Map #14 packet 2 end to end: a Qwen or GLM tool block refused
    /// admission stops a streaming generation at the refusing piece (later
    /// pieces are never requested) and fails it with the typed code,
    /// publishing no held tool text; non-streaming (whose partition runs
    /// after generation) answers the JSON 503. Each reports one server
    /// failure.
    #[test]
    fn tool_block_refusals_stop_streaming_and_fail_typed() {
        use std::sync::atomic::Ordering;
        let glm_tools = || OutputProtocol::Glm5NextTools {
            definitions: vec![
                qwen_llm::glm5_next_chat::ToolDefinition::from_value(&json!({
                    "name": "f", "parameters": {"type": "object",
                        "properties": {"a": {"type": "integer"}}}}))
                .unwrap(),
            ],
            max_bytes: 1 << 20,
        };
        let cases: [(&str, Vec<&str>, Option<OutputProtocol>, usize); 2] = [
            (
                "qwen",
                vec![
                    "<think>plan</think>",
                    "Calling.\n",
                    "<tool_call>\n<function=f>\n",
                    "<parameter=a>\n1\n</parameter>\n",
                    "</function>\n</tool_call>",
                ],
                None,
                3,
            ),
            (
                "glm",
                vec![
                    "plan</think>",
                    "<tool_call>f<arg_key>a</arg_key>",
                    "<arg_value>1</arg_value></tool_call>",
                ],
                Some(glm_tools()),
                2,
            ),
        ];
        for (family, pieces, protocol, refusing_piece) in cases {
            for streaming in [true, false] {
                let mut backend = MockBackend::new(&pieces, StopReason::Eos);
                if let Some(protocol) = protocol.clone() {
                    backend.protocol = protocol;
                }
                // Room for collection steps (128 KiB), not a 64 KiB tool
                // step at 256x.
                backend.headroom = Some(|| Some(1 << 20));
                let failures = Arc::clone(&backend.server_failures);
                let attempts = Arc::clone(&backend.attempts);
                let body = if streaming {
                    r#"{"model":"qwen-test","input":"q","stream":true}"#
                } else {
                    r#"{"model":"qwen-test","input":"q"}"#
                };
                let response = roundtrip(backend, &post("/v1/responses", body));
                if streaming {
                    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
                    let failed = sse_payload(&response, "response.failed");
                    assert_eq!(
                        failed["response"]["error"]["code"], "memory_admission_denied",
                        "{family}: {failed}"
                    );
                    assert!(!body_of(&response).contains("<tool_call>"), "{response}");
                    assert!(!body_of(&response).contains("function_call"), "{response}");
                    assert!(!body_of(&response).contains("response.completed"));
                    assert_eq!(
                        attempts.load(Ordering::SeqCst),
                        refusing_piece,
                        "{family}: generation stops at the refusing piece"
                    );
                } else {
                    assert!(response.starts_with("HTTP/1.1 503"), "{family}: {response}");
                    let envelope: Value = serde_json::from_str(body_of(&response)).unwrap();
                    assert_eq!(envelope["error"]["code"], "memory_admission_denied");
                    assert_eq!(attempts.load(Ordering::SeqCst), pieces.len());
                }
                assert_eq!(failures.load(Ordering::SeqCst), 1, "{family} {streaming}");
            }
        }
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
