//! Native Lens HTTP control plane. These routes never borrow the model owner.
//!
//! Admission is capability-owned; reserving a queue slot precedes durable
//! acceptance, which precedes enqueueing. Delivery failure after acceptance
//! cannot be turned into a pre-admission rejection or a fresh execution.

use super::http::HttpRequest;
pub(crate) mod access;
pub(crate) mod input;
use super::jobs::state::{JobError, JobStatus};
use super::jobs::store::{JobStore, Limits, StoreError};
use crate::ordinary_executor::ExecutionControl;
use serde_json::{Value, json};
use std::io::{self, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

#[cfg(test)]
mod tests;

pub(crate) struct AcceptedJob {
    pub(crate) id: String,
    #[cfg(test)]
    pub(crate) request: Value,
    pub(crate) control: ExecutionControl,
}

pub(crate) trait ReservedSubmission: Send {
    fn archive_bytes(&self) -> u64 {
        0
    }
    /// Err guarantees that the job was not delivered. Dropping an unused
    /// reservation releases its slot; enqueue never waits for GPU completion.
    fn enqueue(self: Box<Self>, job: AcceptedJob) -> Result<(), String>;
}

/// Implemented by a CPU admission adapter for a qualified native executor,
/// never by a second HTTP service or a socket-lifetime generation loop.
pub(crate) trait Admission: Send + Sync {
    fn capabilities(&self) -> Value;
    fn assets(&self) -> Value;
    /// Nonblocking reservation in the existing model owner's bounded queue.
    fn reserve(&self, request: &input::Request) -> Result<Box<dyn ReservedSubmission>, ApiError>;
}

pub(crate) struct LensApi {
    model_id: String,
    store: Option<Arc<JobStore>>,
    admission: Option<Arc<dyn Admission>>,
    submissions: Mutex<()>,
    access: access::BrowserAccess,
}

pub(crate) struct ApiError {
    pub(crate) status: u16,
    pub(crate) error: JobError,
    rejection: Option<String>,
}

impl ApiError {
    pub(crate) fn new(status: u16, code: &str, message: impl Into<String>) -> Self {
        Self {
            status,
            rejection: None,
            error: JobError {
                r#type: match status {
                    429 | 503 => "server_busy",
                    500 => "server_error",
                    _ => "invalid_request_error",
                }
                .into(),
                code: code.into(),
                param: None,
                message: message.into(),
            },
        }
    }
    fn invalid(message: impl Into<String>) -> Self {
        Self::new(400, "invalid_request", message)
    }

    fn not_accepted(mut self, key: &str) -> Self {
        if matches!(self.status, 400 | 412 | 413 | 429 | 503) {
            self.rejection = Some(key.into());
        }
        self
    }

    fn after_acceptance(error: StoreError) -> Self {
        Self::new(
            500,
            "post_acceptance_failure",
            format!("Job was accepted; recover the same key: {error}"),
        )
    }
}

impl From<StoreError> for ApiError {
    fn from(error: StoreError) -> Self {
        match error {
            StoreError::Invalid(message) => Self::invalid(message),
            StoreError::NotFound => Self::new(404, "unknown_job", "Unknown job"),
            StoreError::Conflict => Self::new(409, "idempotency_conflict", error.to_string()),
            StoreError::Full => Self::new(429, "capacity_exceeded", error.to_string()),
            StoreError::Deleted => Self::new(410, "job_deleted", error.to_string()),
            StoreError::RecoveryRequired => {
                Self::new(500, "history_recovery_required", error.to_string())
            }
            // A rename may have succeeded. A 503 would incorrectly tell the
            // browser it is safe to abandon this key and create new work.
            StoreError::Storage(_) => Self::new(500, "storage_unavailable", error.to_string()),
        }
    }
}

struct Reply {
    status: u16,
    location: Option<String>,
    body: Value,
    binary: Option<Vec<u8>>,
}
impl Reply {
    fn ok(body: impl serde::Serialize) -> Result<Self, ApiError> {
        let body = serde_json::to_value(body)
            .map_err(|error| ApiError::new(500, "serialization_failed", error.to_string()))?;
        Ok(Self {
            status: 200,
            location: None,
            body,
            binary: None,
        })
    }
}

impl LensApi {
    pub(super) fn history_enabled(&self) -> bool {
        self.store.is_some()
    }
    pub(super) fn trusted(&self, request: &HttpRequest) -> bool {
        self.access.check(request).is_ok()
    }
    pub(super) fn matches(path: &str) -> bool {
        let path = path.split('?').next().unwrap_or_default();
        path == "/v1/lens" || path.starts_with("/v1/lens/")
    }

    pub(super) fn request_byte_limit(&self) -> usize {
        self.store
            .as_ref()
            .map_or_else(Limits::default, |store| store.limits())
            .max_request_bytes
    }

    pub(super) fn is_read_only(&self, request: &HttpRequest) -> bool {
        if request.method != "GET" || !self.trusted(request) {
            return false;
        }
        let path = request.path.split('?').next().unwrap_or_default();
        if matches!(
            path,
            "/v1/lens/capabilities" | "/v1/lens/assets" | "/v1/lens/jobs"
        ) {
            return true;
        }
        let Some(tail) = path.strip_prefix("/v1/lens/jobs/") else {
            return false;
        };
        let (id, action) = tail.split_once('/').unwrap_or((tail, ""));
        !id.is_empty()
            && (matches!(action, "" | "request" | "result")
                || action.strip_prefix("arrays/").is_some_and(|offset| {
                    !offset.is_empty() && offset.bytes().all(|byte| byte.is_ascii_digit())
                }))
    }

    pub(crate) fn new(
        model_id: String,
        store: Option<Arc<JobStore>>,
        admission: Option<Arc<dyn Admission>>,
    ) -> Self {
        Self {
            model_id,
            store,
            admission,
            submissions: Mutex::new(()),
            access: access::BrowserAccess::default(),
        }
    }

    pub(super) fn with_access(mut self, access: access::BrowserAccess) -> Self {
        self.access = access;
        self
    }

    fn store(&self) -> Result<&JobStore, ApiError> {
        self.store.as_deref().ok_or_else(|| {
            ApiError::new(
                503,
                "history_not_configured",
                "Start qwen serve with --lens-data-dir to enable durable Lens history",
            )
        })
    }

    fn capabilities(&self) -> Value {
        let limits = self
            .store
            .as_ref()
            .map_or_else(Limits::default, |store| store.limits());
        let mut capabilities = if self.store.is_some() {
            self.admission.as_ref().map(|admission| admission.capabilities())
        } else { None }.unwrap_or_else(|| json!({
            "schema_version": 1, "available": false,
            "unavailable_reason": if self.store.is_none() { "history_not_configured" } else { "diagnostic_executor_not_connected" },
            "model": { "id": self.model_id, "identity": null, "template": null, "layers": null, "vocabulary_size": null },
            "input_kinds": [], "generation_modes": [], "assistant_prefill_channels": [],
            "operations": [], "readout_modes": [], "capture_stage": "post_block_after_operations",
            "limits": { "max_new_tokens": 0, "max_context_tokens": 0, "max_operations": 0,
                "max_directions": 0, "max_readouts": 0, "max_top_k": 0 }
        }));
        let queue_limit = capabilities["limits"]["max_queued_jobs"]
            .as_u64()
            .unwrap_or(limits.max_active_jobs as u64)
            .min(limits.max_active_jobs as u64);
        for (key, value) in [
            ("max_body_bytes", limits.max_request_bytes as u64),
            ("max_queued_jobs", queue_limit),
            ("max_retained_jobs", limits.max_retained_jobs as u64),
            ("max_retry_identities", limits.max_retry_identities as u64),
            ("max_history_bytes", limits.max_store_bytes),
            ("max_result_page_records", limits.max_page_records as u64),
            ("max_result_page_bytes", limits.max_page_bytes as u64),
        ] {
            capabilities["limits"][key] = value.into();
        }
        #[cfg(test)]
        if std::env::var("QWEN_LENS_BROWSER_CHILD").as_deref() == Ok("1") {
            capabilities["fixture_owner"] = std::env::var("QWEN_LENS_BROWSER_NONCE").ok().into();
        }
        if let Some(recovery) = self
            .store
            .as_ref()
            .and_then(|store| store.recovery_report())
        {
            capabilities["available"] = false.into();
            capabilities["unavailable_reason"] = "history_recovery_required".into();
            capabilities["storage_recovery"] = json!(recovery);
        }
        capabilities
    }

    pub(crate) fn handle(&self, request: &HttpRequest, stream: &TcpStream) -> io::Result<bool> {
        let path = request.path.split('?').next().unwrap_or_default();
        if path != "/v1/lens" && !path.starts_with("/v1/lens/") {
            return Ok(false);
        }
        let reply = self
            .access
            .check(request)
            .and_then(|()| self.dispatch(request))
            .unwrap_or_else(|error| {
                let mut body = json!({"error": error.error});
                if let Some(key) = error.rejection {
                    body["admission"] =
                        json!({"schema_version":1,"idempotency_key":key,"state":"not_accepted"});
                }
                Reply {
                    status: error.status,
                    location: None,
                    body,
                    binary: None,
                }
            });
        write_reply(stream, reply)?;
        Ok(true)
    }

    fn dispatch(&self, request: &HttpRequest) -> Result<Reply, ApiError> {
        let (path, query) = request.path.split_once('?').unwrap_or((&request.path, ""));
        if request.method == "GET" && !request.body.is_empty() {
            return Err(ApiError::invalid("GET does not accept a body"));
        }
        match (request.method.as_str(), path) {
            ("GET", "/v1/lens/capabilities") => {
                no_query(query)?;
                Reply::ok(self.capabilities())
            }
            ("GET", "/v1/lens/assets") => {
                no_query(query)?;
                Reply::ok(self.admission.as_ref().map_or_else(
                    || json!({"schema_version":1,"assets":[]}),
                    |admission| admission.assets(),
                ))
            }
            ("POST", "/v1/lens/jobs") => {
                no_query(query)?;
                self.submit(&request.body)
            }
            ("GET", "/v1/lens/jobs") => {
                let page =
                    PageQuery::parse(query, self.store()?.limits().max_page_records.min(64))?;
                Reply::ok(self.store()?.history(page.cursor.as_deref(), page.limit)?)
            }
            _ => {
                let tail = path
                    .strip_prefix("/v1/lens/jobs/")
                    .ok_or_else(|| ApiError::new(404, "unknown_route", "Unknown Lens route"))?;
                let (id, action) = tail.split_once('/').unwrap_or((tail, ""));
                if id.is_empty() {
                    return Err(ApiError::new(404, "unknown_job", "Unknown job"));
                }
                match (request.method.as_str(), action) {
                    ("GET", action) if action.starts_with("arrays/") => {
                        no_query(query)?;
                        let locator = action.strip_prefix("arrays/").unwrap();
                        if locator.is_empty() || !locator.bytes().all(|b| b.is_ascii_digit()) {
                            return Err(ApiError::invalid("invalid array locator"));
                        }
                        let offset = locator
                            .parse::<u64>()
                            .map_err(|_| ApiError::invalid("invalid array locator"))?;
                        Ok(Reply {
                            status: 200,
                            location: None,
                            body: Value::Null,
                            binary: Some(self.store()?.array(id, offset)?),
                        })
                    }
                    ("GET", "") => {
                        no_query(query)?;
                        Reply::ok(self.store()?.status(id)?)
                    }
                    ("GET", "request") => {
                        no_query(query)?;
                        Reply::ok(
                            json!({"schema_version":1,"job_id":id,"request":self.store()?.request(id)?}),
                        )
                    }
                    ("GET", "result") => {
                        let page = PageQuery::parse(
                            query,
                            self.store()?.limits().max_page_records.min(64),
                        )?;
                        Reply::ok(
                            self.store()?
                                .result(id, page.cursor.as_deref(), page.limit)?,
                        )
                    }
                    ("POST", "cancel" | "delete") => {
                        no_query(query)?;
                        let store = self.store()?;
                        if !request.body.is_empty()
                            && serde_json::from_slice::<Value>(&request.body).ok()
                                != Some(json!({}))
                        {
                            return Err(ApiError::invalid("Control body must be empty or {}"));
                        }
                        Reply::ok(if action == "delete" {
                            store.delete(id)?
                        } else {
                            store.cancel(id)?
                        })
                    }
                    _ => Err(ApiError::new(
                        404,
                        "unknown_route",
                        "Unknown Lens route or method",
                    )),
                }
            }
        }
    }

    fn submit(&self, bytes: &[u8]) -> Result<Reply, ApiError> {
        let store = self.store()?;
        if bytes.len() > store.limits().max_request_bytes {
            return Err(ApiError::new(
                413,
                "body_too_large",
                "Lens request exceeds advertised body limit",
            ));
        }
        let request: Value = serde_json::from_slice(bytes)
            .map_err(|error| ApiError::invalid(format!("Invalid JSON: {error}")))?;
        let key = request
            .get("idempotency_key")
            .and_then(Value::as_str)
            .filter(|key| !key.is_empty() && key.len() <= 256)
            .ok_or_else(|| ApiError::invalid("idempotency_key must contain 1..256 bytes"))?;
        if request.get("schema_version").and_then(Value::as_u64) != Some(1) {
            return Err(ApiError::invalid("schema_version must be 1"));
        }
        let observations = ["/diagnostics/readouts", "/diagnostics/residual_pairs"]
            .iter()
            .any(|path| {
                request
                    .pointer(path)
                    .and_then(Value::as_array)
                    .is_some_and(|rows| !rows.is_empty())
            });
        // Serialize submissions, not metadata or the model queue. This makes
        // recovery of a concurrent duplicate independent of free queue slots.
        let _submissions = self
            .submissions
            .lock()
            .unwrap_or_else(|cause| cause.into_inner());
        if let Some(status) = store.lookup(key, &request, observations)? {
            return Reply::ok(status);
        }
        let parsed = input::Request::parse(&request)
            .map_err(|error| ApiError::invalid(error.to_string()).not_accepted(key))?;
        let admission = self.admission.as_ref().ok_or_else(|| {
            ApiError::new(
                503,
                "unsupported_capability",
                "The native diagnostic executor is not connected; no job was accepted",
            )
            .not_accepted(key)
        })?;
        let reservation = admission
            .reserve(&parsed)
            .map_err(|error| error.not_accepted(key))?;
        let accepted = store
            .accept_with_archive(key, &request, observations, reservation.archive_bytes())
            .map_err(|error| ApiError::from(error).not_accepted(key))?;
        if !accepted.created {
            return Reply::ok(accepted.status);
        }
        let id = accepted.status.id.clone();
        let control = store.control(&id).map_err(ApiError::after_acceptance)?;
        if let Err(message) = reservation.enqueue(AcceptedJob {
            id: id.clone(),
            #[cfg(test)]
            request,
            control,
        }) {
            tracing::error!(
                job_id = id,
                "Lens dispatch failed after acceptance: {message}"
            );
            let error = JobError {
                r#type: "server_error".into(),
                code: "dispatch_failed".into(),
                param: None,
                message:
                    "The model queue stopped after acceptance; this execution was not retried."
                        .into(),
            };
            let publication = store.fail_dispatch(&id, error);
            store.execution_settled(&id);
            publication.map_err(ApiError::after_acceptance)?;
        }
        let status: JobStatus = store.status(&id).map_err(ApiError::after_acceptance)?;
        Ok(Reply {
            status: 202,
            location: Some(format!("/v1/lens/jobs/{id}")),
            body: serde_json::to_value(status).expect("job status JSON"),
            binary: None,
        })
    }
}

fn no_query(query: &str) -> Result<(), ApiError> {
    if query.is_empty() {
        Ok(())
    } else {
        Err(ApiError::invalid(
            "This route does not accept query parameters",
        ))
    }
}

struct PageQuery {
    cursor: Option<String>,
    limit: usize,
}
impl PageQuery {
    fn parse(query: &str, default_limit: usize) -> Result<Self, ApiError> {
        let mut cursor = None;
        let mut limit = None;
        if !query.is_empty() {
            for pair in query.split('&') {
                let (key, value) = pair
                    .split_once('=')
                    .ok_or_else(|| ApiError::invalid("Malformed query parameter"))?;
                let value = decode_query_value(value)?;
                match key {
                    "cursor" if cursor.is_none() && !value.is_empty() => cursor = Some(value),
                    "limit"
                        if limit.is_none()
                            && !value.is_empty()
                            && value.bytes().all(|byte| byte.is_ascii_digit()) =>
                    {
                        limit = Some(
                            value
                                .parse::<usize>()
                                .map_err(|_| ApiError::invalid("Invalid limit"))?,
                        );
                    }
                    _ => {
                        return Err(ApiError::invalid(
                            "Unknown, empty or duplicate query parameter",
                        ));
                    }
                }
            }
        }
        Ok(Self {
            cursor,
            limit: limit.unwrap_or(default_limit),
        })
    }
}

fn decode_query_value(value: &str) -> Result<String, ApiError> {
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' => {
                let hex = bytes
                    .get(index + 1..index + 3)
                    .and_then(|bytes| std::str::from_utf8(bytes).ok())
                    .and_then(|hex| u8::from_str_radix(hex, 16).ok())
                    .ok_or_else(|| ApiError::invalid("Invalid query encoding"))?;
                output.push(hex);
                index += 3;
            }
            b'+' => {
                output.push(b' ');
                index += 1;
            }
            byte => {
                output.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(output).map_err(|_| ApiError::invalid("Query must be UTF-8"))
}

fn write_reply(mut stream: &TcpStream, reply: Reply) -> io::Result<()> {
    let content_type = if reply.binary.is_some() {
        "application/octet-stream"
    } else {
        "application/json"
    };
    let bytes = match reply.binary {
        Some(bytes) => bytes,
        None => serde_json::to_vec(&reply.body)?,
    };
    let reason = match reply.status {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        410 => "Gone",
        412 => "Precondition Failed",
        413 => "Content Too Large",
        429 => "Too Many Requests",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {} {reason}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\ncache-control: no-store\r\nx-content-type-options: nosniff\r\nconnection: close\r\n",
        reply.status,
        bytes.len()
    )?;
    if let Some(location) = reply.location {
        write!(stream, "location: {location}\r\n")?;
    }
    stream.write_all(b"\r\n")?;
    stream.write_all(&bytes)?;
    stream.flush()
}
