use super::*;
use crate::serve::jobs::state::JobState;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct TestRoot(PathBuf);
impl TestRoot {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
                "qwen-lens-http-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            )))
    }
    fn store(&self) -> Arc<JobStore> {
        Arc::new(JobStore::open(&self.0, Limits::default()).unwrap())
    }
}
impl Drop for TestRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn request(key: &str) -> Value {
    let mut request: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/lens_http_v1/request.json"
    ))
    .unwrap();
    request["idempotency_key"] = key.into();
    request
}

// This fake exercises transport/admission ordering only. It never claims to
// validate the model's diagnostic plan or execute a transformer forward.
struct FakeAdmission {
    store: Arc<JobStore>,
    pending: Mutex<Vec<AcceptedJob>>,
    used: AtomicUsize,
    fail_delivery: AtomicBool,
    paused_delivery: Mutex<
        Option<(
            std::sync::mpsc::SyncSender<String>,
            std::sync::mpsc::Receiver<()>,
        )>,
    >,
}
impl FakeAdmission {
    fn new(store: Arc<JobStore>) -> Arc<Self> {
        Arc::new(Self {
            store,
            pending: Mutex::new(Vec::new()),
            used: AtomicUsize::new(0),
            fail_delivery: AtomicBool::new(false),
            paused_delivery: Mutex::new(None),
        })
    }
    fn take(&self) -> AcceptedJob {
        let job = self.pending.lock().unwrap().pop().unwrap();
        self.used.fetch_sub(1, Ordering::AcqRel);
        job
    }
}
struct FakeReservation {
    queue: Arc<FakeAdmission>,
    delivered: bool,
}
impl Drop for FakeReservation {
    fn drop(&mut self) {
        if !self.delivered {
            self.queue.used.fetch_sub(1, Ordering::AcqRel);
        }
    }
}
impl ReservedSubmission for FakeReservation {
    fn enqueue(mut self: Box<Self>, job: AcceptedJob) -> Result<(), String> {
        assert_eq!(
            self.queue.store.status(&job.id).unwrap().state,
            JobState::Queued
        );
        assert_eq!(self.queue.store.request(&job.id).unwrap(), job.request);
        if let Some((entered, release)) = self.queue.paused_delivery.lock().unwrap().take() {
            entered.send(job.id.clone()).unwrap();
            release
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
        }
        if self.queue.fail_delivery.load(Ordering::Acquire) {
            return Err("model owner stopped".into());
        }
        self.queue.pending.lock().unwrap().push(job);
        self.delivered = true;
        Ok(())
    }
}
impl Admission for Arc<FakeAdmission> {
    fn capabilities(&self) -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/lens_http_v1/capabilities.json"
        ))
        .unwrap()
    }
    fn assets(&self) -> Value {
        json!({"schema_version":1, "assets":[]})
    }
    fn reserve(&self, _: &super::input::Request) -> Result<Box<dyn ReservedSubmission>, ApiError> {
        self.used
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| ApiError::new(429, "queue_full", "Queue is full"))?;
        Ok(Box::new(FakeReservation {
            queue: Arc::clone(self),
            delivered: false,
        }))
    }
}
fn enabled(store: Arc<JobStore>, queue: Arc<FakeAdmission>) -> Arc<LensApi> {
    Arc::new(LensApi::new(
        "test-model".into(),
        Some(store),
        Some(Arc::new(queue)),
    ))
}

struct GuardedAdmission {
    queue: Arc<FakeAdmission>,
    model: &'static str,
    fit: &'static str,
}
impl Admission for GuardedAdmission {
    fn capabilities(&self) -> Value {
        self.queue.capabilities()
    }
    fn assets(&self) -> Value {
        self.queue.assets()
    }
    fn reserve(&self, request: &input::Request) -> Result<Box<dyn ReservedSubmission>, ApiError> {
        crate::serve::native::preconditions::check(request, self.model, |alias| {
            (alias == "j").then_some(self.fit)
        })?;
        self.queue.reserve(request)
    }
}

#[test]
fn stale_bindings_reject_before_acceptance_but_never_block_accepted_key_recovery() {
    let root = TestRoot::new();
    let store = root.store();
    let queue = FakeAdmission::new(store.clone());
    let make = |model, fit| {
        Arc::new(LensApi::new(
            "test".into(),
            Some(store.clone()),
            Some(Arc::new(GuardedAdmission {
                queue: queue.clone(),
                model,
                fit,
            })),
        ))
    };
    let mut value = request("guarded");
    value["diagnostics"]["directions"][0]["lens"] = "j".into();
    value["diagnostics"]["readouts"][0]["lens"] = "j".into();
    value["preconditions"] = json!({"model_identity":"model-a","asset_identities":{"j":"fit-a"}});
    for (model, fit) in [("model-b", "fit-a"), ("model-a", "fit-b")] {
        let (headers, body) = roundtrip(
            make(model, fit),
            "POST",
            "/v1/lens/jobs",
            &value.to_string(),
        );
        assert!(headers.starts_with("HTTP/1.1 412 Precondition Failed"));
        assert_eq!(body["error"]["code"], "binding_mismatch");
        assert_eq!(body["admission"]["state"], "not_accepted");
        assert_eq!(body["admission"]["idempotency_key"], "guarded");
        assert!(store.lookup("guarded", &value, true).unwrap().is_none());
        assert_eq!(queue.used.load(Ordering::Acquire), 0);
    }
    let (_, accepted) = roundtrip(
        make("model-a", "fit-a"),
        "POST",
        "/v1/lens/jobs",
        &value.to_string(),
    );
    assert_eq!(queue.used.load(Ordering::Acquire), 1);
    let changed = make("model-b", "fit-b");
    let (headers, recovered) =
        roundtrip(changed.clone(), "POST", "/v1/lens/jobs", &value.to_string());
    assert!(headers.starts_with("HTTP/1.1 200"));
    assert_eq!(accepted, recovered);
    let unavailable = Arc::new(LensApi::new("other".into(), Some(store.clone()), None));
    assert_eq!(
        roundtrip(unavailable, "POST", "/v1/lens/jobs", &value.to_string()).1,
        accepted
    );
    value["preconditions"]["asset_identities"]["j"] = "fit-b".into();
    assert!(
        roundtrip(changed.clone(), "POST", "/v1/lens/jobs", &value.to_string())
            .0
            .starts_with("HTTP/1.1 409")
    );
    value["idempotency_key"] = "new-key".into();
    assert!(
        roundtrip(changed, "POST", "/v1/lens/jobs", &value.to_string())
            .0
            .starts_with("HTTP/1.1 412")
    );
    assert_eq!(queue.pending.lock().unwrap().len(), 1);
}

fn roundtrip(api: Arc<LensApi>, method: &str, path: &str, body: &str) -> (String, Value) {
    roundtrip_headers(api, method, path, body, "localhost", None)
}

fn roundtrip_headers(
    api: Arc<LensApi>,
    method: &str,
    path: &str,
    body: &str,
    host: &str,
    origin: Option<&str>,
) -> (String, Value) {
    let (headers, body) = roundtrip_binary(api, method, path, body, host, origin);
    (headers, serde_json::from_slice(&body).unwrap())
}

fn roundtrip_binary(
    api: Arc<LensApi>,
    method: &str,
    path: &str,
    body: &str,
    host: &str,
    origin: Option<&str>,
) -> (String, Vec<u8>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    client
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .unwrap();
    let worker = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        handle_api(stream, api).unwrap();
    });
    let origin = origin.map_or(String::new(), |origin| format!("origin: {origin}\r\n"));
    write!(client, "{method} {path} HTTP/1.1\r\nhost: {host}\r\n{origin}content-type: application/json\r\ncontent-length: {}\r\n\r\n{body}", body.len()).unwrap();
    let mut reply = Vec::new();
    client.read_to_end(&mut reply).unwrap();
    worker.join().unwrap();
    let boundary = reply.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
    (
        String::from_utf8(reply[..boundary].to_vec()).unwrap(),
        reply[boundary + 4..].to_vec(),
    )
}

fn handle_api(stream: TcpStream, api: Arc<LensApi>) -> io::Result<()> {
    let mut activity = crate::serve::owner_activity::OwnerActivity::default();
    let (sender, _receiver) = std::sync::mpsc::sync_channel(1);
    let profile = crate::serve::control::Profile {
        assets: None,
        model_id: "test-model".into(),
        request: crate::serve::request_profile::RequestProfile::UnboundQwen,
        lens: api,
        classified: None,
        gate: Default::default(),
        activity: activity.admission(),
        sender,
        trace: None,
    };
    let result = crate::serve::control::handle(stream, &profile);
    activity.drain_finished(|| {});
    assert!(activity.is_settled());
    result
}

#[test]
fn pair_only_acceptance_requests_observations_and_counts_pair_records() {
    let root = TestRoot::new();
    let store = root.store();
    let queue = FakeAdmission::new(store.clone());
    let api = enabled(store.clone(), queue.clone());
    let mut value = request("pair-only");
    value["diagnostics"] = json!({"directions":[],"operations":[],"readouts":[],
        "residual_pairs":[{"id":"pair","scope":{"layers":{"kind":"values","values":[0]},"prefill":{"kind":"values","values":[0]}}}]});
    let (headers, response) = roundtrip(api, "POST", "/v1/lens/jobs", &value.to_string());
    assert!(headers.starts_with("HTTP/1.1 202"), "{headers} {response}");
    assert_ne!(response["observations"]["state"], "not_requested");
    let job = queue.take();
    store.start(&job.id, 1).unwrap();
    store
        .append(&job.id, &[json!({"kind":"residual_pair","id":"pair"})])
        .unwrap();
    assert_eq!(
        store
            .status(&job.id)
            .unwrap()
            .observations
            .committed_records,
        1
    );
}

#[test]
fn retained_binary_routes_are_read_only_verified_and_local_without_an_executor() {
    let root = TestRoot::new();
    let store = root.store();
    let id = store
        .accept_with_archive("binary", &request("binary"), true, 8)
        .unwrap()
        .status
        .id;
    store.start(&id, 2).unwrap();
    let bytes: Vec<u8> = [1.25f32, -3.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    store
        .append_array(&id, json!({"key":"fixture"}), &bytes)
        .unwrap();
    let before = store.status(&id).unwrap();
    let api = Arc::new(LensApi::new(
        "unavailable".into(),
        Some(store.clone()),
        None,
    ));
    let path = format!("/v1/lens/jobs/{id}/arrays/0");
    let (headers, body) = roundtrip_binary(api.clone(), "GET", &path, "", "localhost", None);
    assert!(headers.starts_with("HTTP/1.1 200"));
    assert!(headers.contains("content-type: application/octet-stream"));
    assert_eq!(body, bytes);
    assert_eq!(store.status(&id).unwrap(), before);
    for bad in [
        format!("{path}?x=1"),
        format!("/v1/lens/jobs/{id}/arrays/../request"),
        format!("/v1/lens/jobs/{id}/arrays/1"),
    ] {
        assert!(
            roundtrip(api.clone(), "GET", &bad, "")
                .0
                .starts_with("HTTP/1.1 400")
        );
    }
    assert!(
        roundtrip_headers(
            api.clone(),
            "GET",
            &path,
            "",
            "localhost",
            Some("http://evil.example")
        )
        .0
        .starts_with("HTTP/1.1 403")
    );
    std::fs::write(root.0.join(&id).join("arrays.bin"), [0; 8]).unwrap();
    assert!(
        roundtrip(api, "GET", &path, "")
            .0
            .starts_with("HTTP/1.1 500")
    );
}

#[test]
fn acceptance_is_durable_before_enqueue_and_response_has_location() {
    let root = TestRoot::new();
    let store = root.store();
    let queue = FakeAdmission::new(Arc::clone(&store));
    let api = enabled(Arc::clone(&store), Arc::clone(&queue));
    let (headers, status) = roundtrip(
        api,
        "POST",
        "/v1/lens/jobs",
        &request("accepted").to_string(),
    );
    assert!(headers.starts_with("HTTP/1.1 202 Accepted"));
    assert!(headers.contains(&format!(
        "location: /v1/lens/jobs/{}",
        status["id"].as_str().unwrap()
    )));
    assert!(headers.contains("cache-control: no-store"));
    let job = queue.take();
    assert_eq!(job.id, status["id"]);
    assert!(job.control.checkpoint().is_ok());
}

#[test]
fn duplicate_recovers_while_queue_is_full_and_conflict_never_enqueues() {
    let root = TestRoot::new();
    let store = root.store();
    let queue = FakeAdmission::new(Arc::clone(&store));
    let api = enabled(Arc::clone(&store), Arc::clone(&queue));
    let (_, initial) = roundtrip(
        Arc::clone(&api),
        "POST",
        "/v1/lens/jobs",
        &request("duplicate").to_string(),
    );
    let (headers, repeated) = roundtrip(
        Arc::clone(&api),
        "POST",
        "/v1/lens/jobs",
        &request("duplicate").to_string(),
    );
    assert!(headers.starts_with("HTTP/1.1 200"));
    assert_eq!(initial, repeated);
    let mut changed = request("duplicate");
    changed["generation"]["max_new_tokens"] = 99.into();
    let (headers, error) = roundtrip(
        Arc::clone(&api),
        "POST",
        "/v1/lens/jobs",
        &changed.to_string(),
    );
    assert!(headers.starts_with("HTTP/1.1 409"));
    assert_eq!(error["error"]["code"], "idempotency_conflict");
    let (headers, _) = roundtrip(
        api,
        "POST",
        "/v1/lens/jobs",
        &request("another").to_string(),
    );
    assert!(headers.starts_with("HTTP/1.1 429"));
    assert_eq!(store.history(None, 10).unwrap().jobs.len(), 1);
    assert_eq!(queue.pending.lock().unwrap().len(), 1);
}

#[test]
fn failed_post_acceptance_delivery_is_a_terminal_accepted_job_not_a_rejection() {
    let root = TestRoot::new();
    let store = root.store();
    let queue = FakeAdmission::new(Arc::clone(&store));
    queue.fail_delivery.store(true, Ordering::Release);
    let api = enabled(store, Arc::clone(&queue));
    let (headers, status) = roundtrip(
        Arc::clone(&api),
        "POST",
        "/v1/lens/jobs",
        &request("failed-delivery").to_string(),
    );
    assert!(headers.starts_with("HTTP/1.1 202"));
    assert_eq!(status["state"], "failed");
    assert_eq!(status["generation"]["error"]["code"], "dispatch_failed");
    assert_eq!(queue.used.load(Ordering::Acquire), 0);
    let (headers, retry) = roundtrip(
        api,
        "POST",
        "/v1/lens/jobs",
        &request("failed-delivery").to_string(),
    );
    assert!(headers.starts_with("HTTP/1.1 200"));
    assert_eq!(retry, status);
}

#[test]
fn disconnected_acceptance_response_does_not_cancel_or_duplicate_work() {
    let root = TestRoot::new();
    let store = root.store();
    let queue = FakeAdmission::new(Arc::clone(&store));
    let api = enabled(store, Arc::clone(&queue));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let server_api = Arc::clone(&api);
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        // A failed socket write is expected; acceptance remains durable.
        let _ = handle_api(stream, server_api);
    });
    let body = request("disconnect").to_string();
    write!(
        client,
        "POST /v1/lens/jobs HTTP/1.1\r\nhost: localhost\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    drop(client);
    server.join().unwrap();
    let job = queue.take();
    assert!(job.control.checkpoint().is_ok());
    let (headers, retry) = roundtrip(api, "POST", "/v1/lens/jobs", &body);
    assert!(headers.starts_with("HTTP/1.1 200"));
    assert_eq!(retry["id"], job.id);
    assert!(queue.pending.lock().unwrap().is_empty());
}

#[test]
fn reopened_history_and_request_recovery_work_without_a_diagnostic_executor() {
    let root = TestRoot::new();
    let id;
    {
        let store = root.store();
        id = store
            .accept("recover", &request("recover"), true)
            .unwrap()
            .status
            .id;
    }
    let store = root.store();
    let api = Arc::new(LensApi::new("different-model".into(), Some(store), None));
    let (headers, status) = roundtrip(
        Arc::clone(&api),
        "POST",
        "/v1/lens/jobs",
        &request("recover").to_string(),
    );
    assert!(headers.starts_with("HTTP/1.1 200"));
    assert_eq!(status["state"], "interrupted");
    let (_, history) = roundtrip(Arc::clone(&api), "GET", "/v1/lens/jobs?limit=1", "");
    assert_eq!(history["jobs"][0]["id"], id);
    assert_eq!(
        history["request_previews"][&id]["text"],
        request("recover")["input"]["messages"][0]["content"]
    );
    assert_eq!(history["request_previews"].as_object().unwrap().len(), 1);
    let (_, again) = roundtrip(Arc::clone(&api), "GET", "/v1/lens/jobs?limit=1", "");
    assert_eq!(again, history);
    let (_, saved) = roundtrip(
        Arc::clone(&api),
        "GET",
        &format!("/v1/lens/jobs/{id}/request"),
        "",
    );
    assert_eq!(saved["job_id"], id);
    assert_eq!(saved["request"], request("recover"));
    let (_, page) = roundtrip(
        Arc::clone(&api),
        "GET",
        &format!("/v1/lens/jobs/{id}/result"),
        "",
    );
    assert_eq!(page["complete"], true);
    let (headers, _) = roundtrip(api, "POST", "/v1/lens/jobs", &request("new").to_string());
    assert!(headers.starts_with("HTTP/1.1 503"));
}

#[test]
fn result_polling_is_cpu_only_and_cancel_signals_the_accepted_job() {
    let root = TestRoot::new();
    let store = root.store();
    let queue = FakeAdmission::new(Arc::clone(&store));
    let api = enabled(Arc::clone(&store), Arc::clone(&queue));
    let (_, accepted) = roundtrip(
        Arc::clone(&api),
        "POST",
        "/v1/lens/jobs",
        &request("read").to_string(),
    );
    let id = accepted["id"].as_str().unwrap();
    let job = queue.take();
    store.start(id, 2).unwrap();
    store
        .append(
            id,
            &[json!({"kind":"prepared_input"}), json!({"kind":"readout"})],
        )
        .unwrap();
    let (_, first) = roundtrip(
        Arc::clone(&api),
        "GET",
        &format!("/v1/lens/jobs/{id}/result?limit=1"),
        "",
    );
    let (_, second) = roundtrip(
        Arc::clone(&api),
        "GET",
        &format!(
            "/v1/lens/jobs/{id}/result?limit=1&cursor={}",
            first["next_cursor"].as_str().unwrap()
        ),
        "",
    );
    assert_eq!(first["records"][0]["seq"], 0);
    assert_eq!(second["records"][0]["seq"], 1);
    assert!(job.control.checkpoint().is_ok());
    assert!(queue.pending.lock().unwrap().is_empty());
    let (_, cancelled) = roundtrip(
        Arc::clone(&api),
        "POST",
        &format!("/v1/lens/jobs/{id}/cancel"),
        "{}",
    );
    assert_eq!(cancelled["state"], "running");
    assert_eq!(cancelled["cancel_requested"], true);
    assert!(job.control.checkpoint().is_err());
    let (_, repeated) = roundtrip(api, "POST", &format!("/v1/lens/jobs/{id}/cancel"), "");
    assert_eq!(cancelled, repeated);
}

#[test]
fn unavailable_capabilities_are_explicit_and_never_claim_zero_layer_models() {
    let api = Arc::new(LensApi::new("loaded".into(), None, None));
    let (headers, caps) = roundtrip(Arc::clone(&api), "GET", "/v1/lens/capabilities", "");
    assert!(headers.starts_with("HTTP/1.1 200"));
    assert_eq!(caps["available"], false);
    assert_eq!(caps["model"]["id"], "loaded");
    assert_eq!(caps["model"]["layers"], Value::Null);
    let (_, assets) = roundtrip(Arc::clone(&api), "GET", "/v1/lens/assets", "");
    assert_eq!(assets["assets"], json!([]));
    for body in ["not JSON", "[]", "{}"] {
        let (headers, error) = roundtrip(api.clone(), "POST", "/v1/lens/jobs/missing/cancel", body);
        assert!(headers.starts_with("HTTP/1.1 503"));
        assert_eq!(error["error"]["code"], "history_not_configured");
    }
    let (headers, error) = roundtrip(api, "GET", "/v1/lens/jobs", "");
    assert!(headers.starts_with("HTTP/1.1 503"));
    assert_eq!(error["error"]["code"], "history_not_configured");
}

#[test]
fn production_unavailable_response_matches_the_shared_browser_fixture() {
    let root = TestRoot::new();
    let api = Arc::new(LensApi::new(
        "example-model".into(),
        Some(root.store()),
        None,
    ));
    let (_, caps) = roundtrip(api, "GET", "/v1/lens/capabilities", "");
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/lens_http_v1/capabilities_unavailable.json"
    ))
    .unwrap();
    assert_eq!(caps, fixture);
}

#[test]
fn malformed_requests_never_create_jobs_and_pagination_rejects_ambiguity() {
    let root = TestRoot::new();
    let store = root.store();
    let queue = FakeAdmission::new(Arc::clone(&store));
    let api = enabled(Arc::clone(&store), queue);
    for body in [
        "{",
        "{}",
        "{\"schema_version\":2,\"idempotency_key\":\"x\"}",
    ] {
        assert!(
            roundtrip(Arc::clone(&api), "POST", "/v1/lens/jobs", body)
                .0
                .starts_with("HTTP/1.1 400")
        );
    }
    let mut invalid = request("raw");
    invalid["input"]["kind"] = "raw".into();
    assert!(
        roundtrip(
            Arc::clone(&api),
            "POST",
            "/v1/lens/jobs",
            &invalid.to_string()
        )
        .0
        .starts_with("HTTP/1.1 400")
    );
    for query in [
        "limit=0",
        "limit=1&limit=2",
        "cursor=",
        "cursor=%GG",
        "cursor=%ff",
        "unknown=1",
        "limit=-1",
        "limit=999999999999999999999999999999999999999",
    ] {
        assert!(
            roundtrip(
                Arc::clone(&api),
                "GET",
                &format!("/v1/lens/jobs?{query}"),
                ""
            )
            .0
            .starts_with("HTTP/1.1 400"),
            "{query}"
        );
    }
    assert!(store.history(None, 10).unwrap().jobs.is_empty());
}

#[test]
fn body_limit_is_enforced_before_admission_and_errors_preserve_uncertainty() {
    let root = TestRoot::new();
    let store = root.store();
    let queue = FakeAdmission::new(Arc::clone(&store));
    let api = enabled(Arc::clone(&store), queue);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    client
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .unwrap();
    let worker = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        handle_api(stream, api).unwrap();
    });
    // Refusal must not wait for or allocate the declared body.
    write!(
        client,
        "POST /v1/lens/jobs HTTP/1.1\r\nhost: localhost\r\ncontent-length: {}\r\n\r\n",
        store.limits().max_request_bytes + 1
    )
    .unwrap();
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    worker.join().unwrap();
    assert!(response.starts_with("HTTP/1.1 413"));
    assert!(store.history(None, 10).unwrap().jobs.is_empty());
    let error: ApiError = StoreError::Storage(io::Error::other("uncertain rename")).into();
    assert_eq!(error.status, 500);
    assert_eq!(error.error.code, "storage_unavailable");
}

#[test]
fn rebinding_hosts_and_foreign_origins_cannot_read_prompts_or_cancel() {
    let root = TestRoot::new();
    let store = root.store();
    let id = store
        .accept("private", &request("private"), true)
        .unwrap()
        .status
        .id;
    let api = Arc::new(LensApi::new(
        "test-model".into(),
        Some(Arc::clone(&store)),
        None,
    ));
    for (host, origin) in [
        ("rebind.example:8737", None),
        ("localhost", Some("http://evil.example")),
        ("localhost", Some("null")),
        ("localhost", Some("http://localhost:9999")),
    ] {
        let (headers, error) = roundtrip_headers(
            Arc::clone(&api),
            "GET",
            &format!("/v1/lens/jobs/{id}/request"),
            "",
            host,
            origin,
        );
        assert!(headers.starts_with("HTTP/1.1 403"));
        assert!(error.get("request").is_none());
        let (headers, _) = roundtrip_headers(
            Arc::clone(&api),
            "POST",
            &format!("/v1/lens/jobs/{id}/cancel"),
            "",
            host,
            origin,
        );
        assert!(headers.starts_with("HTTP/1.1 403"));
    }
    assert_eq!(store.status(&id).unwrap().state, JobState::Queued);
    for (host, origin) in [
        ("localhost", Some("http://localhost")),
        ("127.0.0.1:8737", None),
        ("[::1]:8737", Some("http://[::1]:8737")),
    ] {
        assert!(
            roundtrip_headers(
                Arc::clone(&api),
                "GET",
                &format!("/v1/lens/jobs/{id}/request"),
                "",
                host,
                origin
            )
            .0
            .starts_with("HTTP/1.1 200")
        );
    }
}

#[test]
fn cancellation_racing_failed_delivery_returns_the_accepted_terminal_job() {
    let root = TestRoot::new();
    let store = root.store();
    let queue = FakeAdmission::new(Arc::clone(&store));
    queue.fail_delivery.store(true, Ordering::Release);
    let (entered, waiting) = std::sync::mpsc::sync_channel(1);
    let (release, held) = std::sync::mpsc::sync_channel(1);
    *queue.paused_delivery.lock().unwrap() = Some((entered, held));
    let api = enabled(Arc::clone(&store), queue);
    let submit_api = Arc::clone(&api);
    let submission = std::thread::spawn(move || {
        roundtrip(
            submit_api,
            "POST",
            "/v1/lens/jobs",
            &request("race").to_string(),
        )
    });
    let id = waiting
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    let (_, cancelled) = roundtrip(
        Arc::clone(&api),
        "POST",
        &format!("/v1/lens/jobs/{id}/cancel"),
        "{}",
    );
    release.send(()).unwrap();
    let (headers, accepted) = submission.join().unwrap();
    assert!(headers.starts_with("HTTP/1.1 202"));
    assert_eq!(accepted, cancelled);
    assert_eq!(accepted["state"], "cancelled");
    let (headers, retry) = roundtrip(api, "POST", "/v1/lens/jobs", &request("race").to_string());
    assert!(headers.starts_with("HTTP/1.1 200"));
    assert_eq!(accepted, retry);
}

#[test]
fn only_pre_acceptance_decisions_provide_a_keyed_rejection() {
    let root = TestRoot::new();
    let store = root.store();
    let api = Arc::new(LensApi::new("test-model".into(), Some(store), None));
    let (headers, error) = roundtrip(
        api,
        "POST",
        "/v1/lens/jobs",
        &request("rejected").to_string(),
    );
    assert!(headers.starts_with("HTTP/1.1 503"));
    assert_eq!(
        error["admission"],
        json!({"schema_version":1,"idempotency_key":"rejected","state":"not_accepted"})
    );
    for error in [
        ApiError::new(409, "idempotency_conflict", "conflict"),
        ApiError::new(500, "storage_unavailable", "uncertain"),
    ] {
        assert!(error.not_accepted("rejected").rejection.is_none());
    }
}
