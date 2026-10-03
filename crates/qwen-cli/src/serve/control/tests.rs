use super::*;
use crate::serve::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use crate::serve::items::ServeRequest;
use std::io::Read;
use std::sync::{
    atomic::AtomicUsize,
    mpsc::{Receiver, Sender, channel},
};

const WAIT: Duration = Duration::from_secs(5);

#[test]
#[ignore = "CPU browser fixture; invoked by web/baseline-browser-check.ts"]
fn browser_baseline_child() {
    assert_eq!(std::env::var("QWEN_LENS_BROWSER_CHILD").as_deref(), Ok("1"));
    crate::shutdown::install().unwrap();
    let mut fixture = native::CpuFixture::new();
    Arc::get_mut(&mut fixture.profile).unwrap().plain_readouts =
        std::env::var("QWEN_LENS_BROWSER_READOUTS").as_deref() == Ok("1");
    let _fitted_files = if std::env::var("QWEN_LENS_BROWSER_FITTED").as_deref() == Ok("1") {
        let (files, registry) = native::registry::tests::fitted_fixture();
        let profile = Arc::get_mut(&mut fixture.profile).unwrap();
        profile.plain_readouts = true;
        profile.layers = 3;
        profile.registry = Some(registry);
        Some(files)
    } else {
        None
    };
    let assets = crate::serve::assets::WebAssets::open(std::path::Path::new(
        &std::env::var("QWEN_LENS_BROWSER_WEB_ROOT").unwrap(),
    ))
    .unwrap();
    struct BrowserBackend(Arc<native::Profile>);
    impl GenerationBackend for BrowserBackend {
        fn model_id(&self) -> &str {
            "test"
        }
        fn native_profile(&self) -> Result<Option<Arc<native::Profile>>> {
            Ok(Some(Arc::clone(&self.0)))
        }
        fn generate_native(
            &mut self,
            prepared: &native::Prepared,
            sink: &native::Sink,
        ) -> native::Outcome {
            if self.0.plain_readouts {
                return native::run_cpu_readouts(prepared, sink, &self.0.tokenizer);
            }
            native::run_tokens(
                prepared,
                sink,
                &[],
                |_, _, _| {
                    let mut logits = vec![0.0; 261];
                    logits[120] = 100.0;
                    Ok(logits)
                },
                |_| Ok(b"x".to_vec()),
            )
        }
        fn generate(
            &mut self,
            _: &ServeRequest,
            _: &str,
            _: &mut dyn GenerationSink,
        ) -> Result<GenerationOutcome, BackendFailure> {
            panic!("browser baseline must use native jobs")
        }
    }
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    println!(
        "qwen-lens-browser-ready:http://{}",
        listener.local_addr().unwrap()
    );
    std::io::stdout().flush().unwrap();
    let cause = crate::serve::accept_loop(
        listener,
        "test",
        0.0,
        &mut BrowserBackend(Arc::clone(&fixture.profile)),
        &mut None,
        Some(crate::serve::Workbench {
            store: Arc::clone(&fixture.store),
            assets: Some(Arc::new(assets)),
        }),
    )
    .unwrap_err();
    assert!(
        cause.to_string().contains("termination signal 15"),
        "{cause:#}"
    );
    println!("qwen-lens-browser-stopped");
}

struct Backend {
    profile: Arc<native::Profile>,
    owner: std::thread::ThreadId,
    entered: Sender<ExecutionGate>,
    release: Receiver<()>,
    calls: Arc<AtomicUsize>,
    ordinary: Arc<AtomicUsize>,
    callbacks: Arc<Mutex<Vec<&'static str>>>,
}
impl GenerationBackend for Backend {
    fn model_id(&self) -> &str {
        "test"
    }
    fn native_profile(&self) -> Result<Option<Arc<native::Profile>>> {
        Ok(Some(Arc::clone(&self.profile)))
    }
    fn generate_native(
        &mut self,
        prepared: &native::Prepared,
        sink: &native::Sink,
    ) -> native::Outcome {
        assert_eq!(self.owner, std::thread::current().id());
        self.calls.fetch_add(1, Ordering::AcqRel);
        if self.profile.plain_readouts {
            self.entered.send(sink.server.clone()).unwrap();
            self.release.recv_timeout(WAIT).unwrap();
            return native::run_cpu_readouts(prepared, sink, &self.profile.tokenizer);
        }
        native::run_tokens(
            prepared,
            sink,
            &[],
            |_, position, _| {
                if position == 0 {
                    self.entered.send(sink.server.clone()).unwrap();
                    self.release
                        .recv_timeout(WAIT)
                        .context("release fixture forward")?;
                }
                let mut logits = vec![0.0; 261];
                logits[120] = 10.0;
                Ok(logits)
            },
            |_| Ok(b"x".to_vec()),
        )
    }
    fn generate(
        &mut self,
        _: &ServeRequest,
        _: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        assert_eq!(self.owner, std::thread::current().id());
        assert!(sink.transport_reserve_bytes() >= CPU_RESERVE_BYTES);
        self.ordinary.fetch_add(1, Ordering::AcqRel);
        sink.piece(b"ordinary").map_err(BackendFailure::Aborted)?;
        Ok(GenerationOutcome {
            end: crate::serve::output_partition::GenerationEnd::TokenLimit,
            usage: crate::serve::events::Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
            },
            stats: None,
        })
    }
    fn request_finished(&mut self) {
        assert_eq!(self.owner, std::thread::current().id());
        self.callbacks.lock().unwrap().push("finished");
    }
    fn shutdown(&mut self) {
        assert_eq!(self.owner, std::thread::current().id());
        self.callbacks.lock().unwrap().push("shutdown");
    }
}

struct Server {
    address: std::net::SocketAddr,
    stop: Arc<AtomicBool>,
    release: Sender<()>,
    entered: Receiver<ExecutionGate>,
    calls: Arc<AtomicUsize>,
    ordinary: Arc<AtomicUsize>,
    callbacks: Arc<Mutex<Vec<&'static str>>>,
    thread: Option<JoinHandle<Result<()>>>,
}
impl Server {
    fn start(fixture: &native::CpuFixture) -> Self {
        Self::start_with_assets(fixture, None)
    }
    fn start_with_assets(
        fixture: &native::CpuFixture,
        assets: Option<Arc<crate::serve::assets::WebAssets>>,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let (entry_sender, entered) = channel();
        let (release, released) = channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let ordinary = Arc::new(AtomicUsize::new(0));
        let callbacks = Arc::new(Mutex::new(Vec::new()));
        let mut backend = Backend {
            profile: Arc::clone(&fixture.profile),
            owner: std::thread::current().id(),
            entered: entry_sender,
            release: released,
            calls: Arc::clone(&calls),
            ordinary: Arc::clone(&ordinary),
            callbacks: Arc::clone(&callbacks),
        };
        let store = Arc::clone(&fixture.store);
        let thread = std::thread::spawn(move || {
            backend.owner = std::thread::current().id();
            crate::serve::accept_loop_with_workbench(
                listener,
                "test",
                0.0,
                &mut backend,
                &mut None,
                Some(crate::serve::Workbench { store, assets }),
                |_| {
                    anyhow::ensure!(!stopping.load(Ordering::Acquire), "test shutdown");
                    Ok(())
                },
            )
        });
        Self {
            address,
            stop,
            release,
            entered,
            calls,
            ordinary,
            callbacks,
            thread: Some(thread),
        }
    }
    fn send(&self, method: &str, path: &str, body: &str) -> TcpStream {
        let mut stream = TcpStream::connect(self.address).unwrap();
        stream.set_read_timeout(Some(WAIT)).unwrap();
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nhost: localhost\r\ncontent-length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        stream
    }
    fn request(&self, method: &str, path: &str, body: &str) -> (String, serde_json::Value) {
        let mut response = String::new();
        self.send(method, path, body)
            .read_to_string(&mut response)
            .unwrap();
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        (head.into(), serde_json::from_str(body).unwrap())
    }
    fn wait_terminal(&self, id: &str) -> serde_json::Value {
        let start = std::time::Instant::now();
        loop {
            let (head, value) = self.request("GET", &format!("/v1/lens/jobs/{id}"), "");
            if head.starts_with("HTTP/1.1 200")
                && matches!(
                    value["state"].as_str(),
                    Some("completed" | "cancelled" | "failed" | "interrupted")
                )
            {
                return value;
            }
            assert!(start.elapsed() < WAIT, "job did not settle: {head} {value}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = self.release.send(());
        if let Some(thread) = self.thread.take() {
            assert!(
                thread
                    .join()
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("test shutdown")
            );
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = self.release.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[test]
fn qualified_plain_http_submission_reaches_the_shared_owner_and_saved_results() {
    let mut fixture = native::CpuFixture::new();
    Arc::get_mut(&mut fixture.profile).unwrap().plain_readouts = true;
    let mut server = Server::start(&fixture);
    let (_, caps) = server.request("GET", "/v1/lens/capabilities", "");
    assert_eq!(caps["execution"]["baseline_only"], false);
    let mut request = fixture.request("http-plain");
    request["preconditions"]["asset_identities"] = serde_json::json!({"plain":"cpu-fixture"});
    request["diagnostics"] = serde_json::json!({"directions":[],"operations":[],"readouts":[
        {"id":"plain-read","lens":"plain","mode":"full_vocabulary","top_k":2,"scope":{"layers":{"kind":"all"},"decode":{"kind":"all"}}}
    ]});
    let body = request.to_string();
    let (head, accepted) = server.request("POST", "/v1/lens/jobs", &body);
    assert!(head.starts_with("HTTP/1.1 202"), "{head} {accepted}");
    let id = accepted["id"].as_str().unwrap();
    server.entered.recv_timeout(WAIT).unwrap();
    server.release.send(()).unwrap();
    let status = server.wait_terminal(id);
    assert_eq!(status["state"], "completed");
    assert_eq!(status["observations"]["committed_records"], 4);
    let (_, page) = server.request("GET", &format!("/v1/lens/jobs/{id}/result"), "");
    let rows = page["records"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["kind"] == "readout")
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().all(|r| r["readout_id"] == "plain-read"
        && r["phase"] == "decode"
        && r["scores"][0]["token_id"] == 120));
    assert_eq!(server.request("POST", "/v1/lens/jobs", &body).1["id"], id);
    assert_eq!(
        server
            .request("GET", &format!("/v1/lens/jobs/{id}/result"), "")
            .1,
        page
    );
    assert_eq!(server.calls.load(Ordering::Acquire), 1);
    server.stop();
}

#[test]
fn paired_http_history_and_exact_retry_survive_loss_of_capture_capability() {
    let mut fixture = native::CpuFixture::new();
    Arc::get_mut(&mut fixture.profile).unwrap().plain_readouts = true;
    let mut server = Server::start(&fixture);
    assert_eq!(
        server.request("GET", "/v1/lens/capabilities", "").1["residual_pair_capture"],
        true
    );
    let mut request = fixture.request("http-pairs");
    request["diagnostics"] = serde_json::json!({"directions":[],"operations":[],"readouts":[],"residual_pairs":[
        {"id":"pair","scope":{"layers":{"kind":"all"},"decode":{"kind":"all"}}}
    ]});
    let body = request.to_string();
    let (head, accepted) = server.request("POST", "/v1/lens/jobs", &body);
    assert!(head.starts_with("HTTP/1.1 202"), "{head} {accepted}");
    let id = accepted["id"].as_str().unwrap();
    server.entered.recv_timeout(WAIT).unwrap();
    server.release.send(()).unwrap();
    assert_eq!(
        server.wait_terminal(id)["observations"]["committed_records"],
        4
    );
    let result = format!("/v1/lens/jobs/{id}/result");
    let page = server.request("GET", &result, "").1;
    assert_eq!(
        page["records"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["kind"] == "residual_pair")
            .count(),
        4
    );
    server.stop();
    Arc::get_mut(&mut fixture.profile).unwrap().plain_readouts = false;
    let mut server = Server::start(&fixture);
    assert_eq!(
        server.request("GET", "/v1/lens/capabilities", "").1["residual_pair_capture"],
        false
    );
    let (head, recovered) = server.request("POST", "/v1/lens/jobs", &body);
    assert!(head.starts_with("HTTP/1.1 200"), "{head} {recovered}");
    assert_eq!(recovered["id"], id);
    assert_eq!(server.request("GET", &result, "").1, page);
    request["idempotency_key"] = serde_json::json!("new-pairs");
    assert!(
        server
            .request("POST", "/v1/lens/jobs", &request.to_string())
            .0
            .starts_with("HTTP/1.1 400")
    );
    assert_eq!(server.calls.load(Ordering::Acquire), 0);
    server.stop();
}

#[test]
fn fitted_http_pins_registry_identity_and_reopens_original_shared_rows() {
    let mut fixture = native::CpuFixture::new();
    let (_files, registry) = native::registry::tests::fitted_fixture();
    let profile = Arc::get_mut(&mut fixture.profile).unwrap();
    profile.plain_readouts = true;
    profile.layers = 3;
    profile.registry = Some(registry);
    let mut server = Server::start(&fixture);
    let (_, catalog) = server.request("GET", "/v1/lens/assets", "");
    let fit = catalog["assets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["alias"] == "fit")
        .unwrap();
    assert_eq!(fit["direction_rows"], serde_json::json!(["token_id"]));
    let mut request = fixture.request("http-fit");
    request["preconditions"]["asset_identities"] =
        serde_json::json!({"plain":"cpu-fixture","fit":"stale"});
    let row = |id, lens| {
        serde_json::json!({"id":id,"lens":lens,"mode":"full_vocabulary","top_k":2,
        "scope":{"layers":{"kind":"values","values":[0]},"decode":{"kind":"all"}}})
    };
    request["diagnostics"] = serde_json::json!({"directions":[],"operations":[],"readouts":[row("plain","plain"),row("fit-a","fit"),row("fit-b","fit")]});
    assert!(
        server
            .request("POST", "/v1/lens/jobs", &request.to_string())
            .0
            .starts_with("HTTP/1.1 412")
    );
    assert_eq!(server.calls.load(Ordering::Acquire), 0);
    request["preconditions"]["asset_identities"]["fit"] = fit["identity"].clone();
    let (head, accepted) = server.request("POST", "/v1/lens/jobs", &request.to_string());
    assert!(head.starts_with("HTTP/1.1 202"), "{head} {accepted}");
    let id = accepted["id"].as_str().unwrap();
    server.entered.recv_timeout(WAIT).unwrap();
    server.release.send(()).unwrap();
    assert_eq!(server.wait_terminal(id)["state"], "completed");
    let path = format!("/v1/lens/jobs/{id}/result");
    let (_, page) = server.request("GET", &path, "");
    let rows = page["records"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["kind"] == "readout")
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 6);
    assert_eq!(
        page["records"][0]["asset_identities"]["fit"],
        fit["identity"]
    );
    assert_eq!(
        page["records"][0]["readout_admission"]["head_evaluations_upper"],
        4
    );
    for row in rows.iter().filter(|r| r["lens"] == "fit") {
        assert_eq!(row["asset_identity"], fit["identity"]);
        assert_eq!(row["target_layer"], 1);
        assert_eq!(
            row["binding_status"],
            "source_deployment_equivalence_unverified"
        );
        assert!(row["generation_logit_witness"].is_null());
        assert_eq!(
            row["cost"]["readout_ms"].is_null(),
            row["readout_id"] == "fit-b"
        );
    }
    assert_eq!(
        server
            .request("POST", "/v1/lens/jobs", &request.to_string())
            .1["id"],
        id
    );
    assert_eq!(server.request("GET", &path, "").1, page);
    assert_eq!(server.calls.load(Ordering::Acquire), 1);
    server.stop();
}

#[test]
fn retained_archive_overflow_is_refused_before_durable_acceptance() {
    let tokens = (0..4096)
        .map(|i| format!("retained-extra-{i}"))
        .collect::<Vec<_>>();
    let mut fixture = native::CpuFixture::with_extra_tokens(&tokens);
    Arc::get_mut(&mut fixture.profile).unwrap().plain_readouts = true;
    let mut server = Server::start(&fixture);
    let mut request = fixture.request("archive-overflow");
    request["input"]["messages"][0]["content"] = serde_json::json!("a".repeat(1200));
    request["preconditions"]["asset_identities"] = serde_json::json!({"plain":"cpu-fixture"});
    request["diagnostics"] = serde_json::json!({"directions":[],"operations":[],"readouts":[
        {"id":"all","lens":"plain","mode":"full_vocabulary","top_k":1,"retain":"scores_and_residual",
            "scope":{"layers":{"kind":"all"},"prefill":{"kind":"all"}}}]});
    let (head, reply) = server.request("POST", "/v1/lens/jobs", &request.to_string());
    assert!(head.starts_with("HTTP/1.1 400"), "{head} {reply}");
    assert!(
        reply["error"]["message"]
            .as_str()
            .unwrap()
            .contains("retained archive"),
        "{reply}"
    );
    assert_eq!(
        server.request("GET", "/v1/lens/jobs", "").1["jobs"],
        serde_json::json!([])
    );
    assert_eq!(server.calls.load(Ordering::Acquire), 0);
    server.stop();
}

#[test]
fn accepted_job_outlives_socket_and_history_retry_cancel_share_the_resident_owner() {
    let fixture = native::CpuFixture::new();
    let mut server = Server::start(&fixture);
    let request = fixture.request("lost-ack").to_string();
    drop(server.send("POST", "/v1/lens/jobs", &request));
    server.entered.recv_timeout(WAIT).unwrap();
    let (head, history) = server.request("GET", "/v1/lens/jobs", "");
    assert!(head.starts_with("HTTP/1.1 200"), "{head} {history}");
    let id = history["jobs"][0]["id"].as_str().unwrap();
    let (head, recovered) = server.request("POST", "/v1/lens/jobs", &request);
    assert!(head.starts_with("HTTP/1.1 200"), "{head} {recovered}");
    assert_eq!(recovered["id"], id);
    let (_, first) = server.request("GET", &format!("/v1/lens/jobs/{id}/result"), "");
    let (_, second) = server.request("GET", &format!("/v1/lens/jobs/{id}/result"), "");
    assert_eq!(first, second);
    assert_eq!(first["records"][0]["kind"], "prepared_input");
    assert_eq!(server.calls.load(Ordering::Acquire), 1);
    let (head, busy) = server.request(
        "POST",
        "/v1/lens/jobs",
        &fixture.request("new-job").to_string(),
    );
    assert!(head.starts_with("HTTP/1.1 429"), "{head} {busy}");
    assert_eq!(busy["admission"]["state"], "not_accepted");
    let mut ordinary = TcpStream::connect(server.address).unwrap();
    ordinary.set_read_timeout(Some(WAIT)).unwrap();
    ordinary
        .write_all(b"POST /v1/responses HTTP/1.1\r\nhost: localhost\r\ncontent-length: 100\r\n\r\n")
        .unwrap();
    let mut response = String::new();
    ordinary.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 503"));
    let (head, cancelled) = server.request("POST", &format!("/v1/lens/jobs/{id}/cancel"), "{}");
    assert!(head.starts_with("HTTP/1.1 200"), "{head} {cancelled}");
    assert_eq!(cancelled["cancel_requested"], true);
    server.release.send(()).unwrap();
    let terminal = server.wait_terminal(id);
    assert_eq!(terminal["state"], "cancelled");
    assert_eq!(terminal["generation"]["sampled_tokens"], 0);
    let (head, response) = server.request(
        "POST",
        "/v1/responses",
        r#"{"model":"test","input":"hi","max_output_tokens":1}"#,
    );
    assert!(head.starts_with("HTTP/1.1 200"), "{head} {response}");
    assert_eq!(server.ordinary.load(Ordering::Acquire), 1);
    assert_eq!(server.calls.load(Ordering::Acquire), 1);
    server.stop();
    let events = server.callbacks.lock().unwrap();
    assert_eq!(events.last(), Some(&"shutdown"));
    assert_eq!(
        events.iter().filter(|event| **event == "shutdown").count(),
        1
    );
    assert!(events.iter().filter(|event| **event == "finished").count() >= 2);
    drop(TcpListener::bind(server.address).unwrap());
}

#[test]
fn local_control_stop_interrupts_active_native_and_joins_before_backend_shutdown() {
    let fixture = native::CpuFixture::new();
    let mut server = Server::start(&fixture);
    let (_, accepted) = server.request(
        "POST",
        "/v1/lens/jobs",
        &fixture.request("shutdown").to_string(),
    );
    let id = accepted["id"].as_str().unwrap();
    let gate = server.entered.recv_timeout(WAIT).unwrap();
    let mut unfinished = TcpStream::connect(server.address).unwrap();
    unfinished.set_read_timeout(Some(WAIT)).unwrap();
    unfinished.write_all(b"GET /v1/lens").unwrap();
    gate.close();
    server.release.send(()).unwrap();
    let cause = server.thread.take().unwrap().join().unwrap().unwrap_err();
    assert!(cause.is::<ServerStopped>(), "{cause:#}");
    let status = fixture.store.status(id).unwrap();
    assert_eq!(
        status.state,
        crate::serve::jobs::state::JobState::Interrupted
    );
    assert_eq!(status.generation.counters.consumed_prompt_tokens, 1);
    assert_eq!(status.generation.counters.sampled_tokens, 0);
    assert!(status.result.complete);
    assert!(!status.cancel_requested);
    assert_eq!(server.calls.load(Ordering::Acquire), 1);
    let events = server.callbacks.lock().unwrap();
    assert_eq!(events.last(), Some(&"shutdown"));
    assert_eq!(
        events.iter().filter(|event| **event == "shutdown").count(),
        1
    );
    assert!(events.iter().filter(|event| **event == "finished").count() >= 2);
    let mut response = Vec::new();
    if let Err(cause) = unfinished.read_to_end(&mut response) {
        assert!(matches!(
            cause.kind(),
            io::ErrorKind::ConnectionReset | io::ErrorKind::ConnectionAborted
        ));
    }
    drop(TcpListener::bind(server.address).unwrap());
}

#[test]
fn disconnected_startup_probe_does_not_close_the_control_service() {
    use std::os::fd::AsRawFd;
    let fixture = native::CpuFixture::new();
    let mut activity = owner_activity::OwnerActivity::default();
    let (sender, _receiver) = std::sync::mpsc::sync_channel(1);
    let gate = ExecutionGate::default();
    let stopping = Arc::new(AtomicBool::new(false));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let probe = TcpStream::connect(address).unwrap();
    let linger = libc::linger {
        l_onoff: 1,
        l_linger: 0,
    };
    // Own this probe descriptor and reset only its connection before acceptance.
    assert_eq!(
        unsafe {
            libc::setsockopt(
                probe.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_LINGER,
                (&linger as *const libc::linger).cast(),
                std::mem::size_of_val(&linger) as libc::socklen_t,
            )
        },
        0
    );
    drop(probe);
    let acceptor = spawn(
        listener,
        Profile {
            assets: None,
            model_id: "test".into(),
            request: RequestProfile::UnboundQwen,
            lens: Arc::new(LensApi::new(
                "test".into(),
                Some(Arc::clone(&fixture.store)),
                None,
            )),
            gate: gate.clone(),
            activity: activity.admission(),
            sender,
            trace: None,
        },
        Arc::clone(&stopping),
    )
    .unwrap();
    let mut client = TcpStream::connect(address).unwrap();
    client.set_read_timeout(Some(WAIT)).unwrap();
    let mut response = String::new();
    let read = client
        .write_all(b"GET /v1/lens/capabilities HTTP/1.1\r\nhost: localhost\r\n\r\n")
        .and_then(|()| client.read_to_string(&mut response));
    stopping.store(true, Ordering::Release);
    acceptor.join().unwrap().unwrap();
    read.unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    activity.drain_finished(|| {});
    assert!(activity.is_settled());
    drop(TcpListener::bind(address).unwrap());
}

#[test]
fn history_releases_activity_before_waiting_for_store_access() {
    let fixture = native::CpuFixture::new();
    let mut activity = owner_activity::OwnerActivity::default();
    let guard = activity.admission().try_admit().unwrap();
    let (sender, _receiver) = std::sync::mpsc::sync_channel(1);
    let profile = Profile {
        assets: None,
        model_id: "test".into(),
        request: RequestProfile::UnboundQwen,
        lens: Arc::new(LensApi::new(
            "test".into(),
            Some(Arc::clone(&fixture.store)),
            None,
        )),
        gate: ExecutionGate::default(),
        activity: activity.admission(),
        sender,
        trace: None,
    };
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    client.set_read_timeout(Some(WAIT)).unwrap();
    client
        .write_all(b"GET /v1/lens/jobs HTTP/1.1\r\nhost: localhost\r\n\r\n")
        .unwrap();
    let (stream, _) = listener.accept().unwrap();
    let mut worker = None;
    fixture.store.with_history_locked(|| {
        worker = Some(std::thread::spawn(move || {
            handle(stream, &profile, Some(guard))
        }));
        let deadline = std::time::Instant::now() + WAIT;
        while !activity.is_settled() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(activity.is_settled());
        assert!(!worker.as_ref().unwrap().is_finished());
        let mut idle = false;
        activity.idle_if_quiet(|| idle = true);
        assert!(idle);
    });
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    worker.unwrap().join().unwrap().unwrap();
    assert!(response.starts_with("HTTP/1.1 200"));
    activity.drain_finished(|| panic!("read-only history must not debounce idle publication"));
}

#[test]
fn static_assets_share_the_busy_owner_port_without_reserving_execution() {
    let fixture = native::CpuFixture::new();
    std::fs::write(fixture.root.join("index.html"), b"client").unwrap();
    std::fs::write(fixture.root.join("asset-manifest.json"), br#"{"version":1,"entry":"index.html","files":[{"path":"index.html","contentType":"text/html","bytes":6}]}"#).unwrap();
    let assets = Arc::new(crate::serve::assets::WebAssets::open(&fixture.root).unwrap());
    let mut server = Server::start_with_assets(&fixture, Some(assets));
    let (_, accepted) = server.request(
        "POST",
        "/v1/lens/jobs",
        &fixture.request("static-busy").to_string(),
    );
    let id = accepted["id"].as_str().unwrap();
    server.entered.recv_timeout(WAIT).unwrap();
    for method in ["GET", "HEAD"] {
        let mut response = String::new();
        server
            .send(method, "/?history=ignored", "")
            .read_to_string(&mut response)
            .unwrap();
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("HTTP/1.1 200"), "{head}");
        assert!(head.contains("content-length: 6"));
        assert_eq!(body, if method == "GET" { "client" } else { "" });
    }
    let (head, caps) = server.request("GET", "/v1/lens/capabilities", "");
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert_eq!(caps["execution"]["baseline_only"], true);
    let mut invalid = TcpStream::connect(server.address).unwrap();
    invalid.set_read_timeout(Some(WAIT)).unwrap();
    invalid
        .write_all(b"GET / HTTP/1.1\r\nhost: localhost\r\ncontent-length: 100\r\n\r\n")
        .unwrap();
    let mut response = String::new();
    invalid.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 400"));
    server.request("POST", &format!("/v1/lens/jobs/{id}/cancel"), "{}");
    server.release.send(()).unwrap();
    server.wait_terminal(id);
    for path in ["/v1", "/v1/unknown", "/missing.js"] {
        let (head, error) = server.request("GET", path, "");
        assert!(head.starts_with("HTTP/1.1 404"), "{head} {error}");
    }
    assert_eq!(server.ordinary.load(Ordering::Acquire), 0);
    server.stop();
}

#[test]
fn blocked_static_write_releases_activity_and_shutdown_joins_its_worker() {
    use std::os::fd::AsRawFd;
    let fixture = native::CpuFixture::new();
    let length = 4 * 1024 * 1024;
    std::fs::write(fixture.root.join("index.html"), vec![b'x'; length]).unwrap();
    std::fs::write(fixture.root.join("asset-manifest.json"), serde_json::to_vec(&serde_json::json!({
        "version":1,"entry":"index.html","files":[{"path":"index.html","contentType":"text/html","bytes":length}]
    })).unwrap()).unwrap();
    let assets = Arc::new(crate::serve::assets::WebAssets::open(&fixture.root).unwrap());
    let activity = owner_activity::OwnerActivity::default();
    let (sender, _receiver) = std::sync::mpsc::sync_channel(1);
    let gate = ExecutionGate::default();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    client.set_read_timeout(Some(WAIT)).unwrap();
    client
        .write_all(b"GET / HTTP/1.1\r\nhost: localhost\r\n\r\n")
        .unwrap();
    let (stream, _) = listener.accept().unwrap();
    for (socket, option) in [(&client, libc::SO_RCVBUF), (&stream, libc::SO_SNDBUF)] {
        let bytes: libc::c_int = 4096;
        // These owned sockets must backpressure well before the 4 MiB payload.
        assert_eq!(
            unsafe {
                libc::setsockopt(
                    socket.as_raw_fd(),
                    libc::SOL_SOCKET,
                    option,
                    (&bytes as *const libc::c_int).cast(),
                    std::mem::size_of_val(&bytes) as libc::socklen_t,
                )
            },
            0
        );
    }
    let socket = stream.try_clone().unwrap();
    let guard = activity.admission().try_admit().unwrap();
    let profile = Profile {
        model_id: "test".into(),
        request: RequestProfile::UnboundQwen,
        lens: Arc::new(LensApi::new(
            "test".into(),
            Some(Arc::clone(&fixture.store)),
            None,
        )),
        assets: Some(assets),
        gate: gate.clone(),
        activity: activity.admission(),
        sender,
        trace: None,
    };
    let mut worker = Worker {
        socket,
        thread: Some(std::thread::spawn(move || {
            handle(stream, &profile, Some(guard))
        })),
    };
    let mut prefix = [0u8; 16];
    client.read_exact(&mut prefix).unwrap();
    assert_eq!(&prefix[..12], b"HTTP/1.1 200");
    assert!(activity.is_settled());
    assert!(gate.reserve().is_ok());
    assert!(!worker.thread.as_ref().unwrap().is_finished());
    worker.stop();
    worker.join().unwrap();
    assert!(activity.is_settled());
}

#[test]
fn releasing_an_execution_slot_does_not_reopen_closed_admission() {
    let gate = ExecutionGate::default();
    let permit = gate.reserve().unwrap();
    assert!(gate.reserve().is_err());
    drop(permit);
    let permit = gate.reserve().unwrap();
    gate.close();
    assert!(gate.checkpoint().is_err());
    drop(permit);
    assert!(gate.reserve().is_err());
}

#[test]
fn failed_delivery_drops_connection_and_execution_permit_outside_gate_lock() {
    for closed in [false, true] {
        let gate = ExecutionGate::default();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (stream, _) = listener.accept().unwrap();
        let mut activity = owner_activity::OwnerActivity::default();
        let guard = activity.admission().try_admit().unwrap();
        let connection = transport::Connection::prepared(
            stream,
            "test".into(),
            RequestProfile::UnboundQwen,
            None,
            guard,
            http::HttpRequest {
                method: "POST".into(),
                path: "/v1/responses".into(),
                host: Some("localhost".into()),
                origin: None,
                body: br#"{"model":"test","input":"hi"}"#.to_vec(),
            },
            gate.reserve().unwrap(),
        )
        .unwrap();
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        if closed {
            gate.close();
        } else {
            drop(receiver);
        }
        assert!(gate.deliver(&sender, Event::Prepared(connection)).is_err());
        assert_eq!(gate.reserve().is_ok(), !closed);
        let mut finished = 0;
        activity.drain_finished(|| finished += 1);
        assert_eq!(finished, 1);
        assert!(activity.is_settled());
    }
}
