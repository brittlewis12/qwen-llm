use super::*;
use crate::serve::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use crate::serve::items::ServeRequest;
use std::io::Read;
use std::sync::{
    atomic::AtomicUsize,
    mpsc::{Receiver, Sender, channel},
};

const WAIT: Duration = Duration::from_secs(5);

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
            crate::serve::accept_loop_with_history(
                listener,
                "test",
                0.0,
                &mut backend,
                &mut None,
                Some(store),
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
