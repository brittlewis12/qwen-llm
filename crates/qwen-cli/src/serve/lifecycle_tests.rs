use super::*;
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::ThreadId;

const WAIT: Duration = Duration::from_secs(2);

#[derive(Debug, PartialEq)]
enum Event {
    Generate,
    Finished,
    Idle,
    Shutdown,
}

struct LifecycleBackend {
    events: Sender<Event>,
    owner: ThreadId,
    address: SocketAddr,
    fail_render: bool,
    generation_gate: Option<Receiver<()>>,
    idle_stop: Option<Arc<AtomicBool>>,
}

impl LifecycleBackend {
    fn record(&self, event: Event) {
        assert_eq!(std::thread::current().id(), self.owner);
        self.events.send(event).unwrap();
    }
}

impl http::GenerationBackend for LifecycleBackend {
    fn model_id(&self) -> &str {
        "test"
    }

    fn request_profile(&self) -> request_profile::RequestProfile {
        if self.fail_render {
            return request_profile::RequestProfile::OrdinaryQwen {
                template: items::QwenTemplate::Qwen35,
                no_thinking_supported: false,
                style: items::TemplateStyle::House,
            };
        }
        request_profile::RequestProfile::UnboundQwen
    }

    fn generate(
        &mut self,
        _request: &items::ServeRequest,
        _prompt: &str,
        sink: &mut dyn http::GenerationSink,
    ) -> Result<http::GenerationOutcome, http::BackendFailure> {
        self.record(Event::Generate);
        if let Some(gate) = self.generation_gate.take() {
            gate.recv_timeout(WAIT).unwrap();
        }
        sink.piece(b"answer")
            .map_err(http::BackendFailure::Aborted)?;
        Ok(http::GenerationOutcome {
            end: output_partition::GenerationEnd::TokenLimit,
            usage: events::Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
            },
            stats: None,
        })
    }

    fn idle(&mut self) {
        self.record(Event::Idle);
        if let Some(stop) = &self.idle_stop {
            stop.store(true, Ordering::Release);
        }
    }

    fn request_finished(&mut self) {
        self.record(Event::Finished);
    }

    fn shutdown(&mut self) {
        assert!(TcpStream::connect_timeout(&self.address, WAIT).is_err());
        self.record(Event::Shutdown);
    }
}

fn connect(address: SocketAddr) -> TcpStream {
    let stream = TcpStream::connect_timeout(&address, WAIT).unwrap();
    stream.set_read_timeout(Some(WAIT)).unwrap();
    stream.set_write_timeout(Some(WAIT)).unwrap();
    stream
}

fn post() -> String {
    let body =
        r#"{"model":"test","input":"hello","max_output_tokens":1,"x_qwen":{"no_thinking":true}}"#;
    format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

fn stop_after_handling() -> impl FnMut(OwnerCheckpoint) -> Result<()> {
    let mut handling = false;
    move |checkpoint| {
        ensure!(
            !(handling && checkpoint == OwnerCheckpoint::BeforeAdmission),
            "test stop after handling"
        );
        handling |= checkpoint == OwnerCheckpoint::BeforeHandling;
        Ok(())
    }
}

fn first_non_idle(observed: &Receiver<Event>) -> Event {
    loop {
        let event = observed.recv_timeout(WAIT).unwrap();
        if event != Event::Idle {
            return event;
        }
    }
}

#[test]
fn handled_connections_finish_once_before_listener_closed_shutdown() {
    let cases = [
        (post(), false, "200", true),
        (
            "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
            false,
            "200",
            false,
        ),
        (
            "GET /missing HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
            false,
            "404",
            false,
        ),
        ("not an HTTP request\r\n\r\n".into(), false, "400", false),
        (post(), true, "400", false),
        (String::new(), false, "", false),
    ];
    for (request, fail_render, status, generates) in cases {
        let listener = bind_loopback("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (events, observed) = channel();
        let mut client = connect(address);
        if request.is_empty() {
            client.shutdown(Shutdown::Both).unwrap();
        } else {
            client.write_all(request.as_bytes()).unwrap();
        }
        let server = std::thread::spawn(move || {
            let mut backend = LifecycleBackend {
                events,
                owner: std::thread::current().id(),
                address,
                fail_render,
                generation_gate: None,
                idle_stop: None,
            };
            let result = accept_loop_with_checkpoint(
                listener,
                "test",
                0.0,
                &mut backend,
                &mut None,
                stop_after_handling(),
            );
            assert!(result.unwrap_err().to_string().contains("test stop"));
        });
        if !request.is_empty() {
            let mut response = String::new();
            client.read_to_string(&mut response).unwrap();
            assert!(
                response.starts_with(&format!("HTTP/1.1 {status}")),
                "{response}"
            );
        }
        let mut expected = Vec::new();
        if generates {
            expected.push(Event::Generate);
        }
        expected.extend([Event::Finished, Event::Shutdown]);
        for (index, event) in expected.into_iter().enumerate() {
            let actual = if index == 0 {
                first_non_idle(&observed)
            } else {
                observed.recv_timeout(WAIT).unwrap()
            };
            assert_eq!(actual, event);
        }
        server.join().unwrap();
        assert!(observed.try_recv().is_err());
    }
}

#[test]
fn shutdown_at_second_checkpoint_does_not_finish_unhandled_connection() {
    let listener = bind_loopback("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let _client = connect(address);
    let (events, observed) = channel();
    let mut backend = LifecycleBackend {
        events,
        owner: std::thread::current().id(),
        address,
        fail_render: false,
        generation_gate: None,
        idle_stop: None,
    };
    let mut handling_checkpoint = false;
    let result = accept_loop_with_checkpoint(
        listener,
        "test",
        0.0,
        &mut backend,
        &mut None,
        |checkpoint| {
            handling_checkpoint = checkpoint == OwnerCheckpoint::BeforeHandling;
            ensure!(!handling_checkpoint, "shutdown before handling");
            Ok(())
        },
    );
    assert!(result.is_err());
    assert!(handling_checkpoint);
    assert_eq!(first_non_idle(&observed), Event::Shutdown);
    assert!(observed.try_recv().is_err());
}

#[test]
fn idle_callback_and_shutdown_run_on_owner_without_a_wake_connection() {
    let listener = bind_loopback("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (events, observed) = channel();
    let stop = Arc::new(AtomicBool::new(false));
    let mut backend = LifecycleBackend {
        events,
        owner: std::thread::current().id(),
        address,
        fail_render: false,
        generation_gate: None,
        idle_stop: Some(stop.clone()),
    };
    assert!(
        accept_loop_with_checkpoint(listener, "test", 0.0, &mut backend, &mut None, |_| {
            ensure!(!stop.load(Ordering::Acquire), "stop after idle");
            Ok(())
        })
        .is_err()
    );
    assert_eq!(
        observed.try_iter().collect::<Vec<_>>(),
        [Event::Idle, Event::Shutdown]
    );
}

#[test]
fn busy_rejection_does_not_create_an_owner_completion() {
    let listener = bind_loopback("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (events, observed) = channel();
    let (release, gate) = channel();
    let mut first = connect(address);
    first.write_all(post().as_bytes()).unwrap();
    let server = std::thread::spawn(move || {
        let mut backend = LifecycleBackend {
            events,
            owner: std::thread::current().id(),
            address,
            fail_render: false,
            generation_gate: Some(gate),
            idle_stop: None,
        };
        assert!(
            accept_loop_with_checkpoint(
                listener,
                "test",
                0.0,
                &mut backend,
                &mut None,
                stop_after_handling()
            )
            .is_err()
        );
    });
    assert_eq!(first_non_idle(&observed), Event::Generate);
    let mut busy = connect(address);
    let mut response = String::new();
    busy.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    assert!(observed.try_recv().is_err());
    release.send(()).unwrap();
    let mut response = String::new();
    first.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert_eq!(observed.recv_timeout(WAIT).unwrap(), Event::Finished);
    assert_eq!(observed.recv_timeout(WAIT).unwrap(), Event::Shutdown);
    server.join().unwrap();
    assert!(observed.try_recv().is_err());
}

struct PhaseBackend {
    owner: ThreadId,
    address: SocketAddr,
    events: Sender<&'static str>,
    payload: Vec<u8>,
}

impl PhaseBackend {
    fn record(&self, event: &'static str) {
        assert_eq!(std::thread::current().id(), self.owner);
        self.events.send(event).unwrap();
    }
}

impl http::GenerationBackend for PhaseBackend {
    fn model_id(&self) -> &str {
        "test"
    }
    fn generate(
        &mut self,
        _: &items::ServeRequest,
        _: &str,
        sink: &mut dyn http::GenerationSink,
    ) -> Result<http::GenerationOutcome, http::BackendFailure> {
        self.record("generate");
        let result = sink.piece(&self.payload);
        self.record(if result.is_ok() {
            "generated"
        } else {
            "aborted"
        });
        result.map_err(http::BackendFailure::Aborted)?;
        Ok(http::GenerationOutcome {
            end: output_partition::GenerationEnd::TokenLimit,
            usage: events::Usage {
                input_tokens: 1,
                output_tokens: 1,
                cached_tokens: 0,
            },
            stats: None,
        })
    }
    fn request_finished(&mut self) {
        self.record("finished");
    }
    fn shutdown(&mut self) {
        assert!(TcpStream::connect_timeout(&self.address, WAIT).is_err());
        self.record("shutdown");
    }
}

fn small_socket_buffer(socket: &impl std::os::fd::AsRawFd, option: i32) {
    let bytes: libc::c_int = 4096;
    // This test owns the socket and passes a correctly sized integer option.
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

#[test]
fn busy_and_shutdown_cover_reading_generation_backpressure_and_response_writing() {
    for phase in ["reading", "generation", "writing"] {
        let listener = bind_loopback("127.0.0.1:0").unwrap();
        small_socket_buffer(&listener, libc::SO_SNDBUF);
        let address = listener.local_addr().unwrap();
        let mut client = connect(address);
        small_socket_buffer(&client, libc::SO_RCVBUF);
        if phase == "reading" {
            client
                .write_all(b"POST /v1/responses HTTP/1.1\r\n")
                .unwrap();
        } else {
            let body = format!(
                r#"{{"model":"test","input":"hi","stream":{}}}"#,
                phase == "generation"
            );
            client.write_all(format!("POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
        }
        let stopping = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&stopping);
        let (entered, entry) = channel();
        let (events, observed) = channel();
        let (done, completion) = channel();
        let server = std::thread::spawn(move || {
            let mut backend = PhaseBackend {
                owner: std::thread::current().id(),
                address,
                events,
                payload: vec![b'x'; 2 * 1024 * 1024],
            };
            let mut entered = Some(entered);
            let result = accept_loop_with_checkpoint(
                listener,
                "test",
                0.0,
                &mut backend,
                &mut None,
                |point| {
                    if point == OwnerCheckpoint::DuringHandling
                        && let Some(entered) = entered.take()
                    {
                        entered.send(()).unwrap();
                    }
                    ensure!(!stop.load(Ordering::Acquire), "test stop during {phase}");
                    Ok(())
                },
            );
            done.send(result).unwrap();
        });
        entry.recv_timeout(WAIT).unwrap();
        if phase != "reading" {
            assert_eq!(observed.recv_timeout(WAIT).unwrap(), "generate");
            if phase == "writing" {
                assert_eq!(observed.recv_timeout(WAIT).unwrap(), "generated");
            }
            // Seeing a response byte establishes that writing has begun; the
            // deliberately small socket buffers cannot hold the full response.
            client.read_exact(&mut [0u8; 1]).unwrap();
        }
        let mut busy = connect(address);
        let mut response = String::new();
        busy.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 503"), "{phase}: {response}");
        assert!(
            observed.try_recv().is_err(),
            "request finished while {phase} was blocked"
        );
        stopping.store(true, Ordering::Release);
        if phase == "generation" {
            // Reset only this owned test connection to wake a blocked streaming
            // subscriber; a request-side FIN alone is deliberately inconclusive.
            use std::os::fd::AsRawFd;
            let linger = libc::linger {
                l_onoff: 1,
                l_linger: 0,
            };
            assert_eq!(
                unsafe {
                    libc::setsockopt(
                        client.as_raw_fd(),
                        libc::SOL_SOCKET,
                        libc::SO_LINGER,
                        (&linger as *const libc::linger).cast(),
                        std::mem::size_of_val(&linger) as libc::socklen_t,
                    )
                },
                0
            );
            drop(client);
            assert_eq!(observed.recv_timeout(WAIT).unwrap(), "aborted");
        }
        let error = completion
            .recv_timeout(WAIT)
            .expect("shutdown must wake and join the HTTP worker")
            .unwrap_err();
        assert!(error.to_string().contains("test stop"));
        assert_eq!(observed.recv_timeout(WAIT).unwrap(), "finished");
        assert_eq!(observed.recv_timeout(WAIT).unwrap(), "shutdown");
        server.join().unwrap();
        assert!(observed.try_recv().is_err());
    }
}
