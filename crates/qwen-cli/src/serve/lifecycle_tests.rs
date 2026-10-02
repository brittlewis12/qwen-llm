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

    fn render_prompt(&self, request: &items::ServeRequest) -> Result<String, items::ServeError> {
        if self.fail_render {
            return Err(items::ServeError::invalid_request(None, "render refused"));
        }
        self.request_profile().render(request)
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
    let body = r#"{"model":"test","input":"hello","max_output_tokens":1}"#;
    format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

fn stop_after_handling() -> impl FnMut(OwnerCheckpoint) -> Result<()> {
    let mut handling = false;
    move |checkpoint| {
        ensure!(!handling, "test stop after handling");
        handling = checkpoint == OwnerCheckpoint::BeforeHandling;
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
