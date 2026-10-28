//! Owner loop with a backend that keeps several requests in flight
//! (`max_active() > 1`). Single-flight lifecycles stay in `lifecycle_tests`.
use super::*;
use std::collections::HashSet;
use std::io;
use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::ThreadId;
use std::time::Instant;

const WAIT: Duration = Duration::from_secs(5);

struct Active {
    id: String,
    job: Option<batching::Job>,
    emitted: usize,
    parked_since: Option<Instant>,
}

/// Emits `pieces` pieces per request, one per pump, round robin; requests
/// whose id is in `blocked` make no progress. A request whose client does not
/// drain is parked and aborted after `park_timeout`.
struct BatchStub {
    max_active: usize,
    owner: ThreadId,
    address: SocketAddr,
    events: Sender<String>,
    active: Vec<Active>,
    pieces: usize,
    piece_bytes: usize,
    blocked: Arc<Mutex<HashSet<String>>>,
    park_timeout: Duration,
}

impl BatchStub {
    fn new(max_active: usize, address: SocketAddr, events: Sender<String>) -> Self {
        Self {
            max_active,
            owner: std::thread::current().id(),
            address,
            events,
            active: Vec::new(),
            pieces: 3,
            piece_bytes: 0,
            blocked: Arc::default(),
            park_timeout: WAIT,
        }
    }

    fn record(&self, event: String) {
        assert_eq!(std::thread::current().id(), self.owner);
        self.events.send(event).unwrap();
    }

    fn finish(
        &mut self,
        index: usize,
        result: Result<http::GenerationOutcome, http::BackendFailure>,
    ) {
        let mut active = self.active.remove(index);
        let label = match &result {
            Ok(_) => "done",
            Err(http::BackendFailure::Aborted(error))
                if error.kind() == io::ErrorKind::TimedOut =>
            {
                "parked-abort"
            }
            Err(http::BackendFailure::Aborted(_)) => "abort",
            Err(http::BackendFailure::Serve(_)) => "failed",
        };
        self.record(format!("{label} {}", active.id));
        active.job.take().unwrap().sink.complete(result);
    }
}

fn outcome(tokens: usize) -> http::GenerationOutcome {
    http::GenerationOutcome {
        end: output_partition::GenerationEnd::TokenLimit,
        usage: events::Usage {
            input_tokens: 1,
            output_tokens: tokens,
            cached_tokens: 0,
        },
        stats: None,
    }
}

impl http::GenerationBackend for BatchStub {
    fn model_id(&self) -> &str {
        "test"
    }
    fn request_profile(&self) -> request_profile::RequestProfile {
        request_profile::RequestProfile::test_qwen()
    }
    fn generate(
        &mut self,
        _: &items::ServeRequest,
        _: &str,
        _: &mut dyn http::GenerationSink,
    ) -> Result<http::GenerationOutcome, http::BackendFailure> {
        unreachable!("a batched backend is never asked to generate serially")
    }
    fn max_active(&self) -> usize {
        self.max_active
    }
    fn submit(&mut self, job: batching::Job) {
        let prompt = &job.prepared.prompt;
        let start = prompt.find("req-").expect("request label");
        let id = prompt[start..start + 5].to_owned();
        self.record(format!("submit {id}"));
        self.active.push(Active {
            id,
            job: Some(job),
            emitted: 0,
            parked_since: None,
        });
    }
    fn pump(&mut self) -> bool {
        let mut progressed = false;
        let mut index = 0;
        while index < self.active.len() {
            let sink = &mut self.active[index].job.as_mut().unwrap().sink;
            if let Err(error) = sink.check() {
                self.finish(index, Err(http::BackendFailure::Aborted(error)));
                progressed = true;
                continue;
            }
            let id = self.active[index].id.clone();
            if self.blocked.lock().unwrap().contains(&id) {
                index += 1;
                continue;
            }
            match self.active[index].job.as_mut().unwrap().sink.flush() {
                Err(error) => {
                    self.finish(index, Err(http::BackendFailure::Aborted(error)));
                    progressed = true;
                    continue;
                }
                Ok(false) => {
                    let since = *self.active[index]
                        .parked_since
                        .get_or_insert_with(Instant::now);
                    if since.elapsed() >= self.park_timeout {
                        self.finish(
                            index,
                            Err(http::BackendFailure::Aborted(io::Error::new(
                                io::ErrorKind::TimedOut,
                                "client did not drain output",
                            ))),
                        );
                        progressed = true;
                    } else {
                        index += 1;
                    }
                    continue;
                }
                Ok(true) => {
                    if self.active[index].parked_since.take().is_some() {
                        self.record(format!("unparked {id}"));
                    }
                }
            }
            if self.active[index].emitted == self.pieces {
                self.finish(index, Ok(outcome(self.pieces)));
                progressed = true;
                continue;
            }
            let active = &mut self.active[index];
            let piece = if self.piece_bytes == 0 {
                format!("{id}:{} ", active.emitted).into_bytes()
            } else {
                vec![b'x'; self.piece_bytes]
            };
            active.job.as_mut().unwrap().sink.queue(&piece);
            active.emitted += 1;
            let delivered = active.job.as_mut().unwrap().sink.flush();
            let emitted = active.emitted;
            if self.piece_bytes == 0 || emitted == 1 {
                self.record(format!("piece {id} {}", emitted - 1));
            }
            if let Ok(false) = delivered
                && self.active[index].parked_since.is_none()
            {
                self.active[index].parked_since = Some(Instant::now());
                self.record(format!("parked {id}"));
            }
            progressed = true;
            index += 1;
        }
        progressed
    }
    fn abort_all(&mut self) {
        self.record("abort_all".into());
        while !self.active.is_empty() {
            self.finish(
                0,
                Err(http::BackendFailure::Aborted(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "server stopping",
                ))),
            );
        }
    }
    fn request_finished(&mut self) {
        self.record("finished".into());
    }
    fn shutdown(&mut self) {
        assert!(TcpStream::connect_timeout(&self.address, WAIT).is_err());
        self.record("shutdown".into());
    }
}

fn connect(address: SocketAddr) -> TcpStream {
    let stream = TcpStream::connect_timeout(&address, WAIT).unwrap();
    stream.set_read_timeout(Some(WAIT)).unwrap();
    stream.set_write_timeout(Some(WAIT)).unwrap();
    stream
}

fn post(id: usize, stream: bool) -> String {
    let body = format!(
        r#"{{"model":"test","input":"req-{id}","stream":{stream},"max_output_tokens":64,"x_qwen":{{"no_thinking":true}}}}"#
    );
    format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

fn send(address: SocketAddr, id: usize, stream: bool) -> TcpStream {
    let mut client = connect(address);
    client.write_all(post(id, stream).as_bytes()).unwrap();
    client
}

fn read_all(mut client: TcpStream) -> String {
    let mut response = String::new();
    let _ = client.read_to_string(&mut response);
    response
}

fn wait_for(observed: &Receiver<String>, seen: &mut Vec<String>, event: &str) {
    let deadline = Instant::now() + WAIT;
    while !seen.iter().any(|e| e == event) {
        let left = deadline.saturating_duration_since(Instant::now());
        seen.push(
            observed
                .recv_timeout(left)
                .unwrap_or_else(|_| panic!("no {event:?} in {seen:?}")),
        );
    }
}

fn position(seen: &[String], event: &str) -> usize {
    seen.iter()
        .position(|e| e == event)
        .unwrap_or_else(|| panic!("no {event:?} in {seen:?}"))
}

struct Server {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<Result<()>>,
}

fn serve(
    listener: TcpListener,
    depth: usize,
    configure: impl FnOnce(&mut BatchStub) + Send + 'static,
    events: Sender<String>,
) -> Server {
    let address = listener.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let stopping = stop.clone();
    let thread = std::thread::spawn(move || {
        let mut backend = BatchStub::new(3, address, events);
        configure(&mut backend);
        accept_loop_with_queue_depth(listener, &mut backend, depth, |_| {
            ensure!(!stopping.load(Ordering::Acquire), "test stop");
            Ok(())
        })
    });
    Server { stop, thread }
}

impl Server {
    fn stop(self) {
        self.stop.store(true, Ordering::Release);
        let error = self.thread.join().unwrap().unwrap_err();
        assert!(error.to_string().contains("test stop"), "{error:#}");
    }
}

#[test]
fn batched_requests_interleave_and_each_connection_completes_once() {
    let listener = bind_loopback("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (events, observed) = channel();
    let blocked: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(
        ["req-1", "req-2", "req-3"].map(String::from).into(),
    ));
    let gate = blocked.clone();
    let server = serve(listener, 4, move |b| b.blocked = gate, events);
    let clients: Vec<_> = (1..=3).map(|id| send(address, id, id == 2)).collect();
    let mut seen = Vec::new();
    for id in 1..=3 {
        wait_for(&observed, &mut seen, &format!("submit req-{id}"));
    }
    blocked.lock().unwrap().clear();
    for (index, client) in clients.into_iter().enumerate() {
        let id = index + 1;
        let response = read_all(client);
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let text = format!("req-{id}:0 req-{id}:1 req-{id}:2");
        if id == 2 {
            for piece in 0..3 {
                assert!(response.contains(&format!("req-2:{piece} ")), "{response}");
            }
        } else {
            assert!(response.contains(&text), "{response}");
        }
    }
    for id in 1..=3 {
        wait_for(&observed, &mut seen, &format!("done req-{id}"));
    }
    // Every request emitted its first piece before any request finished.
    let first_done = (1..=3)
        .map(|id| position(&seen, &format!("done req-{id}")))
        .min()
        .unwrap();
    for id in 1..=3 {
        assert!(position(&seen, &format!("piece req-{id} 0")) < first_done);
    }
    server.stop();
    let rest: Vec<_> = observed.try_iter().collect();
    seen.extend(rest);
    assert_eq!(seen.iter().filter(|e| *e == "finished").count(), 3);
    assert_eq!(seen.last().unwrap(), "shutdown");
    assert_eq!(position(&seen, "abort_all") + 1, seen.len() - 1);
}

#[test]
fn one_disconnect_aborts_only_that_request() {
    use std::os::fd::AsRawFd;
    let listener = bind_loopback("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (events, observed) = channel();
    let blocked: Arc<Mutex<HashSet<String>>> =
        Arc::new(Mutex::new(["req-1", "req-2"].map(String::from).into()));
    let gate = blocked.clone();
    let server = serve(listener, 4, move |b| b.blocked = gate, events);
    let gone = send(address, 1, false);
    let kept = send(address, 2, true);
    let mut seen = Vec::new();
    wait_for(&observed, &mut seen, "submit req-1");
    wait_for(&observed, &mut seen, "submit req-2");
    // Reset only this owned test connection.
    let linger = libc::linger {
        l_onoff: 1,
        l_linger: 0,
    };
    assert_eq!(
        unsafe {
            libc::setsockopt(
                gone.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_LINGER,
                (&linger as *const libc::linger).cast(),
                std::mem::size_of_val(&linger) as libc::socklen_t,
            )
        },
        0
    );
    drop(gone);
    wait_for(&observed, &mut seen, "abort req-1");
    blocked.lock().unwrap().clear();
    // The stub ends at its token limit: a complete, incomplete-status stream.
    let response = read_all(kept);
    assert!(response.contains("req-2:2 "), "{response}");
    assert!(response.contains("response.incomplete"), "{response}");
    wait_for(&observed, &mut seen, "done req-2");
    server.stop();
    seen.extend(observed.try_iter());
    assert_eq!(seen.iter().filter(|e| *e == "finished").count(), 2);
    assert!(!seen.iter().any(|e| e == "piece req-1 0"), "{seen:?}");
}

#[test]
fn shutdown_aborts_active_requests_and_busy_rejects_queued_connections() {
    let listener = bind_loopback("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (events, observed) = channel();
    let blocked: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(
        ["req-1", "req-2", "req-3", "req-4"]
            .map(String::from)
            .into(),
    ));
    let gate = blocked.clone();
    let server = serve(
        listener,
        4,
        move |b| {
            b.blocked = gate;
            b.max_active = 2;
        },
        events,
    );
    let first = send(address, 1, false);
    let second = send(address, 2, true);
    let mut seen = Vec::new();
    wait_for(&observed, &mut seen, "submit req-1");
    wait_for(&observed, &mut seen, "submit req-2");
    let queued = send(address, 3, false);
    std::thread::sleep(WAIT / 25);
    server.stop();
    let response = read_all(queued);
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    assert!(read_all(first).is_empty());
    let streamed = read_all(second);
    assert!(streamed.starts_with("HTTP/1.1 200"), "{streamed}");
    assert!(!streamed.contains("response.incomplete"), "{streamed}");
    seen.extend(observed.try_iter());
    assert!(!seen.iter().any(|e| e == "submit req-3"), "{seen:?}");
    let abort_all = position(&seen, "abort_all");
    assert_eq!(
        seen[abort_all + 1..],
        [
            "abort req-1",
            "abort req-2",
            "finished",
            "finished",
            "shutdown"
        ]
    );
}

#[test]
fn connections_beyond_max_active_wait_in_arrival_order() {
    let listener = bind_loopback("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (events, observed) = channel();
    let blocked: Arc<Mutex<HashSet<String>>> =
        Arc::new(Mutex::new(["req-1", "req-2"].map(String::from).into()));
    let gate = blocked.clone();
    let server = serve(
        listener,
        4,
        move |b| {
            b.blocked = gate;
            b.max_active = 2;
        },
        events,
    );
    let clients = [send(address, 1, false), send(address, 2, false)];
    let mut seen = Vec::new();
    wait_for(&observed, &mut seen, "submit req-1");
    wait_for(&observed, &mut seen, "submit req-2");
    let third = send(address, 3, false);
    std::thread::sleep(WAIT / 50);
    let fourth = send(address, 4, false);
    std::thread::sleep(WAIT / 25);
    seen.extend(observed.try_iter());
    assert!(
        !seen.iter().any(|e| e.starts_with("submit req-3")),
        "{seen:?}"
    );
    // Request 1 finishes; request 2 stays active, so only the oldest waiting
    // connection is admitted.
    blocked.lock().unwrap().remove("req-1");
    let [first, second] = clients;
    assert!(read_all(first).starts_with("HTTP/1.1 200"));
    assert!(read_all(third).starts_with("HTTP/1.1 200"));
    wait_for(&observed, &mut seen, "done req-3");
    blocked.lock().unwrap().clear();
    assert!(read_all(second).starts_with("HTTP/1.1 200"));
    assert!(read_all(fourth).starts_with("HTTP/1.1 200"));
    wait_for(&observed, &mut seen, "done req-4");
    server.stop();
    let submits: Vec<_> = seen
        .iter()
        .filter(|e| e.starts_with("submit"))
        .cloned()
        .collect();
    assert_eq!(
        submits[2..],
        ["submit req-3".to_owned(), "submit req-4".to_owned()]
    );
    assert!(position(&seen, "done req-3") < position(&seen, "submit req-4"));
}

fn small_receive_buffer(socket: &TcpStream) {
    use std::os::fd::AsRawFd;
    let bytes: libc::c_int = 4096;
    // This test owns the socket and passes a correctly sized integer option.
    assert_eq!(
        unsafe {
            libc::setsockopt(
                socket.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_RCVBUF,
                (&bytes as *const libc::c_int).cast(),
                std::mem::size_of_val(&bytes) as libc::socklen_t,
            )
        },
        0
    );
}

#[test]
fn a_client_that_does_not_drain_parks_without_stalling_others() {
    let listener = bind_loopback("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (events, observed) = channel();
    let blocked: Arc<Mutex<HashSet<String>>> =
        Arc::new(Mutex::new(["req-1", "req-2"].map(String::from).into()));
    let gate = blocked.clone();
    let server = serve(
        listener,
        4,
        move |b| {
            b.blocked = gate;
            b.pieces = 400;
            b.piece_bytes = 64 * 1024;
            b.park_timeout = Duration::from_millis(400);
        },
        events,
    );
    let mut slow = connect(address);
    small_receive_buffer(&slow);
    slow.write_all(post(1, true).as_bytes()).unwrap();
    let fast = send(address, 2, false);
    let mut seen = Vec::new();
    wait_for(&observed, &mut seen, "submit req-1");
    wait_for(&observed, &mut seen, "submit req-2");
    blocked.lock().unwrap().clear();
    wait_for(&observed, &mut seen, "parked req-1");
    // The other request keeps generating and finishes while req-1 is parked.
    let response = read_all(fast);
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    wait_for(&observed, &mut seen, "done req-2");
    wait_for(&observed, &mut seen, "parked-abort req-1");
    assert!(position(&seen, "done req-2") < position(&seen, "parked-abort req-1"));
    let _ = slow.shutdown(Shutdown::Both);
    server.stop();
    seen.extend(observed.try_iter());
    assert_eq!(seen.iter().filter(|e| *e == "finished").count(), 2);
}
