use super::*;
use crate::serve::events::Usage;
use crate::serve::output_partition::GenerationEnd;
use crate::serve::owner_activity::OwnerActivity;
use std::sync::mpsc::channel;

const WAIT: Duration = Duration::from_secs(2);

fn control() -> Arc<Control> {
    Arc::new(Control {
        execution: ExecutionControl::default(),
        server: None,
        owner: std::thread::current(),
    })
}

fn outcome() -> GenerationOutcome {
    GenerationOutcome {
        end: GenerationEnd::TokenLimit,
        usage: Usage {
            input_tokens: 1,
            output_tokens: 1,
            cached_tokens: 0,
        },
        stats: None,
    }
}

struct Backend<F>(F);
impl<F: FnMut(&mut dyn GenerationSink) -> Result<GenerationOutcome, BackendFailure>>
    GenerationBackend for Backend<F>
{
    fn model_id(&self) -> &str {
        "test"
    }
    fn generate(
        &mut self,
        _request: &ServeRequest,
        _prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        (self.0)(sink)
    }
}

fn pipeline() -> (OwnerActivity, HttpProxy, Receiver<Work>) {
    let activity = OwnerActivity::default();
    let guard = activity.admission().try_admit().unwrap();
    let (work, incoming) = sync_channel(1);
    (
        activity,
        HttpProxy {
            extra_cpu_reserve: 0,
            model_id: "test".into(),
            profile: RequestProfile::UnboundQwen,
            work,
            control: control(),
            activity: Arc::new(guard),
        },
        incoming,
    )
}

fn prepared() -> Arc<PreparedResponse> {
    Arc::new(PreparedResponse {
        request: crate::serve::items::parse_request(
            &serde_json::json!({"model":"test", "input":"hi"}),
        )
        .unwrap(),
        prompt: "already rendered".into(),
    })
}

struct Collect(Vec<u8>);
impl GenerationSink for Collect {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.0.extend_from_slice(bytes);
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn bridge_reservation_covers_stack_and_all_live_chunk_slots() {
    assert!(
        BUFFER_RESERVE_BYTES >= (WORKER_STACK_BYTES + (PIECE_CAPACITY + 2) * CHUNK_BYTES) as u64
    );
}

#[test]
fn owner_sink_includes_control_allowance_exactly_once() {
    for extra_cpu_reserve in [
        0,
        super::super::control::STATIC_CPU_RESERVE_BYTES,
        super::super::control::CPU_RESERVE_BYTES,
    ] {
        let (pieces, _) = sync_channel(1);
        let (_, processed) = sync_channel(1);
        let sink = OwnerSink {
            pieces,
            processed,
            control: control(),
            extra_cpu_reserve,
        };
        let expected = BUFFER_RESERVE_BYTES + extra_cpu_reserve;
        assert_eq!(sink.transport_reserve_bytes(), expected);
        super::super::transport_memory::admit_resident_transport(
            sink.transport_reserve_bytes(),
            Some(expected),
        )
        .unwrap();
        assert!(
            super::super::transport_memory::admit_resident_transport(
                sink.transport_reserve_bytes(),
                Some(expected - 1)
            )
            .is_err()
        );
    }
}

#[test]
fn owner_can_settle_a_started_connection_without_dispatching_work() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let _client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (stream, _) = listener.accept().unwrap();
    let mut activity = OwnerActivity::default();
    let guard = activity.admission().try_admit().unwrap();
    let backend = Backend(|_: &mut dyn GenerationSink| panic!("no owner dispatch requested"));
    let connection = Connection::start(stream, &backend, None, guard).unwrap();
    activity.idle_if_quiet(|| panic!("started connection still owns activity"));
    connection.stop_and_join().unwrap();
    let mut completions = 0;
    activity.drain_finished(|| completions += 1);
    assert_eq!(completions, 1);
    assert!(activity.is_settled());
}

#[test]
fn explicit_owner_settlement_reports_worker_panic_after_releasing_activity() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let _client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (socket, _) = listener.accept().unwrap();
    let mut activity = OwnerActivity::default();
    let guard = activity.admission().try_admit().unwrap();
    let (sender, incoming) = sync_channel(1);
    let thread = std::thread::spawn(move || {
        let _guard = guard;
        let _sender = sender;
        panic!("worker supervision fixture")
    });
    let connection = Connection {
        _execution: None,
        incoming: Some(incoming),
        worker: Worker {
            socket,
            control: control(),
            thread: Some(thread),
        },
    };
    assert!(
        connection
            .stop_and_join()
            .unwrap_err()
            .to_string()
            .contains("worker panicked")
    );
    let mut completions = 0;
    activity.drain_finished(|| completions += 1);
    assert_eq!(completions, 1);
    assert!(activity.is_settled());
}

#[test]
fn full_piece_channel_waits_without_spurious_failure_or_extra_copy() {
    let (pieces, incoming) = sync_channel(1);
    pieces
        .send(Piece {
            bytes: vec![1],
            new_piece_bytes: Some(1),
            last: false,
        })
        .ok()
        .unwrap();
    let (_, processed) = sync_channel(1);
    let mut sink = OwnerSink {
        extra_cpu_reserve: 0,
        pieces,
        processed,
        control: control(),
    };
    let pending = vec![2; CHUNK_BYTES];
    let address = pending.as_ptr();
    let mut waits = 0;
    sink.send_chunk(
        Piece {
            bytes: pending,
            new_piece_bytes: Some(CHUNK_BYTES),
            last: true,
        },
        || {
            waits += 1;
            assert_eq!(incoming.try_recv().unwrap().bytes, [1]);
        },
    )
    .unwrap();
    assert_eq!(waits, 1);
    let delivered = incoming.try_recv().unwrap();
    assert_eq!(delivered.bytes.as_ptr(), address);
    assert_eq!(delivered.bytes, vec![2; CHUNK_BYTES]);
}

#[test]
fn cancellation_wakes_full_channel_without_consuming_queued_output() {
    let (pieces, incoming) = sync_channel(1);
    pieces
        .send(Piece {
            bytes: vec![1],
            new_piece_bytes: Some(1),
            last: false,
        })
        .ok()
        .unwrap();
    let (_, processed) = sync_channel(1);
    let control = control();
    let mut sink = OwnerSink {
        extra_cpu_reserve: 0,
        pieces,
        processed,
        control: Arc::clone(&control),
    };
    let error = sink
        .send_chunk(
            Piece {
                bytes: vec![2],
                new_piece_bytes: Some(1),
                last: true,
            },
            || control.cancel(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::ConnectionAborted);
    assert_eq!(incoming.try_recv().unwrap().bytes, [1]);
    assert!(incoming.try_recv().is_err());
}

#[test]
fn bridge_shares_preparation_and_drains_split_pieces_before_terminal() {
    for fail in [false, true] {
        let (mut activity, mut proxy, incoming) = pipeline();
        let prepared = prepared();
        let original = Arc::clone(&prepared);
        let payload = "a".repeat(CHUNK_BYTES - 1) + "\u{1f642}" + &"b".repeat(CHUNK_BYTES * 4);
        let expected = payload.as_bytes().to_vec();
        let worker = std::thread::spawn(move || {
            let _cancel = CancelOnDrop(Arc::clone(&proxy.control));
            let mut sink = Collect(Vec::new());
            let result = proxy.generate_prepared(prepared, &mut sink);
            (sink.0, result)
        });
        let work = incoming.recv_timeout(WAIT).unwrap();
        assert!(Arc::ptr_eq(&original, &work.prepared));
        let mut backend = Backend(|sink: &mut dyn GenerationSink| {
            assert_eq!(sink.transport_reserve_bytes(), BUFFER_RESERVE_BYTES);
            sink.piece(payload.as_bytes())
                .map_err(BackendFailure::Aborted)?;
            if fail {
                Err(ServeError::server_error("after output").into())
            } else {
                Ok(outcome())
            }
        });
        execute(work, &mut backend);
        let (bytes, result) = worker.join().unwrap();
        assert_eq!(bytes, expected);
        assert_eq!(result.is_err(), fail);
        let mut calls = 0;
        activity.drain_finished(|| calls += 1);
        assert_eq!(calls, 1);
        assert!(activity.is_settled());
    }
}

#[test]
fn subscriber_heartbeats_while_waiting_and_missing_terminal_is_failure() {
    struct Heartbeat(std::sync::mpsc::Sender<()>);
    impl GenerationSink for Heartbeat {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            Ok(())
        }
        fn tick(&mut self) -> io::Result<()> {
            self.0.send(()).unwrap();
            Ok(())
        }
    }
    let (mut activity, mut proxy, incoming) = pipeline();
    let (tick, ticked) = channel();
    let worker = std::thread::spawn(move || {
        let _cancel = CancelOnDrop(Arc::clone(&proxy.control));
        proxy.generate_prepared(prepared(), &mut Heartbeat(tick))
    });
    let work = incoming.recv_timeout(WAIT).unwrap();
    ticked.recv_timeout(WAIT).unwrap();
    drop(work);
    let Err(BackendFailure::Serve(error)) = worker.join().unwrap() else {
        panic!("missing terminal must fail")
    };
    assert!(error.message.contains("without outcome"));
    activity.drain_finished(|| {});
    assert!(activity.is_settled());
}

#[test]
fn failed_subscriber_cancels_but_does_not_finish_active_generation() {
    struct FailSink;
    impl GenerationSink for FailSink {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            Err(aborted("test disconnect"))
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let (mut activity, mut proxy, incoming) = pipeline();
    let worker = std::thread::spawn(move || {
        let _cancel = CancelOnDrop(Arc::clone(&proxy.control));
        proxy.generate_prepared(prepared(), &mut FailSink)
    });
    let work = incoming.recv_timeout(WAIT).unwrap();
    let control = Arc::clone(&work.control);
    let (release, released) = channel();
    let owner = std::thread::spawn(move || {
        execute(
            work,
            &mut Backend(|sink: &mut dyn GenerationSink| {
                assert!(sink.piece(b"one").is_err());
                released.recv_timeout(WAIT).unwrap();
                assert!(sink.tick().is_err());
                Err(BackendFailure::Aborted(aborted("cancelled")))
            }),
        );
    });
    assert!(matches!(
        worker.join().unwrap(),
        Err(BackendFailure::Aborted(_))
    ));
    assert!(control.execution.is_cancelled());
    activity.drain_finished(|| panic!("generation is still active"));
    activity.idle_if_quiet(|| panic!("cannot publish idle with in-flight generation"));
    release.send(()).unwrap();
    owner.join().unwrap();
    let mut completions = 0;
    activity.drain_finished(|| completions += 1);
    assert_eq!(completions, 1);
    assert!(activity.is_settled());
}

#[test]
fn cancelled_queued_work_never_calls_backend() {
    let (mut activity, mut proxy, incoming) = pipeline();
    let control = Arc::clone(&proxy.control);
    let worker = std::thread::spawn(move || {
        let _cancel = CancelOnDrop(Arc::clone(&proxy.control));
        proxy.generate_prepared(prepared(), &mut Collect(Vec::new()))
    });
    let work = incoming.recv_timeout(WAIT).unwrap();
    control.cancel();
    execute(
        work,
        &mut Backend(|_: &mut dyn GenerationSink| panic!("cancelled work must not execute")),
    );
    assert!(matches!(
        worker.join().unwrap(),
        Err(BackendFailure::Aborted(_))
    ));
    activity.drain_finished(|| {});
    assert!(activity.is_settled());
}

/// Idle residency must not pulse after a fault: the owner reports exactly
/// the generations that ended in a server-side (5xx) failure, and nothing
/// for success, a client error or a transport abort.
#[test]
fn only_server_side_failures_are_reported_to_the_backend() {
    type Outcome = fn() -> Result<GenerationOutcome, BackendFailure>;
    struct Counting {
        result: Outcome,
        failures: usize,
    }
    impl GenerationBackend for Counting {
        fn model_id(&self) -> &str {
            "test"
        }
        fn generate(
            &mut self,
            _request: &ServeRequest,
            _prompt: &str,
            _sink: &mut dyn GenerationSink,
        ) -> Result<GenerationOutcome, BackendFailure> {
            (self.result)()
        }
        fn request_failed_on_server(&mut self) {
            self.failures += 1;
        }
    }
    let cases: [(Outcome, usize); 5] = [
        (|| Ok(outcome()), 0),
        (|| Err(ServeError::server_error("fault").into()), 1),
        (
            || {
                let mut error = ServeError::server_error("memory");
                error.status = 503;
                Err(error.into())
            },
            1,
        ),
        (
            || Err(ServeError::invalid_request(None, "client").into()),
            0,
        ),
        (
            || {
                Err(BackendFailure::Aborted(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "gone",
                )))
            },
            0,
        ),
    ];
    for (result, expected) in cases {
        let (mut activity, mut proxy, incoming) = pipeline();
        let worker = std::thread::spawn(move || {
            proxy.generate_prepared(prepared(), &mut Collect(Vec::new()))
        });
        let work = incoming.recv_timeout(WAIT).unwrap();
        let mut backend = Counting {
            result,
            failures: 0,
        };
        execute(work, &mut backend);
        let _ = worker.join().unwrap();
        assert_eq!(backend.failures, expected);
        activity.drain_finished(|| {});
    }
}

#[test]
fn owner_waits_for_downstream_processing_and_cancellation_interrupts_ack_wait() {
    struct Blocked {
        entered: std::sync::mpsc::Sender<()>,
        release: Receiver<()>,
    }
    impl GenerationSink for Blocked {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            self.entered.send(()).unwrap();
            self.release.recv_timeout(WAIT).unwrap();
            Ok(())
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    for cancel in [false, true] {
        let (mut activity, mut proxy, incoming) = pipeline();
        let control = Arc::clone(&proxy.control);
        let (entered, entry) = channel();
        let (release, released) = channel();
        let worker = std::thread::spawn(move || {
            let _cancel = CancelOnDrop(Arc::clone(&proxy.control));
            proxy.generate_prepared(
                prepared(),
                &mut Blocked {
                    entered,
                    release: released,
                },
            )
        });
        let work = incoming.recv_timeout(WAIT).unwrap();
        let (advanced, advancement) = channel();
        let owner = std::thread::spawn(move || {
            execute(
                work,
                &mut Backend(|sink: &mut dyn GenerationSink| {
                    let result = sink.piece(b"close reasoning");
                    advanced.send(result.is_ok()).unwrap();
                    result.map_err(BackendFailure::Aborted)?;
                    Ok(outcome())
                }),
            )
        });
        entry.recv_timeout(WAIT).unwrap();
        assert!(
            advancement.try_recv().is_err(),
            "owner advanced before downstream processing returned"
        );
        if cancel {
            control.cancel();
            assert!(!advancement.recv_timeout(WAIT).unwrap());
            owner.join().unwrap();
            activity.idle_if_quiet(|| panic!("blocked response processing still owns activity"));
            release.send(()).unwrap();
            assert!(worker.join().unwrap().is_err());
        } else {
            release.send(()).unwrap();
            assert!(advancement.recv_timeout(WAIT).unwrap());
            owner.join().unwrap();
            assert!(worker.join().unwrap().is_ok());
        }
        let mut calls = 0;
        activity.drain_finished(|| calls += 1);
        assert_eq!(calls, 1);
        assert!(activity.is_settled());
    }
}

#[test]
fn panicking_subscriber_cancels_owner_and_releases_one_activity() {
    struct PanicSink;
    impl GenerationSink for PanicSink {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            panic!("test worker failure")
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let (mut activity, mut proxy, incoming) = pipeline();
    let control = Arc::clone(&proxy.control);
    let worker = std::thread::spawn(move || {
        let _cancel = CancelOnDrop(Arc::clone(&proxy.control));
        proxy.generate_prepared(prepared(), &mut PanicSink)
    });
    let work = incoming.recv_timeout(WAIT).unwrap();
    execute(
        work,
        &mut Backend(|sink: &mut dyn GenerationSink| {
            assert!(sink.piece(b"one").is_err());
            Err(BackendFailure::Aborted(aborted("worker panic")))
        }),
    );
    assert!(worker.join().is_err());
    assert!(control.execution.is_cancelled());
    let mut calls = 0;
    activity.drain_finished(|| calls += 1);
    assert_eq!(calls, 1);
    assert!(activity.is_settled());
}

#[test]
fn real_streaming_partition_preserves_utf8_and_reasoning_across_bridge_boundaries() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    let reasoning = "a".repeat(CHUNK_BYTES - 8) + "\u{1f642}" + &"b".repeat(CHUNK_BYTES - 6);
    let text = "answer \u{1f642}";
    let output = format!("<think>{reasoning}</think>{text}");
    let mut envelopes = Vec::new();
    for bridged in [false, true] {
        let output = output.clone();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let owner = std::thread::current().id();
            let (stream, _) = listener.accept().unwrap();
            let mut backend = Backend(|sink: &mut dyn GenerationSink| {
                assert_eq!(std::thread::current().id(), owner);
                sink.piece(output.as_bytes())
                    .map_err(BackendFailure::Aborted)?;
                let mut result = outcome();
                result.end = GenerationEnd::StopToken(0);
                Ok(result)
            });
            if bridged {
                let mut activity = OwnerActivity::default();
                let guard = activity.admission().try_admit().unwrap();
                handle_connection(stream, &mut backend, None, guard, || Ok(())).unwrap();
                activity.drain_finished(|| {});
                assert!(activity.is_settled());
            } else {
                http::handle_connection(&stream, &mut backend, None).unwrap();
            }
        });
        let body = r#"{"model":"test","input":"hi","stream":true}"#;
        let mut client = TcpStream::connect(address).unwrap();
        client.set_read_timeout(Some(WAIT)).unwrap();
        client.write_all(format!("POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        server.join().unwrap();
        assert!(response.ends_with("data: [DONE]\n\n"));
        for (event, expected) in [
            ("response.reasoning.delta", reasoning.as_str()),
            ("response.output_text.delta", text),
        ] {
            let prefix = format!("event: {event}\n");
            let mut text = String::new();
            for block in response
                .split("\n\n")
                .filter(|block| block.starts_with(&prefix))
            {
                let data = block
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .unwrap();
                let payload: serde_json::Value = serde_json::from_str(data).unwrap();
                text.push_str(payload["delta"].as_str().unwrap());
            }
            assert_eq!(text, expected);
        }
        let terminal = response
            .split("\n\n")
            .find(|block| block.starts_with("event: response.completed\n"))
            .unwrap();
        let data = terminal
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap();
        let envelope: serde_json::Value = serde_json::from_str(data).unwrap();
        envelopes.push(envelope["response"].clone());
    }
    for envelope in &envelopes {
        assert_eq!(envelope["status"], "completed");
        assert_eq!(envelope["output"][0]["type"], "reasoning");
        assert_eq!(envelope["output"][0]["content"][0]["text"], reasoning);
        assert_eq!(envelope["output"][1]["content"][0]["text"], text);
    }
    assert_eq!(envelopes[0]["usage"], envelopes[1]["usage"]);
}
