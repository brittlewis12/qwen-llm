use super::*;
use serde_json::json;
use std::sync::mpsc::{Receiver, channel};

const WAIT: Duration = Duration::from_secs(2);

fn queued_log(capacity: usize) -> (TraceLog, Receiver<Record>) {
    let (sender, receiver) = sync_channel(capacity);
    (
        TraceLog {
            sender: Arc::new(Mutex::new(Some(sender))),
            worker: None,
        },
        receiver,
    )
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn concurrent_subscribers_preserve_correlation_and_each_request_order() {
    let capture = Capture::default();
    let log = TraceLog::with_writer(capture.clone()).unwrap();
    let first = log.subscriber();
    let second = log.subscriber();
    assert_ne!(first.request_id, second.request_id);
    let first_id = first.request_id.clone();
    let second_id = second.request_id.clone();
    let (first_sent, first_received) = channel();
    let (second_sent, second_received) = channel();
    let first_worker = std::thread::spawn(move || {
        for sequence in 0..3 {
            first.line(|| json!({"kind":"event", "sequence":sequence}));
            first_sent.send(()).unwrap();
            second_received.recv_timeout(WAIT).unwrap();
        }
    });
    let second_worker = std::thread::spawn(move || {
        for sequence in 0..3 {
            first_received.recv_timeout(WAIT).unwrap();
            second.line(|| json!({"kind":"event", "sequence":sequence}));
            second_sent.send(()).unwrap();
        }
    });
    for worker in [first_worker, second_worker] {
        worker.join().unwrap();
    }
    drop(log);
    let bytes = capture.0.lock().unwrap();
    let rows = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 6);
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(
            row["trace_request_id"],
            if index % 2 == 0 {
                first_id.as_str()
            } else {
                second_id.as_str()
            }
        );
    }
    for id in [first_id, second_id] {
        let sequences = rows
            .iter()
            .filter(|row| row["trace_request_id"] == id)
            .map(|row| row["sequence"].as_u64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(sequences, [0, 1, 2]);
    }
}

#[test]
fn full_queue_disables_all_subscribers_without_future_value_construction() {
    let (log, receiver) = queued_log(1);
    let first = log.subscriber();
    let second = log.subscriber();
    first.line(|| json!({"kind":"queued"}));
    second.line(|| json!({"kind":"full"}));
    for subscriber in [&first, &second, &log.subscriber()] {
        assert!(!subscriber.is_enabled());
        subscriber.line(|| panic!("disabled tracing must not construct payloads"));
    }
    assert_eq!(receiver.recv().unwrap().value["kind"], "queued");
    assert!(receiver.recv().is_err());
}

#[test]
fn disconnected_writer_disables_every_subscriber() {
    let (log, receiver) = queued_log(1);
    let first = log.subscriber();
    let second = log.subscriber();
    drop(receiver);
    first.line(|| json!({"kind":"lost"}));
    assert!(!second.is_enabled());
    second.line(|| panic!("writer has disconnected"));
}

#[test]
fn writer_failure_is_shared_before_any_subscriber_sends_again() {
    struct FailFlush;
    impl Write for FailFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("test disk failure"))
        }
    }
    let mut log = TraceLog::with_writer(FailFlush).unwrap();
    let first = log.subscriber();
    let second = log.subscriber();
    first.line(|| json!({"kind":"request"}));
    log.worker.take().unwrap().join().unwrap();
    assert!(!first.is_enabled());
    assert!(!second.is_enabled());
    second.line(|| panic!("writer failure must stop payload construction"));
}

#[test]
fn dropping_subscribers_does_not_close_writer_but_owner_drop_does() {
    struct NotifyDrop {
        capture: Capture,
        closed: std::sync::mpsc::Sender<()>,
    }
    impl Write for NotifyDrop {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.capture.write(bytes)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.capture.flush()
        }
    }
    impl Drop for NotifyDrop {
        fn drop(&mut self) {
            self.closed.send(()).unwrap();
        }
    }
    let capture = Capture::default();
    let (closed, closure) = channel();
    let log = TraceLog::with_writer(NotifyDrop {
        capture: capture.clone(),
        closed,
    })
    .unwrap();
    let first = log.subscriber();
    first.line(|| json!({"kind":"first"}));
    drop(first);
    let surviving = log.subscriber();
    assert!(surviving.is_enabled());
    surviving.line(|| json!({"kind":"second"}));
    assert!(closure.try_recv().is_err());
    drop(log);
    closure
        .try_recv()
        .expect("writer exited rather than surviving grace-period detachment");
    assert!(!surviving.is_enabled());
    surviving.line(|| panic!("subscriber cannot retain writer ownership"));
    let bytes = capture.0.lock().unwrap();
    assert_eq!(std::str::from_utf8(&bytes).unwrap().lines().count(), 2);
}

#[test]
fn owner_close_during_payload_construction_prevents_late_enqueue() {
    let (log, receiver) = queued_log(1);
    let subscriber = log.subscriber();
    let (entered, entry) = channel();
    let (release, released) = channel();
    let worker = std::thread::spawn(move || {
        subscriber.line(|| {
            entered.send(()).unwrap();
            released.recv_timeout(WAIT).unwrap();
            json!({"kind":"late"})
        });
    });
    entry.recv_timeout(WAIT).unwrap();
    drop(log);
    release.send(()).unwrap();
    worker.join().unwrap();
    assert!(receiver.recv().is_err());
}

#[test]
fn stalled_trace_worker_is_detached_instead_of_blocking_shutdown() {
    let (release_sender, release_receiver) = channel();
    let (done_sender, done_receiver) = channel();
    let worker = std::thread::spawn(move || {
        release_receiver.recv().unwrap();
        done_sender.send(()).unwrap();
    });
    assert!(join_trace_worker_with_grace(worker, Duration::ZERO).is_none());
    release_sender.send(()).unwrap();
    done_receiver
        .recv_timeout(WAIT)
        .expect("detached worker remains able to finish");
}
