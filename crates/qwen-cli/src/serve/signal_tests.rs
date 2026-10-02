use super::*;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::os::fd::AsRawFd;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{Receiver, channel};

const CHILD: &str = "serve::signal_tests::bridge_signal_child";
const CHILD_ENV: &str = "QWEN_HTTP_BRIDGE_SIGNAL_CHILD";
const PREFIX: &str = "qwen-bridge-signal-test:";

fn marker(value: &str) {
    println!("{PREFIX}{value}");
    std::io::stdout().flush().unwrap();
}

pub(super) fn bridge_wait_observed() {
    static OBSERVED: AtomicBool = AtomicBool::new(false);
    if std::env::var(CHILD_ENV).as_deref() == Ok("1") && !OBSERVED.swap(true, Ordering::AcqRel) {
        marker("bridge_wait");
    }
}

fn small_buffer(socket: &impl AsRawFd, option: libc::c_int) {
    let bytes: libc::c_int = 4096;
    // The fixture owns this socket and the option is a correctly sized integer.
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

struct SignalBackend {
    owner: std::thread::ThreadId,
    address: SocketAddr,
    finished: usize,
    shutdown: usize,
}

impl http::GenerationBackend for SignalBackend {
    fn model_id(&self) -> &str {
        "test"
    }
    fn generate(
        &mut self,
        _: &items::ServeRequest,
        _: &str,
        sink: &mut dyn http::GenerationSink,
    ) -> Result<http::GenerationOutcome, http::BackendFailure> {
        assert_eq!(std::thread::current().id(), self.owner);
        marker("generate");
        let error = sink
            .piece(&vec![b'x'; 2 * 1024 * 1024])
            .expect_err("signal must interrupt the blocked original piece");
        marker("aborted");
        Err(http::BackendFailure::Aborted(error))
    }
    fn request_finished(&mut self) {
        assert_eq!(std::thread::current().id(), self.owner);
        self.finished += 1;
        marker("finished");
    }
    fn shutdown(&mut self) {
        assert_eq!(std::thread::current().id(), self.owner);
        assert!(TcpStream::connect_timeout(&self.address, Duration::from_millis(500)).is_err());
        self.shutdown += 1;
        marker("shutdown");
    }
}

#[test]
#[ignore = "CPU subprocess fixture; invoked by sigterm_settles_blocked_bridge_without_client_disconnect"]
fn bridge_signal_child() {
    assert_eq!(std::env::var(CHILD_ENV).as_deref(), Ok("1"));
    crate::shutdown::install().unwrap();
    let listener = bind_loopback("127.0.0.1:0").unwrap();
    small_buffer(&listener, libc::SO_SNDBUF);
    let address = listener.local_addr().unwrap();
    let mut backend = SignalBackend {
        owner: std::thread::current().id(),
        address,
        finished: 0,
        shutdown: 0,
    };
    marker(&format!("ready {address}"));
    let error = accept_loop(listener, "test", 0.0, &mut backend, &mut None).unwrap_err();
    assert!(
        error
            .to_string()
            .contains(&format!("termination signal {}", libc::SIGTERM)),
        "{error:#}"
    );
    assert_eq!(backend.finished, 1);
    assert_eq!(backend.shutdown, 1);
    marker("stopped");
}

struct OwnedChild {
    child: Option<Child>,
    reader: Option<JoinHandle<()>>,
}

impl OwnedChild {
    fn sigterm(&mut self, deadline: Instant) {
        let child = self.child.as_mut().expect("child has not been reaped");
        assert!(
            child.try_wait().unwrap().is_none(),
            "child exited before signal"
        );
        remaining(deadline);
        // Only the unreaped child spawned by this test is signalled. Its PID
        // cannot be reused until this parent reaps it.
        assert_eq!(
            unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) },
            0
        );
    }

    fn wait_until(&mut self, deadline: Instant) -> ExitStatus {
        loop {
            remaining(deadline);
            if let Some(status) = self.child.as_mut().unwrap().try_wait().unwrap() {
                self.child.take();
                self.reader.take().unwrap().join().unwrap();
                remaining(deadline);
                return status;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            if !matches!(child.try_wait(), Ok(Some(_))) {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

fn remaining(deadline: Instant) -> Duration {
    let remaining = deadline.saturating_duration_since(Instant::now());
    assert!(
        !remaining.is_zero(),
        "subprocess protocol exceeded overall deadline"
    );
    remaining
}

fn next_marker(receiver: &Receiver<String>, deadline: Instant) -> String {
    receiver
        .recv_timeout(remaining(deadline))
        .expect("child marker missing before overall deadline")
}

#[test]
fn sigterm_settles_blocked_bridge_without_client_disconnect() {
    let deadline = Instant::now() + Duration::from_secs(5);
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            CHILD,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut child = OwnedChild {
        child: Some(child),
        reader: None,
    };
    let stdout = child.child.as_mut().unwrap().stdout.take().unwrap();
    let (sender, receiver) = channel();
    child.reader = Some(std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            // libtest can prefix the fixture's first line with its test name.
            if let Some((_, value)) = line.split_once(PREFIX)
                && sender.send(value.to_owned()).is_err()
            {
                break;
            }
        }
    }));
    let ready = next_marker(&receiver, deadline);
    let address: SocketAddr = ready
        .strip_prefix("ready ")
        .expect("fixture readiness marker")
        .parse()
        .unwrap();
    assert!(address.ip().is_loopback());
    let mut client = TcpStream::connect_timeout(&address, remaining(deadline)).unwrap();
    small_buffer(&client, libc::SO_RCVBUF);
    client.set_read_timeout(Some(remaining(deadline))).unwrap();
    client.set_write_timeout(Some(remaining(deadline))).unwrap();
    let body = r#"{"model":"test","input":"hi","stream":true}"#;
    client.write_all(format!("POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
    assert_eq!(next_marker(&receiver, deadline), "generate");
    assert_eq!(next_marker(&receiver, deadline), "bridge_wait");
    client.set_read_timeout(Some(remaining(deadline))).unwrap();
    client.read_exact(&mut [0u8; 1]).unwrap();
    child.sigterm(deadline);
    let mut remaining = Vec::new();
    loop {
        let value = next_marker(&receiver, deadline);
        let stopped = value == "stopped";
        remaining.push(value);
        if stopped {
            break;
        }
    }
    assert!(child.wait_until(deadline).success());
    remaining.extend(receiver.try_iter());
    assert_eq!(remaining, ["aborted", "finished", "shutdown", "stopped"]);
    // Keep the peer alive and unread through child exit: disconnect cancellation
    // must not conceal a broken process-signal path.
    drop(client);
}
