//! One bounded trace writer; per-request subscribers never own its shutdown.

use serde::Serialize;
use serde_json::Value;
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// This bounds queued records, not their bytes. Concurrent transport admission
// must also budget payloads being constructed, queued and written.
const TRACE_QUEUE_CAPACITY: usize = 8;
const TRACE_SHUTDOWN_GRACE: Duration = Duration::from_millis(250);
const TRACE_SHUTDOWN_POLL_INTERVAL: Duration = Duration::from_millis(5);

#[derive(Serialize)]
struct Record {
    trace_request_id: String,
    #[serde(flatten)]
    value: Value,
}

type SharedSender = Arc<Mutex<Option<SyncSender<Record>>>>;

pub(crate) struct TraceLog {
    sender: SharedSender,
    worker: Option<JoinHandle<()>>,
}

pub(crate) struct TraceSubscriber {
    sender: SharedSender,
    request_id: String,
}

impl TraceLog {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "trace path is not a regular file",
            ));
        }
        if metadata.uid() != unsafe { libc::geteuid() } {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "trace file must be owned by the current user",
            ));
        }
        if metadata.permissions().mode() & 0o077 != 0 {
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        }
        let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFL, flags & !libc::O_NONBLOCK) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Self::with_writer(BufWriter::new(file))
    }

    fn with_writer(writer: impl Write + Send + 'static) -> io::Result<Self> {
        let (sender, receiver) = sync_channel::<Record>(TRACE_QUEUE_CAPACITY);
        let sender = Arc::new(Mutex::new(Some(sender)));
        let worker_sender = Arc::clone(&sender);
        let worker = std::thread::Builder::new()
            .name("qwen-sse-trace".into())
            .spawn(move || {
                let mut writer = writer;
                for record in receiver {
                    let result = (|| {
                        serde_json::to_writer(&mut writer, &record)
                            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                        writer.write_all(b"\n")?;
                        writer.flush()
                    })();
                    if let Err(error) = result {
                        worker_sender.lock().unwrap_or_else(|error| error.into_inner()).take();
                        tracing::warn!(target: "qwen_diag", "serve: disabling SSE trace after write failure: {error}");
                        break;
                    }
                }
            })?;
        Ok(Self {
            sender,
            worker: Some(worker),
        })
    }

    pub(crate) fn subscriber(&self) -> TraceSubscriber {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        TraceSubscriber {
            sender: Arc::clone(&self.sender),
            request_id: format!(
                "trace_{:x}_{now:x}_{:x}",
                std::process::id(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ),
        }
    }
}

impl TraceSubscriber {
    pub(crate) fn is_enabled(&self) -> bool {
        self.sender.lock().is_ok_and(|sender| sender.is_some())
    }

    pub(crate) fn line(&self, make_value: impl FnOnce() -> Value) {
        if !self.is_enabled() {
            return;
        }
        // Payload construction and serialization never hold the sender mutex.
        let record = Record {
            trace_request_id: self.request_id.clone(),
            value: make_value(),
        };
        let Ok(mut sender) = self.sender.lock() else {
            return;
        };
        let Some(channel) = sender.as_ref() else {
            return;
        };
        let result = channel.try_send(record);
        if result.is_err() {
            sender.take();
        }
        drop(sender);
        match result {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                tracing::warn!(target: "qwen_diag", "serve: disabling SSE trace because its bounded queue is full");
            }
            Err(TrySendError::Disconnected(_)) => {
                tracing::warn!(target: "qwen_diag", "serve: disabling SSE trace because its writer stopped");
            }
        }
    }
}

fn join_trace_worker_with_grace(
    worker: JoinHandle<()>,
    grace: Duration,
) -> Option<std::thread::Result<()>> {
    let started = Instant::now();
    while !worker.is_finished() {
        let remaining = grace.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return None;
        }
        std::thread::sleep(remaining.min(TRACE_SHUTDOWN_POLL_INTERVAL));
    }
    Some(worker.join())
}

impl Drop for TraceLog {
    fn drop(&mut self) {
        self.sender
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        let Some(worker) = self.worker.take() else {
            return;
        };
        match join_trace_worker_with_grace(worker, TRACE_SHUTDOWN_GRACE) {
            Some(Ok(())) => {}
            Some(Err(_)) => {
                tracing::warn!(target: "qwen_diag", "serve: SSE trace writer panicked");
            }
            None => {
                tracing::warn!(target: "qwen_diag", "serve: SSE trace writer did not stop within {} ms; detaching so shutdown can continue (queued trace events may be lost)", TRACE_SHUTDOWN_GRACE.as_millis());
            }
        }
    }
}

#[cfg(test)]
mod tests;
