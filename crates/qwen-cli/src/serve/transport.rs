//! One admitted HTTP worker communicates with the resident owner. This separates
//! ownership, not admission policy: no new connection overlaps response cleanup.

use super::http::{
    self, BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink, PreparedResponse,
};
use super::items::{ServeError, ServeRequest};
use super::owner_activity::ActivityGuard;
use super::request_profile::RequestProfile;
use super::trace::TraceSubscriber;
use anyhow::{Context, Result};
use std::io;
use std::net::{Shutdown, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::thread::{JoinHandle, Thread};
use std::time::Duration;

const POLL: Duration = Duration::from_millis(10);
const CHUNK_BYTES: usize = 4096;
const PIECE_CAPACITY: usize = 2;
const WORKER_STACK_BYTES: usize = 2 * 1024 * 1024;
// Two queued chunks, one producer chunk, one consumer chunk. Double their
// payload size for allocator slack and small channel/control metadata. Request
// storage is shared, not copied. Reserve the worker's complete configured stack
// too: pages not yet touched may be absent from the process memory signal.
const BUFFER_RESERVE_BYTES: u64 =
    (WORKER_STACK_BYTES + (PIECE_CAPACITY + 2) * CHUNK_BYTES * 2) as u64;

struct Control {
    cancelled: AtomicBool,
    owner: Thread,
}

impl Control {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.owner.unpark();
    }

    fn checkpoint(&self) -> io::Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(aborted("HTTP subscriber stopped"));
        }
        crate::shutdown::checkpoint().map_err(|error| aborted(error.to_string()))
    }
}

fn aborted(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::ConnectionAborted, message.into())
}

struct CancelOnDrop(Arc<Control>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

struct Work {
    prepared: Arc<PreparedResponse>,
    pieces: SyncSender<Piece>,
    processed: Receiver<()>,
    terminal: SyncSender<Result<GenerationOutcome, BackendFailure>>,
    control: Arc<Control>,
    _activity: Arc<ActivityGuard>,
}

struct OwnerSink {
    pieces: SyncSender<Piece>,
    processed: Receiver<()>,
    control: Arc<Control>,
}

struct Piece {
    bytes: Vec<u8>,
    new_piece_bytes: Option<usize>,
    last: bool,
}

impl OwnerSink {
    fn send_chunk(&mut self, mut pending: Piece, mut wait: impl FnMut()) -> io::Result<()> {
        loop {
            self.tick()?;
            match self.pieces.try_send(pending) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Disconnected(_)) => {
                    self.control.cancel();
                    return Err(aborted("HTTP piece receiver stopped"));
                }
                Err(TrySendError::Full(bytes)) => {
                    pending = bytes;
                    #[cfg(test)]
                    super::signal_tests::bridge_wait_observed();
                    wait();
                }
            }
        }
    }
}

impl GenerationSink for OwnerSink {
    fn transport_reserve_bytes(&self) -> u64 {
        BUFFER_RESERVE_BYTES
    }

    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.tick()?;
        let chunks = bytes.chunks(CHUNK_BYTES);
        let count = chunks.len();
        for (index, chunk) in chunks.enumerate() {
            // Receiving a piece or cancelling unparks the owner; the timeout
            // also observes process shutdown signals.
            self.send_chunk(
                Piece {
                    bytes: chunk.to_vec(),
                    new_piece_bytes: (index == 0).then_some(bytes.len()),
                    last: index + 1 == count,
                },
                || std::thread::park_timeout(POLL),
            )?;
        }
        if count > 0 {
            loop {
                self.tick()?;
                match self.processed.recv_timeout(POLL) {
                    Ok(()) => break,
                    Err(RecvTimeoutError::Timeout) => {
                        #[cfg(test)]
                        super::signal_tests::bridge_wait_observed();
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        return Err(aborted("HTTP piece processing stopped"));
                    }
                }
            }
        }
        Ok(())
    }

    fn tick(&mut self) -> io::Result<()> {
        self.control.checkpoint()
    }
}

struct HttpProxy {
    model_id: String,
    profile: RequestProfile,
    work: SyncSender<Work>,
    control: Arc<Control>,
    activity: Arc<ActivityGuard>,
}

impl GenerationBackend for HttpProxy {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn request_profile(&self) -> RequestProfile {
        self.profile.clone()
    }

    fn generate(
        &mut self,
        _request: &ServeRequest,
        _prompt: &str,
        _sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        Err(ServeError::server_error("HTTP transport requires owned preparation").into())
    }

    fn generate_prepared(
        &mut self,
        prepared: Arc<PreparedResponse>,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        self.control.checkpoint().map_err(BackendFailure::Aborted)?;
        let (pieces, incoming) = sync_channel(PIECE_CAPACITY);
        let (terminal, outcome) = sync_channel(1);
        let (acknowledge, processed) = sync_channel(1);
        self.work
            .try_send(Work {
                prepared,
                pieces,
                processed,
                terminal,
                control: Arc::clone(&self.control),
                _activity: Arc::clone(&self.activity),
            })
            .map_err(|_| ServeError::server_error("HTTP model owner unavailable"))?;
        loop {
            self.control.checkpoint().map_err(BackendFailure::Aborted)?;
            match incoming.recv_timeout(POLL) {
                Ok(piece) => {
                    self.control.owner.unpark();
                    sink.piece_fragment(&piece.bytes, piece.new_piece_bytes)
                        .map_err(BackendFailure::Aborted)?;
                    if piece.last {
                        acknowledge.try_send(()).map_err(|_| {
                            BackendFailure::Aborted(aborted(
                                "model owner stopped before piece acknowledgement",
                            ))
                        })?;
                    }
                }
                Err(RecvTimeoutError::Timeout) => sink.tick().map_err(BackendFailure::Aborted)?,
                Err(RecvTimeoutError::Disconnected) => {
                    return outcome.try_recv().unwrap_or_else(|_| {
                        Err(ServeError::server_error("model owner stopped without outcome").into())
                    });
                }
            }
        }
    }
}

struct Worker {
    socket: TcpStream,
    control: Arc<Control>,
    thread: Option<JoinHandle<io::Result<()>>>,
}

impl Worker {
    fn stop(&self) {
        self.control.cancel();
        let _ = self.socket.shutdown(Shutdown::Both);
    }

    fn join(&mut self) -> Result<()> {
        let result = self
            .thread
            .take()
            .expect("worker joined once")
            .join()
            .map_err(|_| anyhow::anyhow!("HTTP request worker panicked"))?;
        if let Err(error) = result {
            tracing::info!(target: "qwen_diag", "serve: connection aborted: {error}");
        }
        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if self.thread.is_some() {
            self.stop();
            let _ = self.join();
        }
    }
}

fn execute(work: Work, backend: &mut dyn GenerationBackend) {
    let mut sink = OwnerSink {
        pieces: work.pieces,
        processed: work.processed,
        control: work.control,
    };
    let result = match sink.tick() {
        Ok(()) => backend.generate_prepared(work.prepared, &mut sink),
        Err(error) => Err(BackendFailure::Aborted(error)),
    };
    let _ = work.terminal.try_send(result);
    // Closing the piece stream follows publishing the terminal. The subscriber
    // drains prior pieces before inspecting it, including on generation errors.
    drop(sink);
}

pub(super) fn handle_connection(
    stream: TcpStream,
    backend: &mut dyn GenerationBackend,
    trace: Option<TraceSubscriber>,
    guard: ActivityGuard,
    mut checkpoint: impl FnMut() -> Result<()>,
) -> Result<()> {
    let socket = stream.try_clone().context("clone HTTP shutdown handle")?;
    let control = Arc::new(Control {
        cancelled: AtomicBool::new(false),
        owner: std::thread::current(),
    });
    let (sender, incoming): (_, Receiver<Work>) = sync_channel(1);
    let mut proxy = HttpProxy {
        model_id: backend.model_id().to_owned(),
        profile: backend.request_profile(),
        work: sender,
        control: Arc::clone(&control),
        activity: Arc::new(guard),
    };
    let cancel = CancelOnDrop(Arc::clone(&control));
    let thread = std::thread::Builder::new()
        .name("qwen-http-request".into())
        .stack_size(WORKER_STACK_BYTES)
        .spawn(move || {
            let _cancel = cancel;
            http::handle_connection(&stream, &mut proxy, trace)
        })
        .context("spawn HTTP request worker")?;
    let mut worker = Worker {
        socket,
        control,
        thread: Some(thread),
    };
    let result = (|| {
        loop {
            checkpoint()?;
            match incoming.recv_timeout(POLL) {
                Ok(work) => {
                    checkpoint()?;
                    execute(work, backend);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        Ok(())
    })();
    // Drop queued work and close producers before joining a waiting subscriber.
    drop(incoming);
    if result.is_err() {
        worker.stop();
    }
    worker.join()?;
    result
}

#[cfg(test)]
mod tests;
