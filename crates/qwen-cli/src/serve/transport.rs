//! One admitted HTTP worker communicates with the resident owner. This separates
//! ownership, not admission policy: no new connection overlaps response cleanup.

use super::http::{
    self, BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink, PreparedResponse,
};
use super::items::{ServeError, ServeRequest};
use super::owner_activity::ActivityGuard;
use super::request_profile::RequestProfile;
use super::trace::TraceSubscriber;
use crate::ordinary_executor::ExecutionControl;
use anyhow::{Context, Result};
use std::io;
use std::net::{Shutdown, TcpStream};
use std::sync::Arc;
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
pub(super) const BUFFER_RESERVE_BYTES: u64 =
    (WORKER_STACK_BYTES + (PIECE_CAPACITY + 2) * CHUNK_BYTES * 2) as u64;

struct Control {
    execution: ExecutionControl,
    owner: Thread,
    server: Option<super::control::ExecutionGate>,
}

impl Control {
    fn cancel(&self) {
        self.execution.cancel();
        self.owner.unpark();
    }

    fn checkpoint(&self) -> io::Result<()> {
        if let Some(server) = &self.server {
            server
                .checkpoint()
                .map_err(|cause| aborted(cause.to_string()))?;
        }
        self.execution
            .checkpoint()
            .map_err(|_| aborted("HTTP subscriber stopped"))?;
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
    extra_cpu_reserve: u64,
}

struct OwnerSink {
    pieces: SyncSender<Piece>,
    processed: Receiver<()>,
    control: Arc<Control>,
    extra_cpu_reserve: u64,
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
        BUFFER_RESERVE_BYTES + self.extra_cpu_reserve
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
    extra_cpu_reserve: u64,
}

impl GenerationBackend for HttpProxy {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn request_profile(&self) -> RequestProfile {
        self.profile.clone()
    }

    /// Worker-side failures after generation (partition) reach the owner
    /// with this connection's completion.
    fn request_failed_on_server(&mut self) {
        self.activity.mark_server_failure();
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
                extra_cpu_reserve: self.extra_cpu_reserve,
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

#[derive(Debug)]
pub(super) struct WorkerPanicked;
impl std::fmt::Display for WorkerPanicked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HTTP request worker panicked")
    }
}
impl std::error::Error for WorkerPanicked {}

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
            .map_err(|_| WorkerPanicked)?;
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
        extra_cpu_reserve: work.extra_cpu_reserve,
    };
    let result = match sink.tick() {
        Ok(()) => backend.generate_prepared(work.prepared, &mut sink),
        Err(error) => Err(BackendFailure::Aborted(error)),
    };
    if let Err(BackendFailure::Serve(error)) = &result
        && error.status >= 500
    {
        backend.request_failed_on_server();
    }
    let _ = work.terminal.try_send(result);
    // Closing the piece stream follows publishing the terminal. The subscriber
    // drains prior pieces before inspecting it, including on generation errors.
    drop(sink);
}

pub(super) struct Connection {
    incoming: Option<Receiver<Work>>,
    worker: Worker,
    _execution: Option<super::control::ExecutionPermit>,
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq)]
pub(super) enum StartFault {
    Clone,
    Spawn,
    WorkerPanic,
}
#[cfg(test)]
std::thread_local! { static START_FAULT: std::cell::Cell<Option<StartFault>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub(super) fn inject_start_fault(fault: StartFault) {
    START_FAULT.set(Some(fault));
}

impl Connection {
    pub(super) fn start(
        stream: TcpStream,
        backend: &dyn GenerationBackend,
        trace: Option<TraceSubscriber>,
        guard: ActivityGuard,
    ) -> Result<Self> {
        Self::spawn(
            stream,
            backend.model_id().to_owned(),
            backend.request_profile(),
            trace,
            guard,
            None,
            None,
            0,
        )
    }

    pub(super) fn prepared(
        stream: TcpStream,
        model_id: String,
        profile: RequestProfile,
        trace: Option<TraceSubscriber>,
        guard: ActivityGuard,
        request: http::HttpRequest,
        execution: super::control::ExecutionPermit,
        extra_cpu_reserve: u64,
    ) -> Result<Self> {
        Self::spawn(
            stream,
            model_id,
            profile,
            trace,
            guard,
            Some(request),
            Some(execution),
            extra_cpu_reserve,
        )
    }

    fn spawn(
        stream: TcpStream,
        model_id: String,
        profile: RequestProfile,
        trace: Option<TraceSubscriber>,
        guard: ActivityGuard,
        request: Option<http::HttpRequest>,
        execution: Option<super::control::ExecutionPermit>,
        extra_cpu_reserve: u64,
    ) -> Result<Self> {
        #[cfg(test)]
        let fault = START_FAULT.take();
        #[cfg(test)]
        anyhow::ensure!(
            fault != Some(StartFault::Clone),
            "injected socket clone failure"
        );
        let socket = stream.try_clone().context("clone HTTP shutdown handle")?;
        let control = Arc::new(Control {
            execution: ExecutionControl::default(),
            server: execution
                .as_ref()
                .map(super::control::ExecutionPermit::gate),
            owner: execution
                .as_ref()
                .map_or_else(std::thread::current, super::control::ExecutionPermit::owner),
        });
        let (sender, incoming) = sync_channel(1);
        let mut proxy = HttpProxy {
            model_id,
            profile,
            work: sender,
            control: Arc::clone(&control),
            activity: Arc::new(guard),
            extra_cpu_reserve,
        };
        let cancel = CancelOnDrop(Arc::clone(&control));
        #[cfg(test)]
        anyhow::ensure!(
            fault != Some(StartFault::Spawn),
            "injected thread spawn failure"
        );
        let thread = std::thread::Builder::new()
            .name("qwen-http-request".into())
            .stack_size(WORKER_STACK_BYTES)
            .spawn(move || {
                let _cancel = cancel;
                #[cfg(test)]
                assert!(
                    fault != Some(StartFault::WorkerPanic),
                    "injected HTTP worker panic"
                );
                match request {
                    Some(request) => http::handle_request(&stream, &mut proxy, trace, request),
                    None => http::handle_connection(&stream, &mut proxy, trace),
                }
            })
            .context("spawn HTTP request worker")?;
        Ok(Self {
            incoming: Some(incoming),
            _execution: execution,
            worker: Worker {
                socket,
                control,
                thread: Some(thread),
            },
        })
    }

    /// Advance only from the resident owner; true means the CPU worker is joined.
    pub(super) fn advance(
        &mut self,
        backend: &mut dyn GenerationBackend,
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<bool> {
        checkpoint()?;
        match self
            .incoming
            .as_ref()
            .expect("connection not settled")
            .recv_timeout(POLL)
        {
            Ok(work) => {
                checkpoint()?;
                execute(work, backend);
                Ok(false)
            }
            Err(RecvTimeoutError::Timeout) => Ok(false),
            Err(RecvTimeoutError::Disconnected) => {
                self.incoming.take();
                self.worker.join()?;
                Ok(true)
            }
        }
    }

    pub(super) fn stop_and_join(mut self) -> Result<()> {
        self.incoming.take();
        if self.worker.thread.is_some() {
            self.worker.stop();
            self.worker.join()?;
        }
        Ok(())
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        // Release queued work before Worker stops sockets and joins the producer.
        self.incoming.take();
    }
}

#[cfg(test)]
pub(super) fn handle_connection(
    stream: TcpStream,
    backend: &mut dyn GenerationBackend,
    trace: Option<TraceSubscriber>,
    guard: ActivityGuard,
    mut checkpoint: impl FnMut() -> Result<()>,
) -> Result<()> {
    let mut connection = Connection::start(stream, backend, trace, guard)?;
    let result = (|| {
        while !connection.advance(backend, &mut checkpoint)? {}
        Ok(())
    })();
    connection.stop_and_join()?;
    result
}

#[cfg(test)]
mod tests;
