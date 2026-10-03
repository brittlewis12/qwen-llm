//! Bounded same-port control traffic; one execution reservation covers both APIs.

use super::{
    http, lens_http::LensApi, native, owner_activity, request_profile::RequestProfile,
    trace::TraceFactory, transport,
};
use anyhow::{Context, Result};
use std::io::{self, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc::SyncSender,
};
use std::thread::JoinHandle;
use std::time::Duration;

#[cfg(test)]
mod tests;

const MAX_WORKERS: usize = 2;
const STACK_BYTES: usize = 2 * 1024 * 1024;
// Standing future allowance for both control slots: bounded request/result Value
// trees, input spans/tokenization, serialization, read watchdogs and worker stacks.
// An ordinary request owns the sole execution slot before body allocation, so
// another ordinary preparation cannot grow behind an executing native job.
pub(super) const CPU_RESERVE_BYTES: u64 = MAX_WORKERS as u64 * 256 * 1024 * 1024;

#[derive(Default)]
struct GateState {
    busy: bool,
    closed: bool,
}
#[derive(Clone)]
pub(super) struct ExecutionGate(Arc<Mutex<GateState>>, std::thread::Thread);
impl Default for ExecutionGate {
    fn default() -> Self {
        Self(Arc::default(), std::thread::current())
    }
}
pub(super) struct ExecutionPermit {
    gate: ExecutionGate,
}

#[derive(Debug)]
pub(crate) struct ServerStopped;
impl std::fmt::Display for ServerStopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HTTP control service stopped")
    }
}
impl std::error::Error for ServerStopped {}
impl ExecutionGate {
    pub(super) fn checkpoint(&self) -> Result<(), ServerStopped> {
        let state = self.0.lock().map_err(|_| ServerStopped)?;
        if state.closed {
            Err(ServerStopped)
        } else {
            Ok(())
        }
    }
    pub(super) fn reserve(&self) -> Result<ExecutionPermit, &'static str> {
        let mut state = self.0.lock().map_err(|_| "execution admission failed")?;
        if state.closed {
            return Err("model queue is stopping");
        }
        if state.busy {
            return Err("model queue is full");
        }
        state.busy = true;
        Ok(ExecutionPermit { gate: self.clone() })
    }
    pub(super) fn close(&self) {
        self.0
            .lock()
            .unwrap_or_else(|cause| cause.into_inner())
            .closed = true;
    }
    pub(super) fn deliver(
        &self,
        sender: &SyncSender<Event>,
        event: Event,
    ) -> Result<(), &'static str> {
        let state = self.0.lock().map_err(|_| "execution admission failed")?;
        if state.closed {
            drop(state);
            drop(event);
            return Err("model queue is stopping");
        }
        let result = sender.try_send(event);
        drop(state);
        result.map_err(|_| "model owner stopped before delivery")
    }
}
impl ExecutionPermit {
    pub(super) fn gate(&self) -> ExecutionGate {
        self.gate.clone()
    }
    pub(super) fn owner(&self) -> std::thread::Thread {
        self.gate.1.clone()
    }
}
impl Drop for ExecutionPermit {
    fn drop(&mut self) {
        self.gate
            .0
            .lock()
            .unwrap_or_else(|cause| cause.into_inner())
            .busy = false;
    }
}

pub(super) enum Event {
    Incoming(TcpStream),
    Prepared(transport::Connection),
    Native(native::Queued),
}

pub(super) struct Profile {
    pub(super) model_id: String,
    pub(super) request: RequestProfile,
    pub(super) lens: Arc<LensApi>,
    pub(super) gate: ExecutionGate,
    pub(super) activity: owner_activity::Admission,
    pub(super) sender: SyncSender<Event>,
    pub(super) trace: Option<TraceFactory>,
}

struct Worker {
    socket: TcpStream,
    thread: Option<JoinHandle<io::Result<()>>>,
}
impl Worker {
    fn stop(&self) {
        let _ = self.socket.shutdown(Shutdown::Both);
    }
    fn join(&mut self) -> Result<()> {
        if let Some(thread) = self.thread.take() {
            if let Err(cause) = thread
                .join()
                .map_err(|_| anyhow::anyhow!("HTTP control worker panicked"))?
            {
                tracing::info!("HTTP control connection aborted: {cause}");
            }
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

#[derive(Debug)]
enum Refused {
    Busy,
    TooLarge,
    GetBody,
}
impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Busy => "server is processing another request",
            Self::TooLarge => "Lens request exceeds advertised body limit",
            Self::GetBody => "Lens GET does not accept a body",
        })
    }
}
impl std::error::Error for Refused {}

fn error_reply(mut stream: &TcpStream, status: u16, message: &str) -> io::Result<()> {
    let body = serde_json::to_vec(
        &serde_json::json!({"error":{"type":"invalid_request_error","code":"invalid_request","message":message}}),
    )?;
    let reason = match status {
        408 => "Request Timeout",
        413 => "Content Too Large",
        _ => "Bad Request",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)
}

pub(super) fn handle(
    stream: TcpStream,
    profile: &Profile,
    mut activity: Option<owner_activity::ActivityGuard>,
) -> io::Result<()> {
    let mut execution = None;
    let request =
        http::read_http_request_with_admission(&stream, Duration::from_secs(30), |head, length| {
            if LensApi::matches(&head.path) {
                if length > profile.lens.request_byte_limit() {
                    return Err(io::Error::other(Refused::TooLarge));
                }
                if head.method == "GET" && length != 0 {
                    return Err(io::Error::other(Refused::GetBody));
                }
                if profile.lens.is_read_only(head) {
                    activity.take().expect("HTTP activity").release_read_only();
                }
            } else {
                execution = Some(
                    profile
                        .gate
                        .reserve()
                        .map_err(|_| io::Error::other(Refused::Busy))?,
                );
            }
            Ok(())
        });
    let request = match request {
        Ok(Some(request)) => request,
        Ok(None) => return Ok(()),
        Err(cause) => {
            if matches!(
                cause
                    .get_ref()
                    .and_then(|cause| cause.downcast_ref::<Refused>()),
                Some(Refused::Busy)
            ) {
                return http::write_busy_response(&stream);
            }
            let status = if matches!(
                cause
                    .get_ref()
                    .and_then(|cause| cause.downcast_ref::<Refused>()),
                Some(Refused::TooLarge)
            ) {
                413
            } else if cause.kind() == io::ErrorKind::TimedOut {
                408
            } else {
                400
            };
            return error_reply(&stream, status, &cause.to_string());
        }
    };
    if LensApi::matches(&request.path) {
        profile.lens.handle(&request, &stream)?;
        return Ok(());
    }
    let trace = profile.trace.as_ref().map(TraceFactory::subscriber);
    let connection = transport::Connection::prepared(
        stream,
        profile.model_id.clone(),
        profile.request.clone(),
        trace,
        activity.take().expect("ordinary HTTP activity"),
        request,
        execution.take().expect("ordinary execution reservation"),
    )
    .map_err(io::Error::other)?;
    profile
        .gate
        .deliver(&profile.sender, Event::Prepared(connection))
        .map_err(io::Error::other)
}

pub(super) fn admit_memory() -> Result<()> {
    super::transport_memory::admit_resident_transport(
        CPU_RESERVE_BYTES,
        qwen_llm::metal::MetalContext::process_limit_bytes_remaining(),
    )
    .map_err(|cause| anyhow::anyhow!("{}", cause.message))
}

pub(super) fn spawn(
    listener: TcpListener,
    profile: Profile,
    stopping: Arc<AtomicBool>,
) -> Result<JoinHandle<Result<()>>> {
    admit_memory()?;
    listener.set_nonblocking(true)?;
    let profile = Arc::new(profile);
    std::thread::Builder::new()
        .name("qwen-http-control".into())
        .spawn(move || {
            let mut workers: Vec<Worker> = Vec::new();
            let result = (|| -> Result<()> {
                while !stopping.load(Ordering::Acquire) {
                    let mut index = 0;
                    while index < workers.len() {
                        if workers[index]
                            .thread
                            .as_ref()
                            .is_some_and(JoinHandle::is_finished)
                        {
                            workers.swap_remove(index).join()?;
                        } else {
                            index += 1;
                        }
                    }
                    match listener.accept() {
                        Ok((stream, _)) => {
                            if let Err(cause) = http::configure_stream(&stream) {
                                tracing::info!("HTTP control socket configuration failed: {cause}");
                                continue;
                            }
                            if workers.len() >= MAX_WORKERS || admit_memory().is_err() {
                                let _ = http::write_busy_response(&stream);
                                continue;
                            }
                            let Some(activity) = profile.activity.try_admit() else {
                                break;
                            };
                            let socket = stream.try_clone()?;
                            let profile = Arc::clone(&profile);
                            let thread = std::thread::Builder::new()
                                .name("qwen-http-control-request".into())
                                .stack_size(STACK_BYTES)
                                .spawn(move || handle(stream, &profile, Some(activity)))?;
                            workers.push(Worker {
                                socket,
                                thread: Some(thread),
                            });
                        }
                        Err(cause) if cause.kind() == io::ErrorKind::WouldBlock => {
                            super::wait_for_connection(&listener)?
                        }
                        Err(cause) => return Err(cause.into()),
                    }
                }
                Ok(())
            })();
            stopping.store(true, Ordering::Release);
            profile.gate.close();
            for worker in &workers {
                worker.stop();
            }
            let mut join_error = None;
            for mut worker in workers {
                if let Err(cause) = worker.join() {
                    join_error.get_or_insert(cause);
                }
            }
            if let Some(cause) = join_error {
                return Err(cause);
            }
            result
        })
        .context("spawn bounded HTTP control acceptor")
}
