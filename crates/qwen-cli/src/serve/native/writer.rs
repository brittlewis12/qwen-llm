//! Bounded CPU publication, owned and joined by the admitted execution.

use super::{Outcome, Prepared, RECORD_BYTES};
use crate::ordinary_executor::ExecutionControl;
use crate::serve::jobs::{
    state::{Counters, JobError, Phase},
    store::JobStore,
};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::json;
use std::io::{self, Write};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
    mpsc::{Receiver, SyncSender, sync_channel},
};
use std::thread::JoinHandle;

const EVENTS: usize = 128;
const QUEUE_BYTES: usize = 8 * RECORD_BYTES;
const STACK_BYTES: usize = 2 * 1024 * 1024;
pub(crate) const CPU_UPPER_BYTES: u64 = (QUEUE_BYTES + 72 * RECORD_BYTES + STACK_BYTES) as u64;

pub(super) fn error(code: &str, message: &str) -> JobError {
    JobError {
        r#type: "server_error".into(),
        code: code.into(),
        param: None,
        message: message.into(),
    }
}

pub(super) fn encode(value: &impl Serialize) -> Result<Vec<u8>> {
    struct Bounded(Vec<u8>);
    impl Write for Bounded {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > (RECORD_BYTES - 128).saturating_sub(self.0.len()) {
                return Err(io::Error::other("native record exceeds byte budget"));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut output = Bounded(Vec::new());
    serde_json::to_writer(&mut output, value).context("encode bounded native record")?;
    Ok(output.0)
}

#[derive(Default)]
struct Budget {
    used: AtomicUsize,
    peak: AtomicUsize,
}
struct Bytes {
    bytes: usize,
    budget: Arc<Budget>,
}
impl Drop for Bytes {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

enum Event {
    Record(Vec<u8>, Bytes, Phase, Counters),
    Progress(Phase, Counters),
}
const _: () = assert!(EVENTS * std::mem::size_of::<Event>() < RECORD_BYTES);

pub(crate) struct Sink {
    pub(crate) control: ExecutionControl,
    pub(in crate::serve) server: crate::serve::control::ExecutionGate,
    sender: SyncSender<Event>,
    failure: Arc<Mutex<Option<JobError>>>,
    budget: Arc<Budget>,
}
impl Sink {
    pub(super) fn fail_recording(&self, code: &str, message: &str) {
        self.failure
            .lock()
            .unwrap()
            .get_or_insert_with(|| error(code, message));
        self.control.cancel();
    }
    fn send(&self, event: Event) {
        if self.sender.try_send(event).is_err() {
            self.fail_recording(
                "artifact_backpressure",
                "The bounded artifact writer stopped or fell behind; publication is incomplete.",
            );
        }
    }
    pub(super) fn record(&self, record: impl Serialize, phase: Phase, counters: &Counters) {
        let bytes = match encode(&record) {
            Ok(bytes) => bytes,
            Err(cause) => {
                tracing::error!("native record encoding: {cause:#}");
                self.fail_recording(
                    "artifact_record_too_large",
                    "Native record exceeded its byte budget.",
                );
                return;
            }
        };
        let amount = bytes.capacity();
        let Ok(previous) =
            self.budget
                .used
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                    used.checked_add(amount)
                        .filter(|total| *total <= QUEUE_BYTES)
                })
        else {
            self.fail_recording(
                "artifact_backpressure",
                "Retained record byte queue is full.",
            );
            return;
        };
        self.budget
            .peak
            .fetch_max(previous + amount, Ordering::AcqRel);
        self.send(Event::Record(
            bytes,
            Bytes {
                bytes: amount,
                budget: Arc::clone(&self.budget),
            },
            phase,
            counters.clone(),
        ));
    }
    pub(super) fn progress(&self, phase: Phase, counters: &Counters) {
        self.send(Event::Progress(phase, counters.clone()));
    }
}

pub(super) struct Writer {
    sink: Option<Sink>,
    terminal: Option<SyncSender<Outcome>>,
    ready: Receiver<Result<Readiness>>,
    worker: Option<JoinHandle<Result<()>>>,
}
pub(super) enum Readiness {
    Execute,
    Settled,
}
impl Writer {
    pub(super) fn spawn(
        store: Arc<JobStore>,
        id: String,
        control: ExecutionControl,
        prepared: &Prepared,
        server: crate::serve::control::ExecutionGate,
    ) -> Result<Self> {
        let record = prepared.record.clone();
        ensure!(
            record.len() <= RECORD_BYTES,
            "prepared record exceeds limit"
        );
        let initial = Counters {
            prompt_tokens: prepared.prompt.len() as u64,
            ..Counters::default()
        };
        let (sender, incoming) = sync_channel(EVENTS);
        let (terminal, outcome) = sync_channel::<Outcome>(1);
        let (started, ready) = sync_channel(1);
        let failure: Arc<Mutex<Option<JobError>>> = Arc::default();
        let worker_failure = Arc::clone(&failure);
        let worker_control = control.clone();
        let budget = Arc::new(Budget::default());
        let worker_budget = Arc::clone(&budget);
        let worker = std::thread::Builder::new().name("qwen-native-artifacts".into()).stack_size(STACK_BYTES).spawn(move || {
            let setup = (|| -> Result<Readiness> {
                if store.start(&id, initial.prompt_tokens)? == crate::serve::jobs::store::Start::Settled {
                    return Ok(Readiness::Settled);
                }
                store.append(&id, &[serde_json::from_slice(&record)?])?;
                Ok(Readiness::Execute)
            })();
            if matches!(setup, Ok(Readiness::Settled)) {
                let _ = started.send(Ok(Readiness::Settled));
                return Ok(());
            }
            if let Err(cause) = setup {
                if let Ok(status) = store.status(&id) {
                    if status.state.terminal() {
                        let _ = started.send(Ok(Readiness::Settled));
                        return Ok(());
                    }
                    let error = error("artifact_prepare_failed", "Native publication setup failed before model execution.");
                    if store.finish_generation(&id, crate::serve::jobs::state::StopReason::ExecutionError, status.generation.counters, Some(error.clone())).is_ok() {
                        let _ = store.finalize(&id, Some(error));
                    }
                }
                let message = format!("prepare native publication: {cause:#}");
                let _ = started.send(Err(anyhow::anyhow!(message.clone())));
                return Err(anyhow::anyhow!(message));
            }
            let _ = started.send(Ok(Readiness::Execute));
            let mut last = initial;
            let mut disk_failed = false;
            for event in incoming {
                if disk_failed { continue; }
                let result = (|| -> Result<()> {
                    let (phase, counters) = match event {
                        Event::Record(bytes, _permit, phase, counters) => {
                            store.append(&id, &[serde_json::from_slice(&bytes)?])?;
                            (phase, counters)
                        }
                        Event::Progress(phase, counters) => (phase, counters),
                    };
                    store.progress(&id, phase, counters.clone())?;
                    last = counters;
                    Ok(())
                })();
                if let Err(cause) = result {
                    tracing::error!(job_id = id, "native publication: {cause:#}");
                    worker_failure.lock().unwrap().get_or_insert_with(|| error("artifact_write_failed", "Native result publication failed; generation outcome is tracked separately."));
                    worker_control.cancel();
                    disk_failed = true;
                }
            }
            let outcome = outcome.recv().unwrap_or_else(|_| Outcome::interrupted(last));
            store.finish_generation(&id, outcome.reason, outcome.counters.clone(), outcome.error.clone())?;
            if !disk_failed {
                let terminal = json!({"kind":"generation_terminal","state":outcome.state_name(),"stop_reason":outcome.reason,
                    "sampled_tokens":outcome.counters.sampled_tokens,"consumed_generated_tokens":outcome.counters.consumed_generated_tokens,
                    "error":outcome.error,"artifact_writer":{"peak_record_bytes":worker_budget.peak.load(Ordering::Acquire),"peak_array_bytes":0},
                    "cost":{"wall_ms":outcome.wall_ms,"prefill_ms":null,"decode_ms":null,"readout_ms":null}});
                if let Err(cause) = store.append(&id, &[terminal]) {
                    tracing::error!(job_id = id, "native terminal record: {cause:#}");
                    worker_failure.lock().unwrap().get_or_insert_with(|| error("artifact_write_failed", "Terminal record publication failed."));
                }
            }
            store.finalize(&id, worker_failure.lock().unwrap().clone())?;
            Ok(())
        }).context("spawn native artifact writer")?;
        Ok(Self {
            sink: Some(Sink {
                control,
                server,
                sender,
                failure,
                budget,
            }),
            terminal: Some(terminal),
            ready,
            worker: Some(worker),
        })
    }
    pub(super) fn sink(&self) -> &Sink {
        self.sink.as_ref().expect("writer not settled")
    }
    pub(super) fn wait_ready(&self) -> Result<Readiness> {
        self.ready
            .recv()
            .context("native writer stopped before readiness")?
    }
    pub(super) fn join_settled(mut self) -> Result<()> {
        self.sink.take();
        self.terminal.take();
        self.join()
    }
    pub(super) fn finish(mut self, outcome: Outcome) -> Result<()> {
        let result = self
            .terminal
            .take()
            .expect("terminal available")
            .try_send(outcome)
            .map_err(|_| anyhow::anyhow!("native writer stopped before outcome"));
        self.sink.take();
        self.join()?;
        result
    }
    fn join(&mut self) -> Result<()> {
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("native writer panicked"))??;
        }
        Ok(())
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        self.sink.take();
        self.terminal.take();
        if let Err(cause) = self.join() {
            tracing::error!("settle native writer: {cause:#}");
        }
    }
}

#[cfg(test)]
mod tests;
