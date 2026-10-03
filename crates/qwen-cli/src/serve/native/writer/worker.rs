//! FIFO publication with bounded metadata batches and explicit array barriers.

use super::*;
use crate::serve::jobs::store::{Start, check_progress};
use serde_json::json;

pub(super) struct State {
    pub(super) store: Arc<JobStore>,
    pub(super) id: String,
    pub(super) control: ExecutionControl,
    pub(super) server: crate::serve::control::ExecutionGate,
    pub(super) failure: Arc<Mutex<Option<JobError>>>,
    pub(super) budget: Arc<Budget>,
    pub(super) array_budget: Arc<Budget>,
    pub(super) waits: Arc<AtomicUsize>,
}

impl State {
    fn fail(&self, message: &str) {
        self.store
            .publication_failed(&self.id, error("artifact_write_failed", message));
        self.failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(|| error("artifact_write_failed", message));
        self.control.cancel();
    }

    fn setup(&self, record: &[u8], initial: &Counters) -> Result<Readiness> {
        if self.store.start(&self.id, initial.prompt_tokens)? == Start::Settled {
            return Ok(Readiness::Settled);
        }
        self.store
            .append_encoded_progress(&self.id, &[record], Phase::Prefill, initial.clone())?;
        Ok(Readiness::Execute)
    }

    pub(super) fn run(
        self,
        record: Vec<u8>,
        initial: Counters,
        incoming: Receiver<Event>,
        started: SyncSender<Result<Readiness>>,
        outcome: Receiver<Outcome>,
    ) -> Result<()> {
        match self.setup(&record, &initial) {
            Ok(Readiness::Settled) => {
                let _ = started.send(Ok(Readiness::Settled));
                return Ok(());
            }
            Ok(Readiness::Execute) => {
                let _ = started.send(Ok(Readiness::Execute));
            }
            Err(cause) => {
                if let Ok(status) = self.store.status(&self.id) {
                    if status.state.terminal() {
                        let _ = started.send(Ok(Readiness::Settled));
                        return Ok(());
                    }
                    let error = error(
                        "artifact_prepare_failed",
                        "Native publication setup failed before model execution.",
                    );
                    if self
                        .store
                        .finish_generation(
                            &self.id,
                            crate::serve::jobs::state::StopReason::ExecutionError,
                            status.generation.counters,
                            Some(error.clone()),
                        )
                        .is_ok()
                    {
                        let _ = self.store.finalize(&self.id, Some(error));
                    }
                }
                let message = format!("prepare native publication: {cause:#}");
                let _ = started.send(Err(anyhow::anyhow!(message.clone())));
                return Err(anyhow::anyhow!(message));
            }
        }
        drop(record);
        let mut last = initial;
        let mut phase = Phase::Prefill;
        let mut disk_failed = false;
        let mut stage_used = false;
        let mut records_started = false;
        let mut pending = None;
        let mut batches = 0u64;
        while let Some(event) = pending.take().or_else(|| incoming.recv().ok()) {
            let event = match event {
                Event::Stage(request) => {
                    if disk_failed || stage_used || records_started {
                        request.refuse(
                            "staging requires a healthy writer before generation, once per job",
                        );
                    } else {
                        stage_used = true;
                        request.run(
                            || {
                                super::super::execute::preparation_checkpoint(
                                    &self.control,
                                    &self.server,
                                )
                            },
                            qwen_llm::metal::MetalContext::process_limit_bytes_remaining,
                        );
                    }
                    continue;
                }
                event => event,
            };
            if disk_failed {
                continue;
            }
            records_started = true;
            let result = match event {
                Event::Array(ref record, ref payload, ref _permit, next_phase, ref counters) => {
                    (|| -> Result<()> {
                        check_progress(Some(phase), &last, next_phase, counters)?;
                        self.store.append_array(
                            &self.id,
                            serde_json::from_slice(record)?,
                            &payload.bytes,
                        )?;
                        self.store
                            .progress(&self.id, next_phase, counters.clone())?;
                        phase = next_phase;
                        last = counters.clone();
                        Ok(())
                    })()
                }
                event => self
                    .batch(event, &incoming, &mut pending, &mut phase, &mut last)
                    .map(|published| batches += u64::from(published)),
            };
            if let Err(cause) = result {
                tracing::error!(job_id = self.id, "native publication: {cause:#}");
                self.fail(
                    "Native result publication failed; generation outcome is tracked separately.",
                );
                disk_failed = true;
            }
        }
        let outcome = outcome
            .recv()
            .unwrap_or_else(|_| Outcome::interrupted(last));
        self.store.finish_generation(
            &self.id,
            outcome.reason,
            outcome.counters.clone(),
            outcome.error.clone(),
        )?;
        if !disk_failed {
            let terminal = json!({
                "kind":"generation_terminal", "state":outcome.state_name(), "stop_reason":outcome.reason,
                "sampled_tokens":outcome.counters.sampled_tokens,
                "consumed_generated_tokens":outcome.counters.consumed_generated_tokens, "error":outcome.error,
                "artifact_writer":{
                    "peak_record_bytes":self.budget.peak.load(Ordering::Acquire),
                    "peak_array_bytes":self.array_budget.peak.load(Ordering::Acquire),
                    "metadata_batches":batches, "backpressure_waits":self.waits.load(Ordering::Acquire)
                },
                "cost":{"wall_ms":outcome.wall_ms,"prefill_ms":null,"decode_ms":null,"readout_ms":null}
            });
            if let Err(cause) = self.store.append(&self.id, &[terminal]) {
                tracing::error!(job_id = self.id, "native terminal record: {cause:#}");
                self.fail("Terminal record publication failed.");
            }
        }
        self.store.finalize(
            &self.id,
            self.failure
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
        )?;
        Ok(())
    }

    fn batch(
        &self,
        first: Event,
        incoming: &Receiver<Event>,
        pending: &mut Option<Event>,
        phase: &mut Phase,
        last: &mut Counters,
    ) -> Result<bool> {
        let bound = |event: &Event| match event {
            Event::Record(bytes, ..) => Some(bytes.len().saturating_add(28)),
            Event::Progress(..) => Some(0),
            _ => None,
        };
        let mut bytes = bound(&first).context("metadata batch requires a record or progress")?;
        let mut events = vec![first];
        while events.len() < EVENTS {
            let Ok(event) = incoming.try_recv() else {
                break;
            };
            match bound(&event) {
                Some(size) if size <= self.store.limits().max_batch_bytes.saturating_sub(bytes) => {
                    bytes += size;
                    events.push(event);
                }
                _ => {
                    *pending = Some(event);
                    break;
                }
            }
        }
        let (mut next_phase, mut counters) = (*phase, last.clone());
        let mut records = Vec::new();
        for event in &events {
            let (p, c) = match event {
                Event::Record(bytes, _permit, p, c) => {
                    records.push(bytes.as_slice());
                    (*p, c)
                }
                Event::Progress(p, c) => (*p, c),
                _ => unreachable!("barriers are not batched"),
            };
            check_progress(Some(next_phase), &counters, p, c)?;
            next_phase = p;
            counters = c.clone();
        }
        if records.is_empty() && next_phase == *phase && counters == *last {
            return Ok(false);
        }
        self.store
            .append_encoded_progress(&self.id, &records, next_phase, counters.clone())?;
        *phase = next_phase;
        *last = counters;
        // Original Vec storage and permits remain live through durable publication.
        Ok(true)
    }
}
