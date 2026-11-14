//! Owner-thread continuous batching for V4.1 (`QWEN_SERVE_V41_MAX_ACTIVE` > 1).
//!
//! Every `pump` is one bounded quantum: reap requests whose client is gone,
//! deliver output to parked clients, admit at most one waiting request, then
//! either one prefill call for the oldest prefilling request or one decode
//! step over every unparked decoding request (`Forward::step_batch`, plain
//! `step` when only one decodes). Under mixed load, normal prefill quanta
//! earn a bounded wall-time budget for decoding. The rollback alternates
//! single steps with small prefill calls.
//!
//! Each request runs the serial path's phases ([`Settings::begin`],
//! [`Settings::prefill_slice`], [`Settings::finish_prefill`],
//! [`DecodeCursor`], [`Settings::finish`]); only scheduling differs:
//! - Fair prefilling retains serial's `V41_PREFILL_ROWS` geometry, CED
//!   eligibility and header folding. The requested decode-time share earns
//!   at most one second per prefill quantum; single-kind service earns no
//!   credit. Rollback uses [`LOADED_PREFILL_ROWS`] while others decode,
//!   changing that prompt's partition and possibly its F32 rounding.
//!   Neither policy preempts a native call already in progress.
//! - A batched step never receives a request's cancellation flag, only the
//!   process-shutdown flag: one disconnect must not poison the others. Each
//!   request's own cancellation is observed between steps and drops it alone.
//! - Members are checked before a batch (ready, below capacity); a refused
//!   batch leaves every state unchanged and falls back to per-request steps,
//!   so only an offender fails. A batch that fails after it started has
//!   poisoned every member, so every member fails (500).
//! - Output is queued and delivered without waiting. A request whose client
//!   has not drained its earlier output is parked: it leaves the batch until
//!   the client catches up, and is aborted after [`PARK_TIMEOUT`].
//! - Admission adds the caller reserves of requests already in flight to the
//!   new request's reserve; the engine admits each state against current
//!   memory signals, which include the other requests' resident states.
//! - With a [`SamplePool`], the members of a step that returned logits are
//!   sampled on its threads before any member is consumed; consumption then
//!   proceeds in the same order with each member's result where it would
//!   have sampled (see [`super::sample_pool`]).
use super::super::batching::{BatchSink, Job, shutting_down};
use super::cursor::{Advance, DecodeCursor};
use super::execution::{
    BudgetedSink, Engine, Run, admit_output, cancel_of, cancelled, check, interrupted,
};
use super::sample_pool::{SampleJob, SamplePool, Sampled};
use super::*;
use qwen_llm::sampling::{SampledToken, Sampler};
use std::collections::VecDeque;
use std::sync::Arc;

/// Rollback prompt rows per call while other requests are decoding.
pub(super) const LOADED_PREFILL_ROWS: usize = 512;

pub(super) fn loaded_prefill_rows(settings: &Settings) -> usize {
    if settings.prefill_fair {
        crate::family_options::V41_PREFILL_ROWS
    } else {
        LOADED_PREFILL_ROWS.min(crate::family_options::V41_PREFILL_ROWS)
    }
}
/// A parked request whose client drains nothing for this long is aborted
/// (the HTTP socket write timeout).
pub(super) const PARK_TIMEOUT: Duration = Duration::from_secs(30);

/// The operations a batch needs beyond one request's [`Engine`].
pub(super) trait BatchEngine<S>: Engine<S> {
    /// One token per state. `cancel` is only ever the process-shutdown flag.
    fn step_batch(
        &mut self,
        states: &mut [&mut S],
        tokens: &[u32],
        cancel: &AtomicBool,
    ) -> std::result::Result<Vec<Vec<f32>>, String>;
    /// Whether [`Self::step_batch_sampled`] is available.
    fn supports_sampled_step(&self) -> bool {
        false
    }
    /// One token per state, sampled on the device where the request's policy
    /// allows it and on the host otherwise. Members whose sampler is fast
    /// eligible skip the logit readback; exact members are host-sampled after
    /// the step, so their tokens stay identical to [`Self::step_batch`] plus a
    /// host sample at the same draw. `explicit_seeds` is per member.
    fn step_batch_sampled(
        &mut self,
        states: &mut [&mut S],
        tokens: &[u32],
        samplers: &mut [&mut Sampler],
        explicit_seeds: &[bool],
        cancel: &AtomicBool,
    ) -> std::result::Result<Vec<u32>, String> {
        let _ = (states, tokens, samplers, explicit_seeds, cancel);
        Err("sampled batch step is unsupported".into())
    }
    /// Ready for another token: neither provisional nor poisoned, below capacity.
    fn can_step(&self, state: &S) -> bool;
    /// Poisoned by partial execution (a batch that failed after it started).
    fn poisoned(&self, state: &S) -> bool;
    fn position(&self, state: &S) -> usize;
}

/// One member's forward result: logits still to sample, or a token a device
/// sampler already selected.
enum SteppedStep {
    Logits(Vec<f32>),
    Token(u32),
}

/// A request index, its forward result, the forward's milliseconds and the
/// width of the step that produced it.
type Stepped = (usize, std::result::Result<SteppedStep, String>, f64, usize);

struct Pending {
    job: Job,
    submitted: Instant,
}

struct Active<S> {
    sink: Box<dyn BatchSink>,
    run: Option<Run<S>>,
    cursor: Option<DecodeCursor>,
    /// The published token awaiting its forward.
    next: Option<u32>,
    /// Generation done and committed; waiting for the client to drain.
    finished: Option<GenerationOutcome>,
    own_reserve: u64,
    emitted: usize,
    queue_ms: f64,
    parked_since: Option<Instant>,
    parked_ms: f64,
    width_sum: usize,
    width_max: usize,
    stepped: usize,
    batched_transitions: usize,
    loaded_prefill_calls: usize,
    token_io: Option<memory_io::Observations>,
    prefill_io: Option<memory_io::ChunkObservations>,
}

impl<S> Active<S> {
    fn decoding(&self) -> bool {
        self.cursor.is_some() || self.finished.is_some()
    }
    fn parked(&self) -> bool {
        self.parked_since.is_some()
    }
    fn unpark(&mut self) {
        if let Some(since) = self.parked_since.take() {
            self.parked_ms += since.elapsed().as_secs_f64() * 1e3;
        }
    }
}

pub(super) struct Scheduler<S> {
    /// One clock for every request's phase timings across pumps.
    epoch: Instant,
    max_active: usize,
    park_timeout: Duration,
    pending: VecDeque<Pending>,
    active: Vec<Active<S>>,
    prefer_prefill: bool,
    /// Decode service earned by the preceding mixed-load prefill quantum.
    /// Never carried across idle or single-kind service.
    decode_budget: Duration,
    /// 5xx completions since the last pump, for the owner's failure hook.
    server_failures: usize,
    /// Samples a multi-member step's members in parallel; None samples each
    /// on the owner as it is consumed.
    sample_pool: Option<SamplePool>,
    /// Members sampled through the pool (tests assert it was exercised).
    #[cfg(test)]
    pool_sampled: usize,
}

/// Adapts the request's owner sink to the serial phases' sink contract:
/// pieces are queued, never awaited.
struct Publish<'a>(&'a mut dyn BatchSink);

impl GenerationSink for Publish<'_> {
    fn cancellation_flag(&self) -> Option<Arc<AtomicBool>> {
        Some(self.0.cancellation_flag())
    }
    fn transport_reserve_bytes(&self) -> u64 {
        self.0.transport_reserve_bytes()
    }
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.0.check()?;
        self.0.queue(bytes);
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        self.0.check()
    }
}

/// Wraps prefill calls with the request's opt-in process IO observation.
pub(super) struct Observed<'a, E> {
    pub(super) engine: &'a mut E,
    pub(super) prefill_io: Option<&'a mut memory_io::ChunkObservations>,
    pub(super) token_io: Option<&'a mut memory_io::Observations>,
}

impl<S, E: BatchEngine<S>> Engine<S> for Observed<'_, E> {
    fn admit(
        &mut self,
        positions: usize,
        prefill_tokens: usize,
        reserve: u64,
    ) -> std::result::Result<(), StateError> {
        self.engine.admit(positions, prefill_tokens, reserve)
    }
    fn new_state(&mut self, required: usize) -> std::result::Result<S, ServeError> {
        self.engine.new_state(required)
    }
    fn prefill(
        &mut self,
        state: &mut S,
        tokens: &[u32],
        cancel: &AtomicBool,
        with_logits: bool,
    ) -> std::result::Result<Vec<f32>, ServeError> {
        let position = self.engine.position(state);
        memory_io::observe_chunk(
            self.prefill_io.as_deref_mut(),
            position,
            tokens.len(),
            || qwen_llm::pid_metrics::PidSnapshot::now().ok(),
            || self.engine.prefill(state, tokens, cancel, with_logits),
        )
    }
    fn cold_prefill(
        &mut self,
        state: &mut S,
        tokens: &[u32],
        cancel: &AtomicBool,
        begin: bool,
        max_rows: usize,
    ) -> std::result::Result<Option<qwen_llm::deepseek_v41::forward::ColdPrefillSlice>, ServeError>
    {
        self.engine
            .cold_prefill(state, tokens, cancel, begin, max_rows)
    }
    fn step(
        &mut self,
        state: &mut S,
        token: u32,
        cancel: &AtomicBool,
    ) -> std::result::Result<Vec<f32>, String> {
        let position = self.engine.position(state);
        memory_io::observe(
            self.token_io.as_deref_mut(),
            position,
            || qwen_llm::pid_metrics::PidSnapshot::now().ok(),
            || self.engine.step(state, token, cancel),
        )
    }
    fn now(&mut self) -> Duration {
        self.engine.now()
    }
}

impl<S> Scheduler<S> {
    pub(super) fn new(max_active: usize) -> Self {
        Self {
            epoch: Instant::now(),
            max_active: max_active.max(1),
            park_timeout: PARK_TIMEOUT,
            pending: VecDeque::new(),
            active: Vec::new(),
            prefer_prefill: false,
            decode_budget: Duration::ZERO,
            server_failures: 0,
            sample_pool: None,
            #[cfg(test)]
            pool_sampled: 0,
        }
    }

    /// Sample multi-member steps on `workers` persistent threads beside the
    /// owner (0 keeps sampling serial on the owner).
    pub(super) fn with_sample_workers(mut self, workers: usize) -> std::io::Result<Self> {
        self.sample_pool = (workers > 0)
            .then(|| SamplePool::new(workers))
            .transpose()?;
        Ok(self)
    }

    pub(super) fn sample_workers(&self) -> usize {
        self.sample_pool.as_ref().map_or(0, SamplePool::workers)
    }

    #[cfg(test)]
    pub(super) fn pool_sampled(&self) -> usize {
        self.pool_sampled
    }

    #[cfg(test)]
    pub(super) fn with_park_timeout(mut self, timeout: Duration) -> Self {
        self.park_timeout = timeout;
        self
    }

    pub(super) fn max_active(&self) -> usize {
        self.max_active
    }

    /// The engine clock's origin; phase timings subtract readings taken in
    /// different pumps, so the engine must not restart it per pump.
    pub(super) fn epoch(&self) -> Instant {
        self.epoch
    }

    pub(super) fn submit(&mut self, job: Job) {
        job.sink
            .lifecycle()
            .mark(super::super::lifecycle_timing::Stage::Submitted);
        self.pending.push_back(Pending {
            job,
            submitted: Instant::now(),
        });
    }

    /// A waiting request will be admitted by the next pump.
    pub(super) fn admitting(&self) -> bool {
        !self.pending.is_empty() && self.active.len() < self.max_active
    }

    #[cfg(test)]
    pub(super) fn in_flight(&self) -> (usize, usize) {
        (self.pending.len(), self.active.len())
    }

    #[cfg(test)]
    pub(super) fn with_state(&mut self, index: usize, change: impl FnOnce(&mut S)) {
        change(&mut self.active[index].run.as_mut().expect("run").state);
    }

    /// 5xx completions since the last call.
    pub(super) fn take_server_failures(&mut self) -> usize {
        std::mem::take(&mut self.server_failures)
    }

    fn complete(
        &mut self,
        sink: Box<dyn BatchSink>,
        result: std::result::Result<GenerationOutcome, BackendFailure>,
    ) {
        if let Err(BackendFailure::Serve(error)) = &result
            && error.status >= 500
        {
            self.server_failures += 1;
        }
        sink.complete(result);
    }

    /// Remove request `index` and publish `result`. A failed request's state
    /// is dropped, never committed.
    fn retire(
        &mut self,
        index: usize,
        result: std::result::Result<GenerationOutcome, BackendFailure>,
    ) {
        let active = self.active.remove(index);
        self.complete(active.sink, result);
    }

    /// A request failure; any failure of a cancelled request is its abort.
    fn fail(&mut self, index: usize, failure: BackendFailure) {
        let cancelled = self.active[index]
            .run
            .as_ref()
            .is_some_and(|run| cancelled(run.cancel()))
            || self.active[index].sink.check().is_err();
        let failure = match failure {
            BackendFailure::Serve(_) if cancelled => BackendFailure::Aborted(interrupted()),
            failure => failure,
        };
        self.retire(index, Err(failure));
    }

    /// Complete everything now: started requests abort, waiting ones get
    /// the busy response.
    pub(super) fn abort_all(&mut self) {
        while let Some(pending) = self.pending.pop_front() {
            self.complete(pending.job.sink, Err(shutting_down().into()));
        }
        while !self.active.is_empty() {
            self.retire(
                0,
                Err(BackendFailure::Aborted(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "V4.1 request aborted: server stopping",
                ))),
            );
        }
    }

    /// One bounded quantum; true when any request made progress.
    pub(super) fn pump(
        &mut self,
        settings: &Settings,
        engine: &mut impl BatchEngine<S>,
        reuse: &mut impl reuse::Reuse<S>,
    ) -> bool {
        // Every GPU-feeding call in this round (plain and sliced prefill, CED,
        // serial and batched steps, device sampling) commits and then blocks on
        // command completion on this thread. One scope per round keeps it
        // user-initiated across all of them and restores it when the round ends.
        let _qos = qwen_llm::metal::GpuThreadQos::enter();
        let mut progressed = self.reap();
        progressed |= self.deliver();
        if self.active.len() < self.max_active
            && let Some(pending) = self.pending.pop_front()
        {
            self.admit(pending, settings, engine, reuse);
            progressed = true;
        }
        let prefilling = self
            .active
            .iter()
            .position(|a| a.cursor.is_none() && a.finished.is_none());
        let decoders: Vec<usize> = (0..self.active.len())
            .filter(|&i| self.active[i].next.is_some() && !self.active[i].parked())
            .collect();
        let mixed = prefilling.is_some() && !decoders.is_empty();
        if !mixed {
            self.decode_budget = Duration::ZERO;
        }
        let prefill_turn = match (prefilling, decoders.is_empty()) {
            (Some(_), true) => true,
            (Some(_), false) if settings.prefill_fair => self.decode_budget.is_zero(),
            (Some(_), false) => {
                self.prefer_prefill = !self.prefer_prefill;
                !self.prefer_prefill
            }
            (None, _) => false,
        };
        let started = engine.now();
        if prefill_turn {
            let interleaved =
                self.prefill(prefilling.expect("prefill turn"), settings, engine, reuse);
            if interleaved {
                self.decode_budget = Duration::ZERO;
            } else if settings.prefill_fair && mixed {
                self.decode_budget = (engine.now() - started)
                    .mul_f64(
                        settings.decode_percent as f64 / (100 - settings.decode_percent) as f64,
                    )
                    .min(Duration::from_secs(1));
            }
            progressed = true;
        } else if !decoders.is_empty() {
            self.decode(decoders, settings, engine, reuse);
            self.decode_budget = self.decode_budget.saturating_sub(engine.now() - started);
            progressed = true;
        }
        progressed
    }

    /// Drop requests whose client stopped, or whose server is stopping.
    fn reap(&mut self) -> bool {
        let mut progressed = false;
        let mut index = 0;
        while index < self.pending.len() {
            if let Err(error) = self.pending[index].job.sink.check() {
                let pending = self.pending.remove(index).expect("pending index");
                self.complete(pending.job.sink, Err(BackendFailure::Aborted(error)));
                progressed = true;
            } else {
                index += 1;
            }
        }
        let mut index = 0;
        while index < self.active.len() {
            if let Err(error) = self.active[index].sink.check() {
                self.retire(index, Err(BackendFailure::Aborted(error)));
                progressed = true;
            } else {
                index += 1;
            }
        }
        progressed
    }

    /// Deliver queued output of parked and finished requests.
    fn deliver(&mut self) -> bool {
        let mut progressed = false;
        let mut index = 0;
        while index < self.active.len() {
            let active = &mut self.active[index];
            if !active.parked() && active.finished.is_none() {
                index += 1;
                continue;
            }
            match active.sink.flush() {
                Err(error) => {
                    self.retire(index, Err(BackendFailure::Aborted(error)));
                    progressed = true;
                    continue;
                }
                Ok(true) => {
                    progressed |= active.parked();
                    active.unpark();
                    if let Some(outcome) = active.finished.take() {
                        self.retire(index, Ok(outcome));
                        progressed = true;
                        continue;
                    }
                }
                Ok(false) => {
                    let since = *active.parked_since.get_or_insert_with(Instant::now);
                    if since.elapsed() >= self.park_timeout {
                        self.retire(
                            index,
                            Err(BackendFailure::Aborted(io::Error::new(
                                io::ErrorKind::TimedOut,
                                "V4.1 client did not drain output within the write timeout",
                            ))),
                        );
                        progressed = true;
                        continue;
                    }
                }
            }
            index += 1;
        }
        progressed
    }

    fn admit(
        &mut self,
        pending: Pending,
        settings: &Settings,
        engine: &mut impl BatchEngine<S>,
        reuse: &mut impl reuse::Reuse<S>,
    ) {
        let Pending { mut job, submitted } = pending;
        let timing = job.sink.lifecycle();
        let _lifecycle = timing.enter();
        timing.mark(super::super::lifecycle_timing::Stage::AdmissionStart);
        let admission_span = super::super::lifecycle_timing::Span::current("admission", 0, 0);
        let queue_ms = submitted.elapsed().as_secs_f64() * 1e3;
        let super::super::output_partition::OutputProtocol::DeepSeekV41 { max_bytes, .. } =
            job.prepared.validated.output()
        else {
            let error =
                ServeError::server_error("V4.1 prepared request has a non-V4.1 output protocol");
            self.complete(job.sink, Err(error.into()));
            return;
        };
        let others: u64 = self.active.iter().map(|a| a.own_reserve).sum();
        let echo = job.prepared.request().echo_stats;
        let mut prefill_io = echo.then(memory_io::ChunkObservations::default);
        let begun = settings.begin(
            *max_bytes,
            job.prepared.request(),
            &job.prepared.prompt,
            job.prepared.boundaries,
            job.prepared.tokens.as_deref(),
            &mut Publish(job.sink.as_mut()),
            others,
            &mut Observed {
                engine,
                token_io: None,
                prefill_io: prefill_io.as_mut(),
            },
            reuse,
        );
        let run = match begun {
            Ok(run) => run,
            Err(failure) => {
                self.complete(job.sink, Err(failure));
                return;
            }
        };
        drop(admission_span);
        timing.mark(super::super::lifecycle_timing::Stage::Admitted);
        timing.extent("deepseek_v41", run.ids.len(), run.position);
        self.active.push(Active {
            own_reserve: run.reserve.saturating_sub(others),
            sink: job.sink,
            run: Some(run),
            cursor: None,
            next: None,
            finished: None,
            emitted: 0,
            queue_ms,
            parked_since: None,
            parked_ms: 0.0,
            width_sum: 0,
            width_max: 0,
            stepped: 0,
            batched_transitions: 0,
            loaded_prefill_calls: 0,
            token_io: echo.then(memory_io::Observations::default),
            prefill_io,
        });
        let index = self.active.len() - 1;
        if self.active[index].run.as_ref().expect("begun").prefilled() {
            self.start_decode(index, settings, engine, reuse);
        }
    }

    fn prefill(
        &mut self,
        index: usize,
        settings: &Settings,
        engine: &mut impl BatchEngine<S>,
        reuse: &mut impl reuse::Reuse<S>,
    ) -> bool {
        let timing = self.active[index].sink.lifecycle();
        let _lifecycle = timing.enter();
        let run = self.active[index].run.as_ref().expect("prefill run");
        let _round = super::super::lifecycle_timing::Round::new(
            "prefill",
            std::iter::once(timing),
            run.position,
            run.ids.len().saturating_sub(run.position),
        );
        let loaded = self
            .active
            .iter()
            .enumerate()
            .any(|(other, active)| other != index && active.decoding());
        let rows = if loaded {
            loaded_prefill_rows(settings)
        } else {
            crate::family_options::V41_PREFILL_ROWS
        };
        if loaded && settings.prefill_fair && settings.layer_slice_ms > 0 {
            // Keep the provisional appender outside every decode scan. Peer
            // retirement may compact the active vector between checkpoints.
            let mut active = self.active.remove(index);
            let mut rounds = 0;
            let result = {
                let run = active.run.as_mut().expect("prefilling run");
                let position = engine.position(&run.state);
                let input_rows = run.next_endpoint(rows).saturating_sub(run.position);
                memory_io::observe_chunk(
                    active.prefill_io.as_mut(),
                    position,
                    input_rows,
                    || qwen_llm::pid_metrics::PidSnapshot::now().ok(),
                    || {
                        settings.prefill_layer_slice(
                            run,
                            rows,
                            &mut Publish(active.sink.as_mut()),
                            engine,
                            reuse,
                            Some(&mut |engine, reuse| {
                                self.reap();
                                self.deliver();
                                let members = (0..self.active.len())
                                    .filter(|&i| {
                                        self.active[i].decoding() && !self.active[i].parked()
                                    })
                                    .collect::<Vec<_>>();
                                if !members.is_empty() {
                                    self.decode(members, settings, engine, reuse);
                                    rounds += 1;
                                }
                            }),
                        )
                    },
                )
            };
            if rounds > 0 && result.is_ok() {
                let members = (0..self.active.len())
                    .filter(|&i| self.active[i].decoding() && !self.active[i].parked())
                    .collect::<Vec<_>>();
                if !members.is_empty() {
                    self.decode(members, settings, engine, reuse);
                    rounds += 1;
                }
            }
            active
                .run
                .as_mut()
                .expect("prefill run")
                .prefill_work
                .peer_rounds += rounds;
            let index = index.min(self.active.len());
            self.active.insert(index, active);
            if let Err(failure) = result {
                self.fail(index, failure);
            } else if self.active[index]
                .run
                .as_ref()
                .expect("prefill run")
                .prefilled()
            {
                self.start_decode(index, settings, engine, reuse);
            }
            return rounds > 0;
        }
        let active = &mut self.active[index];
        let run = active.run.as_mut().expect("prefilling request has a run");
        if loaded
            && run.next_endpoint(rows) < run.next_endpoint(crate::family_options::V41_PREFILL_ROWS)
        {
            active.loaded_prefill_calls += 1;
        }
        let result = settings.prefill_slice(
            run,
            rows,
            &mut Publish(active.sink.as_mut()),
            &mut Observed {
                engine,
                token_io: None,
                prefill_io: active.prefill_io.as_mut(),
            },
            reuse,
        );
        if let Err(failure) = result {
            self.fail(index, failure);
            return false;
        }
        if run.prefilled() {
            self.start_decode(index, settings, engine, reuse);
        }
        false
    }

    fn start_decode(
        &mut self,
        index: usize,
        settings: &Settings,
        engine: &mut impl BatchEngine<S>,
        reuse: &mut impl reuse::Reuse<S>,
    ) {
        let active = &mut self.active[index];
        let _lifecycle = active.sink.lifecycle().enter();
        let run = active.run.as_mut().expect("prefilled request has a run");
        if let Err(failure) =
            settings.finish_prefill(run, &mut Publish(active.sink.as_mut()), engine, reuse)
        {
            self.fail(index, failure);
            return;
        }
        match DecodeCursor::new(super::super::decode_loop::DecodeRequest {
            family: "V4.1",
            logits: run.logits.clone(),
            max_tokens: run.maximum,
            stop_tokens: &[settings.eos as i32],
            vocab_size: run.vocab as u32,
        }) {
            Ok(cursor) => {
                active.cursor = Some(cursor);
                self.advance(index, None, settings, reuse);
            }
            Err(failure) => self.fail(index, failure),
        }
    }

    /// Select and publish the next token, or finish the request. `presampled`
    /// is this request's sampler already applied to the cursor's logits.
    fn advance(
        &mut self,
        index: usize,
        presampled: Option<Sampled>,
        settings: &Settings,
        reuse: &mut impl reuse::Reuse<S>,
    ) {
        let active = &mut self.active[index];
        let run = active.run.as_mut().expect("decoding request has a run");
        let cursor = active
            .cursor
            .as_mut()
            .expect("decoding request has a cursor");
        let flag = run.flag.clone();
        let cancel = cancel_of(&flag);
        let mut publish = Publish(active.sink.as_mut());
        let mut budgeted = BudgetedSink {
            sink: &mut publish,
            cancel,
            admit: || admit_output(reuse, run.reserve, settings.headroom),
            ceiling: run.ceiling,
            emitted: active.emitted,
        };
        let advanced = cursor.advance(
            &mut run.sampler,
            presampled,
            settings.tokenizer.as_ref(),
            &mut budgeted,
            crate::shutdown::checkpoint,
        );
        active.emitted = budgeted.emitted;
        let generation = match advanced {
            Err(failure) => return self.fail(index, failure),
            Ok(Advance::Forward(token)) => {
                active.next = Some(token);
                return self.publish(index);
            }
            Ok(Advance::Done(generation)) => generation,
        };
        if let Err(failure) = check(&mut Publish(active.sink.as_mut()), cancel) {
            return self.fail(index, failure);
        }
        let run = active.run.take().expect("decoding request has a run");
        active.cursor = None;
        let batch = serde_json::json!({
            "max_active": self.max_active,
            "queue_ms": active.queue_ms,
            "parked_ms": active.parked_ms,
            "stepped_transitions": active.stepped,
            "width_max": active.width_max,
            "width_mean": (active.stepped > 0).then(|| active.width_sum as f64 / active.stepped as f64),
            "batched_transitions": active.batched_transitions,
            "prefill_loaded_rows": loaded_prefill_rows(settings),
            "prefill_loaded_calls": active.loaded_prefill_calls,
            "prefill_fair": settings.prefill_fair,
            "decode_percent": settings.prefill_fair.then_some(settings.decode_percent),
        });
        let mut outcome = settings.finish(run, &generation, reuse, Some(batch));
        if let (Some(token_io), Some(prefill_io)) = (&active.token_io, &active.prefill_io) {
            let phases = outcome
                .stats
                .as_mut()
                .expect("shared stats")
                .phases
                .as_mut()
                .expect("V4.1 phases");
            phases["process_token_io"] = token_io.json();
            phases["process_prefill_io"] = prefill_io.json();
        }
        active.finished = Some(outcome);
        self.publish(index);
    }

    /// Deliver what the request queued; park it if its client lags, and
    /// complete it once a finished request's output is delivered.
    fn publish(&mut self, index: usize) {
        let active = &mut self.active[index];
        match active.sink.flush() {
            Err(error) => self.retire(index, Err(BackendFailure::Aborted(error))),
            Ok(true) => {
                if let Some(outcome) = active.finished.take() {
                    self.retire(index, Ok(outcome));
                }
            }
            Ok(false) => {
                active.parked_since.get_or_insert_with(Instant::now);
            }
        }
    }

    fn decode(
        &mut self,
        members: Vec<usize>,
        settings: &Settings,
        engine: &mut impl BatchEngine<S>,
        reuse: &mut impl reuse::Reuse<S>,
    ) {
        let _round = super::super::lifecycle_timing::Round::new(
            "decode",
            members.iter().map(|&i| self.active[i].sink.lifecycle()),
            0,
            1,
        );
        let mut results: Vec<Stepped> = Vec::with_capacity(members.len());
        if members.len() == 1 {
            // Serial step with the request's own cancellation, as `step` alone.
            let index = members[0];
            results.push(self.step_one(index, engine));
        } else {
            let (batch, single): (Vec<usize>, Vec<usize>) = members.into_iter().partition(|&i| {
                engine.can_step(&self.active[i].run.as_ref().expect("decoding run").state)
            });
            // An offender steps alone and fails with the engine's own refusal.
            for index in single {
                results.push(self.step_one(index, engine));
            }
            if batch.len() == 1 {
                results.push(self.step_one(batch[0], engine));
            } else if batch.len() > 1 {
                if settings.gpu_sample && engine.supports_sampled_step() {
                    results.extend(self.step_batch_sampled(batch, engine));
                } else {
                    results.extend(self.step_batch(batch, engine));
                }
            }
        }
        results.sort_by_key(|(index, ..)| std::cmp::Reverse(*index));
        let mut presampled = self.presample(&mut results);
        // Highest index first so retiring a request keeps lower indices valid.
        for ((index, result, elapsed_ms, width), presampled) in
            results.into_iter().zip(presampled.drain(..))
        {
            self.consume(
                index, result, elapsed_ms, width, presampled, settings, reuse,
            );
        }
    }

    /// Run the sampler of every member that returned logits on the pool, one
    /// result per entry of `results` (None where the owner samples or nothing
    /// is sampled). Samplers and logits are moved out and back; nothing else
    /// changes before consumption.
    fn presample(&mut self, results: &mut [Stepped]) -> Vec<Option<Sampled>> {
        let mut presampled: Vec<Option<Sampled>> = results.iter().map(|_| None).collect();
        let Some(pool) = self.sample_pool.as_mut() else {
            return presampled;
        };
        let members: Vec<usize> = (0..results.len())
            .filter(|&k| matches!(results[k].1, Ok(SteppedStep::Logits(_))))
            .collect();
        if members.len() < 2 {
            return presampled;
        }
        let jobs = members
            .iter()
            .map(|&k| {
                let run = self.active[results[k].0]
                    .run
                    .as_mut()
                    .expect("decoding run");
                let placeholder =
                    qwen_llm::sampling::Sampler::new(qwen_llm::sampling::SamplingConfig::default())
                        .expect("default sampler");
                let logits = match &mut results[k].1 {
                    Ok(SteppedStep::Logits(logits)) => std::mem::take(logits),
                    _ => unreachable!("filtered to logit results"),
                };
                SampleJob {
                    sampler: std::mem::replace(&mut run.sampler, placeholder),
                    logits,
                    sampled: None,
                }
            })
            .collect();
        for (&k, job) in members.iter().zip(pool.sample_all(jobs)) {
            let run = self.active[results[k].0]
                .run
                .as_mut()
                .expect("decoding run");
            run.sampler = job.sampler;
            match &mut results[k].1 {
                Ok(SteppedStep::Logits(logits)) => *logits = job.logits,
                _ => unreachable!("filtered to logit results"),
            }
            presampled[k] = job.sampled;
        }
        #[cfg(test)]
        {
            self.pool_sampled += members.len();
        }
        presampled
    }

    fn step_one(&mut self, index: usize, engine: &mut impl BatchEngine<S>) -> Stepped {
        let active = &mut self.active[index];
        let run = active.run.as_mut().expect("decoding run");
        let token = active.next.expect("published token");
        let flag = run.flag.clone();
        let position = engine.position(&run.state);
        let started = Instant::now();
        let result = memory_io::observe(
            active.token_io.as_mut(),
            position,
            || qwen_llm::pid_metrics::PidSnapshot::now().ok(),
            || engine.step(&mut run.state, token, cancel_of(&flag)),
        );
        (
            index,
            result.map(SteppedStep::Logits),
            started.elapsed().as_secs_f64() * 1e3,
            1,
        )
    }

    fn step_batch(
        &mut self,
        members: Vec<usize>,
        engine: &mut impl BatchEngine<S>,
    ) -> Vec<Stepped> {
        let width = members.len();
        let observe = self.active[members[0]].token_io.is_some();
        let before = observe
            .then(|| qwen_llm::pid_metrics::PidSnapshot::now().ok())
            .flatten();
        let started = Instant::now();
        let mut tokens = Vec::with_capacity(width);
        let mut positions = Vec::with_capacity(width);
        let mut states: Vec<&mut S> = Vec::with_capacity(width);
        for (index, active) in self.active.iter_mut().enumerate() {
            if members.contains(&index) {
                tokens.push(active.next.expect("published token"));
                let run = active.run.as_mut().expect("decoding run");
                positions.push(engine.position(&run.state));
                states.push(&mut run.state);
            }
        }
        let result = engine.step_batch(&mut states, &tokens, crate::shutdown::cancellation_flag());
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let after = observe
            .then(|| qwen_llm::pid_metrics::PidSnapshot::now().ok())
            .flatten();
        let refused = result.is_err() && states.iter().all(|state| !engine.poisoned(state));
        drop(states);
        for (&index, &position) in members.iter().zip(&positions) {
            let mut snapshots = [before, after].into_iter();
            let ok = result.is_ok();
            let _ = memory_io::observe(
                self.active[index].token_io.as_mut(),
                position,
                || snapshots.next().flatten(),
                || if ok { Ok(()) } else { Err(()) },
            );
        }
        match result {
            Ok(logits) => members
                .into_iter()
                .zip(logits)
                .map(|(index, logits)| (index, Ok(SteppedStep::Logits(logits)), elapsed_ms, width))
                .collect(),
            Err(error) if refused => {
                // Nothing was touched: each request steps alone, so only an
                // offender fails.
                tracing::info!(target: "qwen_diag",
                    "serve: deepseek_v41 batched step of {width} refused ({error}); stepping each request alone");
                members
                    .into_iter()
                    .map(|index| self.step_one(index, engine))
                    .collect()
            }
            Err(error) => {
                tracing::warn!(target: "qwen_diag",
                    "serve: deepseek_v41 batched step of {width} failed after it started; failing every member: {error}");
                members
                    .into_iter()
                    .map(|index| {
                        (
                            index,
                            Err(format!("batched step of {width} requests failed: {error}")),
                            elapsed_ms,
                            width,
                        )
                    })
                    .collect()
            }
        }
    }

    /// One batched step whose members are sampled on the device where policy
    /// allows it. Every result is a token: fast members skip the logit readback,
    /// exact members are host-sampled by the engine after the step.
    fn step_batch_sampled(
        &mut self,
        members: Vec<usize>,
        engine: &mut impl BatchEngine<S>,
    ) -> Vec<Stepped> {
        let width = members.len();
        let observe = self.active[members[0]].token_io.is_some();
        let before = observe
            .then(|| qwen_llm::pid_metrics::PidSnapshot::now().ok())
            .flatten();
        let started = Instant::now();
        let mut tokens = Vec::with_capacity(width);
        let mut positions = Vec::with_capacity(width);
        let mut explicit = Vec::with_capacity(width);
        let mut states: Vec<&mut S> = Vec::with_capacity(width);
        let mut samplers: Vec<&mut Sampler> = Vec::with_capacity(width);
        for (index, active) in self.active.iter_mut().enumerate() {
            if members.contains(&index) {
                tokens.push(active.next.expect("published token"));
                let run = active.run.as_mut().expect("decoding run");
                positions.push(engine.position(&run.state));
                explicit.push(run.explicit_seed);
                states.push(&mut run.state);
                samplers.push(&mut run.sampler);
            }
        }
        let result = engine.step_batch_sampled(
            &mut states,
            &tokens,
            &mut samplers,
            &explicit,
            crate::shutdown::cancellation_flag(),
        );
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let after = observe
            .then(|| qwen_llm::pid_metrics::PidSnapshot::now().ok())
            .flatten();
        let refused = result.is_err() && states.iter().all(|state| !engine.poisoned(state));
        drop(states);
        drop(samplers);
        for (&index, &position) in members.iter().zip(&positions) {
            let mut snapshots = [before, after].into_iter();
            let ok = result.is_ok();
            let _ = memory_io::observe(
                self.active[index].token_io.as_mut(),
                position,
                || snapshots.next().flatten(),
                || if ok { Ok(()) } else { Err(()) },
            );
        }
        match result {
            Ok(sampled) => members
                .into_iter()
                .zip(sampled)
                .map(|(index, token)| (index, Ok(SteppedStep::Token(token)), elapsed_ms, width))
                .collect(),
            Err(error) if refused => {
                tracing::info!(target: "qwen_diag",
                    "serve: deepseek_v41 sampled step of {width} refused ({error}); stepping each request alone");
                members
                    .into_iter()
                    .map(|index| self.step_one(index, engine))
                    .collect()
            }
            Err(error) => {
                tracing::warn!(target: "qwen_diag",
                    "serve: deepseek_v41 sampled step of {width} failed after it started; failing every member: {error}");
                members
                    .into_iter()
                    .map(|index| {
                        (
                            index,
                            Err(format!("sampled step of {width} requests failed: {error}")),
                            elapsed_ms,
                            width,
                        )
                    })
                    .collect()
            }
        }
    }

    /// Complete one request's transition with its forward result.
    #[allow(clippy::too_many_arguments)]
    fn consume(
        &mut self,
        index: usize,
        result: std::result::Result<SteppedStep, String>,
        elapsed_ms: f64,
        width: usize,
        presampled: Option<Sampled>,
        settings: &Settings,
        reuse: &mut impl reuse::Reuse<S>,
    ) {
        let active = &mut self.active[index];
        let token = active.next.take().expect("published token");
        let run = active.run.as_mut().expect("decoding run");
        let (logits, device_token): (std::result::Result<Vec<f32>, anyhow::Error>, Option<u32>) =
            match result {
                Err(error) => (Err(anyhow::Error::msg(error)), None),
                Ok(SteppedStep::Logits(next))
                    if next.len() != run.vocab || !next.iter().all(|v| v.is_finite()) =>
                {
                    (Err(anyhow::anyhow!("V4.1 returned malformed logits")), None)
                }
                Ok(SteppedStep::Logits(next)) => {
                    run.consumed.push(token);
                    run.logits = next.clone();
                    (Ok(next), None)
                }
                Ok(SteppedStep::Token(sampled)) => {
                    run.consumed.push(token);
                    (Ok(Vec::new()), Some(sampled))
                }
            };
        if logits.is_ok() {
            active.stepped += 1;
            active.width_sum += width;
            active.width_max = active.width_max.max(width);
            active.batched_transitions += usize::from(width > 1);
        }
        let cursor = active.cursor.as_mut().expect("decoding cursor");
        match device_token {
            Some(sampled) => match cursor.on_token(elapsed_ms, crate::shutdown::checkpoint) {
                Ok(()) => self.advance(
                    index,
                    Some(Ok(SampledToken {
                        token: sampled as i32,
                        candidate_index: 0,
                    })),
                    settings,
                    reuse,
                ),
                Err(failure) => self.fail(index, failure),
            },
            None => match cursor.on_logits(logits, elapsed_ms, crate::shutdown::checkpoint) {
                Ok(()) => self.advance(index, presampled, settings, reuse),
                Err(failure) => self.fail(index, failure),
            },
        }
    }
}
