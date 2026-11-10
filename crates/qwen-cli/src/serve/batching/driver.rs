//! Owner-thread scheduling over a family's resumable request phases.
//! Native state, cache policy and admission stay in the runtime adapter.
use super::{BatchSink, Job, shutting_down};
use crate::serve::{
    backend_v41::sample_pool::{SampleJob, SamplePool, Sampled},
    decode_cursor::{Advance, DecodeCursor},
    http::{BackendFailure, GenerationOutcome, GenerationSink, PreparedResponse},
};
use qwen_llm::sampling::{SampledToken, Sampler};
use std::{
    collections::VecDeque,
    io,
    time::{Duration, Instant},
};

pub(crate) enum Begun<R> {
    Run(Box<R>),
    Complete(GenerationOutcome),
}

/// One member's completed batch step: full logits for host sampling, or the
/// token a family already sampled on the GPU.
pub(crate) enum StepOutput {
    Logits(Vec<f32>),
    Token(u32),
}

pub(crate) trait Runtime {
    type Run;
    type State;
    fn extra_reserve(&self) -> u64;
    fn begin(
        &mut self,
        prepared: &PreparedResponse,
        sink: &mut dyn GenerationSink,
    ) -> Result<Begun<Self::Run>, BackendFailure>;
    fn rows(&self, run: &Self::Run) -> usize;
    fn now(&mut self) -> Instant {
        Instant::now()
    }
    fn prefill(
        &mut self,
        run: &mut Self::Run,
        rows: usize,
        sink: &mut dyn GenerationSink,
        reserve: u64,
    ) -> Result<Option<DecodeCursor>, BackendFailure>;
    fn advance(
        &mut self,
        run: &mut Self::Run,
        cursor: &mut DecodeCursor,
        sink: &mut dyn GenerationSink,
        reserve: u64,
        presampled: Option<Sampled>,
    ) -> Result<Advance, BackendFailure>;
    /// The member's sampler, for a pool that samples a whole step before any
    /// member is consumed. Runtimes without pooled sampling leave this None.
    fn sampler_mut(_run: &mut Self::Run) -> Option<&mut Sampler> {
        None
    }
    /// Optional bounded feedback burst, only with one active request and no
    /// queued peer. Returning None retains ordinary token-at-a-time advance.
    fn advance_single(
        &mut self,
        _run: &mut Self::Run,
        _cursor: &mut DecodeCursor,
        _sink: &mut dyn GenerationSink,
        _reserve: u64,
    ) -> Option<Result<Advance, BackendFailure>> {
        None
    }
    fn state(run: &mut Self::Run) -> &mut Self::State;
    fn can_step(&self, state: &Self::State) -> bool;
    fn poisoned(state: &Self::State) -> bool;
    fn step(&mut self, state: &mut Self::State, token: u32) -> Result<Vec<f32>, String>;
    fn step_batch(
        &mut self,
        states: &mut [&mut Self::State],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, String>;
    /// Optional batch step that samples every member on the GPU and returns
    /// only tokens. Returning None keeps `step_batch` and host sampling.
    fn step_batch_sampled(
        &mut self,
        _runs: &mut [&mut Self::Run],
        _tokens: &[u32],
    ) -> Option<Result<Vec<u32>, String>> {
        None
    }
    fn finish(
        &mut self,
        run: Self::Run,
        generation: &crate::GenerationResult,
        batch: serde_json::Value,
    ) -> Result<GenerationOutcome, BackendFailure>;
}

pub(crate) struct Publish<'a> {
    pub sink: &'a mut dyn BatchSink,
    pub reserve: u64,
}
impl GenerationSink for Publish<'_> {
    fn cancellation_flag(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        Some(self.sink.cancellation_flag())
    }
    fn transport_reserve_bytes(&self) -> u64 {
        self.reserve
    }
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.sink.check()?;
        self.sink.queue(bytes);
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        self.sink.check()
    }
    fn flush_ready(&mut self) -> io::Result<bool> {
        self.sink.check()?;
        self.sink.flush()
    }
}
struct Pending {
    job: Job,
    submitted: Instant,
}
struct Active<R> {
    sink: Box<dyn BatchSink>,
    run: Option<Box<R>>,
    cursor: Option<DecodeCursor>,
    next: Option<u32>,
    finished: Option<GenerationOutcome>,
    reserve: u64,
    queue_ms: f64,
    parked: Option<Instant>,
    parked_ms: f64,
    steps: usize,
    feedback_steps: usize,
    width_sum: usize,
    width_max: usize,
    batched: usize,
    loaded_prefills: usize,
}
/// One bounded prefill call or one decode round per pump. Client delivery
/// never waits: parked clients leave decode rounds until they drain.
pub(crate) struct Driver<R> {
    max_active: usize,
    pending: VecDeque<Pending>,
    active: Vec<Active<R>>,
    prefer_prefill: bool,
    prefill_fair: bool,
    decode_percent: usize,
    decode_budget: Duration,
    native_failed: bool,
    park_timeout: Duration,
    sample_pool: Option<SamplePool>,
}
impl<R> Driver<R> {
    pub(crate) fn new(max_active: usize) -> Self {
        Self {
            max_active,
            pending: VecDeque::new(),
            active: Vec::new(),
            prefer_prefill: false,
            prefill_fair: true,
            decode_percent: 10,
            decode_budget: Duration::ZERO,
            native_failed: false,
            park_timeout: Duration::from_secs(30),
            sample_pool: None,
        }
    }
    /// Sample a multi-member step on persistent threads beside the owner.
    /// `0` keeps sampling serial on the owner, exactly as before.
    pub(crate) fn with_sample_workers(mut self, workers: usize) -> std::io::Result<Self> {
        self.sample_pool = (workers > 0)
            .then(|| SamplePool::new(workers))
            .transpose()?;
        Ok(self)
    }
    pub(crate) fn sample_workers(&self) -> usize {
        self.sample_pool.as_ref().map_or(0, SamplePool::workers)
    }
    pub(crate) fn with_prefill_policy(mut self, fair: bool, percent: usize) -> Self {
        assert!((1..=90).contains(&percent));
        self.prefill_fair = fair;
        self.decode_percent = percent;
        self
    }
    pub(crate) fn max_active(&self) -> usize {
        self.max_active
    }
    pub(crate) fn submit(&mut self, job: Job) {
        self.pending.push_back(Pending {
            job,
            submitted: Instant::now(),
        });
    }
    pub(crate) fn take_native_failure(&mut self) -> bool {
        std::mem::take(&mut self.native_failed)
    }
    pub(crate) fn abort_all(&mut self) {
        for p in self.pending.drain(..) {
            p.job.sink.complete(Err(shutting_down().into()));
        }
        for a in self.active.drain(..) {
            a.sink.complete(Err(BackendFailure::Aborted(io::Error::new(
                io::ErrorKind::Interrupted,
                "serve is shutting down",
            ))));
        }
    }
    fn total_reserve<E: Runtime<Run = R>>(&self, runtime: &E) -> Option<u64> {
        self.active
            .iter()
            .try_fold(runtime.extra_reserve(), |n, a| n.checked_add(a.reserve))
    }
    fn retire(&mut self, index: usize, result: Result<GenerationOutcome, BackendFailure>) {
        self.active.remove(index).sink.complete(result);
    }
    fn fail(&mut self, index: usize, error: BackendFailure) {
        let error = match self.active[index].sink.check() {
            Err(e) => BackendFailure::Aborted(e),
            Ok(()) => error,
        };
        self.retire(index, Err(error));
    }
    fn publish(&mut self, index: usize) -> bool {
        let a = &mut self.active[index];
        match a.sink.flush() {
            Err(e) => {
                self.retire(index, Err(BackendFailure::Aborted(e)));
                true
            }
            Ok(true) => {
                if let Some(start) = a.parked.take() {
                    a.parked_ms += start.elapsed().as_secs_f64() * 1e3;
                }
                if let Some(out) = a.finished.take() {
                    self.retire(index, Ok(out));
                    true
                } else {
                    false
                }
            }
            Ok(false) => {
                a.parked.get_or_insert_with(Instant::now);
                false
            }
        }
    }
    fn advance<E: Runtime<Run = R>>(
        &mut self,
        index: usize,
        presampled: Option<Sampled>,
        runtime: &mut E,
    ) {
        let Some(reserve) = self.total_reserve(runtime) else {
            return self.fail(
                index,
                crate::serve::output_memory::size_overflow("batch transport").into(),
            );
        };
        let lone = self.active.len() == 1 && self.pending.is_empty();
        let a = &mut self.active[index];
        let run = a.run.as_mut().expect("decoding run");
        let cursor = a.cursor.as_mut().expect("cursor");
        let before = cursor.completed_transitions();
        let mut sink = Publish {
            sink: a.sink.as_mut(),
            reserve,
        };
        let outcome = if lone {
            runtime.advance_single(run, cursor, &mut sink, reserve)
        } else {
            None
        }
        .unwrap_or_else(|| runtime.advance(run, cursor, &mut sink, reserve, presampled));
        let extra = cursor.completed_transitions().saturating_sub(before);
        a.steps += extra;
        a.feedback_steps += extra;
        a.width_sum += extra;
        if extra > 0 {
            a.width_max = a.width_max.max(1);
        }
        match outcome {
            Err(e) => self.fail(index, e),
            Ok(Advance::Forward(token)) => {
                a.next = Some(token);
                self.publish(index);
            }
            Ok(Advance::Done(generation)) => {
                if let Err(e) = a.sink.check() {
                    return self.retire(index, Err(BackendFailure::Aborted(e)));
                }
                let run = a.run.take().expect("run");
                a.cursor = None;
                let stats = serde_json::json!({"max_active":self.max_active,"queue_ms":a.queue_ms,"parked_ms":a.parked_ms,"stepped_transitions":a.steps,"gpu_feedback_transitions":a.feedback_steps,"width_max":a.width_max,"width_mean":(a.steps>0).then(||a.width_sum as f64/a.steps as f64),"batched_transitions":a.batched,"prefill_loaded_rows":if self.prefill_fair {runtime.rows(&run)} else {512},"prefill_fair":self.prefill_fair,"decode_percent":self.prefill_fair.then_some(self.decode_percent),"prefill_loaded_calls":a.loaded_prefills});
                match runtime.finish(*run, &generation, stats) {
                    Ok(out) => {
                        self.active[index].finished = Some(out);
                        self.publish(index);
                    }
                    Err(e) => self.fail(index, e),
                }
            }
        }
    }
    /// Run every member's sampler on the pool before any member is consumed.
    /// Samplers and logits move out and back; a job keeps its result, logits
    /// and next RNG state exactly as owner sampling would.
    fn presample<E: Runtime<Run = R>>(
        &mut self,
        results: &mut [(usize, Result<StepOutput, String>, f64, usize)],
    ) -> Vec<Option<Sampled>> {
        let mut presampled: Vec<Option<Sampled>> = results.iter().map(|_| None).collect();
        let Some(pool) = self.sample_pool.as_mut() else {
            return presampled;
        };
        let members: Vec<usize> = (0..results.len())
            .filter(|&k| matches!(results[k].1, Ok(StepOutput::Logits(_))))
            .collect();
        if members.len() < 2 {
            return presampled;
        }
        let mut jobs = Vec::with_capacity(members.len());
        for &k in &members {
            let index = results[k].0;
            let run = self.active[index].run.as_mut().expect("decoding run");
            let sampler =
                E::sampler_mut(run).expect("a pooled runtime exposes every member's sampler");
            let placeholder = Sampler::new(qwen_llm::sampling::SamplingConfig::default())
                .expect("default sampler");
            let logits = match &mut results[k].1 {
                Ok(StepOutput::Logits(v)) => std::mem::take(v),
                _ => unreachable!("filtered to logits"),
            };
            jobs.push(SampleJob {
                sampler: std::mem::replace(sampler, placeholder),
                logits,
                sampled: None,
            });
        }
        for (&k, job) in members.iter().zip(pool.sample_all(jobs)) {
            let index = results[k].0;
            let run = self.active[index].run.as_mut().expect("decoding run");
            *E::sampler_mut(run).expect("sampler") = job.sampler;
            if let Ok(StepOutput::Logits(v)) = &mut results[k].1 {
                *v = job.logits;
            }
            presampled[k] = job.sampled;
        }
        presampled
    }

    fn step_one<E: Runtime<Run = R>>(
        &mut self,
        index: usize,
        runtime: &mut E,
    ) -> (usize, Result<StepOutput, String>, f64, usize) {
        let a = &mut self.active[index];
        let start = Instant::now();
        let result = runtime
            .step(
                E::state(a.run.as_mut().expect("run")),
                a.next.expect("pending token"),
            )
            .map(StepOutput::Logits);
        if result.is_err() && E::poisoned(E::state(a.run.as_mut().expect("run"))) {
            self.native_failed = true;
        }
        (index, result, start.elapsed().as_secs_f64() * 1e3, 1)
    }
    fn decode<E: Runtime<Run = R>>(&mut self, members: Vec<usize>, runtime: &mut E) {
        let (mut batch, mut results) = (Vec::new(), Vec::new());
        for i in members {
            if runtime.can_step(E::state(self.active[i].run.as_mut().expect("run"))) {
                batch.push(i)
            } else {
                results.push(self.step_one(i, runtime));
            }
        }
        if batch.len() == 1 {
            results.push(self.step_one(batch[0], runtime));
        }
        if batch.len() > 1 {
            let tokens: Vec<_> = batch
                .iter()
                .map(|&i| self.active[i].next.expect("pending token"))
                .collect();
            let start = Instant::now();
            let mut runs: Vec<&mut R> = self
                .active
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| batch.contains(i))
                .map(|(_, a)| &mut **a.run.as_mut().expect("run"))
                .collect();
            let sampled = runtime.step_batch_sampled(&mut runs, &tokens);
            drop(runs);
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            match sampled {
                Some(Ok(sampled)) if sampled.len() == batch.len() => {
                    for (i, token) in batch.iter().copied().zip(sampled) {
                        results.push((i, Ok(StepOutput::Token(token)), elapsed, batch.len()));
                    }
                }
                Some(Ok(_)) => {
                    self.native_failed = true;
                    let message = "batched GPU sampling returned wrong row count".to_string();
                    for i in batch.iter().copied() {
                        results.push((i, Err(message.clone()), elapsed, batch.len()));
                    }
                }
                Some(Err(message)) => {
                    self.native_failed = true;
                    let message = format!("batched GPU sampling failed: {message}");
                    for i in batch.iter().copied() {
                        results.push((i, Err(message.clone()), elapsed, batch.len()));
                    }
                }
                None => {
                    let start = Instant::now();
                    let mut states: Vec<_> = self
                        .active
                        .iter_mut()
                        .enumerate()
                        .filter(|(i, _)| batch.contains(i))
                        .map(|(_, a)| E::state(a.run.as_mut().expect("run")))
                        .collect();
                    let result = runtime.step_batch(&mut states, &tokens);
                    let poisoned = states.iter().any(|s| E::poisoned(s));
                    let elapsed = start.elapsed().as_secs_f64() * 1e3;
                    match result {
                        Ok(logits) if logits.len() == batch.len() => {
                            for (i, v) in batch.iter().copied().zip(logits) {
                                results.push((i, Ok(StepOutput::Logits(v)), elapsed, batch.len()));
                            }
                        }
                        Err(_) if !poisoned => {
                            for i in batch {
                                results.push(self.step_one(i, runtime));
                            }
                        }
                        result => {
                            self.native_failed = true;
                            let message = result.err().unwrap_or_else(|| {
                                "batched forward returned wrong row count".into()
                            });
                            let message = format!("batched decode step failed: {message}");
                            for i in batch.iter().copied() {
                                results.push((i, Err(message.clone()), elapsed, batch.len()));
                            }
                        }
                    }
                }
            }
        }
        // Descending indices allow a completion/failure to retire one member
        // without changing the identity of any result still to be delivered.
        results.sort_by_key(|x| std::cmp::Reverse(x.0));
        let mut presampled = self.presample::<E>(&mut results);
        for ((index, result, elapsed, width), presampled) in
            results.into_iter().zip(presampled.drain(..))
        {
            let a = &mut self.active[index];
            a.next = None;
            let outcome = match result {
                Err(e) => {
                    self.fail(
                        index,
                        crate::serve::items::ServeError::server_error(e).into(),
                    );
                    continue;
                }
                Ok(StepOutput::Logits(logits)) => a
                    .cursor
                    .as_mut()
                    .expect("cursor")
                    .on_logits(Ok(logits), elapsed, crate::shutdown::checkpoint)
                    .map(|()| presampled),
                Ok(StepOutput::Token(token)) => a
                    .cursor
                    .as_mut()
                    .expect("cursor")
                    .on_token(elapsed, crate::shutdown::checkpoint)
                    .map(|()| {
                        Some(Ok(SampledToken {
                            token: token as i32,
                            candidate_index: 0,
                        }))
                    }),
            };
            match outcome {
                Err(e) => self.fail(index, e),
                Ok(presampled) => {
                    let a = &mut self.active[index];
                    a.steps += 1;
                    a.width_sum += width;
                    a.width_max = a.width_max.max(width);
                    a.batched += usize::from(width > 1);
                    self.advance(index, presampled, runtime);
                }
            }
        }
    }

    pub(crate) fn pump<E: Runtime<Run = R>>(&mut self, runtime: &mut E) -> bool {
        let mut progress = false;
        for i in (0..self.active.len()).rev() {
            if let Err(e) = self.active[i].sink.check() {
                self.retire(i, Err(BackendFailure::Aborted(e)));
                progress = true;
                continue;
            }
            if self.active[i]
                .parked
                .is_some_and(|t| t.elapsed() >= self.park_timeout)
            {
                self.retire(
                    i,
                    Err(BackendFailure::Aborted(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "slow batch client",
                    ))),
                );
                progress = true;
                continue;
            }
            progress |= self.publish(i);
        }
        while self
            .pending
            .front()
            .is_some_and(|p| p.job.sink.check().is_err())
        {
            let p = self.pending.pop_front().expect("front");
            p.job
                .sink
                .complete(Err(BackendFailure::Aborted(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "request cancelled before start",
                ))));
            progress = true;
        }
        if !self.pending.is_empty() && self.active.len() < self.max_active {
            let p = self.pending.pop_front().expect("pending");
            let mut sink = p.job.sink;
            let queue_ms = p.submitted.elapsed().as_secs_f64() * 1e3;
            let own = sink.transport_reserve_bytes();
            let reserve = self.total_reserve(runtime).and_then(|n| n.checked_add(own));
            match reserve {
                None => sink.complete(Err(crate::serve::output_memory::size_overflow(
                    "batch transport",
                )
                .into())),
                Some(reserve) => match runtime.begin(
                    &p.job.prepared,
                    &mut Publish {
                        sink: sink.as_mut(),
                        reserve,
                    },
                ) {
                    Err(e) => sink.complete(Err(e)),
                    Ok(result) => {
                        let (run, finished) = match result {
                            Begun::Run(r) => (Some(r), None),
                            Begun::Complete(o) => (None, Some(o)),
                        };
                        self.active.push(Active {
                            sink,
                            run,
                            cursor: None,
                            next: None,
                            finished,
                            reserve: own,
                            queue_ms,
                            parked: None,
                            parked_ms: 0.0,
                            steps: 0,
                            feedback_steps: 0,
                            width_sum: 0,
                            width_max: 0,
                            batched: 0,
                            loaded_prefills: 0,
                        });
                    }
                },
            }
            progress = true;
        }
        let prefill = self.active.iter().position(|a| {
            a.run.is_some() && a.cursor.is_none() && a.finished.is_none() && a.parked.is_none()
        });
        let members: Vec<_> = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, a)| a.next.is_some() && a.parked.is_none())
            .map(|(i, _)| i)
            .collect();
        let mixed = prefill.is_some() && !members.is_empty();
        if !mixed {
            self.decode_budget = Duration::ZERO;
        }
        let prefill_turn = members.is_empty()
            || if self.prefill_fair {
                self.decode_budget.is_zero()
            } else {
                self.prefer_prefill
            };
        let started = runtime.now();
        if let Some(i) = prefill.filter(|_| prefill_turn) {
            let Some(reserve) = self.total_reserve(runtime) else {
                self.fail(
                    i,
                    crate::serve::output_memory::size_overflow("batch transport").into(),
                );
                return true;
            };
            let a = &mut self.active[i];
            let run = a.run.as_mut().expect("prefill run");
            let loaded = !members.is_empty();
            let rows = if loaded && !self.prefill_fair {
                512
            } else {
                runtime.rows(run).max(1)
            };
            match runtime.prefill(
                run,
                rows,
                &mut Publish {
                    sink: a.sink.as_mut(),
                    reserve,
                },
                reserve,
            ) {
                Err(e) => self.fail(i, e),
                Ok(cursor) => {
                    if loaded {
                        a.loaded_prefills += 1;
                    }
                    if let Some(cursor) = cursor {
                        a.cursor = Some(cursor);
                        self.advance(i, None, runtime);
                    }
                }
            }
            if self.prefill_fair && mixed {
                self.decode_budget = runtime
                    .now()
                    .saturating_duration_since(started)
                    .mul_f64(self.decode_percent as f64 / (100 - self.decode_percent) as f64)
                    .min(Duration::from_secs(1));
            }
            self.prefer_prefill = false;
            return true;
        }
        if !members.is_empty() {
            self.decode(members, runtime);
            self.decode_budget = self
                .decode_budget
                .saturating_sub(runtime.now().saturating_duration_since(started));
            self.prefer_prefill = true;
            return true;
        }
        progress
    }
}

#[cfg(test)]
#[path = "driver_tests.rs"]
mod tests;
