use super::*;
use crate::serve::{
    events::{ServeStats, Usage},
    output_partition::GenerationEnd,
    request_profile::RequestProfile,
};
use qwen_llm::{
    sampling::{Sampler, SamplingConfig},
    tokenizer::{TokError, Tokenize},
};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

struct Bytes;
impl Tokenize for Bytes {
    fn encode(&self, s: &str, _: bool) -> Result<Vec<i32>, TokError> {
        Ok(s.bytes().map(i32::from).collect())
    }
    fn try_decode_piece_bytes(&self, t: i32) -> Result<Vec<u8>, TokError> {
        Ok(vec![
            u8::try_from(t).map_err(|_| TokError::InvalidToken(t))?,
        ])
    }
    fn try_decode_piece(&self, t: i32) -> Result<String, TokError> {
        Ok(String::from_utf8_lossy(&self.try_decode_piece_bytes(t)?).into_owned())
    }
    fn try_decode(&self, t: &[i32]) -> Result<String, TokError> {
        Ok(String::from_utf8_lossy(&t.iter().map(|t| *t as u8).collect::<Vec<_>>()).into_owned())
    }
    fn decode(&self, t: &[i32]) -> String {
        self.try_decode(t).unwrap()
    }
    fn n_vocab(&self) -> u32 {
        256
    }
    fn bos(&self) -> Option<i32> {
        None
    }
    fn eos(&self) -> Option<i32> {
        Some(0)
    }
}
#[derive(Default)]
struct State {
    pos: usize,
    consumed: Vec<u32>,
    poisoned: bool,
    unready: bool,
}
struct Run {
    state: State,
    left: usize,
    prompt: usize,
    maximum: usize,
    sampler: Sampler,
}
#[derive(Default)]
struct Fake {
    feedback_bursts: bool,
    group_bursts: bool,
    token_batch: bool,
    token_batches: usize,
    bursts: usize,
    groups: usize,
    steps: Vec<usize>,
    prefills: Vec<usize>,
    reserves: Vec<u64>,
    commits: Vec<Vec<u32>>,
    refuse: usize,
    fail: usize,
    clock: Option<Instant>,
    prefill_service: Duration,
    decode_service: Duration,
    cancel_during: Option<Arc<AtomicBool>>,
    sampler_draws: Vec<usize>,
    sampling: Option<SamplingConfig>,
}
fn logits(s: &State) -> Vec<f32> {
    let mut v = vec![-100.; 256];
    v[b'a' as usize + s.consumed.len() % 26] = 100.;
    v
}
impl Runtime for Fake {
    type Run = Run;
    type State = State;
    fn now(&mut self) -> Instant {
        self.clock.unwrap_or_else(Instant::now)
    }
    fn extra_reserve(&self) -> u64 {
        7
    }
    fn begin(
        &mut self,
        p: &PreparedResponse,
        sink: &mut dyn GenerationSink,
    ) -> Result<Begun<Run>, BackendFailure> {
        self.reserves.push(sink.transport_reserve_bytes());
        let n = p.prompt.parse::<usize>().unwrap();
        Ok(Begun::Run(Box::new(Run {
            state: State::default(),
            left: n,
            prompt: n,
            maximum: p.request().max_output_tokens.unwrap_or(8),
            sampler: Sampler::new(self.sampling.unwrap_or(SamplingConfig {
                temperature: 0.,
                top_k: 1,
                top_p: 1.,
                min_p: 0.,
                seed: 1,
            }))
            .unwrap(),
        })))
    }
    fn rows(&self, _: &Run) -> usize {
        2048
    }
    fn prefill(
        &mut self,
        r: &mut Run,
        rows: usize,
        _: &mut dyn GenerationSink,
        reserve: u64,
    ) -> Result<Option<DecodeCursor>, BackendFailure> {
        if let Some(c) = &mut self.clock {
            *c += self.prefill_service;
        }
        let n = rows.min(r.left);
        self.prefills.push(n);
        self.reserves.push(reserve);
        r.left -= n;
        r.state.pos += n;
        if r.left > 0 {
            return Ok(None);
        }
        DecodeCursor::new(crate::serve::decode_loop::DecodeRequest {
            family: "scripted",
            logits: logits(&r.state),
            max_tokens: r.maximum,
            stop_tokens: &[0],
            vocab_size: 256,
        })
        .map(Some)
    }
    fn advance(
        &mut self,
        r: &mut Run,
        c: &mut DecodeCursor,
        s: &mut dyn GenerationSink,
        _: u64,
        presampled: Option<crate::serve::backend_v41::sample_pool::Sampled>,
    ) -> Result<Advance, BackendFailure> {
        c.advance(&mut r.sampler, presampled, &Bytes, s, || Ok(()))
    }
    fn sampler_mut(r: &mut Self::Run) -> Option<&mut Sampler> {
        Some(&mut r.sampler)
    }
    fn advance_single(
        &mut self,
        r: &mut Run,
        c: &mut DecodeCursor,
        s: &mut dyn GenerationSink,
        _: u64,
    ) -> Option<Result<Advance, BackendFailure>> {
        if !self.feedback_bursts {
            return None;
        }
        self.bursts += 1;
        Some((|| {
            let mut tokens = Vec::new();
            let limit = c.remaining_tokens().min(3);
            for i in 0..limit {
                if i > 0 {
                    r.state.pos += 1;
                    r.state.consumed.push(tokens[i - 1] as u32);
                }
                let token = b'a' as i32 + r.state.consumed.len() as i32 % 26;
                tokens.push(token);
                s.piece(&[token as u8])
                    .and_then(|()| s.tick())
                    .map_err(BackendFailure::Aborted)?;
                if !s.flush_ready().map_err(BackendFailure::Aborted)? {
                    break;
                }
            }
            let stats = qwen_llm::metal::GpuTokenLoopStats {
                sampled_tokens: tokens.len(),
                emitted_tokens: tokens.len(),
                forwarded_tokens: tokens.len() - 1,
                ..Default::default()
            };
            Ok(c.append_gpu_burst(&tokens, stats, 0.))
        })())
    }
    fn advance_group(
        &mut self,
        runs: &mut [&mut Run],
        cursors: &mut [&mut DecodeCursor],
        sinks: &mut [&mut dyn GenerationSink],
        tokens: &[u32],
        _reserves: &[u64],
    ) -> Option<Result<Vec<Advance>, BackendFailure>> {
        if !self.group_bursts {
            return None;
        }
        assert_eq!(runs.len(), cursors.len());
        assert_eq!(runs.len(), sinks.len());
        assert_eq!(runs.len(), tokens.len());
        self.groups += 1;
        Some((|| {
            let mut out = Vec::with_capacity(runs.len());
            for (i, ((run, cursor), sink)) in runs
                .iter_mut()
                .zip(cursors.iter_mut())
                .zip(sinks.iter_mut())
                .enumerate()
            {
                run.state.pos += 1;
                run.state.consumed.push(tokens[i]);
                cursor.on_logits(Ok(logits(&run.state)), 1., || Ok(()))?;
                let mut burst = Vec::new();
                for k in 0..cursor.remaining_tokens().min(2) {
                    if k > 0 {
                        run.state.pos += 1;
                        run.state.consumed.push(burst[k - 1] as u32);
                    }
                    let token = b'a' as i32 + run.state.consumed.len() as i32 % 26;
                    burst.push(token);
                    sink.piece(&[token as u8])
                        .and_then(|()| sink.tick())
                        .map_err(BackendFailure::Aborted)?;
                    if !sink.flush_ready().map_err(BackendFailure::Aborted)? {
                        break;
                    }
                }
                let stats = qwen_llm::metal::GpuTokenLoopStats {
                    sampled_tokens: burst.len(),
                    emitted_tokens: burst.len(),
                    forwarded_tokens: burst.len() - 1,
                    ..Default::default()
                };
                out.push(cursor.append_gpu_burst(&burst, stats, 0.));
            }
            Ok(out)
        })())
    }
    fn state(r: &mut Run) -> &mut State {
        &mut r.state
    }
    fn can_step(&self, s: &State) -> bool {
        !s.poisoned && !s.unready
    }
    fn poisoned(s: &State) -> bool {
        s.poisoned
    }
    fn step(&mut self, s: &mut State, t: u32) -> Result<Vec<f32>, String> {
        if let Some(c) = &mut self.clock {
            *c += self.decode_service;
        }
        self.steps.push(1);
        if !self.can_step(s) {
            return Err("unready state".into());
        }
        s.pos += 1;
        s.consumed.push(t);
        Ok(logits(s))
    }
    fn step_batch(
        &mut self,
        states: &mut [&mut State],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if let Some(c) = &mut self.clock {
            *c += self.decode_service;
        }
        self.steps.push(states.len());
        if self.refuse > 0 {
            self.refuse -= 1;
            return Err("preflight declined".into());
        }
        if let Some(flag) = self.cancel_during.take() {
            flag.store(true, Ordering::Release);
        }
        if self.fail > 0 {
            self.fail -= 1;
            for s in states {
                s.poisoned = true;
            }
            return Err("started batch failed".into());
        }
        Ok(states
            .iter_mut()
            .zip(tokens)
            .map(|(s, t)| {
                s.pos += 1;
                s.consumed.push(*t);
                logits(s)
            })
            .collect())
    }
    fn step_batch_sampled(
        &mut self,
        runs: &mut [&mut Run],
        tokens: &[u32],
    ) -> Option<Result<Vec<u32>, String>> {
        if !self.token_batch {
            return None;
        }
        self.token_batches += 1;
        let mut out = Vec::with_capacity(runs.len());
        for (run, &token) in runs.iter_mut().zip(tokens) {
            run.state.pos += 1;
            run.state.consumed.push(token);
            out.push(b'a' as u32 + run.state.consumed.len() as u32 % 26);
        }
        Some(Ok(out))
    }
    fn finish(
        &mut self,
        r: Run,
        g: &crate::GenerationResult,
        batch: serde_json::Value,
    ) -> Result<GenerationOutcome, BackendFailure> {
        assert!(!r.state.poisoned);
        assert_eq!(r.state.pos, r.prompt + g.transitions);
        assert_eq!(
            r.state.consumed,
            g.tokens[..g.transitions]
                .iter()
                .map(|x| *x as u32)
                .collect::<Vec<_>>()
        );
        self.sampler_draws.push(r.sampler.draws());
        self.commits.push(r.state.consumed);
        Ok(GenerationOutcome {
            end: GenerationEnd::TokenLimit,
            usage: Usage {
                input_tokens: r.prompt,
                output_tokens: g.tokens.len(),
                cached_tokens: 0,
            },
            stats: Some(ServeStats {
                matched_tokens: 0,
                restore_ms: 0.,
                prompt_tokens: r.prompt,
                seed: None,
                phases: Some(serde_json::json!({"batch":batch})),
            }),
        })
    }
}
#[derive(Default)]
struct Seen {
    queued: Vec<u8>,
    bytes: Vec<u8>,
    terminal: Option<Result<GenerationOutcome, (bool, u16)>>,
    credit: bool,
    completed: usize,
}
struct Sink {
    seen: Rc<RefCell<Seen>>,
    cancel: Arc<AtomicBool>,
    reserve: u64,
}
impl BatchSink for Sink {
    fn cancellation_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }
    fn transport_reserve_bytes(&self) -> u64 {
        self.reserve
    }
    fn check(&self) -> io::Result<()> {
        if self.cancel.load(Ordering::Acquire) {
            Err(io::Error::new(io::ErrorKind::ConnectionAborted, "gone"))
        } else {
            Ok(())
        }
    }
    fn queue(&mut self, b: &[u8]) {
        self.seen.borrow_mut().queued.extend_from_slice(b)
    }
    fn flush(&mut self) -> io::Result<bool> {
        let mut s = self.seen.borrow_mut();
        if s.credit {
            let b = std::mem::take(&mut s.queued);
            s.bytes.extend(b);
        }
        Ok(s.queued.is_empty())
    }
    fn complete(self: Box<Self>, r: Result<GenerationOutcome, BackendFailure>) {
        let mut s = self.seen.borrow_mut();
        s.completed += 1;
        assert_eq!(s.completed, 1);
        s.terminal = Some(r.map_err(|e| match e {
            BackendFailure::Aborted(_) => (true, 0),
            BackendFailure::Serve(e) => (false, e.status),
        }));
    }
}
fn submit(
    d: &mut Driver<Run>,
    prompt: usize,
    credit: bool,
    reserve: u64,
) -> (Rc<RefCell<Seen>>, Arc<AtomicBool>) {
    let profile = RequestProfile::test_qwen();
    let request = profile
        .parse(&serde_json::json!({"model":"test","input":"x","max_output_tokens":8}))
        .unwrap();
    let p = PreparedResponse {
        validated: profile.validate(request).unwrap(),
        prompt: prompt.to_string(),
        boundaries: None,
        tokens: None,
    };
    let seen = Rc::new(RefCell::new(Seen {
        credit,
        ..Seen::default()
    }));
    let cancel = Arc::new(AtomicBool::new(false));
    d.submit(Job {
        prepared: Arc::new(p),
        sink: Box::new(Sink {
            seen: seen.clone(),
            cancel: cancel.clone(),
            reserve,
        }),
    });
    (seen, cancel)
}
fn finish(d: &mut Driver<Run>, e: &mut Fake) {
    for _ in 0..200 {
        if d.active.is_empty() && d.pending.is_empty() {
            return;
        }
        d.pump(e);
    }
    panic!("scheduler did not drain");
}
fn terminal_ok(s: &Rc<RefCell<Seen>>) {
    assert!(s.borrow().terminal.as_ref().unwrap().is_ok());
    assert_eq!(s.borrow().bytes, b"abcdefgh");
}
#[test]
fn scheduler_batches_independent_streams_without_changing_token_lifecycle() {
    let mut d = Driver::new(3);
    let mut e = Fake::default();
    let cs: Vec<_> = (0..3).map(|_| submit(&mut d, 5, true, 100)).collect();
    finish(&mut d, &mut e);
    for (s, _) in cs {
        terminal_ok(&s);
    }
    assert!(e.steps.contains(&3));
    assert_eq!(e.commits.len(), 3);
    assert!(e.reserves.contains(&307));
}
#[test]
fn scheduler_serial_limit_keeps_one_forward_per_transition() {
    let mut d = Driver::new(1);
    let mut e = Fake::default();
    let cs: Vec<_> = (0..3).map(|_| submit(&mut d, 5, true, 100)).collect();
    finish(&mut d, &mut e);
    for (s, _) in cs {
        terminal_ok(&s);
    }
    assert!(e.steps.iter().all(|n| *n == 1));
}
fn run_sampling(
    workers: usize,
    sampling: Option<SamplingConfig>,
) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>) {
    let mut d = Driver::new(3).with_sample_workers(workers).unwrap();
    assert_eq!(d.sample_workers(), workers);
    let mut e = Fake {
        sampling,
        ..Fake::default()
    };
    let cs: Vec<_> = (0..3).map(|_| submit(&mut d, 5, true, 100)).collect();
    finish(&mut d, &mut e);
    if sampling.is_none() {
        for (s, _) in cs {
            terminal_ok(&s);
        }
    }
    (e.commits, e.sampler_draws, e.steps)
}

#[test]
fn pooled_sampling_preserves_tokens_rng_draws_and_step_schedule() {
    let serial = run_sampling(0, None);
    let pooled = run_sampling(2, None);
    assert_eq!(pooled.0, serial.0);
    assert_eq!(pooled.1, serial.1);
    assert_eq!(pooled.2, serial.2);
}

#[test]
fn pooled_sampling_matches_agent_temperature_rng() {
    // GLM's released defaults: temperature 1 and nucleus 0.95.
    let sampling = Some(SamplingConfig {
        temperature: 1.,
        top_k: 0,
        top_p: 0.95,
        min_p: 0.,
        seed: 42,
    });
    let serial = run_sampling(0, sampling);
    let pooled = run_sampling(2, sampling);
    assert_eq!(pooled.0, serial.0);
    assert_eq!(pooled.1, serial.1);
    assert_eq!(pooled.2, serial.2);
    assert!(serial.1.iter().all(|&draws| draws > 0));
}

#[test]
fn scheduler_loaded_prefill_is_bounded_and_alternates_with_decode() {
    let mut d = Driver::new(3).with_prefill_policy(false, 10);
    let mut e = Fake::default();
    let a = submit(&mut d, 5, true, 100);
    let b = submit(&mut d, 1800, true, 100);
    finish(&mut d, &mut e);
    terminal_ok(&a.0);
    terminal_ok(&b.0);
    assert!(e.prefills.contains(&512));
    assert!(!e.prefills.contains(&1800));
}
#[test]
fn scheduler_preflight_decline_retries_only_as_individual_steps() {
    let mut d = Driver::new(3);
    let mut e = Fake {
        refuse: 1,
        ..Fake::default()
    };
    let cs: Vec<_> = (0..3).map(|_| submit(&mut d, 5, true, 100)).collect();
    finish(&mut d, &mut e);
    for (s, _) in cs {
        terminal_ok(&s);
    }
    assert_eq!(e.refuse, 0);
    assert!(!d.take_native_failure());
}
#[test]
fn scheduler_post_start_failure_never_retries_the_poisoned_batch() {
    let mut d = Driver::new(3);
    let mut e = Fake {
        fail: 1,
        ..Fake::default()
    };
    let cs: Vec<_> = (0..3).map(|_| submit(&mut d, 5, true, 100)).collect();
    finish(&mut d, &mut e);
    assert_eq!(
        cs.iter()
            .filter(|(s, _)| s.borrow().terminal.as_ref().unwrap().is_err())
            .count(),
        2
    );
    assert_eq!(e.commits.len(), 1);
    assert!(d.take_native_failure());
}
#[test]
fn scheduler_cancel_during_batch_drops_only_that_request() {
    let mut d = Driver::new(3);
    let mut e = Fake::default();
    let cs: Vec<_> = (0..3).map(|_| submit(&mut d, 5, true, 100)).collect();
    e.cancel_during = Some(cs[0].1.clone());
    finish(&mut d, &mut e);
    assert!(cs[0].0.borrow().terminal.as_ref().unwrap().is_err());
    for (s, _) in &cs[1..] {
        terminal_ok(s);
    }
    assert_eq!(e.commits.len(), 2);
    assert!(!d.take_native_failure());
}
#[test]
fn scheduler_slow_client_parks_without_blocking_other_requests() {
    let mut d = Driver::new(3);
    let mut e = Fake::default();
    let a = submit(&mut d, 5, false, 100);
    let b = submit(&mut d, 5, true, 100);
    let c = submit(&mut d, 5, true, 100);
    for _ in 0..60 {
        d.pump(&mut e);
    }
    assert!(a.0.borrow().terminal.is_none());
    terminal_ok(&b.0);
    terminal_ok(&c.0);
    a.0.borrow_mut().credit = true;
    finish(&mut d, &mut e);
    terminal_ok(&a.0);
}
#[test]
fn scheduler_park_timeout_releases_only_the_slow_client() {
    let mut d = Driver::new(3);
    d.park_timeout = Duration::ZERO;
    let mut e = Fake::default();
    let a = submit(&mut d, 5, false, 100);
    let b = submit(&mut d, 5, true, 100);
    finish(&mut d, &mut e);
    assert!(a.0.borrow().terminal.as_ref().unwrap().is_err());
    terminal_ok(&b.0);
}
#[test]
fn scheduler_shutdown_completes_active_and_pending_exactly_once() {
    let mut d = Driver::new(2);
    let mut e = Fake::default();
    let cs: Vec<_> = (0..3).map(|_| submit(&mut d, 5, true, 100)).collect();
    d.pump(&mut e);
    d.abort_all();
    d.abort_all();
    assert_eq!(
        cs[0]
            .0
            .borrow()
            .terminal
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_err(),
        &(true, 0)
    );
    for (s, _) in &cs[1..] {
        assert_eq!(
            s.borrow().terminal.as_ref().unwrap().as_ref().unwrap_err(),
            &(false, 503)
        );
    }
    assert!(e.commits.is_empty());
}
#[test]
fn scheduler_reserve_overflow_refuses_before_starting_a_request() {
    let mut d = Driver::new(3);
    let mut e = Fake::default();
    let a = submit(&mut d, 5, true, u64::MAX);
    finish(&mut d, &mut e);
    assert_eq!(
        a.0.borrow()
            .terminal
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_err(),
        &(false, 500)
    );
    assert!(e.reserves.is_empty());
}

#[test]
fn scheduler_unready_member_fails_alone_before_forming_the_batch() {
    let mut d = Driver::new(3);
    let mut e = Fake::default();
    let cs: Vec<_> = (0..3).map(|_| submit(&mut d, 5, true, 100)).collect();
    for _ in 0..20 {
        d.pump(&mut e);
        if d.active.len() == 3 && d.active.iter().all(|a| a.cursor.is_some()) {
            break;
        }
    }
    assert_eq!(d.active.len(), 3);
    d.active[1].run.as_mut().unwrap().state.unready = true;
    finish(&mut d, &mut e);
    terminal_ok(&cs[0].0);
    terminal_ok(&cs[2].0);
    assert_eq!(
        cs[1]
            .0
            .borrow()
            .terminal
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap_err(),
        &(false, 500)
    );
    assert_eq!(e.commits.len(), 2);
    assert!(!d.take_native_failure());
}

#[test]
fn scheduler_fair_prefill_keeps_normal_geometry_and_pays_elapsed_decode_service() {
    let mut d = Driver::new(3);
    let mut e = Fake {
        clock: Some(Instant::now()),
        prefill_service: Duration::from_millis(900),
        decode_service: Duration::from_millis(30),
        ..Fake::default()
    };
    let a = submit(&mut d, 5, true, 100);
    d.pump(&mut e);
    let b = submit(&mut d, 5000, true, 100);
    d.pump(&mut e);
    assert_eq!(e.prefills, vec![5, 2048]);
    assert_eq!(d.decode_budget, Duration::from_millis(100));
    let n = e.prefills.len();
    for _ in 0..4 {
        d.pump(&mut e);
    }
    assert_eq!(e.prefills.len(), n);
    assert!(d.decode_budget.is_zero());
    d.pump(&mut e);
    assert_eq!(e.prefills.last(), Some(&2048));
    finish(&mut d, &mut e);
    terminal_ok(&a.0);
    terminal_ok(&b.0);
}
#[test]
fn scheduler_fair_service_credit_is_capped_and_cleared_without_mixed_work() {
    let mut d = Driver::new(3).with_prefill_policy(true, 90);
    let mut e = Fake {
        clock: Some(Instant::now()),
        prefill_service: Duration::from_secs(20),
        decode_service: Duration::from_millis(30),
        ..Fake::default()
    };
    let a = submit(&mut d, 5, true, 100);
    d.pump(&mut e);
    let b = submit(&mut d, 5000, true, 100);
    d.pump(&mut e);
    assert_eq!(d.decode_budget, Duration::from_secs(1));
    a.1.store(true, Ordering::Release);
    d.pump(&mut e);
    assert!(d.decode_budget.is_zero());
    finish(&mut d, &mut e);
    terminal_ok(&b.0);
}

#[test]
fn feedback_bursts_yield_before_existing_peers_and_keep_width_accounting() {
    let mut d = Driver::new(2);
    let mut e = Fake {
        feedback_bursts: true,
        ..Default::default()
    };
    let a = submit(&mut d, 5, true, 100);
    d.pump(&mut e);
    assert_eq!(e.bursts, 1);
    assert_eq!(d.active[0].steps, 2);
    assert_eq!(d.active[0].width_sum, 2);
    let b = submit(&mut d, 5, true, 100);
    d.pump(&mut e);
    let bursts = e.bursts;
    for _ in 0..5 {
        if d.active.len() != 2 {
            break;
        }
        d.pump(&mut e);
        if d.active.len() == 2 {
            assert_eq!(e.bursts, bursts);
        }
    }
    finish(&mut d, &mut e);
    terminal_ok(&a.0);
    terminal_ok(&b.0);
}

#[test]
fn scheduler_token_batch_skips_host_sampling_and_keeps_lifecycle() {
    let mut d = Driver::new(3);
    let mut e = Fake {
        token_batch: true,
        ..Fake::default()
    };
    let cs: Vec<_> = (0..3).map(|_| submit(&mut d, 5, true, 100)).collect();
    finish(&mut d, &mut e);
    for (s, _) in cs {
        terminal_ok(&s);
    }
    assert!(e.token_batches > 0);
    assert!(
        !e.steps.contains(&3),
        "the token path must not also run step_batch"
    );
    assert_eq!(e.commits.len(), 3);
    assert!(!d.take_native_failure());
}

#[test]
fn scheduler_group_burst_batches_all_members_and_keeps_lifecycle() {
    let mut d = Driver::new(3);
    let mut e = Fake {
        group_bursts: true,
        ..Fake::default()
    };
    let cs: Vec<_> = (0..3).map(|_| submit(&mut d, 5, true, 100)).collect();
    finish(&mut d, &mut e);
    for (s, _) in cs {
        terminal_ok(&s);
    }
    assert!(e.groups > 0);
    assert!(
        !e.steps.contains(&3),
        "a full-width round must use the group path, not step_batch"
    );
    assert_eq!(e.commits.len(), 3);
    assert!(!d.take_native_failure());
}

#[test]
fn scheduler_group_burst_backpressure_keeps_each_member_forwardable() {
    let mut d = Driver::new(2);
    let mut e = Fake {
        group_bursts: true,
        ..Fake::default()
    };
    let a = submit(&mut d, 5, false, 100);
    let b = submit(&mut d, 5, true, 100);
    for _ in 0..40 {
        d.pump(&mut e);
        if b.0.borrow().terminal.is_some() {
            break;
        }
    }
    terminal_ok(&b.0);
    assert!(a.0.borrow().terminal.is_none());
    a.0.borrow_mut().credit = true;
    finish(&mut d, &mut e);
    terminal_ok(&a.0);
}

#[test]
fn feedback_burst_backpressure_parks_before_any_forward() {
    let mut d = Driver::new(2);
    let mut e = Fake {
        feedback_bursts: true,
        ..Default::default()
    };
    let a = submit(&mut d, 5, false, 100);
    d.pump(&mut e);
    assert_eq!(e.bursts, 1);
    assert_eq!(
        d.active[0].run.as_ref().unwrap().state.consumed,
        Vec::<u32>::new()
    );
    assert!(d.active[0].parked.is_some());
    let b = submit(&mut d, 5, true, 100);
    d.pump(&mut e);
    for _ in 0..20 {
        d.pump(&mut e);
        if b.0.borrow().terminal.is_some() {
            break;
        }
    }
    terminal_ok(&b.0);
    a.0.borrow_mut().credit = true;
    finish(&mut d, &mut e);
    terminal_ok(&a.0);
}
