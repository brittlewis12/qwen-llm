//! Batch scheduler lifecycle against a scripted engine and in-memory sinks.
use super::scheduler::{BatchEngine, LOADED_PREFILL_ROWS, Scheduler};
use super::*;
use crate::serve::batching::{BatchSink, Job};
use crate::serve::http::PreparedResponse;
use serde_json::{Value, json};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

const VOCAB: usize = 258;

/// A sequence whose next token is a pure function of its consumed tokens, so
/// batched and serial execution must agree exactly.
#[derive(Debug)]
struct Seq {
    tokens: Vec<u32>,
    prompt: usize,
    poisoned: bool,
    unready: bool,
    capacity: usize,
}

fn next_logits(seq: &Seq) -> Vec<f32> {
    let generated = seq.tokens.len() - seq.prompt;
    let budget = 3 + seq.prompt % 5;
    let token = if generated >= budget {
        tokenizer_fixture::EOS
    } else {
        u32::from(b'a') + seq.tokens.iter().sum::<u32>() % 26
    };
    let mut logits = vec![0.0; VOCAB];
    logits[token as usize] = 100.0;
    logits
}

#[derive(Default)]
struct Log {
    /// (rows, with_logits) per prefill call, tagged by prompt length.
    prefill: Vec<(usize, usize)>,
    /// Width of each step call (1 = plain step).
    steps: Vec<usize>,
    /// Whether every batched step received only the shutdown flag.
    batch_flags_are_shutdown: Vec<bool>,
    reserves: Vec<u64>,
    events: Vec<String>,
}

#[derive(Default)]
struct StubEngine {
    log: Rc<RefCell<Log>>,
    refuse_batches: usize,
    fail_batches: usize,
}

impl execution::Engine<Seq> for StubEngine {
    fn admit(&mut self, _: usize, _: usize, reserve: u64) -> std::result::Result<(), StateError> {
        self.log.borrow_mut().reserves.push(reserve);
        Ok(())
    }
    fn new_state(&mut self, required: usize) -> std::result::Result<Seq, ServeError> {
        Ok(Seq {
            tokens: Vec::new(),
            prompt: 0,
            poisoned: false,
            unready: false,
            capacity: required,
        })
    }
    fn prefill(
        &mut self,
        state: &mut Seq,
        tokens: &[u32],
        _: &AtomicBool,
        with_logits: bool,
    ) -> std::result::Result<Vec<f32>, ServeError> {
        let mut log = self.log.borrow_mut();
        log.prefill.push((tokens.len(), state.tokens.len()));
        log.events.push(format!("prefill {}", tokens.len()));
        state.tokens.extend_from_slice(tokens);
        if !with_logits {
            return Ok(Vec::new());
        }
        state.prompt = state.tokens.len();
        Ok(next_logits(state))
    }
    fn step(
        &mut self,
        state: &mut Seq,
        token: u32,
        _: &AtomicBool,
    ) -> std::result::Result<Vec<f32>, String> {
        self.log.borrow_mut().steps.push(1);
        self.log.borrow_mut().events.push("step 1".into());
        if state.unready || state.poisoned {
            return Err("sequence is provisional or poisoned by partial execution".into());
        }
        state.tokens.push(token);
        Ok(next_logits(state))
    }
    fn now(&mut self) -> Duration {
        Duration::ZERO
    }
}

impl BatchEngine<Seq> for StubEngine {
    fn step_batch(
        &mut self,
        states: &mut [&mut Seq],
        tokens: &[u32],
        cancel: &AtomicBool,
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        {
            let mut log = self.log.borrow_mut();
            log.steps.push(states.len());
            log.events.push(format!("step {}", states.len()));
            log.batch_flags_are_shutdown
                .push(std::ptr::eq(cancel, crate::shutdown::cancellation_flag()));
        }
        if states.iter().any(|state| state.unready || state.poisoned) {
            return Err("sequence is provisional or poisoned by partial execution".into());
        }
        if self.refuse_batches > 0 {
            self.refuse_batches -= 1;
            return Err("batch scratch refused".into());
        }
        if self.fail_batches > 0 {
            self.fail_batches -= 1;
            for state in states.iter_mut() {
                state.poisoned = true;
            }
            return Err("command buffer fault".into());
        }
        Ok(states
            .iter_mut()
            .zip(tokens)
            .map(|(state, &token)| {
                state.tokens.push(token);
                next_logits(state)
            })
            .collect())
    }
    fn can_step(&self, state: &Seq) -> bool {
        !state.unready && !state.poisoned && state.tokens.len() < state.capacity
    }
    fn poisoned(&self, state: &Seq) -> bool {
        state.poisoned
    }
    fn position(&self, state: &Seq) -> usize {
        state.tokens.len()
    }
}

#[derive(Default)]
struct Recording {
    commits: Vec<Vec<u32>>,
}

impl reuse::Reuse<Seq> for Recording {
    fn take(
        &mut self,
        _: &[u32],
        _: usize,
        _: u64,
    ) -> std::result::Result<Option<reuse::Restored<Seq>>, ServeError> {
        Ok(None)
    }
    fn checkpoint(&mut self, _: &Seq, _: &[u32], _: &[f32], _: u64) {}
    fn commit(&mut self, live: reuse::Live<Seq>) {
        assert!(!live.state.poisoned);
        self.commits.push(live.tokens);
    }
    fn capacity(&self, _: &Seq, required: usize) -> usize {
        required
    }
}

type Terminal = std::result::Result<(GenerationEnd, usize, Value), (bool, u16, String)>;

#[derive(Default)]
struct SinkState {
    delivered: Vec<u8>,
    queued: Vec<u8>,
    /// Bytes the client may still accept; None = unlimited.
    credit: Option<usize>,
    terminal: Option<Terminal>,
}

struct TestSink {
    state: Rc<RefCell<SinkState>>,
    cancel: Arc<AtomicBool>,
}

impl BatchSink for TestSink {
    fn cancellation_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }
    fn transport_reserve_bytes(&self) -> u64 {
        1000
    }
    fn check(&self) -> io::Result<()> {
        if self.cancel.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "client gone",
            ));
        }
        Ok(())
    }
    fn queue(&mut self, bytes: &[u8]) {
        self.state.borrow_mut().queued.extend_from_slice(bytes);
    }
    fn flush(&mut self) -> io::Result<bool> {
        let mut state = self.state.borrow_mut();
        let take = state
            .credit
            .map_or(state.queued.len(), |credit| credit.min(state.queued.len()));
        let moved: Vec<u8> = state.queued.drain(..take).collect();
        if let Some(credit) = &mut state.credit {
            *credit -= moved.len();
        }
        state.delivered.extend(moved);
        Ok(state.queued.is_empty())
    }
    fn complete(self: Box<Self>, result: std::result::Result<GenerationOutcome, BackendFailure>) {
        let mut state = self.state.borrow_mut();
        assert!(state.terminal.is_none(), "one completion per request");
        state.terminal = Some(match result {
            Ok(outcome) => Ok((
                outcome.end,
                outcome.usage.output_tokens,
                outcome
                    .stats
                    .and_then(|stats| stats.phases)
                    .unwrap_or(Value::Null),
            )),
            Err(BackendFailure::Aborted(error)) => Err((true, 0, error.to_string())),
            Err(BackendFailure::Serve(error)) => Err((false, error.status, error.message)),
        });
    }
}

fn settings() -> Settings {
    let mut settings = Settings::new(
        tokenizer_fixture::tokenizer(Some("answer")),
        16384,
        64,
        qwen_llm::sampling::SamplingConfig::default(),
        tokenizer_fixture::EOS,
        "test".into(),
    )
    .unwrap();
    settings.headroom = || finite_headroom(1 << 30);
    settings
}

fn prepared(settings: &Settings, input: &str, stats: bool) -> Arc<PreparedResponse> {
    let profile = settings.profile();
    let request = profile
        .parse(&json!({"model":"test","input":input,"x_qwen":{"stats":stats}}))
        .unwrap();
    let validated = profile.validate(request).unwrap();
    let (prompt, boundaries) = profile.render_prepared(validated.request()).unwrap();
    Arc::new(PreparedResponse {
        validated,
        prompt,
        boundaries,
    })
}

struct Client {
    state: Rc<RefCell<SinkState>>,
    cancel: Arc<AtomicBool>,
}

impl Client {
    fn terminal(&self) -> Option<Terminal> {
        self.state.borrow().terminal.clone()
    }
    fn text(&self) -> String {
        String::from_utf8(self.state.borrow().delivered.clone()).unwrap()
    }
}

fn submit(scheduler: &mut Scheduler<Seq>, prepared: Arc<PreparedResponse>) -> Client {
    let state = Rc::new(RefCell::new(SinkState::default()));
    let cancel = Arc::new(AtomicBool::new(false));
    scheduler.submit(Job {
        prepared,
        sink: Box::new(TestSink {
            state: state.clone(),
            cancel: cancel.clone(),
        }),
    });
    Client { state, cancel }
}

fn drain(
    scheduler: &mut Scheduler<Seq>,
    settings: &Settings,
    engine: &mut StubEngine,
    reuse: &mut Recording,
) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if scheduler.in_flight() == (0, 0) {
            return;
        }
        scheduler.pump(settings, engine, reuse);
    }
    panic!("scheduler did not drain");
}

/// The serial path's output for the same request and engine.
fn serial(settings: &Settings, prepared: &PreparedResponse) -> (String, GenerationEnd, usize) {
    struct Collect(Vec<u8>);
    impl GenerationSink for Collect {
        fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
            self.0.extend_from_slice(bytes);
            Ok(())
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let super::super::output_partition::OutputProtocol::DeepSeekV41 { max_bytes, .. } =
        prepared.validated.output()
    else {
        unreachable!()
    };
    let engine = RefCell::new(StubEngine::default());
    let mut sink = Collect(Vec::new());
    let outcome = settings
        .execute_with_ceiling(
            *max_bytes,
            prepared.request(),
            &prepared.prompt,
            prepared.boundaries,
            &mut sink,
            |_, _, _| Ok(()),
            |required| execution::Engine::new_state(&mut *engine.borrow_mut(), required),
            &mut Recording::default(),
            |state, tokens, cancel, with_logits| {
                execution::Engine::prefill(
                    &mut *engine.borrow_mut(),
                    state,
                    tokens,
                    cancel,
                    with_logits,
                )
            },
            |state, token, cancel| {
                execution::Engine::step(&mut *engine.borrow_mut(), state, token, cancel)
            },
            || Duration::ZERO,
        )
        .unwrap();
    (
        String::from_utf8(sink.0).unwrap(),
        outcome.end,
        outcome.usage.output_tokens,
    )
}

#[test]
fn batched_requests_match_serial_outputs_and_report_batch_fields() {
    let settings = settings();
    let inputs = ["first prompt", "a second, longer prompt", "third"];
    let mut scheduler = Scheduler::new(3);
    let mut engine = StubEngine::default();
    let mut reuse = Recording::default();
    let clients: Vec<_> = inputs
        .iter()
        .map(|input| submit(&mut scheduler, prepared(&settings, input, true)))
        .collect();
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    let log = engine.log.borrow();
    assert!(log.steps.contains(&3), "{:?}", log.steps);
    assert!(
        log.batch_flags_are_shutdown
            .iter()
            .all(|&shutdown| shutdown)
    );
    assert_eq!(reuse.commits.len(), 3);
    for (client, input) in clients.iter().zip(inputs) {
        let (text, end, tokens) = serial(&settings, &prepared(&settings, input, true));
        let Some(Ok((batched_end, batched_tokens, phases))) = client.terminal() else {
            panic!("{input}: {:?}", client.terminal());
        };
        assert_eq!(client.text(), text, "{input}");
        assert_eq!((batched_end, batched_tokens), (end, tokens), "{input}");
        let batch = &phases["batch"];
        assert_eq!(batch["max_active"], 3);
        assert_eq!(batch["width_max"], 3);
        assert!(batch["batched_transitions"].as_u64().unwrap() > 0);
        assert_eq!(
            batch["stepped_transitions"], phases["transitions"],
            "{phases}"
        );
        assert!(batch["width_mean"].as_f64().unwrap() > 1.0);
        assert_eq!(batch["prefill_loaded_rows"], LOADED_PREFILL_ROWS);
        assert_eq!(phases["family"], "deepseek_v41");
        assert!(phases["process_token_io"].is_object());
    }
}

#[test]
fn prefill_is_bounded_while_others_decode_and_full_when_idle() {
    let settings = settings();
    let long = "x".repeat(3000);
    let mut scheduler = Scheduler::new(2);
    let mut engine = StubEngine::default();
    let mut reuse = Recording::default();
    let mut unbounded = prepared(&settings, &long, false);
    Arc::get_mut(&mut unbounded).unwrap().boundaries = None;
    // Alone: today's single call over the whole uncut prompt.
    let alone = submit(&mut scheduler, unbounded.clone());
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    assert!(matches!(alone.terminal(), Some(Ok(_))));
    let prompt = engine.log.borrow().prefill[0].0;
    assert!(prompt > 2 * LOADED_PREFILL_ROWS);
    assert_eq!(engine.log.borrow().prefill.len(), 1);
    engine.log.borrow_mut().events.clear();
    engine.log.borrow_mut().prefill.clear();

    // Under load: a decoding request, then the long prompt.
    let mut short = prepared(&settings, "hi", false);
    Arc::get_mut(&mut short).unwrap().boundaries = None;
    let short = submit(&mut scheduler, short);
    scheduler.pump(&settings, &mut engine, &mut reuse);
    let loaded = submit(&mut scheduler, unbounded);
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    let log = engine.log.borrow();
    let rows: Vec<usize> = log.prefill[1..].iter().map(|(rows, _)| *rows).collect();
    assert_eq!(rows.iter().sum::<usize>(), prompt);
    // Capped calls while the short request decodes; once it is done the
    // remainder goes in one full call.
    let capped = rows
        .iter()
        .take_while(|&&rows| rows == LOADED_PREFILL_ROWS)
        .count();
    assert!(capped >= 1 && rows.len() - capped <= 1, "{rows:?}");
    // The short request's decode steps ran between the long prefill calls.
    let prefills: Vec<usize> = (0..log.events.len())
        .filter(|&i| log.events[i].starts_with("prefill"))
        .collect();
    let (first, last) = (prefills[1], *prefills.last().unwrap());
    assert!(
        log.events[first..last]
            .iter()
            .any(|e| e.starts_with("step")),
        "{:?}",
        log.events
    );
    let Some(Ok((_, _, _))) = short.terminal() else {
        panic!("{:?}", short.terminal())
    };
    assert!(matches!(loaded.terminal(), Some(Ok(_))));
    // Equal text despite the different prefill partition (exact stub).
    assert_eq!(loaded.text(), alone.text());
}

#[test]
fn prefill_cap_is_reported_per_request() {
    let settings = settings();
    let long = "y".repeat(1500);
    let mut scheduler = Scheduler::new(2);
    let mut engine = StubEngine::default();
    let mut reuse = Recording::default();
    let short = submit(&mut scheduler, prepared(&settings, "hi", true));
    scheduler.pump(&settings, &mut engine, &mut reuse);
    let mut unbounded = prepared(&settings, &long, true);
    Arc::get_mut(&mut unbounded).unwrap().boundaries = None;
    let loaded = submit(&mut scheduler, unbounded);
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    let Some(Ok((_, _, phases))) = loaded.terminal() else {
        panic!()
    };
    assert!(phases["batch"]["prefill_loaded_calls"].as_u64().unwrap() >= 1);
    let Some(Ok((_, _, phases))) = short.terminal() else {
        panic!()
    };
    assert_eq!(phases["batch"]["prefill_loaded_calls"], 0);
}

#[test]
fn a_cancelled_request_drops_alone_and_batches_never_see_its_flag() {
    let settings = settings();
    let mut scheduler = Scheduler::new(3);
    let mut engine = StubEngine::default();
    let mut reuse = Recording::default();
    let clients: Vec<_> = ["one", "two two", "three three three"]
        .iter()
        .map(|input| submit(&mut scheduler, prepared(&settings, input, false)))
        .collect();
    // Admit and prefill all three, then take one batched step.
    while engine.log.borrow().steps.iter().all(|&w| w < 3) {
        scheduler.pump(&settings, &mut engine, &mut reuse);
    }
    clients[1].cancel.store(true, Ordering::Release);
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    assert!(matches!(clients[1].terminal(), Some(Err((true, _, _)))));
    for index in [0, 2] {
        assert!(matches!(clients[index].terminal(), Some(Ok(_))), "{index}");
    }
    let log = engine.log.borrow();
    assert!(
        log.batch_flags_are_shutdown
            .iter()
            .all(|&shutdown| shutdown)
    );
    // After the cancellation the remaining two keep batching.
    let cancelled_at = log.steps.iter().position(|&w| w == 3).unwrap();
    assert!(
        log.steps[cancelled_at + 1..].contains(&2),
        "{:?}",
        log.steps
    );
    assert_eq!(reuse.commits.len(), 2);
}

#[test]
fn a_refused_batch_falls_back_to_single_steps_and_only_an_offender_fails() {
    let settings = settings();
    // Refusal with healthy members: every request continues alone.
    let mut scheduler = Scheduler::new(2);
    let mut engine = StubEngine {
        refuse_batches: 1,
        ..StubEngine::default()
    };
    let mut reuse = Recording::default();
    let clients: Vec<_> = ["alpha", "beta beta"]
        .iter()
        .map(|input| submit(&mut scheduler, prepared(&settings, input, false)))
        .collect();
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    for (client, input) in clients.iter().zip(["alpha", "beta beta"]) {
        assert!(matches!(client.terminal(), Some(Ok(_))));
        assert_eq!(
            client.text(),
            serial(&settings, &prepared(&settings, input, false)).0
        );
    }
    let steps = engine.log.borrow().steps.clone();
    let refused = steps.iter().position(|&w| w == 2).unwrap();
    assert_eq!(steps[refused + 1..refused + 3], [1, 1]);

    // An unready member is stepped alone (and fails); the others batch.
    let mut scheduler = Scheduler::new(3);
    let mut engine = StubEngine::default();
    let clients: Vec<_> = ["one", "two two", "three three three"]
        .iter()
        .map(|input| submit(&mut scheduler, prepared(&settings, input, false)))
        .collect();
    while engine.log.borrow().steps.iter().all(|&w| w < 3) {
        scheduler.pump(&settings, &mut engine, &mut reuse);
    }
    // Poison request 0's state the way a provisional state would refuse.
    scheduler.with_state(0, |seq| seq.unready = true);
    let before = engine.log.borrow().steps.len();
    scheduler.pump(&settings, &mut engine, &mut reuse);
    assert_eq!(engine.log.borrow().steps[before..], [1, 2]);
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    let Some(Err((false, 500, message))) = clients[0].terminal() else {
        panic!("{:?}", clients[0].terminal())
    };
    assert!(message.contains("provisional or poisoned"), "{message}");
    assert!(matches!(clients[1].terminal(), Some(Ok(_))));
    assert!(matches!(clients[2].terminal(), Some(Ok(_))));
}

#[test]
fn a_batch_that_fails_after_starting_fails_every_member_without_commit() {
    let settings = settings();
    let mut scheduler = Scheduler::new(2);
    let mut engine = StubEngine {
        fail_batches: 1,
        ..StubEngine::default()
    };
    let mut reuse = Recording::default();
    let clients: Vec<_> = ["alpha", "beta beta"]
        .iter()
        .map(|input| submit(&mut scheduler, prepared(&settings, input, false)))
        .collect();
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    for client in &clients {
        let Some(Err((false, 500, message))) = client.terminal() else {
            panic!("{:?}", client.terminal())
        };
        assert!(
            message.contains("batched step of 2 requests failed"),
            "{message}"
        );
        assert!(message.contains("command buffer fault"), "{message}");
    }
    assert!(reuse.commits.is_empty());
    assert_eq!(scheduler.take_server_failures(), 2);
}

#[test]
fn a_parked_request_leaves_the_batch_resumes_and_times_out() {
    let settings = settings();
    let mut scheduler = Scheduler::new(2).with_park_timeout(Duration::from_millis(50));
    let mut engine = StubEngine::default();
    let mut reuse = Recording::default();
    let slow = submit(&mut scheduler, prepared(&settings, "slow client", false));
    let fast = submit(&mut scheduler, prepared(&settings, "fast", false));
    slow.state.borrow_mut().credit = Some(1);
    for _ in 0..6 {
        scheduler.pump(&settings, &mut engine, &mut reuse);
    }
    // One byte delivered; the request is parked and the other steps alone.
    assert_eq!(slow.state.borrow().delivered.len(), 1);
    let steps = engine.log.borrow().steps.clone();
    assert!(!steps.contains(&2), "{steps:?}");
    // The client catches up: the request rejoins the batch.
    slow.state.borrow_mut().credit = None;
    let fast_done = fast.terminal().is_some();
    drain_until(&mut scheduler, &settings, &mut engine, &mut reuse, || {
        slow.terminal().is_some()
    });
    assert!(matches!(slow.terminal(), Some(Ok(_))));
    assert_eq!(
        slow.text(),
        serial(&settings, &prepared(&settings, "slow client", false)).0
    );
    if !fast_done {
        assert!(engine.log.borrow().steps.contains(&2));
    }
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    assert!(matches!(fast.terminal(), Some(Ok(_))));

    // A client that never drains is aborted after the park timeout.
    let stuck = submit(&mut scheduler, prepared(&settings, "stuck", false));
    stuck.state.borrow_mut().credit = Some(0);
    let started = Instant::now();
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    assert!(started.elapsed() >= Duration::from_millis(50));
    let Some(Err((true, _, message))) = stuck.terminal() else {
        panic!("{:?}", stuck.terminal())
    };
    assert!(message.contains("did not drain"), "{message}");
}

fn drain_until(
    scheduler: &mut Scheduler<Seq>,
    settings: &Settings,
    engine: &mut StubEngine,
    reuse: &mut Recording,
    done: impl Fn() -> bool,
) {
    for _ in 0..10_000 {
        if done() {
            return;
        }
        scheduler.pump(settings, engine, reuse);
    }
    panic!("condition not reached");
}

#[test]
fn abort_all_aborts_started_requests_and_rejects_waiting_ones_busy() {
    let settings = settings();
    let mut scheduler = Scheduler::new(1);
    let mut engine = StubEngine::default();
    let mut reuse = Recording::default();
    let started = submit(&mut scheduler, prepared(&settings, "started", false));
    let waiting = submit(&mut scheduler, prepared(&settings, "waiting", false));
    scheduler.pump(&settings, &mut engine, &mut reuse);
    assert_eq!(scheduler.in_flight(), (1, 1));
    scheduler.abort_all();
    assert_eq!(scheduler.in_flight(), (0, 0));
    assert!(matches!(started.terminal(), Some(Err((true, _, _)))));
    let Some(Err((false, 503, _))) = waiting.terminal() else {
        panic!("{:?}", waiting.terminal())
    };
    assert!(reuse.commits.is_empty());
}

#[test]
fn waiting_requests_start_in_order_and_admission_adds_in_flight_reserves() {
    let settings = settings();
    let mut scheduler = Scheduler::new(2);
    let mut engine = StubEngine::default();
    let mut reuse = Recording::default();
    let clients: Vec<_> = ["one", "two", "three"]
        .iter()
        .map(|input| submit(&mut scheduler, prepared(&settings, input, false)))
        .collect();
    scheduler.pump(&settings, &mut engine, &mut reuse);
    scheduler.pump(&settings, &mut engine, &mut reuse);
    assert_eq!(scheduler.in_flight(), (1, 2));
    let reserves = engine.log.borrow().reserves.clone();
    assert_eq!(reserves.len(), 2);
    assert_eq!(
        reserves[1],
        2 * reserves[0],
        "second admission adds the first's reserve"
    );
    drain_until(&mut scheduler, &settings, &mut engine, &mut reuse, || {
        clients[0].terminal().is_some() || clients[1].terminal().is_some()
    });
    assert!(clients[2].terminal().is_none());
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    for client in &clients {
        assert!(matches!(client.terminal(), Some(Ok(_))));
    }
    // Commits arrive in finishing order; the third started last.
    assert_eq!(engine.log.borrow().reserves.len(), 3);
}

/// The scheduler behind the real owner loop and HTTP workers.
struct HttpStub {
    settings: Settings,
    scheduler: Scheduler<Seq>,
    engine: StubEngine,
    reuse: Recording,
    /// Hold every request until this many were submitted.
    gather: usize,
}

impl GenerationBackend for HttpStub {
    fn model_id(&self) -> &str {
        "test"
    }
    fn request_profile(&self) -> RequestProfile {
        self.settings.profile()
    }
    fn output_headroom(&self) -> super::super::http::Headroom {
        || Some(1 << 30)
    }
    fn generate(
        &mut self,
        _: &ServeRequest,
        _: &str,
        _: &mut dyn GenerationSink,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        unreachable!("batched")
    }
    fn max_active(&self) -> usize {
        self.scheduler.max_active()
    }
    fn submit(&mut self, job: Job) {
        self.scheduler.submit(job);
    }
    fn pump(&mut self) -> bool {
        let (pending, active) = self.scheduler.in_flight();
        if self.gather > 0 {
            if pending + active < self.gather {
                return false;
            }
            self.gather = 0;
        }
        self.scheduler
            .pump(&self.settings, &mut self.engine, &mut self.reuse)
    }
    fn abort_all(&mut self) {
        self.scheduler.abort_all();
    }
}

#[test]
fn http_requests_batch_through_the_owner_loop_with_serial_text() {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let inputs = ["first prompt", "a second, longer prompt", "third"];
    let stop = Arc::new(AtomicBool::new(false));
    let stopping = stop.clone();
    let (widths, observed) = std::sync::mpsc::channel();
    let server = std::thread::spawn(move || {
        let mut backend = HttpStub {
            settings: settings(),
            scheduler: Scheduler::new(3),
            engine: StubEngine::default(),
            reuse: Recording::default(),
            gather: 3,
        };
        let result = crate::serve::accept_loop_with_queue_depth(listener, &mut backend, 4, |_| {
            anyhow::ensure!(!stopping.load(Ordering::Acquire), "test stop");
            Ok(())
        });
        widths
            .send((
                backend.engine.log.borrow().steps.clone(),
                backend.reuse.commits.len(),
            ))
            .unwrap();
        result
    });
    let clients: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let body = json!({"model":"test","input":input,"stream":index == 1}).to_string();
            let mut socket = std::net::TcpStream::connect(address).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            socket
                .write_all(
                    format!(
                        "POST /v1/responses HTTP/1.1\r\nhost: localhost\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .unwrap();
            socket
        })
        .collect();
    let settings = settings();
    for (mut socket, input) in clients.into_iter().zip(inputs) {
        let mut wire = String::new();
        socket.read_to_string(&mut wire).unwrap();
        assert!(wire.starts_with("HTTP/1.1 200"), "{wire}");
        let (text, _, _) = serial(&settings, &prepared(&settings, input, false));
        let visible: String = text.chars().filter(|c| c.is_ascii_lowercase()).collect();
        let deltas: String = wire
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str::<Value>(data).ok())
            .filter(|event| event["type"] == "response.output_text.delta")
            .filter_map(|event| event["delta"].as_str().map(str::to_owned))
            .collect();
        if deltas.is_empty() {
            assert!(wire.contains(&visible), "{input}: {visible} in {wire}");
        } else {
            assert_eq!(deltas, visible, "{input}");
        }
    }
    stop.store(true, Ordering::Release);
    assert!(server.join().unwrap().is_err());
    let (steps, commits) = observed.recv().unwrap();
    assert!(steps.contains(&3), "{steps:?}");
    assert_eq!(commits, 3);
}
