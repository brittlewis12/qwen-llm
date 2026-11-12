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

fn next_logits(seq: &Seq, shape: Logits) -> Vec<f32> {
    let generated = seq.tokens.len() - seq.prompt;
    let budget = 3 + seq.prompt % 5;
    if shape == Logits::Spread && generated < budget + 4 {
        // Comparable weights over 'a'..='z' from a hash of the history, so a
        // sampled request's stream depends on every draw.
        let mut hash = seq
            .tokens
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |hash, &token| {
                (hash ^ u64::from(token)).wrapping_mul(0x0100_0000_01b3)
            });
        let mut logits = vec![-30.0; VOCAB];
        for token in b'a'..=b'z' {
            hash = (hash ^ u64::from(token)).wrapping_mul(0x0100_0000_01b3);
            logits[usize::from(token)] = (hash >> 40) as f32 / (1u64 << 24) as f32 * 3.0;
        }
        return logits;
    }
    let token = if generated >= budget || shape == Logits::Spread {
        tokenizer_fixture::EOS
    } else {
        u32::from(b'a') + seq.tokens.iter().sum::<u32>() % 26
    };
    let mut logits = vec![0.0; VOCAB];
    logits[token as usize] = 100.0;
    logits
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Logits {
    /// One dominant token: greedy and sampled decoding agree.
    #[default]
    Peaked,
    /// Several plausible tokens for a few more steps, then EOS.
    Spread,
}

#[derive(Default)]
struct Log {
    /// (rows, with_logits) per prefill call, tagged by prompt length.
    prefill: Vec<(usize, usize)>,
    /// Width of each step call (1 = plain step).
    steps: Vec<usize>,
    /// Whether every batched step received only the shutdown flag.
    batch_flags_are_shutdown: Vec<bool>,
    /// Per sampled step, each member's explicit-seed intent, in batch order.
    sampled_explicit: Vec<Vec<bool>>,
    reserves: Vec<u64>,
    events: Vec<String>,
}

#[derive(Default)]
struct StubEngine {
    log: Rc<RefCell<Log>>,
    refuse_batches: usize,
    fail_batches: usize,
    /// Synthetic clock: each prefilled row and each step costs 1 ms.
    clock: Duration,
    logits: Logits,
    /// Prompt length whose decode logits carry a NaN after its first step.
    nan_prompt: Option<usize>,
    cold: bool,
    cold_encoded: usize,
    /// Synthetic service time for a decode quantum (default 1 ms).
    step_ms: u64,
    /// Serve batched steps through `step_batch_sampled` (device-style tokens).
    sampled_step: bool,
    layered: bool,
    cancel_layer: Option<usize>,
}

impl StubEngine {
    fn logits(&self, seq: &Seq) -> Vec<f32> {
        let mut logits = next_logits(seq, self.logits);
        if self.nan_prompt == Some(seq.prompt) && seq.tokens.len() > seq.prompt + 1 {
            logits[usize::from(b'q')] = f32::NAN;
        }
        logits
    }
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
        self.clock += Duration::from_millis(tokens.len() as u64);
        log.events.push(format!("prefill {}", tokens.len()));
        state.tokens.extend_from_slice(tokens);
        if !with_logits {
            return Ok(Vec::new());
        }
        state.prompt = state.tokens.len();
        Ok(self.logits(state))
    }
    fn prefill_layered(
        &mut self,
        state: &mut Seq,
        tokens: &[u32],
        cancel: &AtomicBool,
        with_logits: bool,
        target_ms: usize,
        _: usize,
        between: &mut dyn FnMut(&mut Self),
    ) -> Result<Vec<f32>, ServeError> {
        if !self.layered || target_ms == 0 || tokens.len() < 300 {
            return self.prefill(state, tokens, cancel, with_logits);
        }
        state.unready = true;
        for layer in 0..3 {
            self.clock += Duration::from_millis(100);
            self.log.borrow_mut().events.push(format!("layer {layer}"));
            if self.cancel_layer == Some(layer) {
                cancel.store(true, Ordering::Release);
            }
            if layer < 2 {
                between(self);
            }
            if cancel.load(Ordering::Acquire) {
                state.poisoned = true;
                return Err(ServeError::server_error("cancelled partial layer append"));
            }
        }
        state.tokens.extend_from_slice(tokens);
        state.unready = false;
        if with_logits {
            state.prompt = state.tokens.len();
            Ok(self.logits(state))
        } else {
            Ok(Vec::new())
        }
    }
    fn cold_prefill_layered(
        &mut self,
        state: &mut Seq,
        tokens: &[u32],
        cancel: &AtomicBool,
        begin: bool,
        max_rows: usize,
        target_ms: usize,
        _: usize,
        between: &mut dyn FnMut(&mut Self),
    ) -> Result<Option<qwen_llm::deepseek_v41::forward::ColdPrefillSlice>, ServeError> {
        if !self.cold || !self.layered || target_ms == 0 {
            return self.cold_prefill(state, tokens, cancel, begin, max_rows);
        }
        let previous = state.unready;
        state.unready = true;
        for layer in 0..3 {
            self.clock += Duration::from_millis(100);
            if self.cancel_layer == Some(layer) {
                cancel.store(true, Ordering::Release);
            }
            if layer < 2 {
                between(self);
            }
            if cancel.load(Ordering::Acquire) {
                state.poisoned = true;
                return Err(ServeError::server_error("cancelled CED layer"));
            }
        }
        state.unready = previous;
        self.cold_prefill(state, tokens, cancel, begin, max_rows)
    }
    fn cold_prefill(
        &mut self,
        state: &mut Seq,
        tokens: &[u32],
        _: &AtomicBool,
        begin: bool,
        max_rows: usize,
    ) -> Result<Option<qwen_llm::deepseek_v41::forward::ColdPrefillSlice>, ServeError> {
        if !self.cold {
            return Ok(None);
        }
        if begin {
            assert!(!state.unready);
            state.unready = true;
            self.cold_encoded = 0;
        } else {
            assert!(state.unready);
        }
        let rows = max_rows.min(tokens.len() - self.cold_encoded);
        self.cold_encoded += rows;
        self.clock += Duration::from_millis(rows as u64);
        self.log.borrow_mut().events.push(format!("ced {rows}"));
        let complete = self.cold_encoded == tokens.len();
        let logits = if complete {
            state.tokens.extend_from_slice(tokens);
            state.prompt = state.tokens.len();
            state.unready = false;
            self.logits(state)
        } else {
            vec![]
        };
        Ok(Some(qwen_llm::deepseek_v41::forward::ColdPrefillSlice {
            encoder_rows: rows,
            complete,
            logits,
        }))
    }
    fn step(
        &mut self,
        state: &mut Seq,
        token: u32,
        _: &AtomicBool,
    ) -> std::result::Result<Vec<f32>, String> {
        self.log.borrow_mut().steps.push(1);
        self.log.borrow_mut().events.push("step 1".into());
        self.clock += Duration::from_millis(self.step_ms.max(1));
        if state.unready || state.poisoned {
            return Err("sequence is provisional or poisoned by partial execution".into());
        }
        state.tokens.push(token);
        Ok(self.logits(state))
    }
    fn now(&mut self) -> Duration {
        self.clock
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
        self.clock += Duration::from_millis(self.step_ms.max(1));
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
                self.logits(state)
            })
            .collect())
    }
    fn can_step(&self, state: &Seq) -> bool {
        !state.unready && !state.poisoned && state.tokens.len() < state.capacity
    }
    fn supports_sampled_step(&self) -> bool {
        self.sampled_step
    }
    fn step_batch_sampled(
        &mut self,
        states: &mut [&mut Seq],
        tokens: &[u32],
        samplers: &mut [&mut qwen_llm::sampling::Sampler],
        explicit_seeds: &[bool],
        cancel: &AtomicBool,
    ) -> std::result::Result<Vec<u32>, String> {
        {
            let mut log = self.log.borrow_mut();
            log.steps.push(states.len());
            log.events.push(format!("sampled {}", states.len()));
            log.sampled_explicit.push(explicit_seeds.to_vec());
            log.batch_flags_are_shutdown
                .push(std::ptr::eq(cancel, crate::shutdown::cancellation_flag()));
        }
        self.clock += Duration::from_millis(self.step_ms.max(1));
        if states.iter().any(|state| state.unready || state.poisoned) {
            return Err("sequence is provisional or poisoned by partial execution".into());
        }
        // Mirror `ForwardEngine`: explicit members stay exact host sampling,
        // implicit members would be device-sampled (argmax stands in here).
        states
            .iter_mut()
            .zip(tokens)
            .enumerate()
            .map(|(i, (state, &token))| {
                state.tokens.push(token);
                let logits = self.logits(state);
                if explicit_seeds[i] {
                    samplers[i]
                        .sample(&logits)
                        .map(|sampled| sampled.token as u32)
                        .map_err(|error| error.to_string())
                } else {
                    Ok(logits
                        .iter()
                        .enumerate()
                        .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
                            if v > best.1 { (i, v) } else { best }
                        })
                        .0 as u32)
                }
            })
            .collect()
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
    ready_short: bool,
}

impl reuse::Reuse<Seq> for Recording {
    fn take(
        &mut self,
        ids: &[u32],
        required: usize,
        _: u64,
    ) -> std::result::Result<Option<reuse::Restored<Seq>>, ServeError> {
        if self.ready_short && ids.len() < 512 {
            let state = Seq {
                tokens: ids.to_vec(),
                prompt: ids.len(),
                poisoned: false,
                unready: false,
                capacity: required,
            };
            let logits = next_logits(&state, Logits::Peaked);
            return Ok(Some(reuse::Restored {
                live: reuse::Live {
                    state,
                    tokens: ids.to_vec(),
                    logits,
                },
                source: "snapshot_ram",
            }));
        }
        Ok(None)
    }
    fn checkpoint(&mut self, state: &Seq, _: &[u32], _: &[f32], _: u64) {
        assert!(!state.unready);
    }
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
    // Keep the original small-quantum lifecycle witnesses as rollback checks.
    settings.prefill_fair = false;
    settings
}

fn prepared(settings: &Settings, input: &str, stats: bool) -> Arc<PreparedResponse> {
    prepared_json(
        settings,
        json!({"model":"test","input":input,"x_qwen":{"stats":stats}}),
    )
}

/// A request with a fixed sampler seed (absent seeds are fresh draws).
fn seeded(settings: &Settings, input: &str, seed: u64) -> Arc<PreparedResponse> {
    prepared_json(
        settings,
        json!({"model":"test","input":input,"x_qwen":{"seed":seed}}),
    )
}

fn prepared_json(settings: &Settings, body: Value) -> Arc<PreparedResponse> {
    let profile = settings.profile();
    let request = profile.parse(&body).unwrap();
    let validated = profile.validate(request).unwrap();
    let (prompt, boundaries) = profile.render_prepared(validated.request()).unwrap();
    Arc::new(PreparedResponse {
        validated,
        prompt,
        boundaries,
        tokens: None,
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
    serial_with(settings, prepared, StubEngine::default())
}

fn serial_with(
    settings: &Settings,
    prepared: &PreparedResponse,
    engine: StubEngine,
) -> (String, GenerationEnd, usize) {
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
    let engine = RefCell::new(engine);
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
fn sampled_batch_steps_match_serial_greedy_outputs() {
    let mut settings = settings();
    settings.gpu_sample = true;
    let inputs = ["first prompt", "a second, longer prompt", "third"];
    let mut scheduler = Scheduler::new(3);
    let mut engine = StubEngine {
        sampled_step: true,
        ..Default::default()
    };
    let mut reuse = Recording::default();
    let clients: Vec<_> = inputs
        .iter()
        .map(|input| submit(&mut scheduler, prepared(&settings, input, true)))
        .collect();
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    {
        let log = engine.log.borrow();
        assert!(
            log.events.iter().any(|event| event == "sampled 3"),
            "{:?}",
            log.events
        );
        assert!(
            log.batch_flags_are_shutdown
                .iter()
                .all(|&shutdown| shutdown),
            "{:?}",
            log.batch_flags_are_shutdown
        );
    }
    assert_eq!(reuse.commits.len(), 3);
    for (client, input) in clients.iter().zip(inputs) {
        let (text, end, tokens) = serial(&settings, &prepared(&settings, input, true));
        let Some(Ok((batched_end, batched_tokens, phases))) = client.terminal() else {
            panic!("{input}: {:?}", client.terminal());
        };
        assert_eq!(client.text(), text, "{input}");
        assert_eq!((batched_end, batched_tokens), (end, tokens), "{input}");
        assert!(phases["batch"]["batched_transitions"].as_u64().unwrap() > 0);
    }
}

#[test]
fn sampled_steps_carry_seed_intent_after_normalization() {
    let mut settings = settings();
    settings.gpu_sample = true;
    let mut scheduler = Scheduler::new(2);
    let mut engine = StubEngine {
        sampled_step: true,
        ..Default::default()
    };
    let mut reuse = Recording::default();
    // `prepared` and `seeded` both pass through parse -> normalize, which fills
    // an absent seed with a fresh draw. Only the captured intent differs.
    let _implicit = submit(&mut scheduler, prepared(&settings, "first prompt", false));
    let _explicit = submit(&mut scheduler, seeded(&settings, "second prompt", 7));
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    let log = engine.log.borrow();
    let seen = log.sampled_explicit.last().expect("a sampled batch");
    assert_eq!(seen.len(), 2, "{seen:?}");
    assert_eq!(
        seen.iter().filter(|&&explicit| !explicit).count(),
        1,
        "the unseeded request must reach the engine as implicit: {seen:?}"
    );
    assert_eq!(
        seen.iter().filter(|&&explicit| explicit).count(),
        1,
        "the seeded request must reach the engine as explicit: {seen:?}"
    );
}

#[test]
fn seeded_requests_keep_host_tokens_with_gpu_sampling_on_or_off() {
    // The fixture tokenizer maps one token per byte, so delivered text plus the
    // token count is the generated token stream.
    fn two_seeded(gpu_sample: bool) -> Vec<(String, usize)> {
        let mut settings = settings();
        settings.sampling = qwen_llm::sampling::SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 42,
        };
        settings.gpu_sample = gpu_sample;
        let mut scheduler = Scheduler::new(2);
        let mut engine = StubEngine {
            sampled_step: true,
            ..Default::default()
        };
        let mut reuse = Recording::default();
        let clients: Vec<_> = ["first prompt", "a second, longer prompt"]
            .iter()
            .map(|input| submit(&mut scheduler, seeded(&settings, input, 7)))
            .collect();
        drain(&mut scheduler, &settings, &mut engine, &mut reuse);
        clients
            .iter()
            .map(|client| {
                let text = client.text();
                match client.terminal() {
                    Some(Ok((_, count, _))) => (text, count),
                    other => panic!("{other:?}"),
                }
            })
            .collect()
    }
    assert_eq!(
        two_seeded(true),
        two_seeded(false),
        "explicitly seeded requests must take the unchanged host-sampling path"
    );
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
    let calls = phases["batch"]["prefill_loaded_calls"].as_u64().unwrap();
    assert!(calls >= 1);
    // Active prefill time excludes decode service between the slices.
    let rows = phases["prefill_tokens"].as_f64().unwrap();
    assert_eq!(phases["prefill_ms"].as_f64().unwrap(), rows, "{phases}");
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
    // The test reads the clients one after another, so a request whose
    // worker lags under machine load parks and leaves the batch: the widest
    // step is 2 or 3 depending on scheduling, never a fixed pattern.
    assert!(steps.iter().any(|&width| width >= 2), "{steps:?}");
    assert_eq!(commits, 3);
}

/// Text and terminal (without timing phases) of every client.
type Observed = Vec<(
    String,
    std::result::Result<(GenerationEnd, usize), (bool, u16, String)>,
)>;

/// Run `inputs` concurrently with sampling on `workers` pool threads (0 = on
/// the owner) against spread logits under `settings`' sampler.
fn sampled_batch(
    settings: &Settings,
    inputs: &[&str],
    workers: usize,
    nan_prompt: Option<usize>,
) -> (Observed, usize, Vec<Vec<u32>>, Vec<usize>) {
    let mut scheduler = Scheduler::new(inputs.len())
        .with_sample_workers(workers)
        .unwrap();
    assert_eq!(scheduler.sample_workers(), workers);
    let mut engine = StubEngine {
        logits: Logits::Spread,
        nan_prompt,
        ..StubEngine::default()
    };
    let mut reuse = Recording::default();
    let clients: Vec<_> = inputs
        .iter()
        .enumerate()
        .map(|(seed, input)| submit(&mut scheduler, seeded(settings, input, seed as u64)))
        .collect();
    drain(&mut scheduler, settings, &mut engine, &mut reuse);
    let observed = clients
        .iter()
        .map(|client| {
            let terminal = match client.terminal().expect("terminal") {
                Ok((end, tokens, _)) => Ok((end, tokens)),
                Err(failure) => Err(failure),
            };
            (client.text(), terminal)
        })
        .collect();
    let steps = engine.log.borrow().steps.clone();
    (observed, scheduler.pool_sampled(), reuse.commits, steps)
}

#[test]
fn pooled_sampling_matches_owner_sampling_and_the_serial_path() {
    let mut settings = settings();
    // The V4.1 release shape: full-vocabulary sampling at temperature 1.
    settings.sampling.temperature = 1.0;
    let inputs = [
        "first prompt",
        "a second, longer prompt",
        "third",
        "fourth one",
    ];
    let (owner, owner_pooled, owner_commits, owner_steps) =
        sampled_batch(&settings, &inputs, 0, None);
    assert_eq!(owner_pooled, 0);
    assert!(owner_steps.contains(&4), "{owner_steps:?}");
    for workers in [1, 3, 5] {
        let (pooled, count, commits, steps) = sampled_batch(&settings, &inputs, workers, None);
        assert!(count > 8, "workers={workers}: {count} pooled samples");
        assert_eq!(pooled, owner, "workers={workers}");
        assert_eq!(commits, owner_commits, "workers={workers}");
        assert_eq!(steps, owner_steps, "workers={workers}");
    }

    let greedy = self::settings();
    assert_eq!(greedy.sampling.temperature, 0.0);
    let mut differs_from_greedy = false;
    for (seed, ((text, terminal), input)) in owner.iter().zip(inputs).enumerate() {
        let seed = seed as u64;
        let spread = || StubEngine {
            logits: Logits::Spread,
            ..StubEngine::default()
        };
        let (serial_text, end, tokens) =
            serial_with(&settings, &seeded(&settings, input, seed), spread());
        assert_eq!(text, &serial_text, "{input}");
        assert_eq!(terminal, &Ok((end, tokens)), "{input}");
        let (greedy_text, ..) = serial_with(&greedy, &seeded(&greedy, input, seed), spread());
        differs_from_greedy |= greedy_text != serial_text;
    }
    assert!(
        differs_from_greedy,
        "spread logits must make sampling matter"
    );
}

/// A member whose logits turn malformed fails alone, on the owner, with the
/// same terminal whether or not its sampler already ran on the pool.
#[test]
fn pooled_sampling_preserves_owner_failure_handling() {
    let mut settings = settings();
    settings.sampling.temperature = 1.0;
    let inputs = ["first prompt", "a second, longer prompt", "third"];
    let nan_prompt = Some(seeded(&settings, inputs[1], 1).prompt.len());
    let (owner, ..) = sampled_batch(&settings, &inputs, 0, nan_prompt);
    let (pooled, count, ..) = sampled_batch(&settings, &inputs, 2, nan_prompt);
    assert!(count > 0);
    assert_eq!(pooled, owner);
    let failed: Vec<_> = owner
        .iter()
        .filter(|(_, terminal)| terminal.is_err())
        .collect();
    assert_eq!(failed.len(), 1, "{owner:?}");
    assert!(
        matches!(&failed[0].1, Err((false, 500, message)) if message.contains("malformed logits")),
        "{failed:?}"
    );
}

#[test]
fn cold_ced_yields_then_uses_loaded_quanta_while_another_request_decodes() {
    let settings = settings();
    let mut scheduler = Scheduler::new(2);
    let mut engine = StubEngine {
        cold: true,
        ..StubEngine::default()
    };
    let mut reuse = Recording {
        ready_short: true,
        ..Recording::default()
    };
    let mut request = prepared(&settings, &"x".repeat(12000), true);
    Arc::get_mut(&mut request).unwrap().boundaries = None;
    let cold = submit(&mut scheduler, request);
    for _ in 0..4 {
        scheduler.pump(&settings, &mut engine, &mut reuse);
        if engine.cold_encoded > 0 {
            break;
        }
    }
    assert_eq!(engine.cold_encoded, 4096);
    let short = submit(&mut scheduler, prepared(&settings, "hi", false));
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    assert!(matches!(cold.terminal(), Some(Ok(_))));
    assert!(matches!(short.terminal(), Some(Ok(_))));
    let log = engine.log.borrow();
    assert!(
        log.events.iter().any(|e| e == "ced 512"),
        "{:?}",
        log.events
    );
    let first = log
        .events
        .iter()
        .position(|e| e.starts_with("ced"))
        .unwrap();
    let last = log
        .events
        .iter()
        .rposition(|e| e.starts_with("ced"))
        .unwrap();
    assert!(
        log.events[first..last]
            .iter()
            .any(|e| e.starts_with("step"))
    );
    let Some(Ok((_, _, phases))) = cold.terminal() else {
        unreachable!()
    };
    assert_eq!(phases["ced_episodes"], 1);
    let ranges = phases["prefill_chunk_ranges"].as_array().unwrap();
    assert!(
        ranges
            .iter()
            .any(|r| r[1].as_u64().unwrap() - r[0].as_u64().unwrap() == 512)
    );
}

#[test]
fn fair_prefill_starts_ced_under_load_and_keeps_decoding_between_quanta() {
    let mut settings = settings();
    settings.prefill_fair = true;
    let mut scheduler = Scheduler::new(2);
    let mut engine = StubEngine {
        cold: true,
        ..StubEngine::default()
    };
    let mut reuse = Recording {
        ready_short: true,
        ..Recording::default()
    };
    let short = submit(&mut scheduler, prepared(&settings, "hi", true));
    scheduler.pump(&settings, &mut engine, &mut reuse);
    let mut request = prepared(&settings, &"x".repeat(12000), true);
    Arc::get_mut(&mut request).unwrap().boundaries = None;
    let cold = submit(&mut scheduler, request);
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    assert!(matches!(short.terminal(), Some(Ok(_))));
    let Some(Ok((_, _, phases))) = cold.terminal() else {
        panic!("{:?}", cold.terminal());
    };
    assert_eq!(phases["ced_episodes"], 1);
    assert_eq!(phases["batch"]["prefill_loaded_calls"], 0);
    assert_eq!(phases["batch"]["prefill_loaded_rows"], 4096);
    let log = engine.log.borrow();
    let first = log
        .events
        .iter()
        .position(|e| e.starts_with("ced"))
        .unwrap();
    let last = log
        .events
        .iter()
        .rposition(|e| e.starts_with("ced"))
        .unwrap();
    assert!(
        log.events[first..last]
            .iter()
            .any(|e| e.starts_with("step"))
    );
    assert!(!log.events.iter().any(|e| e == "ced 512"));
}

#[test]
fn fair_decode_service_tracks_elapsed_time_with_a_bounded_burst() {
    // Each 4096-row prefill costs 4096 ms, each decode step costs 200 ms.
    // 1% earns 41 ms (one step), 10% earns 455 ms (three steps); 90%
    // reaches the one-second cap (five steps), then prefill resumes.
    for (percent, expected) in [(1, 1), (10, 3), (90, 5)] {
        let mut settings = settings();
        settings.prefill_fair = true;
        settings.decode_percent = percent;
        let mut scheduler = Scheduler::new(2);
        let mut engine = StubEngine {
            cold: true,
            step_ms: 200,
            ..StubEngine::default()
        };
        let mut reuse = Recording {
            ready_short: true,
            ..Recording::default()
        };
        // Seven generated tokens, so even the longest burst leaves work.
        let short = (1..=5)
            .map(|n| prepared(&settings, &"hi".repeat(n), true))
            .find(|request| {
                settings
                    .tokenizer
                    .encode(&request.prompt, false)
                    .unwrap()
                    .len()
                    % 5
                    == 4
            })
            .unwrap();
        let short = submit(&mut scheduler, short);
        scheduler.pump(&settings, &mut engine, &mut reuse);
        let mut request = prepared(&settings, &"x".repeat(12000), true);
        Arc::get_mut(&mut request).unwrap().boundaries = None;
        let cold = submit(&mut scheduler, request);
        drain(&mut scheduler, &settings, &mut engine, &mut reuse);
        assert!(matches!(short.terminal(), Some(Ok(_))));
        assert!(matches!(cold.terminal(), Some(Ok(_))));
        let log = engine.log.borrow();
        let ced: Vec<_> = log
            .events
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.starts_with("ced").then_some(i))
            .collect();
        let steps = log.events[ced[0] + 1..ced[1]]
            .iter()
            .filter(|e| e.starts_with("step"))
            .count();
        assert_eq!(steps, expected, "decode {percent}%: {:?}", log.events);
    }
}

#[test]
fn fair_cancelled_ced_discards_provisional_state_and_preserves_other_decoders() {
    let mut settings = settings();
    settings.prefill_fair = true;
    let mut scheduler = Scheduler::new(2);
    let mut engine = StubEngine {
        cold: true,
        ..StubEngine::default()
    };
    let mut reuse = Recording {
        ready_short: true,
        ..Recording::default()
    };
    let short = submit(&mut scheduler, prepared(&settings, "hi", true));
    scheduler.pump(&settings, &mut engine, &mut reuse);
    let mut request = prepared(&settings, &"x".repeat(12000), true);
    Arc::get_mut(&mut request).unwrap().boundaries = None;
    let cold = submit(&mut scheduler, request);
    scheduler.pump(&settings, &mut engine, &mut reuse);
    assert_eq!(engine.cold_encoded, 4096);
    cold.cancel.store(true, Ordering::Release);
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    assert!(matches!(cold.terminal(), Some(Err((true, _, _)))));
    assert!(matches!(short.terminal(), Some(Ok(_))));
    assert_eq!(reuse.commits.len(), 1, "provisional CED was not committed");
}

#[test]
fn layer_slices_keep_appender_out_of_peer_rounds_and_exclude_peer_time() {
    let mut settings = settings();
    settings.prefill_fair = true;
    settings.cold_ced = false;
    settings.layer_slice_ms = 200;
    let mut scheduler = Scheduler::new(3);
    let mut engine = StubEngine {
        layered: true,
        step_ms: 20,
        ..Default::default()
    };
    let mut reuse = Recording {
        ready_short: true,
        ..Default::default()
    };
    let a = submit(&mut scheduler, prepared(&settings, "hi", true));
    scheduler.pump(&settings, &mut engine, &mut reuse);
    let b = submit(&mut scheduler, prepared(&settings, "hello", true));
    scheduler.pump(&settings, &mut engine, &mut reuse);
    let mut request = prepared(&settings, &"x".repeat(1500), true);
    Arc::get_mut(&mut request).unwrap().boundaries = None;
    let app = submit(&mut scheduler, request);
    scheduler.pump(&settings, &mut engine, &mut reuse);
    drain(&mut scheduler, &settings, &mut engine, &mut reuse);
    assert!(matches!(a.terminal(), Some(Ok(_))) && matches!(b.terminal(), Some(Ok(_))));
    let Some(Ok((_, _, phases))) = app.terminal() else {
        panic!("{:?}", app.terminal());
    };
    assert_eq!(
        phases["prefill_ms"], 300.0,
        "only native slice work is charged"
    );
    let log = engine.log.borrow();
    let first = log.events.iter().position(|e| e == "layer 0").unwrap();
    let last = log.events.iter().position(|e| e == "layer 2").unwrap();
    assert!(
        log.events[first..last]
            .iter()
            .any(|e| e.starts_with("step"))
    );
    assert!(
        !log.events[first..last].iter().any(|e| e == "step 3"),
        "provisional appender excluded"
    );
}

#[test]
fn cancellation_between_layers_discards_appender_and_keeps_peers_consistent() {
    for (cold, layer) in [false, true]
        .into_iter()
        .flat_map(|cold| (0..3).map(move |layer| (cold, layer)))
    {
        let mut settings = settings();
        settings.prefill_fair = true;
        settings.cold_ced = cold;
        settings.layer_slice_ms = 200;
        let mut scheduler = Scheduler::new(2);
        let mut engine = StubEngine {
            cold,
            layered: true,
            cancel_layer: Some(layer),
            ..Default::default()
        };
        let mut reuse = Recording {
            ready_short: true,
            ..Default::default()
        };
        let peer = submit(&mut scheduler, prepared(&settings, "hi", true));
        scheduler.pump(&settings, &mut engine, &mut reuse);
        let mut request = prepared(
            &settings,
            &"x".repeat(if cold { 12000 } else { 1500 }),
            true,
        );
        Arc::get_mut(&mut request).unwrap().boundaries = None;
        let app = submit(&mut scheduler, request);
        drain(&mut scheduler, &settings, &mut engine, &mut reuse);
        assert!(matches!(app.terminal(), Some(Err((true, _, _)))));
        assert!(matches!(peer.terminal(), Some(Ok(_))));
        assert_eq!(reuse.commits.len(), 1, "partial appender never retained");
    }
}
