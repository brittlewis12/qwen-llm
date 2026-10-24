use super::*;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize},
    mpsc,
};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(5);
thread_local! { static OUTPUT_HEADROOM: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }

fn prompt_len(backend: &Stub, input: &str) -> usize {
    let request = backend
        .settings
        .profile()
        .parse(&json!({"model":"test", "input":input}))
        .unwrap();
    let prompt = backend.settings.profile().render(&request).unwrap();
    backend
        .settings
        .tokenizer
        .encode(&prompt, false)
        .unwrap()
        .len()
}

#[test]
fn v41_serve_preparation_uses_validated_context_alias() {
    for experts in [272, 320, 384] {
        let gguf = crate::deepseek_v41_fixture::gguf(experts, "deepseek41.max_position_embeddings");
        assert!(gguf.get_u64("deepseek41.context_length").is_none());
        let (mut args, _) = crate::Args::parse_with_explicit([
            "qwen",
            "serve",
            "-m",
            "fixture.gguf",
            "--max-context-tokens",
            "64",
            "--max-tokens",
            "8",
        ]);
        let crate::cli::Invocation::Serve(mut invocation) = crate::cli::normalize(&mut args) else {
            panic!("serve invocation")
        };
        let prepared = Prepared::new(&gguf, &invocation).unwrap();
        assert_eq!(prepared.settings.context, 64);
        assert_eq!(prepared.settings.limits.default_max_tokens, 8);
        invocation.max_context_tokens = Some(65);
        assert!(Prepared::new(&gguf, &invocation).is_err());
    }
}

#[test]
fn v41_typed_state_pressure_is_503_and_never_steps() {
    struct Sink;
    impl GenerationSink for Sink {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            Ok(())
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let backend = stub("answer");
    let request = backend
        .settings
        .profile()
        .parse(&json!({"model":"test","input":"hi","max_output_tokens":1}))
        .unwrap();
    let denied = qwen_llm::metal::MemoryAdmissionDenied::new(
        qwen_llm::metal::MetalMemoryAdmissionReason::WorkingSetInsufficient,
        Some(4096),
        MetalMemorySignals {
            host_budget: Some(qwen_llm::metal::HostMemoryBudget {
                physical_bytes: 1024 << 30,
                conservative_available_bytes: 1024 << 30,
                external_gap_bytes: 0,
                private_shadow_bytes: 0,
            }),
            recommended_max_bytes: 1024,
            current_allocated_bytes: 0,
            process_limit_remaining_bytes: Some(1024),
        },
        Some(1024),
    )
    .expect("test reason refuses admission");
    let result = backend.settings.execute::<()>(
        &request,
        "prompt",
        None,
        &mut Sink,
        |_, _| Ok(()),
        |_| {
            Err(state_error(
                qwen_llm::deepseek_v41::forward::StateError::MemoryAdmissionDenied(denied),
            ))
        },
        &mut reuse::NoReuse,
        |_, _, _, _| panic!("state refusal must precede prefill"),
        |_, _, _| panic!("state refusal must precede forward"),
        || Duration::ZERO,
    );
    let Err(BackendFailure::Serve(error)) = result else {
        panic!("expected typed refusal")
    };
    assert_eq!(error.status, 503);
    assert_eq!(error.code, Some("memory_admission_denied"));
    let invalid = state_error(qwen_llm::deepseek_v41::forward::StateError::Invalid(
        "bad geometry".into(),
    ));
    assert_eq!(invalid.status, 500);
    assert_ne!(invalid.code, Some("memory_admission_denied"));
}

#[test]
fn v41_state_refusal_is_http_503_and_next_request_starts_fresh() {
    let mut backend = stub("answer");
    backend.state_failure = Some(
        qwen_llm::deepseek_v41::forward::StateError::MemoryAdmissionDenied(
            qwen_llm::metal::MemoryAdmissionDenied::new(
                qwen_llm::metal::MetalMemoryAdmissionReason::ProcessInsufficient,
                Some(4096),
                MetalMemorySignals {
                    host_budget: Some(qwen_llm::metal::HostMemoryBudget {
                        physical_bytes: 1024 << 30,
                        conservative_available_bytes: 1024 << 30,
                        external_gap_bytes: 0,
                        private_shadow_bytes: 0,
                    }),
                    recommended_max_bytes: 1 << 40,
                    current_allocated_bytes: 0,
                    process_limit_remaining_bytes: Some(1024),
                },
                Some(1 << 40),
            )
            .expect("test reason refuses admission"),
        ),
    );
    let body = json!({"model":"test","input":"hi","max_output_tokens":1});
    let wire = exchange(&mut backend, post(&body));
    assert!(wire.starts_with("HTTP/1.1 503"), "{wire}");
    assert!(wire.contains("memory_admission_denied"), "{wire}");
    assert_eq!(backend.starts, 1);
    assert_eq!(backend.steps.load(Ordering::Acquire), 0);
    let wire = exchange(&mut backend, post(&body));
    assert!(wire.starts_with("HTTP/1.1 200"), "{wire}");
    assert_eq!(backend.starts, 2);
    assert_eq!(backend.drops.load(Ordering::Acquire), 1);
}

#[test]
fn v41_cancellation_during_state_construction_remains_aborted() {
    struct Sink(Arc<AtomicBool>);
    impl GenerationSink for Sink {
        fn cancellation_flag(&self) -> Option<Arc<AtomicBool>> {
            Some(self.0.clone())
        }
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            Ok(())
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let backend = stub("answer");
    let request = backend
        .settings
        .profile()
        .parse(&json!({"model":"test","input":"hi","max_output_tokens":1}))
        .unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let result = backend.settings.execute::<()>(
        &request,
        "prompt",
        None,
        &mut Sink(cancel.clone()),
        |_, _| Ok(()),
        |_| {
            cancel.store(true, Ordering::Release);
            Err(ServeError::server_error("cancelled allocation"))
        },
        &mut reuse::NoReuse,
        |_, _, _, _| panic!("cancelled construction must not prefill"),
        |_, _, _| panic!("cancelled construction must not step"),
        || Duration::ZERO,
    );
    assert!(
        matches!(result, Err(BackendFailure::Aborted(error)) if error.kind() == io::ErrorKind::Interrupted)
    );
}

fn io_snapshot(pageins: u64, bytes: u64) -> qwen_llm::pid_metrics::PidSnapshot {
    qwen_llm::pid_metrics::PidSnapshot {
        pageins,
        diskio_bytesread: bytes,
        diskio_byteswritten: 0,
        resident_size: 0,
        phys_footprint: 0,
    }
}

#[test]
fn token_io_aggregates_consumed_steps() {
    let mut observations = memory_io::Observations::default();
    for (position, pageins, bytes) in [(0, 3, 100), (1, 7, 300), (2, 0, 0)] {
        let mut snapshots = [
            Some(io_snapshot(10, 1000)),
            Some(io_snapshot(10 + pageins, 1000 + bytes)),
        ]
        .into_iter();
        memory_io::observe(
            Some(&mut observations),
            position,
            || snapshots.next().unwrap(),
            || Ok::<_, ()>(()),
        )
        .unwrap();
    }
    let stats = observations.json();
    assert_eq!(stats["attempted_tokens"], 3);
    assert_eq!(stats["successful_tokens"], 3);
    assert_eq!(
        stats["process_pageins"],
        json!({"count":3,"total":10,"max":7,"per_token_mean":10.0/3.0})
    );
    assert_eq!(
        stats["process_disk_read_bytes"],
        json!({"count":3,"total":400,"max":300,"per_token_mean":400.0/3.0})
    );
}

#[test]
fn token_io_missing_and_regressed_counters_are_not_zero() {
    for missing_before in [false, true] {
        let mut observations = memory_io::Observations::default();
        let mut snapshots = if missing_before {
            [None, Some(io_snapshot(4, 40))]
        } else {
            [Some(io_snapshot(4, 40)), None]
        }
        .into_iter();
        memory_io::observe(
            Some(&mut observations),
            0,
            || snapshots.next().unwrap(),
            || Ok::<_, ()>(()),
        )
        .unwrap();
        let stats = observations.json();
        for metric in ["process_pageins", "process_disk_read_bytes"] {
            assert_eq!(
                stats[metric],
                json!({"count":0,"total":null,"max":null,"per_token_mean":null})
            );
        }
    }
    let mut observations = memory_io::Observations::default();
    assert!(observations.json()["process_pageins"]["total"].is_null());
    let mut snapshots = [Some(io_snapshot(4, 40)), Some(io_snapshot(3, 60))].into_iter();
    memory_io::observe(
        Some(&mut observations),
        0,
        || snapshots.next().unwrap(),
        || Ok::<_, ()>(()),
    )
    .unwrap();
    memory_io::observe(
        Some(&mut observations),
        1,
        || Some(io_snapshot(5, 60)),
        || Ok::<_, ()>(()),
    )
    .unwrap();
    let stats = observations.json();
    assert_eq!(stats["process_pageins"]["count"], 1);
    assert!(stats["process_pageins"]["total"].is_null());
    assert!(stats["process_pageins"]["max"].is_null());
    assert!(stats["process_pageins"]["per_token_mean"].is_null());
    assert_eq!(stats["process_disk_read_bytes"]["total"], 20);
}

#[test]
fn token_io_nonstats_never_snapshots_and_preserves_errors() {
    let error = memory_io::observe(
        None,
        9,
        || panic!("nonstats snapshot"),
        || Err::<(), _>("cancelled"),
    );
    assert_eq!(error, Err("cancelled"));
    let value = memory_io::observe(None, 9, || panic!("nonstats snapshot"), || Ok::<_, ()>(42));
    assert_eq!(value, Ok(42));
    let mut observations = memory_io::Observations::default();
    let mut snapshots = 0;
    let error = memory_io::observe(
        Some(&mut observations),
        9,
        || {
            snapshots += 1;
            Some(io_snapshot(snapshots, snapshots * 10))
        },
        || Err::<(), _>("original forward error"),
    );
    assert_eq!(error, Err("original forward error"));
    assert_eq!(snapshots, 2);
    assert_eq!(observations.json()["attempted_tokens"], 1);
    assert_eq!(observations.json()["successful_tokens"], 0);
    assert_eq!(observations.json()["process_pageins"]["total"], 1);
}

#[test]
fn prefill_io_counts_rows_but_reports_chunk_observation_maxima() {
    let mut observations = memory_io::ChunkObservations::default();
    for (position, rows, delta) in [(0, 32, 64), (32, 3, 6)] {
        let mut snapshots = [
            Some(io_snapshot(10, 100)),
            Some(io_snapshot(10 + delta, 100 + delta)),
        ]
        .into_iter();
        memory_io::observe_chunk(
            Some(&mut observations),
            position,
            rows,
            || snapshots.next().unwrap(),
            || Ok::<_, ()>(()),
        )
        .unwrap();
    }
    let stats = observations.json();
    assert_eq!(stats["scope"], "whole_process_native_prefill_chunk");
    assert_eq!(stats["attempted_tokens"], 35);
    assert_eq!(stats["successful_tokens"], 35);
    assert_eq!(stats["attempted_chunks"], 2);
    assert_eq!(
        stats["process_pageins"],
        json!({"count":2,"total":70,"max_chunk":64,"per_token_mean":2.0})
    );
    let result = memory_io::observe_chunk::<(), _>(
        None,
        0,
        32,
        || panic!("nonstats must not observe"),
        || Err("cancelled"),
    );
    assert_eq!(result, Err("cancelled"));
}

struct State(usize, Arc<AtomicUsize>);
impl Drop for State {
    fn drop(&mut self) {
        self.1.fetch_add(1, Ordering::SeqCst);
    }
}

struct Stub {
    settings: Settings,
    output_headroom: super::super::http::Headroom,
    starts: usize,
    emission: u32,
    state_failure: Option<qwen_llm::deepseek_v41::forward::StateError>,
    drops: Arc<AtomicUsize>,
    block: Option<mpsc::Sender<()>>,
    owner: std::thread::ThreadId,
    available: Arc<AtomicU64>,
    steps: Arc<AtomicUsize>,
    drop_after: Option<usize>,
    admitted_positions: Vec<usize>,
    // This model deliberately cannot cross threads, just like a borrowed owner.
    _not_send: std::rc::Rc<()>,
}

fn stub(emission: &str) -> Stub {
    let tokenizer = tokenizer_fixture::tokenizer(Some(emission));
    let mut settings = Settings::new(
        tokenizer,
        16384,
        8,
        SamplingConfig::default(),
        tokenizer_fixture::EOS,
        "test".into(),
    )
    .unwrap();
    settings.headroom = || finite_headroom(1 << 30);
    Stub {
        settings,
        output_headroom: || Some(1 << 30),
        starts: 0,
        emission: tokenizer_fixture::EMISSION,
        state_failure: None,
        drops: Arc::new(AtomicUsize::new(0)),
        block: None,
        owner: std::thread::current().id(),
        available: Arc::new(AtomicU64::new(1 << 40)),
        steps: Arc::new(AtomicUsize::new(0)),
        drop_after: None,
        admitted_positions: vec![],
        _not_send: std::rc::Rc::new(()),
    }
}

impl GenerationBackend for Stub {
    fn model_id(&self) -> &str {
        "test"
    }
    fn request_profile(&self) -> RequestProfile {
        self.settings.profile()
    }
    fn output_headroom(&self) -> super::super::http::Headroom {
        self.output_headroom
    }
    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        assert_eq!(self.owner, std::thread::current().id());
        let drops = self.drops.clone();
        let mut block = self.block.take();
        let available = self.available.clone();
        let steps = self.steps.clone();
        let drop_after = self.drop_after;
        let headroom = self.settings.headroom;
        let emission = self.emission;
        let consume = std::cell::RefCell::new(
            |state: &mut State, tokens: &[u32], cancel: &AtomicBool| -> Result<(), String> {
                if let Some(entered) = block.take() {
                    entered.send(()).unwrap();
                    let start = Instant::now();
                    while !cancel.load(Ordering::Acquire) && start.elapsed() < WAIT {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    assert!(
                        cancel.load(Ordering::Acquire),
                        "HTTP disconnect must reach a layer without another sink tick"
                    );
                    return Err("cancelled inside synthetic layer".into());
                }
                for _ in tokens {
                    state.0 += 1;
                    let count = steps.fetch_add(1, Ordering::SeqCst) + 1;
                    if drop_after == Some(count) {
                        available.store(1, Ordering::Release);
                        OUTPUT_HEADROOM.with(|value| value.set(1));
                    }
                }
                Ok(())
            },
        );
        let epoch = Instant::now();
        self.settings.execute(
            request,
            prompt,
            None,
            sink,
            |positions, reserve| {
                self.admitted_positions.push(positions);
                let estimate = qwen_llm::deepseek_v41::forward::Config::default()
                    .estimate_request_memory(positions)
                    .unwrap();
                let remaining = headroom().map(|headroom| {
                    let current = self.available.load(Ordering::Acquire);
                    current.min(headroom.effective_remaining_bytes())
                });
                admit_request_memory(
                    estimate,
                    reserve,
                    MetalMemorySignals {
                        host_budget: Some(qwen_llm::metal::HostMemoryBudget {
                            physical_bytes: 1024 << 30,
                            conservative_available_bytes: 1024 << 30,
                            external_gap_bytes: 0,
                            private_shadow_bytes: 0,
                        }),
                        recommended_max_bytes: 1 << 40,
                        current_allocated_bytes: 0,
                        process_limit_remaining_bytes: remaining,
                    },
                )
                .map_err(StateError::MemoryAdmissionDenied)
            },
            |_| {
                self.starts += 1;
                if let Some(error) = self.state_failure.take() {
                    return Err(state_error(error));
                }
                Ok(State(0, drops))
            },
            &mut reuse::NoReuse,
            |state, tokens, cancel, with_logits| {
                consume.borrow_mut()(state, tokens, cancel).map_err(ServeError::server_error)?;
                if !with_logits {
                    return Ok(vec![]);
                }
                let mut logits = vec![0.; 258];
                logits[emission as usize] = 100.;
                Ok(logits)
            },
            |state, token, cancel| {
                consume.borrow_mut()(state, &[token], cancel)?;
                let mut logits = vec![0.; 258];
                logits[tokenizer_fixture::EOS as usize] = 100.;
                Ok(logits)
            },
            || epoch.elapsed(),
        )
    }
}

fn post(body: &Value) -> String {
    let body = body.to_string();
    format!(
        "POST /v1/responses HTTP/1.1\r\nhost: localhost\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
}

fn exchange(backend: &mut Stub, request: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let client = std::thread::spawn(move || {
        let mut socket = TcpStream::connect(addr).unwrap();
        socket.set_read_timeout(Some(WAIT)).unwrap();
        socket.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        socket.read_to_string(&mut response).unwrap();
        response
    });
    let (socket, _) = listener.accept().unwrap();
    let mut activity = super::super::owner_activity::OwnerActivity::default();
    let guard = activity.admission().try_admit().unwrap();
    super::super::transport::handle_connection(socket, backend, None, guard, || Ok(())).unwrap();
    let mut finished = 0;
    activity.drain_finished(|| finished += 1);
    assert_eq!(finished, 1);
    assert!(activity.is_settled());
    client.join().unwrap()
}

fn response(wire: &str, stream: bool) -> Value {
    terminal_response(wire, stream, "response.completed")
}

fn terminal_response(wire: &str, stream: bool, terminal: &str) -> Value {
    assert!(wire.starts_with("HTTP/1.1 200"), "{wire}");
    if stream {
        wire.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str::<Value>(data).ok())
            .find(|v| v["type"] == terminal)
            .unwrap()["response"]
            .clone()
    } else {
        serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap()
    }
}

#[test]
fn v41_backend_http_fresh_state_reasoning_eos_and_token_limit() {
    let mut backend = stub("plan</think>answer");
    for (index, (stream, max)) in [(false, 8), (true, 8), (false, 1)].into_iter().enumerate() {
        let wire = exchange(
            &mut backend,
            post(
                &json!({"model":"test","input":"Hello", "stream":stream, "max_output_tokens":max, "reasoning":{"effort":"high"},"x_qwen":{"seed":42}}),
            ),
        );
        let response = if max == 1 {
            assert!(wire.starts_with("HTTP/1.1 200"));
            serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap()
        } else {
            response(&wire, stream)
        };
        assert_eq!(response["output"][0]["type"], "reasoning");
        assert_eq!(response["output"][0]["content"][0]["text"], "plan");
        assert_eq!(response["output"][1]["content"][0]["text"], "answer");
        assert_eq!(
            response["usage"]["input_tokens_details"]["cached_tokens"],
            0
        );
        assert_eq!(
            response["usage"]["output_tokens"],
            if max == 1 { 1 } else { 2 }
        );
        assert_eq!(backend.starts, index + 1);
        assert_eq!(backend.drops.load(Ordering::SeqCst), index + 1);
    }
}

#[test]
fn v41_backend_flushes_incomplete_utf8_at_eos() {
    let mut backend = stub("answer");
    backend.emission = 0xc3;
    for stream in [false, true] {
        let reply = response(
            &exchange(
                &mut backend,
                post(&json!({"model":"test","input":"hi","stream":stream})),
            ),
            stream,
        );
        assert_eq!(reply["output"][0]["content"][0]["text"], "\u{fffd}");
        assert_eq!(reply["usage"]["output_tokens"], 2);
    }
}

#[test]
fn v41_http_partial_tool_is_incomplete_at_budget_but_failed_at_eos() {
    let output = "answer<\u{ff5c}DSML\u{ff5c} calls><\u{ff5c}DSML\u{ff5c} invoke name=\"lookup\"><\u{ff5c}DSML\u{ff5c} parameter name=\"q\" string=\"true\">partial";
    for stream in [false, true] {
        for maximum in [1, 8] {
            let mut backend = stub(output);
            let wire = exchange(
                &mut backend,
                post(&json!({"model":"test", "input":"hi",
                "stream":stream, "max_output_tokens":maximum,
                "tools":[{"type":"function", "name":"lookup", "parameters":{"type":"object"}}]})),
            );
            assert!(!wire.contains("\"type\":\"function_call\""), "{wire}");
            assert!(
                !wire.contains("partial"),
                "tool arguments must stay withheld: {wire}"
            );
            if maximum == 1 {
                assert!(wire.starts_with("HTTP/1.1 200"), "{wire}");
                let reply: Value = if stream {
                    wire.lines()
                        .filter_map(|line| line.strip_prefix("data: "))
                        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                        .find(|event| event["type"] == "response.incomplete")
                        .expect("budget termination event")["response"]
                        .clone()
                } else {
                    serde_json::from_str(wire.split_once("\r\n\r\n").unwrap().1).unwrap()
                };
                assert_eq!(reply["status"], "incomplete");
                assert_eq!(reply["incomplete_details"]["reason"], "max_output_tokens");
                assert_eq!(reply["output"][0]["content"][0]["text"], "answer");
            } else if stream {
                assert!(wire.contains("response.failed"), "{wire}");
                assert!(!wire.contains("response.completed"), "{wire}");
            } else {
                assert!(wire.starts_with("HTTP/1.1 500"), "{wire}");
            }
        }
    }
}

#[test]
fn v41_backend_tools_multiturn_non_generation_and_preflight_rejections() {
    let output = "plan</think><\u{ff5c}DSML\u{ff5c} calls><\u{ff5c}DSML\u{ff5c} invoke name=\"lookup\"></\u{ff5c}DSML\u{ff5c} invoke></\u{ff5c}DSML\u{ff5c} calls>";
    let mut backend = stub(output);
    for stream in [false, true] {
        let body = json!({"model":"test","input":[{"role":"user","content":"first"},{"type":"function_call","call_id":"c","name":"lookup","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":"found"},{"role":"user","content":"next"}],"reasoning":{"effort":"high"},"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"stream":stream});
        let reply = response(&exchange(&mut backend, post(&body)), stream);
        assert_eq!(reply["output"][1]["type"], "function_call");
        assert_eq!(reply["output"][1]["name"], "lookup");
        assert_eq!(reply["output"][1]["arguments"], "{}");
    }
    let starts = backend.starts;
    let wire = exchange(
        &mut backend,
        "GET /v1/models HTTP/1.1\r\nhost: localhost\r\n\r\n".into(),
    );
    assert!(wire.starts_with("HTTP/1.1 200") && wire.contains("test"));
    for body in [
        json!({"model":"test","input":"hi","max_output_tokens":16385}),
        json!({"model":"test","input":"hi","x_qwen":{"history_thinking":"preserve"}}),
        json!({"model":"test","input":"hi","prompt_cache_key":"x"}),
    ] {
        let wire = exchange(&mut backend, post(&body));
        assert!(wire.starts_with("HTTP/1.1 400"), "{wire}");
    }
    backend.settings.headroom = || finite_headroom(1);
    backend.output_headroom = || Some(1);
    let wire = exchange(&mut backend, post(&json!({"model":"test","input":"hi"})));
    assert!(wire.starts_with("HTTP/1.1 503"), "{wire}");
    assert_eq!(backend.starts, starts);
}

#[test]
fn v41_http_complete_tool_at_budget_is_retained_in_incomplete_response() {
    let output = "<\u{ff5c}DSML\u{ff5c} calls><\u{ff5c}DSML\u{ff5c} invoke name=\"lookup\"></\u{ff5c}DSML\u{ff5c} invoke></\u{ff5c}DSML\u{ff5c} calls>";
    for stream in [false, true] {
        let mut backend = stub(output);
        let wire = exchange(
            &mut backend,
            post(&json!({"model":"test", "input":"hi",
            "stream":stream, "max_output_tokens":1,
            "tools":[{"type":"function", "name":"lookup", "parameters":{"type":"object"}}]})),
        );
        let reply = terminal_response(&wire, stream, "response.incomplete");
        assert_eq!(reply["status"], "incomplete");
        assert_eq!(reply["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(reply["output"][0]["type"], "function_call");
        assert_eq!(reply["output"][0]["name"], "lookup");
        assert_eq!(reply["output"][0]["arguments"], "{}");
    }
}

#[test]
fn v41_backend_http_disconnect_cancels_inside_layer_and_next_request_recovers() {
    use std::os::fd::AsRawFd;
    let mut backend = stub("answer");
    let (entered, entry) = mpsc::channel();
    backend.block = Some(entered);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let client = std::thread::spawn(move || {
        let mut socket = TcpStream::connect(addr).unwrap();
        socket
            .write_all(post(&json!({"model":"test","input":"hi"})).as_bytes())
            .unwrap();
        entry.recv_timeout(WAIT).unwrap();
        let linger = libc::linger {
            l_onoff: 1,
            l_linger: 0,
        };
        assert_eq!(
            unsafe {
                libc::setsockopt(
                    socket.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_LINGER,
                    (&linger as *const libc::linger).cast(),
                    std::mem::size_of_val(&linger) as libc::socklen_t,
                )
            },
            0
        );
        drop(socket);
    });
    let (socket, _) = listener.accept().unwrap();
    let mut activity = super::super::owner_activity::OwnerActivity::default();
    let guard = activity.admission().try_admit().unwrap();
    let _ =
        super::super::transport::handle_connection(socket, &mut backend, None, guard, || Ok(()));
    client.join().unwrap();
    activity.drain_finished(|| {});
    assert!(activity.is_settled());
    assert_eq!(backend.drops.load(Ordering::SeqCst), 1);
    let wire = exchange(&mut backend, post(&json!({"model":"test","input":"retry"})));
    assert_eq!(
        response(&wire, false)["output"][0]["content"][0]["text"],
        "answer"
    );
    assert_eq!(backend.drops.load(Ordering::SeqCst), 2);
}

#[test]
fn v41_backend_defaults_prompt_budget_and_loopback_only() {
    let mut backend = stub("answer");
    backend.settings.sampling = SamplingConfig {
        temperature: 0.7,
        top_k: 2,
        top_p: 0.9,
        min_p: 0.,
        seed: 123,
    };
    let profile = backend.settings.profile();
    let mut request = profile
        .parse(&json!({"model":"test","input":"hi"}))
        .unwrap();
    profile.normalize(&mut request).unwrap();
    assert_eq!((request.temperature, request.top_k), (Some(0.7), Some(2)));
    let first_seed = request.seed.expect("normalized request has a seed");
    assert_ne!(first_seed, 123, "startup seed is not a request default");
    profile.normalize(&mut request).unwrap();
    assert_eq!(
        request.seed,
        Some(first_seed),
        "normalization is idempotent"
    );
    let mut second = profile
        .parse(&json!({"model":"test","input":"hi"}))
        .unwrap();
    profile.normalize(&mut second).unwrap();
    assert_ne!(second.seed, Some(first_seed));
    assert_eq!(
        sampling_with_defaults(&request, backend.settings.sampling).seed,
        first_seed
    );
    let mut request = profile
        .parse(&json!({"model":"test","input":"hi","temperature":0,"x_qwen":{"seed":7}}))
        .unwrap();
    profile.normalize(&mut request).unwrap();
    assert_eq!((request.temperature, request.seed), (Some(0.), Some(7)));
    backend.settings.context = 4;
    let wire = exchange(
        &mut backend,
        post(&json!({"model":"test","input":"one two three four five","max_output_tokens":1})),
    );
    assert!(wire.starts_with("HTTP/1.1 400"), "{wire}");
    assert_eq!(backend.starts, 0);
    assert!(super::super::bind_loopback("0.0.0.0:0").is_err());
    // Native tokenizer/config mismatch must still fail before execution.
    assert!(
        Settings::new(
            backend.settings.tokenizer,
            64,
            8,
            SamplingConfig::default(),
            1,
            "test".into()
        )
        .is_err()
    );
}

#[test]
fn v41_request_memory_admits_creation_once_and_checks_output_growth() {
    let mut backend = stub("answer");
    backend.available.store(16 << 20, Ordering::Release);
    let wire = exchange(
        &mut backend,
        post(&json!({"model":"test","input":"hi","max_output_tokens":1})),
    );
    assert!(wire.starts_with("HTTP/1.1 503"), "{wire}");
    assert!(wire.contains("memory_admission_denied"));
    assert_eq!(backend.starts, 0);
    assert_eq!(backend.steps.load(Ordering::Acquire), 0);
    assert_eq!(backend.drops.load(Ordering::Acquire), 0);
    assert_eq!(backend.admitted_positions, [prompt_len(&backend, "hi")]);

    backend.available.store(128 << 20, Ordering::Release);
    backend.settings.context = 256;
    backend.drop_after = Some(1);
    OUTPUT_HEADROOM.with(|value| value.set(128 << 20));
    backend.settings.headroom = || finite_headroom(OUTPUT_HEADROOM.with(|value| value.get()));
    backend.output_headroom = || Some(OUTPUT_HEADROOM.with(|value| value.get()));
    let wire = exchange(
        &mut backend,
        post(&json!({"model":"test","input":"one two three four five six","max_output_tokens":1})),
    );
    assert!(wire.starts_with("HTTP/1.1 503"), "{wire}");
    assert!(wire.contains("memory_admission_denied"));
    assert_eq!(backend.starts, 1);
    assert_eq!(
        backend.steps.load(Ordering::Acquire),
        prompt_len(&backend, "one two three four five six")
            .min(crate::family_options::V41_PREFILL_ROWS),
        "pressure must stop before the next bounded prefill chunk"
    );
    assert_eq!(backend.drops.load(Ordering::Acquire), 1);
    let checks = &backend.admitted_positions[1..];
    assert_eq!(checks.len(), 1, "only creation prices the request state");
    assert!(checks[0] > 1 && checks[0] < 256);
    assert!(checks.iter().all(|n| *n == checks[0]));
}

#[test]
fn v41_request_memory_prices_actual_use_not_million_token_capacity() {
    let mut backend = stub("answer");
    backend.settings.context = 1_048_576;
    backend.available.store(64 << 20, Ordering::Release);
    let wire = exchange(
        &mut backend,
        post(&json!({"model":"test","input":"hi","max_output_tokens":1})),
    );
    assert!(wire.starts_with("HTTP/1.1 200"), "{wire}");
    assert_eq!(backend.starts, 1);
    let prompt_tokens = prompt_len(&backend, "hi");
    assert_eq!(backend.steps.load(Ordering::Acquire), prompt_tokens);
    assert_eq!(backend.admitted_positions, [prompt_tokens]);
}

#[test]
fn v41_allocated_state_is_not_readmitted_during_prefill_or_decode() {
    let mut backend = stub("answer");
    backend.drop_after = Some(1);
    let wire = exchange(
        &mut backend,
        post(&json!({
            "model":"test", "input":"hi", "max_output_tokens":3
        })),
    );
    assert!(wire.starts_with("HTTP/1.1 200"), "{wire}");
    assert_eq!(backend.available.load(Ordering::Acquire), 1);
    let prompt_tokens = prompt_len(&backend, "hi");
    assert_eq!(backend.admitted_positions, [prompt_tokens + 2]);
    assert_eq!(backend.steps.load(Ordering::Acquire), prompt_tokens + 1);
    assert_eq!(backend.drops.load(Ordering::Acquire), 1);
}

#[test]
fn v41_request_memory_splits_metal_cpu_and_preserves_typed_refusals() {
    let estimate = qwen_llm::deepseek_v41::forward::Config::default()
        .estimate_request_memory(1)
        .unwrap();
    let extra_cpu = 4096;
    let signals = MetalMemorySignals {
        host_budget: Some(qwen_llm::metal::HostMemoryBudget {
            physical_bytes: 1024 << 30,
            conservative_available_bytes: 1024 << 30,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
        recommended_max_bytes: (200 << 30) + estimate.metal_bytes,
        current_allocated_bytes: 200 << 30,
        process_limit_remaining_bytes: Some(estimate.total_bytes + extra_cpu),
    };
    // Resident weights are in current allocation; CPU KV/output aren't charged
    // to the device working set. A one-byte shortage on either side refuses.
    admit_request_memory(estimate, extra_cpu, signals).unwrap();
    let omitted = MetalMemorySignals {
        process_limit_remaining_bytes: Some(0),
        ..signals
    };
    admit_request_memory(estimate, extra_cpu, omitted).unwrap();
    let mut host_short = omitted;
    let host = host_short.host_budget.as_mut().unwrap();
    // Caller bytes fit; the request's CPU demand still must be charged.
    host.conservative_available_bytes = estimate.cpu_bytes + extra_cpu - 1;
    let denied = admit_request_memory(estimate, extra_cpu, host_short).unwrap_err();
    assert_eq!(
        denied.reason(),
        qwen_llm::metal::MetalMemoryAdmissionReason::HostAvailabilityInsufficient
    );
    assert_eq!(
        super::super::transport_memory::memory_refusal("V4.1 request", &denied).status,
        503
    );
    assert_eq!(
        admit_request_memory(
            estimate,
            extra_cpu,
            MetalMemorySignals {
                process_limit_remaining_bytes: None,
                ..signals
            },
        )
        .unwrap_err()
        .reason(),
        qwen_llm::metal::MetalMemoryAdmissionReason::ProcessSignalUnavailable
    );
    let cpu_short = MetalMemorySignals {
        process_limit_remaining_bytes: Some(estimate.total_bytes + extra_cpu - 1),
        ..signals
    };
    let error = admit_request_memory(estimate, extra_cpu, cpu_short).unwrap_err();
    assert_eq!(
        error.reason(),
        qwen_llm::metal::MetalMemoryAdmissionReason::ProcessInsufficient
    );
    let mapped = super::super::transport_memory::memory_refusal("V4.1 request", &error);
    assert_eq!(
        (mapped.status, mapped.code),
        (503, Some("memory_admission_denied"))
    );
    let metal_short = MetalMemorySignals {
        current_allocated_bytes: signals.current_allocated_bytes + 1,
        ..signals
    };
    assert_eq!(
        admit_request_memory(estimate, extra_cpu, metal_short)
            .unwrap_err()
            .reason(),
        qwen_llm::metal::MetalMemoryAdmissionReason::WorkingSetInsufficient
    );
    assert_eq!(
        admit_request_memory(
            estimate,
            extra_cpu,
            MetalMemorySignals {
                process_limit_remaining_bytes: Some(0),
                ..metal_short
            },
        )
        .unwrap_err()
        .reason(),
        qwen_llm::metal::MetalMemoryAdmissionReason::WorkingSetInsufficient
    );
    assert_eq!(
        admit_request_memory(estimate, u64::MAX, signals)
            .unwrap_err()
            .reason(),
        qwen_llm::metal::MetalMemoryAdmissionReason::RequiredBytesOverflow
    );
    assert!(caller_cpu_reserve(u64::MAX, 1, 129280).is_none());
}

#[test]
fn v41_native_phase_clock_excludes_preparation_and_counts_eos_and_max_one() {
    use std::cell::Cell;
    struct TimedSink<'a>(&'a Cell<u64>);
    impl GenerationSink for TimedSink<'_> {
        fn tick(&mut self) -> io::Result<()> {
            self.0.set(self.0.get() + 2);
            Ok(())
        }
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            self.0.set(self.0.get() + 5);
            Ok(())
        }
    }
    for (maximum, eos_first, generated, transitions, decode_work) in [
        (3, false, 2, 1, 16),
        (1, false, 1, 0, 9),
        (3, true, 1, 0, 2),
    ] {
        let backend = stub("answer");
        let request = backend
            .settings
            .profile()
            .parse(&json!({"model":"test","input":"hi","max_output_tokens":maximum}))
            .unwrap();
        let clock = Cell::new(0);
        let outcome = backend
            .settings
            .execute(
                &request,
                "prompt",
                None,
                &mut TimedSink(&clock),
                |_, _| {
                    clock.set(clock.get() + 3);
                    Ok(())
                },
                |_| {
                    clock.set(clock.get() + 2000);
                    Ok(0)
                },
                &mut reuse::NoReuse,
                |state, tokens, _, with_logits| {
                    assert!(with_logits);
                    clock.set(clock.get() + 7 * tokens.len() as u64);
                    *state += tokens.len();
                    let mut logits = vec![0.; 258];
                    logits[if eos_first {
                        tokenizer_fixture::EOS
                    } else {
                        tokenizer_fixture::EMISSION
                    } as usize] = 100.;
                    Ok(logits)
                },
                |state, _, _| {
                    clock.set(clock.get() + 7);
                    *state += 1;
                    let mut logits = vec![0.; 258];
                    logits[tokenizer_fixture::EOS as usize] = 100.;
                    Ok(logits)
                },
                || Duration::from_millis(clock.get()),
            )
            .unwrap();
        let stats = outcome.stats.unwrap();
        assert_eq!(
            (stats.prompt_tokens, stats.matched_tokens, stats.restore_ms),
            (6, 0, 0.)
        );
        assert_eq!(outcome.usage.output_tokens, generated);
        assert_eq!(
            outcome.end,
            if maximum == 1 && !eos_first {
                GenerationEnd::TokenLimit
            } else {
                GenerationEnd::StopToken(tokenizer_fixture::EOS as i32)
            }
        );
        let phases = stats.phases.unwrap();
        assert_eq!(phases["prefill_ms"], 48.);
        assert_eq!(clock.get(), 2 + 3 + 2000 + 48 + decode_work);
        // Decode uses the shared loop's real clock, not this prefill clock.
        assert!(phases["decode_ms"].as_f64().unwrap() >= phases["transition_ms"].as_f64().unwrap());
        assert_eq!(phases["transitions"], transitions);
        if transitions == 0 {
            assert_eq!(phases["transition_ms"], 0.);
        } else {
            assert!(phases["transition_ms"].as_f64().unwrap() > 0.);
        }
        assert_eq!(phases["generated_tokens"], generated);
        assert_eq!(
            phases["prefill_mode"],
            crate::family_options::V41_PREFILL_MODE
        );
        assert_eq!(phases["prefill_rows"], 6);
        assert_eq!(phases["prefill_chunks"], 1);
        assert_eq!(phases["ced"], false);
    }
}

#[test]
fn v41_native_phase_stats_are_opt_in_on_existing_http_echo() {
    let mut backend = stub("answer");
    for (echo, stream) in [(false, false), (true, false), (false, true), (true, true)] {
        let wire = exchange(
            &mut backend,
            post(&json!({"model":"test","input":"hi","stream":stream,"x_qwen":{"stats":echo}})),
        );
        let reply = response(&wire, stream);
        if echo {
            assert_eq!(reply["x_qwen"]["version"], "serve_stats_v1");
            assert_eq!(reply["x_qwen"]["matched_tokens"], 0);
            assert_eq!(reply["x_qwen"]["phases"]["family"], "deepseek_v41");
            assert_eq!(reply["x_qwen"]["phases"]["transitions"], 1);
        } else {
            assert!(reply.get("x_qwen").is_none());
        }
    }
}

#[test]
fn v41_http_sampling_seeds_are_per_request_and_explicit_seeds_are_echoed() {
    let mut backend = stub("answer");
    let mut seeds = vec![];
    for stream in [false, true] {
        for explicit in [None, Some(7)] {
            let mut body = json!({"model":"test", "input":"hi", "stream":stream,
                "x_qwen":{"stats":true}});
            if let Some(seed) = explicit {
                body["x_qwen"]["seed"] = json!(seed);
            }
            let reply = response(&exchange(&mut backend, post(&body)), stream);
            let seed = reply["x_qwen"]["seed"]
                .as_u64()
                .expect("effective sampler seed");
            if let Some(explicit) = explicit {
                assert_eq!(seed, explicit);
            } else {
                seeds.push(seed);
            }
        }
    }
    assert_ne!(seeds[0], seeds[1]);
}
