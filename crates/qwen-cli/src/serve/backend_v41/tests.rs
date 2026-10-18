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
    let denied = qwen_llm::metal::MemoryAdmissionDenied {
        reason: qwen_llm::metal::MetalMemoryAdmissionReason::WorkingSetInsufficient,
        required_bytes: Some(4096),
        signals: MetalMemorySignals {
            recommended_max_bytes: 1024,
            current_allocated_bytes: 0,
            process_limit_remaining_bytes: Some(1024),
        },
        working_set_headroom_bytes: Some(1024),
    };
    let result = backend.settings.execute::<()>(
        &request,
        "prompt",
        &mut Sink,
        |_, _| Ok(()),
        |_| {
            Err(state_error(
                qwen_llm::deepseek_v41::forward::StateError::MemoryAdmissionDenied(denied),
            ))
        },
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
            qwen_llm::metal::MemoryAdmissionDenied {
                reason: qwen_llm::metal::MetalMemoryAdmissionReason::ProcessInsufficient,
                required_bytes: Some(4096),
                signals: MetalMemorySignals {
                    recommended_max_bytes: 1 << 40,
                    current_allocated_bytes: 0,
                    process_limit_remaining_bytes: Some(1024),
                },
                working_set_headroom_bytes: Some(1 << 40),
            },
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
        &mut Sink(cancel.clone()),
        |_, _| Ok(()),
        |_| {
            cancel.store(true, Ordering::Release);
            Err(ServeError::server_error("cancelled allocation"))
        },
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

struct State(usize, Arc<AtomicUsize>);
impl Drop for State {
    fn drop(&mut self) {
        self.1.fetch_add(1, Ordering::SeqCst);
    }
}

struct Stub {
    settings: Settings,
    starts: usize,
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
    use tokenizers::{decoders::byte_level::ByteLevel, models::wordlevel::WordLevel};
    let vocabulary = [
        "[UNK]".to_owned(),
        deepseek_recipe_encoding::v4::EOS_TOKEN.into(),
        emission.into(),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, word)| (word, i as u32))
    .collect();
    let model = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteLevel::default()));
    let mut settings = Settings::new(
        tokenizer,
        64,
        8,
        SamplingConfig::default(),
        1,
        "test".into(),
    )
    .unwrap();
    settings.headroom = || Some(0);
    Stub {
        settings,
        starts: 0,
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
    fn output_headroom(&self) -> Headroom {
        self.settings.headroom
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
        let epoch = Instant::now();
        self.settings.execute(
            request,
            prompt,
            sink,
            |positions, reserve| {
                self.admitted_positions.push(positions);
                let estimate = qwen_llm::deepseek_v41::forward::Config::default()
                    .estimate_request_memory(positions)
                    .unwrap();
                let remaining = headroom().map(|n| {
                    let current = self.available.load(Ordering::Acquire);
                    if n == 0 { current } else { current.min(n) }
                });
                admit_request_memory(
                    estimate,
                    reserve,
                    MetalMemorySignals {
                        recommended_max_bytes: 1 << 40,
                        current_allocated_bytes: 0,
                        process_limit_remaining_bytes: remaining,
                    },
                )
                .map_err(|e| super::super::transport_memory::memory_refusal("V4.1 request", &e))
            },
            |_| {
                self.starts += 1;
                if let Some(error) = self.state_failure.take() {
                    return Err(state_error(error));
                }
                Ok(State(0, drops))
            },
            |state, token, cancel| {
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
                state.0 += 1;
                let count = steps.fetch_add(1, Ordering::SeqCst) + 1;
                if drop_after == Some(count) {
                    available.store(1, Ordering::Release);
                }
                let mut input = vec![0.; 3];
                input[token as usize] = 1.;
                let normalized =
                    qwen_llm::deepseek_v41::operators::rms_norm(&input, &[1.; 3], 1e-20)?;
                qwen_llm::deepseek_v41::operators::Linear {
                    input: 3,
                    output: 3,
                    weight: vec![0., 0., 0., 0., 0., 10., 10., 0., 0.],
                }
                .forward(&normalized)
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
    assert!(wire.starts_with("HTTP/1.1 200"), "{wire}");
    if stream {
        wire.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str::<Value>(data).ok())
            .find(|v| v["type"] == "response.completed")
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
    let mut backend = stub("\u{c3}");
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
        json!({"model":"test","input":"hi","max_output_tokens":65}),
        json!({"model":"test","input":"hi","x_qwen":{"prefix_reuse":true}}),
        json!({"model":"test","input":"hi","prompt_cache_key":"x"}),
    ] {
        let wire = exchange(&mut backend, post(&body));
        assert!(wire.starts_with("HTTP/1.1 400"), "{wire}");
    }
    backend.settings.headroom = || Some(1);
    let wire = exchange(&mut backend, post(&json!({"model":"test","input":"hi"})));
    assert!(wire.starts_with("HTTP/1.1 503"), "{wire}");
    assert_eq!(backend.starts, starts);
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
    assert_eq!(
        (request.temperature, request.top_k, request.seed),
        (Some(0.7), Some(2), Some(123))
    );
    let mut request = profile
        .parse(&json!({"model":"test","input":"hi","temperature":0,"x_qwen":{"seed":7}}))
        .unwrap();
    profile.normalize(&mut request).unwrap();
    assert_eq!((request.temperature, request.seed), (Some(0.), Some(7)));
    backend.settings.context = 4;
    backend
        .settings
        .tokenizer
        .hf_mut()
        .with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    let wire = exchange(
        &mut backend,
        post(&json!({"model":"test","input":"one two three four five","max_output_tokens":1})),
    );
    assert!(wire.starts_with("HTTP/1.1 400"), "{wire}");
    assert_eq!(backend.starts, 0);
    assert!(super::super::bind_loopback("0.0.0.0:0").is_err());
    let mut tokenizer = backend.settings.tokenizer;
    tokenizer
        .hf_mut()
        .with_decoder(None::<tokenizers::decoders::DecoderWrapper>);
    assert!(
        Settings::new(
            tokenizer,
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
fn v41_request_memory_refuses_before_state_and_rechecks_mid_prefill() {
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
    assert_eq!(backend.admitted_positions, [1]);

    backend.available.store(128 << 20, Ordering::Release);
    backend.settings.context = 256;
    backend
        .settings
        .tokenizer
        .hf_mut()
        .with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    backend.drop_after = Some(1);
    let wire = exchange(
        &mut backend,
        post(&json!({"model":"test","input":"one two three four five six","max_output_tokens":1})),
    );
    assert!(wire.starts_with("HTTP/1.1 503"), "{wire}");
    assert!(wire.contains("memory_admission_denied"));
    assert_eq!(backend.starts, 1);
    assert_eq!(
        backend.steps.load(Ordering::Acquire),
        1,
        "second prefill forward must not run"
    );
    assert_eq!(backend.drops.load(Ordering::Acquire), 1);
    let checks = &backend.admitted_positions[1..];
    assert_eq!(
        checks.len(),
        3,
        "preflight and both attempted prefill steps"
    );
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
    assert_eq!(backend.steps.load(Ordering::Acquire), 1);
    assert_eq!(backend.admitted_positions, [1, 1]);
}

#[test]
fn v41_request_memory_splits_metal_cpu_and_preserves_typed_refusals() {
    let estimate = qwen_llm::deepseek_v41::forward::Config::default()
        .estimate_request_memory(1)
        .unwrap();
    let extra_cpu = 4096;
    let signals = MetalMemorySignals {
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
        .reason,
        qwen_llm::metal::MetalMemoryAdmissionReason::ProcessSignalUnavailable
    );
    let cpu_short = MetalMemorySignals {
        process_limit_remaining_bytes: Some(estimate.total_bytes + extra_cpu - 1),
        ..signals
    };
    let error = admit_request_memory(estimate, extra_cpu, cpu_short).unwrap_err();
    assert_eq!(
        error.reason,
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
            .reason,
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
        .reason,
        qwen_llm::metal::MetalMemoryAdmissionReason::WorkingSetInsufficient
    );
    assert_eq!(
        admit_request_memory(estimate, u64::MAX, signals)
            .unwrap_err()
            .reason,
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
    for (maximum, eos_first, generated, transitions, decode_ms) in [
        (3, false, 2, 1, 27.),
        (1, false, 1, 0, 13.),
        (3, true, 1, 0, 6.),
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
                &mut TimedSink(&clock),
                |_, _| {
                    clock.set(clock.get() + 3);
                    Ok(())
                },
                |_| {
                    clock.set(clock.get() + 2000);
                    Ok(0)
                },
                |state, _, _| {
                    clock.set(clock.get() + 7);
                    *state += 1;
                    Ok(if eos_first || *state > 1 {
                        vec![0., 10., 0.]
                    } else {
                        vec![0., 0., 10.]
                    })
                },
                || Duration::from_millis(clock.get()),
            )
            .unwrap();
        let stats = outcome.stats.unwrap();
        assert_eq!(
            (stats.prompt_tokens, stats.matched_tokens, stats.restore_ms),
            (1, 0, 0.)
        );
        assert_eq!(outcome.usage.output_tokens, generated);
        assert_eq!(
            outcome.end,
            if maximum == 1 && !eos_first {
                GenerationEnd::TokenLimit
            } else {
                GenerationEnd::StopToken(1)
            }
        );
        let phases = stats.phases.unwrap();
        assert_eq!(phases["prefill_ms"], 14.);
        assert_eq!(phases["decode_ms"], decode_ms);
        assert_eq!(phases["transitions"], transitions);
        assert_eq!(
            phases["transition_ms"],
            if transitions == 1 { 12. } else { 0. }
        );
        assert_eq!(phases["generated_tokens"], generated);
        assert_eq!(phases["prefill_mode"], "single_token_forward");
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
