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

struct State(usize, Arc<AtomicUsize>);
impl Drop for State {
    fn drop(&mut self) {
        self.1.fetch_add(1, Ordering::SeqCst);
    }
}

struct Stub {
    settings: Settings,
    starts: usize,
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
        .with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    let wire = exchange(
        &mut backend,
        post(&json!({"model":"test","input":"one two three four five","max_output_tokens":1})),
    );
    assert!(wire.starts_with("HTTP/1.1 400"), "{wire}");
    assert_eq!(backend.starts, 0);
    assert!(super::super::bind_loopback("0.0.0.0:0").is_err());
    let mut tokenizer = backend.settings.tokenizer;
    tokenizer.with_decoder(None::<tokenizers::decoders::DecoderWrapper>);
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
        admit_request_memory(estimate, u64::MAX, signals)
            .unwrap_err()
            .reason,
        qwen_llm::metal::MetalMemoryAdmissionReason::RequiredBytesOverflow
    );
    assert!(caller_cpu_reserve(u64::MAX, 1, 129280).is_none());
}
