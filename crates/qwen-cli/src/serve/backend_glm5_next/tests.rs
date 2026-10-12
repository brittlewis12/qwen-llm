use super::*;
use crate::serve::items::TemplateStyle;
use crate::serve::output_partition::{GenerationEnd, OutputPartition, OutputProtocol};
use crate::serve::partition::PartitionEvent;
use serde_json::{Value, json};

mod chunk_checkpoint_ab;

#[test]
#[ignore = "GLM53_GGUF leased session-recreation cost screen; MTL_DEBUG_LAYER=1"]
fn gpu_cold_session_recreation_cost() {
    struct TimedSink {
        started: Instant,
        first_piece_ms: Option<f64>,
    }
    impl GenerationSink for TimedSink {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            self.first_piece_ms
                .get_or_insert_with(|| self.started.elapsed().as_secs_f64() * 1e3);
            Ok(())
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let path = gguf_path();
    let source = GgufFile::open(&path).unwrap();
    let prepared = Prepared::new(&source, &invocation(&path, Some(32768), Some(1))).unwrap();
    let ctx = MetalContext::new().unwrap();
    let weights = load(&ctx, &source, &prepared).unwrap();
    let cache =
        crate::serve::SnapshotCachePlan::resolve(Some(0), Default::default(), ctx.memory_signals())
            .unwrap();
    let mut backend = Glm5NextBackend::new(
        &ctx,
        &weights,
        prepared,
        "glm".into(),
        std::time::Duration::ZERO,
        cache,
    );
    backend.prefix_reuse = true;
    backend.warm_up().unwrap();
    let long: String = (0..120)
        .map(|i| format!("Rule {i}: when asked about topic {i}, answer in one short sentence. "))
        .collect();
    let mut expected = [None, None];
    for repeat in 0..4 {
        for recreate in if repeat % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        } {
            backend.force_cold_reallocation = recreate;
            for (shape, instructions, input) in [
                ("short", "Answer briefly.", "What is 6 times 7?"),
                ("long", long.as_str(), "What is 5 times 9?"),
            ] {
                let request = prepared_request(
                    &backend,
                    json!({"model":"glm", "instructions":instructions,
                    "input":[{"role":"user","content":input}],
                    "reasoning":{"effort":"low"}, "temperature":0,
                    "max_output_tokens":1}),
                );
                let mut sink = TimedSink {
                    started: Instant::now(),
                    first_piece_ms: None,
                };
                let outcome = backend.generate_prepared(request, &mut sink).unwrap();
                let wall_ms = sink.started.elapsed().as_secs_f64() * 1e3;
                assert_eq!(outcome.usage.cached_tokens, 0);
                assert_eq!(backend.cache.len(), 0);
                let (drop_ms, alloc_ms, reset_ms) = backend.last_cold_session_ms.unwrap();
                assert_eq!(reset_ms == 0.0, recreate);
                let expected = &mut expected[usize::from(shape == "long")];
                if let Some(bits) = expected {
                    assert_eq!(&backend.last_prefill_logits, bits);
                } else {
                    *expected = Some(backend.last_prefill_logits.clone());
                }
                eprintln!(
                    "cold_session_cost {}",
                    json!({
                        "repeat":repeat, "recreate":recreate, "shape":shape, "capacity":backend.prepared.capacity,
                        "rows":backend.frozen_rows, "prompt_tokens":backend.history.len(),
                        "drop_ms":drop_ms, "alloc_ms":alloc_ms, "reset_ms":reset_ms,
                        "first_piece_ms":sink.first_piece_ms, "wall_ms":wall_ms,
                    })
                );
            }
        }
    }
    backend.force_cold_reallocation = false;
    for lineage in [PackedLineage::Exact, PackedLineage::Fast] {
        backend.lineage = lineage;
        let request = prepared_request(
            &backend,
            json!({"model":"glm",
            "input":"Reply with the number 7.", "temperature":0,
            "reasoning":{"effort":"low"}, "max_output_tokens":1}),
        );
        backend
            .generate_prepared(request, &mut Sink::default())
            .unwrap();
        assert_eq!(
            backend.last_cold_session_ms.unwrap().2,
            0.0,
            "incompatible arithmetic must recreate, not reset"
        );
        assert_eq!(
            backend.session.as_ref().unwrap().packed_lineage(),
            Some(lineage)
        );
    }
    struct CancelledTransport;
    impl GenerationSink for CancelledTransport {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            unreachable!()
        }
        fn tick(&mut self) -> io::Result<()> {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "cancel before reset",
            ))
        }
    }
    let old_history = backend.history.clone();
    let old_position = backend.session.as_ref().unwrap().position();
    let request = prepared_request(
        &backend,
        json!({"model":"glm",
        "input":"An unrelated request", "temperature":0, "max_output_tokens":1}),
    );
    assert!(
        backend
            .generate_prepared(request, &mut CancelledTransport)
            .is_err()
    );
    assert_eq!(backend.history, old_history);
    assert_eq!(backend.session.as_ref().unwrap().position(), old_position);
    assert!(!backend.session.as_ref().unwrap().is_poisoned());
}

fn invocation(
    path: &str,
    capacity: Option<usize>,
    maximum: Option<usize>,
) -> crate::cli::ServeInvocation {
    crate::cli::ServeInvocation {
        lens_data_dir: None,
        lens_config: None,
        web_root: None,
        lens_allowed_origin: Vec::new(),
        model: path.into(),
        addr: "invalid-listen-address".into(),
        max_tokens: maximum,
        max_context_tokens: capacity,
        snapshot_cache_mib: None,
        snapshot_policy: Default::default(),
        durable: crate::serve::durable::DurableSnapshotConfig::off(),
        drafter: None,
        trace_sse: None,
        template_style: Default::default(),
        idle_residency_secs: None,
    }
}

fn gguf_path() -> String {
    std::env::var("GLM53_GGUF").expect("GLM53_GGUF (GLM-5.3-Flash shard 1)")
}

#[test]
fn snapshot_namespaces_isolate_prefill_rows() {
    let ns = CacheNamespace {
        lineage: PackedLineage::Fast,
        schedule: Schedule::FastSharedSplitV1,
        rows: 512,
    };
    let mut cache = SnapshotCache::new(1024, Default::default());
    assert!(cache.insert_strict_in(ns, vec![1, 2], "snapshot", 10));
    assert!(cache.peek_exact_in(&ns, &[1, 2]).is_some());
    assert!(
        cache
            .peek_exact_in(&CacheNamespace { rows: 2048, ..ns }, &[1, 2])
            .is_none()
    );
}

#[test]
#[ignore = "GLM53_GGUF CPU metadata-only real ledger planning; no Metal context"]
fn cpu_prefill_plan_generic_hc_and_sparse_ledger() {
    use qwen_llm::glm5_next::memory::{Glm5NextMemoryLedger, apple_16k_price};
    use qwen_llm::metal::MetalMemorySignals;
    let path = gguf_path();
    let source = GgufFile::open(&path).unwrap();
    let prepared = Prepared::new(&source, &invocation(&path, Some(4096), Some(1))).unwrap();
    let ledger = |rows, generic| {
        Glm5NextMemoryLedger::new_with_generic_hc(
            prepared.artifact.config(),
            0,
            4096,
            rows as u64,
            generic,
            &apple_16k_price,
        )
        .unwrap()
    };
    let generic = ledger(2048, true);
    assert!(generic.session_buffer_bytes() > ledger(2048, false).session_buffer_bytes());
    assert!(
        generic
            .terms()
            .iter()
            .any(|&(name, bytes)| name == "packed_sparse" && bytes > 0)
    );
    let price = prefill_plan::Price::from_ledger(&generic);
    assert!(
        price.admission > price.buffers,
        "dynamic reserve stays in admission"
    );
    let signals = MetalMemorySignals {
        recommended_max_bytes: price.admission - 1,
        current_allocated_bytes: 0,
        process_limit_remaining_bytes: Some(0),
    };
    let plan = prefill_plan::select(
        &prefill_plan::candidates(4096, true),
        signals,
        0,
        None,
        |rows| Ok(prefill_plan::Price::from_ledger(&ledger(rows, true))),
    )
    .unwrap();
    assert!(plan.rows < 2048);
    assert_eq!(ledger(plan.rows, true).capacity(), 4096);
}

#[test]
#[ignore = "GLM53_GGUF adaptive 2048-row serve cold/hit/cancel/continuation; MTL_DEBUG_LAYER=1; requires snapshot headroom"]
fn gpu_adaptive_prefill_cold_hit_and_continuation() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let path = gguf_path();
    let source = GgufFile::open(&path).unwrap();
    let prepared = Prepared::new(&source, &invocation(&path, Some(4096), Some(1))).unwrap();
    let ctx = MetalContext::new().unwrap();
    let weights = load(&ctx, &source, &prepared).unwrap();
    let cache = crate::serve::SnapshotCachePlan::resolve(
        Some(8192),
        Default::default(),
        ctx.memory_signals(),
    )
    .unwrap();
    let mut backend = Glm5NextBackend::new(
        &ctx,
        &weights,
        prepared,
        "glm".into(),
        std::time::Duration::ZERO,
        cache,
    );
    backend.prefix_reuse = true;
    backend.fast_snapshots = true;
    backend.lineage = PackedLineage::Fast;
    backend.warm_up().unwrap();
    assert_eq!(backend.frozen_rows, None);
    let instructions: String = (0..120)
        .map(|i| format!("Rule {i}: when asked about topic {i}, answer in one short sentence. "))
        .collect();
    let prompt = prepared_request(
        &backend,
        json!({
            "model":"glm", "instructions":instructions,
            "input":[{"role":"user","content":QUESTION}],
            "reasoning":{"effort":"low"}, "temperature":0, "max_output_tokens":1
        }),
    );
    let encode = |text: &str| {
        decode_loop::encode_checked(
            backend.prepared.artifact.tokenizer(),
            text,
            false,
            backend.prepared.artifact.config().vocab_size,
            FAMILY,
        )
    };
    let tokens = encode(&prompt.prompt).unwrap();
    let cuts = cut_positions(
        encode,
        &prompt.prompt,
        &tokens,
        prompt.boundaries,
        Schedule::FastSharedSplitV1,
    );
    assert_eq!(cuts.len(), 1);
    let shared = cuts[0];
    assert!(
        shared > 2048 && shared < tokens.len(),
        "fixture must exercise a full 2048-row chunk plus tail: shared={shared}, prompt={}",
        tokens.len()
    );
    assert!(tokens.len() + 32 < backend.prepared.capacity);
    let prefix_bytes = SnapshotCache::<Glm5NextSnapshot, CacheNamespace>::entry_bytes(
        shared,
        snapshot_bytes(&weights.config, shared as u64).unwrap(),
    )
    .unwrap();
    assert!(prefix_bytes <= backend.cache.max_bytes());
    let scenario = plan_session(
        &ctx,
        &weights,
        &backend.prepared,
        2048,
        Sink::default().transport_reserve_bytes(),
        Some(prefix_bytes),
    )
    .unwrap();
    assert_eq!(
        scenario.rows, 2048,
        "this opt-in gate requires 2048-row session plus useful-prefix headroom: {scenario:?}"
    );
    assert!(scenario.preserved);
    let run = |label: &str, backend: &mut Glm5NextBackend<'_, '_>| {
        let started = Instant::now();
        let mut sink = Sink::default();
        let outcome = backend
            .generate_prepared(prompt.clone(), &mut sink)
            .unwrap();
        eprintln!(
            "adaptive {label}: rows={:?} shared_rows={shared} prompt_rows={} cached_rows={} elapsed_ms={:.1}",
            backend.frozen_rows,
            tokens.len(),
            outcome.usage.cached_tokens,
            started.elapsed().as_secs_f64() * 1e3
        );
        (
            outcome.usage.cached_tokens,
            backend.last_prefill_logits.clone(),
            sink.bytes,
        )
    };
    let cold = run("cold", &mut backend);
    assert_eq!(cold.0, 0);
    assert_eq!(
        backend.frozen_rows,
        Some(2048),
        "actual allocation must exercise the admitted 2048-row scenario"
    );
    assert_eq!(backend.prepared.capacity, 4096);
    assert_eq!(
        backend.session.as_ref().unwrap().ledger().prefill_rows(),
        2048
    );
    let namespace = CacheNamespace {
        lineage: PackedLineage::Fast,
        schedule: Schedule::FastSharedSplitV1,
        rows: 2048,
    };
    assert!(
        backend
            .cache
            .peek_exact_in(&namespace, &tokens[..shared])
            .is_some()
    );
    assert!(
        backend
            .cache
            .peek_exact_in(
                &CacheNamespace {
                    rows: 512,
                    ..namespace
                },
                &tokens[..shared]
            )
            .is_none()
    );
    backend.session = None;
    backend.history.clear();
    let hit = run("snapshot_hit", &mut backend);
    assert_eq!(hit.0, shared);
    assert_eq!(hit.1, cold.1, "hit and miss use the same frozen schedule");
    assert_eq!(hit.2, cold.2);

    backend.session = None;
    backend.history.clear();
    backend.cache = SnapshotCache::new(
        backend.cache.max_bytes(),
        backend.snapshot_cache_plan.policy,
    );
    // Admission tick, first chunk tick, then cancellation before chunk two.
    let mut cancelled = Sink {
        abort_tick: Some(3),
        ..Sink::default()
    };
    let started = Instant::now();
    assert!(matches!(
        backend.generate_prepared(prompt.clone(), &mut cancelled),
        Err(BackendFailure::Aborted(_))
    ));
    eprintln!(
        "adaptive cancelled: rows={:?} committed_rows={} elapsed_ms={:.1}",
        backend.frozen_rows,
        backend.history.len(),
        started.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(backend.frozen_rows, Some(2048));
    assert_eq!(backend.history, tokens[..2048]);
    assert_eq!(backend.session.as_ref().unwrap().position(), 2048);
    assert!(backend.live_canonical);
    assert_eq!(
        backend.cache.len(),
        0,
        "cancel occurs before the shared cut"
    );
    let retry = run("cancel_retry", &mut backend);
    assert_eq!(retry.0, 2048);
    assert_eq!(
        retry.1, cold.1,
        "chunk-boundary retry keeps the uninterrupted schedule"
    );
    assert_eq!(retry.2, cold.2);
    assert!(
        backend
            .cache
            .peek_exact_in(&namespace, &tokens[..shared])
            .is_some()
    );

    // With one output token no sampled token is forwarded. Append a special
    // token to keep the original generation header's tokenization intact.
    let continuation = format!("{}<|user|>Continue.", prompt.prompt);
    let previous = backend.history.len();
    assert_eq!(backend.history, tokens);
    let continued_tokens = decode_loop::encode_checked(
        backend.prepared.artifact.tokenizer(),
        &continuation,
        false,
        backend.prepared.artifact.config().vocab_size,
        FAMILY,
    )
    .unwrap();
    assert!(continued_tokens.starts_with(&backend.history));
    assert!((1..=32).contains(&(continued_tokens.len() - previous)));
    let started = Instant::now();
    let continued = backend
        .generate(&prompt.request, &continuation, &mut Sink::default())
        .unwrap();
    eprintln!(
        "adaptive short_suffix: rows={:?} suffix_rows={} cached_rows={} elapsed_ms={:.1}",
        backend.frozen_rows,
        continued_tokens.len() - previous,
        continued.usage.cached_tokens,
        started.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(continued.usage.cached_tokens, previous);
    assert_eq!(backend.history, continued_tokens);
    assert_eq!(backend.frozen_rows, Some(2048));

    // Test-only pinned schedules isolate the throughput tradeoff using the
    // same resident weights, request, snapshot policy, and transport reserve.
    // Reverse the order on the second pair; compare identity within a plan,
    // not between different floating-point execution schedules.
    for rows in [512, 2048, 2048, 512] {
        backend.session = None;
        backend.history.clear();
        backend.cache = SnapshotCache::new(
            backend.cache.max_bytes(),
            backend.snapshot_cache_plan.policy,
        );
        backend.frozen_rows = Some(rows);
        let cold = run("comparison_cold", &mut backend);
        assert_eq!(cold.0, 0);
        eprintln!(
            "adaptive comparison_memory: rows={rows} session_buffer_bytes={}",
            backend
                .session
                .as_ref()
                .unwrap()
                .ledger()
                .session_buffer_bytes()
        );
        backend.session = None;
        backend.history.clear();
        let hit = run("comparison_hit", &mut backend);
        assert_eq!(hit.0, shared);
        assert_eq!(hit.1, cold.1);
        assert_eq!(hit.2, cold.2);
    }
}

/// `QWEN_GLM_FAST_PRECISION`: unset is `half`; each named precision parses
/// (surrounding space ignored); anything else is a startup error that names
/// the accepted values.
#[test]
fn fast_precision_setting_parses_or_refuses() {
    assert_eq!(parse_fast_precision(None).unwrap(), FastPrecision::Half);
    for precision in FastPrecision::ALL {
        assert_eq!(
            parse_fast_precision(Some(precision.name())).unwrap(),
            precision
        );
    }
    assert_eq!(
        parse_fast_precision(Some(" dense_f32\n")).unwrap(),
        FastPrecision::DenseF32
    );
    for bad in ["", "F32", "fp32", "full"] {
        let error = parse_fast_precision(Some(bad)).unwrap_err().to_string();
        assert!(error.contains("half, dense_f32, f32"), "{bad:?}: {error}");
    }
}

#[test]
#[ignore = "CPU/header-only GLM53_GGUF startup refusals; never binds a socket or initializes Metal"]
fn cpu_startup_refuses_bad_limits_before_listener_or_metal() {
    let path = gguf_path();
    for (capacity, maximum, change, expected) in [
        (None, Some(8), None, "requires --max-context-tokens"),
        (Some(64), None, None, "requires explicit --max-tokens"),
        (Some(64), Some(65), None, "exceeds --max-context-tokens"),
        (Some(4 << 20), Some(8), None, "exceeds model context"),
        (Some(0), Some(0), None, "must be greater than 0"),
        (
            Some(64),
            Some(8),
            Some("drafter"),
            "--drafter is not supported for",
        ),
        (
            Some(64),
            Some(8),
            Some("upstream"),
            "--template-style upstream",
        ),
    ] {
        let mut invocation = invocation(&path, capacity, maximum);
        match change {
            Some("drafter") => invocation.drafter = Some("nonexistent-drafter.gguf".into()),
            Some("upstream") => invocation.template_style = TemplateStyle::Upstream,
            _ => {}
        }
        let error = crate::serve::run_serve(invocation).unwrap_err();
        assert!(format!("{error:#}").contains(expected), "{error:#}");
    }
}

#[derive(Default)]
struct Sink {
    bytes: Vec<u8>,
    ticks: usize,
    pieces: usize,
    abort_tick: Option<usize>,
    /// Disconnect on this (1-based) piece.
    abort_piece: Option<usize>,
}

impl GenerationSink for Sink {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.pieces += 1;
        if self.abort_piece == Some(self.pieces) {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "test disconnect"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        self.ticks += 1;
        if self.abort_tick == Some(self.ticks) {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "test disconnect"));
        }
        Ok(())
    }
}

fn request(backend: &Glm5NextBackend<'_, '_>, body: Value) -> (ServeRequest, String) {
    let mut request = backend.parse_request(&body).unwrap();
    backend.normalize_request(&mut request).unwrap();
    let prompt = backend.render_prompt(&request).unwrap();
    (request, prompt)
}

/// (reasoning, answer) through the serve output grammar.
fn split(bytes: &[u8], end: GenerationEnd) -> (String, String) {
    let mut partition = OutputPartition::new(OutputProtocol::Glm5NextChat);
    let mut events = Vec::new();
    partition.push(bytes, &mut events);
    partition.finish(end, &mut events).unwrap();
    let (mut reasoning, mut answer) = (String::new(), String::new());
    for event in events {
        match event {
            PartitionEvent::Reasoning(text) => reasoning.push_str(&text),
            PartitionEvent::Visible(text) => answer.push_str(&text),
            PartitionEvent::ReasoningClosed => {}
            PartitionEvent::FunctionCall(_) => panic!("no-tools grammar parsed a call"),
        }
    }
    (reasoning, answer)
}

fn wire(backend: &mut Glm5NextBackend<'_, '_>, body: Value) -> String {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::time::Duration;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let client = std::thread::spawn(move || {
        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(600)))
            .unwrap();
        let body = body.to_string();
        write!(stream, "POST /v1/responses HTTP/1.1\r\nhost: localhost\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}", body.len()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    });
    let (stream, _) = listener.accept().unwrap();
    // The Metal backend stays on this thread; only the socket client is spawned.
    crate::serve::http::handle_connection(&stream, backend, None).unwrap();
    drop(stream);
    client.join().unwrap()
}

/// The run lane's cold path for the same prompt: a fresh session, packed
/// prefill, the shared serial loop.
fn reference_bytes(
    backend: &Glm5NextBackend<'_, '_>,
    request: &ServeRequest,
    prompt: &str,
) -> Vec<u8> {
    let tokenizer = backend.prepared.artifact.tokenizer();
    let tokens: Vec<u32> = tokenizer
        .encode(prompt, false)
        .unwrap()
        .into_iter()
        .map(|id| id as u32)
        .collect();
    let mut session = backend.fresh_session(0, backend.lineage).unwrap();
    let logits = session.prefill_packed(backend.ctx, &tokens).unwrap();
    let mut sampler = Sampler::new(render::sampling(request)).unwrap();
    let mut bytes = Vec::new();
    crate::generate_serial(
        logits,
        request.max_output_tokens.unwrap(),
        &CHAT_STOPS,
        &mut sampler,
        |token| {
            bytes.extend_from_slice(tokenizer.try_decode_piece_bytes_exact(token)?);
            Ok(())
        },
        |token| Ok(session.forward(backend.ctx, token as u32)?),
    )
    .unwrap();
    bytes
}

const QUESTION: &str = "What is 6 times 7? Reply with the number only.";

#[test]
#[ignore = "GLM53_GGUF GPU serve correctness under MTL_DEBUG_LAYER=1; production CLI lease, no server process"]
fn gpu_live_session_extends_resumes_and_resets() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let path = gguf_path();
    let source = GgufFile::open(&path).unwrap();
    let prepared = Prepared::new(&source, &invocation(&path, Some(1024), Some(64))).unwrap();
    assert_eq!(prepared.prefill_rows, 1024);
    // A CLI test links the production library: this takes the real lease.
    let ctx = MetalContext::new().unwrap();
    let weights = load(&ctx, &source, &prepared).unwrap();
    // No snapshot cache: this test pins live-session behaviour alone.
    let no_snapshots =
        crate::serve::SnapshotCachePlan::resolve(Some(0), Default::default(), ctx.memory_signals())
            .unwrap();
    let mut backend = Glm5NextBackend::new(
        &ctx,
        &weights,
        prepared,
        "glm".into(),
        std::time::Duration::ZERO,
        no_snapshots,
    );
    backend.frozen_rows = Some(512);
    eprintln!("warm_up_ms={:.1}", backend.warm_up().unwrap());
    let greedy = |input: Value, effort: &str| {
        json!({"model":"glm","input":input,"reasoning":{"effort":effort},
            "temperature":0,"max_output_tokens":64})
    };

    // Cold request: the run lane's bytes, history = prompt + forwarded.
    let (first_req, first_prompt) = request(&backend, greedy(json!(QUESTION), "low"));
    assert_eq!(
        backend.output_protocol(&first_req),
        OutputProtocol::Glm5NextChat
    );
    let mut first = Sink::default();
    let started = Instant::now();
    let outcome = backend
        .generate(&first_req, &first_prompt, &mut first)
        .unwrap();
    eprintln!(
        "first request ms={:.1} usage={:?}",
        started.elapsed().as_secs_f64() * 1e3,
        outcome.usage
    );
    assert_eq!(outcome.usage.cached_tokens, 0);
    assert_eq!(
        first.bytes,
        reference_bytes(&backend, &first_req, &first_prompt)
    );
    assert_eq!(
        backend.history.len(),
        outcome.usage.input_tokens + outcome.usage.output_tokens - 1
    );
    let (reasoning, answer) = split(&first.bytes, outcome.end);
    assert!(answer.contains("42"), "{reasoning:?} / {answer:?}");

    // Replaying the exchange plus a new turn extends the live session.
    let replay = |next: &str| {
        json!([
            {"role":"user","content":QUESTION},
            {"type":"reasoning","content":reasoning},
            {"role":"assistant","content":answer},
            {"role":"user","content":next}])
    };
    let (second_req, second_prompt) = request(&backend, greedy(replay("And 6 times 8?"), "low"));
    assert!(second_prompt.starts_with(&first_prompt));
    let history = backend.history.len();
    let mut second = Sink::default();
    let outcome = backend
        .generate(&second_req, &second_prompt, &mut second)
        .unwrap();
    assert_eq!(
        outcome.usage.cached_tokens, history,
        "replayed history must re-tokenize to the consumed tokens"
    );
    let (_, second_answer) = split(&second.bytes, outcome.end);
    assert!(second_answer.contains("48"), "{second_answer:?}");

    // A cold run of the same prompt: semantically the same under Fast...
    backend.prefix_reuse = false;
    let mut cold = Sink::default();
    let outcome = backend
        .generate(&second_req, &second_prompt, &mut cold)
        .unwrap();
    assert_eq!(outcome.usage.cached_tokens, 0);
    assert!(split(&cold.bytes, outcome.end).1.contains("48"));
    eprintln!("fast warm==cold bytes: {}", cold.bytes == second.bytes);
    backend.prefix_reuse = true;
    // ...and bitwise under Exact, whose prefill matches serial decode.
    backend.lineage = PackedLineage::Exact;
    backend.session = None;
    backend.history.clear();
    let mut exact_first = Sink::default();
    let outcome = backend
        .generate(&first_req, &first_prompt, &mut exact_first)
        .unwrap();
    let exact_history = backend.history.clone();
    let first_len = outcome.usage.input_tokens;
    let (r, a) = split(&exact_first.bytes, outcome.end);
    let exact_replay = json!([
        {"role":"user","content":QUESTION},
        {"type":"reasoning","content":r},
        {"role":"assistant","content":a},
        {"role":"user","content":"And 6 times 8?"}]);
    let (exact_req, exact_prompt) = request(&backend, greedy(exact_replay, "low"));
    let history = backend.history.len();
    let mut warm = Sink::default();
    let outcome = backend
        .generate(&exact_req, &exact_prompt, &mut warm)
        .unwrap();
    assert_eq!(outcome.usage.cached_tokens, history);
    backend.prefix_reuse = false;
    let mut cold = Sink::default();
    backend
        .generate(&exact_req, &exact_prompt, &mut cold)
        .unwrap();
    assert_eq!(warm.bytes, cold.bytes, "Exact: warm continuation == cold");
    // The join itself: the logits after (prefill, decode forwards, suffix
    // prefill) equal one cold prefill's, bit for bit.
    let exact_tokens: Vec<u32> = backend
        .prepared
        .artifact
        .tokenizer()
        .encode(&exact_prompt, false)
        .unwrap()
        .into_iter()
        .map(|id| id as u32)
        .collect();
    assert!(exact_tokens.starts_with(&exact_history));
    let mut cold_session = backend.fresh_session(0, backend.lineage).unwrap();
    let cold_logits = cold_session.prefill_packed(&ctx, &exact_tokens).unwrap();
    drop(cold_session);
    let mut warm_session = backend.fresh_session(0, backend.lineage).unwrap();
    warm_session
        .prefill_packed(&ctx, &exact_history[..first_len])
        .unwrap();
    for &token in &exact_history[first_len..] {
        warm_session.forward(&ctx, token).unwrap();
    }
    let warm_logits = warm_session
        .prefill_packed(&ctx, &exact_tokens[exact_history.len()..])
        .unwrap();
    drop(warm_session);
    assert!(
        warm_logits
            .iter()
            .zip(&cold_logits)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "Exact warm and cold logits differ"
    );
    backend.prefix_reuse = true;
    backend.lineage = PackedLineage::Fast;
    backend.session = None;
    backend.history.clear();

    // A different effort rewrites the prefix: a fresh session.
    backend
        .generate(&first_req, &first_prompt, &mut Sink::default())
        .unwrap();
    let (high_req, high_prompt) = request(&backend, greedy(replay("And 6 times 8?"), "high"));
    let outcome = backend
        .generate(&high_req, &high_prompt, &mut Sink::default())
        .unwrap();
    assert_eq!(outcome.usage.cached_tokens, 0);

    // A two-chunk prompt cancelled between chunks keeps the committed chunk;
    // the retry resumes from it with the cold run's bytes.
    let long = format!(
        "{} Summarize the text above in five words.",
        "The quick brown fox jumps over the lazy dog. ".repeat(70)
    );
    let (long_req, long_prompt) = request(&backend, greedy(json!(long), "low"));
    backend.prefix_reuse = false;
    let mut long_cold = Sink::default();
    let cold_outcome = backend
        .generate(&long_req, &long_prompt, &mut long_cold)
        .unwrap();
    assert!(
        cold_outcome.usage.input_tokens > 512,
        "{:?}",
        cold_outcome.usage
    );
    backend.prefix_reuse = true;
    backend.session = None;
    backend.history.clear();
    // Ticks: admission, chunk 1, chunk 2.
    let mut cancelled = Sink {
        abort_tick: Some(3),
        ..Sink::default()
    };
    assert!(matches!(
        backend.generate(&long_req, &long_prompt, &mut cancelled),
        Err(BackendFailure::Aborted(_))
    ));
    assert_eq!(backend.history.len(), 512);
    let mut resumed = Sink::default();
    let outcome = backend
        .generate(&long_req, &long_prompt, &mut resumed)
        .unwrap();
    assert_eq!(outcome.usage.cached_tokens, 512);
    assert_eq!(resumed.bytes, long_cold.bytes);

    // An abort right after prefill (tick 3 of a one-chunk prompt), on the
    // first decoded piece, or after several forwards clears the session; each
    // retry is cold and equal to an uninterrupted run.
    let (count_req, count_prompt) = request(
        &backend,
        greedy(json!("Count from 1 to 20, separated by spaces."), "low"),
    );
    let mut counted = Sink::default();
    backend
        .generate(&count_req, &count_prompt, &mut counted)
        .unwrap();
    assert!(counted.pieces > 8, "{}", counted.pieces);
    for sink in [
        Sink {
            abort_tick: Some(3),
            ..Sink::default()
        },
        Sink {
            abort_piece: Some(1),
            ..Sink::default()
        },
        Sink {
            abort_piece: Some(6),
            ..Sink::default()
        },
    ] {
        let mut sink = sink;
        assert!(matches!(
            backend.generate(&count_req, &count_prompt, &mut sink),
            Err(BackendFailure::Aborted(_))
        ));
        assert!(backend.history.is_empty() && backend.session.is_none());
        let mut retry = Sink::default();
        let outcome = backend
            .generate(&count_req, &count_prompt, &mut retry)
            .unwrap();
        assert_eq!(outcome.usage.cached_tokens, 0);
        assert_eq!(retry.bytes, counted.bytes);
    }

    // Over capacity is refused before the session: no ticks, history kept.
    let history = backend.history.clone();
    let (mut oversized, prompt) = request(&backend, greedy(replay("Next?"), "low"));
    oversized.max_output_tokens = Some(1024);
    let mut sink = Sink::default();
    assert!(matches!(
        backend.generate(&oversized, &prompt, &mut sink),
        Err(BackendFailure::Serve(_))
    ));
    assert_eq!(sink.ticks, 0);
    assert_eq!(backend.history, history);

    // JSON and SSE carry the reasoning item and the answer.
    for stream in [false, true] {
        let mut body = greedy(json!(QUESTION), "low");
        body["stream"] = stream.into();
        let response = wire(&mut backend, body);
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let body = response.split_once("\r\n\r\n").unwrap().1;
        let envelope = if stream {
            let block = body
                .split("\n\n")
                .find(|block| block.starts_with("event: response.completed\n"))
                .unwrap_or_else(|| panic!("{body}"));
            let data = block
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap();
            serde_json::from_str::<Value>(data).unwrap()["response"].clone()
        } else {
            serde_json::from_str::<Value>(body).unwrap()
        };
        let output = envelope["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["text"], reasoning);
        assert_eq!(output[1]["content"][0]["text"], answer);
        assert_eq!(envelope["reasoning"], json!({"effort":"low"}));
    }
}

fn refusal(
    reason: qwen_llm::metal::MetalMemoryAdmissionReason,
) -> qwen_llm::metal::MemoryAdmissionDenied {
    qwen_llm::metal::MemoryAdmissionDenied {
        reason,
        required_bytes: Some(3 << 30),
        signals: qwen_llm::metal::MetalMemorySignals {
            recommended_max_bytes: 112 << 30,
            current_allocated_bytes: 111 << 30,
            process_limit_remaining_bytes: Some(8 << 30),
        },
        working_set_headroom_bytes: Some(1 << 30),
    }
}

/// Lane audit B3: only a typed pressure refusal of the session is a 503;
/// telemetry refusals, geometry and kernel validation failures stay 500.
#[test]
fn session_errors_map_typed_pressure_to_503_and_the_rest_to_500() {
    use qwen_llm::metal::MetalMemoryAdmissionReason as R;
    let admission = |reason| Glm5NextMetalError::MemoryAdmission {
        denied: refusal(reason),
        budget_bytes: 1 << 30,
        advice: qwen_llm::glm5_next_metal::CapacityAdvice::NotEvaluated,
    };
    let pressure = admission(R::BothInsufficient);
    assert!(pressure.is_memory_pressure());
    let error = session_error(pressure);
    assert_eq!(
        (error.status, error.error_type, error.code),
        (503, "server_busy", Some("memory_admission_denied"))
    );
    assert!(
        error.message.contains("both_insufficient"),
        "{}",
        error.message
    );

    let telemetry = admission(R::InvalidWorkingSetSignal);
    assert!(!telemetry.is_memory_pressure());
    let error = session_error(telemetry);
    assert_eq!(
        (error.status, error.code),
        (500, Some("memory_signal_invalid"))
    );

    for other in [
        Glm5NextMetalError::Invalid("capacity must be positive".into()),
        Glm5NextMetalError::Poisoned,
        Glm5NextMetalError::KernelValidation {
            stage: "route",
            block: 7,
            row: None,
            status: -1,
        },
    ] {
        assert!(!other.is_memory_pressure());
        let error = session_error(other);
        assert_eq!((error.status, error.code), (500, None), "{}", error.message);
    }
}

/// A request reads its prompt with its own lineage, and a live session is
/// reused only by requests of the lineage it was built with.
#[test]
fn live_session_reuse_requires_the_same_prefill_lineage() {
    use crate::serve::items::PrefillLineage;
    let mut request = ServeRequest::default();
    assert_eq!(
        request_lineage(&request, PackedLineage::Fast),
        PackedLineage::Fast
    );
    request.prefill_lineage = Some(PrefillLineage::Exact);
    assert_eq!(
        request_lineage(&request, PackedLineage::Fast),
        PackedLineage::Exact
    );
    request.prefill_lineage = Some(PrefillLineage::Fast);
    assert_eq!(
        request_lineage(&request, PackedLineage::Exact),
        PackedLineage::Fast
    );

    assert_eq!(
        reuse_len(218, Some(PackedLineage::Fast), PackedLineage::Fast),
        218
    );
    assert_eq!(
        reuse_len(218, Some(PackedLineage::Fast), PackedLineage::Exact),
        0
    );
    assert_eq!(
        reuse_len(218, Some(PackedLineage::Exact), PackedLineage::Fast),
        0
    );
    assert_eq!(reuse_len(0, None, PackedLineage::Exact), 0);
    assert_eq!(lineage_name(PackedLineage::Exact), "exact");
}

/// Exact always has a snapshot schedule when the cache has a budget; Fast
/// only with its lever on (the default); no budget means no splits and no
/// captures.
#[test]
fn snapshot_schedules_follow_lineage_budget_and_lever() {
    use PackedLineage::{Exact, Fast};
    assert_eq!(snapshot_schedule(1, false, Exact), Some(Schedule::ExactV1));
    assert_eq!(snapshot_schedule(1, true, Exact), Some(Schedule::ExactV1));
    assert_eq!(snapshot_schedule(1, false, Fast), None);
    assert_eq!(
        snapshot_schedule(1, true, Fast),
        Some(Schedule::FastSharedSplitV1)
    );
    for lineage in [Exact, Fast] {
        assert_eq!(snapshot_schedule(0, true, lineage), None);
    }
}

/// Cuts are renderer boundaries mapped through the tokenizer and kept only
/// when the boundary's tokens are a strict prefix of the prompt's.
#[test]
fn cuts_are_verified_renderer_boundaries() {
    // A toy tokenizer: one id per char, except "XY", which merges.
    let encode = |text: &str| -> Result<Vec<u32>, ServeError> {
        let mut ids = Vec::new();
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            if c == 'X' && chars.peek() == Some(&'Y') {
                chars.next();
                ids.push(1000);
            } else {
                ids.push(c as u32);
            }
        }
        Ok(ids)
    };
    let prompt = "sysSTUFF<u>hiXY<a>";
    let tokens = encode(prompt).unwrap();
    let at = |shared: Option<usize>, header: Option<usize>| {
        Some(PromptBoundaries {
            shared_prefix_end: shared,
            generation_header_start: header,
        })
    };
    let header = prompt.find("<a>");
    assert_eq!(
        cut_positions(
            encode,
            prompt,
            &tokens,
            at(Some(8), header),
            Schedule::ExactV1
        ),
        vec![8, 14]
    );
    assert_eq!(
        cut_positions(
            encode,
            prompt,
            &tokens,
            at(Some(8), header),
            Schedule::FastSharedSplitV1
        ),
        vec![8]
    );
    assert!(cut_positions(encode, prompt, &tokens, None, Schedule::ExactV1).is_empty());
    // Between X and Y the boundary's tokens are not a prefix: dropped.
    let inside = prompt.find('Y');
    assert_eq!(
        cut_positions(
            encode,
            prompt,
            &tokens,
            at(inside, header),
            Schedule::ExactV1
        ),
        vec![14]
    );
    // Empty or whole-prompt prefixes and out-of-range offsets cut nothing.
    for offset in [0, prompt.len(), prompt.len() + 5] {
        assert!(
            cut_positions(
                encode,
                prompt,
                &tokens,
                at(Some(offset), None),
                Schedule::ExactV1
            )
            .is_empty(),
            "{offset}"
        );
    }
    // Equal boundaries cut once.
    assert_eq!(
        cut_positions(
            encode,
            prompt,
            &tokens,
            at(Some(8), Some(8)),
            Schedule::ExactV1
        ),
        vec![8]
    );
}

/// The serve renderer reports where the shared instructions-and-tools
/// prefix ends and where the generation header starts; authored text that
/// spells a marker moves neither.
#[test]
fn serve_render_reports_the_shared_prefix_and_header() {
    let profile = crate::serve::request_profile::RequestProfile::Glm5Next {
        default_max_tokens: 64,
        capacity: 4096,
        max_piece_bytes: 64,
    };
    let body = json!({
        "model": "glm",
        "instructions": "Be brief. <|user|> here is just text.",
        "tools": [{"type": "function", "name": "get_weather", "description": "Weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "input": [{"role": "user", "content": "Weather in Paris?"}],
        "reasoning": {"effort": "low"}
    });
    let mut request = profile.parse(&body).unwrap();
    profile.normalize(&mut request).unwrap();
    let (prompt, boundaries) = profile.render_prepared(&request).unwrap();
    assert_eq!(prompt, profile.render(&request).unwrap());
    let boundaries = boundaries.unwrap();
    let shared = boundaries.shared_prefix_end.unwrap();
    assert!(prompt[..shared].ends_with("<|user|> here is just text."));
    assert!(prompt[shared..].starts_with("<|user|>Weather in Paris?"));
    let header = boundaries.generation_header_start.unwrap();
    assert_eq!(&prompt[header..], "<|assistant|><think>");
    // A different conversation with the same instructions and tools shares
    // the prefix byte for byte.
    let mut other = body.clone();
    other["input"] = json!([{"role": "user", "content": "And in Rome?"}]);
    let mut other = profile.parse(&other).unwrap();
    profile.normalize(&mut other).unwrap();
    let (other_prompt, other_boundaries) = profile.render_prepared(&other).unwrap();
    assert_eq!(other_boundaries.unwrap().shared_prefix_end, Some(shared));
    assert_eq!(other_prompt[..shared], prompt[..shared]);
}

/// Session snapshot relief is reserved for process-only memory refusals.
#[test]
fn process_only_session_refusals_release_snapshots_and_keep_the_hit() {
    use qwen_llm::metal::MetalMemoryAdmissionReason as R;
    use qwen_llm::snapshot_policy::SnapshotPolicyConfig;
    let ns = CacheNamespace {
        lineage: PackedLineage::Exact,
        schedule: Schedule::ExactV1,
        rows: 512,
    };
    let filled = || {
        let mut cache: SnapshotCache<&'static str, CacheNamespace> =
            SnapshotCache::new(1 << 20, SnapshotPolicyConfig::LRU);
        assert!(cache.insert_strict_in(ns, vec![1], "keep", 10));
        assert!(cache.insert_strict_in(ns, vec![2], "drop", 10));
        cache
    };
    let admission = |reason| Glm5NextMetalError::MemoryAdmission {
        denied: refusal(reason),
        budget_bytes: 1 << 30,
        advice: qwen_llm::glm5_next_metal::CapacityAdvice::NotEvaluated,
    };

    // Working-set-only and mixed denials are not relieved by dropping CPU
    // snapshots, and their typed reasons and denial details are preserved.
    for reason in [R::WorkingSetInsufficient, R::BothInsufficient] {
        let mut cache = filled();
        let mut attempts = 0;
        let result: Result<(), Glm5NextMetalError> =
            with_snapshot_release(&mut cache, None, || {
                attempts += 1;
                Err(admission(reason))
            });
        assert_eq!(attempts, 1);
        assert!(cache.entry_for_in(&ns, &[1]).is_some());
        assert!(cache.entry_for_in(&ns, &[2]).is_some());
        match result.unwrap_err() {
            Glm5NextMetalError::MemoryAdmission { denied, .. } => {
                assert_eq!(denied, refusal(reason));
            }
            error => panic!("unexpected refusal: {error}"),
        }
    }

    // A process-only refusal releases unpinned snapshots except `keep` and
    // retries exactly once.
    let mut cache = filled();
    let keep = cache.entry_for_in(&ns, &[1]);
    let mut attempts = 0;
    let result = with_snapshot_release(&mut cache, keep, || {
        attempts += 1;
        if attempts == 1 {
            Err(admission(R::ProcessInsufficient))
        } else {
            Ok(attempts)
        }
    });
    assert_eq!(result.unwrap(), 2);
    assert_eq!(attempts, 2);
    assert!(cache.entry_for_in(&ns, &[1]).is_some());
    assert!(cache.entry_for_in(&ns, &[2]).is_none());

    // Persistent process pressure retries once and returns the second typed
    // refusal.
    let mut cache = filled();
    let mut attempts = 0;
    let result: Result<(), Glm5NextMetalError> = with_snapshot_release(&mut cache, None, || {
        attempts += 1;
        Err(admission(R::ProcessInsufficient))
    });
    assert!(matches!(
        result.unwrap_err(),
        Glm5NextMetalError::MemoryAdmission { denied, .. }
            if denied.reason == R::ProcessInsufficient
    ));
    assert_eq!(attempts, 2);
    assert_eq!(cache.len(), 0);

    // An empty cache returns the original process refusal without retry.
    let mut attempts = 0;
    let result: Result<(), Glm5NextMetalError> = with_snapshot_release(&mut cache, None, || {
        attempts += 1;
        Err(admission(R::ProcessInsufficient))
    });
    assert!(matches!(
        result.unwrap_err(),
        Glm5NextMetalError::MemoryAdmission { denied, .. }
            if denied.reason == R::ProcessInsufficient
    ));
    assert_eq!(attempts, 1);

    // Invalid memory telemetry also preserves every cache entry.
    let mut cache = filled();
    let result: Result<(), Glm5NextMetalError> = with_snapshot_release(&mut cache, None, || {
        Err(admission(R::InvalidWorkingSetSignal))
    });
    assert!(matches!(
        result.unwrap_err(),
        Glm5NextMetalError::MemoryAdmission { denied, .. }
            if denied.reason == R::InvalidWorkingSetSignal
    ));
    assert_eq!(cache.len(), 2);
}

#[test]
fn release_keep_pin_is_cleared_when_eviction_panics() {
    use qwen_llm::snapshot_policy::{Clock, SnapshotPolicyConfig};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    struct PanicClock(Arc<AtomicBool>);

    impl Clock for PanicClock {
        fn now(&self) -> Duration {
            assert!(
                !self.0.load(Ordering::Relaxed),
                "injected eviction clock panic"
            );
            Duration::ZERO
        }
    }

    let panic_clock = Arc::new(AtomicBool::new(false));
    let namespace = CacheNamespace {
        lineage: PackedLineage::Exact,
        schedule: Schedule::ExactV1,
        rows: 512,
    };
    let mut cache: SnapshotCache<&'static str, CacheNamespace> = SnapshotCache::with_clock(
        1_000,
        SnapshotPolicyConfig::default(),
        Arc::new(PanicClock(Arc::clone(&panic_clock))),
    );
    for token in [1, 2, 3] {
        assert!(cache.insert_strict_in(namespace, vec![token], "snapshot", 10));
    }
    let keep = cache.entry_for_in(&namespace, &[1]);

    panic_clock.store(true, Ordering::Relaxed);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cache.evict_for_except(u64::MAX, keep);
    }));
    assert!(result.is_err());
    panic_clock.store(false, Ordering::Relaxed);

    assert_eq!(cache.pinned_bytes(), 0);
    assert_eq!(cache.len(), 3);
}

/// A request rendered as serve renders it, with its boundaries.
fn prepared_request(backend: &Glm5NextBackend<'_, '_>, body: Value) -> Arc<PreparedResponse> {
    let mut request = backend.parse_request(&body).unwrap();
    backend.normalize_request(&mut request).unwrap();
    let (prompt, boundaries) = backend.render_prepared(&request).unwrap();
    Arc::new(PreparedResponse {
        request,
        prompt,
        boundaries,
    })
}

/// Map #15 serve gates. A snapshot restore continues exactly the trajectory
/// a miss runs on the same schedule: Exact hits equal a cold single prefill
/// bitwise (Exact is segmentation-invariant); Fast hits equal a miss
/// that splits at the same shared-prefix cut, as does a full cache; Fast and
/// Exact entries never serve each other; a cancelled split prefill resumes
/// onto the same schedule.
#[test]
#[ignore = "GLM53_GGUF GPU serve snapshot gates under MTL_DEBUG_LAYER=1; production CLI lease, no server process"]
fn gpu_snapshots_continue_the_captured_trajectory() {
    use qwen_llm::snapshot_policy::SnapshotPolicyConfig;
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let path = gguf_path();
    let source = GgufFile::open(&path).unwrap();
    let prepared = Prepared::new(&source, &invocation(&path, Some(4096), Some(64))).unwrap();
    let ctx = MetalContext::new().unwrap();
    let weights = load(&ctx, &source, &prepared).unwrap();
    let plan = |mib| {
        crate::serve::SnapshotCachePlan::resolve(
            Some(mib),
            Default::default(),
            ctx.memory_signals(),
        )
        .unwrap()
    };
    let mut backend = Glm5NextBackend::new(
        &ctx,
        &weights,
        prepared,
        "glm".into(),
        std::time::Duration::ZERO,
        plan(8192),
    );
    // These existing trajectory gates explicitly exercise the 512-row contract.
    backend.frozen_rows = Some(512);
    backend.warm_up().unwrap();
    // ~1.3K tokens of instructions plus a tool: the shared cut is not a
    // multiple of the 512-row chunk, so a split moves chunk alignment.
    let instructions: String = (0..60)
        .map(|i| format!("Rule {i}: when asked about topic {i}, answer in one short sentence. "))
        .collect();
    let body_with = |instructions: &str, input: Value| {
        json!({"model":"glm","instructions":instructions,
            "tools":[{"type":"function","name":"get_weather","description":"Current weather for a city.",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],
            "input":input,
            "reasoning":{"effort":"low"},"temperature":0,"max_output_tokens":8})
    };
    let body =
        |question: &str| body_with(&instructions, json!([{"role":"user","content":question}]));
    let a = prepared_request(
        &backend,
        body("What is 6 times 7? Reply with the number only."),
    );
    let b = prepared_request(
        &backend,
        body("What is 5 times 9? Reply with the number only."),
    );
    let tokenizer = backend.prepared.artifact.tokenizer();
    let shared = tokenizer
        .encode(
            &a.prompt[..a.boundaries.unwrap().shared_prefix_end.unwrap()],
            false,
        )
        .unwrap()
        .len();
    let a_len = tokenizer.encode(&a.prompt, false).unwrap().len();
    assert!(shared > 1024 && shared % 512 != 0, "shared prefix {shared}");
    eprintln!("shared prefix {shared} tokens, request A {a_len} tokens");

    // Runs one request; returns (cached tokens, prefill logit bits, bytes).
    let run = |backend: &mut Glm5NextBackend<'_, '_>, prepared: &Arc<PreparedResponse>| {
        let mut sink = Sink::default();
        let outcome = backend
            .generate_prepared(Arc::clone(prepared), &mut sink)
            .unwrap();
        (
            outcome.usage.cached_tokens,
            backend.last_prefill_logits.clone(),
            sink.bytes,
        )
    };
    let reset = |backend: &mut Glm5NextBackend<'_, '_>, mib: u64, fast: bool, lineage| {
        backend.session = None;
        backend.history.clear();
        backend.deny_captures = false;
        backend.cache = SnapshotCache::new(mib << 20, SnapshotPolicyConfig::default());
        backend.fast_snapshots = fast;
        backend.lineage = lineage;
    };
    let mut failures = Vec::new();
    let mut check = |label: &str, ok: bool| {
        eprintln!("{label}: {ok}");
        if !ok {
            failures.push(label.to_string());
        }
    };

    // Exact references: one cold prefill each (no cache, so no cuts).
    reset(&mut backend, 0, false, PackedLineage::Exact);
    let (_, exact_a, exact_a_bytes) = run(&mut backend, &a);
    reset(&mut backend, 0, false, PackedLineage::Exact);
    let (_, exact_b, exact_b_bytes) = run(&mut backend, &b);
    // Exact with snapshots: A misses and captures at both cuts; B restores
    // the shared prefix; an identical retry of A restores its transcript.
    reset(&mut backend, 8192, false, PackedLineage::Exact);
    let (cached, logits, bytes) = run(&mut backend, &a);
    check("exact miss: no reuse", cached == 0);
    check(
        "exact miss == cold",
        logits == exact_a && bytes == exact_a_bytes,
    );
    check("exact miss captured both cuts", backend.cache.len() == 2);
    let (cached, logits, bytes) = run(&mut backend, &b);
    check("exact B restores the shared prefix", cached == shared);
    check(
        "exact B hit == cold",
        logits == exact_b && bytes == exact_b_bytes,
    );
    let (cached, logits, bytes) = run(&mut backend, &a);
    check("exact A retry restores its transcript", cached == a_len - 2);
    check(
        "exact A retry == cold",
        logits == exact_a && bytes == exact_a_bytes,
    );

    // Fast snapshots: the reference is a miss on the same schedule.
    reset(&mut backend, 8192, true, PackedLineage::Fast);
    let (cached, _, _) = run(&mut backend, &a);
    check("fast A miss: no reuse", cached == 0);
    reset(&mut backend, 8192, true, PackedLineage::Fast);
    let (_, fast_b_miss, fast_b_bytes) = run(&mut backend, &b);
    check(
        "fast B miss captured the shared cut",
        backend.cache.len() == 1,
    );
    backend.session = None;
    backend.history.clear();
    let (cached, logits, bytes) = run(&mut backend, &b);
    check("fast B restores the shared prefix", cached == shared);
    check(
        "fast B hit == fast B miss",
        logits == fast_b_miss && bytes == fast_b_bytes,
    );
    let (cached, logits, _) = run(&mut backend, &a);
    check("fast A hit (from B's capture)", cached == shared);
    let fast_a_hit = logits;
    // A full cache (1 MiB) captures nothing but splits the same way.
    reset(&mut backend, 1, true, PackedLineage::Fast);
    let (cached, logits, bytes) = run(&mut backend, &b);
    check(
        "fast full: no reuse, nothing cached",
        cached == 0 && backend.cache.len() == 0,
    );
    check(
        "fast full == fast miss",
        logits == fast_b_miss && bytes == fast_b_bytes,
    );
    // Fast with the lever off is a single cold prefill.
    reset(&mut backend, 8192, false, PackedLineage::Fast);
    let (_, unsplit, _) = run(&mut backend, &a);
    check(
        "fast with the lever off captures nothing",
        backend.cache.len() == 0,
    );
    eprintln!(
        "fast split vs unsplit A logits equal: {}",
        unsplit == fast_a_hit
    );

    // Lineages never share entries: Fast entries serve no Exact request.
    reset(&mut backend, 8192, true, PackedLineage::Fast);
    run(&mut backend, &a);
    backend.lineage = PackedLineage::Exact;
    let (cached, logits, _) = run(&mut backend, &b);
    check("exact B ignores fast entries", cached == 0);
    check("exact B after fast entries == cold", logits == exact_b);

    // A cancelled split prefill (tick 3: admission, chunk 1, chunk 2 of
    // the shared segment) resumes on the same schedule.
    for (lineage, fast) in [(PackedLineage::Exact, false), (PackedLineage::Fast, true)] {
        reset(&mut backend, 8192, fast, lineage);
        let reference = run(&mut backend, &b);
        reset(&mut backend, 8192, fast, lineage);
        let mut cancelled = Sink {
            abort_tick: Some(3),
            ..Sink::default()
        };
        let aborted = backend.generate_prepared(Arc::clone(&b), &mut cancelled);
        check(
            &format!("{lineage:?} cancel keeps 512 committed"),
            matches!(aborted, Err(BackendFailure::Aborted(_))) && backend.history.len() == 512,
        );
        let (cached, logits, bytes) = run(&mut backend, &b);
        check(
            &format!("{lineage:?} resume == uninterrupted"),
            cached == 512 && logits == reference.1 && bytes == reference.2,
        );
    }

    // Fast snapshots are taken and used only at this request's own
    // verified cut. Two instruction sets where the first's shared prefix is
    // a strict token prefix of the second's.
    let short = format!("{}\n", instructions.trim_end());
    let long = format!("{short}Extra rule: answer politely.\n");
    let question =
        json!([{"role":"user","content":"What is 6 times 7? Reply with the number only."}]);
    let first = prepared_request(&backend, body_with(&short, question.clone()));
    let second = prepared_request(&backend, body_with(&long, question));
    // The GGUF tokenizer, independent of the backend's borrow.
    let gguf_tokenizer = qwen_llm::tokenizer::Tokenizer::from_gguf(&source).unwrap();
    let encode = |text: &str| -> Vec<u32> {
        gguf_tokenizer
            .encode(text, false)
            .unwrap()
            .into_iter()
            .map(|id| id as u32)
            .collect()
    };
    assert_eq!(encode(&a.prompt).len(), a_len);
    let first_tokens = encode(&first.prompt);
    let second_tokens = encode(&second.prompt);
    let s1 = encode(&first.prompt[..first.boundaries.unwrap().shared_prefix_end.unwrap()]).len();
    let s2 = encode(&second.prompt[..second.boundaries.unwrap().shared_prefix_end.unwrap()]).len();
    assert!(
        s1 < s2 && second_tokens.starts_with(&first_tokens[..s1]),
        "fixture: the first shared prefix ({s1}) must be a token prefix of the second ({s2})"
    );
    let fast_ns = CacheNamespace {
        lineage: PackedLineage::Fast,
        schedule: Schedule::FastSharedSplitV1,
        rows: 512,
    };
    reset(&mut backend, 8192, true, PackedLineage::Fast);
    let (_, second_miss, second_bytes) = run(&mut backend, &second);
    // A cached shorter shared prefix is not this request's cut: a miss.
    reset(&mut backend, 8192, true, PackedLineage::Fast);
    run(&mut backend, &first);
    backend.session = None;
    backend.history.clear();
    let (cached, logits, bytes) = run(&mut backend, &second);
    check("fast ignores another request's shorter cut", cached == 0);
    check(
        "fast after a shorter cut == miss",
        logits == second_miss && bytes == second_bytes,
    );
    // A live session cancelled right after the first request's cut (tick 5:
    // admission, three chunks, then the next segment) continues into the
    // second request, but its state is off the chunk grid: no Fast capture.
    reset(&mut backend, 8192, true, PackedLineage::Fast);
    let mut cancelled = Sink {
        abort_tick: Some(5),
        ..Sink::default()
    };
    let aborted = backend.generate_prepared(Arc::clone(&first), &mut cancelled);
    check(
        "fast cancel right after the cut",
        matches!(aborted, Err(BackendFailure::Aborted(_))) && backend.history.len() == s1,
    );
    let (cached, _, _) = run(&mut backend, &second);
    check(
        "off-grid live state is continued but not published",
        cached == s1
            && backend
                .cache
                .entry_for_in(&fast_ns, &second_tokens[..s2])
                .is_none(),
    );
    // Without renderer boundaries a Fast request neither restores nor
    // captures.
    reset(&mut backend, 8192, true, PackedLineage::Fast);
    run(&mut backend, &a);
    backend.session = None;
    backend.history.clear();
    let entries = backend.cache.len();
    let mut sink = Sink::default();
    let outcome = backend.generate(&a.request, &a.prompt, &mut sink).unwrap();
    check(
        "fast without boundaries: no restore, no capture",
        outcome.usage.cached_tokens == 0 && backend.cache.len() == entries,
    );

    // Denied captures keep the same splits.
    for (lineage, fast, reference) in [
        (PackedLineage::Fast, true, &fast_b_miss),
        (PackedLineage::Exact, false, &exact_b),
    ] {
        reset(&mut backend, 8192, fast, lineage);
        backend.deny_captures = true;
        let (cached, logits, _) = run(&mut backend, &b);
        check(
            &format!("{lineage:?} denied: same splits, nothing cached"),
            cached == 0 && backend.cache.len() == 0 && logits == *reference,
        );
    }

    // A continued live request keeps the shared entry it passed: with room
    // for one entry beside a pinned one, its transcript capture is refused
    // rather than evicting the shared snapshot.
    reset(&mut backend, 8192, false, PackedLineage::Exact);
    let mut sink = Sink::default();
    let outcome = backend
        .generate_prepared(Arc::clone(&a), &mut sink)
        .unwrap();
    let (reasoning, answer) = split(&sink.bytes, outcome.end);
    let continued = prepared_request(
        &backend,
        body_with(
            &instructions,
            json!([
                {"role":"user","content":"What is 6 times 7? Reply with the number only."},
                {"type":"reasoning","content":reasoning},
                {"role":"assistant","content":answer},
                {"role":"user","content":"And 5 times 9?"}]),
        ),
    );
    let entry = |n: usize| {
        qwen_llm::glm5_next_metal::snapshot_bytes(&weights.config, n as u64).unwrap() + 4 * n as u64
    };
    let continued_len = encode(&continued.prompt).len();
    let budget = entry(continued_len - 2) + entry(shared) / 2;
    backend.session = None;
    backend.history.clear();
    backend.cache = SnapshotCache::new(budget, SnapshotPolicyConfig::default());
    run(&mut backend, &a);
    let exact_ns = CacheNamespace {
        lineage: PackedLineage::Exact,
        schedule: Schedule::ExactV1,
        rows: 512,
    };
    let a_tokens = encode(&a.prompt);
    check(
        "tight budget keeps the shared entry, refuses the transcript",
        backend.cache.len() == 1
            && backend
                .cache
                .entry_for_in(&exact_ns, &a_tokens[..shared])
                .is_some(),
    );
    let (cached, _, _) = run(&mut backend, &continued);
    check(
        "continued request reuses the live session past both cuts",
        cached > a_len,
    );
    check(
        "the passed shared entry survives the continued capture",
        backend
            .cache
            .entry_for_in(&exact_ns, &a_tokens[..shared])
            .is_some(),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
