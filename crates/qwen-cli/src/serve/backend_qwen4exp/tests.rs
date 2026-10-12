use super::*;
use qwen_llm::metal::MetalContext;
use qwen_llm::qwen4exp::Qwen4ExpConfig;
use serde_json::json;
use std::io;

fn gpu_backend(snapshot_cache_mib: Option<u64>) -> FlashNextBackend {
    let path = qwen_llm::test_fixtures::QWEN4EXP_Q3_K_XL.required();
    let gguf = Box::leak(Box::new(GgufFile::open(&path).unwrap()));
    let invocation = crate::cli::ServeInvocation {
        lens_data_dir: None,
        lens_config: None,
        web_root: None,
        lens_allowed_origin: Vec::new(),
        model: path.clone(),
        addr: "127.0.0.1:0".into(),
        max_tokens: Some(8),
        max_context_tokens: Some(256),
        snapshot_cache_mib,
        snapshot_policy: Default::default(),
        durable: crate::serve::durable::DurableSnapshotConfig::off(),
        drafter: None,
        trace_sse: None,
        template_style: Default::default(),
        idle_residency_secs: None,
    };
    let prepared = Prepared::new(gguf, &invocation).unwrap();
    let snapshot_cache_mib =
        snapshot_cache_mib.unwrap_or_else(|| snapshot_cache_mib_for_test(gguf, &prepared));
    FlashNextBackend::new(
        MetalContext::new().unwrap(),
        gguf,
        "flash-next-tests".into(),
        prepared,
        Some(snapshot_cache_mib),
        Default::default(),
    )
    .unwrap()
}

fn snapshot_cache_mib_for_test(gguf: &GgufFile, prepared: &Prepared) -> u64 {
    const MIB: u64 = 1 << 20;
    let config = Qwen4ExpConfig::from_gguf(gguf).unwrap();
    let geometry = qwen_llm::qwen4exp_text_session::Qwen4ExpTextSessionMetalGeometry::from_config(
        &config,
        prepared.capacity.qsa_physical_capacity(),
    )
    .unwrap();
    let max_prefix = prepared.capacity.forward_limit().saturating_sub(1);
    let payload_bytes = geometry.snapshot_bytes(max_prefix).unwrap();
    let entry_bytes = SnapshotCache::<()>::entry_bytes(max_prefix, payload_bytes).unwrap();
    entry_bytes
        .checked_mul(2)
        .expect("snapshot cache margin overflow")
        .div_ceil(MIB)
}

#[derive(Default)]
struct Sink {
    bytes: Vec<u8>,
    ticks: usize,
    pieces: usize,
    abort_tick: Option<usize>,
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

fn build_request(backend: &FlashNextBackend, input: &str) -> (ServeRequest, String) {
    let body = json!({
        "model":"flash-next-tests",
        "input":[{"role":"user","content":input}],
        "temperature":0,
        "max_output_tokens":8,
        "seed":42
    });
    let mut request = backend.parse_request(&body).unwrap();
    backend.normalize_request(&mut request).unwrap();
    let prompt = backend.render_prompt(&request).unwrap();
    (request, prompt)
}

fn execute(
    backend: &mut FlashNextBackend,
    request: &ServeRequest,
    prompt: &str,
    sink: &mut Sink,
) -> Result<GenerationOutcome, BackendFailure> {
    backend.generate(request, prompt, sink)
}

#[test]
fn startup_limits_use_flash_next_profile_and_require_bounded_capacity() {
    let config = Qwen4ExpConfig::flash_next_reference();
    let profile = crate::serve::profile(qwen_llm::model_family::ModelFamily::Qwen4Exp);
    for (context, output, expected) in [
        (None, Some(8), "requires --max-context-tokens"),
        (Some(64), None, "requires explicit --max-tokens"),
        (Some(0), Some(1), "must be greater than 0"),
        (Some(64), Some(65), "exceeds --max-context-tokens"),
    ] {
        let error = crate::serve::resolve_serve_limits(
            profile,
            Some(config.context_length as usize),
            context,
            output,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains(expected), "{error}");
    }

    let limits = crate::serve::resolve_serve_limits(
        profile,
        Some(config.context_length as usize),
        Some(256),
        Some(64),
    )
    .unwrap()
    .unwrap();
    assert_eq!((limits.context_tokens, limits.max_tokens), (256, 64));
}

#[test]
fn session_capacity_bounds_forward_limit() {
    let config = Qwen4ExpConfig::flash_next_reference();
    for forward_limit in [1, 64, 255, 256, 4096] {
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, forward_limit).unwrap();
        assert_eq!(capacity.forward_limit(), forward_limit);
        assert!(capacity.qsa_physical_capacity() >= forward_limit);
        assert!(capacity.qsa_physical_capacity() <= config.context_length as usize);
    }
    assert!(Qwen4ExpSessionCapacity::for_forward_limit(&config, 0).is_err());
    assert!(
        Qwen4ExpSessionCapacity::for_forward_limit(&config, config.context_length as usize + 1)
            .is_err()
    );
}

#[test]
fn invalid_sampling_is_refused_before_model_work() {
    let request = crate::serve::items::parse_request(&serde_json::json!({
        "model":"flash-next", "input":"hello", "temperature":-1.0
    }));
    assert!(request.is_err());
    assert!(
        request
            .unwrap_err()
            .message
            .contains("temperature must be >= 0")
    );
}

#[test]
fn transcript_plan_splits_only_when_boundary_is_ahead_of_cache() {
    assert_eq!(transcript_split(None, 4), None);
    assert_eq!(transcript_split(Some(8), 4), Some(8));
    assert_eq!(transcript_split(Some(4), 4), None);
    assert_eq!(transcript_split(Some(2), 4), None);
}

#[test]
fn prefill_failure_maps_checkpoint_abort_separately_from_runtime_error() {
    let abort = map_prefill_result::<()>(
        Err(Qwen4ExpRuntimeError::Invalid("prefill failed".into())),
        Some(io::Error::new(io::ErrorKind::BrokenPipe, "client gone")),
    );
    assert!(matches!(abort, Err(BackendFailure::Aborted(_))));

    let failure = map_prefill_result::<()>(
        Err(Qwen4ExpRuntimeError::Invalid("prefill failed".into())),
        None,
    );
    assert!(matches!(failure, Err(BackendFailure::Serve(error)) if error.status == 500));
}

#[test]
fn failed_snapshot_restore_evicts_the_entry_and_resets_the_runner() {
    let mut cache = SnapshotCache::new(1024, Default::default());
    assert!(cache.insert_strict(vec![1, 2], "snapshot", 16));
    let mut runner = 99;
    let mut reset = false;
    let matched = restore_best_prefix(
        &mut cache,
        &[1, 2, 3],
        &mut runner,
        |_, _| Err("corrupt snapshot"),
        |runner| {
            *runner = 0;
            reset = true;
            Ok::<_, &str>(())
        },
    )
    .unwrap();
    assert_eq!(matched, 0);
    assert!(reset);
    assert_eq!(runner, 0);
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.indexed_bytes(), 0);
}

#[test]
fn flash_next_workspace_cleanup_runs_after_transport_abort() {
    let mut reset = false;
    let outcome: Result<(), BackendFailure> = Err(BackendFailure::Aborted(io::Error::new(
        io::ErrorKind::BrokenPipe,
        "client disconnected",
    )));
    let result = with_request_workspace(
        outcome,
        "workspace",
        |workspace| {
            assert_eq!(workspace, "workspace");
            reset = true;
            Ok::<(), &str>(())
        },
        |error| panic!("unexpected workspace reset failure: {error}"),
    );
    assert!(reset);
    assert!(matches!(result, Err(BackendFailure::Aborted(_))));
}

#[test]
fn flash_next_workspace_reset_failure_reaches_fail_stop_handler() {
    let mut failure = None;
    let result = with_request_workspace(
        Ok::<_, &str>("streamed success"),
        "workspace",
        |_| Err("reset failed"),
        |error| failure = Some(error),
    );
    assert_eq!(result, Ok("streamed success"));
    assert_eq!(failure, Some("reset failed"));
}

#[test]
#[ignore = "GPU-only QWEN4EXP_Q3_K_XL resident/cold parity and abort retry coverage"]
fn gpu_qwen4exp_resident_abort_retry_and_transcript_cache_parity() {
    let mut cold_backend = gpu_backend(Some(0));
    let (cold_request, cold_prompt) =
        build_request(&cold_backend, "What is 2 + 2? Answer briefly.");
    let mut cold_sink = Sink::default();
    let cold_outcome = execute(
        &mut cold_backend,
        &cold_request,
        &cold_prompt,
        &mut cold_sink,
    )
    .unwrap();
    assert_eq!(cold_outcome.usage.cached_tokens, 0);
    let cold_bytes = cold_sink.bytes;
    drop(cold_backend);

    let mut backend = gpu_backend(None);
    let (request, prompt) = build_request(&backend, "What is 2 + 2? Answer briefly.");
    let mut warm = Sink::default();
    execute(&mut backend, &request, &prompt, &mut warm).unwrap();
    assert_eq!(warm.bytes, cold_bytes);

    let token_ids = decode_loop::encode_checked(
        &backend.tokenizer,
        &prompt,
        false,
        backend.vocab_size,
        FAMILY,
    )
    .unwrap();
    let boundary = backend
        .im_start
        .and_then(|im_start| transcript_boundary(&token_ids, im_start))
        .expect("rendered Flash-Next prompt has a transcript boundary");
    assert!(backend.cache.entry_for(&token_ids[..boundary]).is_some());
    let mut resent = Sink::default();
    let resent_outcome = execute(&mut backend, &request, &prompt, &mut resent).unwrap();
    assert_eq!(resent_outcome.usage.cached_tokens, boundary);
    assert_eq!(resent.bytes, cold_bytes);

    let late_abort_tick = warm.ticks.saturating_sub(1).max(2);
    for mut aborted in [
        Sink {
            abort_tick: Some(1),
            ..Sink::default()
        },
        Sink {
            abort_piece: Some(1),
            ..Sink::default()
        },
        Sink {
            abort_tick: Some(late_abort_tick),
            ..Sink::default()
        },
    ] {
        assert!(matches!(
            execute(&mut backend, &request, &prompt, &mut aborted),
            Err(BackendFailure::Aborted(_))
        ));
        let mut retry = Sink::default();
        execute(&mut backend, &request, &prompt, &mut retry).unwrap();
        assert_eq!(retry.bytes, cold_bytes);
    }

    let over_capacity = "overflow ".repeat(512);
    let (request, prompt) = build_request(&backend, &over_capacity);
    let mut refused = Sink::default();
    assert!(execute(&mut backend, &request, &prompt, &mut refused).is_err());
    assert_eq!(refused.ticks, 0);
}

#[test]
#[ignore = "GPU-only QWEN4EXP_Q3_K_XL transcript snapshot restore parity"]
fn gpu_qwen4exp_transcript_restore_matches_cold_run() {
    let question = "Summarize why cached prefixes are safe. Keep it short.";
    let mut resident = gpu_backend(None);
    let (request, prompt) = build_request(&resident, question);
    let token_ids = decode_loop::encode_checked(
        &resident.tokenizer,
        &prompt,
        false,
        resident.vocab_size,
        FAMILY,
    )
    .unwrap();
    let boundary = resident
        .im_start
        .and_then(|im_start| transcript_boundary(&token_ids, im_start))
        .expect("rendered Flash-Next prompt has a transcript boundary");
    let mut first = Sink::default();
    execute(&mut resident, &request, &prompt, &mut first).unwrap();
    assert!(resident.cache.entry_for(&token_ids[..boundary]).is_some());
    let mut restored = Sink::default();
    let outcome = execute(&mut resident, &request, &prompt, &mut restored).unwrap();
    assert_eq!(outcome.usage.cached_tokens, boundary);

    let mut cold = gpu_backend(Some(0));
    let (cold_request, cold_prompt) = build_request(&cold, question);
    let mut cold_sink = Sink::default();
    execute(&mut cold, &cold_request, &cold_prompt, &mut cold_sink).unwrap();
    assert_eq!(restored.bytes, cold_sink.bytes);
}
