use super::*;
use qwen_llm::metal::{MetalMemorySignals, evaluate_metal_memory_admission_with_cpu_bytes};
use qwen_llm::tokenizer::TokError;

struct Bytes;
impl Tokenize for Bytes {
    fn encode(&self, text: &str, _: bool) -> Result<Vec<i32>, TokError> {
        Ok(text.bytes().map(i32::from).collect())
    }
    fn try_decode_piece_bytes(&self, token: i32) -> Result<Vec<u8>, TokError> {
        if token == 255 {
            return Err(TokError::InvalidToken(token));
        }
        Ok(vec![token as u8])
    }
    fn try_decode_piece(&self, token: i32) -> Result<String, TokError> {
        Ok(String::from_utf8(self.try_decode_piece_bytes(token)?).unwrap())
    }
    fn try_decode(&self, tokens: &[i32]) -> Result<String, TokError> {
        tokens.iter().map(|&id| self.try_decode_piece(id)).collect()
    }
    fn decode(&self, tokens: &[i32]) -> String {
        self.try_decode(tokens).unwrap()
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
struct Sink {
    bytes: Vec<u8>,
    events: Vec<&'static str>,
    ticks: usize,
    pieces: usize,
    abort_tick: Option<usize>,
    abort_piece: Option<usize>,
}
impl GenerationSink for Sink {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.events.push("piece");
        self.pieces += 1;
        if self.abort_piece == Some(self.pieces) {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "piece cancelled"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn tick(&mut self) -> io::Result<()> {
        self.events.push("tick");
        self.ticks += 1;
        if self.abort_tick == Some(self.ticks) {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionReset,
                "tick cancelled",
            ));
        }
        Ok(())
    }
}

fn packet(tokens: &[i32], finish: Option<Qwen4ExpDFlashFinish>) -> Qwen4ExpDFlashPacket {
    Qwen4ExpDFlashPacket {
        tokens: tokens.to_vec(),
        finish,
    }
}

#[test]
fn committed_packets_count_terminal_without_rendering_it_and_keep_t_minus_one() {
    let mut sink = Sink::default();
    let mut output = PacketOutput::new(8, Duration::from_millis(3));
    output
        .emit(&packet(&[97], None), &[0], &Bytes, &mut sink)
        .unwrap();
    output.record_step(Duration::from_millis(11));
    output
        .emit(
            &packet(&[98, 99, 0], Some(Qwen4ExpDFlashFinish::Eos)),
            &[0],
            &Bytes,
            &mut sink,
        )
        .unwrap();
    let generation = output.finish(3).unwrap();
    assert_eq!(generation.tokens, [97, 98, 99, 0]);
    assert_eq!(generation.transitions, 3);
    assert_eq!(generation.first_token_selection_ms, 3.0);
    assert_eq!(generation.first_transition_ms, Some(11.0));
    assert_eq!(generation.transition_ms, 11.0);
    assert_eq!(sink.bytes, b"abc");
    assert_eq!(
        sink.events,
        [
            "tick", "piece", "tick", "tick", "piece", "tick", "piece", "tick"
        ]
    );
    let outcome = super::super::super::outcome::finish_generation(9, &generation, 0, 0.0);
    assert_eq!(outcome.usage.input_tokens, 9);
    assert_eq!(outcome.usage.output_tokens, 4);
    assert_eq!(outcome.usage.cached_tokens, 0);
}

#[test]
fn initial_eos_and_ragged_cap_never_require_an_extra_transition() {
    let mut sink = Sink::default();
    let mut output = PacketOutput::new(1, Duration::ZERO);
    output
        .emit(
            &packet(&[0], Some(Qwen4ExpDFlashFinish::Eos)),
            &[0],
            &Bytes,
            &mut sink,
        )
        .unwrap();
    let result = output.finish(0).unwrap();
    assert_eq!(result.stop_reason, crate::StopReason::Eos);
    assert!(result.first_token_callback_ms.is_none());
    assert!(sink.bytes.is_empty());
    let mut output = PacketOutput::new(3, Duration::ZERO);
    output
        .emit(&packet(&[97], None), &[0], &Bytes, &mut sink)
        .unwrap();
    output
        .emit(
            &packet(&[98, 99], Some(Qwen4ExpDFlashFinish::TokenLimit)),
            &[0],
            &Bytes,
            &mut sink,
        )
        .unwrap();
    assert_eq!(
        output.finish(2).unwrap().stop_reason,
        crate::StopReason::TokenLimit
    );
}

#[test]
fn invalid_packet_is_refused_before_any_output() {
    for bad in [
        packet(&[], None),
        packet(&[97, 0, 98], Some(Qwen4ExpDFlashFinish::Eos)),
        packet(&[97, 98], Some(Qwen4ExpDFlashFinish::TokenLimit)),
        packet(&[97, -1], None),
        packet(&[256], None),
        packet(&[97], Some(Qwen4ExpDFlashFinish::Cancelled)),
        packet(&[97, 98, 99], None),
        packet(&[97, 98, 99, 100], None),
    ] {
        let mut output = PacketOutput::new(3, Duration::ZERO);
        let mut sink = Sink::default();
        assert!(matches!(
            output.emit(&bad, &[0], &Bytes, &mut sink),
            Err(BackendFailure::Serve(_))
        ));
        assert!(sink.events.is_empty());
    }
}

#[test]
fn abort_before_between_and_inside_packets_preserves_io_type_and_returns_workspace() {
    for (abort_tick, abort_piece, expected) in
        [(Some(1), None, 0), (Some(3), None, 1), (None, Some(3), 2)]
    {
        let mut output = PacketOutput::new(4, Duration::ZERO);
        let mut sink = Sink {
            abort_tick,
            abort_piece,
            ..Default::default()
        };
        let result = output
            .emit(&packet(&[97], None), &[0], &Bytes, &mut sink)
            .and_then(|()| {
                output.emit(
                    &packet(&[98, 99, 100], Some(Qwen4ExpDFlashFinish::TokenLimit)),
                    &[0],
                    &Bytes,
                    &mut sink,
                )
            });
        let mut returned = 0;
        let result = super::super::with_request_workspace(
            result,
            7,
            |workspace| {
                assert_eq!(workspace, 7);
                returned += 1;
                Ok::<_, ()>(())
            },
            |_| panic!("return failed"),
        );
        assert_eq!(returned, 1);
        assert_eq!(sink.bytes.len(), expected);
        assert!(matches!(result, Err(BackendFailure::Aborted(error))
            if error.kind() == if abort_piece.is_some() { io::ErrorKind::BrokenPipe } else { io::ErrorKind::ConnectionReset }));
    }
}

#[test]
fn tokenizer_failure_is_server_error_not_cancellation() {
    let mut output = PacketOutput::new(1, Duration::ZERO);
    let mut sink = Sink::default();
    assert!(matches!(
        output.emit(
            &packet(&[255], Some(Qwen4ExpDFlashFinish::TokenLimit)),
            &[0],
            &Bytes,
            &mut sink
        ),
        Err(BackendFailure::Serve(_))
    ));
    assert!(sink.bytes.is_empty());
}

#[test]
fn prefill_cancellation_keeps_original_io_error() {
    let error = Qwen4ExpDFlashError::Target(Qwen4ExpRuntimeError::Checkpoint("cancelled".into()));
    let failure = map_start_error(
        error,
        Some(io::Error::new(io::ErrorKind::BrokenPipe, "original")),
    );
    assert!(
        matches!(failure, BackendFailure::Aborted(error) if error.kind() == io::ErrorKind::BrokenPipe && error.to_string() == "original")
    );
}

#[test]
fn aggregate_pressure_counts_cpu_reserve_and_releases_only_process_pressure() {
    use std::cell::Cell;
    let remaining = Cell::new(100);
    let signals = || MetalMemorySignals {
        current_allocated_bytes: 0,
        recommended_max_bytes: 1_000,
        process_limit_remaining_bytes: Some(remaining.get()),
        host_budget: Some(qwen_llm::metal::HostMemoryBudget {
            physical_bytes: 1024 << 30,
            conservative_available_bytes: 1024 << 30,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
    };
    let attempt = || {
        evaluate_metal_memory_admission_with_cpu_bytes(80, 30, 0, signals(), true)
            .refusal()
            .map_or(Ok(()), |denied| {
                Err(Qwen4ExpDFlashError::Target(
                    Qwen4ExpRuntimeError::MemoryAdmission(denied),
                ))
            })
    };
    assert!(
        evaluate_metal_memory_admission_with_cpu_bytes(80, 0, 0, signals(), true).is_admitted()
    );
    assert!(
        matches!(map_error(attempt().unwrap_err()), BackendFailure::Serve(error) if error.status == 503)
    );
    let result = super::super::super::memory_relief::retry(
        FAMILY,
        attempt(),
        relief_request,
        |deficit| {
            assert_eq!(deficit, 10);
            remaining.set(110);
            super::super::super::memory_relief::test_evicted(10)
        },
        attempt,
    );
    assert!(result.is_ok());
    for reason in [
        MetalMemoryAdmissionReason::WorkingSetInsufficient,
        MetalMemoryAdmissionReason::BothInsufficient,
        MetalMemoryAdmissionReason::ProcessSignalUnavailable,
        MetalMemoryAdmissionReason::RequiredBytesOverflow,
    ] {
        let denied = MemoryAdmissionDenied::new(reason, Some(110), signals(), Some(100)).unwrap();
        let first = Err(Qwen4ExpDFlashError::Target(
            Qwen4ExpRuntimeError::MemoryAdmission(denied),
        ));
        let result = super::super::super::memory_relief::retry(
            FAMILY,
            first,
            relief_request,
            |_| panic!("non-process-only refusal must not evict"),
            || panic!("must not retry"),
        );
        let expected = if reason.is_pressure() { 503 } else { 500 };
        assert!(
            matches!(map_error(result.unwrap_err()), BackendFailure::Serve(error) if error.status == expected)
        );
    }
}

#[test]
#[ignore = "exclusive GPU lease; local QWEN4EXP_Q3_K_XL_RUNTIME_GGUF and QWEN4EXP_DFLASH_GGUF"]
fn real_cold_backend_packets_cancel_and_reuse_workspace() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let path = qwen_llm::test_fixtures::QWEN4EXP_Q3_K_XL.required();
    let gguf = Box::leak(Box::new(GgufFile::open(&path).unwrap()));
    let invocation = crate::cli::ServeInvocation {
        lens_data_dir: None,
        lens_config: None,
        web_root: None,
        lens_allowed_origin: Vec::new(),
        model: path.clone(),
        addr: "127.0.0.1:0".into(),
        max_tokens: Some(16),
        max_context_tokens: Some(256),
        snapshot_cache_mib: Some(0),
        snapshot_policy: Default::default(),
        durable: crate::serve::durable::DurableSnapshotConfig::off(),
        drafter: Some(std::env::var("QWEN4EXP_DFLASH_GGUF").unwrap().into()),
        trace_sse: None,
        template_style: Default::default(),
        idle_residency_secs: None,
    };
    let prepared = Prepared::new(gguf, &invocation).unwrap();
    let ctx = MetalContext::new().unwrap();
    let mut backend = FlashNextBackend::new(
        ctx,
        gguf,
        "flash-next-test".into(),
        prepared,
        Some(0),
        Default::default(),
    )
    .unwrap();
    assert!(backend.dflash_head.is_some());
    backend.stop_tokens.clear();
    for (temperature, abort_tick, abort_piece) in [
        (0.0, None, None),
        (1.0, Some(1), None),
        (1.0, Some(3), None),
        (0.0, None, Some(2)),
        (1.0, None, None),
    ] {
        let body = serde_json::json!({"model":"flash-next-test", "input":"Count to ten.", "temperature":temperature,
            "seed":42, "max_output_tokens":16});
        let mut request = backend.parse_request(&body).unwrap();
        backend.normalize_request(&mut request).unwrap();
        let prompt = backend.render_prompt(&request).unwrap();
        let mut sink = Sink {
            abort_tick,
            abort_piece,
            ..Default::default()
        };
        let result = backend.generate(&request, &prompt, &mut sink);
        if abort_tick.is_some() || abort_piece.is_some() {
            assert!(matches!(result, Err(BackendFailure::Aborted(_))));
        } else {
            let outcome = result.unwrap();
            assert_eq!(outcome.usage.output_tokens, 16);
            assert_eq!(outcome.usage.cached_tokens, 0);
            assert_eq!(outcome.stats.unwrap().restore_ms, 0.0);
        }
        assert_eq!(backend.cache.len(), 0);
        let runner = backend.loaded.create_runner(&backend.ctx).unwrap();
        assert_eq!(runner.next_position(), 0);
        assert!(!runner.is_poisoned());
        let workspace = runner.into_workspace();
        backend.loaded.restore_workspace(workspace).unwrap();
    }
}
