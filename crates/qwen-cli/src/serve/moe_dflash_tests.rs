//! Opt-in Metal regression; register as a child of `serve::backend`.
//! Requires local QWEN_MOE_DFLASH_TARGET and QWEN_MOE_DFLASH_DRAFTER GGUFs.
//! Run this test alone: the 35B-A3B target and BF16 DFlash2 drafter need
//! roughly 23 GiB for weights, plus request scratch and a 1 GiB snapshot cache.

use super::*;
use crate::serve::output_partition::OutputPartition;
use crate::serve::partition::PartitionEvent;
use serde_json::{Value, json};
use std::path::PathBuf;

const MODEL_ID: &str = "moe-dflash-regression";
const CONTEXT: usize = 1024;
const OUTPUT_TOKENS: usize = 128;
const CACHE_BYTES: u64 = 1024 * 1024 * 1024;
const CODE_PROMPT: &str = "Write a Python function merge_sorted(a, b) that merges two sorted lists in linear time. Return only code.";

#[derive(Default)]
struct Sink {
    bytes: Vec<u8>,
    pieces: usize,
    abort_after: Option<usize>,
}

impl GenerationSink for Sink {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        if self.abort_after == Some(self.pieces) {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "MoE DFlash regression disconnect",
            ));
        }
        self.pieces += 1;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn tick(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn required_model_path(name: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var_os(name).unwrap_or_else(|| {
        panic!("set {name} to a local GGUF before explicitly running this ignored Metal test")
    }));
    assert!(
        path.is_file(),
        "{name} is not a local file: {}",
        path.display()
    );
    path
}

fn request(backend: &EngineBackend, input: Value) -> ServeRequest {
    let mut request = backend
        .parse_request(&json!({
            "model": MODEL_ID,
            "input": input,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": OUTPUT_TOKENS,
            "x_qwen": {"no_thinking": true, "stats": true, "seed": 42},
        }))
        .expect("parse real serving request");
    backend
        .normalize_request(&mut request)
        .expect("normalize real serving request");
    assert_eq!(request_sampler(&request).unwrap().config().temperature, 0.0);
    request
}

fn run(
    backend: &mut EngineBackend,
    request: &ServeRequest,
    stage: &str,
) -> (Vec<u8>, GenerationOutcome, String) {
    let prompt = backend
        .render_prompt(request)
        .expect("render serving prompt");
    let prompt_tokens = backend.tokenizer.encode(&prompt, false).unwrap().len();
    assert!(
        prompt_tokens + OUTPUT_TOKENS <= CONTEXT,
        "{stage}: context bound"
    );
    let mut sink = Sink::default();
    let outcome = backend
        .generate(request, &prompt, &mut sink)
        .unwrap_or_else(|error| panic!("{stage}: serving generation failed: {error:?}"));
    assert_eq!(outcome.usage.input_tokens, prompt_tokens, "{stage}");
    assert!(
        (1..=OUTPUT_TOKENS).contains(&outcome.usage.output_tokens),
        "{stage}: {outcome:?}"
    );
    assert!(outcome.usage.cached_tokens <= prompt_tokens, "{stage}");
    let stats = outcome.stats.as_ref().expect("serving outcome stats");
    assert_eq!(stats.prompt_tokens, prompt_tokens, "{stage}");
    assert_eq!(stats.matched_tokens, outcome.usage.cached_tokens, "{stage}");
    assert!(
        stats.restore_ms.is_finite() && stats.restore_ms >= 0.0,
        "{stage}"
    );

    let mut partition = OutputPartition::new(backend.output_protocol(request).unwrap());
    let mut events = Vec::new();
    partition.push(&sink.bytes, &mut events);
    partition
        .finish(outcome.end, &mut events)
        .expect("finish serving output protocol");
    let visible: String = events
        .into_iter()
        .filter_map(|event| match event {
            PartitionEvent::Visible(text) => Some(text),
            _ => None,
        })
        .collect();
    assert!(!visible.trim().is_empty(), "{stage}: no visible answer");
    assert!(
        !visible.contains('\u{fffd}'),
        "{stage}: invalid UTF-8 output"
    );
    eprintln!("{stage}: {outcome:?}");
    (sink.bytes, outcome, visible)
}

fn assert_speculation(outcome: &GenerationOutcome, stage: &str) {
    let phases = outcome
        .stats
        .as_ref()
        .and_then(|stats| stats.phases.as_ref())
        .unwrap_or_else(|| {
            panic!("{stage}: expose DFlash spec_steps and drafter_calls in ServeStats.phases; logs alone cannot rule out serial fallback")
        });
    let dflash = phases.get("dflash").unwrap_or(phases);
    for counter in ["spec_steps", "drafter_calls"] {
        assert!(
            dflash
                .get(counter)
                .and_then(Value::as_u64)
                .is_some_and(|n| n > 0),
            "{stage}: expected nonzero {counter}, not silent serial fallback: {phases}"
        );
    }
    // Rejection and backoff depend on model numerics; report them, but do
    // not require this fixture to reject an arbitrary number of drafts.
    eprintln!("{stage}: DFlash counters {dflash}");
}

fn assert_cached_capture(backend: &EngineBackend, request: &ServeRequest) -> usize {
    let prompt = backend.render_prompt(request).unwrap();
    let ids = backend.tokenizer.encode(&prompt, false).unwrap();
    let lookup = backend
        .loaded
        .lookup_cached_prefix(&ids)
        .expect("served request must retain a reusable checkpoint");
    let matched = lookup.matched_prefix_len();
    assert!(lookup.restored_prefix_len() > 0);
    let mut sequence = backend
        .loaded
        .create_sequence(SequenceConfig::new(CONTEXT))
        .expect("bounded capture witness sequence");
    let restored = backend
        .loaded
        .restore_prepared_cached_prefix(lookup, &mut sequence, &ids)
        .expect("restore served checkpoint with drafter capture");
    assert_eq!(sequence.position(), restored.restored_prefix_len);
    assert_eq!(restored.matched_prefix_len, matched);
    let head = backend.dflash_head.as_ref().unwrap();
    let features = head.target_layer_ids.len() * backend.loaded.arch().hidden_size as usize;
    let columns = restored
        .restored_prefix_len
        .min(dflash_capture_window_limit(head));
    let tail = restored
        .capture_tail
        .expect("cached DFlash hidden-state tail");
    assert_eq!(
        tail.len(),
        columns * features,
        "complete restored capture window"
    );
    assert!(tail.iter().all(|value| value.is_finite()));
    assert!(
        tail.iter().any(|value| *value != 0.0),
        "capture must not be unseeded zeros"
    );
    matched
}

#[test]
#[ignore = "real Metal MoE DFlash serving; requires QWEN_MOE_DFLASH_TARGET and QWEN_MOE_DFLASH_DRAFTER; run alone with sufficient memory"]
fn moe_dflash_serving_cold_restore_cancel_retry_and_next_turn() {
    let target = required_model_path("QWEN_MOE_DFLASH_TARGET");
    let drafter = required_model_path("QWEN_MOE_DFLASH_DRAFTER");
    let runtime = Runtime::metal().expect("initialize real Metal runtime");
    let loaded = runtime
        .load_model(&target)
        .expect("load local MoE target once");
    assert_eq!(loaded.arch().kind, qwen_llm::model::ArchKind::Moe);
    loaded.set_prefix_cache_max_bytes(CACHE_BYTES);
    let family = ModelFamily::detect(loaded.gguf()).expect("identify target family");
    let template = crate::prompt_template::serve_qwen_template(family, loaded.gguf())
        .expect("identify real serving template");
    let no_thinking = crate::supports_qwen_no_thinking_prompt(family, loaded.gguf());
    assert!(no_thinking, "fixture needs the release's no-thinking mode");
    let mut backend = EngineBackend::new(
        loaded,
        MODEL_ID.into(),
        OUTPUT_TOKENS,
        Some(CONTEXT),
        CONTEXT,
        Some(&drafter),
        template,
        no_thinking,
    )
    .expect("construct MoE serving backend with DFlash enabled");
    // Repeated requests must call the drafter, not replay prior output.
    // Set instance state rather than mutate process-global environment.
    backend.dflash_prefix_replay = DflashPrefixReplayCache::default();
    let first = request(&backend, json!(CODE_PROMPT));

    let head = backend.dflash_head.take().expect("loaded DFlash head");
    backend.loaded.clear_prefix_cache();
    let (serial, serial_outcome, _) = run(&mut backend, &first, "serial reference");
    assert_eq!(serial_outcome.usage.cached_tokens, 0);
    backend.dflash_head = Some(head);
    backend.loaded.clear_prefix_cache();
    assert_eq!(backend.loaded.prefix_cache_stats().entries, 0);

    let (cold, cold_outcome, cold_visible) = run(&mut backend, &first, "cold DFlash");
    assert_eq!(cold_outcome.usage.cached_tokens, 0);
    assert_speculation(&cold_outcome, "cold DFlash");
    // One bounded greedy fixture checks emitted output, not bitwise logits
    // or recurrent-state identity across packed and serial schedules.
    assert_eq!(
        cold, serial,
        "bounded greedy code fixture differs from serial"
    );
    assert_eq!(cold_outcome.end, serial_outcome.end);
    assert_eq!(
        cold_outcome.usage.output_tokens,
        serial_outcome.usage.output_tokens
    );
    assert!(cold_visible.contains("def merge_sorted"), "{cold_visible}");

    let matched = assert_cached_capture(&backend, &first);
    let (warm, warm_outcome, _) = run(&mut backend, &first, "restored DFlash");
    assert_eq!(warm_outcome.usage.cached_tokens, matched);
    assert_speculation(&warm_outcome, "restored DFlash");

    let prompt = backend.render_prompt(&first).unwrap();
    let mut disconnect = Sink {
        abort_after: Some(1),
        ..Sink::default()
    };
    let error = backend
        .generate(&first, &prompt, &mut disconnect)
        .expect_err("disconnect after an emitted piece must abort generation");
    match error {
        BackendFailure::Aborted(error) => assert_eq!(error.kind(), io::ErrorKind::BrokenPipe),
        other => panic!("disconnect must remain a transport abort, got {other:?}"),
    }
    assert_eq!(disconnect.pieces, 1);
    assert!(!disconnect.bytes.is_empty());
    assert!(warm.starts_with(&disconnect.bytes));
    backend.request_finished();

    // Do not clear the cache: retry must survive the cancelled session's
    // partial decode without poisoning the previously committed capture.
    let matched = assert_cached_capture(&backend, &first);
    let (retry, retry_outcome, retry_visible) = run(&mut backend, &first, "retry DFlash");
    assert_eq!(retry_outcome.usage.cached_tokens, matched);
    assert_speculation(&retry_outcome, "retry DFlash");
    assert_eq!(
        retry, warm,
        "same restored schedule must recover after abort"
    );
    assert_eq!(retry_outcome.end, warm_outcome.end);
    assert_eq!(
        retry_outcome.usage.output_tokens,
        warm_outcome.usage.output_tokens
    );

    let next = request(
        &backend,
        json!([
            {"role": "user", "content": CODE_PROMPT},
            {"role": "assistant", "content": retry_visible},
            {"role": "user", "content": "What is the sorted merge of [1, 3, 5] and [2, 4, 6]? Return only the resulting list."},
        ]),
    );
    assert_ne!(backend.render_prompt(&next).unwrap(), prompt);
    let matched = assert_cached_capture(&backend, &next);
    let first_prompt_tokens = backend.tokenizer.encode(&prompt, false).unwrap().len();
    assert!(
        matched > first_prompt_tokens,
        "next turn must reuse completed assistant tokens, not just the transcript boundary: {matched} <= {first_prompt_tokens}"
    );
    let (_, next_outcome, next_visible) = run(&mut backend, &next, "next turn DFlash");
    assert_eq!(next_outcome.usage.cached_tokens, matched);
    assert_speculation(&next_outcome, "next turn DFlash");
    assert!(
        matches!(next_outcome.end, GenerationEnd::StopToken(_)),
        "next turn must finish within {OUTPUT_TOKENS} tokens"
    );
    let compact: String = next_visible
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    assert!(
        compact.contains("[1,2,3,4,5,6]"),
        "next turn: {next_visible}"
    );
}
