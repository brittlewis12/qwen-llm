//! Actual resident-backend repeated-turn screen; no HTTP or cold-process claim.

use super::*;
use serde_json::json;

#[derive(Default)]
struct Sink {
    bytes: Vec<u8>,
    fail_tick: bool,
    fail_piece: Option<usize>,
    pieces: usize,
}

impl GenerationSink for Sink {
    fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.pieces += 1;
        if self.fail_piece == Some(self.pieces) {
            return Err(io::Error::other("injected decode disconnect"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn tick(&mut self) -> io::Result<()> {
        if self.fail_tick {
            Err(io::Error::other("injected prefill disconnect"))
        } else {
            Ok(())
        }
    }
}

fn request(backend: &MuseGlimmerBackend, system: &str, followup: bool) -> (ServeRequest, String) {
    let input = if followup {
        json!([
            {"role":"user", "content":"begin"},
            {"role":"assistant", "content":"You taste copper. The world steadies. What's your name?"},
            {"role":"user", "content":"Mara. I get up and look for somewhere warm, with a power outlet."}
        ])
    } else {
        json!("begin")
    };
    let mut request = crate::open_responses::items::parse_request(&json!({
        "model":"muse-prefix-pilot", "instructions":system, "input":input,
        "temperature":0.0, "seed":42, "max_output_tokens":16,
        "reasoning":{"effort":"high"},
    }))
    .unwrap();
    backend.normalize_request(&mut request).unwrap();
    let prompt = backend.render_prompt(&request).unwrap();
    (request, prompt)
}

fn run(
    backend: &mut MuseGlimmerBackend,
    request: &ServeRequest,
    prompt: &str,
) -> (GenerationOutcome, Vec<u8>, f64) {
    let mut sink = Sink::default();
    let start = Instant::now();
    let outcome = backend
        .generate(request, prompt, &mut sink)
        .unwrap_or_else(|_| panic!("Muse backend request failed"));
    (outcome, sink.bytes, start.elapsed().as_secs_f64() * 1e3)
}

#[test]
#[ignore = "serial Metal, real Muse repeated-turn correctness and full-backend wall"]
fn muse_live_prefix_reuse_backend_packet() {
    let path = std::path::Path::new(qwen_llm::test_fixtures::MUSE_GLIMMER_Q8_0.path());
    let gguf = GgufFile::open(path).unwrap();
    let ctx = MetalContext::new().unwrap();
    let mut backend =
        MuseGlimmerBackend::new(ctx, gguf, path, "muse-prefix-pilot".into(), 16, 8192).unwrap();
    backend.prefix_reuse = true;
    let (short, short_prompt) = request(
        &backend,
        "You are a concise storyteller. Describe a winter street.",
        false,
    );
    let (first, bytes, _) = run(&mut backend, &short, &short_prompt);
    assert_eq!(first.usage.cached_tokens, 0);
    assert_eq!(first.usage.output_tokens, 16);
    let (hit, hit_bytes, _) = run(&mut backend, &short, &short_prompt);
    assert_eq!(hit_bytes, bytes);
    assert_eq!(hit.usage.cached_tokens, first.usage.input_tokens - 1);
    let retained = backend.consumed_tokens.clone();
    let mut over_capacity = short.clone();
    over_capacity.max_output_tokens = Some(8193);
    assert!(
        backend
            .generate(&over_capacity, &short_prompt, &mut Sink::default())
            .is_err()
    );
    assert_eq!(backend.consumed_tokens, retained);
    for mut aborted in [
        Sink {
            fail_tick: true,
            ..Sink::default()
        },
        Sink {
            fail_piece: Some(2),
            ..Sink::default()
        },
    ] {
        assert!(
            backend
                .generate(&short, &short_prompt, &mut aborted)
                .is_err()
        );
        assert!(backend.consumed_tokens.is_empty());
        let (recovered, recovered_bytes, _) = run(&mut backend, &short, &short_prompt);
        assert_eq!(recovered.usage.cached_tokens, 0);
        assert_eq!(recovered_bytes, bytes);
    }
    println!(
        "MUSE_PREFIX_JSON {}",
        json!({"kind":"failure_oracles", "passed":true})
    );

    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../docs/bench/tokenizer-messages/current-reva-short-qwen36.json"
    ))
    .unwrap();
    let system = fixture["messages"][0]["content"].as_str().unwrap();
    let (prime, prime_prompt) = request(&backend, system, false);
    let (next, next_prompt) = request(&backend, system, true);
    let ids = backend.encode(&next_prompt).unwrap();
    assert!(
        (4096..=7168).contains(&ids.len()),
        "frozen long cell must stay below attention fallback"
    );
    let ids_i32: Vec<_> = ids.iter().map(|&t| t as i32).collect();
    println!(
        "MUSE_PREFIX_JSON {}",
        json!({"kind":"fixture", "prompt_tokens":ids.len(),
        "prompt_sha256":qwen_llm::tokenizer::token_ids_sha256_i32le(&ids_i32),
        "capacity":backend.capacity, "temperature":0.0, "max_output_tokens":16,
        "transcript":"Current system; authored short assistant reply; Mara followup"})
    );

    let (_, prime_bytes, _) = run(&mut backend, &prime, &prime_prompt);
    backend.prefix_reuse = false;
    let (reference, expected, oracle_ms) = run(&mut backend, &next, &next_prompt);
    assert_eq!(reference.usage.cached_tokens, 0);
    assert_eq!(reference.usage.output_tokens, 16);
    let expected_history = backend.consumed_tokens.clone();
    println!(
        "MUSE_PREFIX_JSON {}",
        json!({"kind":"oracle", "arm":"A", "wall_ms":oracle_ms})
    );
    backend.prefix_reuse = true;
    let (_, repeated_prime, _) = run(&mut backend, &prime, &prime_prompt);
    assert_eq!(repeated_prime, prime_bytes);
    let (candidate, actual, oracle_ms) = run(&mut backend, &next, &next_prompt);
    assert_eq!(actual, expected);
    assert_eq!(backend.consumed_tokens, expected_history);
    assert!(candidate.usage.cached_tokens * 10 >= ids.len() * 9);
    println!(
        "MUSE_PREFIX_JSON {}",
        json!({"kind":"oracle", "arm":"B", "wall_ms":oracle_ms,
        "cached_tokens":candidate.usage.cached_tokens, "emission_and_consumed_ids_equal":true})
    );

    // The full reference and candidate requests above condition both paths.
    // Every measured arm starts after the SAME prime request; no KV readbacks.
    for (index, enabled) in [false, true, true, false].into_iter().enumerate() {
        backend.prefix_reuse = true;
        let (_, prime_output, _) = run(&mut backend, &prime, &prime_prompt);
        assert_eq!(prime_output, prime_bytes);
        backend.prefix_reuse = enabled;
        let (outcome, output, wall_ms) = run(&mut backend, &next, &next_prompt);
        assert_eq!(output, expected);
        assert_eq!(backend.consumed_tokens, expected_history);
        assert_eq!(outcome.usage.output_tokens, 16);
        println!(
            "MUSE_PREFIX_JSON {}",
            json!({"kind":"sample", "index":index,
            "arm":if enabled {"B"} else {"A"}, "wall_ms":wall_ms,
            "input_tokens":outcome.usage.input_tokens, "cached_tokens":outcome.usage.cached_tokens,
            "output_tokens":outcome.usage.output_tokens, "consumed_tokens":backend.consumed_tokens.len(),
            "observed_weights":backend.loaded.observed_weight_bytes(),
            "observed_session":backend.loaded.observed_session_bytes()})
        );
    }
}
