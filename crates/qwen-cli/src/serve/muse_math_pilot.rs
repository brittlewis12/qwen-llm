#[test]
#[ignore = "serial Metal, optimized resident Muse math/reuse/cancellation composition"]
fn muse_served_math_prefix_composition() {
    let path = Path::new(qwen_llm::test_fixtures::MUSE_GLIMMER_Q8_0.path());
    let gguf = GgufFile::open(path).unwrap();
    let ctx = MetalContext::new().unwrap();
    let mut backend = MuseGlimmerBackend::new_with_options(
        ctx,
        gguf,
        path,
        "muse-prefix-pilot".into(),
        4,
        2048,
        MuseGlimmerRuntimeOptions {
            matrix_prefill: true,
            split_decode: true,
        },
    )
    .unwrap();
    backend.prefix_reuse = true;
    let system = format!(
        "Tell a concise story about the following street. {}",
        "A winter street has lamps, rain, a station and a cafe. ".repeat(80)
    );
    let (mut first, prompt) = request(&backend, &system, false);
    first.max_output_tokens = Some(4);
    let ids = backend.encode(&prompt).unwrap();
    assert!(
        (1024..=1536).contains(&ids.len()),
        "modest native fixture length {}",
        ids.len()
    );
    qwen_llm::metal::dispatch_census_begin();
    let (cold, expected, cold_ms) = run(&mut backend, &first, &prompt);
    let census = qwen_llm::metal::dispatch_census_take();
    let count = |name| census.iter().filter(|row| row.kernel == name).count();
    let tiled = count("kernel_muse_prefill_tiled_f32_h128");
    let split = count("kernel_muse_split_attention_h128");
    eprintln!(
        "MUSE_SERVE_MATH_JSON {}",
        json!({"kind":"dispatch_preflight","input_tokens":ids.len(),"output_tokens":cold.usage.output_tokens,"tiled":tiled,"split":split,"reduce":count("kernel_muse_split_attention_reduce_h128")})
    );
    assert_eq!(tiled, ids.len() / 128 * 52);
    assert_eq!(split, 52 * 3);
    assert_eq!(count("kernel_muse_split_attention_reduce_h128"), split);
    assert_eq!(cold.usage.output_tokens, 4);
    assert_eq!(cold.usage.cached_tokens, 0);
    let (hit, actual, warm_ms) = run(&mut backend, &first, &prompt);
    assert_eq!(actual, expected);
    assert_eq!(hit.usage.cached_tokens, ids.len() - 1);
    let retained = backend.consumed_tokens.clone();
    let mut too_large = first.clone();
    too_large.max_output_tokens = Some(2049);
    assert!(
        backend
            .generate(&too_large, &prompt, &mut Sink::default())
            .is_err()
    );
    assert_eq!(backend.consumed_tokens, retained);
    eprintln!(
        "MUSE_SERVE_MATH_JSON {}",
        json!({"kind":"dispatch_and_retry","input_tokens":ids.len(),"tiled_dispatches":tiled,"split_dispatches":split,"cold_backend_ms_with_census":cold_ms,"warm_backend_ms":warm_ms,"cached_tokens":hit.usage.cached_tokens,"output_equal":true})
    );
    let (mut next, next_prompt) = request(&backend, &system, true);
    next.max_output_tokens = Some(4);
    let (warm, warm_bytes, _) = run(&mut backend, &next, &next_prompt);
    let expected_history = backend.consumed_tokens.clone();
    assert!(warm.usage.cached_tokens > 1024);
    backend.prefix_reuse = false;
    let (reset, reset_bytes, _) = run(&mut backend, &next, &next_prompt);
    assert_eq!(reset.usage.cached_tokens, 0);
    assert_eq!(warm_bytes, reset_bytes);
    assert_eq!(backend.consumed_tokens, expected_history);
    let (mut safety, safety_prompt) =
        request(&backend, &"A street has lamps and rain. ".repeat(24), false);
    safety.max_output_tokens = Some(4);
    assert!((145..512).contains(&backend.encode(&safety_prompt).unwrap().len()));
    let (_, safety_bytes, _) = run(&mut backend, &safety, &safety_prompt);
    for (reuse, mut aborted) in [
        (
            false,
            Sink {
                fail_tick_at: Some(2),
                ..Sink::default()
            },
        ),
        (
            true,
            Sink {
                fail_piece: Some(2),
                ..Sink::default()
            },
        ),
    ] {
        backend.prefix_reuse = reuse;
        assert!(
            backend
                .generate(&safety, &safety_prompt, &mut aborted)
                .is_err()
        );
        if !reuse {
            assert_eq!(
                backend
                    .loaded
                    .create_runner(&backend.ctx)
                    .unwrap()
                    .next_position(),
                128
            );
        }
        assert!(backend.consumed_tokens.is_empty());
        backend.prefix_reuse = true;
        let (recovery, bytes, _) = run(&mut backend, &safety, &safety_prompt);
        assert_eq!(recovery.usage.cached_tokens, 0);
        assert_eq!(bytes, safety_bytes);
    }
    safety.temperature = Some(0.7);
    safety.seed = Some(99);
    backend.prefix_reuse = false;
    let (_, sampled, _) = run(&mut backend, &safety, &safety_prompt);
    backend.prefix_reuse = true;
    let (retry, retried, _) = run(&mut backend, &safety, &safety_prompt);
    assert_eq!(sampled, retried);
    assert_eq!(
        retry.usage.cached_tokens,
        backend.encode(&safety_prompt).unwrap().len() - 1
    );
    eprintln!(
        "MUSE_SERVE_MATH_JSON {}",
        json!({"kind":"composition","followup_warm_reset_equal":true,"detected_prefill_decode_abort_recovery_cold":true,"sampled_retry_cohort_equal":true,"observed_weights":backend.loaded.observed_weight_bytes(),"observed_session":backend.loaded.observed_session_bytes()})
    );
}
