fn long_context_tokens(path: &str, config: &MuseGlimmerConfig) -> Vec<u32> {
    let request = crate::muse_glimmer_request::MuseGlimmerRequest::from_json(include_str!(
        "../../../docs/bench/tokenizer-messages/current-marcus-long.json"
    ))
    .unwrap();
    let rendered = crate::muse_glimmer_prompt::render_muse_glimmer_atem_prompt_annotated(
        &request.messages,
        &crate::muse_glimmer_prompt::MuseGlimmerPromptOptions {
            profile: config.chat_template_profile,
            add_generation_prompt: false,
            ..Default::default()
        },
    )
    .unwrap()
    .text;
    let tokenizer = LlamaCppTokenizer::open(path).unwrap();
    let ids = tokenizer.encode(&rendered, false).unwrap();
    assert!(ids.len() >= 32784);
    eprintln!(
        "MUSE_LONG_JSON {}",
        serde_json::json!({"kind":"fixture", "name":"native Current Marcus transcript prefixes; embedded thinking retained as content", "total_tokens":ids.len(), "token_sha256":crate::tokenizer::token_ids_sha256_i32le(&ids)})
    );
    ids.into_iter()
        .map(|id| u32::try_from(id).unwrap())
        .collect()
}

fn long_context_prefix_hash(session: &MuseGlimmerTextSession, end: usize) -> blake3::Hash {
    let mut hash = blake3::Hasher::new();
    for layer in 0..session.geometry.layer_count {
        let (key, value) = session.cache_prefix_views(layer, end).unwrap();
        for tensor in [key, value] {
            unsafe {
                hash.update(std::slice::from_raw_parts(
                    (tensor.buffer.contents().as_ptr() as *const u8).add(tensor.offset as usize),
                    end * session.geometry.kv_width * 2,
                ));
            }
        }
    }
    hash.finalize()
}

#[test]
fn optimized_prefill_eligibility_uses_absolute_end() {
    for (base, rows, expected) in [
        (0, 128, true),
        (7040, 128, true),
        (7152, 16, true),
        (7160, 16, false),
        (7168, 16, false),
        (0, 0, false),
        (usize::MAX, 16, false),
    ] {
        assert_eq!(optimized_prefill_range(base, rows, 7168), expected);
    }
}

#[test]
#[ignore = "serial Metal, Muse capacity reservation and 7168 prefill/decode boundary"]
fn optimized_prefill_large_reservation_boundary() {
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.path();
    let gguf = GgufFile::open(path).unwrap();
    let config = MuseGlimmerConfig::from_gguf(&gguf).unwrap();
    let tokens = long_context_tokens(path, &config);
    let ctx = MetalContext::new().unwrap();
    let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).unwrap();
    let weights =
        MuseGlimmerMetalWeights::realize(&ctx, &gguf, plan.admit(ctx.memory_signals()).unwrap())
            .unwrap()
            .into_weights();
    let exact = MuseGlimmerTextForward::new(&ctx, &weights).unwrap();
    let optimized =
        MuseGlimmerTextForward::new_with_optimized_prefill(&ctx, &weights, true).unwrap();
    let mut reference = MuseGlimmerTextSession::new(&ctx, weights.config(), 7275).unwrap();
    let mut candidate =
        MuseGlimmerTextSession::new_with_split_decode(&ctx, weights.config(), 7275, true).unwrap();
    let a = exact.prefill(&tokens[..6884], &mut reference).unwrap();
    let b = optimized.prefill(&tokens[..6884], &mut candidate).unwrap();
    let endpoint = compare_logits(&b, &a);
    assert!(
        endpoint.cosine > 0.999_99 && endpoint.relative_rms < 0.002 && endpoint.max_abs < 0.1,
        "reservation endpoint {endpoint:?}"
    );
    assert_eq!(greedy_argmax(&a), greedy_argmax(&b));
    let original_prefix = long_context_prefix_hash(&candidate, 6884);
    let mut worst_rms = 0.0_f64;
    let mut worst_abs = 0.0_f32;
    let mut minimum_cosine = 1.0_f64;
    let mut top1_matches = 0;
    for position in 6884..7275 {
        let a = exact
            .forward_token(tokens[position], &mut reference)
            .unwrap();
        let b = optimized
            .forward_generated_token(tokens[position], &mut candidate)
            .unwrap();
        let comparison = compare_logits(&b, &a);
        assert!(
            comparison.cosine > 0.999_99
                && comparison.relative_rms < 0.006
                && comparison.max_abs < 0.3,
            "reservation position={position} {comparison:?}"
        );
        let top1_equal = greedy_argmax(&a) == greedy_argmax(&b);
        top1_matches += usize::from(top1_equal);
        if (7160..=7176).contains(&position) {
            assert!(top1_equal, "boundary top1 position={position}");
        }
        worst_rms = worst_rms.max(comparison.relative_rms);
        worst_abs = worst_abs.max(comparison.max_abs);
        minimum_cosine = minimum_cosine.min(comparison.cosine);
    }
    assert_eq!(candidate.next_position(), 7275);
    assert_eq!(long_context_prefix_hash(&candidate, 6884), original_prefix);
    for base in [7160, 7168] {
        candidate.rewind_prefix(base).unwrap();
        let original = exact
            .prefill(&tokens[base..base + 16], &mut candidate)
            .unwrap();
        let original_kv = long_context_prefix_hash(&candidate, base + 16);
        candidate.rewind_prefix(base).unwrap();
        let fallback = optimized
            .prefill(&tokens[base..base + 16], &mut candidate)
            .unwrap();
        assert_logits_bitwise_equal(
            "ineligible incremental chunk uses original graph",
            &original,
            &fallback,
        );
        assert_eq!(long_context_prefix_hash(&candidate, base + 16), original_kv);
    }
    eprintln!(
        "MUSE_LONG_JSON {}",
        serde_json::json!({"kind":"reservation_boundary", "prompt_tokens":6884, "teacher_forced_transitions":391,"capacity":7275,"endpoint_cosine":endpoint.cosine,"endpoint_rms":endpoint.relative_rms,"endpoint_max_abs":endpoint.max_abs,"continuation_min_cosine":minimum_cosine,"continuation_max_rms":worst_rms,"continuation_max_abs":worst_abs,"top1_matches":top1_matches,"boundary_17_top1_equal":true,"prefix_immutable":true,"straddling_and_beyond_chunk_bitwise_fallback":true,"session_driver_bytes":candidate.observed_allocation_delta()})
    );
}
