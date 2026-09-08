fn long_context_messages() -> Vec<crate::muse_glimmer_prompt::MuseGlimmerMessage> {
    let request = crate::muse_glimmer_request::MuseGlimmerRequest::from_json(include_str!(
        "../../../docs/bench/tokenizer-messages/current-marcus-long.json"
    ))
    .unwrap();
    let mut messages: Vec<crate::muse_glimmer_prompt::MuseGlimmerMessage> = Vec::new();
    for message in request.messages {
        assert!(
            message.reasoning_content.is_none()
                && message.recipient.is_none()
                && message.end_turn.is_none()
                && message.tool_calls.is_empty()
        );
        if let Some(last) = messages.last_mut()
            && last.role == message.role
        {
            last.content.push_str("\n\n");
            last.content.push_str(&message.content);
        } else {
            messages.push(message);
        }
    }
    messages
}

fn long_context_tokens(path: &str, config: &MuseGlimmerConfig) -> Vec<u32> {
    let messages = long_context_messages();
    let rendered = crate::muse_glimmer_prompt::render_muse_glimmer_atem_prompt_annotated(
        &messages,
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
        serde_json::json!({"kind":"fixture", "name":"native Current Marcus transcript prefixes; adjacent roles joined with two newlines; embedded thinking retained as content", "total_tokens":ids.len(), "token_sha256":crate::tokenizer::token_ids_sha256_i32le(&ids)})
    );
    ids.into_iter()
        .map(|id| u32::try_from(id).unwrap())
        .collect()
}

#[test]
#[ignore = "CPU-only authenticated Muse long fixture identity"]
fn muse_long_fixture_identity() {
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.path();
    let gguf = GgufFile::open(path).unwrap();
    long_context_tokens(path, &MuseGlimmerConfig::from_gguf(&gguf).unwrap());
}

#[test]
#[ignore = "CPU-only export of an authenticated native long CLI request"]
fn export_muse_long_cli_fixture() {
    use crate::muse_glimmer_prompt::MuseGlimmerMessageRole;
    use std::io::Write;
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.path();
    let gguf = GgufFile::open(path).unwrap();
    let config = MuseGlimmerConfig::from_gguf(&gguf).unwrap();
    let tokenizer = LlamaCppTokenizer::open(path).unwrap();
    let messages = long_context_messages();
    for end in (1..=messages.len()).rev() {
        if messages[end - 1].role != MuseGlimmerMessageRole::User {
            continue;
        }
        let rendered = crate::muse_glimmer_prompt::render_muse_glimmer_atem_prompt_annotated(
            &messages[..end],
            &crate::muse_glimmer_prompt::MuseGlimmerPromptOptions {
                profile: config.chat_template_profile,
                ..Default::default()
            },
        )
        .unwrap()
        .text;
        let ids = tokenizer.encode(&rendered, false).unwrap();
        if ids.len() > 32768 {
            continue;
        }
        assert!(ids.len() >= 24576);
        let wire: Vec<serde_json::Value> = messages[..end]
            .iter()
            .map(|message| {
                let role = match message.role {
                    MuseGlimmerMessageRole::System => "system",
                    MuseGlimmerMessageRole::User => "user",
                    MuseGlimmerMessageRole::Assistant => "assistant",
                    _ => panic!("unexpected tool fixture"),
                };
                serde_json::json!({"role":role,"content":message.content})
            })
            .collect();
        let document = serde_json::json!({"messages":wire});
        let metadata = serde_json::json!({"tokens":ids.len(),"token_sha256":crate::tokenizer::token_ids_sha256_i32le(&ids),"messages":end,"description":"largest complete normalized Current Marcus request ending in a user turn below32K"});
        for (name, value) in [
            ("long-cli-fixture.json", document),
            ("long-cli-fixture-meta.json", metadata.clone()),
        ] {
            let path = std::path::Path::new("target/profiles/muse-live-prefix").join(name);
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .unwrap();
            serde_json::to_writer(&mut file, &value).unwrap();
            writeln!(file).unwrap();
        }
        eprintln!("MUSE_LONG_JSON {}", metadata);
        return;
    }
    panic!("no eligible long CLI fixture");
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

fn bounded_7k_forward<'ctx, 'model>(
    ctx: &'ctx MetalContext,
    weights: &'model MuseGlimmerMetalWeights,
) -> MuseGlimmerTextForward<'ctx, 'model> {
    let mut forward =
        MuseGlimmerTextForward::new_with_optimized_prefill(ctx, weights, true).unwrap();
    forward.packed_prefill_max_end = 7168;
    forward.packed_online_max_end = 7168;
    forward
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
fn optimized_math_uses_admitted_model_context() {
    let config = MuseGlimmerConfig::unsloth_release_reference();
    let end = config.context_length as usize;
    let geometry = MuseGlimmerTextGeometry::from_config(&config, end).unwrap();
    for base in [32760, 32768, 65528, end - 16] {
        assert!(optimized_prefill_range(base, 16, end));
    }
    assert!(!optimized_prefill_range(end - 8, 16, end));
    assert!(!optimized_prefill_range(usize::MAX, 16, end));
    assert!(geometry.cache_write_offset(0, end - 1).is_ok());
    assert!(geometry.cache_write_offset(0, end).is_err());
    assert!(MuseGlimmerTextGeometry::from_config(&config, end + 1).is_err());
}

fn check_local_generated_math(
    forward: &MuseGlimmerTextForward<'_, '_>,
    session: &mut MuseGlimmerTextSession,
    token: u32,
) -> Vec<f32> {
    let position = session.next_position();
    let prefix = long_context_prefix_hash(session, position);
    let reference = forward.forward_token(token, session).unwrap();
    session.rewind_prefix(position).unwrap();
    let actual = forward.forward_generated_token(token, session).unwrap();
    let comparison = compare_logits(&actual, &reference);
    assert!(
        comparison.cosine > 0.999_99 && comparison.relative_rms < 0.002 && comparison.max_abs < 0.1,
        "local generated math {comparison:?}"
    );
    assert_eq!(long_context_prefix_hash(session, position), prefix);
    eprintln!(
        "MUSE_CONTEXT_JSON {}",
        serde_json::json!({"kind":"local_decode","position":position,"cosine":comparison.cosine,"relative_rms":comparison.relative_rms,"max_abs":comparison.max_abs,"top1_equal":comparison.reference_argmax == comparison.candidate_argmax,"prefix_immutable":true})
    );
    actual
}

#[test]
#[ignore = "serial Metal, one optimized traversal for horizon checks and current phase attribution"]
fn optimized_horizon_and_current_profile() {
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
    let forward = MuseGlimmerTextForward::new_with_optimized_prefill(&ctx, &weights, true).unwrap();
    assert_eq!(
        forward.packed_prefill_max_end,
        config.context_length as usize
    );
    assert_eq!(
        forward.packed_online_max_end,
        config.context_length as usize
    );
    let mut session =
        MuseGlimmerTextSession::new_with_split_decode(&ctx, weights.config(), 33281, true).unwrap();
    let mut previous = 0;
    for end in [8192, 32768] {
        let started = std::time::Instant::now();
        forward
            .prefill(&tokens[previous..end], &mut session)
            .unwrap();
        eprintln!(
            "MUSE_CONTEXT_JSON {}",
            serde_json::json!({"kind":"optimized_prefix","start":previous,"end":end,"wall_ms":started.elapsed().as_secs_f64()*1e3})
        );
        let prefix = long_context_prefix_hash(&session, end);
        profile_actual_matrix_chunk(&forward, &mut session, &tokens[end - 128..end], end - 128);
        assert_eq!(long_context_prefix_hash(&session, end), prefix);
        check_local_generated_math(&forward, &mut session, tokens[end]);
        session.rewind_prefix(end).unwrap();
        profile_actual_decode(&forward, &mut session, tokens[end]);
        assert_eq!(session.next_position(), end);
        previous = end;
    }
    let prefix = long_context_prefix_hash(&session, 32768);
    let started = std::time::Instant::now();
    for position in 32768..33280 {
        let output = if [32783, 32784, 33279].contains(&position) {
            check_local_generated_math(&forward, &mut session, tokens[position])
        } else {
            forward
                .forward_generated_token(tokens[position], &mut session)
                .unwrap()
        };
        assert!(output.iter().all(|value| value.is_finite()));
    }
    eprintln!(
        "MUSE_CONTEXT_JSON {}",
        serde_json::json!({"kind":"teacher_forced_horizon","start":32768,"transitions":512,"wall_ms_including_local_checks":started.elapsed().as_secs_f64()*1e3,"prefix_immutable":long_context_prefix_hash(&session,32768)==prefix})
    );
    assert_eq!(long_context_prefix_hash(&session, 32768), prefix);
    profile_actual_decode(&forward, &mut session, tokens[33280]);
    forward
        .forward_generated_token(tokens[33280], &mut session)
        .unwrap();
    assert_eq!(session.next_position(), 33281);
    assert!(
        forward
            .forward_generated_token(tokens[33280], &mut session)
            .is_err()
    );
    assert_eq!(session.next_position(), 33281);
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
    let optimized = bounded_7k_forward(&ctx, &weights);
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

#[test]
#[ignore = "serial Metal, live 8K Muse chunk extended-attention qualification"]
fn long_attention_live_8k_chunk() {
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
    let bounded = bounded_7k_forward(&ctx, &weights);
    let mut candidate = bounded_7k_forward(&ctx, &weights);
    candidate.packed_online_max_end = 32768;
    let mut session = MuseGlimmerTextSession::new(&ctx, weights.config(), 8192).unwrap();
    bounded.prefill(&tokens[..8064], &mut session).unwrap();
    live_long_attention_chunk(&bounded, &candidate, &mut session, &tokens, 8064);
}

fn live_long_attention_chunk(
    bounded: &MuseGlimmerTextForward<'_, '_>,
    candidate: &MuseGlimmerTextForward<'_, '_>,
    session: &mut MuseGlimmerTextSession,
    tokens: &[u32],
    base: usize,
) {
    live_long_attention_chunk_with_tiling(bounded, candidate, session, tokens, base, false);
}

fn live_long_attention_chunk_with_tiling(
    bounded: &MuseGlimmerTextForward<'_, '_>,
    candidate: &MuseGlimmerTextForward<'_, '_>,
    session: &mut MuseGlimmerTextSession,
    tokens: &[u32],
    base: usize,
    tiled: bool,
) {
    let prefix_hash = long_context_prefix_hash(session, base);
    let written_kv = |session: &MuseGlimmerTextSession| {
        let mut values = Vec::new();
        for layer in 0..session.geometry.layer_count {
            let offset = session.geometry.cache_write_offset(layer, base).unwrap() * 2;
            for tensor in [&session.key_cache, &session.value_cache] {
                unsafe {
                    let slice = std::slice::from_raw_parts(
                        (tensor.buffer.contents().as_ptr() as *const u8)
                            .add(tensor.offset as usize + offset)
                            as *const u16,
                        128 * 256,
                    );
                    values.extend(
                        slice
                            .iter()
                            .map(|&bits| half::f16::from_bits(bits).to_f32()),
                    );
                }
            }
        }
        values
    };
    let run = |forward: &MuseGlimmerTextForward<'_, '_>, session: &mut MuseGlimmerTextSession| {
        session.rewind_prefix(base).unwrap();
        let started = std::time::Instant::now();
        let logits = forward.prefill(&tokens[base..base + 128], session).unwrap();
        (started.elapsed().as_secs_f64() * 1e3, logits)
    };
    let a = run(bounded, session);
    let residual_a = read_f32(
        &session
            .packed
            .views(&session.geometry, 128)
            .unwrap()
            .residual,
    );
    let kv_a = written_kv(session);
    assert_eq!(long_context_prefix_hash(session, base), prefix_hash);
    for layer in 0..session.geometry.layer_count {
        let offset = session.geometry.cache_write_offset(layer, base).unwrap() * 2;
        for tensor in [&session.key_cache, &session.value_cache] {
            unsafe {
                (tensor.buffer.contents().as_ptr() as *mut u8)
                    .add(tensor.offset as usize + offset)
                    .write_bytes(0xff, 128 * 256 * 2);
            }
        }
    }
    let b = crate::muse_glimmer_metal::with_tiled_prefill(tiled, || run(candidate, session));
    let residual_b = read_f32(
        &session
            .packed
            .views(&session.geometry, 128)
            .unwrap()
            .residual,
    );
    let kv_b = written_kv(session);
    assert_eq!(long_context_prefix_hash(session, base), prefix_hash);
    let logits = compare_logits(&b.1, &a.1);
    let residual = compare_logits(&residual_b, &residual_a);
    let kv = compare_logits(&kv_b, &kv_a);
    assert!(
        logits.cosine > 0.999_99 && logits.relative_rms < 0.002 && logits.max_abs < 0.1,
        "live chunk logits {logits:?}"
    );
    assert_eq!(greedy_argmax(&a.1), greedy_argmax(&b.1));
    assert!(
        residual.cosine > 0.999_99 && residual.relative_rms < 0.002,
        "all row residual {residual:?}"
    );
    assert!(
        kv.cosine > 0.9999 && kv.relative_rms < 0.01,
        "written KV {kv:?}"
    );
    eprintln!(
        "MUSE_LONG_JSON {}",
        serde_json::json!({"kind":"live_chunk","base":base,"rows":128,"tiled":tiled,"A_ms":a.0,"B_ms":b.0,"logits_cosine":logits.cosine,"logits_rms":logits.relative_rms,"logits_max_abs":logits.max_abs,"residual_cosine":residual.cosine,"residual_rms":residual.relative_rms,"residual_max_abs":residual.max_abs,"kv_cosine":kv.cosine,"kv_rms":kv.relative_rms,"kv_max_abs":kv.max_abs,"prefix_immutable":true,"matrix_limit":candidate.packed_prefill_max_end,"online_limit":candidate.packed_online_max_end})
    );
    if tiled {
        for (kind, candidate, reference, width, cosine, rms) in [
            (
                "residual",
                &residual_b,
                &residual_a,
                session.geometry.hidden_size,
                0.999_99,
                0.002,
            ),
            ("KV", &kv_b, &kv_a, session.geometry.kv_width, 0.9999, 0.01),
        ] {
            let mut worst_rms = 0.0_f64;
            let mut minimum_cosine = 1.0_f64;
            for (index, (candidate, reference)) in candidate
                .chunks_exact(width)
                .zip(reference.chunks_exact(width))
                .enumerate()
            {
                let row = compare_logits(candidate, reference);
                assert!(
                    row.cosine > cosine && row.relative_rms < rms,
                    "tiled {kind} row={index} {row:?}"
                );
                worst_rms = worst_rms.max(row.relative_rms);
                minimum_cosine = minimum_cosine.min(row.cosine);
            }
            eprintln!(
                "MUSE_TILED_JSON {}",
                serde_json::json!({"kind":"per_row_numerics","base":base,"tensor":kind,"minimum_cosine":minimum_cosine,"worst_relative_rms":worst_rms})
            );
        }
        let b_hash = long_context_prefix_hash(session, base + 128);
        run(bounded, session);
        crate::muse_glimmer_metal::with_tiled_prefill(true, || run(candidate, session));
        let mut endpoints = Vec::new();
        for (pair, order) in [[false, true], [true, false]].into_iter().enumerate() {
            for tiled in order {
                let before = crate::muse_glimmer_metal::tiled_prefill_dispatch_count();
                let (wall_ms, logits) =
                    crate::muse_glimmer_metal::with_tiled_prefill(tiled, || {
                        run(if tiled { candidate } else { bounded }, session)
                    });
                assert_eq!(
                    crate::muse_glimmer_metal::tiled_prefill_dispatch_count() - before,
                    if tiled {
                        session.geometry.layer_count as u64
                    } else {
                        0
                    }
                );
                endpoints.push((tiled, logits));
                eprintln!(
                    "MUSE_TILED_JSON {}",
                    serde_json::json!({"kind":"live_chunk_timing","base":base,"rows":128,"pair":pair,"tiled":tiled,"wall_ms":wall_ms})
                );
            }
        }
        for (tiled, logits) in endpoints {
            assert_logits_bitwise_equal(
                "timed chunk matches untimed endpoint",
                &logits,
                if tiled { &b.1 } else { &a.1 },
            );
        }
        let restored =
            crate::muse_glimmer_metal::with_tiled_prefill(true, || run(candidate, session));
        assert_logits_bitwise_equal("tiled chunk restored after timing", &b.1, &restored.1);
        assert_eq!(long_context_prefix_hash(session, base + 128), b_hash);
        assert_eq!(long_context_prefix_hash(session, base), prefix_hash);
    }
}

#[test]
#[ignore = "serial Metal, one tiled traversal with current-online live chunk comparisons"]
fn tiled_prefill_live_chunk_transfer() {
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.path();
    let gguf = GgufFile::open(path).unwrap();
    let config = MuseGlimmerConfig::from_gguf(&gguf).unwrap();
    let tokens = long_context_tokens(path, &config);
    let ctx = MetalContext::new().unwrap();
    let transaction = ctx.begin_allocation_transaction();
    let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).unwrap();
    let geometry = MuseGlimmerTextGeometry::from_config(&config, 32784).unwrap();
    let session_plan =
        MuseGlimmerTextSessionMemoryPlan::for_geometry_with_split_decode(&ctx, &geometry, true)
            .unwrap();
    let admission = evaluate_metal_memory_admission_with_cpu_bytes(
        plan.memory_plan()
            .priced_upper_bytes()
            .checked_add(session_plan.priced_upper_bytes())
            .unwrap(),
        64 * 1024 * 1024,
        MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    assert!(admission.admitted, "tiled transfer admission {admission:?}");
    let weights =
        MuseGlimmerMetalWeights::realize(&ctx, &gguf, plan.admit(ctx.memory_signals()).unwrap())
            .unwrap()
            .into_weights();
    let forward = MuseGlimmerTextForward::new_with_optimized_prefill(&ctx, &weights, true).unwrap();
    let mut session =
        MuseGlimmerTextSession::new_with_split_decode(&ctx, weights.config(), 32784, true).unwrap();
    drop(transaction);
    let mut previous = 0;
    let mut logits = Vec::new();
    for end in [8192, 32768] {
        let started = std::time::Instant::now();
        logits = crate::muse_glimmer_metal::with_tiled_prefill(true, || {
            forward.prefill(&tokens[previous..end], &mut session)
        })
        .unwrap();
        eprintln!(
            "MUSE_TILED_JSON {}",
            serde_json::json!({"kind":"candidate_traversal_diagnostic","start":previous,"end":end,"wall_ms":started.elapsed().as_secs_f64()*1000.0})
        );
        let original_hash = long_context_prefix_hash(&session, end);
        live_long_attention_chunk_with_tiling(
            &forward,
            &forward,
            &mut session,
            &tokens,
            end - 128,
            true,
        );
        assert_eq!(long_context_prefix_hash(&session, end), original_hash);
        assert_eq!(session.next_position(), end);
        assert_logits_bitwise_equal(
            "tiled traversal endpoint restored",
            &logits,
            &read_f32(&session.logits),
        );
        previous = end;
    }
    let prefix = long_context_prefix_hash(&session, 32768);
    let expected = [
        913, 49098, 26, 352, 4557, 24, 54633, 26137, 26, 589, 48570, 948, 19044, 70912, 398, 6837,
        24,
    ];
    for (step, &token) in expected.iter().enumerate() {
        assert_eq!(
            greedy_argmax(&logits),
            token,
            "tiled historical greedy step={step}"
        );
        if step < 16 {
            logits = forward
                .forward_generated_token(token, &mut session)
                .unwrap();
        }
    }
    assert_eq!(long_context_prefix_hash(&session, 32768), prefix);
    eprintln!(
        "MUSE_TILED_JSON {}",
        serde_json::json!({"kind":"historical_greedy_check","tokens":expected,"prefix_immutable":true,"session_driver_bytes":session.observed_allocation_delta()})
    );
}

#[test]
#[ignore = "serial Metal, complete fresh 8K Muse joint range qualification"]
fn long_prefill_fresh_8k() {
    long_prefill_fresh_qualification(8192);
}

#[test]
#[ignore = "serial Metal, expensive complete fresh 32K Muse joint range qualification"]
fn long_prefill_fresh_32k() {
    long_prefill_fresh_qualification(32768);
}

#[test]
#[ignore = "serial Metal, delivered 32K prefill constructor and incremental fallback boundaries"]
fn long_prefill_32k_delivery_boundaries() {
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
    let mut forward =
        MuseGlimmerTextForward::new_with_optimized_prefill(&ctx, &weights, true).unwrap();
    // Preserve this historical bounded-policy regression independently of rollout.
    forward.packed_prefill_max_end = 32768;
    forward.packed_online_max_end = 32768;
    assert_eq!(forward.packed_prefill_max_end, 32768);
    assert_eq!(forward.packed_online_max_end, 32768);
    let mut session =
        MuseGlimmerTextSession::new_with_split_decode(&ctx, weights.config(), 32784, true).unwrap();
    let mut logits = forward.prefill(&tokens[..32768], &mut session).unwrap();
    let prefix = long_context_prefix_hash(&session, 32768);
    let expected = [
        913, 49098, 26, 352, 4557, 24, 54633, 26137, 26, 589, 48570, 948, 19044, 70912, 398, 6837,
        24,
    ];
    for (step, &token) in expected.iter().enumerate() {
        assert_eq!(
            greedy_argmax(&logits),
            token,
            "delivered greedy step={step}"
        );
        if step < 16 {
            logits = forward
                .forward_generated_token(token, &mut session)
                .unwrap();
        }
    }
    assert_eq!(long_context_prefix_hash(&session, 32768), prefix);
    for base in [32760, 32768] {
        session.rewind_prefix(base).unwrap();
        let reference = exact
            .prefill(&tokens[base..base + 16], &mut session)
            .unwrap();
        let reference_kv = long_context_prefix_hash(&session, base + 16);
        session.rewind_prefix(base).unwrap();
        let actual = forward
            .prefill(&tokens[base..base + 16], &mut session)
            .unwrap();
        assert_logits_bitwise_equal("delivered32K straddle/beyond fallback", &reference, &actual);
        assert_eq!(long_context_prefix_hash(&session, base + 16), reference_kv);
        assert_eq!(session.next_position(), base + 16);
    }
    eprintln!(
        "MUSE_LONG_JSON {}",
        serde_json::json!({"kind":"delivery32k","independent_greedy_ids":expected,"constructor_limits":32768,"straddle_and_beyond_fallback_bitwise":true,"generation_prefix_immutable":true,"session_driver_bytes":session.observed_allocation_delta()})
    );
}

fn long_prefill_fresh_qualification(count: usize) {
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.path();
    let gguf = GgufFile::open(path).unwrap();
    let config = MuseGlimmerConfig::from_gguf(&gguf).unwrap();
    let tokens = long_context_tokens(path, &config);
    let ctx = MetalContext::new().unwrap();
    let transaction = ctx.begin_allocation_transaction();
    let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).unwrap();
    let geometry = MuseGlimmerTextGeometry::from_config(&config, count + 16).unwrap();
    let session_plan =
        MuseGlimmerTextSessionMemoryPlan::for_geometry_with_split_decode(&ctx, &geometry, true)
            .unwrap();
    let metal_bytes = plan
        .memory_plan()
        .priced_upper_bytes()
        .checked_add(session_plan.priced_upper_bytes().checked_mul(2).unwrap())
        .unwrap();
    let admission = evaluate_metal_memory_admission_with_cpu_bytes(
        metal_bytes,
        64 * 1024 * 1024,
        MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    assert!(
        admission.admitted,
        "two-session qualification admission {admission:?}"
    );
    let weights =
        MuseGlimmerMetalWeights::realize(&ctx, &gguf, plan.admit(ctx.memory_signals()).unwrap())
            .unwrap()
            .into_weights();
    let bounded = bounded_7k_forward(&ctx, &weights);
    let mut extended =
        MuseGlimmerTextForward::new_with_optimized_prefill(&ctx, &weights, true).unwrap();
    extended.packed_prefill_max_end = count;
    extended.packed_online_max_end = count;
    let mut a =
        MuseGlimmerTextSession::new_with_split_decode(&ctx, weights.config(), count + 16, true)
            .unwrap();
    let mut b =
        MuseGlimmerTextSession::new_with_split_decode(&ctx, weights.config(), count + 16, true)
            .unwrap();
    drop(transaction);
    let run_fresh = |forward: &MuseGlimmerTextForward<'_, '_>,
                     session: &mut MuseGlimmerTextSession,
                     arm: &str| {
        session.reset().unwrap();
        for tensor in [&session.key_cache, &session.value_cache] {
            unsafe {
                (tensor.buffer.contents().as_ptr() as *mut u8)
                    .add(tensor.offset as usize)
                    .write_bytes(0xff, session.geometry.cache_elements().unwrap() * 2);
            }
        }
        let started = std::time::Instant::now();
        let mut chunks = 0;
        let logits = forward.prefill_with_command_checkpoint(&tokens[..count], session, || {
            chunks += 1;
            if chunks % 64 == 0 { eprintln!("MUSE_LONG_JSON {}", serde_json::json!({"kind":"progress","arm":arm,"tokens":count,"chunks":chunks,"elapsed_s":started.elapsed().as_secs_f64()})); }
            Ok(())
        }).unwrap();
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        assert!(logits.iter().all(|value| value.is_finite()));
        eprintln!(
            "MUSE_LONG_JSON {}",
            serde_json::json!({"kind":"fresh_diagnostic","arm":arm,"tokens":count,"wall_ms":wall_ms,"prefill_tps":count as f64 * 1000.0 / wall_ms})
        );
        (wall_ms, logits)
    };
    let (b_ms, mut b_logits) = run_fresh(&extended, &mut b, "B");
    let b_hash = long_context_prefix_hash(&b, count);
    let mut attention_only = bounded_7k_forward(&ctx, &weights);
    attention_only.packed_online_max_end = count;
    let probe_started = std::time::Instant::now();
    live_long_attention_chunk(&bounded, &attention_only, &mut b, &tokens, count - 128);
    let probe_ms = probe_started.elapsed().as_secs_f64() * 1e3;
    b.rewind_prefix(count - 128).unwrap();
    let restore_started = std::time::Instant::now();
    let restored = extended
        .prefill(&tokens[count - 128..count], &mut b)
        .unwrap();
    let restore_ms = restore_started.elapsed().as_secs_f64() * 1e3;
    assert_logits_bitwise_equal(
        "restore original fresh extended endpoint after probe",
        &b_logits,
        &restored,
    );
    assert_eq!(long_context_prefix_hash(&b, count), b_hash);
    assert_eq!(b.next_position(), count);
    eprintln!(
        "MUSE_LONG_JSON {}",
        serde_json::json!({"kind":"probe_restoration","tokens":count,"probe_ms":probe_ms,"restore_ms":restore_ms,"fresh_endpoint_and_all_active_kv_bitwise":true})
    );
    let (a_ms, mut a_logits) = run_fresh(&bounded, &mut a, "A");
    let a_hash = long_context_prefix_hash(&a, count);
    let endpoint = compare_logits(&b_logits, &a_logits);
    assert!(
        endpoint.cosine > 0.999_99 && endpoint.relative_rms < 0.002 && endpoint.max_abs < 0.1,
        "fresh endpoint {endpoint:?}"
    );
    let mut dot = 0.0_f64;
    let mut aa = 0.0_f64;
    let mut bb = 0.0_f64;
    let mut difference = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for layer in 0..a.geometry.layer_count {
        let (ak, av) = a.cache_prefix_views(layer, count).unwrap();
        let (bk, bv) = b.cache_prefix_views(layer, count).unwrap();
        for (a, b) in [(ak, bk), (av, bv)] {
            unsafe {
                let a = std::slice::from_raw_parts(
                    (a.buffer.contents().as_ptr() as *const u8).add(a.offset as usize)
                        as *const u16,
                    count * 256,
                );
                let b = std::slice::from_raw_parts(
                    (b.buffer.contents().as_ptr() as *const u8).add(b.offset as usize)
                        as *const u16,
                    count * 256,
                );
                for (&a, &b) in a.iter().zip(b) {
                    let (a, b) = (
                        half::f16::from_bits(a).to_f64(),
                        half::f16::from_bits(b).to_f64(),
                    );
                    assert!(a.is_finite() && b.is_finite());
                    dot += a * b;
                    aa += a * a;
                    bb += b * b;
                    difference += (a - b).powi(2);
                    max_abs = max_abs.max((a - b).abs());
                }
            }
        }
    }
    let kv_cosine = dot / (aa * bb).sqrt();
    let kv_rms = (difference / aa).sqrt();
    assert!(
        kv_cosine > 0.9999 && kv_rms < 0.01,
        "fresh active KV cosine={kv_cosine} RMS={kv_rms}"
    );
    let mut emitted = Vec::new();
    for step in 0..17 {
        let comparison = compare_logits(&b_logits, &a_logits);
        let (rms_gate, abs_gate) = if step == 0 {
            (0.002, 0.1)
        } else {
            (0.006, 0.3)
        };
        assert!(
            comparison.cosine > 0.999_99
                && comparison.relative_rms < rms_gate
                && comparison.max_abs < abs_gate,
            "fresh continuation step={step} {comparison:?}"
        );
        let a_token = greedy_argmax(&a_logits);
        let b_token = greedy_argmax(&b_logits);
        assert_eq!(a_token, b_token, "independent greedy step={step}");
        emitted.push(a_token);
        eprintln!(
            "MUSE_LONG_JSON {}",
            serde_json::json!({"kind":"continuation","tokens":count,"step":step,"cosine":comparison.cosine,"relative_rms":comparison.relative_rms,"max_abs":comparison.max_abs})
        );
        if step < 16 {
            a_logits = bounded.forward_generated_token(a_token, &mut a).unwrap();
            b_logits = extended.forward_generated_token(b_token, &mut b).unwrap();
        }
    }
    assert_eq!(long_context_prefix_hash(&a, count), a_hash);
    assert_eq!(long_context_prefix_hash(&b, count), b_hash);
    assert_eq!(a.next_position(), count + 16);
    assert_eq!(b.next_position(), count + 16);
    eprintln!(
        "MUSE_LONG_JSON {}",
        serde_json::json!({"kind":"fresh_qualification","tokens":count,"A_ms":a_ms,"B_ms":b_ms,"baseline":"delivered bounded hybrid, not all-original arithmetic","kv_cosine":kv_cosine,"kv_rms":kv_rms,"kv_max_abs":max_abs,"prefixes_immutable":true,"independent_greedy_ids":emitted,"session_driver_bytes_each":b.observed_allocation_delta(),"aggregate_required_bytes":admission.required_bytes,"cpu_oracle_allowance_bytes":64 * 1024 * 1024})
    );
}
