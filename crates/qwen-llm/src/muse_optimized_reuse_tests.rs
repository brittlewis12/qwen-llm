#[test]
#[ignore = "serial Metal, optimized warm/reset numerical and KV composition through split-visible context"]
fn optimized_live_prefix_math_composition() {
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.path();
    let gguf = GgufFile::open(path).unwrap();
    let config = MuseGlimmerConfig::from_gguf(&gguf).unwrap();
    let root = long_context_tokens(path, &config)[..1169].to_vec();
    let ctx = MetalContext::new().unwrap();
    let transaction = ctx.begin_allocation_transaction();
    let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).unwrap();
    let geometry = MuseGlimmerTextGeometry::from_config(&config, 1184).unwrap();
    let session_plan =
        MuseGlimmerTextSessionMemoryPlan::for_geometry_with_split_decode(&ctx, &geometry, true)
            .unwrap();
    let admission = evaluate_metal_memory_admission_with_cpu_bytes(
        plan.memory_plan().priced_upper_bytes() + 2 * session_plan.priced_upper_bytes(),
        32 * 1024 * 1024,
        MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    assert!(admission.admitted);
    let weights =
        MuseGlimmerMetalWeights::realize(&ctx, &gguf, plan.admit(ctx.memory_signals()).unwrap())
            .unwrap()
            .into_weights();
    let forward = MuseGlimmerTextForward::new_with_tiled_prefill(&ctx, &weights, true).unwrap();
    let mut live =
        MuseGlimmerTextSession::new_with_split_decode(&ctx, weights.config(), 1184, true).unwrap();
    let mut fresh =
        MuseGlimmerTextSession::new_with_split_decode(&ctx, weights.config(), 1184, true).unwrap();
    drop(transaction);
    forward.prefill(&root, &mut live).unwrap();
    let mut history = root.clone();
    let mut short_branch = root[..1057].to_vec();
    short_branch[1040] += 1;
    let mut long_branch = root.clone();
    long_branch[1040] += 2;
    let mut unrelated = root[..1153].to_vec();
    unrelated[0] += 1;
    for (case, mut prompt) in [
        root.clone(),
        root[..1152].to_vec(),
        short_branch,
        long_branch,
        unrelated,
        Vec::new(),
    ]
    .into_iter()
    .enumerate()
    {
        let generated_history = prompt.is_empty();
        if generated_history {
            prompt = history.clone();
            prompt.extend_from_slice(&root[..16]);
        }
        let reused = history
            .iter()
            .zip(&prompt)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt.len() - 1);
        if generated_history {
            assert_eq!(reused, history.len());
        }
        assert_eq!(
            (reused, prompt.len() - reused),
            [
                (1168, 1),
                (1151, 1),
                (1040, 17),
                (1040, 129),
                (0, 1153),
                (1154, 16)
            ][case]
        );
        let immutable = (reused != 0).then(|| long_context_prefix_hash(&live, reused));
        live.rewind_prefix(reused).unwrap();
        let actual = forward.prefill(&prompt[reused..], &mut live).unwrap();
        if let Some(immutable) = immutable {
            assert_eq!(long_context_prefix_hash(&live, reused), immutable);
        }
        fresh.reset().unwrap();
        let expected = forward.prefill(&prompt, &mut fresh).unwrap();
        let comparison = compare_logits(&actual, &expected);
        assert!(
            comparison.cosine > 0.999_99
                && comparison.relative_rms < 0.002
                && comparison.max_abs < 0.1,
            "rewind {reused} {comparison:?}"
        );
        let mut dot = 0.0_f64;
        let mut aa = 0.0_f64;
        let mut bb = 0.0_f64;
        let mut difference = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for layer in 0..52 {
            let (ak, av) = live.cache_prefix_views(layer, prompt.len()).unwrap();
            let (bk, bv) = fresh.cache_prefix_views(layer, prompt.len()).unwrap();
            for (a, b) in [(ak, bk), (av, bv)] {
                unsafe {
                    let bits = |tensor: &MetalTensor| {
                        std::slice::from_raw_parts(
                            (tensor.buffer.contents().as_ptr() as *const u8)
                                .add(tensor.offset as usize)
                                as *const u16,
                            prompt.len() * 256,
                        )
                    };
                    for (&a, &b) in bits(&a).iter().zip(bits(&b)) {
                        let a = half::f16::from_bits(a).to_f64();
                        let b = half::f16::from_bits(b).to_f64();
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
        let kv_rms = (difference / bb).sqrt();
        assert!(
            kv_cosine > 0.9999 && kv_rms < 0.01,
            "warm/reset KV {kv_cosine} {kv_rms}"
        );
        let token = greedy_argmax(&expected);
        let actual = forward.forward_generated_token(token, &mut live).unwrap();
        let expected = forward.forward_generated_token(token, &mut fresh).unwrap();
        let continuation = compare_logits(&actual, &expected);
        let written = compare_logits(
            &replay_tail(&live, prompt.len(), 1),
            &replay_tail(&fresh, prompt.len(), 1),
        );
        assert!(
            written.cosine > 0.9999 && written.relative_rms < 0.01,
            "generated row KV {written:?}"
        );
        assert!(
            continuation.cosine > 0.999_99
                && continuation.relative_rms < 0.006
                && continuation.max_abs < 0.3,
            "warm/reset continuation {continuation:?}"
        );
        eprintln!(
            "MUSE_REUSE_MATH_JSON {}",
            serde_json::json!({"prompt":prompt.len(),"reused":reused,"computed":prompt.len()-reused,"generated_history":generated_history,"logit_rms":comparison.relative_rms,"logit_max_abs":comparison.max_abs,"top1_equal":comparison.reference_argmax==comparison.candidate_argmax,"kv_rms":kv_rms,"kv_cosine":kv_cosine,"kv_max_abs":max_abs,"continuation_max_abs":continuation.max_abs,"new_generated_kv_rms":written.relative_rms,"prefix_immutable":true})
        );
        history = prompt;
        history.push(token);
    }
}
