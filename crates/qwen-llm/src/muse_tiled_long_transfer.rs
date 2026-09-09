fn replay_tail(session: &MuseGlimmerTextSession, base: usize, rows: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(52 * 2 * rows * 256);
    for layer in 0..52 {
        let offset = session.geometry.cache_write_offset(layer, base).unwrap() * 2;
        for tensor in [&session.key_cache, &session.value_cache] {
            unsafe {
                let bits = std::slice::from_raw_parts(
                    (tensor.buffer.contents().as_ptr() as *const u8)
                        .add(tensor.offset as usize + offset) as *const u16,
                    rows * 256,
                );
                values.extend(bits.iter().map(|&bits| half::f16::from_bits(bits).to_f32()));
            }
        }
    }
    values
}

#[test]
#[ignore = "serial Metal, frozen tiled32K local transfer; persisted common prefix, no full reference"]
fn tiled_prefill_32k_local_transfer() {
    const BASE: usize = 32640;
    const ROWS: usize = 128;
    const CONTINUATION: usize = 8;
    let root = std::path::Path::new("target/profiles/muse-tiled-diagnostic");
    let seed: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("prefix.json")).unwrap()).unwrap();
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.path();
    let metadata = std::fs::metadata(path).unwrap();
    assert_eq!(seed["identity"]["model"], path);
    assert_eq!(seed["identity"]["model_bytes"], metadata.len());
    assert_eq!(
        seed["identity"]["model_modified_ns"],
        metadata
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_string()
    );
    let gguf = GgufFile::open(path).unwrap();
    let config = MuseGlimmerConfig::from_gguf(&gguf).unwrap();
    let tokens = long_context_tokens(path, &config);
    assert_eq!(
        seed["identity"]["prefix_tokens"],
        serde_json::json!(&tokens[..DIAGNOSTIC_BASE])
    );
    let identity = serde_json::json!({"version":1,"seed_identity":seed["identity"],"seed_hash":seed["hash"],"extension_math":"current_online_matrix","base":BASE,"prefix_tokens":&tokens[..BASE]});
    let manifest_path = root.join("prefix32.json");
    let manifest: Option<serde_json::Value> = manifest_path
        .exists()
        .then(|| serde_json::from_reader(std::fs::File::open(&manifest_path).unwrap()).unwrap());
    if let Some(manifest) = &manifest {
        assert_eq!(manifest["identity"], identity);
    } else {
        assert!(
            !root.join("prefix32.bin").exists(),
            "incomplete prior prefix artifact retained"
        );
    }
    let ctx = MetalContext::new().unwrap();
    let transaction = ctx.begin_allocation_transaction();
    let plan = MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).unwrap();
    let geometry =
        MuseGlimmerTextGeometry::from_config(&config, BASE + ROWS + CONTINUATION).unwrap();
    let session_plan =
        MuseGlimmerTextSessionMemoryPlan::for_geometry_with_split_decode(&ctx, &geometry, true)
            .unwrap();
    let admission = evaluate_metal_memory_admission_with_cpu_bytes(
        plan.memory_plan()
            .priced_upper_bytes()
            .checked_add(session_plan.priced_upper_bytes())
            .unwrap(),
        128 * 1024 * 1024,
        MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    assert!(admission.admitted, "32K replay admission {admission:?}");
    let weights =
        MuseGlimmerMetalWeights::realize(&ctx, &gguf, plan.admit(ctx.memory_signals()).unwrap())
            .unwrap()
            .into_weights();
    let forward = MuseGlimmerTextForward::new_with_optimized_prefill(&ctx, &weights, true).unwrap();
    let mut session = MuseGlimmerTextSession::new_with_split_decode(
        &ctx,
        weights.config(),
        BASE + ROWS + CONTINUATION,
        true,
    )
    .unwrap();
    drop(transaction);
    let (file_name, end, expected_hash) = if let Some(manifest) = &manifest {
        ("prefix32.bin", BASE, manifest["hash"].as_str().unwrap())
    } else {
        (
            "prefix.bin",
            DIAGNOSTIC_BASE,
            seed["hash"].as_str().unwrap(),
        )
    };
    let mut file = std::fs::File::open(root.join(file_name)).unwrap();
    assert_eq!(
        file.metadata().unwrap().len(),
        (52 * 2 * end * 256 * 2) as u64
    );
    replay_prefix_io(&session, &mut file, true, end);
    session.next_position = end;
    assert_eq!(
        long_context_prefix_hash(&session, end).to_hex().to_string(),
        expected_hash
    );
    if end < BASE {
        let started = std::time::Instant::now();
        forward.prefill(&tokens[end..BASE], &mut session).unwrap();
        eprintln!(
            "MUSE_TILED32_JSON {}",
            serde_json::json!({"kind":"prefix_extension","start":end,"end":BASE,"wall_ms":started.elapsed().as_secs_f64()*1000.0})
        );
        assert_eq!(
            long_context_prefix_hash(&session, end).to_hex().to_string(),
            expected_hash
        );
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("prefix32.bin"))
            .unwrap();
        replay_prefix_io(&session, &mut file, false, BASE);
        file.sync_all().unwrap();
        let manifest = serde_json::json!({"identity":identity,"hash":long_context_prefix_hash(&session, BASE).to_hex().to_string(),"producer_source":std::env::var("MUSE_DIAGNOSTIC_SOURCE").unwrap_or_default()});
        serde_json::to_writer(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(manifest_path)
                .unwrap(),
            &manifest,
        )
        .unwrap();
    }
    let prefix = long_context_prefix_hash(&session, BASE);
    eprintln!(
        "MUSE_TILED32_JSON {}",
        serde_json::json!({"kind":"common_prefix","hash":prefix.to_hex().to_string(),"base":BASE,"restored32":end==BASE,"session_driver_bytes":session.observed_allocation_delta()})
    );
    let run = |session: &mut MuseGlimmerTextSession, tiled: bool| {
        session.rewind_prefix(BASE).unwrap();
        let before = crate::muse_glimmer_metal::tiled_prefill_dispatch_count();
        let started = std::time::Instant::now();
        let logits = crate::muse_glimmer_metal::with_tiled_prefill(tiled, || {
            forward.prefill(&tokens[BASE..BASE + ROWS], session)
        })
        .unwrap();
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(
            crate::muse_glimmer_metal::tiled_prefill_dispatch_count() - before,
            if tiled { 52 } else { 0 }
        );
        assert_eq!(session.next_position(), BASE + ROWS);
        (wall_ms, logits)
    };
    let mut oracles = Vec::new();
    for tiled in [false, true] {
        for layer in 0..52 {
            let offset = session.geometry.cache_write_offset(layer, BASE).unwrap() * 2;
            for tensor in [&session.key_cache, &session.value_cache] {
                unsafe {
                    (tensor.buffer.contents().as_ptr() as *mut u8)
                        .add(tensor.offset as usize + offset)
                        .write_bytes(0xff, (ROWS + CONTINUATION) * 256 * 2);
                }
            }
        }
        let (_, logits) = run(&mut session, tiled);
        let chunk_hash = long_context_prefix_hash(&session, BASE + ROWS);
        let residual = read_f32(
            &session
                .packed
                .views(&session.geometry, ROWS)
                .unwrap()
                .residual,
        );
        let mut logits = vec![logits];
        for &token in &tokens[BASE + ROWS..BASE + ROWS + CONTINUATION] {
            logits.push(
                forward
                    .forward_generated_token(token, &mut session)
                    .unwrap(),
            );
        }
        assert_eq!(long_context_prefix_hash(&session, BASE), prefix);
        assert_eq!(long_context_prefix_hash(&session, BASE + ROWS), chunk_hash);
        oracles.push((
            logits,
            residual,
            replay_tail(&session, BASE, ROWS + CONTINUATION),
            chunk_hash,
        ));
    }
    for (step, (a, b)) in oracles[0].0.iter().zip(&oracles[1].0).enumerate() {
        let comparison = compare_logits(b, a);
        eprintln!(
            "MUSE_TILED32_JSON {}",
            serde_json::json!({"kind":"logits","step":step,"cosine":comparison.cosine,"relative_rms":comparison.relative_rms,"max_abs":comparison.max_abs,"top1_equal":comparison.reference_argmax==comparison.candidate_argmax})
        );
        assert!(
            comparison.cosine > 0.999_99
                && comparison.relative_rms < if step == 0 { 0.002 } else { 0.006 }
                && comparison.max_abs < if step == 0 { 0.1 } else { 0.3 },
            "logits step={step}: {comparison:?}"
        );
        assert_eq!(comparison.reference_argmax, comparison.candidate_argmax);
    }
    for (kind, a, b, width, cosine, rms) in [
        (
            "residual",
            &oracles[0].1,
            &oracles[1].1,
            6656,
            0.999_99,
            0.002,
        ),
        (
            "written_KV",
            &oracles[0].2,
            &oracles[1].2,
            256,
            0.9999,
            0.01,
        ),
    ] {
        let comparison = compare_logits(b, a);
        let worst_row_rms = a
            .chunks_exact(width)
            .zip(b.chunks_exact(width))
            .map(|(a, b)| compare_logits(b, a).relative_rms)
            .fold(0.0_f64, f64::max);
        eprintln!(
            "MUSE_TILED32_JSON {}",
            serde_json::json!({"kind":kind,"cosine":comparison.cosine,"relative_rms":comparison.relative_rms,"max_abs":comparison.max_abs,"diagnostic_worst_row_rms":worst_row_rms})
        );
        assert!(
            comparison.cosine > cosine && comparison.relative_rms < rms,
            "aggregate {kind}: {comparison:?}"
        );
    }
    run(&mut session, false);
    run(&mut session, true);
    let mut measured = Vec::new();
    for (pair, order) in [[false, true], [true, false]].into_iter().enumerate() {
        for tiled in order {
            let (wall_ms, logits) = run(&mut session, tiled);
            eprintln!(
                "MUSE_TILED32_JSON {}",
                serde_json::json!({"kind":"chunk_wall","pair":pair,"tiled":tiled,"wall_ms":wall_ms})
            );
            measured.push((tiled, logits));
        }
    }
    for (tiled, logits) in measured {
        assert_logits_bitwise_equal("timed endpoint", &logits, &oracles[usize::from(tiled)].0[0]);
    }
    let (_, restored) = run(&mut session, true);
    assert_logits_bitwise_equal("restored B endpoint", &restored, &oracles[1].0[0]);
    assert_eq!(
        long_context_prefix_hash(&session, BASE + ROWS),
        oracles[1].3
    );
    assert_eq!(long_context_prefix_hash(&session, BASE), prefix);
    eprintln!(
        "MUSE_TILED32_JSON {}",
        serde_json::json!({"kind":"complete","fixed_token_continuations":CONTINUATION,"timed_endpoints_bitwise":true,"prefix_immutable":true,"restoration_bitwise":true})
    );
}
