use super::*;

const ROWS: usize = 1024;

#[test]
fn router_n1024_planner_reachability() {
    let direct = plan_qwen4exp_prefill_execution(ROWS, Some(2048), true, 2051).unwrap();
    assert_eq!(direct.packed_ranges, vec![0..1024]);
    let selected = plan_qwen4exp_prefill_execution(3072, Some(2048), true, 2051).unwrap();
    assert_eq!(
        selected.packed_ranges,
        vec![0..2048, 2048..2051, 2051..3072]
    );
}

fn observe_prefill(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
) -> (Observation, Qwen4ExpPrefillTiming) {
    let logits = runner.prefill(tokens).unwrap().to_vec();
    let timing = runner.last_prefill_timing().unwrap();
    let observed = Observation {
        logits: vec![logits],
        hyper: vec![runner.workspace.final_hyper_for_tests()],
        state: snapshot_persistent_state(runner),
        timing: Vec::new(),
    };
    (observed, timing)
}

fn causal_metadata(runner: &Qwen4ExpTextRunner<'_, '_, '_>) -> String {
    format!(
        "position={} qsa={:?} ple={:?}",
        runner.next_position(),
        runner.workspace.qsa_committed_lengths(),
        runner.workspace.ple_prior_tokens()
    )
}

fn assert_continuation_replay(
    artifact: &std::path::Path,
    step: usize,
    baseline: &Observation,
    candidate: &Observation,
) {
    assert!(baseline.state.is_empty());
    assert_eq!(candidate.state.len(), 121);
    assert_eq!(baseline.logits.len(), candidate.logits.len());
    for (a, b) in baseline.logits.iter().zip(&candidate.logits) {
        assert_f32_bits_eq("N1024 continuation logits", a, b);
    }
    for (a, b) in baseline.hyper.iter().zip(&candidate.hyper) {
        assert_f32_bits_eq("N1024 continuation hyper", a, b);
    }
    for (index, bytes) in candidate.state.iter().enumerate() {
        let expected = std::fs::read(artifact.join(format!("A-step{step}.state{index}"))).unwrap();
        assert_eq!(
            &expected, bytes,
            "N1024 step{step} persistent tensor{index}"
        );
    }
}

#[test]
#[ignore = "production lease; one exact N1024 native qualification and ordinary prefill ABBA"]
fn packed_router_n1024_native_qualification() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    let artifact = parent.join(format!("qwen4exp-router-n1024-{}", std::process::id()));
    std::fs::create_dir(&artifact).unwrap();
    eprintln!("router_n1024 artifacts={}", artifact.display());
    let bytes = include_bytes!(
        "../../../../../docs/bench/2026-08-29-qwen4exp-selected-semantic/natural-ssh.u32le"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "874537119c68f6c566c4288ba17c1099694416edb001c4003249570894438e97"
    );
    let tokens: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    let prompt = tokens[..ROWS]
        .iter()
        .flat_map(|&token| {
            tokenizer
                .try_decode_piece_bytes_exact(token as i32)
                .unwrap()
                .to_vec()
        })
        .collect::<Vec<_>>();
    let prompt = String::from_utf8(prompt).unwrap();
    let encoded: Vec<u32> = tokenizer
        .encode(&prompt, false)
        .unwrap()
        .into_iter()
        .map(|v| v as u32)
        .collect();
    assert_eq!(&encoded, &tokens[..ROWS]);
    std::fs::write(artifact.join("prompt.txt"), prompt).unwrap();
    let ctx = MetalContext::new().unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, ROWS + 4).unwrap();
    let mut loaded =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, ROWS).unwrap();
    assert!(
        loaded.guarded_topk_enabled()
            && loaded.hc_up_mix_enabled()
            && !loaded.split_decode_enabled()
    );
    let mut runner = loaded.create_runner(&ctx).unwrap();
    let plan = runner.workspace.memory_plan().clone();
    assert!(
        !plan
            .allocations()
            .iter()
            .any(|a| a.name == "session.qsa_split")
    );
    let tensors = runner.workspace.persistent_state_tensors();
    assert_eq!(tensors.len(), 121);
    let state_bytes: usize = tensors.iter().map(|t| t.n_bytes() as usize).sum();
    assert!(state_bytes * 4 + config.vocab_size as usize * 5 * 4 < 1024 * 1024 * 1024);
    let empty = runner.workspace.checkpoint_for_tests();
    let mut baseline = Vec::new();
    let mut metadata = Vec::new();
    let mut baseline_census: Option<Vec<DispatchCensusRow>> = None;
    for (name, strict) in [("A", false), ("B", true)] {
        runner.workspace.restore_checkpoint_for_tests(&empty);
        with_qwen4exp_packed_router_e8p32_strict_override(strict, || {
            crate::metal::dispatch_census_begin();
            let (prefill, timing) = observe_prefill(&mut runner, &tokens[..ROWS]);
            let census = crate::metal::dispatch_census_take();
            defaults::save(&artifact, &format!("{name}-prefill"), &prefill);
            std::fs::write(
                artifact.join(format!("{name}-prefill.txt")),
                format!("{timing:?}\n{}\n{census:#?}", causal_metadata(&runner)),
            )
            .unwrap();
            assert_eq!(timing.command_count, 1);
            assert_eq!(timing.packed_token_count, ROWS);
            assert!(!timing.contains_selection);
            if strict {
                assert_replay("N1024 prefill", &baseline[0], &prefill);
                assert_eq!(metadata[0], causal_metadata(&runner));
                assert_router_candidate_census(
                    "N1024",
                    ROWS,
                    48,
                    baseline_census.as_ref().unwrap(),
                    &census,
                );
            } else {
                baseline.push(prefill);
                metadata.push(causal_metadata(&runner));
                baseline_census = Some(census);
            }
            for step in 0..4 {
                crate::metal::dispatch_census_begin();
                let mut observed = observe_product_at(
                    &mut runner,
                    &tokens[ROWS + step..ROWS + step + 1],
                    true,
                    ROWS + step,
                );
                let census = crate::metal::dispatch_census_take();
                defaults::save(&artifact, &format!("{name}-step{step}"), &observed);
                std::fs::write(
                    artifact.join(format!("{name}-step{step}.txt")),
                    format!("{}\n{census:#?}", causal_metadata(&runner)),
                )
                .unwrap();
                defaults::witness(&census, 1, [true, false, true]);
                if strict {
                    assert_continuation_replay(&artifact, step, &baseline[step + 1], &observed);
                    assert_eq!(metadata[step + 1], causal_metadata(&runner));
                } else {
                    observed.state.clear();
                    baseline.push(observed);
                    metadata.push(causal_metadata(&runner));
                }
            }
        });
        assert_eq!(runner.workspace.memory_plan(), &plan);
    }
    eprintln!("router_n1024 exact prefill and4 continuation states PASS");
    let mut timings = Vec::new();
    for (index, strict) in [false, true, true, false].into_iter().enumerate() {
        runner.workspace.restore_checkpoint_for_tests(&empty);
        let (observed, timing) = with_qwen4exp_packed_router_e8p32_strict_override(strict, || {
            observe_prefill(&mut runner, &tokens[..ROWS])
        });
        defaults::save(&artifact, &format!("timed{index}"), &observed);
        std::fs::write(
            artifact.join(format!("timed{index}.txt")),
            format!(
                "strict={strict} timing={timing:?}\n{}",
                causal_metadata(&runner)
            ),
        )
        .unwrap();
        assert_replay("timed N1024 replay", &baseline[0], &observed);
        assert_eq!(metadata[0], causal_metadata(&runner));
        assert_eq!(timing.command_count, 1);
        assert_eq!(timing.packed_token_count, ROWS);
        timings.push(timing);
    }
    let mut keep = true;
    for axis in ["GPU", "wall"] {
        let values: Vec<_> = timings
            .iter()
            .map(|t| {
                if axis == "GPU" {
                    t.complete_gpu_ms().unwrap()
                } else {
                    t.total_wall_ms
                }
            })
            .collect();
        assert!(values.iter().all(|v| v.is_finite() && *v > 0.0));
        let a = (values[0] + values[3]) * 0.5;
        let b = (values[1] + values[2]) * 0.5;
        let spread = (values[0] - values[3]).abs() / a;
        let useful = if axis == "GPU" {
            b <= a * 0.99 && values[1] < values[0] && values[2] < values[3]
        } else {
            b <= a
        };
        keep &= spread <= 0.05 && useful;
        eprintln!(
            "router_n1024 {axis} ABBA_ms={values:?} saved_pct={} spread={spread} useful={useful}",
            (1.0 - b / a) * 100.0
        );
    }
    eprintln!(
        "router_n1024 verdict={}",
        if keep {
            "KEEP"
        } else {
            "HOLD_REMOVE_WIDTH_NO_RETRY"
        }
    );
}
