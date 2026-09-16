use super::*;

#[test]
fn decode_policy_defaults_and_independent_qualification() {
    let on = Qwen4ExpDecodeOptions {
        guarded_topk: true,
        split_qsa: true,
        hc_up_mix: true,
    };
    let off = Qwen4ExpDecodeOptions {
        guarded_topk: false,
        split_qsa: false,
        hc_up_mix: false,
    };
    assert_eq!(Qwen4ExpDecodeOptions::default(), on);
    assert_eq!(on.qualified("Apple M4 Max", 2048, [true; 3]), on);
    for device in [
        "Apple M4",
        "Apple M4 Pro",
        "Apple M3 Max",
        "Apple M5 Max",
        "",
    ] {
        assert_eq!(on.qualified(device, 8192, [true; 3]), off);
    }
    for bits in 0..8 {
        let flags = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0];
        let expected = Qwen4ExpDecodeOptions {
            guarded_topk: flags[0],
            split_qsa: flags[1],
            hc_up_mix: flags[2],
        };
        assert_eq!(on.qualified("Apple M4 Max", 2048, flags), expected);
        assert_eq!(
            expected.qualified("Apple M4 Max", 2048, [true; 3]),
            expected
        );
    }
    let capacity =
        Qwen4ExpSessionCapacity::for_forward_limit(&Qwen4ExpConfig::flash_next_reference(), 2047)
            .unwrap();
    assert_eq!(capacity.qsa_physical_capacity, 2048);
    assert_eq!(
        on.qualified("Apple M4 Max", capacity.forward_limit, [true; 3]),
        Qwen4ExpDecodeOptions {
            split_qsa: false,
            ..on
        }
    );
}

#[test]
fn decode_policy_optional_scratch_admission_truth_table() {
    for requested in [false, true] {
        for baseline in [false, true] {
            for optimized in [false, true] {
                assert_eq!(
                    optional_split_admission_fallback(requested, baseline, optimized),
                    (requested, baseline, optimized) == (true, true, false)
                );
            }
        }
    }
}

fn witness(census: &[DispatchCensusRow], steps: usize, flags: [bool; 3]) {
    for (name, per_step, enabled) in [
        (crate::qwen4exp_moe::guarded_topk::KERNEL, 48, flags[0]),
        ("kernel_qwen4exp_qsa_split_f16", 12, flags[1]),
        ("kernel_qwen4exp_qsa_split_merge_f32", 12, flags[1]),
        ("kernel_qwen4exp_qsa_split_softmax_f32", 12, flags[1]),
        ("kernel_qwen4exp_hc_up_mix_q8_k320", 97, flags[2]),
        ("kernel_qwen4exp_qsa_attention_logits_f16", 12, true),
    ] {
        assert_eq!(
            census.iter().filter(|r| r.kernel == name).count(),
            if enabled { steps * per_step } else { 0 },
            "{name}"
        );
    }
}

fn save(directory: &std::path::Path, name: &str, observation: &Observation) {
    let bytes: Vec<u8> = observation
        .logits
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(directory.join(format!("{name}.f32le")), bytes).unwrap();
    for (index, bytes) in observation.state.iter().enumerate() {
        std::fs::write(directory.join(format!("{name}.state{index}")), bytes).unwrap();
    }
}

#[test]
#[ignore = "production lease; default identity, independent rollbacks, dissimilar QSA guardrail; no timing bracket"]
fn qualified_defaults_product_closure() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    let artifact = parent.join(format!("qwen4exp-defaults-{}", std::process::id()));
    std::fs::create_dir(&artifact).unwrap();
    eprintln!("defaults artifacts={}", artifact.display());
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
    assert_eq!(tokens.len(), PREFIX + 32);
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, 2578 + 8).unwrap();
    let mut loaded =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, 2578).unwrap();
    assert!(
        loaded.guarded_topk_enabled()
            && loaded.split_decode_enabled()
            && loaded.hc_up_mix_enabled()
    );
    let mut runner = loaded.create_runner(&ctx).unwrap();
    let tensors = runner.workspace.persistent_state_tensors();
    assert_eq!(tensors.len(), 121);
    let state_bytes: usize = tensors.iter().map(|t| t.n_bytes() as usize).sum();
    assert!(state_bytes * 4 + config.vocab_size as usize * 32 * 4 * 3 < 1024 * 1024 * 1024);
    let empty = runner.workspace.checkpoint_for_tests();
    let _selected = Qwen4ExpPackedSelectedQsaOverride::set(true);
    crate::metal::dispatch_census_begin();
    runner.prefill(&tokens[..PREFIX]).unwrap();
    let census = crate::metal::dispatch_census_take();
    for name in [
        crate::qwen4exp_moe::guarded_topk::KERNEL,
        "kernel_qwen4exp_qsa_split_f16",
        "kernel_qwen4exp_qsa_split_merge_f32",
        "kernel_qwen4exp_hc_up_mix_q8_k320",
    ] {
        assert!(!census.iter().any(|r| r.kernel == name), "packed {name}");
    }
    assert_eq!(
        runner.last_prefill_timing().unwrap().packed_token_count,
        PREFIX
    );
    {
        let checkpoint = runner.workspace.checkpoint_for_tests();
        let old_prefix = prefix_bytes(&runner);
        crate::metal::dispatch_census_begin();
        let defaults = observe_product_at(&mut runner, &tokens[PREFIX..], true, PREFIX);
        witness(&crate::metal::dispatch_census_take(), 32, [true; 3]);
        save(&artifact, "defaults-32x248320", &defaults);
        assert_state_bytes_eq("default old prefix", &old_prefix, &prefix_bytes(&runner));
        for (name, flags) in [
            ("explicit", [true, true, true]),
            ("topk-off", [false, true, true]),
            ("hc-off", [true, true, false]),
            ("qsa-off", [true, false, true]),
        ] {
            runner.workspace.restore_checkpoint_for_tests(&checkpoint);
            runner
                .workspace
                .configure_guarded_topk(&ctx, flags[0])
                .unwrap();
            runner
                .workspace
                .set_split_decode_for_tests(&ctx, flags[1])
                .unwrap();
            runner
                .workspace
                .configure_hc_up_mix(&ctx, flags[2])
                .unwrap();
            crate::metal::dispatch_census_begin();
            let observed = observe_product_at(&mut runner, &tokens[PREFIX..], true, PREFIX);
            witness(&crate::metal::dispatch_census_take(), 32, flags);
            save(&artifact, name, &observed);
            if flags[1] && flags[2] {
                assert_replay(name, &defaults, &observed);
            } else {
                assert_numeric(&observed, &defaults, &tensors);
            }
            assert_state_bytes_eq(name, &old_prefix, &prefix_bytes(&runner));
            eprintln!("defaults {name} PASS steps=32 states=121");
        }
    }
    runner.workspace.restore_checkpoint_for_tests(&empty);
    drop(empty);
    runner.workspace.configure_guarded_topk(&ctx, true).unwrap();
    runner.workspace.configure_hc_up_mix(&ctx, true).unwrap();
    runner
        .workspace
        .set_split_decode_for_tests(&ctx, true)
        .unwrap();
    let bytes = include_bytes!(
        "../../../../../docs/bench/2026-08-29-qwen4exp-selected-semantic/known-answer.u32le"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "94305afe3a2a668e2095b31db79598950498ce4867c9f164c985d3c311902ada"
    );
    let prompt: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(prompt.len(), 2578);
    let endpoint = runner.prefill(&prompt).unwrap().to_vec();
    assert_eq!(
        runner.last_prefill_timing().unwrap().packed_token_count,
        prompt.len()
    );
    let checkpoint = runner.workspace.checkpoint_for_tests();
    runner
        .workspace
        .set_split_decode_for_tests(&ctx, false)
        .unwrap();
    let mut continuation = Vec::new();
    let mut baseline = Observation {
        logits: Vec::new(),
        hyper: Vec::new(),
        state: Vec::new(),
        timing: Vec::new(),
    };
    let mut token = argmax(&endpoint) as u32;
    crate::metal::dispatch_census_begin();
    for step in 0..8 {
        continuation.push(token);
        let mut observed = observe_product_at(&mut runner, &[token], false, prompt.len() + step);
        token = argmax(&observed.logits[0]) as u32;
        baseline.logits.append(&mut observed.logits);
        baseline.hyper.append(&mut observed.hyper);
    }
    witness(
        &crate::metal::dispatch_census_take(),
        8,
        [true, false, true],
    );
    baseline.state = snapshot_persistent_state(&runner);
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    crate::metal::dispatch_census_begin();
    let candidate = run_product_at(&mut runner, &continuation, true, true, prompt.len());
    witness(&crate::metal::dispatch_census_take(), 8, [true; 3]);
    save(&artifact, "known-answer-qsa-off-8x248320", &baseline);
    save(&artifact, "known-answer-qsa-on-8x248320", &candidate);
    std::fs::write(
        artifact.join("known-answer-continuation.u32le"),
        continuation
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_numeric_at(&baseline, &candidate, &tensors, prompt.len());
    eprintln!(
        "defaults dissimilar QSA numerical PASS steps=8 states=121 continuation={continuation:?}"
    );
}
