use super::*;

fn run_hc_product(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    hc: bool,
    split: bool,
    state: bool,
    prefix: usize,
) -> Observation {
    runner
        .workspace
        .configure_hc_up_mix(runner.ctx, hc)
        .unwrap();
    run_product_at(runner, tokens, split, state, prefix)
}

fn save_rows(directory: &std::path::Path, name: &str, observation: &Observation) {
    let rows: Vec<u8> = observation
        .logits
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(directory.join(format!("{name}.f32le")), rows).unwrap();
}

fn witness(census: &[crate::metal::DispatchCensusRow], steps: usize, split: bool) {
    assert_eq!(
        census
            .iter()
            .filter(|r| r.kernel == "kernel_qwen4exp_hc_up_mix_q8_k320")
            .count(),
        steps * 97
    );
    assert!(
        !census
            .iter()
            .any(|r| r.kernel == "kernel_qwen4exp_hc_gated_mean_f32")
    );
    for name in [
        "kernel_qwen4exp_qsa_split_f16",
        "kernel_qwen4exp_qsa_split_merge_f32",
    ] {
        assert_eq!(
            census.iter().filter(|r| r.kernel == name).count(),
            if split { steps * 12 } else { 0 }
        );
    }
    assert_eq!(
        census
            .iter()
            .filter(|r| r.kernel == "kernel_qwen4exp_qsa_attention_logits_f16")
            .count(),
        if split { 0 } else { steps * 12 }
    );
}

#[test]
#[ignore = "serial production lease; HC actual-product composition and incremental decode timing"]
fn product_hc_up_shared_prefix() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    std::fs::create_dir_all(&parent).unwrap();
    let artifact = parent.join(format!("qwen4exp-product-hc-{}", std::process::id()));
    std::fs::create_dir(&artifact).unwrap();
    eprintln!("product_hc artifacts={}", artifact.display());
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
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, PREFIX + 32).unwrap();
    let mut loaded = Qwen4ExpLoadedModel::load_with_decode_options(
        &ctx,
        &gguf,
        capacity,
        Some(PREFIX),
        Qwen4ExpDecodeOptions {
            split_qsa: true,
            hc_up_mix: true,
        },
    )
    .unwrap();
    assert!(loaded.hc_up_mix_enabled() && loaded.split_decode_enabled());
    let mut runner = loaded.create_runner(&ctx).unwrap();
    let tensors = runner.workspace.persistent_state_tensors();
    let state_bytes: usize = tensors.iter().map(|t| t.n_bytes() as usize).sum();
    assert!(state_bytes * 4 + config.vocab_size as usize * 32 * 4 * 3 < 1024 * 1024 * 1024);
    {
        let empty = runner.workspace.checkpoint_for_tests();
        assert_eq!(runner.next_position(), 0);
        assert!(runner.logits().is_err());
        let baseline = run_hc_product(&mut runner, &tokens[..4], false, false, true, 0);
        runner.workspace.restore_checkpoint_for_tests(&empty);
        assert_eq!(runner.next_position(), 0);
        assert!(runner.logits().is_err());
        let replay = run_hc_product(&mut runner, &tokens[..4], false, false, true, 0);
        assert_replay("empty scalar restored baseline", &baseline, &replay);
        drop(replay);
        runner.workspace.restore_checkpoint_for_tests(&empty);
        crate::metal::dispatch_census_begin();
        let candidate = run_hc_product(&mut runner, &tokens[..4], true, false, true, 0);
        let census = crate::metal::dispatch_census_take();
        save_rows(&artifact, "empty-scalar-baseline-4x248320", &baseline);
        save_rows(&artifact, "empty-scalar-candidate-4x248320", &candidate);
        witness(&census, 4, false);
        assert_numeric_at(&baseline, &candidate, &tensors, 0);
        runner.workspace.restore_checkpoint_for_tests(&empty);
        assert!(runner.logits().is_err());
        eprintln!("product_hc empty_state_scalar_prefix PASS steps=4");
    }
    runner.workspace.configure_hc_up_mix(&ctx, true).unwrap();
    runner
        .workspace
        .set_split_decode_for_tests(&ctx, true)
        .unwrap();
    let _selected = Qwen4ExpPackedSelectedQsaOverride::set(true);
    crate::metal::dispatch_census_begin();
    let endpoint = runner.prefill(&tokens[..PREFIX]).unwrap().to_vec();
    let census = crate::metal::dispatch_census_take();
    assert!(
        !census
            .iter()
            .any(|r| r.kernel == "kernel_qwen4exp_hc_up_mix_q8_k320")
    );
    assert_eq!(
        runner.last_prefill_timing().unwrap().packed_token_count,
        PREFIX
    );
    eprintln!(
        "product_hc prefix {:?}",
        runner.last_prefill_timing().unwrap()
    );
    let checkpoint = runner.workspace.checkpoint_for_tests();
    let prefix = prefix_bytes(&runner);
    let mut baseline = run_hc_product(&mut runner, &tokens[PREFIX..], false, true, true, PREFIX);
    assert_state_bytes_eq(
        "product HC baseline old prefix",
        &prefix,
        &prefix_bytes(&runner),
    );
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    assert_f32_bits_eq(
        "product HC restored endpoint",
        &endpoint,
        &runner.logits().unwrap().to_vec(),
    );
    let replay = run_hc_product(&mut runner, &tokens[PREFIX..], false, true, true, PREFIX);
    assert_replay("product HC restored baseline", &baseline, &replay);
    drop(replay);
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    crate::metal::dispatch_census_begin();
    let mut candidate = run_hc_product(&mut runner, &tokens[PREFIX..], true, true, true, PREFIX);
    let census = crate::metal::dispatch_census_take();
    save_rows(&artifact, "split-on-baseline-32x248320", &baseline);
    save_rows(&artifact, "split-on-candidate-32x248320", &candidate);
    witness(&census, 32, true);
    assert_numeric(&baseline, &candidate, &tensors);
    assert_state_bytes_eq(
        "product HC candidate old prefix",
        &prefix,
        &prefix_bytes(&runner),
    );
    baseline.state.clear();
    candidate.state.clear();
    eprintln!(
        "product_hc split=true numerical PASS steps=32 states={}",
        tensors.len()
    );
    {
        runner.workspace.restore_checkpoint_for_tests(&checkpoint);
        let baseline = run_hc_product(&mut runner, &tokens[PREFIX..], false, false, true, PREFIX);
        assert_state_bytes_eq(
            "independent HC baseline old prefix",
            &prefix,
            &prefix_bytes(&runner),
        );
        runner.workspace.restore_checkpoint_for_tests(&checkpoint);
        crate::metal::dispatch_census_begin();
        let candidate = run_hc_product(&mut runner, &tokens[PREFIX..], true, false, true, PREFIX);
        let census = crate::metal::dispatch_census_take();
        save_rows(&artifact, "split-off-baseline-32x248320", &baseline);
        save_rows(&artifact, "split-off-candidate-32x248320", &candidate);
        witness(&census, 32, false);
        assert_numeric(&baseline, &candidate, &tensors);
        assert_state_bytes_eq(
            "independent HC candidate old prefix",
            &prefix,
            &prefix_bytes(&runner),
        );
        eprintln!(
            "product_hc split=false numerical PASS steps=32 states={}",
            tensors.len()
        );
    }
    let mut timings = Vec::new();
    for measured in [false, true] {
        for hc in [false, true, true, false] {
            runner.workspace.restore_checkpoint_for_tests(&checkpoint);
            let observed = run_hc_product(
                &mut runner,
                &tokens[PREFIX..PREFIX + 4],
                hc,
                true,
                false,
                PREFIX,
            );
            assert_replay(
                "actual-product timed replay",
                if hc { &candidate } else { &baseline },
                &observed,
            );
            if measured {
                let gpu: f64 = observed.timing.iter().map(|t| t.gpu_ms.unwrap()).sum();
                let wall: f64 = observed.timing.iter().map(|t| t.total_wall_ms).sum();
                assert!(gpu.is_finite() && gpu > 0.0 && wall.is_finite() && wall > 0.0);
                timings.push((gpu, wall));
            }
        }
    }
    for (label, axis, floor) in [("GPU", 0, 0.05), ("wall", 1, 0.03)] {
        let v: Vec<f64> = timings
            .iter()
            .map(|t| if axis == 0 { t.0 } else { t.1 })
            .collect();
        let a = (v[0] + v[3]) * 0.5;
        let b = (v[1] + v[2]) * 0.5;
        let spread = (v[0] - v[3]).abs() / a;
        let useful =
            b <= a * (1.0 - floor) && v[1] <= v[0] * (1.0 - floor) && v[2] <= v[3] * (1.0 - floor);
        let verdict = if spread > 0.05 {
            "INCONCLUSIVE"
        } else if useful {
            "USEFUL"
        } else {
            "HOLD"
        };
        eprintln!(
            "product_hc {label} four_forwards_abba_ms={v:?} saved_pct={} spread={spread} verdict={verdict}",
            (1.0 - b / a) * 100.0
        );
    }
}
