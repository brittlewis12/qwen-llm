use super::*;
use crate::qwen4exp_metal::hc_up_probe;
use objc2_metal::MTLComputePipelineState;

fn run_hc(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    candidate: bool,
    state: bool,
) -> Observation {
    let (observed, calls) =
        hc_up_probe::with_probe(candidate, || run_product(runner, tokens, true, state));
    assert_eq!(calls, tokens.len() * 97, "all native HC reads witnessed");
    observed
}

#[test]
#[ignore = "serial production GPU lease; research HC body native 32-token qualification"]
fn native_hc_up_shared_prefix() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    std::fs::create_dir_all(&parent).unwrap();
    let artifact = parent.join(format!("qwen4exp-native-hc-up-{}", std::process::id()));
    std::fs::create_dir(&artifact).expect("reserve native HC artifact directory");
    eprintln!("native_hc raw_logit_artifacts={}", artifact.display());
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
    let ctx = MetalContext::new().expect("real Metal required");
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_up_mix_q8_k320").unwrap();
    assert_eq!(pipeline.threadExecutionWidth(), 32);
    assert!(pipeline.maxTotalThreadsPerThreadgroup() >= 128);
    assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, PREFIX + 32).unwrap();
    let mut loaded = Qwen4ExpLoadedModel::load_with_decode_options(
        &ctx,
        &gguf,
        capacity,
        Some(PREFIX),
        Qwen4ExpDecodeOptions {
            guarded_topk: false,
            split_qsa: true,
            hc_up_mix: false,
        },
    )
    .unwrap();
    let mut runner = loaded.create_runner(&ctx).unwrap();
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
        "native_hc prefix {:?}",
        runner.last_prefill_timing().unwrap()
    );
    let checkpoint = runner.workspace.checkpoint_for_tests();
    let tensors = runner.workspace.persistent_state_tensors();
    let state_bytes: usize = tensors.iter().map(|t| t.n_bytes() as usize).sum();
    assert!(state_bytes * 4 + config.vocab_size as usize * 32 * 4 * 3 < 1024 * 1024 * 1024);
    let prefix = prefix_bytes(&runner);
    let baseline = run_hc(&mut runner, &tokens[PREFIX..], false, true);
    assert_state_bytes_eq("HC baseline old prefix", &prefix, &prefix_bytes(&runner));
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    assert_f32_bits_eq(
        "HC restored endpoint",
        &endpoint,
        &runner.logits().unwrap().to_vec(),
    );
    let replay = run_hc(&mut runner, &tokens[PREFIX..], false, true);
    assert_replay("HC restored baseline", &baseline, &replay);
    drop(replay);
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    crate::metal::dispatch_census_begin();
    let candidate = run_hc(&mut runner, &tokens[PREFIX..], true, true);
    let census = crate::metal::dispatch_census_take();
    for (name, observation) in [("baseline", &baseline), ("candidate", &candidate)] {
        let rows: Vec<u8> = observation
            .logits
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(artifact.join(format!("{name}-32x248320.f32le")), rows).unwrap();
    }
    assert_eq!(
        census
            .iter()
            .filter(|r| r.kernel == "kernel_qwen4exp_hc_up_mix_q8_k320")
            .count(),
        32 * 97
    );
    assert!(
        !census
            .iter()
            .any(|r| r.kernel == "kernel_qwen4exp_hc_gated_mean_f32")
    );
    assert_eq!(
        census
            .iter()
            .filter(|r| r.kernel == "kernel_qwen4exp_qsa_split_f16")
            .count(),
        32 * 12
    );
    assert_numeric(&baseline, &candidate, &tensors);
    assert_state_bytes_eq("HC candidate old prefix", &prefix, &prefix_bytes(&runner));
    eprintln!(
        "native_hc numerical PASS rows=32 persistent_tensors={} hc_calls={}",
        tensors.len(),
        32 * 97
    );
    let mut timings = Vec::new();
    for measured in [false, true] {
        for arm in [false, true, true, false] {
            runner.workspace.restore_checkpoint_for_tests(&checkpoint);
            let observed = run_hc(&mut runner, &tokens[PREFIX..PREFIX + 4], arm, false);
            assert_replay(
                "HC timed replay",
                if arm { &candidate } else { &baseline },
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
            "native_hc {label} four_forwards_abba_ms={v:?} saved_pct={} spread={spread} verdict={verdict}",
            (1.0 - b / a) * 100.0
        );
    }
}
