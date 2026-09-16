use super::*;
use crate::qwen4exp_moe::singleton_observe::topk_native::{Bank, preflight, with_mode};

fn save(path: &std::path::Path, name: &str, o: &Observation) {
    let values: Vec<f32> = o.logits.iter().flatten().copied().collect();
    std::fs::write(
        path.join(format!("{name}-logits.f32le")),
        bytemuck::cast_slice(&values),
    )
    .unwrap();
}

#[test]
#[ignore = "serial production lease; finite native parallel routing qualification, research only"]
fn native_parallel_topk_shared_prefix() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    std::fs::create_dir_all(&parent).unwrap();
    let artifact = parent.join(format!("qwen4exp-native-topk-{}", std::process::id()));
    std::fs::create_dir(&artifact).unwrap();
    eprintln!("native_topk artifacts={}", artifact.display());
    let data = include_bytes!(
        "../../../../../docs/bench/2026-08-29-qwen4exp-selected-semantic/natural-ssh.u32le"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(data)),
        "874537119c68f6c566c4288ba17c1099694416edb001c4003249570894438e97"
    );
    let tokens: Vec<u32> = data
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(tokens.len(), PREFIX + 32);
    let ctx = MetalContext::new().unwrap();
    preflight(&ctx);
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
            hc_up_mix: false,
        },
    )
    .unwrap();
    let mut runner = loaded.create_runner(&ctx).unwrap();
    let _selected = Qwen4ExpPackedSelectedQsaOverride::set(true);
    runner.prefill(&tokens[..PREFIX]).unwrap();
    let checkpoint = runner.workspace.checkpoint_for_tests();
    let mut ordinary = run_product(&mut runner, &tokens[PREFIX..], true, true);
    save(&artifact, "ordinary", &ordinary);
    let a = Bank::new(&ctx, 32 * 48);
    let b = Bank::new(&ctx, 32 * 48);
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    let (captured, _) = with_mode(false, Some(&a), || {
        run_product(&mut runner, &tokens[PREFIX..], true, true)
    });
    a.save(&artifact, "incumbent");
    save(&artifact, "captured-incumbent", &captured);
    assert_replay("native router capture", &ordinary, &captured);
    drop(captured);
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    crate::metal::dispatch_census_begin();
    let (mut candidate, calls) = with_mode(true, Some(&b), || {
        run_product(&mut runner, &tokens[PREFIX..], true, true)
    });
    let census = crate::metal::dispatch_census_take();
    b.save(&artifact, "parallel");
    save(&artifact, "parallel", &candidate);
    assert_eq!(calls, 32 * 48);
    assert_eq!(
        census
            .iter()
            .filter(|r| r.kernel == "kernel_topk_logits_softmax_parallel_f32")
            .count(),
        32 * 48
    );
    assert!(
        !census
            .iter()
            .any(|r| r.kernel == "kernel_topk_logits_softmax_f32"
                || r.kernel == "kernel_qwen4exp_hc_up_mix_q8_k320")
    );
    assert_eq!(
        census
            .iter()
            .filter(|r| r.kernel == "kernel_qwen4exp_qsa_split_f16")
            .count(),
        32 * 12
    );
    a.assert_equivalent(&b);
    assert_replay("native parallel router", &ordinary, &candidate);
    assert_eq!(ordinary.state.len(), 121);
    for o in [&ordinary, &candidate] {
        assert!(
            o.logits
                .iter()
                .chain(&o.hyper)
                .flatten()
                .all(|v| v.is_finite())
        );
    }
    ordinary.state.clear();
    candidate.state.clear();
    eprintln!(
        "native_topk numerical PASS 32fullrows/hyper/121states bitwise;1536routerrows finite/CPU-exact order/weights bitwise"
    );
    let mut timings = Vec::new();
    for warm in [true, false] {
        for (i, parallel) in [false, true, true, false].into_iter().enumerate() {
            runner.workspace.restore_checkpoint_for_tests(&checkpoint);
            let (observed, calls) = with_mode(parallel, None, || {
                run_product(&mut runner, &tokens[PREFIX..PREFIX + 4], true, false)
            });
            let name = format!("warm{warm}-{i}");
            save(&artifact, &name, &observed);
            let hyper: Vec<f32> = observed.hyper.iter().flatten().copied().collect();
            std::fs::write(
                artifact.join(format!("{name}-hyper.f32le")),
                bytemuck::cast_slice(&hyper),
            )
            .unwrap();
            let path = artifact.join(format!("{name}-timing.json"));
            let mut evidence = serde_json::json!({"parallel":parallel,"calls":calls,"validation":"pending","forwards":observed.timing.iter().map(|t|serde_json::json!({"gpu_ms":t.gpu_ms,"executor_wall_ms":t.total_wall_ms,"details":format!("{t:?}")})).collect::<Vec<_>>()});
            std::fs::write(&path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
            assert_eq!(calls, 4 * 48);
            assert_replay(
                "native topk timed arm",
                if parallel { &candidate } else { &ordinary },
                &observed,
            );
            let gpu: f64 = observed.timing.iter().map(|t| t.gpu_ms.unwrap()).sum();
            let wall: f64 = observed.timing.iter().map(|t| t.total_wall_ms).sum();
            assert!(gpu.is_finite() && gpu > 0.0 && wall.is_finite() && wall > 0.0);
            evidence["validation"] = serde_json::json!("bitwise_passed");
            evidence["gpu_ms"] = serde_json::json!(gpu);
            evidence["executor_wall_ms"] = serde_json::json!(wall);
            std::fs::write(&path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
            if !warm {
                timings.push((gpu, wall));
            }
        }
    }
    let mut axes = Vec::new();
    for (axis, floor) in [(0, 0.10), (1, 0.05)] {
        let v: Vec<f64> = timings
            .iter()
            .map(|t| if axis == 0 { t.0 } else { t.1 })
            .collect();
        let a = (v[0] + v[3]) * 0.5;
        let b = (v[1] + v[2]) * 0.5;
        let spread = (v[0] - v[3]).abs() / a;
        let useful =
            b <= a * (1.0 - floor) && v[1] <= v[0] * (1.0 - floor) && v[2] <= v[3] * (1.0 - floor);
        let result = serde_json::json!({"axis":if axis==0{"gpu"}else{"executor_wall"},"four_forward_abba_ms":v,"saved_fraction":1.0-b/a,"spread":spread,"verdict":if spread>0.05{"INCONCLUSIVE"}else if useful{"USEFUL"}else{"HOLD"}});
        eprintln!("native_topk {result}");
        axes.push(result);
    }
    std::fs::write(artifact.join("result.json"),serde_json::to_vec_pretty(&serde_json::json!({"numerical":"bitwise","finite_router_rows":1536,"axes":axes,"scope":"research selector; no nonfinite/product qualification"})).unwrap()).unwrap();
}
