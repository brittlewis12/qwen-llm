use super::*;

fn run(
    r: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    enabled: bool,
    split: bool,
    hc: bool,
    state: bool,
    prefix: usize,
) -> Observation {
    r.workspace.configure_guarded_topk(r.ctx, enabled).unwrap();
    r.workspace.configure_hc_up_mix(r.ctx, hc).unwrap();
    run_product_at(r, tokens, split, state, prefix)
}

fn save(path: &std::path::Path, name: &str, o: &Observation) {
    for (label, rows) in [("logits", &o.logits), ("hyper", &o.hyper)] {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        std::fs::write(
            path.join(format!("{name}-{label}.f32le")),
            bytemuck::cast_slice(&flat),
        )
        .unwrap();
    }
}

fn exact(label: &str, a: &Observation, b: &Observation) {
    assert_replay(label, a, b);
    for o in [a, b] {
        assert!(
            o.logits
                .iter()
                .chain(&o.hyper)
                .flatten()
                .all(|v| v.is_finite())
        );
    }
}

fn witness(census: &[crate::metal::DispatchCensusRow], steps: usize, split: bool, hc: bool) {
    assert_eq!(
        census
            .iter()
            .filter(|r| r.kernel == "kernel_qwen4exp_qsa_split_f16")
            .count(),
        if split { steps * 12 } else { 0 }
    );
    assert_eq!(
        census
            .iter()
            .filter(|r| r.kernel == "kernel_qwen4exp_hc_up_mix_q8_k320")
            .count(),
        if hc { steps * 97 } else { 0 }
    );
    assert_eq!(
        census
            .iter()
            .filter(|r| r.kernel == crate::qwen4exp_moe::guarded_topk::KERNEL)
            .count(),
        steps * 48
    );
    assert!(
        !census
            .iter()
            .any(|r| r.kernel == "kernel_topk_logits_softmax_f32"
                || r.kernel == "kernel_topk_logits_softmax_parallel_f32")
    );
}

#[test]
#[ignore = "serial production lease; guarded selector actual-product compositions and native timing"]
fn product_guarded_topk_shared_prefix() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    let artifact = parent.join(format!("qwen4exp-product-topk-{}", std::process::id()));
    std::fs::create_dir(&artifact).unwrap();
    eprintln!("product_topk artifacts={}", artifact.display());
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
            guarded_topk: true,
            split_qsa: true,
            hc_up_mix: false,
        },
    )
    .unwrap();
    assert!(loaded.guarded_topk_enabled());
    let mut r = loaded.create_runner(&ctx).unwrap();
    {
        let empty = r.workspace.checkpoint_for_tests();
        let a = run(&mut r, &tokens[..4], false, false, false, true, 0);
        r.workspace.restore_checkpoint_for_tests(&empty);
        let replay = run(&mut r, &tokens[..4], false, false, false, true, 0);
        save(&artifact, "empty-a", &a);
        save(&artifact, "empty-restored-a", &replay);
        exact("empty incumbent replay", &a, &replay);
        drop(replay);
        r.workspace.restore_checkpoint_for_tests(&empty);
        crate::metal::dispatch_census_begin();
        let b = run(&mut r, &tokens[..4], true, false, false, true, 0);
        let census = crate::metal::dispatch_census_take();
        save(&artifact, "empty-a", &a);
        save(&artifact, "empty-b", &b);
        witness(&census, 4, false, false);
        exact("empty scalar topk", &a, &b);
        r.workspace.restore_checkpoint_for_tests(&empty);
    }
    r.workspace.configure_guarded_topk(&ctx, true).unwrap();
    r.workspace.set_split_decode_for_tests(&ctx, true).unwrap();
    let _selected = Qwen4ExpPackedSelectedQsaOverride::set(true);
    crate::metal::dispatch_census_begin();
    r.prefill(&tokens[..PREFIX]).unwrap();
    let census = crate::metal::dispatch_census_take();
    assert!(
        !census
            .iter()
            .any(|row| row.kernel == crate::qwen4exp_moe::guarded_topk::KERNEL)
    );
    let checkpoint = r.workspace.checkpoint_for_tests();
    let mut a = run(&mut r, &tokens[PREFIX..], false, true, false, true, PREFIX);
    r.workspace.restore_checkpoint_for_tests(&checkpoint);
    let replay = run(&mut r, &tokens[PREFIX..], false, true, false, true, PREFIX);
    save(&artifact, "split-on-a", &a);
    save(&artifact, "split-on-restored-a", &replay);
    exact("incumbent restored", &a, &replay);
    drop(replay);
    r.workspace.restore_checkpoint_for_tests(&checkpoint);
    crate::metal::dispatch_census_begin();
    let mut b = run(&mut r, &tokens[PREFIX..], true, true, false, true, PREFIX);
    let census = crate::metal::dispatch_census_take();
    save(&artifact, "split-on-a", &a);
    save(&artifact, "split-on-b", &b);
    witness(&census, 32, true, false);
    exact("actual product split on", &a, &b);
    assert_eq!(a.state.len(), 121);
    a.state.clear();
    b.state.clear();
    for (label, steps, split, hc) in [
        ("split-off", 32, false, false),
        ("hc-composition", 4, true, true),
    ] {
        r.workspace.restore_checkpoint_for_tests(&checkpoint);
        let x = run(
            &mut r,
            &tokens[PREFIX..PREFIX + steps],
            false,
            split,
            hc,
            true,
            PREFIX,
        );
        r.workspace.restore_checkpoint_for_tests(&checkpoint);
        crate::metal::dispatch_census_begin();
        let y = run(
            &mut r,
            &tokens[PREFIX..PREFIX + steps],
            true,
            split,
            hc,
            true,
            PREFIX,
        );
        let census = crate::metal::dispatch_census_take();
        save(&artifact, &format!("{label}-a"), &x);
        save(&artifact, &format!("{label}-b"), &y);
        witness(&census, steps, split, hc);
        exact(label, &x, &y);
    }
    eprintln!(
        "product_topk numerical PASS empty4/split-on32/split-off32/HC-on4 full logits/hyper/terminal121states bitwise; no research override"
    );
    let mut times = Vec::new();
    for warm in [true, false] {
        for (i, enabled) in [false, true, true, false].into_iter().enumerate() {
            r.workspace.restore_checkpoint_for_tests(&checkpoint);
            let o = run(
                &mut r,
                &tokens[PREFIX..PREFIX + 4],
                enabled,
                true,
                false,
                false,
                PREFIX,
            );
            let name = format!("warm{warm}-{i}");
            save(&artifact, &name, &o);
            let path = artifact.join(format!("{name}-timing.json"));
            let mut evidence = serde_json::json!({"enabled":enabled,"validation":"pending","forwards":o.timing.iter().map(|t|serde_json::json!({"gpu_ms":t.gpu_ms,"wall_ms":t.total_wall_ms})).collect::<Vec<_>>()});
            std::fs::write(&path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
            exact("timed own arm", if enabled { &b } else { &a }, &o);
            let gpu: f64 = o.timing.iter().map(|t| t.gpu_ms.unwrap()).sum();
            let wall: f64 = o.timing.iter().map(|t| t.total_wall_ms).sum();
            assert!(gpu.is_finite() && gpu > 0.0 && wall.is_finite() && wall > 0.0);
            evidence["validation"] = serde_json::json!("bitwise_passed");
            std::fs::write(&path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
            if !warm {
                times.push((gpu, wall));
            }
        }
    }
    let mut results = Vec::new();
    for (axis, floor) in [(0, 0.10), (1, 0.05)] {
        let v: Vec<f64> = times
            .iter()
            .map(|t| if axis == 0 { t.0 } else { t.1 })
            .collect();
        let a = (v[0] + v[3]) * 0.5;
        let b = (v[1] + v[2]) * 0.5;
        let spread = (v[0] - v[3]).abs() / a;
        let useful =
            b <= a * (1.0 - floor) && v[1] <= v[0] * (1.0 - floor) && v[2] <= v[3] * (1.0 - floor);
        let row = serde_json::json!({"axis":if axis==0{"gpu"}else{"executor_wall"},"four_forward_abba_ms":v,"saved_fraction":1.0-b/a,"spread":spread,"verdict":if spread>0.05{"INCONCLUSIVE"}else if useful{"USEFUL"}else{"HOLD"}});
        eprintln!("product_topk {row}");
        results.push(row);
    }
    std::fs::write(
        artifact.join("result.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"numerics":"bitwise","actual_product":true,"results":results}),
        )
        .unwrap(),
    )
    .unwrap();
}
