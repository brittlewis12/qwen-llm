use super::*;
use crate::qwen4exp_moe::singleton_observe::{Capture, profile, with_capture};

#[test]
#[ignore = "serial production lease; finite N512/K10 parallel selector screen, no production routing change"]
fn saved_moe_parallel_topk_screen() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    let source = parent.join("qwen4exp-moe-observe-905");
    assert!(source.is_dir());
    let artifact = parent.join(format!("qwen4exp-topk-screen-{}", std::process::id()));
    std::fs::create_dir(&artifact).unwrap();
    eprintln!("topk_screen artifacts={}", artifact.display());
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, 1).unwrap();
    let loaded = Qwen4ExpLoadedModel::load_with_decode_options(
        &ctx,
        &gguf,
        capacity,
        None,
        Qwen4ExpDecodeOptions::default(),
    )
    .unwrap();
    crate::qwen4exp_moe::singleton_observe::topk_screen::screen(
        &ctx,
        &loaded.weights,
        &source,
        &artifact,
    );
}

#[test]
#[ignore = "serial production lease; versioned saved native layer2 inclusive-interval budget"]
fn saved_moe_interval_budget_v2() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    let source = parent.join("qwen4exp-moe-observe-905");
    assert!(source.is_dir(), "requires preserved observation01 captures");
    let artifact = parent.join(format!("qwen4exp-moe-interval-v2-{}", std::process::id()));
    std::fs::create_dir(&artifact).unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, 1).unwrap();
    let loaded = Qwen4ExpLoadedModel::load_with_decode_options(
        &ctx,
        &gguf,
        capacity,
        None,
        Qwen4ExpDecodeOptions::default(),
    )
    .unwrap();
    crate::qwen4exp_moe::singleton_observe::intervals::observe_saved_layer2(
        &ctx,
        &loaded.weights,
        &source,
        &artifact,
    );
}

#[test]
#[ignore = "serial production lease; saved native layer2 timestamp classification, no prefix or performance retry"]
fn saved_moe_stage_interval_diagnostic() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    let source = parent.join("qwen4exp-moe-observe-905");
    assert!(source.is_dir(), "requires preserved observation01 captures");
    let artifact = parent.join(format!(
        "saved-moe-interval-diagnostic-{}.json",
        std::process::id()
    ));
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, 1).unwrap();
    let loaded = Qwen4ExpLoadedModel::load_with_decode_options(
        &ctx,
        &gguf,
        capacity,
        None,
        Qwen4ExpDecodeOptions::default(),
    )
    .unwrap();
    crate::qwen4exp_moe::singleton_observe::captured_interval_diagnostic(
        &ctx,
        &loaded.weights,
        &source,
        &artifact,
    );
}

#[test]
#[ignore = "serial production lease; bounded native MoE capture and isolated complete-path observation"]
fn native_moe_complete_path_observation() {
    native_moe_observation(false);
}

#[test]
#[ignore = "serial production lease; guarded routing baseline across three native MoE dtype combinations"]
fn native_guarded_moe_budget() {
    native_moe_observation(true);
}

fn native_moe_observation(guarded: bool) {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    std::fs::create_dir_all(&parent).unwrap();
    let artifact = parent.join(format!(
        "qwen4exp-moe-observe-guarded{guarded}-{}",
        std::process::id()
    ));
    std::fs::create_dir(&artifact).unwrap();
    eprintln!("moe_observe artifacts={}", artifact.display());
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
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, PREFIX + 1).unwrap();
    let mut loaded = Qwen4ExpLoadedModel::load_with_decode_options(
        &ctx,
        &gguf,
        capacity,
        Some(PREFIX),
        Qwen4ExpDecodeOptions {
            guarded_topk: guarded,
            split_qsa: true,
            hc_up_mix: false,
        },
    )
    .unwrap();
    let captures = [2, 4, 5]
        .into_iter()
        .map(|layer| Capture::new(&ctx, &loaded.weights, layer))
        .collect();
    let mut runner = loaded.create_runner(&ctx).unwrap();
    assert!(!runner.workspace.hc_up_mix_enabled());
    let _selected = Qwen4ExpPackedSelectedQsaOverride::set(true);
    runner.prefill(&tokens[..PREFIX]).unwrap();
    let checkpoint = runner.workspace.checkpoint_for_tests();
    let baseline = run_product(&mut runner, &tokens[PREFIX..PREFIX + 1], true, true);
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    crate::metal::dispatch_census_begin();
    let (observed, captures) = with_capture(captures, || {
        run_product(&mut runner, &tokens[PREFIX..PREFIX + 1], true, true)
    });
    let census = crate::metal::dispatch_census_take();
    for (name, observation) in [("ordinary", &baseline), ("observed", &observed)] {
        for (label, values) in [
            ("logits", &observation.logits[0]),
            ("hyper", &observation.hyper[0]),
        ] {
            std::fs::write(
                artifact.join(format!("{name}-{label}.f32le")),
                bytemuck::cast_slice(values),
            )
            .unwrap();
        }
    }
    if guarded {
        assert_eq!(
            census
                .iter()
                .filter(|r| r.kernel == crate::qwen4exp_moe::guarded_topk::KERNEL)
                .count(),
            48
        );
        assert!(
            !census
                .iter()
                .any(|r| r.kernel == "kernel_topk_logits_softmax_f32")
        );
    }
    assert_replay("MoE capture preserves native forward", &baseline, &observed);
    for observation in [&baseline, &observed] {
        assert!(
            observation
                .logits
                .iter()
                .chain(&observation.hyper)
                .flatten()
                .all(|v| v.is_finite())
        );
    }
    assert_eq!(baseline.state.len(), 121);
    assert_eq!(
        census
            .iter()
            .filter(|r| r.kernel == "kernel_qwen4exp_qsa_split_f16")
            .count(),
        12
    );
    assert!(
        !census
            .iter()
            .any(|r| r.kernel == "kernel_qwen4exp_hc_up_mix_q8_k320")
    );
    eprintln!("moe_observe native logits/hyper/121states bitwise PASS position={PREFIX}");
    drop(baseline);
    drop(observed);
    drop(checkpoint);
    if guarded {
        crate::qwen4exp_moe::singleton_observe::intervals::observe_guarded_captures(
            &ctx, &captures, &census, &artifact,
        );
    } else {
        profile(&ctx, &captures, &census, &artifact);
    }
}
