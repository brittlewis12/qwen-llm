use super::*;

#[test]
#[ignore = "production lease; CPU-only validation ceiling, real model metadata, no command commit or forward"]
fn scalar_validation_cpu_ceiling() {
    let _lease = crate::metal::acquire_metal_benchmark_lease()
        .expect("production lease and wired gate required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    let artifact = parent.join(format!(
        "qwen4exp-validation-ceiling-{}",
        std::process::id()
    ));
    std::fs::create_dir(&artifact).unwrap();
    eprintln!("validation_ceiling artifacts={}", artifact.display());
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, 2180).unwrap();
    let mut loaded = Qwen4ExpLoadedModel::load(&ctx, &gguf, capacity).unwrap();
    assert!(
        loaded.guarded_topk_enabled()
            && loaded.hc_up_mix_enabled()
            && !loaded.split_decode_enabled()
    );
    let runner = loaded.create_runner(&ctx).unwrap();
    let before = snapshot_persistent_state(&runner);
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    crate::metal::dispatch_census_begin();
    let check = || {
        runner
            .workspace
            .scalar_preflight_only_for_tests(&ctx, &encoder, 42, &runner.weights)
            .unwrap()
    };
    check();
    let census = crate::metal::dispatch_census_take();
    check();
    let times: Vec<_> = (0..32)
        .map(|_| {
            let start = std::time::Instant::now();
            check();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    encoder.end();
    // Deliberately abandon the empty encoder; this packet never commits GPU work.
    drop(command);
    std::fs::write(
        artifact.join("cpu-ms.txt"),
        format!("times_ms={times:?}\ncensus={census:#?}\n"),
    )
    .unwrap();
    assert!(census.is_empty());
    assert_eq!(runner.next_position(), 0);
    assert!(runner.logits().is_err());
    assert_state_bytes_eq(
        "preflight must not mutate state",
        &before,
        &snapshot_persistent_state(&runner),
    );
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let median = (sorted[15] + sorted[16]) * 0.5;
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    eprintln!(
        "validation_ceiling median_ms={median} mean_ms={mean} min_ms={} max_ms={} whole_pass_floor_ms=1 verdict={}; this is not removable static-only work or a speedup",
        sorted[0],
        sorted[31],
        if median >= 1.0 {
            "WORTH_CONTRACT_SPLIT_DESIGN"
        } else {
            "HOLD_NO_OPTIMIZATION"
        }
    );
}
