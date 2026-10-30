//! GLM-local Q8 wide leaf qualification, not a model/default promotion gate.

use super::*;

fn encode(f: &Fixture, ctx: &MetalContext, enc: &KernelEncoder, wide: bool) {
    let _wide = Q8WideScope::new(wide);
    f.encode(ctx, enc, true);
}

fn check_invalid_bindings(f: &Fixture, ctx: &MetalContext) {
    let _tensor = Scope::new(true);
    let _wide = Q8WideScope::new(true);
    let mut readonly = f.y.clone();
    readonly.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
    let mut short = f.y.clone();
    short.shape[1] -= 1;
    let mut alias = f.y.clone();
    alias.buffer = f.x.buffer.clone();
    let mut unaligned = f.x.clone();
    unaligned.offset += 8;
    let mut outside = f.y.clone();
    outside.offset = u64::MAX - 3;
    for (x, y) in [
        (&f.x, &readonly),
        (&f.x, &short),
        (&f.x, &alias),
        (&unaligned, &f.y),
        (&f.x, &outside),
    ] {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        dispatch_census_begin();
        assert!(try_encode(ctx, &enc, HALF, StageMode::Fast, &f.w, x, y, f.k, f.m, f.n).is_err());
        assert!(dispatch_census_take().is_empty());
        assert!(enc.bound_pipeline_name.borrow().is_none());
        enc.end();
    }
    assert!(read(&f.y).iter().all(|v| v.to_bits() == u32::MAX));
    for t in [&f.w, &f.x, &f.y] {
        assert_guards(t);
    }
}

fn qualify(f: &Fixture, ctx: &MetalContext) {
    let mut narrow = Vec::new();
    for wide in [false, true] {
        let tile = if wide { Tile::Q8Wide } else { Tile::Narrow };
        assert!(prepare(ctx, GgmlType::Q8_0, tile).unwrap());
        f.poison_output();
        let census = run(ctx, |enc| encode(f, ctx, enc, wide));
        assert_eq!(census.len(), 1);
        assert_eq!(
            census[0].kernel,
            tile.pipeline_name(GgmlType::Q8_0).unwrap()
        );
        assert_eq!(
            census[0].grid_width,
            f.n.div_ceil(if wide { 64 } else { 32 }) as u64
        );
        assert_eq!(census[0].grid_height, f.m.div_ceil(64) as u64);
        assert_eq!(census[0].grid_depth, 1);
        assert_eq!(census[0].threads_width, 128);
        let result = read(&f.y);
        f.validate(&result, &f.full, 1e-2);
        f.validate(&result, &f.half, 2e-3);
        if !wide {
            narrow = result;
            continue;
        }
        let scale = narrow.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let mut changed = 0usize;
        let mut max_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        for (i, (&a, &b)) in narrow.iter().zip(&result).enumerate() {
            let diff = (f64::from(a) - f64::from(b)).abs();
            assert!(
                diff <= 2e-3 * f64::from(scale) + 1e-5,
                "output {i}: {a} vs {b}"
            );
            changed += usize::from(a.to_bits() != b.to_bits());
            max_abs = max_abs.max(diff);
            sum_sq += diff * diff;
        }
        eprintln!(
            "GLM Q8 K{} M{} N{} all_outputs={} changed_bits={} max_abs={} rms={} full_f64_rel=1e-2 half_f64_rel=2e-3",
            f.k,
            f.m,
            f.n,
            result.len(),
            changed,
            max_abs,
            (sum_sq / result.len() as f64).sqrt()
        );
    }
}

fn timed(f: &Fixture, ctx: &MetalContext, wide: bool) -> (f64, f64) {
    const REPEATS: usize = 4;
    let start = std::time::Instant::now();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    for _ in 0..REPEATS {
        // Includes the actual packed working function, policy, checked bindings,
        // exact-pipeline preparation and dispatch; no leaf-only timing shortcut.
        encode(f, ctx, &enc, wide);
    }
    enc.end();
    command.commit();
    wait_completed(&command).unwrap();
    let wall_ms = start.elapsed().as_secs_f64() * 1e3 / REPEATS as f64;
    let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3 / REPEATS as f64;
    assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
    (gpu_ms, wall_ms)
}

#[test]
#[ignore = "actual M5, production lease, release; MTL_DEBUG_LAYER=1 validation or 0 timing"]
fn m5_glm_q8_wide_leaf_f64_guards_and_continuous_abba() {
    let debug = std::env::var("MTL_DEBUG_LAYER").expect("explicit validation/timing mode");
    assert!(matches!(debug.as_str(), "0" | "1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("explicit production GPU lease");
    let ctx = MetalContext::new().expect("Metal is required, not skipped");
    require_m5(&ctx);
    for (k, m, n) in [
        (4096, 8192, 1024),
        (8192, 4096, 1024),
        (1536, 16384, 1024),
        (4096, 1536, 1024),
        (4096, 8192, 2048),
        (8192, 4096, 2048),
        (1536, 16384, 2048),
        (4096, 1536, 2048),
        (256, 65, 1025),
    ] {
        let f = Fixture::new(&ctx, GgmlType::Q8_0, k, m, n);
        check_invalid_bindings(&f, &ctx);
        qualify(&f, &ctx);
        // Finish all CPU payload access before conditioning and continuous
        // measured ABBA. No poison, readback, census or logging between blocks.
        for wide in [false, true, true, false] {
            timed(&f, &ctx, wide);
        }
        let mut samples = Vec::with_capacity(12);
        for round in 0..3 {
            for (label, wide) in [("A1", false), ("B1", true), ("B2", true), ("A2", false)] {
                samples.push((round, label, timed(&f, &ctx, wide)));
            }
        }
        eprintln!(
            "GLM Q8 wide leaf K{k} M{m} N{n} debug={debug} timing_eligible={} repeats=4 ABBA (gpu_ms,working_function_wall_ms)={samples:?}; not a model promotion gate",
            debug == "0"
        );
        f.validate(&read(&f.y), &f.half, 2e-3);
    }
}
