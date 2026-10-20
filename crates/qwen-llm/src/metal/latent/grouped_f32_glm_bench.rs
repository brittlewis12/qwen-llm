//! Resident shared-operator probes for GLM's two F32 latent projection shapes.
use super::*;

fn shapes() -> impl Iterator<Item = Shape> {
    [(256, 512), (512, 256)].into_iter().flat_map(|(k, m)| {
        [64, 128, 256, 512].into_iter().map(move |rows| Shape {
            k,
            m,
            groups: 64,
            rows,
        })
    })
}

fn fixture(ctx: &MetalContext, shape: Shape) -> Fixture {
    // Production-like alignment, with distinct signed F32 inputs per token/head.
    Fixture::with_prefix(
        ctx,
        shape,
        1.0,
        false,
        256,
        256,
        GroupedF32TestVariant::Production,
    )
}

#[test]
fn grouped_f32_glm_shapes_correctness() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for s in shapes() {
        let f = fixture(&ctx, s);
        let weight_before = tensor_backing_bytes(&f.weight);
        let input_before = tensor_backing_bytes(&f.input);
        f.checked_dispatch(&ctx, true);
        f.checked_dispatch(&ctx, false);
        f.check_values();
        assert_eq!(tensor_backing_bytes(&f.weight), weight_before);
        assert_eq!(tensor_backing_bytes(&f.input), input_before);
        eprintln!("GLM F32 correctness {s:?}: F64/scalar/census/guards passed (not performance)");
    }
}

#[test]
#[ignore = "resident GPU timing; unset MTL_DEBUG_LAYER and QWEN_REQUIRE_METAL_TESTS"]
fn grouped_f32_glm_shapes_performance() {
    for name in ["MTL_DEBUG_LAYER", "QWEN_REQUIRE_METAL_TESTS"] {
        assert!(
            std::env::var_os(name).is_none(),
            "{name} must be unset for performance"
        );
    }
    let ctx = metal_test_context().expect("explicit timing requires Metal");
    const REPEATS: usize = 3;
    const LAUNCHES: usize = 4;
    eprintln!(
        "GLM F32 resident synthetic shared operator only: groups=64, signed F32 operands, 256-byte aligned guarded buffers. Both arms use identical W/input/output bindings. GPU command and encode+commit+wait wall us per launch; excludes allocation, compilation, census, readback, oracle, attention and model execution. All ABBA samples retained, no outlier filtering."
    );
    for s in shapes() {
        let f = fixture(&ctx, s);
        f.checked_dispatch(&ctx, true);
        f.checked_dispatch(&ctx, false);
        f.check_values();
        let timed = |variant| {
            assert!(!dispatch_census_is_active());
            let wall = std::time::Instant::now();
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            for _ in 0..LAUNCHES {
                encode_mat_mat_grouped_f32_for_test(
                    &ctx, &enc, &f.weight, &f.input, &f.output, s.k, s.m, s.groups, s.rows, variant,
                )
                .unwrap();
            }
            enc.end();
            commit_and_wait(&command).unwrap();
            let wall = wall.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64;
            let start = command.GPUStartTime();
            let end = command.GPUEndTime();
            assert!(
                start.is_finite() && end.is_finite() && start > 0.0 && end > start,
                "GPU timestamps unavailable"
            );
            ((end - start) * 1e6 / LAUNCHES as f64, wall)
        };
        use GroupedF32TestVariant::{Production, Scalar};
        timed(Scalar);
        timed(Production);
        let mut samples = [Vec::new(), Vec::new()];
        for repeat in 0..REPEATS {
            for (slot, (arm, variant)) in
                [(0, Scalar), (1, Production), (1, Production), (0, Scalar)]
                    .into_iter()
                    .enumerate()
            {
                let (gpu, wall) = timed(variant);
                samples[arm].push((gpu, wall));
                eprintln!(
                    "glm_f32 sample K={} M={} groups={} rows={} repeat={repeat} slot={slot} arm={variant:?} gpu_us={gpu:.3} wall_us={wall:.3}",
                    s.k, s.m, s.groups, s.rows
                );
            }
        }
        let median = |arm: usize, wall: bool| {
            let mut times: Vec<f64> = samples[arm]
                .iter()
                .map(|&(g, w)| if wall { w } else { g })
                .collect();
            times.sort_by(f64::total_cmp);
            (times[REPEATS - 1] + times[REPEATS]) * 0.5
        };
        let scalar = median(0, false);
        let production = median(1, false);
        eprintln!(
            "glm_f32 median K={} M={} groups={} rows={} ABBA_repeats={REPEATS} launches_per_command={LAUNCHES} scalar_gpu_us={scalar:.3} production_gpu_us={production:.3} speedup={:.3} scalar_wall_us={:.3} production_wall_us={:.3}",
            s.k,
            s.m,
            s.groups,
            s.rows,
            scalar / production,
            median(0, true),
            median(1, true)
        );
        // Timed arms share output; restore the production result for the oracle.
        f.checked_dispatch(&ctx, false);
        f.check_values();
        assert_offset_guards(&f.weight, f.prefix, f.suffix);
        assert_offset_guards(&f.input, f.prefix, f.suffix);
    }
}
