//! One D128 PV chunk per group, with Qregister scores in both comparison arms.
use super::pv_dim2::{CancellationGuard, compare_output, median};
use super::q_register_cache::{poison, score};
use super::*;

const REPEATS: usize = 3;
const LAUNCHES: usize = 32;

fn prepare(f: &Fixture, ctx: &MetalContext, enc: &KernelEncoder) {
    score(f, ctx, enc, true);
    pipeline(ctx, enc, "softmax_rows").unwrap();
    grid(enc, H, f.n, 1, 32);
}

fn encode(f: &Fixture, ctx: &MetalContext, enc: &KernelEncoder, dim1: bool, scope: Scope) {
    if scope == Scope::Sink {
        prepare(f, ctx, enc);
    } else {
        assert_eq!(scope, Scope::Pv);
        f.bind(enc);
    }
    tensor_pipeline(
        ctx,
        enc,
        if dim1 {
            "pv_tensor64_dim1"
        } else {
            "pv_tensor64"
        },
    )
    .unwrap();
    grid(enc, if dim1 { 4 } else { 1 }, f.n, 1, 128);
    if scope == Scope::Sink {
        let pso = pipeline(ctx, enc, "normalize_rows").unwrap();
        dispatch(enc, &pso, f.n * H * M);
    }
}

#[test]
#[ignore = "resident GPU PV dim1 timing; parent measurement queue only; no debug or shader validation"]
fn pv_tensor64_dim1_probe() {
    for name in [
        "MTL_DEBUG_LAYER",
        "MTL_SHADER_VALIDATION",
        "QWEN_REQUIRE_METAL_TESTS",
    ] {
        assert!(std::env::var_os(name).is_none(), "unset {name} for timing");
    }
    let cancellation = CancellationGuard::install();
    let _lease = metal::acquire_metal_benchmark_lease().unwrap();
    cancellation.check();
    let ctx = MetalContext::new().expect("PV dim1 timing requires Metal");
    cancellation.check();
    assert!(
        metal::moe_grouped_tensor_available(&ctx),
        "tensor tier required"
    );
    eprintln!(
        "pv_dim1: K640 N512 H64 D512; A=pv_tensor64 B=pv_tensor64_dim1; one D128 accumulator/group, four groups/query; native K/V F32; Pv=fixed Qregister+softmax scores, no normalize; Sink=Qregister score+softmax+PV+normalize; GPU/wall us per launch; compilation/allocation/readback excluded"
    );
    for (k, n, ragged, first) in production_cases() {
        cancellation.check();
        let f = Fixture::new(&ctx, k, n, ragged, first);
        cancellation.check();
        execute(&ctx, |enc| prepare(&f, &ctx, enc));
        cancellation.check();
        f.check_guards();

        for scope in [Scope::Pv, Scope::Sink] {
            let run = |dim1: bool, timed: bool| {
                cancellation.check();
                assert!(!metal::dispatch_census_is_active());
                // SAFETY: shared one-u32 fixture buffer; all prior commands
                // are terminal, so reset status independently for each arm.
                unsafe {
                    f.status
                        .buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(f.status.offset as usize)
                        .cast::<u32>()
                        .write(0);
                }
                let wall = std::time::Instant::now();
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                for _ in 0..LAUNCHES {
                    encode(&f, &ctx, &enc, dim1, scope);
                }
                enc.end();
                cancellation.check();
                metal::commit_and_wait(&command).unwrap();
                let wall = wall.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64;
                cancellation.check();
                if !timed {
                    return (0.0, 0.0);
                }
                let start = command.GPUStartTime();
                let end = command.GPUEndTime();
                assert!(
                    start.is_finite() && end.is_finite() && start > 0.0 && end > start,
                    "GPU timestamps unavailable"
                );
                ((end - start) * 1e6 / LAUNCHES as f64, wall)
            };

            // Compile and warm both arms before retaining any timing samples.
            run(false, false);
            let baseline = f32_slice(&f.output).unwrap().to_vec();
            let baseline_status = ids(&f.status, 1)[0];
            let nonfinite = compare_output(&f, &baseline, baseline_status, scope);
            poison(&f.output);
            run(true, false);
            assert_eq!(
                compare_output(&f, &baseline, baseline_status, scope),
                nonfinite
            );
            eprintln!(
                "pv_dim1_check K={k} n={n} ragged={ragged} scope={scope:?} output_elements={} finite_bits_exact=true nonfinite_payloads_exact=true baseline_nonfinite={nonfinite} baseline_status={baseline_status} status={:?}",
                baseline.len(),
                ids(&f.status, 1)
            );

            // Readbacks between short commands let the GPU idle and change clocks.
            for _ in 0..2 {
                run(false, false);
                run(true, false);
            }
            let mut samples = [Vec::new(), Vec::new()];
            for repeat in 0..REPEATS {
                for (slot, arm) in [0, 1, 1, 0].into_iter().enumerate() {
                    let (gpu, wall) = run(arm == 1, true);
                    samples[arm].push((gpu, wall));
                    eprintln!(
                        "pv_dim1_sample K={k} n={n} ragged={ragged} scope={scope:?} repeat={repeat} slot={slot} arm={} launches={LAUNCHES} gpu_us={gpu:.6} wall_us={wall:.6}",
                        if arm == 0 { "A" } else { "B" }
                    );
                }
            }
            let a = median(&samples[0], false);
            let b = median(&samples[1], false);
            eprintln!(
                "pv_dim1_median K={k} n={n} ragged={ragged} scope={scope:?} ABBA_repeats={REPEATS} launches={LAUNCHES} baseline_gpu_us={a:.6} dim1_gpu_us={b:.6} speedup={:.6} baseline_wall_us={:.6} dim1_wall_us={:.6}",
                a / b,
                median(&samples[0], true),
                median(&samples[1], true)
            );
            compare_output(&f, &baseline, baseline_status, scope);
            poison(&f.output);
            run(true, false);
            compare_output(&f, &baseline, baseline_status, scope);
        }
    }
    cancellation.check();
}
