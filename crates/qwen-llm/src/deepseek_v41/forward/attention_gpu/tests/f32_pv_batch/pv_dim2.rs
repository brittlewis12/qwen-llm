//! Test-only PV dimension split; native K/V remain F32 in both arms.
use super::*;
use std::sync::atomic::{AtomicI32, Ordering};

const REPEATS: usize = 3;
const LAUNCHES: usize = 4;
static CANCELLED_SIGNAL: AtomicI32 = AtomicI32::new(0);

extern "C" fn cancel(signal: libc::c_int) {
    let _ = CANCELLED_SIGNAL.compare_exchange(0, signal, Ordering::SeqCst, Ordering::SeqCst);
}

pub(super) struct CancellationGuard(Vec<(libc::c_int, libc::sigaction)>);

impl CancellationGuard {
    pub(super) fn install() -> Self {
        let mut guard = Self(Vec::new());
        for signal in [libc::SIGINT, libc::SIGTERM] {
            // SAFETY: initialized sigactions, static C handler, valid mask and
            // saved-action pointers. Only the explicit single-threaded probe installs these.
            unsafe {
                let mut action: libc::sigaction = std::mem::zeroed();
                let mut previous: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = cancel as *const () as libc::sighandler_t;
                assert_eq!(libc::sigemptyset(&mut action.sa_mask), 0);
                assert_eq!(
                    libc::sigaction(signal, &action, &mut previous),
                    0,
                    "install cancellation handler: {}",
                    std::io::Error::last_os_error()
                );
                guard.0.push((signal, previous));
            }
        }
        guard.check();
        guard
    }

    pub(super) fn check(&self) {
        assert_eq!(
            CANCELLED_SIGNAL.load(Ordering::SeqCst),
            0,
            "attention probe cancelled; all submitted commands have drained"
        );
    }
}

impl Drop for CancellationGuard {
    fn drop(&mut self) {
        for (signal, previous) in self.0.iter().rev() {
            // SAFETY: previous is the action saved for this signal at installation.
            if unsafe { libc::sigaction(*signal, previous, std::ptr::null_mut()) } != 0 {
                eprintln!(
                    "restore signal {signal}: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
    }
}

fn prepare_qhalf(f: &Fixture, ctx: &MetalContext, enc: &KernelEncoder) {
    f.bind(enc);
    enc.set_tensor(2, &f.query_half);
    tensor_pipeline(ctx, enc, "score_tensor64_qhalf").unwrap();
    grid(enc, 1, f.n, 1, 128);
    pipeline(ctx, enc, "softmax_rows").unwrap();
    grid(enc, H, f.n, 1, 32);
}

fn encode(f: &Fixture, ctx: &MetalContext, enc: &KernelEncoder, dim2: bool, scope: Scope) {
    if scope == Scope::Sink {
        prepare_qhalf(f, ctx, enc);
    } else {
        assert_eq!(scope, Scope::Pv);
        f.bind(enc);
    }
    tensor_pipeline(
        ctx,
        enc,
        if dim2 {
            "pv_tensor64_dim2"
        } else {
            "pv_tensor64"
        },
    )
    .unwrap();
    grid(enc, if dim2 { 2 } else { 1 }, f.n, 1, 128);
    if scope == Scope::Sink {
        let pso = pipeline(ctx, enc, "normalize_rows").unwrap();
        dispatch(enc, &pso, f.n * H * M);
    }
}

fn compare_output(f: &Fixture, baseline: &[f32], baseline_status: usize, scope: Scope) -> usize {
    let output = f32_slice(&f.output).unwrap();
    assert_eq!(baseline.len(), output.len());
    let mut nonfinite = 0;
    for (i, (&a, &b)) in baseline.iter().zip(output).enumerate() {
        assert_eq!(
            a.is_finite(),
            b.is_finite(),
            "{scope:?} finiteness lane={i}"
        );
        if a.is_finite() {
            assert_eq!(a.to_bits(), b.to_bits(), "{scope:?} finite bits lane={i}");
        } else {
            nonfinite += 1;
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{scope:?} nonfinite payload lane={i}"
            );
        }
    }
    assert!(nonfinite < baseline.len(), "no finite outputs compared");
    // Preserve the baseline's known sink edge without waiving any new bit.
    assert_eq!(ids(&f.status, 1), [baseline_status]);
    f.check_guards();
    nonfinite
}

fn median(samples: &[(f64, f64)], wall: bool) -> f64 {
    let mut values: Vec<_> = samples
        .iter()
        .map(|&(g, w)| if wall { w } else { g })
        .collect();
    values.sort_by(f64::total_cmp);
    assert_eq!(values.len(), 2 * REPEATS);
    (values[REPEATS - 1] + values[REPEATS]) * 0.5
}

#[test]
#[ignore = "resident GPU timing; parent measurement queue only; unset MTL_DEBUG_LAYER and QWEN_REQUIRE_METAL_TESTS"]
fn pv_tensor64_dim2_probe() {
    for name in ["MTL_DEBUG_LAYER", "QWEN_REQUIRE_METAL_TESTS"] {
        assert!(
            std::env::var_os(name).is_none(),
            "{name} must be unset for performance"
        );
    }
    let cancellation = CancellationGuard::install();
    let ctx = MetalContext::new().expect("explicit timing requires Metal");
    cancellation.check();
    assert!(
        metal::moe_grouped_tensor_available(&ctx),
        "Metal tensor ops required"
    );
    eprintln!(
        "pv_dim2: K640 N512 H64 D512; A=pv_tensor64 B=pv_tensor64_dim2; native K/V F32; PV uses fixed Qhalf+softmax scores without normalize; Sink=Qhalf score+softmax+PV+normalize; GPU/wall us per launch; compilation/allocation/readback excluded; no speedup assumption"
    );
    for ragged in [false, true] {
        cancellation.check();
        let f = Fixture::new(&ctx, 640, 512, ragged, 0);
        cancellation.check();
        execute(&ctx, |enc| prepare_qhalf(&f, &ctx, enc));
        cancellation.check();
        f.check_guards();

        for scope in [Scope::Pv, Scope::Sink] {
            let run = |dim2: bool, timed: bool| {
                cancellation.check();
                assert!(!metal::dispatch_census_is_active());
                // SAFETY: shared, aligned one-u32 fixture buffer; every prior
                // command is terminal and no command currently references it.
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
                    encode(&f, &ctx, &enc, dim2, scope);
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

            // Warm compilation and both arms outside the measured samples.
            run(false, false);
            let baseline = f32_slice(&f.output).unwrap().to_vec();
            let baseline_status = ids(&f.status, 1)[0];
            let nonfinite = compare_output(&f, &baseline, baseline_status, scope);
            // SAFETY: shared fixture output, all commands terminal, and no
            // borrowed output slice remains. Poison only the untimed check so
            // missing candidate stores cannot inherit finite baseline lanes.
            unsafe {
                std::slice::from_raw_parts_mut(
                    f.output
                        .buffer
                        .contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(f.output.offset as usize)
                        .cast::<f32>(),
                    baseline.len(),
                )
                .fill(f32::NAN);
            }
            run(true, false);
            assert_eq!(
                compare_output(&f, &baseline, baseline_status, scope),
                nonfinite
            );
            eprintln!(
                "pv_dim2_check ragged={ragged} scope={scope:?} finite_bits_exact=true nonfinite_payloads_exact=true baseline_nonfinite={nonfinite} baseline_status={baseline_status} status={:?}",
                ids(&f.status, 1)
            );

            let mut samples = [Vec::new(), Vec::new()];
            for repeat in 0..REPEATS {
                for (slot, arm) in [0, 1, 1, 0].into_iter().enumerate() {
                    let (gpu, wall) = run(arm == 1, true);
                    samples[arm].push((gpu, wall));
                    eprintln!(
                        "pv_dim2_sample K=640 n=512 ragged={ragged} scope={scope:?} repeat={repeat} slot={slot} arm={} launches={LAUNCHES} gpu_us={gpu:.6} wall_us={wall:.6}",
                        if arm == 0 { "A" } else { "B" }
                    );
                    compare_output(&f, &baseline, baseline_status, scope);
                }
            }
            let a = median(&samples[0], false);
            let b = median(&samples[1], false);
            eprintln!(
                "pv_dim2_median K=640 n=512 ragged={ragged} scope={scope:?} ABBA_repeats={REPEATS} launches={LAUNCHES} baseline_gpu_us={a:.6} dim2_gpu_us={b:.6} speedup={:.6} baseline_wall_us={:.6} dim2_wall_us={:.6}",
                a / b,
                median(&samples[0], true),
                median(&samples[1], true)
            );
        }
    }
    cancellation.check();
}
