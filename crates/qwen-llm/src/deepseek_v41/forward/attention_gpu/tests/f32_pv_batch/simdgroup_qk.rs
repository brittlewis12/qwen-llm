//! Test-only per-simdgroup QK versus the current Q-register score kernel.
use super::pv_dim2::{CancellationGuard, median};
use super::*;

const LAUNCHES: usize = 32;
const BLOCKS: usize = 3;
const POISON: f32 = f32::from_bits(0x7fc1_4b51);

fn encode(f: &Fixture, ctx: &MetalContext, enc: &KernelEncoder, candidate: bool, full: bool) {
    if candidate {
        f.bind(enc);
        enc.set_tensor(2, &f.query_half);
        tensor_pipeline(ctx, enc, "score_simdgroup_qk").unwrap();
        grid(enc, f.k.div_ceil(64), f.n, 1, 128);
    } else {
        super::q_register_cache::score(f, ctx, enc, true);
    }
    if full {
        pipeline(ctx, enc, "softmax_rows").unwrap();
        grid(enc, H, f.n, 1, 32);
        tensor_pipeline(ctx, enc, "pv_tensor64").unwrap();
        grid(enc, 1, f.n, 1, 128);
        let pso = pipeline(ctx, enc, "normalize_rows").unwrap();
        dispatch(enc, &pso, f.n * H * M);
    }
}

fn reset(f: &Fixture) {
    for tensor in [&f.scores, &f.output, &f.values, &f.probabilities] {
        let elements = f32_slice(tensor).unwrap().len();
        // SAFETY: shared fixture views, all prior commands terminal, no live slices.
        unsafe {
            std::slice::from_raw_parts_mut(
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize)
                    .cast::<f32>(),
                elements,
            )
            .fill(POISON);
        }
    }
    // SAFETY: aligned shared one-u32 buffer, no command in flight.
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
}

#[derive(Default, Debug)]
struct Errors {
    finite: usize,
    nonfinite: usize,
    changed: usize,
    max_abs: f64,
    max_rel: f64,
    max_ulp: u32,
    abs_sum: f64,
    squared_sum: f64,
}

impl Errors {
    fn observe(&mut self, a: f32, b: f32) {
        assert_ne!(a.to_bits(), POISON.to_bits(), "missing baseline store");
        assert_ne!(b.to_bits(), POISON.to_bits(), "missing candidate store");
        assert_eq!(a.is_finite(), b.is_finite(), "new nonfinite classification");
        if !a.is_finite() {
            self.nonfinite += 1;
            assert_eq!(a.to_bits(), b.to_bits(), "nonfinite payload/sign changed");
            return;
        }
        self.finite += 1;
        self.changed += usize::from(a.to_bits() != b.to_bits());
        let error = (f64::from(a) - f64::from(b)).abs();
        self.max_abs = self.max_abs.max(error);
        self.max_rel = self
            .max_rel
            .max(error / f64::from(a).abs().max(f64::MIN_POSITIVE));
        let ordered = |x: f32| {
            let bits = x.to_bits();
            if bits & 0x8000_0000 != 0 {
                !bits
            } else {
                bits | 0x8000_0000
            }
        };
        self.max_ulp = self.max_ulp.max(ordered(a).abs_diff(ordered(b)));
        self.abs_sum += error;
        self.squared_sum += error * error;
    }

    fn report(&self, label: &str) {
        let divisor = self.finite.max(1) as f64;
        eprintln!(
            "simdgroup_qk_errors tensor={label} finite={} nonfinite={} changed={} max_abs={:.12e} max_rel={:.12e} max_ulp={} mean_abs={:.12e} rms={:.12e}",
            self.finite,
            self.nonfinite,
            self.changed,
            self.max_abs,
            self.max_rel,
            self.max_ulp,
            self.abs_sum / divisor,
            (self.squared_sum / divisor).sqrt()
        );
    }
}

fn untouched(f: &Fixture, full: bool) {
    for tensor in [&f.values, &f.probabilities] {
        assert!(
            f32_slice(tensor)
                .unwrap()
                .iter()
                .all(|v| v.to_bits() == POISON.to_bits())
        );
    }
    let scores = f32_slice(&f.scores).unwrap();
    for q in 0..f.n {
        for h in 0..H {
            let row = (q * H + h) * SCORE_STRIDE;
            let end = SCORE_STRIDE - usize::from(full);
            assert!(
                scores[row + f.count(q)..row + end]
                    .iter()
                    .all(|v| v.to_bits() == POISON.to_bits()),
                "inactive score store"
            );
        }
    }
    if !full {
        assert!(
            f32_slice(&f.output)
                .unwrap()
                .iter()
                .all(|v| v.to_bits() == POISON.to_bits())
        );
    }
    f.check_guards();
}

fn check(f: &Fixture, ctx: &MetalContext, full: bool, label: &str) -> (usize, usize) {
    reset(f);
    execute(ctx, |enc| encode(f, ctx, enc, false, full));
    untouched(f, full);
    let baseline = f32_slice(&f.scores).unwrap().to_vec();
    let output = if full {
        f32_slice(&f.output).unwrap().to_vec()
    } else {
        Vec::new()
    };
    let status = ids(&f.status, 1);
    reset(f);
    execute(ctx, |enc| encode(f, ctx, enc, true, full));
    untouched(f, full);
    let actual = f32_slice(&f.scores).unwrap();
    let mut scores = Errors::default();
    let mut denominators = Errors::default();
    for q in 0..f.n {
        for h in 0..H {
            let row = (q * H + h) * SCORE_STRIDE;
            for i in 0..f.count(q) {
                scores.observe(baseline[row + i], actual[row + i]);
            }
            if full {
                let i = row + SCORE_STRIDE - 1;
                denominators.observe(baseline[i], actual[i]);
            }
        }
    }
    let mut outputs = Errors::default();
    if full {
        for (&a, &b) in output.iter().zip(f32_slice(&f.output).unwrap()) {
            outputs.observe(a, b);
        }
    }
    assert_eq!(ids(&f.status, 1), status, "status semantics changed");
    let score_count = (0..f.n).map(|q| H * f.count(q)).sum::<usize>();
    assert_eq!(scores.finite + scores.nonfinite, score_count);
    assert_eq!(outputs.finite + outputs.nonfinite, output.len());
    eprintln!(
        "simdgroup_qk_check label={label} K={} n={} count={}..{} first={} scope={} active_scores={score_count} outputs={} status={status:?} finite_bits_exact={} unpromoted=true",
        f.k,
        f.n,
        f.count(0),
        f.count(f.n - 1),
        f.first,
        if full { "FULLSINK" } else { "SCORE" },
        output.len(),
        scores.changed + denominators.changed + outputs.changed == 0
    );
    scores.report("scores");
    if full {
        denominators.report("denominators");
        outputs.report("output");
    }
    (score_count, output.len())
}

fn edge_checks(ctx: &MetalContext) {
    for (k, n, ragged, first) in cases() {
        let f = Fixture::new(ctx, k, n, ragged, first);
        check(&f, ctx, false, "small_or_tail");
        check(&f, ctx, true, "small_or_tail");
    }
    for bad in [u32::MAX, 0] {
        let mut f = Fixture::new(ctx, 512, 8, true, 0);
        let mut selected: Vec<u32> = ids(&f.selected, f.topk * f.n)
            .into_iter()
            .map(|v| v as u32)
            .collect();
        selected[f.topk] = if bad == 0 {
            ((f.start + 2) / f.ratio) as u32
        } else {
            bad
        };
        f.selected = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&selected),
            vec![f.topk as u64, f.n as u64],
            GgmlType::I32,
        )
        .unwrap();
        check(&f, ctx, false, "bad_id");
        check(&f, ctx, true, "bad_id");
        assert_eq!(ids(&f.status, 1), [2]);
    }
    let mut f = Fixture::new(ctx, 512, 8, true, 0);
    let mut q: Vec<u16> = f32_slice(&f.query)
        .unwrap()
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    q[H * M] = half::f16::NAN.to_bits(); // Query 1, head 0 only.
    f.query_half = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&q),
        vec![(H * M) as u64, f.n as u64],
        GgmlType::F16,
    )
    .unwrap();
    check(&f, ctx, false, "head0_nan");
    check(&f, ctx, true, "head0_nan");
    assert_eq!(ids(&f.status, 1), [4]);
}

fn timing(
    f: &Fixture,
    ctx: &MetalContext,
    cancellation: &CancellationGuard,
    full: bool,
    ragged: bool,
) {
    let scope = if full { "FULLSINK" } else { "SCORE" };
    let run = |arm: usize| {
        cancellation.check();
        assert!(!metal::dispatch_census_is_active());
        let wall = std::time::Instant::now();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        for _ in 0..LAUNCHES {
            encode(f, ctx, &enc, arm == 1, full);
        }
        enc.end();
        cancellation.check();
        metal::commit_and_wait(&command).unwrap();
        let wall_us = wall.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64;
        cancellation.check();
        let start = command.GPUStartTime();
        let end = command.GPUEndTime();
        assert!(start.is_finite() && end.is_finite() && start > 0.0 && end > start);
        ((end - start) * 1e6 / LAUNCHES as f64, wall_us, start, end)
    };
    // No tensor readbacks, poisoning, or programmed cooldown in this warm/timed sequence.
    let mut samples = [Vec::new(), Vec::new()];
    for block in 0..BLOCKS + 2 {
        for (slot, arm) in [0, 1, 1, 0].into_iter().enumerate() {
            let (gpu, wall, start, end) = run(arm);
            let warm = block < 2;
            if !warm {
                samples[arm].push((gpu, wall));
            }
            eprintln!(
                "simdgroup_qk_sample K={} n={} ragged={ragged} scope={scope} warm={warm} block={block} slot={slot} arm={arm} launches={LAUNCHES} gpu_start_s={start:.12} gpu_end_s={end:.12} gpu_us={gpu:.6} wall_us={wall:.6}",
                f.k, f.n
            );
        }
    }
    let a = median(&samples[0], false);
    let b = median(&samples[1], false);
    eprintln!(
        "simdgroup_qk_median K={} n={} ragged={ragged} scope={scope} blocks={BLOCKS} launches={LAUNCHES} qregister_gpu_us={a:.6} candidate_gpu_us={b:.6} speedup={:.6} qregister_wall_us={:.6} candidate_wall_us={:.6}",
        f.k,
        f.n,
        a / b,
        median(&samples[0], true),
        median(&samples[1], true)
    );
    f.check_guards();
    assert_eq!(ids(&f.status, 1), [0]);
}

#[test]
#[ignore = "resident per-simdgroup QK probe; parent GPU queue only; no debug or shader validation"]
fn attention_simdgroup_qk_probe() {
    for name in [
        "MTL_DEBUG_LAYER",
        "MTL_SHADER_VALIDATION",
        "QWEN_REQUIRE_METAL_TESTS",
    ] {
        assert!(std::env::var_os(name).is_none(), "unset {name} for timing");
    }
    let cancellation = CancellationGuard::install();
    let _lease = metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().expect("simdgroup QK probe requires Metal");
    assert!(
        metal::moe_grouped_tensor_available(&ctx),
        "tensor tier required"
    );
    eprintln!(
        "simdgroup_qk: A=current score_tensor64_qregister B=test-only score_simdgroup_qk; F16 Q mirror, native F32 K rounded on staging; H16/K64/DK64 per SG, four independent SGs, 128 threads, 16KiB shared; SCORE or unchanged softmax+pv_tensor64+normalize FULLSINK; all finite errors retained, nonfinite payload/status exact; no promotion or model-speed claim"
    );
    edge_checks(&ctx);
    let mut totals = (0, 0);
    for (k, n, ragged, first) in production_cases() {
        cancellation.check();
        let f = Fixture::new(&ctx, k, n, ragged, first);
        let label = if ragged { "ragged" } else { "dense" };
        let (scores, _) = check(&f, &ctx, false, label);
        let (_, outputs) = check(&f, &ctx, true, label);
        totals.0 += scores;
        totals.1 += outputs;
        assert_eq!(ids(&f.status, 1), [0]);
        for full in [false, true] {
            timing(&f, &ctx, &cancellation, full, ragged);
        }
    }
    eprintln!(
        "simdgroup_qk_production_totals active_scores={} outputs={} (derived from fixture)",
        totals.0, totals.1
    );
    cancellation.check();
}
