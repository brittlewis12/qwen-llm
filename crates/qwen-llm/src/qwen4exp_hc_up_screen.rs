use super::*;
use crate::metal::MetalTensorProvenance;
use objc2_metal::MTLCommandQueue;

const HIDDEN: usize = 2560;
const RANK: usize = 320;
const WIDTH: usize = 4 * HIDDEN;
const DATASETS: usize = 12;
const GUARD: usize = 32;

#[path = "qwen4exp_moe_index_probe.rs"]
mod moe_index_probe;

#[path = "qwen4exp_moe_down_index_probe.rs"]
mod moe_down_index_probe;

#[path = "qwen4exp_moe_remaining_index_probe.rs"]
mod moe_remaining_index_probe;

fn bytes(t: &MetalTensor) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast(), t.buffer.length()).to_vec()
    }
}

fn read(t: &MetalTensor) -> Vec<f32> {
    let all = bytes(t);
    bytemuck::cast_slice(&all[t.offset as usize..t.offset as usize + t.n_elements() as usize * 4])
        .to_vec()
}

fn write(t: &MetalTensor, values: &[f32]) {
    assert_eq!(t.n_elements() as usize, values.len());
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize),
            values.len() * 4,
        );
    }
}

fn guarded(ctx: &MetalContext, data: &[u8], shape: Vec<u64>, dtype: GgmlType) -> MetalTensor {
    let mut storage = vec![0xa5; GUARD];
    storage.extend_from_slice(data);
    storage.extend_from_slice(&[0x5a; GUARD]);
    MetalTensor {
        buffer: ctx.buffer_from(&storage).unwrap(),
        offset: GUARD as u64,
        shape,
        dtype,
        provenance: MetalTensorProvenance::OwnedWritable,
    }
}

fn f32_tensor(ctx: &MetalContext, values: &[f32]) -> MetalTensor {
    guarded(
        ctx,
        bytemuck::cast_slice(values),
        vec![values.len() as u64],
        GgmlType::F32,
    )
}

fn guards(t: &MetalTensor) {
    let all = bytes(t);
    assert_eq!(&all[..GUARD], &[0xa5; GUARD]);
    assert_eq!(&all[all.len() - GUARD..], &[0x5a; GUARD]);
}

fn random(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as u32
}

fn sample(state: &mut u64) -> f32 {
    (random(state) >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

fn q8(state: &mut u64, blocks: usize, amplitude: f32) -> Vec<u8> {
    let mut result = Vec::with_capacity(blocks * 34);
    for block in 0..blocks {
        let scale = if block % 101 == 0 {
            0.0
        } else {
            sample(state) * amplitude
        };
        result.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        for _ in 0..32 {
            result.push(random(state) as u8);
        }
    }
    result
}

struct Fixture {
    input: MetalTensor,
    norm: MetalTensor,
    down: MetalTensor,
    up: MetalTensor,
    scratch: GatedResidualMetalScratch,
    frozen: Vec<Vec<u8>>,
}

impl Fixture {
    fn new(ctx: &MetalContext, seed: u64) -> Self {
        let mut state = seed;
        let input = f32_tensor(
            ctx,
            &(0..WIDTH)
                .map(|_| sample(&mut state) * 2.0)
                .collect::<Vec<_>>(),
        );
        let norm = f32_tensor(
            ctx,
            &(0..WIDTH)
                .map(|_| 1.0 + sample(&mut state) * 0.5)
                .collect::<Vec<_>>(),
        );
        let down = guarded(
            ctx,
            &q8(&mut state, WIDTH * RANK / 32, 0.001),
            vec![WIDTH as u64, RANK as u64],
            GgmlType::Q8_0,
        );
        let up = guarded(
            ctx,
            &q8(&mut state, WIDTH * RANK / 32, 0.008),
            vec![RANK as u64, WIDTH as u64],
            GgmlType::Q8_0,
        );
        let scratch = GatedResidualMetalScratch {
            branch_count: 4,
            hidden_size: HIDDEN,
            low_rank: RANK,
            normalized: f32_tensor(ctx, &vec![f32::NAN; WIDTH]),
            low: f32_tensor(ctx, &vec![f32::NAN; RANK]),
            raw_gate: f32_tensor(ctx, &vec![f32::NAN; WIDTH]),
            mixed: f32_tensor(ctx, &vec![f32::NAN; HIDDEN]),
            injection: f32_tensor(ctx, &[0.0; 4]),
            active_command: None,
        };
        let frozen = [&input, &norm, &down, &up].map(bytes).to_vec();
        Self {
            input,
            norm,
            down,
            up,
            scratch,
            frozen,
        }
    }

    fn check(&self) {
        for (tensor, original) in [&self.input, &self.norm, &self.down, &self.up]
            .into_iter()
            .zip(&self.frozen)
        {
            assert_eq!(&bytes(tensor), original, "input/weight storage mutated");
        }
        for tensor in [
            &self.scratch.normalized,
            &self.scratch.low,
            &self.scratch.raw_gate,
            &self.scratch.mixed,
        ] {
            guards(tensor);
            assert!(read(tensor).iter().all(|v| v.is_finite()));
        }
    }

    fn poison(&self, full: bool) {
        write(&self.scratch.raw_gate, &vec![f32::NAN; WIDTH]);
        write(&self.scratch.mixed, &vec![f32::NAN; HIDDEN]);
        if full {
            write(&self.scratch.normalized, &vec![f32::NAN; WIDTH]);
            write(&self.scratch.low, &vec![f32::NAN; RANK]);
        }
    }
}

fn encode(ctx: &MetalContext, enc: &KernelEncoder, fixture: &Fixture, candidate: bool, full: bool) {
    let s = &fixture.scratch;
    if full {
        encode_hc_norm(
            ctx,
            enc,
            &fixture.input,
            &fixture.norm,
            &s.normalized,
            4,
            HIDDEN,
            1e-6,
        )
        .unwrap();
        encode_mat_vec_dispatch(ctx, enc, &fixture.down, &s.normalized, &s.low, WIDTH, RANK)
            .unwrap();
        encode_hc_low_activation(ctx, enc, &s.low, 4).unwrap();
    }
    if candidate {
        let pipeline = ctx.pipeline("kernel_qwen4exp_hc_up_mix_q8_k320").unwrap();
        assert_eq!(pipeline.threadExecutionWidth(), 32);
        assert!(pipeline.maxTotalThreadsPerThreadgroup() >= 128);
        enc.set_pipeline(&pipeline);
        for (index, tensor) in [&fixture.up, &s.low, &s.normalized, &s.raw_gate, &s.mixed]
            .into_iter()
            .enumerate()
        {
            enc.set_tensor(index, tensor);
        }
        enc.dispatch(
            MTLSize {
                width: HIDDEN / 4,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    } else {
        encode_mat_vec_dispatch(ctx, enc, &fixture.up, &s.low, &s.raw_gate, RANK, WIDTH).unwrap();
        encode_hc_gated_mean(ctx, enc, &s.normalized, &s.raw_gate, &s.mixed, 4, HIDDEN).unwrap();
    }
}

fn run(
    ctx: &MetalContext,
    fixtures: &[Fixture],
    candidate: bool,
    full: bool,
    repeats: usize,
) -> (f64, f64) {
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let start = std::time::Instant::now();
    for _ in 0..repeats {
        for fixture in fixtures {
            encode(ctx, &encoder, fixture, candidate, full);
        }
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    let wall = start.elapsed().as_secs_f64() * 1e3 / repeats as f64;
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none(), "{:?}", command.error());
    let gpu = (command.GPUEndTime() - command.GPUStartTime()) * 1e3 / repeats as f64;
    assert!(gpu.is_finite() && gpu > 0.0);
    (gpu, wall)
}

fn oracle(f: &Fixture) -> (Vec<f64>, Vec<f64>) {
    let (gates, mixed, _) = conditioned_oracle(f);
    (gates, mixed)
}

fn conditioned_oracle(f: &Fixture) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let storage = bytes(&f.up);
    let weights = &storage[f.up.offset as usize..];
    let low = read(&f.scratch.low);
    let normalized = read(&f.scratch.normalized);
    let mut gates = vec![0.0; WIDTH];
    let mut sum_abs = vec![0.0; WIDTH];
    for (row, gate) in gates.iter_mut().enumerate() {
        for block in 0..10 {
            let offset = row * 340 + block * 34;
            let scale =
                half::f16::from_bits(u16::from_le_bytes([weights[offset], weights[offset + 1]]))
                    .to_f64();
            for i in 0..32 {
                let term =
                    (weights[offset + 2 + i] as i8) as f64 * scale * low[block * 32 + i] as f64;
                *gate += term;
                sum_abs[row] += term.abs();
            }
        }
    }
    let mixed = (0..HIDDEN)
        .map(|hidden| {
            let mut sum = 0.0;
            for branch in 0..4 {
                let row = branch * HIDDEN + hidden;
                sum += normalized[row] as f64 / (1.0 + (-gates[row]).exp());
            }
            sum / 4.0
        })
        .collect();
    (gates, mixed, sum_abs)
}

fn compare(actual: &[f32], expected: &[f64], label: &str) -> (f64, f64) {
    compare_with_bound(actual, expected, label, |_, b| 3e-5 * (1.0 + b.abs()))
}

fn compare_with_bound(
    actual: &[f32],
    expected: &[f64],
    label: &str,
    bound: impl Fn(usize, f64) -> f64,
) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let (mut error, mut energy, mut max_abs) = (0.0, 0.0, 0.0f64);
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "{label}: nonfinite");
        let delta = (a as f64 - b).abs();
        assert!(
            delta <= bound(index, b),
            "{label}: {a} != {b}, delta={delta}"
        );
        error += delta * delta;
        energy += b * b;
        max_abs = max_abs.max(delta);
    }
    let rms = if energy == 0.0 {
        assert_eq!(error, 0.0, "{label}: zero-energy oracle");
        0.0
    } else {
        (error / energy).sqrt()
    };
    assert!(rms <= 3e-5, "{label}: rms={rms}");
    (max_abs, rms)
}

fn outputs(f: &Fixture) -> (Vec<f32>, Vec<f32>) {
    (read(&f.scratch.raw_gate), read(&f.scratch.mixed))
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "bitwise mismatch at {index}");
    }
}

#[test]
#[ignore = "serial production GPU lease; model-free full-shape HC scheduling screen"]
fn hc_up_k320_screen() {
    screen(false);
}

#[test]
#[ignore = "serial production GPU lease; protocol-2 conditioned HC scheduling screen"]
fn hc_up_k320_screen_v2() {
    screen(true);
}

fn screen(conditioned: bool) {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let ctx = MetalContext::new().expect("real Metal required; no skipped screen");
    assert!(
        crate::metal::mat_vec_q8_0_lcpp_enabled(),
        "screen requires incumbent LCPP route"
    );
    let fixtures: Vec<_> = (0..DATASETS)
        .map(|i| Fixture::new(&ctx, 0x4d595df4d0f33173 + i as u64 * 7919))
        .collect();
    eprintln!(
        "hc_up footprint full_weights_bytes={} up_weights_bytes={}",
        DATASETS * 2 * WIDTH * RANK / 32 * 34,
        DATASETS * WIDTH * RANK / 32 * 34
    );
    run(&ctx, &fixtures, false, true, 1);
    let upstream: Vec<_> = fixtures
        .iter()
        .map(|f| (read(&f.scratch.normalized), read(&f.scratch.low)))
        .collect();
    let expected: Vec<_> = fixtures.iter().map(oracle).collect();
    let baseline: Vec<_> = fixtures.iter().map(outputs).collect();
    for f in &fixtures {
        f.poison(true);
    }
    run(&ctx, &fixtures, true, true, 1);
    let candidate: Vec<_> = fixtures.iter().map(outputs).collect();
    for (i, f) in fixtures.iter().enumerate() {
        assert_bits(&read(&f.scratch.normalized), &upstream[i].0);
        assert_bits(&read(&f.scratch.low), &upstream[i].1);
        for (arm, result) in [("baseline", &baseline[i]), ("candidate", &candidate[i])] {
            let gate = compare(&result.0, &expected[i].0, "raw gate");
            let mixed = compare(&result.1, &expected[i].1, "mixed");
            eprintln!(
                "hc_up numeric dataset={i} arm={arm} raw_max_abs_rms={gate:?} mixed_max_abs_rms={mixed:?}"
            );
        }
        f.check();
    }

    hostile_checks(&ctx, &fixtures, &upstream[0], conditioned);

    for full in [false, true] {
        for arm in [false, true, true, false] {
            run(&ctx, &fixtures, arm, full, 8);
        }
        let mut times = Vec::new();
        for arm in [false, true, true, false] {
            for f in &fixtures {
                f.poison(full);
            }
            times.push(run(&ctx, &fixtures, arm, full, 8));
            for (i, f) in fixtures.iter().enumerate() {
                let observed = outputs(f);
                let expected = if arm { &candidate[i] } else { &baseline[i] };
                assert_bits(&observed.0, &expected.0);
                assert_bits(&observed.1, &expected.1);
                assert_bits(&read(&f.scratch.normalized), &upstream[i].0);
                assert_bits(&read(&f.scratch.low), &upstream[i].1);
            }
        }
        for f in &fixtures {
            f.check();
        }
        let a = (times[0].0 + times[3].0) * 0.5;
        let b = (times[1].0 + times[2].0) * 0.5;
        let spread = (times[0].0 - times[3].0).abs() / a;
        let floor = |a: f64, b: f64| b <= a * 0.9 && (a - b) * 97.0 / DATASETS as f64 >= 0.5;
        let verdict = if spread > 0.05 {
            "INCONCLUSIVE"
        } else if floor(a, b) && floor(times[0].0, times[1].0) && floor(times[3].0, times[2].0) {
            "USEFUL"
        } else {
            "HOLD"
        };
        eprintln!(
            "hc_up timing full={full} gpu_encode_submit_wait_ms_abba={times:?} saved_pct={} control_spread={spread} extrapolated_97_calls_saved_ms={} verdict={verdict}",
            (1.0 - b / a) * 100.0,
            (a - b) * 97.0 / DATASETS as f64
        );
    }
}

fn hostile_checks(
    ctx: &MetalContext,
    fixtures: &[Fixture],
    upstream: &(Vec<f32>, Vec<f32>),
    conditioned: bool,
) {
    let f = &fixtures[0];
    for mode in 0..4 {
        let low: Vec<_> = (0..RANK)
            .map(|i| match mode {
                0 => 0.0,
                1 => upstream.1[i] * 1e-12,
                2 => {
                    if i % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                _ => upstream.1[i] * 256.0,
            })
            .collect();
        write(&f.scratch.low, &low);
        let expected = conditioned_oracle(f);
        for arm in [false, true] {
            f.poison(false);
            run(ctx, &fixtures[..1], arm, false, 1);
            let result = outputs(f);
            let gate = if conditioned && mode == 3 {
                let original_failures = result
                    .0
                    .iter()
                    .zip(&expected.0)
                    .filter(|(a, b)| (**a as f64 - **b).abs() > 3e-5 * (1.0 + b.abs()))
                    .count();
                eprintln!(
                    "hc_up protocol=2 candidate={arm} retained_protocol1_raw_failures={original_failures}"
                );
                let gamma = |m: f64, u: f64| m * u / (1.0 - m * u);
                let gamma64 = gamma(322.0, 2.0f64.powi(-53));
                compare_with_bound(
                    &result.0,
                    &expected.0,
                    "conditioned hostile raw gate",
                    |i, _| {
                        let sum_abs = expected.2[i];
                        assert!(sum_abs.is_finite() && sum_abs >= 0.0);
                        (gamma(45.0, 2.0f64.powi(-24)) + gamma64) * sum_abs / (1.0 - gamma64)
                    },
                )
            } else {
                compare(&result.0, &expected.0, "hostile raw gate")
            };
            let mixed = compare(&result.1, &expected.1, "hostile mixed");
            assert_bits(&read(&f.scratch.low), &low);
            assert_bits(&read(&f.scratch.normalized), &upstream.0);
            f.check();
            eprintln!(
                "hc_up hostile mode={mode} candidate={arm} raw_max_abs_rms={gate:?} mixed_max_abs_rms={mixed:?}"
            );
        }
    }
    write(&f.scratch.low, &upstream.1);
}

#[test]
#[ignore = "serial production GPU lease; diagnostic only, no performance qualification"]
fn hc_up_k320_conditioning_diagnostic() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../target/profiles/qwen4exp-hc-conditioning-{}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).expect("reserve unique diagnostic artifact directory");
    let ctx = MetalContext::new().expect("real Metal required");
    assert!(crate::metal::mat_vec_q8_0_lcpp_enabled());
    let fixtures = [Fixture::new(&ctx, 0x4d595df4d0f33173)];
    let f = &fixtures[0];
    run(&ctx, &fixtures, false, true, 1);
    let baseline_original = outputs(f);
    run(&ctx, &fixtures, true, false, 1);
    let candidate_original = outputs(f);
    let low: Vec<_> = read(&f.scratch.low).iter().map(|x| x * 256.0).collect();
    write(&f.scratch.low, &low);
    let (reference, mixed_reference, sum_abs) = conditioned_oracle(f);
    let mut observations = Vec::new();
    for (arm, name) in [(false, "baseline"), (true, "candidate")] {
        f.poison(false);
        run(&ctx, &fixtures, arm, false, 1);
        let result = outputs(f);
        std::fs::write(
            directory.join(format!("{name}-raw.f32le")),
            bytemuck::cast_slice(&result.0),
        )
        .unwrap();
        std::fs::write(
            directory.join(format!("{name}-mixed.f32le")),
            bytemuck::cast_slice(&result.1),
        )
        .unwrap();
        observations.push(result);
    }
    std::fs::write(
        directory.join("reference-raw.f64le"),
        bytemuck::cast_slice(&reference),
    )
    .unwrap();
    std::fs::write(
        directory.join("reference-mixed.f64le"),
        bytemuck::cast_slice(&mixed_reference),
    )
    .unwrap();
    std::fs::write(
        directory.join("sum-abs-products.f64le"),
        bytemuck::cast_slice(&sum_abs),
    )
    .unwrap();
    eprintln!("hc_up diagnostic artifacts={}", directory.display());
    let gamma = |m: f64, u: f64| m * u / (1.0 - m * u);
    let mut summaries = Vec::new();
    for (index, (observed, original)) in observations
        .iter()
        .zip([baseline_original, candidate_original])
        .enumerate()
    {
        let depth = if index == 0 { 20.0 } else { 45.0 };
        let u32 = 2.0f64.powi(-24);
        let gamma64 = gamma(322.0, 2.0f64.powi(-53));
        let mut failed = 0;
        let mut bound_failed = 0;
        let mut power_two_failed = 0;
        let (mut error, mut energy, mut max_abs, mut max_fraction) = (0.0, 0.0, 0.0f64, 0.0f64);
        let mut worst_condition = 0.0f64;
        for row in 0..WIDTH {
            assert!(observed.0[row].is_finite());
            let delta = (observed.0[row] as f64 - reference[row]).abs();
            let bound = (gamma(depth, u32) + gamma64) * sum_abs[row] / (1.0 - gamma64);
            failed += usize::from(delta > 3e-5 * (1.0 + reference[row].abs()));
            bound_failed += usize::from(delta > bound);
            power_two_failed +=
                usize::from(observed.0[row].to_bits() != (original.0[row] * 256.0).to_bits());
            error += delta * delta;
            energy += reference[row] * reference[row];
            max_abs = max_abs.max(delta);
            if bound > 0.0 {
                max_fraction = max_fraction.max(delta / bound);
            }
            if reference[row] != 0.0 {
                worst_condition = worst_condition.max(sum_abs[row] / reference[row].abs());
            }
        }
        let mixed_failed = observed
            .1
            .iter()
            .zip(&mixed_reference)
            .filter(|(a, b)| (**a as f64 - **b).abs() > 3e-5 * (1.0 + b.abs()))
            .count();
        let mixed_error: f64 = observed
            .1
            .iter()
            .zip(&mixed_reference)
            .map(|(&a, &b)| (a as f64 - b).powi(2))
            .sum();
        let mixed_energy: f64 = mixed_reference.iter().map(|x| x * x).sum();
        let summary = serde_json::json!({"arm": if index==0 {"baseline"} else {"candidate"}, "raw_original_pointwise_failures":failed, "raw_relative_rms":(error/energy).sqrt(), "raw_max_abs":max_abs, "depth":depth, "forward_bound_failures":bound_failed, "max_error_bound_fraction":max_fraction, "max_condition":worst_condition, "power_two_scaling_failures":power_two_failed, "mixed_original_pointwise_failures":mixed_failed, "mixed_relative_rms":(mixed_error/mixed_energy).sqrt()});
        eprintln!("hc_up diagnostic {summary}");
        summaries.push(summary);
    }
    std::fs::write(
        directory.join("summary.json"),
        serde_json::to_vec_pretty(&summaries).unwrap(),
    )
    .unwrap();
    f.check();
}
