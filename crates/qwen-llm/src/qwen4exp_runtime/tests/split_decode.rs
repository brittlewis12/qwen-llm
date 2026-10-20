use super::*;
use crate::qwen4exp_qsa::split_decode_probe::with_probe;

const PREFIX: usize = 2179;
const STEPS: usize = 32;

#[path = "hc_up.rs"]
mod hc_up;

#[path = "hc_up_product.rs"]
mod hc_up_product;

#[path = "moe_observe.rs"]
mod moe_observe;

#[path = "topk_native.rs"]
mod topk_native;

#[path = "topk_product.rs"]
mod topk_product;

#[path = "defaults.rs"]
mod defaults;

#[path = "parent_ledger.rs"]
mod parent_ledger;

#[path = "child_ledger.rs"]
mod child_ledger;

#[path = "validation_ceiling.rs"]
mod validation_ceiling;

#[path = "router_width.rs"]
mod router_width;

struct Observation {
    logits: Vec<Vec<f32>>,
    hyper: Vec<Vec<f32>>,
    state: Vec<Vec<u8>>,
    timing: Vec<Qwen4ExpTokenTiming>,
}

fn assert_replay(label: &str, a: &Observation, b: &Observation) {
    assert!(a.logits.len() >= b.logits.len());
    for step in 0..b.logits.len() {
        assert_f32_bits_eq(
            &format!("{label} logits{step}"),
            &a.logits[step],
            &b.logits[step],
        );
        assert_f32_bits_eq(
            &format!("{label} hyper{step}"),
            &a.hyper[step],
            &b.hyper[step],
        );
    }
    if !b.state.is_empty() {
        assert_state_bytes_eq(label, &a.state, &b.state);
    }
}

fn numerical_state(label: &str, a: &[f32], b: &[f32], rms_limit: f64, abs_limit: f64) {
    assert_eq!(a.len(), b.len());
    let mut error = 0.0_f64;
    let mut energy = 0.0_f64;
    let mut maximum = 0.0_f64;
    for (&a, &b) in a.iter().zip(b) {
        assert!(a.is_finite() && b.is_finite(), "{label} nonfinite");
        let delta = f64::from(a) - f64::from(b);
        error += delta * delta;
        energy += f64::from(a).powi(2);
        maximum = maximum.max(delta.abs());
    }
    let rms = (error / energy.max(1e-30)).sqrt();
    eprintln!("split_qsa state {label} rms={rms:.9e} max_abs={maximum:.9e}");
    assert!(
        rms <= rms_limit && maximum <= abs_limit,
        "{label} state gate"
    );
}

/// Next-token distribution agreement between two logit rows: KL in nats both
/// ways, whether the arg-max agrees, and when it does not, the baseline's own
/// logit margin between the two candidates (a flip is explained by the
/// perturbation when that margin is within the measured logit error).
struct ArmAgreement {
    kl_baseline_candidate: f64,
    kl_candidate_baseline: f64,
    top1_equal: bool,
    flip_margin: f64,
    max_abs: f64,
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let total: f64 = logits.iter().map(|&v| (v as f64 - maximum).exp()).sum();
    let log_total = total.ln() + maximum;
    logits.iter().map(|&v| v as f64 - log_total).collect()
}

fn arm_agreement(baseline: &[f32], candidate: &[f32]) -> ArmAgreement {
    assert_eq!(baseline.len(), candidate.len());
    assert!(baseline.iter().chain(candidate).all(|v| v.is_finite()));
    let (lp, lq) = (log_softmax(baseline), log_softmax(candidate));
    let kl = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&la, &lb)| la.exp() * (la - lb))
            .sum::<f64>()
            .max(0.0)
    };
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (i, &x)| {
                if x > best.1 { (i, x) } else { best }
            })
            .0
    };
    let (b, c) = (argmax(baseline), argmax(candidate));
    ArmAgreement {
        kl_baseline_candidate: kl(&lp, &lq),
        kl_candidate_baseline: kl(&lq, &lp),
        top1_equal: b == c,
        flip_margin: if b == c {
            0.0
        } else {
            f64::from(baseline[b]) - f64::from(baseline[c])
        },
        max_abs: baseline
            .iter()
            .zip(candidate)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs())
            .fold(0.0, f64::max),
    }
}

fn assert_numeric(a: &Observation, b: &Observation, tensors: &[MetalTensor]) {
    assert_numeric_at(a, b, tensors, PREFIX);
}

/// Distribution-level agreement over every step, plus the persistent-state
/// closeness gates the earlier witnesses used. The gate is quality, not
/// exactness: mean KL between arms stays far below quantization-level KL, and
/// any arg-max flip must sit inside the measured logit perturbation.
fn assert_numeric_at(a: &Observation, b: &Observation, tensors: &[MetalTensor], prefix: usize) {
    assert_eq!(a.logits.len(), b.logits.len());
    let steps = a.logits.len();
    let mut kl_sum = 0.0;
    let mut kl_max = 0.0_f64;
    let mut flips = 0;
    for step in 0..steps {
        let agreement = arm_agreement(&a.logits[step], &b.logits[step]);
        let kl = agreement
            .kl_baseline_candidate
            .max(agreement.kl_candidate_baseline);
        kl_sum += kl;
        kl_max = kl_max.max(kl);
        eprintln!(
            "split_qsa step{step} kl_bc={:.3e} kl_cb={:.3e} top1_equal={} flip_margin={:.3e} max_abs={:.3e}",
            agreement.kl_baseline_candidate,
            agreement.kl_candidate_baseline,
            agreement.top1_equal,
            agreement.flip_margin,
            agreement.max_abs
        );
        if !agreement.top1_equal {
            flips += 1;
            assert!(
                agreement.flip_margin <= 4.0 * agreement.max_abs,
                "step{step}: arg-max flip with margin {:.3e} beyond the {:.3e} logit perturbation",
                agreement.flip_margin,
                agreement.max_abs
            );
        }
        numerical_state(
            &format!("hyper{step}"),
            &a.hyper[step],
            &b.hyper[step],
            3e-4,
            0.01,
        );
    }
    let kl_mean = kl_sum / steps as f64;
    eprintln!(
        "split_qsa agreement steps={steps} kl_mean={kl_mean:.3e} kl_max={kl_max:.3e} top1_flips={flips}"
    );
    assert!(kl_mean <= 1e-4, "mean KL {kl_mean:.3e} between arms");
    assert!(kl_max <= 1e-3, "max KL {kl_max:.3e} between arms");
    assert_eq!(a.state.len(), tensors.len());
    assert_eq!(b.state.len(), tensors.len());
    for (i, ((a, b), tensor)) in a.state.iter().zip(&b.state).zip(tensors).enumerate() {
        assert_eq!(a.len(), b.len());
        let (a, b, rms_limit, abs_limit) = match tensor.dtype {
            GgmlType::F32 => (
                a.chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>(),
                b.chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>(),
                3e-4,
                0.01,
            ),
            GgmlType::F16 => {
                let (start, end) = match tensor.shape.as_slice() {
                    [128, _] => ((prefix / 4) * 128 * 2, ((prefix + steps) / 4) * 128 * 2),
                    [256, 2, _] => (prefix * 512 * 2, (prefix + steps) * 512 * 2),
                    shape => panic!("unknown persistent F16 shape {shape:?}"),
                };
                assert_eq!(&a[..start], &b[..start], "old prefix differs {i}");
                assert_eq!(&a[end..], &b[end..], "unused suffix differs {i}");
                let decode = |bytes: &[u8]| {
                    bytes
                        .chunks_exact(2)
                        .map(|v| {
                            half::f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32()
                        })
                        .collect::<Vec<_>>()
                };
                (
                    decode(&a[start..end]),
                    decode(&b[start..end]),
                    1e-3,
                    0.03125,
                )
            }
            dtype => panic!("unknown persistent type {dtype:?}"),
        };
        numerical_state(
            &format!("tensor{i}/{:?}", tensor.dtype),
            &a,
            &b,
            rms_limit,
            abs_limit,
        );
    }
}

/// The committed prefix of every QSA key/value and index cache, for checking
/// that decode arms leave the shared prefix untouched.
fn prefix_bytes(runner: &Qwen4ExpTextRunner<'_, '_, '_>) -> Vec<Vec<u8>> {
    runner
        .workspace
        .qsa_persistent_state_tensors()
        .into_iter()
        .flat_map(|(_, tensors)| {
            tensors
                .into_iter()
                .skip(1)
                .map(|tensor| {
                    let bytes = match tensor.shape.as_slice() {
                        [128, _] => (PREFIX / 4) * 128 * 2,
                        [256, 2, _] => PREFIX * 512 * 2,
                        shape => panic!("unknown cache shape {shape:?}"),
                    };
                    unsafe {
                        let source = tensor
                            .buffer
                            .contents()
                            .as_ptr()
                            .cast::<u8>()
                            .add(tensor.offset as usize);
                        std::slice::from_raw_parts(source, bytes).to_vec()
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn run_product(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    split: bool,
    state: bool,
) -> Observation {
    run_product_at(runner, tokens, split, state, PREFIX)
}

/// Decode `tokens` with split attention forced on or off on every eligible
/// step; past the 2,048-key budget every QSA layer must have been eligible.
fn run_product_at(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    split: bool,
    state: bool,
    prefix: usize,
) -> Observation {
    let (observed, records) =
        with_probe(split, || observe_product_at(runner, tokens, state, prefix));
    for row in &records {
        assert_eq!(row.layer % 4, 3);
        assert!((prefix..prefix + tokens.len()).contains(&row.position));
        assert!(row.ids >= crate::qwen4exp_qsa::split_decode::MIN_IDS);
        assert_eq!(row.split, split);
    }
    if prefix >= 2048 {
        assert_eq!(records.len(), tokens.len() * 12, "eligible QSA steps");
        for (step, rows) in records.chunks_exact(12).enumerate() {
            for (layer, row) in rows.iter().enumerate() {
                assert_eq!(row.layer, (layer * 4 + 3) as u32);
                assert_eq!(row.position, prefix + step);
                assert_eq!(row.ids, 2048 + (prefix + step + 1) % 4);
            }
        }
    }
    observed
}

fn observe_product_at(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    state: bool,
    prefix: usize,
) -> Observation {
    let mut observed = Observation {
        logits: Vec::new(),
        hyper: Vec::new(),
        state: Vec::new(),
        timing: Vec::new(),
    };
    for (step, &token) in tokens.iter().enumerate() {
        assert_eq!(runner.next_position(), prefix + step);
        observed
            .logits
            .push(runner.forward_token(token).unwrap().to_vec());
        observed
            .hyper
            .push(runner.workspace.final_hyper_for_tests());
        observed.timing.push(runner.last_token_timing().unwrap());
        assert!(
            runner
                .workspace
                .qsa_committed_lengths()
                .iter()
                .all(|(_, n)| *n == prefix + step + 1)
        );
        assert_eq!(*runner.workspace.ple_prior_tokens().last().unwrap(), token);
    }
    if state {
        observed.state = snapshot_persistent_state(runner);
    }
    observed
}

/// Split decode attention against the incumbent single-pass kernel on one
/// real 2,179-token prefix and 32 teacher-forced continuation tokens:
/// distribution agreement, persistent-state closeness, dispatch census, and
/// an in-process ABBA timing signal (the promotion timing is measured
/// separately between binaries).
#[test]
#[ignore = "serial Metal; one existing UD-Q3_K_XL prefix and 32 continuation tokens"]
fn product_split_decode_shared_prefix() {
    let _benchmark_lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
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
    assert_eq!(tokens.len(), PREFIX + STEPS);
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, PREFIX + STEPS).unwrap();
    let mut loaded = Qwen4ExpLoadedModel::load_with_decode_options(
        &ctx,
        &gguf,
        capacity,
        Some(PREFIX),
        Qwen4ExpDecodeOptions {
            guarded_topk: false,
            hc_up_mix: false,
        },
    )
    .unwrap();
    let mut runner = loaded.create_runner(&ctx).unwrap();
    assert!(
        runner
            .workspace
            .qsa_split_decode_capable()
            .iter()
            .all(|(_, capable)| *capable)
    );
    let _selected = Qwen4ExpPackedSelectedQsaOverride::set(true);
    crate::metal::dispatch_census_begin();
    let endpoint = runner.prefill(&tokens[..PREFIX]).unwrap().to_vec();
    let prefill_census = crate::metal::dispatch_census_take();
    assert!(
        !prefill_census
            .iter()
            .any(|r| r.kernel == "kernel_qwen4exp_qsa_split_f16")
    );
    assert_eq!(
        runner.last_prefill_timing().unwrap().packed_token_count,
        PREFIX
    );
    eprintln!(
        "split_qsa prefix {:?}",
        runner.last_prefill_timing().unwrap()
    );
    let checkpoint = runner.workspace.checkpoint_for_tests();
    let tensors = runner.workspace.persistent_state_tensors();
    let state_bytes: usize = tensors.iter().map(|t| t.n_bytes() as usize).sum();
    assert!(state_bytes * 4 + config.vocab_size as usize * STEPS * 4 * 3 < 1024 * 1024 * 1024);
    let prefix = prefix_bytes(&runner);
    let baseline = run_product(&mut runner, &tokens[PREFIX..], false, true);
    assert_state_bytes_eq("baseline old prefix", &prefix, &prefix_bytes(&runner));
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    assert_f32_bits_eq(
        "restored endpoint",
        &endpoint,
        &runner.logits().unwrap().to_vec(),
    );
    let replay = run_product(&mut runner, &tokens[PREFIX..], false, true);
    assert_replay("restored baseline", &baseline, &replay);
    drop(replay);
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    crate::metal::dispatch_census_begin();
    let candidate = run_product(&mut runner, &tokens[PREFIX..], true, true);
    let census = crate::metal::dispatch_census_take();
    let count = |kernel: &str| census.iter().filter(|r| r.kernel == kernel).count();
    for name in [
        "kernel_qwen4exp_qsa_attention_logits_f16",
        "kernel_qwen4exp_qsa_split_softmax_f32",
        "kernel_qwen4exp_qsa_split_f16",
        "kernel_qwen4exp_qsa_split_merge_f32",
    ] {
        assert_eq!(count(name), STEPS * 12, "candidate census {name}");
    }
    assert_eq!(count("kernel_qwen4exp_qsa_attention_softmax_value_f16"), 0);
    assert_numeric(&baseline, &candidate, &tensors);
    assert_state_bytes_eq("candidate old prefix", &prefix, &prefix_bytes(&runner));
    eprintln!(
        "split_qsa numerical PASS rows={STEPS} persistent_tensors={}",
        tensors.len()
    );
    let mut timings = Vec::new();
    for measured in [false, true] {
        for split in [false, true, true, false] {
            runner.workspace.restore_checkpoint_for_tests(&checkpoint);
            let observed = run_product(&mut runner, &tokens[PREFIX..PREFIX + 4], split, false);
            assert_replay(
                "timed replay",
                if split { &candidate } else { &baseline },
                &observed,
            );
            if measured {
                let gpu: f64 = observed.timing.iter().map(|t| t.gpu_ms.unwrap()).sum();
                let wall: f64 = observed.timing.iter().map(|t| t.total_wall_ms).sum();
                assert!(gpu.is_finite() && gpu > 0.0 && wall.is_finite() && wall > 0.0);
                timings.push((gpu, wall));
            }
        }
    }
    for (label, axis) in [("GPU", 0), ("wall", 1)] {
        let v: Vec<f64> = timings
            .iter()
            .map(|t| if axis == 0 { t.0 } else { t.1 })
            .collect();
        let a = (v[0] + v[3]) * 0.5;
        let b = (v[1] + v[2]) * 0.5;
        let spread = (v[0] - v[3]).abs() / a;
        eprintln!(
            "split_qsa {label} four_forwards_abba_ms={v:?} saved_pct={:.3} spread={spread:.4}",
            (1.0 - b / a) * 100.0
        );
    }
}
