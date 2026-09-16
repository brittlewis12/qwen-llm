use super::*;
use crate::qwen4exp_qsa::split_decode_probe::{SCRATCH_FLOATS, with_probe};

const PREFIX: usize = 2179;
const STEPS: usize = 4;

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

struct Observation {
    logits: Vec<Vec<f32>>,
    hyper: Vec<Vec<f32>>,
    state: Vec<Vec<u8>>,
    timing: Vec<Qwen4ExpTokenTiming>,
}

fn run(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    scratch: &MetalTensor,
    tokens: &[u32],
    split: bool,
    capture_state: bool,
) -> Observation {
    let (observation, records) = with_probe(split, scratch, || {
        let mut logits = Vec::new();
        let mut hyper = Vec::new();
        let mut timing = Vec::new();
        for (step, &token) in tokens.iter().enumerate() {
            assert_eq!(runner.next_position(), PREFIX + step);
            logits.push(runner.forward_token(token).unwrap().to_vec());
            hyper.push(runner.workspace.final_hyper_for_tests());
            timing.push(runner.last_token_timing().unwrap());
            assert_eq!(runner.next_position(), PREFIX + step + 1);
            assert!(
                runner
                    .workspace
                    .qsa_committed_lengths()
                    .iter()
                    .all(|(_, n)| *n == PREFIX + step + 1)
            );
            assert_eq!(*runner.workspace.ple_prior_tokens().last().unwrap(), token);
        }
        Observation {
            logits,
            hyper,
            timing,
            state: if capture_state {
                snapshot_persistent_state(runner)
            } else {
                Vec::new()
            },
        }
    });
    assert_eq!(records.len(), STEPS * 12);
    for (step, rows) in records.chunks_exact(12).enumerate() {
        for (layer, row) in rows.iter().enumerate() {
            assert_eq!(row.layer, (layer * 4 + 3) as u32);
            assert_eq!(row.position, PREFIX + step);
            assert_eq!(row.ids, 2048 + step);
            assert_eq!(row.split, split);
        }
    }
    observation
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
    eprintln!("native_split state {label} rms={rms:.9e} max_abs={maximum:.9e}");
    assert!(
        rms <= rms_limit && maximum <= abs_limit,
        "{label} state gate"
    );
}

fn assert_numeric(a: &Observation, b: &Observation, tensors: &[MetalTensor]) {
    assert_numeric_at(a, b, tensors, PREFIX);
}

fn assert_numeric_at(a: &Observation, b: &Observation, tensors: &[MetalTensor], prefix: usize) {
    assert_eq!(a.logits.len(), b.logits.len());
    let steps = a.logits.len();
    for step in 0..steps {
        assert_logit_arms_close(
            &format!("native_split step{step}"),
            &a.logits[step],
            &b.logits[step],
        );
        numerical_state(
            &format!("hyper{step}"),
            &a.hyper[step],
            &b.hyper[step],
            3e-4,
            0.01,
        );
    }
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

fn run_product(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    split: bool,
    state: bool,
) -> Observation {
    run_product_at(runner, tokens, split, state, PREFIX)
}

fn run_product_at(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    split: bool,
    state: bool,
    prefix: usize,
) -> Observation {
    runner
        .workspace
        .set_split_decode_for_tests(runner.ctx, split)
        .unwrap();
    observe_product_at(runner, tokens, state, prefix)
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

#[test]
#[ignore = "serial Metal; actual split product bindings, one existing prefix and 32 continuation tokens"]
fn product_split_decode_shared_prefix() {
    let _benchmark_lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    std::fs::create_dir_all(&parent).unwrap();
    let artifact = parent.join(format!("qwen4exp-product-split-{}", std::process::id()));
    std::fs::create_dir(&artifact).unwrap();
    eprintln!("product_split raw_logit_artifacts={}", artifact.display());
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
    assert_eq!(tokens.len(), PREFIX + 32);
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
            guarded_topk: false,
            split_qsa: true,
            hc_up_mix: false,
        },
    )
    .unwrap();
    assert!(loaded.split_decode_enabled());
    let mut runner = loaded.create_runner(&ctx).unwrap();
    assert_eq!(
        runner
            .workspace
            .memory_plan()
            .allocations()
            .iter()
            .filter(|a| a.name == "session.qsa_split")
            .count(),
        1
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
        "product_split prefix {:?}",
        runner.last_prefill_timing().unwrap()
    );
    let checkpoint = runner.workspace.checkpoint_for_tests();
    let tensors = runner.workspace.persistent_state_tensors();
    let state_bytes: usize = tensors.iter().map(|t| t.n_bytes() as usize).sum();
    assert!(state_bytes * 4 + config.vocab_size as usize * 32 * 4 * 3 < 1024 * 1024 * 1024);
    let prefix = prefix_bytes(&runner);
    let baseline = run_product(&mut runner, &tokens[PREFIX..], false, true);
    assert_state_bytes_eq(
        "product baseline old prefix",
        &prefix,
        &prefix_bytes(&runner),
    );
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    assert_f32_bits_eq(
        "product restored endpoint",
        &endpoint,
        &runner.logits().unwrap().to_vec(),
    );
    let replay = run_product(&mut runner, &tokens[PREFIX..], false, true);
    assert_replay("product restored baseline", &baseline, &replay);
    drop(replay);
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    crate::metal::dispatch_census_begin();
    let candidate = run_product(&mut runner, &tokens[PREFIX..], true, true);
    let census = crate::metal::dispatch_census_take();
    for (name, observation) in [("baseline", &baseline), ("candidate", &candidate)] {
        let rows: Vec<u8> = observation
            .logits
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(artifact.join(format!("{name}-32x248320.f32le")), rows).unwrap();
    }
    for name in [
        "kernel_qwen4exp_qsa_split_f16",
        "kernel_qwen4exp_qsa_split_merge_f32",
    ] {
        assert_eq!(
            census.iter().filter(|r| r.kernel == name).count(),
            384,
            "product witness {name}"
        );
    }
    assert!(
        !census
            .iter()
            .any(|r| r.kernel == "kernel_qwen4exp_qsa_attention_logits_f16")
    );
    assert_numeric(&baseline, &candidate, &tensors);
    assert_state_bytes_eq(
        "product candidate old prefix",
        &prefix,
        &prefix_bytes(&runner),
    );
    eprintln!(
        "product_split numerical PASS rows=32 persistent_tensors={} pairs=384",
        tensors.len()
    );
    let mut timings = Vec::new();
    for measured in [false, true] {
        for split in [false, true, true, false] {
            runner.workspace.restore_checkpoint_for_tests(&checkpoint);
            let observed = run_product(&mut runner, &tokens[PREFIX..PREFIX + 4], split, false);
            assert_replay(
                "product timed replay",
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
    for (label, axis, floor) in [("GPU", 0, 0.05), ("wall", 1, 0.03)] {
        let v: Vec<f64> = timings
            .iter()
            .map(|t| if axis == 0 { t.0 } else { t.1 })
            .collect();
        let a = (v[0] + v[3]) * 0.5;
        let b = (v[1] + v[2]) * 0.5;
        let spread = (v[0] - v[3]).abs() / a;
        let useful =
            b <= a * (1.0 - floor) && v[1] <= v[0] * (1.0 - floor) && v[2] <= v[3] * (1.0 - floor);
        let verdict = if spread > 0.05 {
            "INCONCLUSIVE"
        } else if useful {
            "USEFUL"
        } else {
            "HOLD"
        };
        eprintln!(
            "product_split {label} four_forwards_abba_ms={v:?} saved_pct={} spread={spread} verdict={verdict}",
            (1.0 - b / a) * 100.0
        );
    }
}

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

#[test]
#[ignore = "serial Metal; one existing UD-Q3_K_XL packed prefix and shared-state decode qualification"]
fn native_split_decode_shared_prefix() {
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
    assert_eq!(tokens.len(), 2211);
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
            split_qsa: false,
            hc_up_mix: false,
        },
    )
    .unwrap();
    let mut runner = loaded.create_runner(&ctx).unwrap();
    let scratch = MetalTensor::zeros_f32(&ctx, vec![SCRATCH_FLOATS as u64]).unwrap();
    ctx.pipeline("kernel_qwen4exp_qsa_split_f16").unwrap();
    ctx.pipeline("kernel_qwen4exp_qsa_split_merge_f32").unwrap();
    let _selected = Qwen4ExpPackedSelectedQsaOverride::set(true);
    let endpoint = runner.prefill(&tokens[..PREFIX]).unwrap().to_vec();
    eprintln!(
        "native_split prefix {:?}",
        runner.last_prefill_timing().unwrap()
    );
    assert_eq!(
        runner.last_prefill_timing().unwrap().packed_token_count,
        PREFIX
    );
    let checkpoint = runner.workspace.checkpoint_for_tests();
    let tensors = runner.workspace.persistent_state_tensors();
    let state_bytes: usize = tensors.iter().map(|t| t.n_bytes() as usize).sum();
    assert!(
        state_bytes * 4 < 1024 * 1024 * 1024,
        "bounded CPU snapshot budget"
    );
    eprintln!(
        "native_split state_bytes={state_bytes} shared_scratch_bytes={}",
        SCRATCH_FLOATS * 4
    );
    let prefix = prefix_bytes(&runner);
    let continuation = &tokens[PREFIX..PREFIX + STEPS];
    let baseline = run(&mut runner, &scratch, continuation, false, true);
    assert_state_bytes_eq("baseline old prefix", &prefix, &prefix_bytes(&runner));
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    assert_f32_bits_eq(
        "restored endpoint",
        &endpoint,
        &runner.logits().unwrap().to_vec(),
    );
    let replay = run(&mut runner, &scratch, continuation, false, true);
    assert_replay("baseline restored", &baseline, &replay);
    drop(replay);
    runner.workspace.restore_checkpoint_for_tests(&checkpoint);
    crate::metal::dispatch_census_begin();
    let candidate = run(&mut runner, &scratch, continuation, true, true);
    let census = crate::metal::dispatch_census_take();
    for kernel in [
        "kernel_qwen4exp_qsa_split_f16",
        "kernel_qwen4exp_qsa_split_merge_f32",
    ] {
        assert_eq!(
            census.iter().filter(|row| row.kernel == kernel).count(),
            48,
            "native split dispatch {kernel}"
        );
    }
    assert!(
        !census
            .iter()
            .any(|row| row.kernel == "kernel_qwen4exp_qsa_attention_logits_f16")
    );
    assert_numeric(&baseline, &candidate, &tensors);
    assert_state_bytes_eq("candidate old prefix", &prefix, &prefix_bytes(&runner));
    eprintln!(
        "native_split numerical PASS full_logits=4 persistent_tensors={} witness_pairs=48",
        tensors.len()
    );

    for split in [false, true, true, false] {
        runner.workspace.restore_checkpoint_for_tests(&checkpoint);
        let warm = run(&mut runner, &scratch, continuation, split, false);
        assert_replay("warm", if split { &candidate } else { &baseline }, &warm);
    }
    let mut times = Vec::new();
    for split in [false, true, true, false] {
        runner.workspace.restore_checkpoint_for_tests(&checkpoint);
        let observed = run(&mut runner, &scratch, continuation, split, false);
        assert_replay(
            "timed",
            if split { &candidate } else { &baseline },
            &observed,
        );
        let gpu: f64 = observed.timing.iter().map(|t| t.gpu_ms.unwrap()).sum();
        let wall: f64 = observed.timing.iter().map(|t| t.total_wall_ms).sum();
        assert!(gpu.is_finite() && gpu > 0.0 && wall.is_finite() && wall > 0.0);
        eprintln!(
            "native_split arm split={split} token_timings={:?}",
            observed.timing
        );
        times.push((gpu, wall));
    }
    for (label, axis, fraction) in [("GPU", 0, 0.05), ("wall", 1, 0.03)] {
        let v: Vec<f64> = times
            .iter()
            .map(|t| if axis == 0 { t.0 } else { t.1 })
            .collect();
        let a = (v[0] + v[3]) * 0.5;
        let b = (v[1] + v[2]) * 0.5;
        let spread = (v[0] - v[3]).abs() / a;
        let useful = b <= a * (1.0 - fraction)
            && v[1] <= v[0] * (1.0 - fraction)
            && v[2] <= v[3] * (1.0 - fraction);
        let verdict = if spread > 0.05 {
            "INCONCLUSIVE"
        } else if useful {
            "USEFUL"
        } else {
            "HOLD"
        };
        eprintln!(
            "native_split {label} four_forwards_abba_ms={v:?} saved_pct={:.6} A_spread={spread:.9} verdict={verdict}",
            (1.0 - b / a) * 100.0
        );
    }
    for split in [false, true] {
        runner.workspace.restore_checkpoint_for_tests(&checkpoint);
        let (outcome, records) = with_probe(split, &scratch, || {
            runner
                .forward_token_layer_profiled(continuation[0])
                .unwrap()
        });
        assert_eq!(records.len(), 12);
        let reference = if split { &candidate } else { &baseline };
        assert_f32_bits_eq(
            "profile observer logits",
            &reference.logits[0],
            &runner.logits().unwrap().to_vec(),
        );
        assert_f32_bits_eq(
            "profile observer hyper",
            &reference.hyper[0],
            &runner.workspace.final_hyper_for_tests(),
        );
        eprintln!("native_split coarse_profile split={split} outcome={outcome:?}");
    }
}
