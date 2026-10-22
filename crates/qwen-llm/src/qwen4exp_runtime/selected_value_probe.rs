use super::*;
use crate::tokenizer::Tokenizer;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::Write;

fn hash(values: &[f32]) -> String {
    assert!(values.iter().all(|v| v.is_finite()));
    format!("{:x}", Sha256::digest(bytemuck::cast_slice(values)))
}

fn distribution(reference: &[f32], candidate: &[f32], target: u32) -> serde_json::Value {
    assert_eq!(reference.len(), candidate.len());
    assert!(reference.iter().chain(candidate).all(|v| v.is_finite()));
    let top = |v: &[f32]| (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap();
    let za = f64::from(reference[top(reference)]);
    let zb = f64::from(candidate[top(candidate)]);
    let za = za
        + reference
            .iter()
            .map(|&v| (f64::from(v) - za).exp())
            .sum::<f64>()
            .ln();
    let zb = zb
        + candidate
            .iter()
            .map(|&v| (f64::from(v) - zb).exp())
            .sum::<f64>()
            .ln();
    let mut kl = 0.0;
    let mut max_abs = 0.0f64;
    for (&a, &b) in reference.iter().zip(candidate) {
        kl += (f64::from(a) - za).exp() * ((f64::from(a) - za) - (f64::from(b) - zb));
        max_abs = max_abs.max((f64::from(a) - f64::from(b)).abs());
    }
    json!({"kl":kl,"max_abs":max_abs,"top1_equal":top(reference)==top(candidate),
        "baseline_nll":za-f64::from(reference[target as usize]),
        "candidate_nll":zb-f64::from(candidate[target as usize])})
}

#[test]
fn distribution_screen_checks() {
    let same = distribution(&[-1.0, 0.0, 2.0], &[-1.0, 0.0, 2.0], 2);
    assert_eq!(same["kl"], 0.0);
    assert_eq!(same["top1_equal"], true);
    assert_eq!(same["baseline_nll"], same["candidate_nll"]);
    let shifted = distribution(&[-1.0, 0.0, 2.0], &[3.0, 4.0, 6.0], 2);
    assert!(shifted["kl"].as_f64().unwrap().abs() < 1e-12);
    let changed = distribution(&[-1.0, 0.0, 2.0], &[2.0, 0.0, -1.0], 2);
    assert!(changed["kl"].as_f64().unwrap() > 0.0);
    assert_eq!(changed["top1_equal"], false);
}

#[test]
#[ignore = "local Flash target and text; production lease; product cancellation and retry"]
fn product_selected_cancellation_retry() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let text = std::fs::read_to_string(std::env::var_os("QWEN_QSA_TEXT").unwrap()).unwrap();
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let tokens: Vec<u32> = Tokenizer::from_gguf(&gguf)
        .unwrap()
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .map(|v| u32::try_from(v).unwrap())
        .take(10240)
        .collect();
    assert_eq!(tokens.len(), 10240);
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, tokens.len()).unwrap();
    let mut loaded =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, tokens.len()).unwrap();
    let mut runner = loaded.create_runner(&ctx).unwrap();
    crate::metal::dispatch_census_begin();
    let expected = hash(runner.prefill(&tokens).unwrap().as_slice());
    let census = crate::metal::dispatch_census_take();
    assert!(
        census
            .iter()
            .any(|r| r.kernel.contains("gqa12") && !r.kernel.ends_with("_test")),
        "product route must actually run"
    );
    let state = super::strict_inject_probe::state(&runner);
    runner.reset().unwrap();
    let mut checkpoints = 0;
    let error = runner
        .prefill_with_command_checkpoint(&tokens, || {
            checkpoints += 1;
            if checkpoints == 4 {
                Err(Qwen4ExpRuntimeError::Checkpoint(
                    "selected boundary cancellation".into(),
                ))
            } else {
                Ok(())
            }
        })
        .err()
        .expect("cancellation must stop prefill");
    assert!(matches!(error, Qwen4ExpRuntimeError::Prefill { .. }));
    assert_eq!(runner.next_position(), 4099);
    assert!(!runner.workspace.is_poisoned());
    let actual = hash(
        runner
            .prefill_continuation_with_command_checkpoint(&tokens[4099..], || Ok(()))
            .unwrap()
            .as_slice(),
    );
    assert_eq!(actual, expected);
    assert_eq!(super::strict_inject_probe::state(&runner), state);
}

#[test]
#[ignore = "local Flash target and text; production lease; native selected value screen"]
fn ordinary_selected_value_abba() {
    use crate::qwen4exp_qsa::{value_mma_probe::Scratch, value_probe_binding::Scope};
    use std::rc::Rc;
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let text = std::fs::read_to_string(std::env::var_os("QWEN_QSA_TEXT").unwrap()).unwrap();
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("QWEN_QSA_REPORT").unwrap())
        .unwrap();
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let tokens: Vec<u32> = Tokenizer::from_gguf(&gguf)
        .unwrap()
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .map(|v| u32::try_from(v).unwrap())
        .collect();
    const DEPTH: usize = 8192;
    const ROWS: usize = 2048;
    const HELD: usize = 32;
    assert!(tokens.len() >= DEPTH + ROWS + HELD);
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity =
        Qwen4ExpSessionCapacity::for_forward_limit(&config, DEPTH + ROWS + HELD).unwrap();
    let mut loaded =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, DEPTH + ROWS).unwrap();
    assert!(loaded.packed_selected_active());
    assert_eq!(loaded.packed_prefill_capacity(), Some(ROWS));
    let variant = std::env::var("QWEN_QSA_VALUE_ARM").unwrap_or_else(|_| "mma".into());
    let scratch = match variant.as_str() {
        "mma" => Some(Rc::new(Scratch::new(&ctx, 32).unwrap())),
        "gqa12" => Some(Rc::new(Scratch::new_gqa12(&ctx, 32).unwrap())),
        "product" => None,
        _ => panic!("QWEN_QSA_VALUE_ARM must be mma, gqa12 or product"),
    };
    let value_kernel = if variant == "mma" {
        crate::qwen4exp_qsa::value_mma_probe::VALUE
    } else if variant == "gqa12" {
        crate::qwen4exp_qsa::value_mma_probe::VALUE_GQA12
    } else {
        "kernel_qwen4exp_qsa_selected_value_gqa12_f32"
    };
    let softmax_kernel = if variant == "product" {
        "kernel_qwen4exp_qsa_selected_softmax_gqa4_f32"
    } else {
        crate::qwen4exp_qsa::value_mma_probe::SOFTMAX
    };
    let cpu_bytes =
        3 * HELD as u64 * u64::from(config.vocab_size) * 4 + tokens.capacity() as u64 * 4;
    assert!(
        crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
            0,
            cpu_bytes,
            crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        )
        .is_admitted()
    );
    let mut runner = loaded.create_runner(&ctx).unwrap();
    let mut references: Option<Vec<Vec<f32>>> = None;
    let mut candidate_hashes = None;
    let mut prefix_hash = None;
    let anchored = match std::env::var("QWEN_QSA_ANCHORED").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => panic!("QWEN_QSA_ANCHORED must be 0 or 1"),
    };
    let frozen = if anchored {
        runner.prefill(&tokens[..DEPTH]).unwrap();
        let bytes = runner.snapshot_bytes().unwrap();
        assert!(
            crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
                0,
                cpu_bytes.checked_add(bytes).unwrap(),
                crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
                ctx.memory_signals(),
                true,
            )
            .is_admitted()
        );
        Some((
            runner.capture_snapshot().unwrap(),
            super::strict_inject_probe::state(&runner),
        ))
    } else {
        None
    };
    let order = if anchored {
        vec![
            false, true, false, true, true, false, true, false, false, true,
        ]
    } else {
        vec![false, true, false, true, true, false]
    };
    for (index, enabled) in order.into_iter().enumerate() {
        if let Some((snapshot, expected)) = &frozen {
            runner.restore_snapshot(snapshot).unwrap();
            assert_eq!(&super::strict_inject_probe::state(&runner), expected);
        } else {
            runner.reset().unwrap();
            let prefix = hash(runner.prefill(&tokens[..DEPTH]).unwrap().as_slice());
            if let Some(expected) = &prefix_hash {
                assert_eq!(&prefix, expected);
            } else {
                prefix_hash = Some(prefix);
            }
        }
        let _scope = if !enabled {
            Some(Scope::baseline())
        } else {
            scratch.as_ref().map(|scratch| Scope::new(scratch, true))
        };
        if index < 2 {
            crate::metal::dispatch_census_begin();
        }
        let mut rows = vec![
            runner
                .prefill_continuation_with_command_checkpoint(&tokens[DEPTH..DEPTH + ROWS], || {
                    Ok(())
                })
                .unwrap()
                .to_vec(),
        ];
        let timing = runner.last_prefill_timing().unwrap();
        assert_eq!(timing.command_count, 1);
        assert!(timing.contains_selection);
        if index < 2 {
            let census = crate::metal::dispatch_census_take();
            for kernel in [softmax_kernel, value_kernel] {
                assert_eq!(
                    census.iter().filter(|r| r.kernel == kernel).count(),
                    if enabled { 768 } else { 0 }
                );
            }
            if variant == "product" {
                assert!(
                    !census.iter().any(|r| r.kernel.ends_with("_test")),
                    "product comparison must not dispatch research shaders"
                );
            }
            assert_eq!(
                census
                    .iter()
                    .filter(|r| r.kernel
                        == "kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16")
                    .count(),
                if enabled { 0 } else { 768 },
                "control must execute the production GQA4 leaf",
            );
        }
        for &token in &tokens[DEPTH + ROWS..DEPTH + ROWS + HELD - 1] {
            rows.push(runner.forward_token(token).unwrap().as_slice().to_vec());
        }
        let hashes: Vec<_> = rows.iter().map(|r| hash(r)).collect();
        if references.is_none() {
            references = Some(rows.clone());
        }
        let reference = references.as_ref().unwrap();
        if variant != "mma" {
            assert!(
                rows.iter().zip(reference).all(|(a, b)| hash(a) == hash(b)),
                "chronological-reuse candidate must preserve native bits"
            );
        }
        if enabled {
            if let Some(expected) = &candidate_hashes {
                assert_eq!(&hashes, expected);
            } else {
                candidate_hashes = Some(hashes.clone());
            }
        } else {
            assert!(rows.iter().zip(reference).all(|(a, b)| hash(a) == hash(b)));
        }
        let metrics: Vec<_> = rows
            .iter()
            .zip(reference)
            .enumerate()
            .map(|(i, (b, a))| distribution(a, b, tokens[DEPTH + ROWS + i]))
            .collect();
        let record = json!({"index":index,"candidate_enabled":enabled,"variant":variant,
            "qualification":"measurement screen, not production authority","measured":index>=2,
            "anchored":anchored,
            "depth":DEPTH,"rows":ROWS,"held_tokens":HELD,
            "gpu_ms":timing.complete_gpu_ms().unwrap(),"wall_ms":timing.total_wall_ms,
            "logits_hashes":hashes,"metrics":metrics});
        serde_json::to_writer(&mut out, &record).unwrap();
        writeln!(out).unwrap();
        out.flush().unwrap();
        eprintln!(
            "selected_value_native index={index} enabled={enabled} gpu_ms={}",
            timing.complete_gpu_ms().unwrap()
        );
    }
}

#[test]
#[ignore = "local Flash target and text; production lease; selected prefill attribution"]
fn ordinary_selected_parent_cost() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let text = std::fs::read_to_string(std::env::var_os("QWEN_QSA_TEXT").unwrap()).unwrap();
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("QWEN_QSA_REPORT").unwrap())
        .unwrap();
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let tokens: Vec<u32> = Tokenizer::from_gguf(&gguf)
        .unwrap()
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .map(|v| u32::try_from(v).unwrap())
        .collect();
    const DEPTH: usize = 8192;
    const ROWS: usize = 2048;
    assert!(tokens.len() >= DEPTH + ROWS);
    let config = Qwen4ExpConfig::flash_next_reference();
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, DEPTH + ROWS).unwrap();
    let mut loaded =
        Qwen4ExpLoadedModel::load_with_packed_prefill(&ctx, &gguf, capacity, DEPTH + ROWS).unwrap();
    assert_eq!(loaded.packed_prefill_capacity(), Some(ROWS));
    assert!(loaded.packed_selected_active());
    let mut runner = loaded.create_runner(&ctx).unwrap();
    let mut reference = None;
    for label in ["warm", "plain1", "plain2", "profiled"] {
        runner.reset().unwrap();
        runner.prefill(&tokens[..DEPTH]).unwrap();
        let (logits, gpu_ms, wall_ms, profile) = if label == "profiled" {
            let outcome = execute_qwen4exp_text_packed_profiled_sync(
                &ctx,
                &tokens[DEPTH..DEPTH + ROWS],
                runner.ple_table,
                &runner.weights,
                &mut runner.workspace,
            )
            .unwrap();
            let profile = match outcome.profile {
                Ok(p) => json!({"sampling":p.sampling.as_str(),
                    "raw_coverage":p.raw_coverage_assuming_ns,
                    "stages":p.stages.iter().map(|s| json!({"scope":s.label.scope.as_str(),
                        "name":s.label.name,"layer":s.label.layer,"gpu_ms":s.gpu_ms,
                        "fraction":s.fraction_of_gpu})).collect::<Vec<_>>()}),
                Err(e) => json!({"error":e.to_string()}),
            };
            (
                runner.logits().unwrap().to_vec(),
                outcome.token.gpu_ms.unwrap(),
                outcome.token.total_wall_ms,
                Some(profile),
            )
        } else {
            let logits = runner
                .prefill_continuation_with_command_checkpoint(&tokens[DEPTH..DEPTH + ROWS], || {
                    Ok(())
                })
                .unwrap()
                .to_vec();
            let timing = runner.last_prefill_timing().unwrap();
            assert_eq!(timing.command_count, 1);
            assert!(timing.contains_selection);
            (
                logits,
                timing.complete_gpu_ms().unwrap(),
                timing.total_wall_ms,
                None,
            )
        };
        assert_eq!(runner.next_position(), DEPTH + ROWS);
        let fingerprint = hash(&logits);
        if let Some(expected) = &reference {
            assert_eq!(&fingerprint, expected);
        } else {
            reference = Some(fingerprint.clone());
        }
        let record = json!({"label":label,"depth":DEPTH,"rows":ROWS,"gpu_ms":gpu_ms,
            "wall_ms":wall_ms,"logits_sha256":fingerprint,"profile":profile});
        serde_json::to_writer(&mut out, &record).unwrap();
        writeln!(out).unwrap();
        out.flush().unwrap();
        eprintln!("selected_qsa_parent {record}");
    }
}
