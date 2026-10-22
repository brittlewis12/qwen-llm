//! Target-only diagnostic. Requires QWEN4EXP_Q3_K_XL_GGUF and QWEN_PACKET_TEXT
//! (private UTF-8 file); optional QWEN_PACKET_DEPTHS=64,2179,8192.
//! Run only this ignored test under the production lease, with --nocapture.
//! Restores, checks and the common continuation are outside ordinary timings.
//! QWEN_PACKET_HC: baseline (default), inject_rows, mix_fewrow, or
//! mix_fewrow_inject_rows. Injection remains F32; only down/up can use Q8 MMA.
//! QWEN_PACKET_WIDTHS=2,4 narrows the matrix; QWEN_PACKET_AUDIT_ONLY=1 runs
//! restore/order counterfactuals without ordinary timings or profiling.
//! QWEN_PACKET_LOCALIZE=1 additionally captures layer means and expert counts;
//! requires audit-only mode and widths restricted to 2/4.
//! QWEN_PACKET_ROUTERS=1 captures raw routes at layers 20/31 (audit-only).
//! QWEN_PACKET_QUALITY_TOKENS=64 runs continuing teacher-forced arms (audit-only).

use super::*;
use crate::qwen4exp_metal::packet_hc::{Mode, Scope};
use crate::tokenizer::Tokenizer;
use serde_json::json;
use std::collections::BTreeMap;

mod quality;
mod repeatability;

fn bits(values: &[f32]) -> Vec<u32> {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "nonfinite logits"
    );
    values.iter().map(|value| value.to_bits()).collect()
}

fn hc_mode(value: &str) -> Mode {
    match value {
        "baseline" => Mode::Baseline,
        "inject_rows" => Mode::InjectRows,
        "mix_fewrow" => Mode::MixFewrow,
        "mix_fewrow_inject_rows" => Mode::MixFewrowInjectRows,
        _ => panic!(
            "invalid QWEN_PACKET_HC={value}; use baseline/inject_rows/mix_fewrow/mix_fewrow_inject_rows; injection is F32, not Q8"
        ),
    }
}

fn distribution_difference(reference: &[u32], candidate: &[u32]) -> serde_json::Value {
    assert_eq!(reference.len(), candidate.len());
    assert!(!reference.is_empty());
    let a: Vec<f64> = reference
        .iter()
        .map(|&v| f64::from(f32::from_bits(v)))
        .collect();
    let b: Vec<f64> = candidate
        .iter()
        .map(|&v| f64::from(f32::from_bits(v)))
        .collect();
    assert!(a.iter().chain(&b).all(|v| v.is_finite()));
    let top = |x: &[f64]| (0..x.len()).max_by(|&i, &j| x[i].total_cmp(&x[j])).unwrap();
    let log_z = |x: &[f64]| {
        let max = x[top(x)];
        max + x.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
    };
    let (za, zb) = (log_z(&a), log_z(&b));
    let (mut aa, mut bb, mut ab, mut error2, mut max_abs) = (0.0, 0.0, 0.0, 0.0, 0.0_f64);
    let (mut kl_ab, mut kl_ba) = (0.0, 0.0);
    for (&x, &y) in a.iter().zip(&b) {
        aa += x * x;
        bb += y * y;
        ab += x * y;
        error2 += (x - y) * (x - y);
        max_abs = max_abs.max((x - y).abs());
        kl_ab += (x - za).exp() * ((x - za) - (y - zb));
        kl_ba += (y - zb).exp() * ((y - zb) - (x - za));
    }
    json!({"top1_agrees": top(&a) == top(&b), "max_abs": max_abs,
        "cosine": (aa > 0.0 && bb > 0.0).then(|| ab/(aa*bb).sqrt()),
        "relative_rms": (aa > 0.0).then(|| (error2/aa).sqrt()),
        "kl_baseline_candidate": kl_ab, "kl_candidate_baseline": kl_ba})
}

#[test]
fn packet_hc_diagnostic_parser_and_metrics() {
    for mode in [
        Mode::Baseline,
        Mode::InjectRows,
        Mode::MixFewrow,
        Mode::MixFewrowInjectRows,
    ] {
        assert_eq!(hc_mode(mode.name()), mode);
    }
    assert!(std::panic::catch_unwind(|| hc_mode("inject_fewrow")).is_err());
    let values = bits(&[-2.0, 0.5, 3.0]);
    let same = distribution_difference(&values, &values);
    assert_eq!(same["top1_agrees"], true);
    assert_eq!(same["max_abs"], 0.0);
    assert_eq!(same["kl_baseline_candidate"], 0.0);
    let different = distribution_difference(&values, &bits(&[3.0, 0.5, -2.0]));
    assert_eq!(different["top1_agrees"], false);
    assert!(different["kl_baseline_candidate"].as_f64().unwrap() > 0.0);
}

fn run_arm(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    arm: &str,
    captures: &mut Qwen4ExpCaptureBank,
    logits: &mut Qwen4ExpRowLogitsBank,
) -> (Vec<f32>, Vec<Qwen4ExpTokenTiming>) {
    runner.validate_prefill_request(tokens).unwrap();
    if arm == "serial" {
        let mut values = Vec::new();
        let mut timings = Vec::new();
        for &token in tokens {
            values.extend_from_slice(runner.forward_token(token).unwrap().as_slice());
            timings.push(runner.last_token_timing.unwrap());
        }
        return (values, timings);
    }
    let positions = runner.next_position()..runner.next_position() + tokens.len();
    let capture = (arm == "verify_features").then(|| captures.sink(positions.clone()));
    let output = (arm != "packed_last").then(|| logits.sink(positions).unwrap());
    let timing = execute_qwen4exp_text_packed_with_row_outputs_sync(
        runner.ctx,
        tokens,
        runner.ple_table,
        &runner.weights,
        &mut runner.workspace,
        capture.as_ref(),
        output.as_ref(),
    )
    .unwrap();
    let values = match output {
        Some(output) => output.complete().as_slice().to_vec(),
        None => runner.logits().unwrap().to_vec(),
    };
    (values, vec![timing])
}

#[test]
#[ignore = "explicit local Flash target and QWEN_PACKET_TEXT; exclusive production GPU lease"]
fn native_short_packet_cost_and_stage_attribution() {
    let mode = hc_mode(&std::env::var("QWEN_PACKET_HC").unwrap_or_else(|_| "baseline".into()));
    let widths = repeatability::widths(
        &std::env::var("QWEN_PACKET_WIDTHS").unwrap_or_else(|_| "1,2,4,8".into()),
    );
    let audit_only = match std::env::var("QWEN_PACKET_AUDIT_ONLY").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => panic!("QWEN_PACKET_AUDIT_ONLY must be 0 or 1"),
    };
    let localize = match std::env::var("QWEN_PACKET_LOCALIZE").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => panic!("QWEN_PACKET_LOCALIZE must be 0 or 1"),
    };
    let routers = match std::env::var("QWEN_PACKET_ROUTERS").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => panic!("QWEN_PACKET_ROUTERS must be 0 or 1"),
    };
    let quality_tokens: usize = match std::env::var("QWEN_PACKET_QUALITY_TOKENS").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => 0,
        Ok("64") => 64,
        _ => panic!("QWEN_PACKET_QUALITY_TOKENS must be 0 or 64"),
    };
    assert!(
        !(routers || quality_tokens > 0)
            || (audit_only && widths.iter().all(|n| matches!(n, 2 | 4 | 8))),
        "router/quality diagnostics require audit-only mode and widths 2/4/8"
    );
    assert!(
        !localize || (audit_only && widths.iter().all(|n| matches!(n, 2 | 4))),
        "localization requires QWEN_PACKET_AUDIT_ONLY=1 and QWEN_PACKET_WIDTHS=2,4 (or either width)"
    );
    std::env::var_os("QWEN4EXP_Q3_K_XL_GGUF")
        .filter(|path| !path.is_empty())
        .expect("set QWEN4EXP_Q3_K_XL_GGUF to the local target first shard");
    let text = std::fs::read_to_string(
        std::env::var_os("QWEN_PACKET_TEXT").expect("QWEN_PACKET_TEXT is required"),
    )
    .expect("QWEN_PACKET_TEXT must be a readable UTF-8 file");
    let depths: Vec<usize> = std::env::var("QWEN_PACKET_DEPTHS")
        .unwrap_or_else(|_| "64,2179,8192".into())
        .split(',')
        .map(|depth| {
            depth
                .trim()
                .parse()
                .expect("depth must be a positive integer")
        })
        .collect();
    assert!(!depths.is_empty() && depths.iter().all(|&depth| depth > 0));
    let extent = depths
        .iter()
        .max()
        .unwrap()
        .checked_add(quality_tokens.max(8) + 1)
        .unwrap();
    let gguf = GgufFile::open(crate::test_fixtures::QWEN4EXP_Q3_K_XL.required()).unwrap();
    let tokens: Vec<u32> = Tokenizer::from_gguf(&gguf)
        .unwrap()
        .encode(&text, false)
        .unwrap()
        .into_iter()
        .map(|token| u32::try_from(token).unwrap())
        .collect();
    assert!(
        tokens.len() > extent,
        "text needs max(depths)+max(8,quality_tokens)+2 tokens, including continuation target; no repetition/padding"
    );
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let capacity =
        Qwen4ExpSessionCapacity::for_forward_limit(&Qwen4ExpConfig::flash_next_reference(), extent)
            .unwrap();
    let mut loaded = Qwen4ExpLoadedModel::load_with_adaptive_prefill(
        &ctx,
        &gguf,
        capacity,
        extent,
        Qwen4ExpDecodeOptions::default(),
    )
    .unwrap();
    let mut runner = loaded.create_runner(&ctx).unwrap();
    let prefill_rows = runner.workspace.packed_prefill_capacity().unwrap();
    assert!(
        prefill_rows >= 8,
        "admitted workspace must cover packet widths"
    );
    let mut captures = runner
        .allocate_capture_bank(vec![5, 15, 25, 35, 45], 8)
        .unwrap();
    let mut logits = runner.allocate_row_logits_bank(8).unwrap();
    let vocab = runner.weights.geometry.vocab_size();
    let mut all_stable = true;
    for depth in depths {
        runner.reset().unwrap();
        runner.prefill(&tokens[..depth]).unwrap();
        let base_state = repeatability::state(&runner, depth);
        let base = runner.capture_snapshot().unwrap();
        for &width in &widths {
            let suffix = &tokens[depth..depth + width];
            let audit = repeatability::audit(
                &mut runner,
                &base,
                &base_state,
                suffix,
                tokens[depth + width],
                mode,
                &mut captures,
                &mut logits,
            );
            all_stable &= audit.stable;
            if localize && audit.stable {
                repeatability::localize(
                    &mut runner,
                    &base,
                    &base_state,
                    suffix,
                    tokens[depth + width],
                    mode,
                    &audit,
                    &mut logits,
                );
            }
            if audit.stable && audit_only {
                quality::serial_witness(
                    &mut runner,
                    &base,
                    &base_state,
                    suffix,
                    &tokens[depth + 1..depth + width + 2],
                    mode,
                    &audit,
                    &mut captures,
                    &mut logits,
                );
                if routers {
                    quality::router_witness(
                        &mut runner,
                        &base,
                        &base_state,
                        suffix,
                        tokens[depth + width],
                        mode,
                        &audit,
                        &mut captures,
                        &mut logits,
                    );
                }
                if quality_tokens > 0 {
                    quality::rolling(
                        &mut runner,
                        &base,
                        &base_state,
                        &tokens[depth..depth + quality_tokens + 2],
                        width,
                        mode,
                        &audit,
                        &mut captures,
                        &mut logits,
                    );
                }
            }
            if !audit.stable || audit_only {
                continue;
            }
            // No override is active during prefix construction or snapshot capture.
            let _hc_scope = Scope::new(mode);
            let mut expected = BTreeMap::<&str, Vec<u32>>::new();
            let mut expected_future = BTreeMap::<&str, Vec<u32>>::new();
            for (iteration, order) in [
                ["serial", "packed_last", "verify_all", "verify_features"],
                ["serial", "packed_last", "verify_all", "verify_features"],
                ["verify_features", "verify_all", "packed_last", "serial"],
            ]
            .into_iter()
            .enumerate()
            {
                for arm in order {
                    runner.restore_snapshot(&base).unwrap();
                    let start = Instant::now();
                    let (values, commands) =
                        run_arm(&mut runner, suffix, arm, &mut captures, &mut logits);
                    let wall_ms = start.elapsed().as_secs_f64() * 1e3;
                    assert_eq!(runner.next_position(), depth + width);
                    assert_eq!(
                        values.len(),
                        vocab * if arm == "packed_last" { 1 } else { width }
                    );
                    let actual = bits(&values);
                    let future = bits(
                        runner
                            .forward_token(tokens[depth + width])
                            .unwrap()
                            .as_slice(),
                    );
                    if iteration == 0 && arm == "verify_all" {
                        let same =
                            actual == audit.candidate.rows && future == audit.candidate.future;
                        repeatability::emit_logits_comparison(
                            depth,
                            width,
                            mode,
                            "candidate_first_vs_after_serial_and_packed_last",
                            &audit.candidate.rows,
                            &actual,
                            &audit.candidate.future,
                            &future,
                            vocab,
                        );
                        assert!(
                            same,
                            "candidate changed after serial/packed-last conditioning; not a warmup exemption"
                        );
                    }
                    assert_eq!(runner.next_position(), depth + width + 1);
                    assert!(
                        runner
                            .workspace
                            .qsa_committed_lengths()
                            .iter()
                            .all(|(_, length)| *length == depth + width + 1)
                    );
                    if iteration == 0 {
                        expected.insert(arm, actual);
                        expected_future.insert(arm, future);
                    } else {
                        assert!(
                            actual == expected[arm],
                            "repeat logits {depth}/{width}/{arm}"
                        );
                        assert!(
                            future == expected_future[arm],
                            "repeat continuation {depth}/{width}/{arm}"
                        );
                    }
                    let gpu_ms: Option<f64> = commands.iter().map(|t| t.gpu_ms).sum();
                    eprintln!(
                        "packet_cost {}",
                        json!({
                            "family": "flash_next", "depth": depth, "width": width,
                            "hc_mode": mode.name(),
                            "arm": arm, "warmup": iteration == 0, "iteration": iteration,
                            "wall_ms": wall_ms, "gpu_ms": gpu_ms, "prefill_rows": prefill_rows,
                            "capacity": extent, "readback": "all requested logits; no feature readback",
                            "commands": commands.iter().map(|t| json!({
                                "gpu_ms": t.gpu_ms, "encode_cpu_ms": t.encode_cpu_ms,
                                "completion_wait_ms": t.completion_wait_ms,
                                "executor_wall_ms": t.total_wall_ms,
                            })).collect::<Vec<_>>(),
                        })
                    );
                }
            }
            for arm in ["verify_all", "verify_features"] {
                assert!(
                    expected[arm][(width - 1) * vocab..] == expected["packed_last"],
                    "same-width endpoint {depth}/{width}/{arm}"
                );
                assert!(
                    expected_future[arm] == expected_future["packed_last"],
                    "same-width continuation {depth}/{width}/{arm}"
                );
            }
            assert!(
                expected["verify_all"] == expected["verify_features"],
                "capture changes logits"
            );
            quality::cheap_report(
                depth,
                width,
                mode,
                &expected["serial"],
                &expected_future["serial"],
                &audit.baseline.rows,
                &audit.baseline.future,
                &expected["verify_all"],
                &expected_future["verify_all"],
                &tokens[depth + 1..depth + width + 2],
                vocab,
            );
            {
                let baseline = &audit.baseline.rows;
                let baseline_future = &audit.baseline.future;
                eprintln!(
                    "packet_comparison {}",
                    json!({
                        "family": "flash_next", "depth": depth, "width": width,
                        "hc_mode": mode.name(), "reference_hc_mode": "baseline",
                        "scope": "same prefix, teacher-forced verify_all and one common continuation; characterization only",
                        "rows": baseline.chunks_exact(vocab).zip(expected["verify_all"].chunks_exact(vocab))
                            .map(|(a,b)| distribution_difference(a,b)).collect::<Vec<_>>(),
                        "continuation": distribution_difference(baseline_future, &expected_future["verify_all"]),
                    })
                );
            }
            // Existing packed profiler requires N>1; it measures one head only.
            if width > 1 {
                runner.restore_snapshot(&base).unwrap();
                let start = Instant::now();
                let report = execute_qwen4exp_text_packed_profiled_sync(
                    runner.ctx,
                    suffix,
                    runner.ple_table,
                    &runner.weights,
                    &mut runner.workspace,
                )
                .unwrap();
                let values = runner.logits().unwrap().to_vec();
                let wall_ms = start.elapsed().as_secs_f64() * 1e3;
                assert!(
                    bits(&values) == expected["packed_last"],
                    "profile changes endpoint"
                );
                let mut stages = BTreeMap::<String, f64>::new();
                if let Ok(profile) = &report.profile {
                    for stage in &profile.stages {
                        *stages
                            .entry(format!("{}:{}", stage.depth, stage.label.name))
                            .or_default() += stage.gpu_ms;
                    }
                }
                eprintln!(
                    "packet_profile {}",
                    json!({
                        "family": "flash_next", "depth": depth, "width": width,
                        "hc_mode": mode.name(),
                        "wall_ms": wall_ms, "gpu_ms": report.token.gpu_ms,
                        "sampling": report.sampling.as_str(), "fallback": report.sampling_fallback,
                        "error": report.profile.as_ref().err().map(ToString::to_string),
                        "raw_coverage": report.profile.as_ref().ok().map(|p| p.raw_coverage_assuming_ns),
                        "stages_ms": stages,
                        "scope": "packed_last only; observer perturbs timing; nested stages overlap; detailed layers 5/7 only",
                    })
                );
            }
        }
    }
    assert!(
        all_stable,
        "baseline repeatability/restore control failed; candidate attribution blocked (see packet_audit records)"
    );
}
