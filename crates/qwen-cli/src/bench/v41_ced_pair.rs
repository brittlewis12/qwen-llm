//! Bounded resident Full/CED timing protocol. This is not quality qualification.

use super::*;
use crate::family_bench::{
    FamilyBench, FamilyBenchExtent, FamilySemantics, V41BenchPrefillRep, attach_v41_diagnostics,
    family_row, run_family_pp_row, shape_label,
};
use anyhow::ensure;
use qwen_llm::deepseek_v41::forward::{PrefillExecution, preflight_cold_prefill};
use qwen_llm::model_family::ModelFamily;

// At most 38 cases, 256 layer counters per report, two JSON rows and a 1M-logit head.
const CALLER_CPU_BYTES: u64 = 64 << 20;
const MAX_PP: usize = 8192;

pub(crate) fn validate_args(
    family: ModelFamily,
    pp: &[usize],
    tg: &[usize],
    depths: &[usize],
    runs: usize,
    no_warmup: bool,
    chunk: Option<usize>,
) -> Result<()> {
    ensure!(
        family == ModelFamily::DeepSeekV41,
        "--v41-ced-pair requires DeepSeek V4.1"
    );
    ensure!(
        pp.len() == 1 && (2..=MAX_PP).contains(&pp[0]),
        "--v41-ced-pair requires exactly one --pp in 2..=8192"
    );
    ensure!(
        tg.is_empty() && depths == [0],
        "--v41-ced-pair is pp-only at exactly --depth 0"
    );
    ensure!(
        (2..=16).contains(&runs) && runs % 2 == 0,
        "--v41-ced-pair requires even --runs 2..=16 per policy"
    );
    ensure!(
        !no_warmup && chunk.is_none(),
        "--v41-ced-pair requires warmup and no chunk override"
    );
    Ok(())
}

fn validate_environment(
    get: impl Fn(&str) -> Option<std::ffi::OsString>,
    names: &[String],
) -> Result<()> {
    ensure!(
        get("MTL_DEBUG_LAYER").as_deref() == Some(std::ffi::OsStr::new("1")),
        "--v41-ced-pair requires MTL_DEBUG_LAYER=1"
    );
    for name in names {
        let override_name = name.starts_with("QWEN_V41_")
            || name.starts_with("QWEN_MATMAT_")
            || name.starts_with("QWEN_MATVEC_")
            || name.starts_with("QWEN_PREFILL_")
            || name.starts_with("QWEN_DECODE_")
            || name.starts_with("QWEN_DSV4_")
            || name.starts_with("QWEN_PROMPT_MATMAT_")
            || name == "QWEN_EXPERT_TENSOR_TILE";
        ensure!(
            !override_name,
            "remove runtime/arithmetic/profiling override {name} for --v41-ced-pair"
        );
    }
    if let Some(value) = get("QWEN_METAL_LEASE_SKIP_WIRED_GATE") {
        ensure!(
            qwen_llm::env_flag::env_value_falsy(&value.to_string_lossy()),
            "--v41-ced-pair refuses the wired admission bypass"
        );
    }
    ensure!(
        !qwen_llm::metal::dispatch_census_is_active(),
        "--v41-ced-pair refuses an active dispatch observer"
    );
    Ok(())
}

pub(crate) fn check_environment() -> Result<()> {
    let names: Vec<_> = std::env::vars_os()
        .map(|(name, _)| name.to_string_lossy().into_owned())
        .collect();
    validate_environment(|name| std::env::var_os(name), &names)
}

/// Called before with_family_bench can import any resident weights.
pub(crate) fn preweight_plan(
    ctx: &MetalContext,
    gguf: &GgufFile,
    extent: FamilyBenchExtent,
) -> Result<serde_json::Value> {
    let plan = qwen_llm::deepseek_v41::gguf::plan(gguf)?;
    ensure!(
        plan.config.vocab <= 1_048_576 && plan.layers <= 256,
        "paired result CPU envelope supports at most 1048576 logits and 256 layers"
    );
    let accepted = preflight_cold_prefill(
        ctx,
        gguf,
        extent.prefill_endpoint,
        extent.forwards,
        &[
            PrefillExecution::FullForward,
            PrefillExecution::ShrinkingCed,
        ],
        CALLER_CPU_BYTES,
    )
    .context("paired preweight Full/CED admission")?;
    ensure!(
        accepted.admission.is_admitted(),
        "paired preweight plan was not admitted"
    );
    let ced = accepted
        .ced
        .as_ref()
        .context("paired preweight plan cannot execute CED without fallback")?;
    Ok(serde_json::json!({
        "accepted": accepted.admission.is_admitted(),
        "prompt_rows": extent.prefill_endpoint,
        "capacity": extent.forwards,
        "executions": [PrefillExecution::FullForward, PrefillExecution::ShrinkingCed],
        "resident_metal_view_bytes": accepted.resident_storage.metal_view_bytes,
        "resident_private_shadow_bytes": accepted.resident_storage.private_shadow_bytes,
        "ced_incremental_metal_bytes": ced.incremental_metal_bytes,
        "ced_incremental_cpu_bytes": ced.incremental_cpu_bytes,
        "full_incremental_metal_bytes": accepted.full_incremental_metal_bytes,
        "full_incremental_cpu_bytes": accepted.full_incremental_cpu_bytes,
        "runtime_metal_bytes": accepted.runtime_metal_bytes,
        "runtime_cpu_bytes": accepted.runtime_cpu_bytes,
        "binding_cpu_bytes": accepted.binding_cpu_bytes,
        "caller_cpu_bytes": accepted.caller_cpu_bytes,
        "required_metal_bytes": accepted.required_metal_bytes,
        "required_cpu_bytes": accepted.required_cpu_bytes,
        "admission_reason": format!("{:?}", accepted.admission.reason),
        "admission_required_bytes": accepted.admission.required_bytes,
        "working_set_headroom_bytes": accepted.admission.working_set_headroom_bytes,
        "scope": "preweight_budget_not_reservation_later_state_and_prefill_gates_retained",
    }))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
enum Policy {
    A,
    B,
}

impl Policy {
    fn execution(self) -> PrefillExecution {
        match self {
            Self::A => PrefillExecution::FullForward,
            Self::B => PrefillExecution::ShrinkingCed,
        }
    }
    fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Warmup,
    Pilot,
    Main,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
struct Case {
    /// Zero-based native chronological index, including excluded calls.
    case_index: usize,
    policy: Policy,
    phase: Phase,
    block: usize,
    block_order: &'static str,
    order_in_block: usize,
    retained: bool,
}

fn schedule(runs: usize) -> Result<Vec<Case>> {
    ensure!(
        (2..=16).contains(&runs) && runs % 2 == 0,
        "paired runs must be even, 2..=16 per policy"
    );
    let mut cases = Vec::with_capacity(6 + 2 * runs);
    let mut block = |phase, block, order: &'static str, policies: &[Policy]| {
        for (slot, &policy) in policies.iter().enumerate() {
            cases.push(Case {
                case_index: cases.len(),
                policy,
                phase,
                block,
                block_order: order,
                order_in_block: slot,
                retained: phase == Phase::Main,
            });
        }
    };
    use Policy::{A, B};
    block(Phase::Warmup, 0, "AB", &[A, B]);
    block(Phase::Pilot, 0, "ABBA", &[A, B, B, A]);
    for index in 0..runs / 2 {
        if index % 2 == 0 {
            block(Phase::Main, index, "BAAB", &[B, A, A, B]);
        } else {
            block(Phase::Main, index, "ABBA", &[A, B, B, A]);
        }
    }
    Ok(cases)
}

/// Deliberately evaluates each policy independently, never a pooled ratio.
fn stable(samples: &[f64]) -> Result<()> {
    ensure!(
        samples.len() >= 2,
        "pilot requires two native samples per policy"
    );
    ensure!(
        samples.iter().all(|n| n.is_finite() && *n > 0.0),
        "native samples must be finite and positive: {samples:?}"
    );
    let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max = samples.iter().copied().fold(0.0, f64::max);
    ensure!(
        max / min <= 1.5,
        "within-policy native instability: max/min={} > 1.5, samples={samples:?}",
        max / min
    );
    Ok(())
}

fn drive<T>(
    runs: usize,
    mut checkpoint: impl FnMut() -> Result<()>,
    mut execute: impl FnMut(Case) -> Result<(f64, T)>,
) -> Result<Vec<(Case, T)>> {
    let cases = schedule(runs)?;
    let mut native = [Vec::new(), Vec::new()];
    let mut evidence = Vec::with_capacity(cases.len());
    for case in cases {
        // The previous callback has dropped state and drained its autorelease pool.
        checkpoint()?;
        let (sample, payload) = execute(case).with_context(|| {
            format!(
                "paired case {} {:?} {:?}",
                case.case_index, case.phase, case.policy
            )
        })?;
        if case.phase != Phase::Warmup {
            ensure!(
                sample.is_finite() && sample > 0.0,
                "paired case {} has invalid native time {sample}",
                case.case_index
            );
            native[case.policy.index()].push(sample);
            if native[case.policy.index()].len() >= 2 {
                stable(&native[case.policy.index()]).with_context(|| {
                    format!(
                        "paired {:?} stability gate after case {} (pilot baseline included)",
                        case.policy, case.case_index
                    )
                })?;
            }
        }
        evidence.push((case, payload));
        if (case.phase == Phase::Pilot && case.order_in_block == 3) || case.phase == Phase::Main {
            for policy in [Policy::A, Policy::B] {
                stable(&native[policy.index()]).with_context(|| {
                    format!(
                        "paired {:?} stability gate after case {} (pilot baseline included)",
                        policy, case.case_index
                    )
                })?;
            }
        }
    }
    checkpoint()?;
    Ok(evidence)
}

struct NativeCase {
    sample_ns: u64,
    semantics: FamilySemantics,
    report: V41BenchPrefillRep,
    logits_sha256_f32le: String,
}

fn native_case(
    bench: &mut dyn FamilyBench,
    row_ctx: &SuiteRowContext,
    n: usize,
    seed: u64,
    case: Case,
) -> Result<NativeCase> {
    objc2::rc::autoreleasepool(|_| {
        let result = (|| {
            bench.set_pair_execution(case.policy.execution())?;
            let row = run_family_pp_row(bench, row_ctx, n, 0, 1, true, seed)?;
            let report = bench
                .v41_prefill_diagnostics()
                .context("paired adapter omitted actual report")?;
            report
                .validate_pair(case.policy.execution(), n)
                .with_context(|| format!("actual paired report: {report:?}"))?;
            let logits = bench
                .last_logits()
                .context("paired adapter omitted endpoint logits")?;
            ensure!(
                logits.len() == bench.vocab_size() && logits.iter().all(|v| v.is_finite()),
                "paired endpoint logits are incomplete or non-finite"
            );
            // Reuse the i32-le SHA256 helper on the exact IEEE754 bits, after timing.
            let bits: Vec<i32> = logits.iter().map(|v| v.to_bits() as i32).collect();
            Ok(NativeCase {
                sample_ns: row.samples_ns[0],
                semantics: bench.semantics(),
                report,
                logits_sha256_f32le: token_ids_sha256_i32le(&bits),
            })
        })();
        // Also release on prefill failure, fallback, cancellation or validation error.
        bench.end_pair_case();
        result
    })
}

pub(crate) fn run(
    bench: &mut dyn FamilyBench,
    row_ctx: &SuiteRowContext,
    n: usize,
    runs: usize,
    seed: u64,
    preweight: &serde_json::Value,
) -> Result<Vec<BenchRow>> {
    let evidence = collect(runs, shutdown::checkpoint, |case| {
        native_case(bench, row_ctx, n, seed, case)
    })?;
    aggregate(
        row_ctx,
        n,
        runs,
        seed,
        bench.vocab_size(),
        preweight,
        &evidence,
    )
}

fn collect(
    runs: usize,
    checkpoint: impl FnMut() -> Result<()>,
    mut execute: impl FnMut(Case) -> Result<NativeCase>,
) -> Result<Vec<(Case, NativeCase)>> {
    let mut warm_full = None;
    drive(runs, checkpoint, |case| {
        let sample = execute(case)?;
        if case.case_index == 0 {
            warm_full = Some(sample.logits_sha256_f32le.clone());
        }
        ensure!(
            warm_full.as_ref() == Some(&sample.logits_sha256_f32le),
            "paired case {} endpoint head differs bitwise from warm Full; not quality qualification",
            case.case_index
        );
        tracing::info!(target: "qwen_diag", case=%serde_json::to_string(&case)?, sample_ns=sample.sample_ns,
            report=%serde_json::to_string(&sample.report)?, logits_sha256_f32le=%sample.logits_sha256_f32le, "native paired prefill case");
        Ok((sample.sample_ns as f64, sample))
    })
}

fn aggregate(
    row_ctx: &SuiteRowContext,
    n: usize,
    runs: usize,
    seed: u64,
    vocab: usize,
    preweight: &serde_json::Value,
    evidence: &[(Case, NativeCase)],
) -> Result<Vec<BenchRow>> {
    let chronology: Vec<_> = evidence
        .iter()
        .map(|(case, sample)| {
            serde_json::json!({
                "case": case,
                "requested_execution": case.policy.execution(),
                "sample_ns": sample.sample_ns,
                "pilot_status": match case.phase {
                    Phase::Warmup => "pending",
                    Phase::Pilot if case.order_in_block < 3 => "collecting",
                    _ => "accepted",
                },
                "prefill_mode": sample.semantics.prefill_mode,
                "prefill_chunk": sample.semantics.prefill_chunk,
                "prefill_report": sample.report,
                "endpoint_logits_sha256_f32le": sample.logits_sha256_f32le,
                "endpoint_logits_finite": true,
                "endpoint_bitwise_matches_warm_full": true,
            })
        })
        .collect();
    let ids = synthetic_prompt_ids(n, u32::try_from(vocab)?, seed);
    let mut rows = Vec::with_capacity(2);
    for policy in [Policy::A, Policy::B] {
        let retained: Vec<_> = evidence
            .iter()
            .filter(|(case, _)| case.retained && case.policy == policy)
            .collect();
        let wall: Vec<_> = retained
            .iter()
            .map(|(_, sample)| sample.sample_ns as f64 / 1e6)
            .collect();
        let mut row = family_row(
            row_ctx,
            retained[0].1.semantics,
            shape_label("pp", n, 0),
            n,
            0,
            &wall,
            true,
        );
        row.samples_ns = retained
            .iter()
            .map(|(_, sample)| sample.sample_ns)
            .collect();
        attach_v41_diagnostics(
            &mut row,
            retained
                .iter()
                .map(|(_, sample)| sample.report.clone())
                .collect(),
        );
        let diagnostics = row.diagnostics.get_or_insert_with(|| serde_json::json!({}));
        diagnostics["deepseek_v41"]["paired_benchmark"] = serde_json::json!({
            "protocol": "native_full_ced_abba_baab_v1",
            "policy": policy,
            "requested_execution": policy.execution(),
            "seed": seed,
            "prompt_ids_sha256_i32le": token_ids_sha256_i32le(&ids),
            "capacity": n,
            "retained_runs_per_policy": runs,
            "pilot_status": "accepted",
            "pilot_samples_per_policy": 2,
            "max_within_policy_spread": 1.5,
            "stability_baseline": "pilot_plus_main_rechecked_after_each_main_case",
            "preweight_plan": preweight,
            "native_chronological_cases": chronology,
            "timing_scope": "native_prefill_host_wall_including_endpoint_logits_copy_state_allocation_excluded",
            "quality_qualification": false,
            "host_validity": "external_continuous_monitor_required",
            "metal_debug_layer": std::env::var("MTL_DEBUG_LAYER").ok(),
        });
        rows.push(row);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family_bench::v41_pair_test_bench;
    use qwen_llm::deepseek_v41::forward::{CedCounters, CedMemoryEstimate, PrefillReport};
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    fn row_context() -> SuiteRowContext {
        SuiteRowContext {
            build_commit: "fixture",
            build_dirty: 0,
            model_filename: "fixture.gguf".into(),
            model_size: 123,
            model_n_params: 456,
            arch_kind: "deepseek_v41",
            family: "deepseek_v41",
            power: None,
            qwen_env: Default::default(),
        }
    }

    fn report(execution: PrefillExecution) -> PrefillReport {
        let ced = execution == PrefillExecution::ShrinkingCed;
        PrefillReport {
            execution,
            fallback: None,
            counters: ced.then(|| CedCounters {
                completed_commands: 2,
                encoded_commands: 2,
                ..Default::default()
            }),
            memory: ced.then(|| CedMemoryEstimate {
                incremental_metal_bytes: 123,
                ..Default::default()
            }),
            observed_incremental_metal_bytes: ced.then_some(100),
        }
    }

    #[cfg(target_os = "macos")]
    fn caller_envelope_peak_rss_bytes() -> u64 {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // Darwin reports ru_maxrss in bytes; no allocator or Metal observer is installed.
        assert_eq!(
            unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
            0
        );
        u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).unwrap()
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn caller_cpu_envelope_worst_case_isolated_rss_regression() {
        let module = module_path!().split_once("::").unwrap().1;
        let child_test = format!("{module}::caller_cpu_envelope_worst_case_rss_child");
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &child_test,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .output()
            .expect("run the exact CPU-only envelope child");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("paired-caller-envelope-rss"),
            "CPU-only envelope child failed or did not run: {}\n{stdout}\n{stderr}",
            output.status
        );
        eprint!("{stdout}{stderr}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "CPU-only RSS child; run by the isolated envelope regression"]
    fn caller_cpu_envelope_worst_case_rss_child() {
        use qwen_llm::deepseek_v41::forward::CedLayerCounters;

        const RUNS: usize = 16;
        const LAYERS: usize = 256;
        const VOCAB: usize = 1_048_576;

        let head: Vec<f32> = (0..VOCAB).map(|i| (i + 1) as f32).collect();
        let bits: Vec<i32> = head.iter().map(|v| v.to_bits() as i32).collect();
        let fingerprint = token_ids_sha256_i32le(&bits);
        assert_eq!(fingerprint.len(), 64);
        let head_and_bits_bytes = u64::try_from(
            head.capacity() * std::mem::size_of::<f32>()
                + bits.capacity() * std::mem::size_of::<i32>(),
        )
        .unwrap();
        assert!(head_and_bits_bytes >= 8 << 20);
        let remaining_bytes = CALLER_CPU_BYTES.checked_sub(head_and_bits_bytes).unwrap();
        // Charge live head/bits capacities explicitly, then measure the remaining
        // workload in a fresh single-test process. RSS deltas are observational:
        // startup high-water and allocator reuse prevent a formal heap proof.
        let before = caller_envelope_peak_rss_bytes();

        let ids: Vec<u32> = synthetic_prompt_ids(MAX_PP, VOCAB as u32, u64::MAX)
            .into_iter()
            .map(|id| id as u32)
            .collect();
        let mut bench = v41_pair_test_bench(
            || Ok(()),
            |_, ids, execution| {
                assert_eq!(ids.len(), MAX_PP);
                let mut actual = report(execution);
                if let Some(counters) = actual.counters.as_mut() {
                    // Saturated numeric payloads stress JSON digit storage, not
                    // execution geometry; no GPU work or allocation occurs.
                    *counters = CedCounters {
                        encoder_output_rows: usize::MAX,
                        publication_rows: usize::MAX,
                        decoder_output_rows: usize::MAX,
                        decoder_seed_rows: usize::MAX,
                        encoded_commands: usize::MAX,
                        completed_commands: usize::MAX,
                        layers: (0..LAYERS)
                            .map(|layer| CedLayerCounters {
                                layer,
                                encoder_output_rows: usize::MAX,
                                decoder_output_rows: usize::MAX,
                                seed_rows: usize::MAX,
                                publication_rows: usize::MAX,
                                selection_producer_rows: usize::MAX,
                            })
                            .collect(),
                    };
                    actual.memory = Some(CedMemoryEstimate {
                        retained_residual_bytes: u64::MAX,
                        retained_mix_bytes: u64::MAX,
                        retained_selected_bytes: u64::MAX,
                        retained_candidate_bytes: u64::MAX,
                        priced_retained_bytes: u64::MAX,
                        priced_chunk_bytes: u64::MAX,
                        priced_half_chunk_bytes: u64::MAX,
                        priced_publication_bytes: u64::MAX,
                        priced_seed_bytes: u64::MAX,
                        command_reserve_bytes: u64::MAX,
                        incremental_metal_bytes: u64::MAX,
                        incremental_cpu_bytes: u64::MAX,
                        retained_control_cpu_bytes: u64::MAX,
                        chunk_hash_cpu_bytes: u64::MAX,
                        process_ru_maxrss_before_bytes: Some(u64::MAX),
                        process_ru_maxrss_after_bytes: Some(u64::MAX),
                        observation_method: "Metal allocation samples after allocation, encoding and completion; cumulative process ru_maxrss, not an isolated CED or continuous driver peak",
                    });
                    actual.observed_incremental_metal_bytes = Some(u64::MAX);
                }
                Ok((Vec::new(), actual))
            },
        );
        let evidence = collect(
            RUNS,
            || Ok(()),
            |case| {
                let execution = case.policy.execution();
                bench.set_pair_execution(execution)?;
                bench.begin_pp_rep(0)?;
                bench.prefill(&ids)?;
                let report = bench.v41_prefill_diagnostics().unwrap();
                report.validate_pair(execution, MAX_PP)?;
                let sample = NativeCase {
                    sample_ns: 100_000_000,
                    semantics: bench.semantics(),
                    report,
                    logits_sha256_f32le: fingerprint.clone(),
                };
                bench.end_pair_case();
                Ok(sample)
            },
        )
        .unwrap();
        assert_eq!(evidence.len(), 38);
        let preweight = serde_json::json!({
            "accepted": true,
            "prompt_rows": MAX_PP,
            "capacity": MAX_PP,
            "executions": [PrefillExecution::FullForward, PrefillExecution::ShrinkingCed],
            "resident_metal_view_bytes": u64::MAX,
            "resident_private_shadow_bytes": u64::MAX,
            "ced_incremental_metal_bytes": u64::MAX,
            "ced_incremental_cpu_bytes": u64::MAX,
            "full_incremental_metal_bytes": u64::MAX,
            "full_incremental_cpu_bytes": u64::MAX,
            "runtime_metal_bytes": u64::MAX,
            "runtime_cpu_bytes": u64::MAX,
            "binding_cpu_bytes": u64::MAX,
            "caller_cpu_bytes": CALLER_CPU_BYTES,
            "required_metal_bytes": u64::MAX,
            "required_cpu_bytes": u64::MAX,
            "admission_reason": "AdmittedWithHostBudget",
            "admission_required_bytes": u64::MAX,
            "working_set_headroom_bytes": u64::MAX,
            "scope": "preweight_budget_not_reservation_later_state_and_prefill_gates_retained",
        });
        let rows = aggregate(
            &row_context(),
            MAX_PP,
            RUNS,
            u64::MAX,
            VOCAB,
            &preweight,
            &evidence,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        for row in &rows {
            assert_eq!(row.n_repetitions, RUNS);
            assert_eq!(row.samples_ns.len(), RUNS);
            let diagnostics = &row.diagnostics.as_ref().unwrap()["deepseek_v41"];
            assert_eq!(
                diagnostics["prefill_repetitions"].as_array().unwrap().len(),
                RUNS
            );
            let chronology = diagnostics["paired_benchmark"]["native_chronological_cases"]
                .as_array()
                .unwrap();
            assert_eq!(chronology.len(), 38);
            for (case, (expected, _)) in chronology.iter().zip(&evidence) {
                let layers = &case["prefill_report"]["calls"][0]["counters"]["layers"];
                if expected.policy == Policy::B {
                    assert_eq!(layers.as_array().unwrap().len(), LAYERS);
                } else {
                    assert!(layers.is_null());
                }
            }
        }
        // Exercise the suite's final serialization without constructing an
        // additional parsed DOM that production does not retain.
        let json = serde_json::to_string(&rows).unwrap();
        assert_eq!(json.matches("\"native_chronological_cases\"").count(), 2);
        assert_eq!(json.matches("\"layers\":[").count(), 2 * 19 + RUNS);
        std::hint::black_box((&head, &bits, &rows, &evidence, &json));
        let result_peak_delta = caller_envelope_peak_rss_bytes()
            .checked_sub(before)
            .unwrap();
        let charged_bytes = head_and_bits_bytes.checked_add(result_peak_delta).unwrap();
        println!(
            "paired-caller-envelope-rss head_and_bits_bytes={head_and_bits_bytes} result_peak_delta_bytes={result_peak_delta} remaining_bytes={remaining_bytes} charged_bytes={charged_bytes} limit_bytes={CALLER_CPU_BYTES} json_bytes={} measurement=observational_not_heap_proof",
            json.len()
        );
        assert!(
            result_peak_delta <= remaining_bytes,
            "caller envelope exceeded: live head/bits {head_and_bits_bytes} + result RSS delta {result_peak_delta} = {charged_bytes} > {CALLER_CPU_BYTES}; no extra tolerance"
        );
        drop((head, bits, rows, evidence, json));
    }

    #[test]
    fn bounded_schedule_is_two_policies_with_both_warm_before_pilot() {
        use Policy::{A, B};
        for runs in (2..=16).step_by(2) {
            let cases = schedule(runs).unwrap();
            assert_eq!(cases.len(), 6 + 2 * runs);
            assert_eq!(
                cases.iter().take(6).map(|c| c.policy).collect::<Vec<_>>(),
                [A, B, A, B, B, A]
            );
            assert!(
                cases[..2]
                    .iter()
                    .all(|c| c.phase == Phase::Warmup && !c.retained)
            );
            assert!(
                cases[2..6]
                    .iter()
                    .all(|c| c.phase == Phase::Pilot && !c.retained)
            );
            for policy in [A, B] {
                assert_eq!(cases[2..6].iter().filter(|c| c.policy == policy).count(), 2);
                assert_eq!(
                    cases
                        .iter()
                        .filter(|c| c.retained && c.policy == policy)
                        .count(),
                    runs
                );
            }
            for (index, block) in cases[6..].chunks_exact(4).enumerate() {
                let expected = if index % 2 == 0 {
                    [B, A, A, B]
                } else {
                    [A, B, B, A]
                };
                assert_eq!(block.iter().map(|c| c.policy).collect::<Vec<_>>(), expected);
                assert_eq!(block[0].policy, block[3].policy);
            }
            assert!(cases.iter().enumerate().all(|(i, c)| c.case_index == i));
        }
        for runs in [0, 1, 3, 17, usize::MAX] {
            assert!(schedule(runs).is_err());
        }
    }

    #[test]
    fn gate_is_finite_positive_with_inclusive_ratio_and_no_policy_pooling() {
        assert!(stable(&[2., 3.]).is_ok());
        for values in [
            vec![],
            vec![1.],
            vec![1., 1.50001],
            vec![0., 1.],
            vec![-1., 1.],
            vec![1., f64::NAN],
            vec![1., f64::INFINITY],
        ] {
            assert!(stable(&values).is_err(), "{values:?}");
        }
        let evidence = drive(
            2,
            || Ok(()),
            |case| Ok((if case.policy == Policy::A { 100. } else { 10. }, ())),
        )
        .unwrap();
        assert_eq!(
            evidence.len(),
            10,
            "10x cross-policy difference is not instability"
        );
    }

    #[test]
    fn pilot_and_every_main_case_gate_before_any_remaining_case() {
        for bad_index in 2..14 {
            let calls = Cell::new(0);
            let result = drive(
                4,
                || Ok(()),
                |case| {
                    calls.set(calls.get() + 1);
                    Ok((
                        if case.case_index == bad_index {
                            200.
                        } else {
                            100.
                        },
                        (),
                    ))
                },
            );
            assert!(result.is_err());
            assert_eq!(
                calls.get(),
                match bad_index {
                    2 | 5 => 6,
                    3 | 4 => 5,
                    _ => bad_index + 1,
                }
            );
        }
        for invalid in [0., -1., f64::NAN, f64::INFINITY] {
            let calls = Cell::new(0);
            assert!(
                drive(
                    2,
                    || Ok(()),
                    |case| {
                        calls.set(calls.get() + 1);
                        Ok((if case.case_index == 2 { invalid } else { 1. }, ()))
                    }
                )
                .is_err()
            );
            assert_eq!(calls.get(), 3);
        }
    }

    #[test]
    fn cancellation_and_execution_error_never_schedule_later_cases() {
        for stop in 0..=10 {
            let checkpoints = Cell::new(0);
            let calls = Cell::new(0);
            let result = drive(
                2,
                || {
                    let index = checkpoints.get();
                    checkpoints.set(index + 1);
                    ensure!(index != stop, "fixture cancellation");
                    Ok(())
                },
                |_| {
                    calls.set(calls.get() + 1);
                    Ok((1., ()))
                },
            );
            assert!(result.is_err());
            assert_eq!(calls.get(), stop);
        }
        for stop in 0..10 {
            let calls = Cell::new(0);
            assert!(
                drive(
                    2,
                    || Ok(()),
                    |case| {
                        calls.set(calls.get() + 1);
                        ensure!(case.case_index != stop, "fixture error");
                        Ok((1., ()))
                    }
                )
                .is_err()
            );
            assert_eq!(calls.get(), stop + 1);
        }
    }

    struct ColdState {
        live: Rc<Cell<usize>>,
        rows: usize,
    }
    impl Drop for ColdState {
        fn drop(&mut self) {
            self.live.set(self.live.get() - 1);
        }
    }

    #[test]
    fn native_fake_adapter_preserves_ids_cold_state_full_reports_and_json_statistics() {
        let live = Rc::new(Cell::new(0));
        let calls = RefCell::new(Vec::new());
        let mut bench = v41_pair_test_bench(
            || {
                assert_eq!(
                    live.get(),
                    0,
                    "no overlapping states, including before allocation"
                );
                live.set(1);
                Ok(ColdState {
                    live: live.clone(),
                    rows: 0,
                })
            },
            |state, ids, execution| {
                assert_eq!(state.rows, 0);
                assert_eq!(ids.len(), 3);
                calls.borrow_mut().push((ids.to_vec(), execution));
                state.rows += ids.len();
                Ok((vec![0.; 257], report(execution)))
            },
        );
        let ctx = row_context();
        let evidence = collect(
            2,
            || {
                assert_eq!(live.get(), 0);
                Ok(())
            },
            |case| {
                let mut sample = native_case(&mut bench, &ctx, 3, 42, case)?;
                assert_eq!(
                    live.get(),
                    0,
                    "state released before returning from the case pool"
                );
                // Deterministic test timing; production always uses the native timer unchanged.
                sample.sample_ns = match (case.phase, case.policy) {
                    (Phase::Warmup, _) => 1_000_000_000,
                    (Phase::Pilot, Policy::A) => 100_000_000,
                    (Phase::Pilot, Policy::B) => 10_000_000,
                    (Phase::Main, Policy::A) => 110_000_000,
                    (Phase::Main, Policy::B) => 11_000_000,
                };
                Ok(sample)
            },
        )
        .unwrap();
        let calls = calls.borrow();
        assert_eq!(calls.len(), 10);
        assert!(calls.iter().all(|(ids, _)| *ids == calls[0].0));
        assert_eq!(
            calls.iter().map(|(_, e)| *e).collect::<Vec<_>>(),
            schedule(2)
                .unwrap()
                .iter()
                .map(|c| c.policy.execution())
                .collect::<Vec<_>>()
        );
        let preweight = serde_json::json!({"accepted": true, "caller_cpu_bytes": CALLER_CPU_BYTES});
        let rows = aggregate(&ctx, 3, 2, 42, 257, &preweight, &evidence).unwrap();
        let json = serde_json::to_value(rows).unwrap();
        for (index, expected_ns) in [110_000_000u64, 11_000_000].into_iter().enumerate() {
            let row = &json[index];
            assert_eq!(row["schema_version"], BENCH_SCHEMA_VERSION);
            assert_eq!(row["n_repetitions"], 2);
            assert_eq!(
                row["samples_ns"],
                serde_json::json!([expected_ns, expected_ns])
            );
            assert_eq!(row["avg_ns"], expected_ns);
            assert!(row["avg_gpu_ns"].is_null());
            let diagnostics = &row["diagnostics"]["deepseek_v41"];
            assert_eq!(
                diagnostics["prefill_repetitions"].as_array().unwrap().len(),
                2
            );
            let paired = &diagnostics["paired_benchmark"];
            assert_eq!(paired["preweight_plan"], preweight);
            assert_eq!(paired["quality_qualification"], false);
            assert_eq!(paired["pilot_status"], "accepted");
            let cases = paired["native_chronological_cases"].as_array().unwrap();
            assert_eq!(cases.len(), 10);
            assert_eq!(
                cases
                    .iter()
                    .filter(|c| c["case"]["retained"] == true)
                    .count(),
                4
            );
            assert_eq!(cases[0]["case"]["phase"], "warmup");
            assert_eq!(cases[2]["case"]["phase"], "pilot");
            assert_eq!(cases[5]["pilot_status"], "accepted");
            assert_eq!(
                cases[1]["prefill_report"]["calls"][0]["memory"]["incremental_metal_bytes"],
                123
            );
            assert_eq!(
                cases[1]["prefill_report"]["calls"][0]["counters"]["completed_commands"],
                2
            );
        }
        assert_eq!(
            json[0]["prefill_mode"],
            crate::family_options::V41_PREFILL_MODE
        );
        assert_eq!(json[1]["prefill_mode"], "shrinking_ced+final_logits_copy");
        assert_eq!(
            json[0]["diagnostics"]["deepseek_v41"]["prefill_repetitions"][0]["calls"][0]["execution"],
            "full_forward"
        );
    }

    #[test]
    fn fallback_errors_and_bad_heads_release_state_and_halt_the_protocol() {
        for stop in 0..10 {
            for fault in ["fallback", "wrong_policy", "error", "nonfinite", "bits"] {
                let live = Rc::new(Cell::new(0));
                let calls = Cell::new(0);
                let mut bench = v41_pair_test_bench(
                    || {
                        assert_eq!(live.get(), 0);
                        live.set(1);
                        Ok(ColdState {
                            live: live.clone(),
                            rows: 0,
                        })
                    },
                    |_, _, execution| {
                        let current = calls.get();
                        calls.set(current + 1);
                        let mut actual = report(execution);
                        let mut logits = vec![0.; 257];
                        if current == stop {
                            match fault {
                                "fallback" => actual.fallback = Some("fixture CED refusal".into()),
                                "wrong_policy" => {
                                    actual.execution = if execution == PrefillExecution::FullForward
                                    {
                                        PrefillExecution::ShrinkingCed
                                    } else {
                                        PrefillExecution::FullForward
                                    }
                                }
                                "error" => anyhow::bail!("fixture native failure"),
                                "nonfinite" => logits[256] = f32::NAN,
                                "bits" => logits[256] = -0.,
                                _ => unreachable!(),
                            }
                        }
                        Ok((logits, actual))
                    },
                );
                let ctx = row_context();
                let result = collect(
                    2,
                    || {
                        assert_eq!(live.get(), 0);
                        Ok(())
                    },
                    |case| {
                        let mut sample = native_case(&mut bench, &ctx, 3, 42, case)?;
                        sample.sample_ns = 100;
                        Ok(sample)
                    },
                );
                assert!(result.is_err(), "stop={stop}, fault={fault}");
                // A bad first warm Full establishes the bitwise baseline; warm CED rejects it.
                let expected = if stop == 0 && fault == "bits" {
                    2
                } else {
                    stop + 1
                };
                assert_eq!(calls.get(), expected, "stop={stop}, fault={fault}");
                assert_eq!(live.get(), 0);
            }
        }
    }

    #[test]
    fn malformed_ced_work_or_memory_stops_before_later_cases() {
        for fault in [
            "counters",
            "incomplete",
            "memory",
            "observation",
            "over_bound",
        ] {
            let live = Rc::new(Cell::new(0));
            let calls = Cell::new(0);
            let mut bench = v41_pair_test_bench(
                || {
                    assert_eq!(live.get(), 0);
                    live.set(1);
                    Ok(ColdState {
                        live: live.clone(),
                        rows: 0,
                    })
                },
                |_, _, execution| {
                    calls.set(calls.get() + 1);
                    let mut actual = report(execution);
                    if execution == PrefillExecution::ShrinkingCed {
                        match fault {
                            "counters" => actual.counters = None,
                            "incomplete" => {
                                actual.counters.as_mut().unwrap().completed_commands = 1
                            }
                            "memory" => actual.memory = None,
                            "observation" => actual.observed_incremental_metal_bytes = None,
                            "over_bound" => actual.observed_incremental_metal_bytes = Some(124),
                            _ => unreachable!(),
                        }
                    }
                    Ok((vec![0.; 257], actual))
                },
            );
            let ctx = row_context();
            assert!(
                collect(
                    2,
                    || Ok(()),
                    |case| {
                        let mut sample = native_case(&mut bench, &ctx, 3, 42, case)?;
                        sample.sample_ns = 100;
                        Ok(sample)
                    }
                )
                .is_err(),
                "fault={fault}"
            );
            assert_eq!(calls.get(), 2, "fault={fault}");
            assert_eq!(live.get(), 0);
        }
    }

    #[test]
    fn paired_cli_is_opt_in_and_validation_does_not_mutate_environment() {
        use clap::Parser;
        let args = Args::try_parse_from([
            "qwen-bench",
            "suite",
            "-m",
            "fixture.gguf",
            "--pp",
            "128",
            "--runs",
            "2",
            "--v41-ced-pair",
        ])
        .unwrap();
        let Cmd::Suite(args) = args.cmd else {
            panic!("suite")
        };
        assert!(args.v41_ced_pair);
        assert!(
            validate_args(
                ModelFamily::DeepSeekV41,
                &args.pp,
                &args.tg,
                &args.depth,
                args.runs,
                args.no_warmup,
                args.prefill_chunk
            )
            .is_ok()
        );
        for (family, pp, tg, depths, runs, warmup, chunk) in [
            (
                ModelFamily::DeepSeek4,
                vec![128],
                vec![],
                vec![0],
                2,
                false,
                None,
            ),
            (
                ModelFamily::DeepSeekV41,
                vec![128, 256],
                vec![],
                vec![0],
                2,
                false,
                None,
            ),
            (
                ModelFamily::DeepSeekV41,
                vec![8193],
                vec![],
                vec![0],
                2,
                false,
                None,
            ),
            (
                ModelFamily::DeepSeekV41,
                vec![128],
                vec![1],
                vec![0],
                2,
                false,
                None,
            ),
            (
                ModelFamily::DeepSeekV41,
                vec![128],
                vec![],
                vec![1],
                2,
                false,
                None,
            ),
            (
                ModelFamily::DeepSeekV41,
                vec![128],
                vec![],
                vec![0],
                3,
                false,
                None,
            ),
            (
                ModelFamily::DeepSeekV41,
                vec![128],
                vec![],
                vec![0],
                2,
                true,
                None,
            ),
            (
                ModelFamily::DeepSeekV41,
                vec![128],
                vec![],
                vec![0],
                2,
                false,
                Some(128),
            ),
        ] {
            assert!(validate_args(family, &pp, &tg, &depths, runs, warmup, chunk).is_err());
        }
        let args =
            Args::try_parse_from(["qwen-bench", "suite", "-m", "fixture.gguf", "--pp", "128"])
                .unwrap();
        let Cmd::Suite(args) = args.cmd else {
            panic!("suite")
        };
        assert!(!args.v41_ced_pair);
        assert_eq!(args.runs, 1, "ordinary defaults unchanged");
        let get = |name: &str| (name == "MTL_DEBUG_LAYER").then(|| "1".into());
        assert!(validate_environment(get, &[]).is_ok());
        assert!(validate_environment(|_| None, &[]).is_err());
        for name in [
            "QWEN_V41_PREFILL_EXECUTION",
            "QWEN_V41_SINK_TENSOR",
            "QWEN_V41_PREFILL_STAGES",
            "QWEN_V41_CENSUS",
            "QWEN_MATMAT_F16_HALF_ACT",
            "QWEN_MATVEC_IQ3_XXS_FAST",
        ] {
            assert!(validate_environment(get, &[name.into()]).is_err());
        }
        assert!(
            validate_environment(
                |name| match name {
                    "MTL_DEBUG_LAYER" => Some("1".into()),
                    "QWEN_METAL_LEASE_SKIP_WIRED_GATE" => Some("1".into()),
                    _ => None,
                },
                &[]
            )
            .is_err()
        );
    }
}
