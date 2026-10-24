//! Bounded resident Full/CED timing protocol. This is not quality qualification.

use super::*;
use crate::family_bench::{
    FamilyBench, FamilyBenchExtent, FamilySemantics, V41BenchPrefillRep, attach_v41_diagnostics,
    family_row, run_family_pp_row, shape_label,
};
use anyhow::ensure;
use qwen_llm::deepseek_v41::forward::profile;
use qwen_llm::deepseek_v41::forward::{PrefillExecution, preflight_cold_prefill};
use qwen_llm::metal::MetalPipelineCacheMetrics;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::pid_metrics::PidSnapshot;
use std::io::Write;

// At most 38 cases, 256 layer counters per report, two JSON rows and a 1M-logit head.
const CALLER_CPU_BYTES: u64 = 64 << 20;
const DIAGNOSTIC_CPU_BYTES: u64 = 16 << 20;
// Four encoder chunks plus at most four seed/four output chunks and a scalar
// tail per layer, with room for fixed aggregate host spans. No sampled stages.
const MAX_DIAGNOSTIC_OPERATIONS: usize = profile::PAIRED_OPERATION_CAPACITY;
const MAX_PP: usize = 8192;
const MAX_LAYERS: usize = 256;

fn diagnostic_operation_bound(rows: usize, layers: usize) -> usize {
    let chunks = rows.div_ceil(qwen_llm::deepseek_v41::forward::PREFILL_CHUNK_ROWS);
    chunks + layers * (2 * chunks + 1) + 4
}

fn caller_cpu_bytes(diagnostic: bool) -> u64 {
    CALLER_CPU_BYTES + if diagnostic { DIAGNOSTIC_CPU_BYTES } else { 0 }
}

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
    diagnostic: bool,
) -> Result<serde_json::Value> {
    let plan = qwen_llm::deepseek_v41::gguf::plan(gguf)?;
    ensure!(
        plan.config.vocab <= 1_048_576 && plan.layers <= MAX_LAYERS,
        "paired result CPU envelope supports at most 1048576 logits and 256 layers"
    );
    if diagnostic {
        ensure!(
            extent.prefill_endpoint <= MAX_PP,
            "diagnostic prompt exceeds producer bound"
        );
        ensure!(
            diagnostic_operation_bound(extent.prefill_endpoint, plan.layers)
                <= MAX_DIAGNOSTIC_OPERATIONS,
            "diagnostic producer inventory exceeds its CPU reserve"
        );
    }
    let accepted = preflight_cold_prefill(
        ctx,
        gguf,
        extent.prefill_endpoint,
        extent.forwards,
        &[
            PrefillExecution::FullForward,
            PrefillExecution::ShrinkingCed,
        ],
        caller_cpu_bytes(diagnostic),
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
        "binding_metal_bytes": accepted.binding_metal_bytes,
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

#[derive(Debug)]
struct NativeCase {
    sample_ns: u64,
    semantics: FamilySemantics,
    report: V41BenchPrefillRep,
    logits_sha256_f32le: String,
    instrumented: bool,
}

struct CaseState<'a>(&'a mut dyn FamilyBench);

impl CaseState<'_> {
    fn bench(&mut self) -> &mut dyn FamilyBench {
        &mut *self.0
    }
}

impl Drop for CaseState<'_> {
    fn drop(&mut self) {
        self.0.end_pair_case();
    }
}

fn native_case(
    bench: &mut dyn FamilyBench,
    row_ctx: &SuiteRowContext,
    n: usize,
    seed: u64,
    case: Case,
) -> Result<NativeCase> {
    native_case_with(
        bench,
        row_ctx,
        n,
        seed,
        case,
        |body| objc2::rc::autoreleasepool(|_| body()),
        fingerprint_logits,
    )
}

fn fingerprint_logits(logits: &[f32]) -> String {
    let bits: Vec<i32> = logits.iter().map(|v| v.to_bits() as i32).collect();
    token_ids_sha256_i32le(&bits)
}

fn native_case_with(
    bench: &mut dyn FamilyBench,
    row_ctx: &SuiteRowContext,
    n: usize,
    seed: u64,
    case: Case,
    pool: impl FnOnce(&mut dyn FnMut() -> Result<NativeCase>) -> Result<NativeCase>,
    mut fingerprint: impl FnMut(&[f32]) -> String,
) -> Result<NativeCase> {
    pool(&mut || {
        let mut state = CaseState(&mut *bench);
        let bench = state.bench();
        (|| {
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
            Ok(NativeCase {
                sample_ns: row.samples_ns[0],
                semantics: bench.semantics(),
                report,
                logits_sha256_f32le: fingerprint(logits),
                instrumented: false,
            })
        })()
    })
}

#[derive(serde::Serialize)]
struct DiagnosticResources {
    process_pageins: u64,
    process_disk_read_bytes: u64,
    process_resident_size_bytes: u64,
    process_phys_footprint_bytes: u64,
    device_current_allocated_bytes: u64,
}

impl DiagnosticResources {
    fn capture(ctx: &MetalContext) -> Result<Self> {
        let process = PidSnapshot::now().context("paired diagnostic process snapshot")?;
        Ok(Self {
            process_pageins: process.pageins,
            process_disk_read_bytes: process.diskio_bytesread,
            process_resident_size_bytes: process.resident_size,
            process_phys_footprint_bytes: process.phys_footprint,
            device_current_allocated_bytes: ctx.current_allocated_size(),
        })
    }

    fn delta_since(&self, before: &Self) -> DiagnosticResourceDelta {
        DiagnosticResourceDelta {
            process_pageins: self.process_pageins.saturating_sub(before.process_pageins),
            process_disk_read_bytes: self
                .process_disk_read_bytes
                .saturating_sub(before.process_disk_read_bytes),
            process_resident_size_bytes: i128::from(self.process_resident_size_bytes)
                - i128::from(before.process_resident_size_bytes),
            process_phys_footprint_bytes: i128::from(self.process_phys_footprint_bytes)
                - i128::from(before.process_phys_footprint_bytes),
            device_current_allocated_bytes: i128::from(self.device_current_allocated_bytes)
                - i128::from(before.device_current_allocated_bytes),
        }
    }
}

#[derive(serde::Serialize)]
struct DiagnosticResourceDelta {
    process_pageins: u64,
    process_disk_read_bytes: u64,
    process_resident_size_bytes: i128,
    process_phys_footprint_bytes: i128,
    device_current_allocated_bytes: i128,
}

#[derive(serde::Serialize)]
struct CommandTotals {
    completed_submits: usize,
    invalid_gpu_timestamps: usize,
    gpu_ms_valid_completed_only: Option<f64>,
    encode_ms: f64,
    commit_wait_ms_including_gpu: f64,
    outer_wall_minus_command_encode_and_wait_ms: f64,
}

fn command_totals(report: &profile::Report) -> CommandTotals {
    let mut totals = CommandTotals {
        completed_submits: 0,
        invalid_gpu_timestamps: 0,
        gpu_ms_valid_completed_only: None,
        encode_ms: 0.0,
        commit_wait_ms_including_gpu: 0.0,
        outer_wall_minus_command_encode_and_wait_ms: 0.0,
    };
    for op in &report.operations {
        totals.completed_submits += op.completed_submits;
        totals.invalid_gpu_timestamps += op.invalid_gpu_timestamps;
        if let Some(ms) = op.gpu_ms {
            *totals.gpu_ms_valid_completed_only.get_or_insert(0.0) += ms;
        }
        totals.encode_ms += op.encode_ms;
        totals.commit_wait_ms_including_gpu += op.commit_wait_ms;
    }
    totals.outer_wall_minus_command_encode_and_wait_ms =
        report.token_wall_ms - totals.encode_ms - totals.commit_wait_ms_including_gpu;
    totals
}

fn validate_diagnostic_report(report: &profile::Report) -> Result<()> {
    ensure!(
        report.process_pageins.is_some() && report.process_disk_read_bytes.is_some(),
        "paired diagnostic profile omitted required process I/O counters"
    );
    ensure!(
        report.operations.len() <= MAX_DIAGNOSTIC_OPERATIONS,
        "paired diagnostic profile exceeded its bounded operation inventory"
    );
    Ok(())
}

fn stream_diagnostic(
    writer: &mut impl Write,
    case: Case,
    event: &'static str,
    observations: &impl serde::Serialize,
) -> Result<()> {
    #[derive(serde::Serialize)]
    struct Event<'a, T> {
        event: &'static str,
        case: Case,
        instrumented: bool,
        performance_evidence: bool,
        observations: &'a T,
    }
    serde_json::to_writer(
        &mut *writer,
        &Event {
            event,
            case,
            instrumented: true,
            performance_evidence: false,
            observations,
        },
    )
    .context("stream paired diagnostic observation")?;
    writer.write_all(b"\n")?;
    writer
        .flush()
        .context("flush paired diagnostic observation")
}

fn diagnostic_case(
    ctx: &MetalContext,
    bench: &mut dyn FamilyBench,
    row_ctx: &SuiteRowContext,
    n: usize,
    seed: u64,
    case: Case,
) -> Result<NativeCase> {
    let stderr = std::io::stderr();
    diagnostic_case_with(
        &mut NativeDiagnosticProbe(ctx),
        case,
        || native_case(bench, row_ctx, n, seed, case),
        |event| stream_diagnostic_event(&mut stderr.lock(), case, event),
    )
}

trait DiagnosticProbe {
    type Scope;
    fn resources(&mut self) -> Result<DiagnosticResources>;
    fn pipelines(&self) -> MetalPipelineCacheMetrics;
    fn begin(&mut self) -> Result<Self::Scope>;
    fn finish(&mut self, scope: Self::Scope) -> profile::Report;
}

struct NativeDiagnosticProbe<'a>(&'a MetalContext);

impl DiagnosticProbe for NativeDiagnosticProbe<'_> {
    type Scope = profile::Scope;

    fn resources(&mut self) -> Result<DiagnosticResources> {
        DiagnosticResources::capture(self.0)
    }

    fn pipelines(&self) -> MetalPipelineCacheMetrics {
        self.0.pipeline_cache_metrics()
    }

    fn begin(&mut self) -> Result<Self::Scope> {
        profile::Scope::begin().map_err(anyhow::Error::msg)
    }

    fn finish(&mut self, scope: Self::Scope) -> profile::Report {
        scope.finish()
    }
}

#[derive(serde::Serialize)]
struct DiagnosticObservation<'a> {
    native_case_succeeded: bool,
    collection_succeeded: bool,
    cross_case_gate_status: &'static str,
    native_prefill_wall_ns: Option<u64>,
    native_timing_scope: &'static str,
    profile_and_pipeline_scope: &'static str,
    resource_scope: &'static str,
    gap_interpretation: &'static str,
    resources_before: &'a DiagnosticResources,
    resources_after_pool_drain: Option<&'a DiagnosticResources>,
    resource_delta: Option<DiagnosticResourceDelta>,
    pipeline_cache_delta: [u64; 3],
    pipeline_cache_delta_fields: [&'static str; 3],
    command_totals: CommandTotals,
    profile: &'a profile::Report,
}

enum DiagnosticEvent<'a> {
    Start(&'a DiagnosticResources),
    Finish(DiagnosticObservation<'a>),
}

fn stream_diagnostic_event(
    writer: &mut impl Write,
    case: Case,
    event: DiagnosticEvent<'_>,
) -> Result<()> {
    match event {
        DiagnosticEvent::Start(before) => {
            stream_diagnostic(writer, case, "v41_pair_diagnostic_start", before)
        }
        DiagnosticEvent::Finish(observation) => {
            stream_diagnostic(writer, case, "v41_pair_diagnostic_finish", &observation)
        }
    }
}

fn diagnostic_case_with<P: DiagnosticProbe>(
    probe: &mut P,
    case: Case,
    execute: impl FnOnce() -> Result<NativeCase>,
    mut emit: impl FnMut(DiagnosticEvent<'_>) -> Result<()>,
) -> Result<NativeCase> {
    let before = probe.resources()?;
    let pipeline_before = probe.pipelines();
    emit(DiagnosticEvent::Start(&before))?;
    let scope = probe.begin()?;
    let result = execute();
    // Execution has released state and drained its pool before returning.
    let report = probe.finish(scope);
    let after = probe.resources();
    let pipeline = probe.pipelines().saturating_delta_since(pipeline_before);
    let validation = validate_diagnostic_report(&report);
    emit(DiagnosticEvent::Finish(DiagnosticObservation {
        native_case_succeeded: result.is_ok(),
        collection_succeeded: after.is_ok() && validation.is_ok(),
        cross_case_gate_status: "fingerprint_and_stability_checks_pending",
        native_prefill_wall_ns: result.as_ref().ok().map(|sample| sample.sample_ns),
        native_timing_scope: "prefill_plus_endpoint_copy_state_allocation_excluded_instrumented",
        profile_and_pipeline_scope: "outer_native_case_including_state_allocation_validation_fingerprint_reporting_teardown_and_pool_drain",
        resource_scope: "before_case_to_after_state_drop_and_case_pool_drain_current_values_not_peaks",
        gap_interpretation: "outer_wall_minus_encode_and_wait_is_unattributed_wall_not_cpu_time_gpu_is_already_inside_wait",
        resources_before: &before,
        resources_after_pool_drain: after.as_ref().ok(),
        resource_delta: after.as_ref().ok().map(|a| a.delta_since(&before)),
        pipeline_cache_delta: [
            pipeline.misses,
            pipeline.miss_wall_ns,
            pipeline.compiler_wall_ns,
        ],
        pipeline_cache_delta_fields: ["misses", "miss_wall_ns", "compiler_wall_ns"],
        command_totals: command_totals(&report),
        profile: &report,
    }))?;
    let mut sample = result?;
    after?;
    validation?;
    sample.instrumented = true;
    Ok(sample)
}

/// Suite is the sole caller and owns a fresh, private context for the entire
/// diagnostic run. Counters are enabled once there; only snapshots happen here.
pub(crate) fn run(
    bench: &mut dyn FamilyBench,
    row_ctx: &SuiteRowContext,
    n: usize,
    runs: usize,
    seed: u64,
    preweight: &serde_json::Value,
    diagnostic_ctx: Option<&MetalContext>,
) -> Result<Vec<BenchRow>> {
    let evidence = if let Some(ctx) = diagnostic_ctx {
        ensure!(
            preweight["caller_cpu_bytes"].as_u64() == Some(caller_cpu_bytes(true)),
            "paired diagnostic requires its additional preweight CPU reserve"
        );
        collect(runs, shutdown::checkpoint, |case| {
            diagnostic_case(ctx, bench, row_ctx, n, seed, case)
        })?
    } else {
        collect(runs, shutdown::checkpoint, |case| {
            native_case(bench, row_ctx, n, seed, case)
        })?
    };
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
        if sample.instrumented {
            tracing::info!(target: "qwen_diag", case=%serde_json::to_string(&case)?, sample_ns=sample.sample_ns,
                instrumented=true, performance_evidence=false,
                report=%serde_json::to_string(&sample.report)?, logits_sha256_f32le=%sample.logits_sha256_f32le, "instrumented paired prefill case");
        } else {
            tracing::info!(target: "qwen_diag", case=%serde_json::to_string(&case)?, sample_ns=sample.sample_ns,
                report=%serde_json::to_string(&sample.report)?, logits_sha256_f32le=%sample.logits_sha256_f32le, "native paired prefill case");
        }
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
    let instrumented = evidence
        .first()
        .is_some_and(|(_, sample)| sample.instrumented);
    ensure!(
        evidence
            .iter()
            .all(|(_, sample)| sample.instrumented == instrumented),
        "paired evidence mixes diagnostic and uninstrumented cases"
    );
    let chronology: Vec<_> = evidence
        .iter()
        .map(|(case, sample)| {
            let mut entry = serde_json::json!({
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
            });
            if instrumented {
                entry["instrumented"] = true.into();
                entry["performance_evidence"] = false.into();
            }
            entry
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
        if instrumented {
            diagnostics["instrumented"] = true.into();
            diagnostics["performance_evidence"] = false.into();
            let paired = &mut diagnostics["deepseek_v41"]["paired_benchmark"];
            paired["protocol"] = "instrumented_full_ced_abba_baab_diagnostic_v1".into();
            paired["instrumented"] = true.into();
            paired["performance_evidence"] = false.into();
            paired["diagnostic_observations"] = "streamed_to_stderr_not_retained".into();
            paired["diagnostic_cpu_reserve_bytes"] = DIAGNOSTIC_CPU_BYTES.into();
        }
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

    fn profile_report(operations: Vec<profile::Operation>) -> profile::Report {
        profile::Report {
            token_wall_ms: 100.0,
            process_pageins: Some(3),
            process_disk_read_bytes: Some(4096),
            operations,
        }
    }

    fn profile_operation() -> profile::Operation {
        profile::Operation {
            operation: "ced_decoder_rows",
            dimensions: [0; 4],
            count: 0,
            completed_submits: 1,
            invalid_gpu_timestamps: 0,
            gpu_ms: Some(12.0),
            encode_ms: 5.0,
            commit_wait_ms: 20.0,
            wall_ms: 0.0,
        }
    }

    #[test]
    fn composed_diagnostic_lifecycle_orders_and_cleans_every_boundary() {
        type Trace = Rc<RefCell<Vec<&'static str>>>;
        struct Drain(Trace);
        impl Drop for Drain {
            fn drop(&mut self) {
                self.0.borrow_mut().push("drain");
            }
        }
        struct State {
            trace: Trace,
            live: Rc<Cell<bool>>,
        }
        impl Drop for State {
            fn drop(&mut self) {
                self.trace.borrow_mut().push("state_drop");
                self.live.set(false);
            }
        }
        struct ObservedScope {
            inner: Option<profile::Scope>,
            trace: Trace,
            live: Rc<Cell<bool>>,
        }
        impl Drop for ObservedScope {
            fn drop(&mut self) {
                if let Some(scope) = self.inner.take() {
                    drop(scope);
                    self.trace.borrow_mut().push("scope_drop");
                }
                self.live.set(false);
            }
        }
        struct Probe {
            trace: Trace,
            scope_live: Rc<Cell<bool>>,
            snapshots: u64,
            fault: &'static str,
        }
        impl DiagnosticProbe for Probe {
            type Scope = ObservedScope;
            fn resources(&mut self) -> Result<DiagnosticResources> {
                self.trace.borrow_mut().push(if self.snapshots == 0 {
                    "snapshot"
                } else {
                    "snapshot_after"
                });
                self.snapshots += 1;
                ensure!(
                    !(self.fault == "snapshot_before" && self.snapshots == 1)
                        && !(self.fault == "snapshot_after" && self.snapshots == 2),
                    "snapshot fixture"
                );
                Ok(DiagnosticResources {
                    process_pageins: self.snapshots,
                    process_disk_read_bytes: self.snapshots * 4096,
                    process_resident_size_bytes: 100,
                    process_phys_footprint_bytes: 100,
                    device_current_allocated_bytes: 100,
                })
            }
            fn pipelines(&self) -> MetalPipelineCacheMetrics {
                MetalPipelineCacheMetrics {
                    misses: self.snapshots,
                    ..Default::default()
                }
            }
            fn begin(&mut self) -> Result<Self::Scope> {
                self.trace.borrow_mut().push("scope");
                ensure!(self.fault != "scope_begin", "scope setup fixture");
                let inner = profile::Scope::begin().map_err(anyhow::Error::msg)?;
                self.scope_live.set(true);
                Ok(ObservedScope {
                    inner: Some(inner),
                    trace: self.trace.clone(),
                    live: self.scope_live.clone(),
                })
            }
            fn finish(&mut self, mut scope: Self::Scope) -> profile::Report {
                self.trace.borrow_mut().push("finish");
                let mut report = scope.inner.take().unwrap().finish();
                report.process_pageins = Some(0);
                report.process_disk_read_bytes = if self.fault == "missing_counters" {
                    None
                } else {
                    Some(0)
                };
                report
            }
        }
        struct BrokenWriter;
        impl Write for BrokenWriter {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "lifecycle fixture",
                ))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        for fault in [
            "ok",
            "allocation",
            "prefill",
            "cancel",
            "unwind",
            "fingerprint_unwind",
            "write_start",
            "write_finish",
            "snapshot_before",
            "scope_begin",
            "snapshot_after",
            "missing_counters",
        ] {
            let trace = Trace::default();
            let live = Rc::new(Cell::new(false));
            let scope_live = Rc::new(Cell::new(false));
            let mut probe = Probe {
                trace: trace.clone(),
                scope_live: scope_live.clone(),
                snapshots: 0,
                fault,
            };
            let mut bench = v41_pair_test_bench(
                || {
                    assert!(scope_live.get());
                    assert!(!live.get());
                    trace.borrow_mut().push("allocate");
                    ensure!(fault != "allocation", "allocation fixture");
                    live.set(true);
                    Ok(State {
                        trace: trace.clone(),
                        live: live.clone(),
                    })
                },
                |_, _, execution| {
                    trace.borrow_mut().push("prefill");
                    if fault == "unwind" {
                        panic!("prefill unwind fixture");
                    }
                    ensure!(fault != "prefill", "prefill fixture");
                    ensure!(fault != "cancel", "cancelled fixture");
                    Ok((vec![0.; 257], report(execution)))
                },
            );
            let case = schedule(2).unwrap()[0];
            let mut bytes = Vec::new();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                diagnostic_case_with(
                    &mut probe,
                    case,
                    || {
                        native_case_with(
                            &mut bench,
                            &row_context(),
                            3,
                            1,
                            case,
                            |body| {
                                let _drain = Drain(trace.clone());
                                body()
                            },
                            |logits| {
                                trace.borrow_mut().push("fingerprint");
                                if fault == "fingerprint_unwind" {
                                    panic!("fingerprint unwind fixture");
                                }
                                fingerprint_logits(logits)
                            },
                        )
                    },
                    |event| {
                        let start = matches!(&event, DiagnosticEvent::Start(_));
                        trace
                            .borrow_mut()
                            .push(if start { "start_write" } else { "finish_write" });
                        if (start && fault == "write_start") || (!start && fault == "write_finish")
                        {
                            stream_diagnostic_event(&mut BrokenWriter, case, event)
                        } else {
                            stream_diagnostic_event(&mut bytes, case, event)
                        }
                    },
                )
            }));
            assert!(!live.get(), "state cleanup: {fault}");
            assert!(!scope_live.get(), "scope cleanup: {fault}");
            drop(profile::Scope::begin().expect("real thread-local scope cleaned"));
            bench
                .set_pair_execution(PrefillExecution::FullForward)
                .unwrap();
            if matches!(fault, "unwind" | "fingerprint_unwind") {
                assert!(result.is_err(), "{fault}");
            } else {
                let result = result.expect("only requested faults unwind");
                assert_eq!(result.is_ok(), fault == "ok", "{fault}");
                if fault == "cancel" {
                    assert!(format!("{:#}", result.unwrap_err()).contains("cancelled fixture"));
                }
            }
            let mut expected = vec!["snapshot"];
            if fault != "snapshot_before" {
                expected.push("start_write");
                if fault != "write_start" {
                    expected.push("scope");
                    if fault != "scope_begin" {
                        expected.push("allocate");
                        if fault != "allocation" {
                            expected.push("prefill");
                            if !matches!(fault, "prefill" | "cancel" | "unwind") {
                                expected.push("fingerprint");
                            }
                            expected.push("state_drop");
                        }
                        expected.push("drain");
                        if matches!(fault, "unwind" | "fingerprint_unwind") {
                            expected.push("scope_drop");
                        } else {
                            expected.extend(["finish", "snapshot_after", "finish_write"]);
                        }
                    }
                }
            }
            assert_eq!(*trace.borrow(), expected, "{fault}");
            if !matches!(
                fault,
                "snapshot_before"
                    | "scope_begin"
                    | "write_start"
                    | "write_finish"
                    | "unwind"
                    | "fingerprint_unwind"
            ) {
                let event: serde_json::Value = serde_json::from_str(
                    std::str::from_utf8(&bytes).unwrap().lines().last().unwrap(),
                )
                .unwrap();
                assert_eq!(
                    event["observations"]["native_case_succeeded"],
                    !matches!(fault, "allocation" | "prefill" | "cancel")
                );
                assert_eq!(
                    event["observations"]["collection_succeeded"],
                    !matches!(fault, "snapshot_after" | "missing_counters")
                );
                assert_eq!(event["observations"]["pipeline_cache_delta"][0], 1);
            }
        }
    }

    #[test]
    fn diagnostic_cli_requires_pair_and_is_not_an_ordinary_default() {
        use clap::Parser;
        assert!(
            Args::try_parse_from([
                "qwen-bench",
                "suite",
                "-m",
                "fixture.gguf",
                "--pp",
                "2048",
                "--v41-ced-pair-diagnostic",
            ])
            .is_err()
        );
        let args = Args::try_parse_from([
            "qwen-bench",
            "suite",
            "-m",
            "fixture.gguf",
            "--pp",
            "2048",
            "--runs",
            "2",
            "--v41-ced-pair",
            "--v41-ced-pair-diagnostic",
        ])
        .unwrap();
        let Cmd::Suite(args) = args.cmd else {
            panic!("suite")
        };
        assert!(args.v41_ced_pair && args.v41_ced_pair_diagnostic);
        validate_args(
            ModelFamily::DeepSeekV41,
            &args.pp,
            &args.tg,
            &args.depth,
            args.runs,
            args.no_warmup,
            args.prefill_chunk,
        )
        .unwrap();
        assert_eq!(caller_cpu_bytes(false), CALLER_CPU_BYTES);
        assert_eq!(
            caller_cpu_bytes(true),
            CALLER_CPU_BYTES + DIAGNOSTIC_CPU_BYTES
        );
        assert!(DIAGNOSTIC_CPU_BYTES >= 16 << 20);
    }

    #[test]
    fn recorded_2k_instability_streams_case_8_then_never_schedules_case_9() {
        let seconds = [
            26.2563, 19.2572, 17.0546, 11.3095, 16.9522, 16.9579, 13.0913, 14.9084, 2.3451,
        ];
        let mut bench = v41_pair_test_bench(
            || Ok(()),
            |_, _, execution| Ok((vec![0.; 257], report(execution))),
        );
        let mut streamed = Vec::new();
        let mut calls = Vec::new();
        let error = collect(
            2,
            || Ok(()),
            |case| {
                calls.push(case.case_index);
                let seconds = seconds
                    .get(case.case_index)
                    .expect("must not schedule case 9");
                let mut sample = native_case(&mut bench, &row_context(), 3, 1, case)?;
                sample.sample_ns = (seconds * 1e9) as u64;
                sample.instrumented = true;
                stream_diagnostic(&mut streamed, case, "fixture_finish", &sample.sample_ns)?;
                Ok(sample)
            },
        )
        .unwrap_err();
        assert_eq!(calls, (0..9).collect::<Vec<_>>());
        assert!(format!("{error:#}").contains("stability gate after case 8"));
        let events: Vec<serde_json::Value> = std::str::from_utf8(&streamed)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(events.len(), 9);
        assert_eq!(events[8]["case"]["case_index"], 8);
        assert!(
            events
                .iter()
                .all(|e| e["instrumented"] == true && e["performance_evidence"] == false)
        );
    }

    #[test]
    fn diagnostic_rows_and_every_case_are_marked_without_retaining_profiles() {
        for instrumented in [false, true] {
            let mut bench = v41_pair_test_bench(
                || Ok(()),
                |_, _, execution| Ok((vec![0.; 257], report(execution))),
            );
            let evidence = collect(
                2,
                || Ok(()),
                |case| {
                    let mut sample = native_case(&mut bench, &row_context(), 3, 1, case)?;
                    sample.sample_ns = 100;
                    sample.instrumented = instrumented;
                    Ok(sample)
                },
            )
            .unwrap();
            let rows = aggregate(
                &row_context(),
                3,
                2,
                1,
                257,
                &serde_json::json!({"caller_cpu_bytes": caller_cpu_bytes(instrumented)}),
                &evidence,
            )
            .unwrap();
            for row in rows {
                let diagnostics = row.diagnostics.unwrap();
                let paired = &diagnostics["deepseek_v41"]["paired_benchmark"];
                if instrumented {
                    assert_eq!(diagnostics["instrumented"], true);
                    assert_eq!(diagnostics["performance_evidence"], false);
                    assert_eq!(paired["instrumented"], true);
                    assert_eq!(paired["performance_evidence"], false);
                    assert_eq!(
                        paired["protocol"],
                        "instrumented_full_ced_abba_baab_diagnostic_v1"
                    );
                } else {
                    assert!(diagnostics.get("instrumented").is_none());
                    assert!(paired.get("instrumented").is_none());
                    assert_eq!(paired["protocol"], "native_full_ced_abba_baab_v1");
                }
                for case in paired["native_chronological_cases"].as_array().unwrap() {
                    if instrumented {
                        assert_eq!(case["instrumented"], true);
                        assert_eq!(case["performance_evidence"], false);
                    } else {
                        assert!(case.get("instrumented").is_none());
                    }
                    assert!(case.get("profile").is_none());
                    assert!(case.get("command_totals").is_none());
                }
            }
        }
    }

    #[test]
    fn diagnostic_gap_does_not_subtract_gpu_twice_or_sum_nested_host_spans() {
        let mut host = profile_operation();
        host.completed_submits = 0;
        host.gpu_ms = None;
        host.encode_ms = 0.0;
        host.commit_wait_ms = 0.0;
        host.wall_ms = 40.0;
        let report = profile_report(vec![profile_operation(), host]);
        let totals = command_totals(&report);
        assert_eq!(totals.gpu_ms_valid_completed_only, Some(12.0));
        assert_eq!(totals.outer_wall_minus_command_encode_and_wait_ms, 75.0);
        assert_eq!(totals.completed_submits, 1);
        validate_diagnostic_report(&report).unwrap();
    }

    #[test]
    fn diagnostic_collection_and_stream_errors_stop_the_driver() {
        let mut report = profile_report(vec![]);
        report.process_pageins = None;
        let calls = Cell::new(0);
        assert!(
            drive(
                2,
                || Ok(()),
                |case| {
                    calls.set(calls.get() + 1);
                    if case.case_index == 3 {
                        validate_diagnostic_report(&report)?;
                    }
                    Ok((100.0, ()))
                }
            )
            .is_err()
        );
        assert_eq!(calls.get(), 4);

        struct BrokenWriter;
        impl Write for BrokenWriter {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "fixture",
                ))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        calls.set(0);
        assert!(
            drive(
                2,
                || Ok(()),
                |case| {
                    calls.set(calls.get() + 1);
                    stream_diagnostic(&mut BrokenWriter, case, "fixture", &())?;
                    Ok((100.0, ()))
                }
            )
            .is_err()
        );
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn diagnostic_maximum_profile_is_streamed_one_at_a_time_within_reserve() {
        let max_commands_and_host_spans = diagnostic_operation_bound(MAX_PP, MAX_LAYERS);
        assert!(max_commands_and_host_spans <= MAX_DIAGNOSTIC_OPERATIONS);
        struct CountBytes(usize);
        impl Write for CountBytes {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 += bytes.len();
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let report = profile_report(
            (0..MAX_DIAGNOSTIC_OPERATIONS)
                .map(|_| {
                    let mut op = profile_operation();
                    op.dimensions = [usize::MAX; 4];
                    op.count = usize::MAX;
                    op.completed_submits = usize::MAX;
                    op.invalid_gpu_timestamps = usize::MAX;
                    op.gpu_ms = Some(f64::MAX);
                    op.encode_ms = f64::MAX;
                    op.commit_wait_ms = f64::MAX;
                    op.wall_ms = f64::MAX;
                    op
                })
                .collect(),
        );
        validate_diagnostic_report(&report).unwrap();
        let live_operations =
            report.operations.capacity() * std::mem::size_of::<profile::Operation>();
        for case in schedule(16).unwrap() {
            let mut writer = CountBytes(0);
            stream_diagnostic(&mut writer, case, "fixture_maximum", &report).unwrap();
            // Also charge serialized bytes conservatively although the real
            // writer streams without keeping a JSON string or parsed DOM.
            assert!((live_operations + writer.0 + (1 << 20)) as u64 <= DIAGNOSTIC_CPU_BYTES);
        }
    }

    #[test]
    fn diagnostic_resource_deltas_keep_released_memory_negative() {
        let before = DiagnosticResources {
            process_pageins: 5,
            process_disk_read_bytes: 1024,
            process_resident_size_bytes: 400,
            process_phys_footprint_bytes: 300,
            device_current_allocated_bytes: 200,
        };
        let after = DiagnosticResources {
            process_pageins: 7,
            process_disk_read_bytes: 1536,
            process_resident_size_bytes: 100,
            process_phys_footprint_bytes: 150,
            device_current_allocated_bytes: 50,
        };
        let delta = after.delta_since(&before);
        assert_eq!(delta.process_pageins, 2);
        assert_eq!(delta.process_disk_read_bytes, 512);
        assert_eq!(delta.process_resident_size_bytes, -300);
        assert_eq!(delta.process_phys_footprint_bytes, -150);
        assert_eq!(delta.device_current_allocated_bytes, -150);
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
                    instrumented: false,
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
            "binding_metal_bytes": u64::MAX,
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
    fn native_case_unwind_releases_state_before_returning_to_caller() {
        let live = Rc::new(Cell::new(0));
        let mut bench = v41_pair_test_bench(
            || {
                assert_eq!(live.get(), 0);
                live.set(1);
                Ok(ColdState {
                    live: live.clone(),
                    rows: 0,
                })
            },
            |_, _, _| panic!("native case unwind fixture"),
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            native_case(&mut bench, &row_context(), 3, 1, schedule(2).unwrap()[0])
        }));
        assert!(result.is_err());
        assert_eq!(live.get(), 0);
        bench
            .set_pair_execution(PrefillExecution::FullForward)
            .unwrap();
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
        assert!(!args.v41_ced_pair_diagnostic);
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
