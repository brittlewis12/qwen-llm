//! Bounded resident Full/CED timing protocol. This is not quality qualification.

use super::*;
use crate::family_bench::{
    FamilyBench, FamilyBenchExtent, FamilySemantics, V41BenchPrefillRep, attach_v41_diagnostics,
    family_row, run_family_pp_row, shape_label,
};
use anyhow::ensure;
use qwen_llm::deepseek_v41::forward::profile;
use qwen_llm::deepseek_v41::forward::{
    ColdPrefillPlan, PrefillExecution, plan_cold_prefill_with_target,
};
use qwen_llm::metal::{MetalMemoryAdmission, MetalMemorySignals, MetalPipelineCacheMetrics};
use qwen_llm::model_family::ModelFamily;
use qwen_llm::pid_metrics::PidSnapshot;
use std::collections::BTreeMap;
use std::io::Write;

// At most 38 cases, 256 layer counters per report, two JSON rows and a 1M-logit head.
const CALLER_CPU_BYTES: u64 = 64 << 20;
const DIAGNOSTIC_CPU_BYTES: u64 = 16 << 20;
// Four encoder chunks plus at most four seed/four output chunks and a scalar
// tail per layer, with room for fixed aggregate host spans. No sampled stages.
const MAX_DIAGNOSTIC_OPERATIONS: usize = profile::PAIRED_OPERATION_CAPACITY;
const MAX_PP: usize = 8192;
const MAX_LAYERS: usize = 256;
const P4_SOFT_TARGET: usize = 4_294_967_296usize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum PairSchedule {
    #[default]
    Interleaved,
    FullBlocksFirst,
    CedBlocksFirst,
}

impl PairSchedule {
    pub(crate) fn validate(self, runs: usize, diagnostic: bool) -> Result<()> {
        ensure!(
            (2..=16).contains(&runs) && runs % 2 == 0,
            "paired runs must be even, 2..=16 per policy"
        );
        if self != Self::Interleaved {
            ensure!(
                runs >= 4,
                "blocked pair schedules require --runs >= 4 per policy"
            );
            ensure!(
                !diagnostic,
                "blocked pair schedules are native-only; remove --v41-ced-pair-diagnostic"
            );
        }
        Ok(())
    }
}

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

fn runtime_override(name: &str) -> bool {
    name.starts_with("QWEN_V41_")
        || name.starts_with("QWEN_MATMAT_")
        || name.starts_with("QWEN_MATVEC_")
        || name.starts_with("QWEN_PREFILL_")
        || name.starts_with("QWEN_DECODE_")
        || name.starts_with("QWEN_DSV4_")
        || name.starts_with("QWEN_PROMPT_MATMAT_")
        || name == "QWEN_EXPERT_TENSOR_TILE"
}

const VALIDATION_ENV_KEYS: &[&str] = &[
    "MTL_DEBUG_LAYER",
    "MTL_SHADER_VALIDATION",
    "METAL_DEVICE_WRAPPER_TYPE",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_IMAGE_SUFFIX",
    "MTL_CAPTURE_ENABLED",
];

fn validation_or_injection_setting(name: &str) -> bool {
    name.starts_with("MTL_DEBUG_LAYER")
        || name.starts_with("MTL_SHADER_VALIDATION")
        || VALIDATION_ENV_KEYS.contains(&name)
}

pub(crate) struct Measurement {
    production_timing: bool,
    diagnostic: bool,
    schedule: PairSchedule,
    soft_coalescing_target: Option<usize>,
    environment: BTreeMap<String, Option<String>>,
}

impl Measurement {
    pub(crate) fn capture(production_timing: bool, diagnostic: bool) -> Self {
        Self::from_environment(production_timing, diagnostic, std::env::vars_os())
    }

    pub(crate) fn from_environment(
        production_timing: bool,
        diagnostic: bool,
        vars: impl IntoIterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
    ) -> Self {
        let mut environment: BTreeMap<_, _> = VALIDATION_ENV_KEYS
            .iter()
            .map(|name| ((*name).to_owned(), None))
            .collect();
        for (name, value) in vars {
            let name = name.to_string_lossy();
            if validation_or_injection_setting(&name)
                || runtime_override(&name)
                || name == "QWEN_METAL_LEASE_SKIP_WIRED_GATE"
            {
                let captured = if runtime_override(&name) {
                    "<set; disallowed runtime value not recorded>".to_owned()
                } else {
                    value.to_string_lossy().into_owned()
                };
                environment.insert(name.into_owned(), Some(captured));
            }
        }
        Self {
            production_timing,
            diagnostic,
            schedule: PairSchedule::Interleaved,
            soft_coalescing_target: None,
            environment,
        }
    }

    pub(crate) fn protocol(&self) -> &'static str {
        match (self.schedule, self.production_timing) {
            (PairSchedule::FullBlocksFirst, false) => return "native_full_ced_full_blocks_first_v1",
            (PairSchedule::FullBlocksFirst, true) => {
                return "native_full_ced_full_blocks_first_production_timing_v1";
            }
            (PairSchedule::CedBlocksFirst, false) => return "native_full_ced_ced_blocks_first_v1",
            (PairSchedule::CedBlocksFirst, true) => {
                return "native_full_ced_ced_blocks_first_production_timing_v1";
            }
            _ => (),
        }
        match (self.production_timing, self.diagnostic) {
            (false, false) => "native_full_ced_abba_baab_v2",
            (false, true) => "instrumented_full_ced_abba_baab_diagnostic_v2",
            (true, false) => "native_full_ced_abba_baab_production_timing_v2",
            (true, true) => "instrumented_full_ced_abba_baab_production_timing_diagnostic_v2",
        }
    }

    pub(crate) fn with_schedule(mut self, schedule: PairSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    pub(crate) fn with_p4(mut self, p4: bool) -> Self {
        self.soft_coalescing_target = p4.then_some(P4_SOFT_TARGET);
        self
    }

    pub(crate) fn validate_p4(&self, p4: bool) -> Result<()> {
        ensure!(
            self.soft_coalescing_target == p4.then_some(P4_SOFT_TARGET),
            "paired launch mapping target differs from request"
        );
        Ok(())
    }

    pub(crate) fn validate_schedule(&self, schedule: PairSchedule, runs: usize) -> Result<()> {
        ensure!(
            self.schedule == schedule,
            "paired launch schedule differs from request"
        );
        self.schedule.validate(runs, self.diagnostic)
    }

    fn regime(&self) -> &'static str {
        if self.production_timing {
            "production_timing_requested"
        } else {
            "validation_on_requested"
        }
    }

    fn description(&self) -> serde_json::Value {
        let mut description = serde_json::json!({
            "protocol": self.protocol(),
            "regime": self.regime(),
            "production_timing_requested": self.production_timing,
            "canonical_production_timing_requested": self.production_timing && !self.diagnostic,
            "instrumented": self.diagnostic,
            "resident_soft_coalescing_target_bytes": self.soft_coalescing_target,
            "effective_validation": "not_verified_external_injection_possible",
            "environment_capture": "before_metal_context_creation_no_in_process_mutation",
            "environment_value_encoding": "os_values_lossy_utf8",
            "disallowed_runtime_values": "presence_only_not_recorded",
            "environment": self.environment,
        });
        if self.schedule != PairSchedule::Interleaved {
            description["schema"] = "v41-ced-pair-blocked-v1".into();
            description["schedule"] = serde_json::json!(self.schedule);
            description["schedule_contract"] = "one policy warmup + two excluded pilots + all retained main calls, then the other policy; fresh cold state per call".into();
        }
        description
    }

    pub(crate) fn validate_environment(&self) -> Result<()> {
        if !self.production_timing {
            ensure!(
                self.environment
                    .get("MTL_DEBUG_LAYER")
                    .and_then(|v| v.as_deref())
                    == Some("1"),
                "--v41-ced-pair requires MTL_DEBUG_LAYER=1 unless --v41-ced-pair-production-timing is explicit"
            );
        }
        for (name, value) in &self.environment {
            if value.is_none() {
                continue;
            }
            ensure!(
                !self.production_timing || !validation_or_injection_setting(name),
                "production timing requires {name} unset at process launch (even 0/empty is refused)"
            );
            let error_override = validation_or_injection_setting(name)
                && (name.ends_with("_ERROR_MODE")
                    || name.ends_with("_WARNING_MODE")
                    || name.ends_with("_ENABLE_ERROR_REPORTING")
                    || name.ends_with("_REPORT_TO_STDERR"));
            ensure!(
                !error_override,
                "remove validation error/warning override {name} for --v41-ced-pair"
            );
            ensure!(
                !runtime_override(name),
                "remove runtime/arithmetic/profiling override {name} for --v41-ced-pair"
            );
        }
        if let Some(Some(value)) = self.environment.get("QWEN_METAL_LEASE_SKIP_WIRED_GATE") {
            ensure!(
                qwen_llm::env_flag::env_value_falsy(value),
                "--v41-ced-pair refuses the wired admission bypass"
            );
        }
        ensure!(
            !qwen_llm::metal::dispatch_census_is_active(),
            "--v41-ced-pair refuses an active dispatch observer"
        );
        Ok(())
    }

    pub(crate) fn emit_requested(&self, writer: &mut impl Write) -> Result<()> {
        serde_json::to_writer(
            &mut *writer,
            &serde_json::json!({
                "event": "v41_pair_launch_requested",
                "measurement": self.description(),
                "instrumented": self.diagnostic,
                "performance_evidence": false,
                "status": "environment_check_pending_no_device_created",
            }),
        )?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(())
    }

    #[cfg(test)]
    fn start(self, writer: &mut impl Write) -> Result<Self> {
        self.emit_requested(writer)?;
        self.validate_environment().with_context(|| {
            format!(
                "paired launch protocol={} regime={}",
                self.protocol(),
                self.regime()
            )
        })?;
        Ok(self)
    }
}

pub(crate) struct PairedPreweight {
    pub(crate) plan: ColdPrefillPlan,
    pub(crate) measurement: Measurement,
    report: serde_json::Value,
}

/// Called before with_family_bench can import any resident weights.
pub(crate) fn preweight_plan(
    ctx: &MetalContext,
    gguf: &GgufFile,
    extent: FamilyBenchExtent,
    measurement: Measurement,
) -> Result<PairedPreweight> {
    let diagnostic = measurement.diagnostic;
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
    let plan = plan_cold_prefill_with_target(
        ctx,
        gguf,
        extent.prefill_endpoint,
        extent.forwards,
        &[
            PrefillExecution::FullForward,
            PrefillExecution::ShrinkingCed,
        ],
        caller_cpu_bytes(diagnostic),
        measurement.soft_coalescing_target,
    )
    .context("paired preweight Full/CED planning")?;
    let report = preweight_report(&plan, extent)?;
    Ok(PairedPreweight {
        plan,
        measurement,
        report,
    })
}

fn preweight_report(
    plan: &ColdPrefillPlan,
    extent: FamilyBenchExtent,
) -> Result<serde_json::Value> {
    let ced = plan
        .ced
        .as_ref()
        .context("paired preweight plan cannot execute CED without fallback")?;
    Ok(serde_json::json!({
        "prompt_rows": extent.prefill_endpoint,
        "capacity": extent.forwards,
        "executions": [PrefillExecution::FullForward, PrefillExecution::ShrinkingCed],
        "resident_soft_coalescing_target_bytes": plan.resident_storage.soft_coalescing_target,
        "resident_metal_view_bytes": plan.resident_storage.metal_view_bytes,
        "resident_private_shadow_bytes": plan.resident_storage.private_shadow_bytes,
        "request_metal_bytes": plan.request.metal_bytes,
        "request_cpu_bytes": plan.request.cpu_bytes,
        "ced_incremental_metal_bytes": ced.incremental_metal_bytes,
        "ced_incremental_cpu_bytes": ced.incremental_cpu_bytes,
        "full_chunk_rows": plan.full_chunk_rows,
        "full_incremental_metal_bytes": plan.full_incremental_metal_bytes,
        "full_incremental_cpu_bytes": plan.full_incremental_cpu_bytes,
        "runtime_metal_bytes": plan.runtime_metal_bytes,
        "runtime_cpu_bytes": plan.runtime_cpu_bytes,
        "binding_metal_bytes": plan.binding_metal_bytes,
        "binding_cpu_bytes": plan.binding_cpu_bytes,
        "loader_control_cpu_bytes": plan.loader_control_cpu_bytes,
        "descriptor_window_cpu_bytes": plan.descriptor_window_cpu_bytes,
        "retained_summary_cpu_bytes": plan.retained_summary_cpu_bytes,
        "retained_summary_accounting": "max(loader_summary, caller_descriptor_window); included_once_in_binding_cpu",
        "loader_controls": {"payload_bytes_read_per_plan": plan.loader_controls.payload_bytes_read,
            "cold_normalization_cpu_bytes": plan.loader_controls.cold_cpu_bytes,
            "retained_cpu_bytes": plan.loader_controls.retained_cpu_bytes,
            "source_bookkeeping_bytes": plan.loader_controls.source_bookkeeping_bytes,
            "scope": "included_in_binding_cpu_not_shared_weight_shadow"},
        "caller_cpu_bytes": plan.caller_cpu_bytes,
        "required_metal_bytes": plan.required_metal_bytes,
        "required_cpu_bytes": plan.required_cpu_bytes,
        "load_metal_bytes": plan.load_metal_bytes,
        "load_cpu_bytes": plan.load_cpu_bytes,
        "runtime_gate_cpu_bytes": plan.runtime_gate_cpu_bytes,
        "required_bytes_scope": "full_cold_peak_capacity_not_fresh_load_availability",
        "scope": "staged_budget_not_reservation_fresh_runtime_assessment_before_every_state_inner_gates_retained",
    }))
}

fn admission_report(admission: &MetalMemoryAdmission) -> serde_json::Value {
    serde_json::json!({
        "admitted": admission.is_admitted(),
        "reason": format!("{:?}", admission.reason),
        "required_bytes": admission.required_bytes,
        "cpu_required_bytes": admission.cpu_required_bytes,
        "working_set_headroom_bytes": admission.working_set_headroom_bytes,
        "host_remaining_scope": "additional_cpu_headroom_not_shared_metal_capacity",
    })
}

/// The only resident factory invocation must remain inside this gate.
#[cfg(test)]
pub(crate) fn with_load<R>(
    preweight: Option<&PairedPreweight>,
    signals: impl FnOnce() -> MetalMemorySignals,
    factory: impl FnOnce(Option<&serde_json::Value>) -> Result<R>,
) -> Result<R> {
    with_load_plan(preweight, signals, |report, _| factory(report))
}

pub(crate) fn with_load_plan<R>(
    preweight: Option<&PairedPreweight>,
    signals: impl FnOnce() -> MetalMemorySignals,
    factory: impl FnOnce(Option<&serde_json::Value>, Option<&ColdPrefillPlan>) -> Result<R>,
) -> Result<R> {
    let mut selected = None;
    let report = if let Some(preweight) = preweight {
        let (plan, assessment) = preweight
            .plan
            .select_load(signals())
            .context("paired full cold peak capacity and fresh load admission")?;
        ensure!(
            plan.resident_storage.soft_coalescing_target
                == preweight.measurement.soft_coalescing_target,
            "paired cold mapping target differs from launch"
        );
        ensure!(
            assessment.load_admission.is_admitted(),
            "paired load denied"
        );
        let mut report = preweight.report.clone();
        if plan.capacity != 0 {
            let current = preweight_report(
                &plan,
                FamilyBenchExtent {
                    forwards: plan.capacity,
                    prefill_endpoint: plan.geometry.tokens,
                },
            )?;
            report
                .as_object_mut()
                .context("paired report object")?
                .extend(current.as_object().unwrap().clone());
        }
        report["peak_capacity_passed"] = true.into();
        report["peak_capacity"] = format!("{:?}", assessment.peak_capacity).into();
        report["load_admission"] = admission_report(&assessment.load_admission);
        report["whole_peak_now"] = admission_report(&assessment.whole_peak_now);
        report["whole_peak_now_scope"] = "diagnostic_only_not_runtime_admission".into();
        selected = Some(plan);
        Some(report)
    } else {
        None
    };
    factory(report.as_ref(), selected.as_ref())
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Warmup,
    Pilot,
    Main,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
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

fn schedule(mode: PairSchedule, runs: usize) -> Result<Vec<Case>> {
    mode.validate(runs, false)?;
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
    if mode != PairSchedule::Interleaved {
        let order = if mode == PairSchedule::FullBlocksFirst {
            [A, B]
        } else {
            [B, A]
        };
        for (index, policy) in order.into_iter().enumerate() {
            let label = if policy == A { "A" } else { "B" };
            block(Phase::Warmup, index, label, &[policy]);
            block(
                Phase::Pilot,
                index,
                if policy == A { "AA" } else { "BB" },
                &[policy; 2],
            );
            let main_order = if policy == A {
                "AAAAAAAAAAAAAAAA"
            } else {
                "BBBBBBBBBBBBBBBB"
            };
            block(Phase::Main, index, &main_order[..runs], &vec![policy; runs]);
        }
        return Ok(cases);
    }
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

#[derive(Debug, serde::Serialize)]
struct NativeDispersion {
    min_ns: f64,
    max_ns: f64,
    max_over_min: f64,
}

fn dispersion(samples: &[f64]) -> Result<NativeDispersion> {
    ensure!(
        samples.len() >= 2,
        "dispersion requires at least two native samples per policy"
    );
    ensure!(
        samples.iter().all(|n| n.is_finite() && *n > 0.0),
        "native samples must be finite and positive: {samples:?}"
    );
    let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max = samples.iter().copied().fold(0.0, f64::max);
    Ok(NativeDispersion {
        min_ns: min,
        max_ns: max,
        max_over_min: max / min,
    })
}

fn drive<T>(
    mode: PairSchedule,
    runs: usize,
    mut checkpoint: impl FnMut() -> Result<()>,
    mut execute: impl FnMut(Case) -> Result<(f64, T)>,
) -> Result<Vec<(Case, T)>> {
    let cases = schedule(mode, runs)?;
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
        }
        evidence.push((case, payload));
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
    driver_scheduling_ms_valid_completed_only: Option<f64>,
    invalid_driver_scheduling_timestamps: usize,
    encode_ms: f64,
    commit_wait_ms_including_gpu: f64,
    outer_wall_minus_command_encode_and_wait_ms: f64,
}

fn command_totals(report: &profile::Report) -> CommandTotals {
    let mut totals = CommandTotals {
        completed_submits: 0,
        invalid_gpu_timestamps: 0,
        gpu_ms_valid_completed_only: None,
        driver_scheduling_ms_valid_completed_only: None,
        invalid_driver_scheduling_timestamps: 0,
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
        if let Some(ms) = op.driver_scheduling_ms {
            *totals
                .driver_scheduling_ms_valid_completed_only
                .get_or_insert(0.0) += ms;
        }
        totals.invalid_driver_scheduling_timestamps += op.invalid_driver_scheduling_timestamps;
        totals.encode_ms += op.encode_ms;
        totals.commit_wait_ms_including_gpu += op.commit_wait_ms;
    }
    // The outer case includes work outside native prefill. Neither Metal GPU
    // envelopes nor CPU-driver scheduling are additive to inclusive wait time.
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
    measurement: &Measurement,
    case: Case,
    event: &'static str,
    observations: &impl serde::Serialize,
) -> Result<()> {
    #[derive(serde::Serialize)]
    struct Event<'a, T> {
        event: &'static str,
        protocol: &'static str,
        measurement_regime: &'static str,
        case: Case,
        instrumented: bool,
        performance_evidence: bool,
        observations: &'a T,
    }
    serde_json::to_writer(
        &mut *writer,
        &Event {
            event,
            protocol: measurement.protocol(),
            measurement_regime: measurement.regime(),
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
    measurement: &Measurement,
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
        |event| stream_diagnostic_event(&mut stderr.lock(), measurement, case, event),
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
    driver_scheduling_interpretation: &'static str,
    resources_before: &'a DiagnosticResources,
    resources_after_pool_drain: Option<&'a DiagnosticResources>,
    resource_delta: Option<DiagnosticResourceDelta>,
    pipeline_cache_delta: [u64; 3],
    pipeline_cache_delta_fields: [&'static str; 3],
    pipeline_cache_coverage: &'static str,
    command_totals: CommandTotals,
    profile: &'a profile::Report,
}

enum DiagnosticEvent<'a> {
    Start(&'a DiagnosticResources),
    Finish(DiagnosticObservation<'a>),
}

fn stream_diagnostic_event(
    writer: &mut impl Write,
    measurement: &Measurement,
    case: Case,
    event: DiagnosticEvent<'_>,
) -> Result<()> {
    match event {
        DiagnosticEvent::Start(before) => stream_diagnostic(
            writer,
            measurement,
            case,
            "v41_pair_diagnostic_start",
            before,
        ),
        DiagnosticEvent::Finish(observation) => stream_diagnostic(
            writer,
            measurement,
            case,
            "v41_pair_diagnostic_finish",
            &observation,
        ),
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
        cross_case_gate_status: "fingerprint_and_sample_validation_pending",
        native_prefill_wall_ns: result.as_ref().ok().map(|sample| sample.sample_ns),
        native_timing_scope: "prefill_plus_endpoint_copy_state_allocation_excluded_instrumented",
        profile_and_pipeline_scope: "outer_native_case_including_state_allocation_validation_fingerprint_reporting_teardown_and_pool_drain",
        resource_scope: "before_case_to_after_state_drop_and_case_pool_drain_current_values_not_peaks",
        gap_interpretation: "outer_wall_minus_encode_and_wait_is_unattributed_wall_not_cpu_time_gpu_is_already_inside_wait",
        driver_scheduling_interpretation: "metal_kernel_end_minus_start_cpu_driver_scheduling_not_shader_active_no_instant_alignment_not_additive_to_gpu_or_wait_null_without_valid_completed_pairs_invalid_count_completed_only",
        resources_before: &before,
        resources_after_pool_drain: after.as_ref().ok(),
        resource_delta: after.as_ref().ok().map(|a| a.delta_since(&before)),
        pipeline_cache_delta: [
            pipeline.misses,
            pipeline.miss_wall_ns,
            pipeline.compiler_wall_ns,
        ],
        pipeline_cache_delta_fields: ["misses", "miss_wall_ns", "compiler_wall_ns"],
        pipeline_cache_coverage: "ordinary_product_research_and_available_tensor_cold_lookups_including_failures_compiler_nested_in_miss_wall_absent_tensor_tier_not_a_lookup",
        command_totals: command_totals(&report),
        profile: &report,
    }))?;
    let mut sample = result?;
    after?;
    validation?;
    sample.instrumented = true;
    Ok(sample)
}

/// Suite owns the plan and a fresh context through import, binding and all cases.
pub(crate) fn run(
    bench: &mut dyn FamilyBench,
    row_ctx: &SuiteRowContext,
    n: usize,
    runs: usize,
    seed: u64,
    plan: &ColdPrefillPlan,
    preweight: &serde_json::Value,
    ctx: &MetalContext,
    measurement: &Measurement,
) -> Result<Vec<BenchRow>> {
    let diagnostic = measurement.diagnostic;
    measurement.schedule.validate(runs, diagnostic)?;
    if diagnostic {
        ensure!(
            plan.caller_cpu_bytes == caller_cpu_bytes(true),
            "paired diagnostic requires its additional preweight CPU reserve"
        );
    }
    let evidence = collect_assessed(
        measurement.schedule,
        runs,
        shutdown::checkpoint,
        || {
            let admission = plan
                .assess_runtime(ctx.memory_signals())
                .context("paired fresh postbinding runtime admission")?;
            ensure!(admission.is_admitted(), "paired runtime denied");
            Ok(())
        },
        |case| {
            if diagnostic {
                diagnostic_case(ctx, measurement, bench, row_ctx, n, seed, case)
            } else {
                native_case(bench, row_ctx, n, seed, case)
            }
        },
    )?;
    aggregate(
        row_ctx,
        n,
        runs,
        seed,
        bench.vocab_size(),
        preweight,
        measurement,
        &evidence,
    )
}

fn collect_assessed(
    mode: PairSchedule,
    runs: usize,
    checkpoint: impl FnMut() -> Result<()>,
    mut assess_runtime: impl FnMut() -> Result<()>,
    mut execute: impl FnMut(Case) -> Result<NativeCase>,
) -> Result<Vec<(Case, NativeCase)>> {
    collect(mode, runs, checkpoint, |case| {
        // The prior native case has released state/logits/commands and its pool.
        // Do not put this in checkpoint: drive also checkpoints after all cases.
        assess_runtime()?;
        execute(case)
    })
}

fn collect(
    mode: PairSchedule,
    runs: usize,
    checkpoint: impl FnMut() -> Result<()>,
    mut execute: impl FnMut(Case) -> Result<NativeCase>,
) -> Result<Vec<(Case, NativeCase)>> {
    let mut first_warmup = None;
    drive(mode, runs, checkpoint, |case| {
        let sample = execute(case)?;
        if case.case_index == 0 {
            first_warmup = Some(sample.logits_sha256_f32le.clone());
        }
        ensure!(
            first_warmup.as_ref() == Some(&sample.logits_sha256_f32le),
            "paired case {} endpoint head differs bitwise from first warmup (all must match warm Full); not quality qualification",
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
    measurement: &Measurement,
    evidence: &[(Case, NativeCase)],
) -> Result<Vec<BenchRow>> {
    let expected = schedule(measurement.schedule, runs)?;
    ensure!(
        evidence.len() == expected.len()
            && evidence
                .iter()
                .zip(&expected)
                .all(|((case, _), expected)| case == expected),
        "paired evidence has incomplete or incorrect schedule chronology"
    );
    let instrumented = evidence
        .first()
        .is_some_and(|(_, sample)| sample.instrumented);
    ensure!(
        evidence
            .iter()
            .all(|(_, sample)| sample.instrumented == instrumented),
        "paired evidence mixes diagnostic and uninstrumented cases"
    );
    ensure!(
        measurement.diagnostic == instrumented,
        "paired measurement identity differs from instrumentation"
    );
    let chronology: Vec<_> = evidence
        .iter()
        .enumerate()
        .map(|(index, (case, sample))| {
            let mut entry = serde_json::json!({
                "case": case,
                "requested_execution": case.policy.execution(),
                "sample_ns": sample.sample_ns,
                "pilot_status": match case.phase {
                    Phase::Warmup => "pending",
                    Phase::Pilot if evidence.get(index + 1).is_some_and(|(next, _)| next.phase == Phase::Pilot && next.block == case.block) => "collecting",
                    _ => "completed",
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
        let pilot_plus_main_ns: Vec<_> = evidence
            .iter()
            .filter(|(case, _)| case.phase != Phase::Warmup && case.policy == policy)
            .map(|(_, sample)| sample.sample_ns as f64)
            .collect();
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
        let retained_main_ns: Vec<_> = row.samples_ns.iter().map(|&ns| ns as f64).collect();
        let pilot_plus_main_dispersion = dispersion(&pilot_plus_main_ns)?;
        let retained_main_dispersion = dispersion(&retained_main_ns)?;
        attach_v41_diagnostics(
            &mut row,
            retained
                .iter()
                .map(|(_, sample)| sample.report.clone())
                .collect(),
        );
        let diagnostics = row.diagnostics.get_or_insert_with(|| serde_json::json!({}));
        diagnostics["deepseek_v41"]["paired_benchmark"] = serde_json::json!({
            "protocol": measurement.protocol(),
            "measurement": measurement.description(),
            "policy": policy,
            "requested_execution": policy.execution(),
            "seed": seed,
            "prompt_ids_sha256_i32le": token_ids_sha256_i32le(&ids),
            "capacity": n,
            "retained_runs_per_policy": runs,
            "schedule_status": "completed",
            "pilot_status": "completed",
            "pilot_samples_per_policy": 2,
            "native_dispersion": {
                "pilot_plus_main": pilot_plus_main_dispersion,
                "retained_main": retained_main_dispersion,
            },
            "timing_interpretation": "observational_not_automatic_performance_gain",
            "preweight_plan": preweight,
            "native_chronological_cases": chronology,
            "timing_scope": "native_prefill_host_wall_including_endpoint_logits_copy_state_allocation_excluded",
            "quality_qualification": false,
            "host_validity": "external_continuous_monitor_required",
            "metal_debug_layer": measurement.environment.get("MTL_DEBUG_LAYER").and_then(|v| v.as_deref()),
        });
        if measurement.schedule != PairSchedule::Interleaved {
            let paired = &mut diagnostics["deepseek_v41"]["paired_benchmark"];
            paired["schema"] = "v41-ced-pair-blocked-v1".into();
            paired["schedule"] = serde_json::json!(measurement.schedule);
        }
        if instrumented {
            diagnostics["instrumented"] = true.into();
            diagnostics["performance_evidence"] = false.into();
            let paired = &mut diagnostics["deepseek_v41"]["paired_benchmark"];
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

    fn measurement_vars(production: bool, diagnostic: bool, vars: &[(&str, &str)]) -> Measurement {
        Measurement::from_environment(
            production,
            diagnostic,
            vars.iter()
                .map(|(name, value)| ((*name).into(), (*value).into())),
        )
    }

    fn measurement_fixture(production: bool, diagnostic: bool) -> Measurement {
        measurement_vars(
            production,
            diagnostic,
            if production {
                &[]
            } else {
                &[("MTL_DEBUG_LAYER", "1")]
            },
        )
    }

    const PROVENANCE_ENV_BUDGET_BYTES: usize = 240 << 10;

    fn escaped_many_key_measurement() -> (Measurement, usize, usize) {
        let mut vars: Vec<(std::ffi::OsString, std::ffi::OsString)> =
            vec![("MTL_DEBUG_LAYER".into(), "1".into())];
        let entry_bytes =
            |name: &str, value: &str| name.len() + value.len() + 2 + std::mem::size_of::<usize>();
        let mut raw_bytes = entry_bytes("MTL_DEBUG_LAYER", "1");
        loop {
            let name = format!("MTL_SHADER_VALIDATION_FIXTURE_{:04}", vars.len());
            let value = "\u{0001}".repeat(192);
            let bytes = entry_bytes(&name, &value);
            if raw_bytes + bytes > PROVENANCE_ENV_BUDGET_BYTES {
                break;
            }
            raw_bytes += bytes;
            vars.push((name.into(), value.into()));
        }
        let keys = vars.len();
        (
            Measurement::from_environment(false, false, vars),
            raw_bytes,
            keys,
        )
    }

    #[test]
    fn escaped_many_key_provenance_uses_a_bounded_arg_max_like_fixture() {
        let (measurement, raw_bytes, keys) = escaped_many_key_measurement();
        assert!(raw_bytes <= PROVENANCE_ENV_BUDGET_BYTES);
        assert!(raw_bytes > PROVENANCE_ENV_BUDGET_BYTES - 512);
        assert!(keys >= 1000);
        measurement.validate_environment().unwrap();
        let serialized = serde_json::to_string(&measurement.description()).unwrap();
        assert!(serialized.len() > raw_bytes * 4);
        assert!(serialized.contains("\\u0001"));
        assert!(!serialized.contains('\u{0001}'));
        assert!(
            measurement
                .environment
                .values()
                .flatten()
                .all(|v| !v.contains('\0'))
        );
    }

    #[test]
    fn production_timing_cli_requires_pair_and_remains_explicit() {
        use clap::Parser;
        assert!(
            Args::try_parse_from([
                "qwen-bench",
                "suite",
                "-m",
                "fixture.gguf",
                "--pp",
                "128",
                "--v41-ced-pair-production-timing",
            ])
            .is_err()
        );
        for diagnostic in [false, true] {
            let mut argv = vec![
                "qwen-bench",
                "suite",
                "-m",
                "fixture.gguf",
                "--pp",
                "128",
                "--runs",
                "2",
                "--v41-ced-pair",
                "--v41-ced-pair-production-timing",
            ];
            if diagnostic {
                argv.push("--v41-ced-pair-diagnostic");
            }
            let Cmd::Suite(args) = Args::try_parse_from(argv).unwrap().cmd else {
                panic!("suite")
            };
            assert!(args.v41_ced_pair && args.v41_ced_pair_production_timing);
            assert_eq!(args.v41_ced_pair_diagnostic, diagnostic);
        }
        let Cmd::Suite(args) = Args::try_parse_from([
            "qwen-bench",
            "suite",
            "-m",
            "fixture.gguf",
            "--pp",
            "128",
            "--v41-ced-pair",
        ])
        .unwrap()
        .cmd
        else {
            panic!("suite")
        };
        assert!(!args.v41_ced_pair_production_timing);
        assert!(!args.v41_ced_pair_diagnostic);
    }

    #[test]
    fn production_timing_requires_validation_wrapper_and_injection_settings_strictly_unset() {
        for diagnostic in [false, true] {
            assert!(
                measurement_fixture(true, diagnostic)
                    .validate_environment()
                    .is_ok()
            );
            for name in [
                "MTL_DEBUG_LAYER",
                "MTL_DEBUG_LAYER_ERROR_MODE",
                "MTL_DEBUG_LAYER_WARNING_MODE",
                "MTL_DEBUG_LAYER_VALIDATE_UNRETAINED_RESOURCES",
                "MTL_SHADER_VALIDATION",
                "MTL_SHADER_VALIDATION_DEFAULT_STATE",
                "MTL_SHADER_VALIDATION_ENABLE_ERROR_REPORTING",
                "MTL_SHADER_VALIDATION_REPORT_TO_STDERR",
                "METAL_DEVICE_WRAPPER_TYPE",
                "DYLD_INSERT_LIBRARIES",
                "DYLD_IMAGE_SUFFIX",
                "MTL_CAPTURE_ENABLED",
            ] {
                for value in ["", "0", "1", "false"] {
                    assert!(
                        measurement_vars(true, diagnostic, &[(name, value)])
                            .validate_environment()
                            .is_err(),
                        "{name}={value:?}"
                    );
                }
            }
        }
        for name in [
            "MTL_DEBUG_LAYER_ERROR_MODE",
            "MTL_DEBUG_LAYER_WARNING_MODE",
            "MTL_SHADER_VALIDATION_ENABLE_ERROR_REPORTING",
            "MTL_SHADER_VALIDATION_REPORT_TO_STDERR",
        ] {
            assert!(
                measurement_vars(false, false, &[("MTL_DEBUG_LAYER", "1"), (name, "0")])
                    .validate_environment()
                    .is_err(),
                "{name}"
            );
        }
    }

    #[test]
    fn validation_provenance_preserves_unset_zero_one_and_records_failed_launch_identity() {
        for value in [None, Some(""), Some("0"), Some("1")] {
            let vars: Vec<_> = value
                .map(|value| ("MTL_DEBUG_LAYER", value))
                .into_iter()
                .collect();
            let measurement = measurement_vars(false, false, &vars);
            assert_eq!(
                measurement.description()["environment"]["MTL_DEBUG_LAYER"],
                serde_json::to_value(value).unwrap()
            );
            assert_eq!(
                measurement.validate_environment().is_ok(),
                value == Some("1")
            );
        }
        for shader in ["0", "1"] {
            let measurement = measurement_vars(
                false,
                false,
                &[("MTL_DEBUG_LAYER", "1"), ("MTL_SHADER_VALIDATION", shader)],
            );
            assert!(measurement.validate_environment().is_ok());
            assert_eq!(
                measurement.description()["environment"]["MTL_SHADER_VALIDATION"],
                shader
            );
            assert_eq!(
                measurement.description()["effective_validation"],
                "not_verified_external_injection_possible"
            );
        }
        for production in [false, true] {
            for diagnostic in [false, true] {
                let vars = if production {
                    vec![
                        ("MTL_SHADER_VALIDATION", "0"),
                        ("DYLD_INSERT_LIBRARIES", "fixture-injection"),
                    ]
                } else {
                    vec![]
                };
                let measurement = measurement_vars(production, diagnostic, &vars);
                let expected = measurement.description();
                let protocol = measurement.protocol();
                let mut bytes = Vec::new();
                let error = measurement
                    .start(&mut bytes)
                    .err()
                    .expect("invalid launch refused");
                assert!(format!("{error:#}").contains(protocol));
                let launch: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(launch["event"], "v41_pair_launch_requested");
                assert_eq!(launch["measurement"], expected);
                assert_eq!(
                    launch["status"],
                    "environment_check_pending_no_device_created"
                );
                assert_eq!(launch["performance_evidence"], false);
            }
        }
    }

    #[test]
    fn rejected_runtime_override_values_are_not_logged() {
        let secret = "fixture-value-that-must-not-be-logged";
        let measurement = measurement_vars(
            false,
            false,
            &[
                ("MTL_DEBUG_LAYER", "1"),
                ("QWEN_V41_UNKNOWN_OVERRIDE", secret),
            ],
        );
        let mut bytes = Vec::new();
        assert!(measurement.start(&mut bytes).is_err());
        let launch = String::from_utf8(bytes).unwrap();
        assert!(launch.contains("QWEN_V41_UNKNOWN_OVERRIDE"));
        assert!(!launch.contains(secret));
    }

    #[test]
    fn both_native_regimes_leave_profile_scope_inert() {
        for production in [false, true] {
            let measurement = measurement_fixture(production, false);
            measurement.validate_environment().unwrap();
            let mut bench = v41_pair_test_bench(
                || {
                    drop(profile::Scope::begin().expect("no scope around native state creation"));
                    Ok(())
                },
                |_, _, execution| {
                    drop(profile::Scope::begin().expect("no scope around native prefill"));
                    Ok((vec![0.; 257], report(execution)))
                },
            );
            let sample = native_case(
                &mut bench,
                &row_context(),
                3,
                1,
                schedule(PairSchedule::Interleaved, 2).unwrap()[0],
            )
            .unwrap();
            assert!(!sample.instrumented);
            drop(profile::Scope::begin().expect("native case leaves no profiling scope"));
        }
    }

    #[test]
    fn diagnostic_events_have_distinct_regime_identity_without_repeating_environment() {
        for production in [false, true] {
            let measurement = measurement_fixture(production, true);
            let mut bytes = Vec::new();
            stream_diagnostic(
                &mut bytes,
                &measurement,
                schedule(PairSchedule::Interleaved, 2).unwrap()[0],
                "fixture",
                &(),
            )
            .unwrap();
            let event: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(event["protocol"], measurement.protocol());
            assert_eq!(event["measurement_regime"], measurement.regime());
            assert_eq!(event["instrumented"], true);
            assert_eq!(event["performance_evidence"], false);
            assert!(event.get("environment").is_none());
            assert!(event.get("measurement").is_none());
        }
    }

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
            geometry: qwen_llm::deepseek_v41::forward::PrefillGeometry {
                start: 0,
                tokens: 3,
                selected_rows: 4096,
            },
            admission: None,
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
            dispatches: Vec::new(),
            dispatch_overflow: 0,
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
            driver_scheduling_ms: Some(3.0),
            invalid_driver_scheduling_timestamps: 0,
            encode_ms: 5.0,
            commit_wait_ms: 20.0,
            wall_ms: 0.0,
        }
    }

    const MIB: u64 = 1 << 20;

    fn staged_preweight(diagnostic: bool) -> PairedPreweight {
        use qwen_llm::deepseek_v41::forward::RequestMemoryEstimate;
        use qwen_llm::deepseek_v41::gguf::{ByteCounters, ResidentStoragePlan};
        let caller = caller_cpu_bytes(diagnostic);
        let plan = ColdPrefillPlan {
            prefill_admission: qwen_llm::deepseek_v41::forward::PrefillAdmission {
                intent: qwen_llm::deepseek_v41::forward::PrefillIntent::RequiredFullCed,
                fallback: None,
            },
            geometry: qwen_llm::deepseek_v41::forward::PrefillGeometry {
                start: 0,
                tokens: 0,
                selected_rows: 4096,
            },
            capacity: 0,
            executions: vec![
                PrefillExecution::FullForward,
                PrefillExecution::ShrinkingCed,
            ],
            alternatives: Vec::new(),
            full_chunk_rows: qwen_llm::deepseek_v41::forward::PREFILL_CHUNK_ROWS,
            resident_storage: ResidentStoragePlan {
                page_size: 16384,
                max_buffer_length: 1 << 30,
                soft_coalescing_target: None,
                bytes: ByteCounters::default(),
                windows: vec![],
                metal_view_bytes: 400 * MIB,
                private_shadow_bytes: 20 * MIB,
                shared_eof_window_count: 0,
                private_window_count: 0,
            },
            request: RequestMemoryEstimate {
                cpu_persistent_bytes: 10 * MIB,
                cpu_transient_bytes: 10 * MIB,
                cpu_bytes: 20 * MIB,
                metal_bytes: 100 * MIB,
                total_bytes: 120 * MIB,
            },
            full_incremental_metal_bytes: 200 * MIB,
            full_incremental_cpu_bytes: 10 * MIB,
            ced: Some(CedMemoryEstimate {
                incremental_metal_bytes: 300 * MIB,
                incremental_cpu_bytes: 20 * MIB,
                ..Default::default()
            }),
            runtime_metal_bytes: 400 * MIB,
            runtime_cpu_bytes: 40 * MIB,
            binding_metal_bytes: 10 * MIB,
            binding_cpu_bytes: 30 * MIB,
            loader_control_cpu_bytes: 0,
            loader_controls: Default::default(),
            descriptor_window_cpu_bytes: 0,
            retained_summary_cpu_bytes: 0,
            caller_cpu_bytes: caller,
            required_metal_bytes: 810 * MIB,
            required_cpu_bytes: 90 * MIB + caller,
            load_metal_bytes: 410 * MIB,
            load_cpu_bytes: 50 * MIB + caller,
            runtime_gate_cpu_bytes: 70 * MIB + caller,
        };
        let report = preweight_report(
            &plan,
            FamilyBenchExtent {
                forwards: 3,
                prefill_endpoint: 3,
            },
        )
        .unwrap();
        PairedPreweight {
            plan,
            measurement: measurement_fixture(false, diagnostic),
            report,
        }
    }

    fn staged_signals() -> MetalMemorySignals {
        MetalMemorySignals {
            recommended_max_bytes: 4000 * MIB,
            current_allocated_bytes: 0,
            process_limit_remaining_bytes: Some(4000 * MIB),
            host_budget: Some(qwen_llm::metal::HostMemoryBudget {
                physical_bytes: 5000 * MIB,
                // Load CPU fits; the whole peak's CPU payload does not.
                conservative_available_bytes: 60 * MIB + caller_cpu_bytes(true),
                external_gap_bytes: 0,
                private_shadow_bytes: 0,
            }),
        }
    }

    #[test]
    fn staged_load_allows_only_future_availability_denial_and_reports_static_budgets() {
        for diagnostic in [false, true] {
            let preweight = staged_preweight(diagnostic);
            let calls = Cell::new(0);
            with_load(Some(&preweight), staged_signals, |report| {
                calls.set(calls.get() + 1);
                let report = report.unwrap();
                assert_eq!(report["peak_capacity_passed"], true);
                assert_eq!(report["load_admission"]["admitted"], true);
                assert_eq!(report["whole_peak_now"]["admitted"], false);
                assert_eq!(
                    report["whole_peak_now"]["reason"],
                    "HostAvailabilityInsufficient"
                );
                assert_eq!(
                    report["whole_peak_now_scope"],
                    "diagnostic_only_not_runtime_admission"
                );
                assert!(report.get("accepted").is_none());
                assert!(report.get("runtime_admission").is_none());
                assert_eq!(
                    report["request_metal_bytes"],
                    preweight.plan.request.metal_bytes
                );
                assert_eq!(
                    report["request_cpu_bytes"],
                    preweight.plan.request.cpu_bytes
                );
                for (field, bytes) in [
                    ("required_metal_bytes", preweight.plan.required_metal_bytes),
                    ("required_cpu_bytes", preweight.plan.required_cpu_bytes),
                    ("load_metal_bytes", preweight.plan.load_metal_bytes),
                    ("load_cpu_bytes", preweight.plan.load_cpu_bytes),
                    ("runtime_metal_bytes", preweight.plan.runtime_metal_bytes),
                    ("runtime_cpu_bytes", preweight.plan.runtime_cpu_bytes),
                    (
                        "runtime_gate_cpu_bytes",
                        preweight.plan.runtime_gate_cpu_bytes,
                    ),
                    ("caller_cpu_bytes", caller_cpu_bytes(diagnostic)),
                ] {
                    assert_eq!(report[field].as_u64(), Some(bytes), "{field}");
                }
                Ok(())
            })
            .unwrap();
            assert_eq!(calls.get(), 1);
        }
        with_load(
            None,
            || panic!("ordinary suite must not probe a pair gate"),
            |report| {
                assert!(report.is_none());
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn capacity_and_load_denials_stop_before_the_resident_factory() {
        for fault in [
            "device",
            "ownership",
            "process",
            "host_missing",
            "process_missing",
            "invalid_host",
            "load_availability",
            "overflow",
        ] {
            let mut preweight = staged_preweight(false);
            let mut signals = staged_signals();
            match fault {
                "device" => signals.recommended_max_bytes = preweight.plan.required_metal_bytes - 1,
                "ownership" => {
                    let host = signals.host_budget.as_mut().unwrap();
                    // Isolate physical capacity from the independent device bound.
                    signals.recommended_max_bytes = host.physical_bytes;
                    host.external_gap_bytes = host.physical_bytes
                        - preweight.plan.required_metal_bytes
                        - preweight.plan.required_cpu_bytes
                        + 1;
                }
                "process" => {
                    signals.process_limit_remaining_bytes = Some(
                        preweight.plan.required_metal_bytes + preweight.plan.required_cpu_bytes - 1,
                    )
                }
                "host_missing" => signals.host_budget = None,
                "process_missing" => signals.process_limit_remaining_bytes = None,
                "invalid_host" => {
                    signals
                        .host_budget
                        .as_mut()
                        .unwrap()
                        .conservative_available_bytes = 5001 * MIB
                }
                "load_availability" => {
                    signals
                        .host_budget
                        .as_mut()
                        .unwrap()
                        .conservative_available_bytes = preweight.plan.load_cpu_bytes - 1
                }
                "overflow" => preweight.plan.required_metal_bytes = u64::MAX,
                _ => unreachable!(),
            }
            let result: Result<()> = with_load(
                Some(&preweight),
                || signals,
                |_| panic!("resident factory entered after {fault} denial"),
            );
            assert!(result.is_err(), "{fault}");
        }
    }

    #[test]
    fn fresh_runtime_gate_follows_every_state_drop_and_pool_drain_and_releases_resident() {
        struct Resident(Rc<Cell<bool>>);
        impl Drop for Resident {
            fn drop(&mut self) {
                self.0.set(false);
            }
        }
        struct State(Rc<RefCell<Vec<&'static str>>>);
        impl Drop for State {
            fn drop(&mut self) {
                self.0.borrow_mut().push("state_drop");
            }
        }
        struct Drain(Rc<RefCell<Vec<&'static str>>>);
        impl Drop for Drain {
            fn drop(&mut self) {
                self.0.borrow_mut().push("drain");
            }
        }
        struct Probe {
            trace: Rc<RefCell<Vec<&'static str>>>,
            after: bool,
        }
        impl DiagnosticProbe for Probe {
            type Scope = profile::Scope;
            fn resources(&mut self) -> Result<DiagnosticResources> {
                self.trace.borrow_mut().push(if self.after {
                    "snapshot_after"
                } else {
                    "snapshot"
                });
                self.after = !self.after;
                Ok(DiagnosticResources {
                    process_pageins: 0,
                    process_disk_read_bytes: 0,
                    process_resident_size_bytes: 0,
                    process_phys_footprint_bytes: 0,
                    device_current_allocated_bytes: 0,
                })
            }
            fn pipelines(&self) -> MetalPipelineCacheMetrics {
                Default::default()
            }
            fn begin(&mut self) -> Result<Self::Scope> {
                self.trace.borrow_mut().push("scope");
                profile::Scope::begin().map_err(anyhow::Error::msg)
            }
            fn finish(&mut self, scope: Self::Scope) -> profile::Report {
                self.trace.borrow_mut().push("finish");
                scope.finish();
                profile_report(vec![])
            }
        }
        for diagnostic in [false, true] {
            for stop in 0..=10 {
                let preweight = staged_preweight(diagnostic);
                let resident_live = Rc::new(Cell::new(false));
                let trace = Rc::new(RefCell::new(Vec::new()));
                let assessments = Cell::new(0);
                let states = Cell::new(0);
                let prefills = Cell::new(0);
                let checkpoints = Cell::new(0);
                let result = with_load(Some(&preweight), staged_signals, |_| {
                    resident_live.set(true);
                    let _resident = Resident(resident_live.clone());
                    let mut bench = v41_pair_test_bench(
                        || {
                            states.set(states.get() + 1);
                            trace.borrow_mut().push("state");
                            Ok(State(trace.clone()))
                        },
                        |_, _, execution| {
                            prefills.set(prefills.get() + 1);
                            trace.borrow_mut().push("prefill");
                            Ok((vec![0.; 257], report(execution)))
                        },
                    );
                    let mut probe = Probe {
                        trace: trace.clone(),
                        after: false,
                    };
                    collect_assessed(
                        PairSchedule::Interleaved,
                        2,
                        || {
                            checkpoints.set(checkpoints.get() + 1);
                            Ok(())
                        },
                        || {
                            assert!(resident_live.get());
                            let index = assessments.get();
                            if index > 0 {
                                assert_eq!(
                                    trace.borrow().last(),
                                    Some(&if diagnostic { "finish_event" } else { "drain" })
                                );
                            }
                            assessments.set(index + 1);
                            trace.borrow_mut().push("runtime");
                            let mut signals = staged_signals();
                            signals.current_allocated_bytes = preweight.plan.load_metal_bytes;
                            let host = signals.host_budget.as_mut().unwrap();
                            host.private_shadow_bytes =
                                preweight.plan.resident_storage.private_shadow_bytes;
                            host.conservative_available_bytes = host.private_shadow_bytes
                                + if index == stop {
                                    preweight.plan.runtime_gate_cpu_bytes - 1
                                } else {
                                    600 * MIB
                                };
                            let admission = preweight.plan.assess_runtime(signals)?;
                            assert!(admission.is_admitted());
                            assert_eq!(
                                admission.required_bytes,
                                Some(
                                    preweight.plan.runtime_metal_bytes
                                        + preweight.plan.runtime_gate_cpu_bytes
                                )
                            );
                            Ok(())
                        },
                        |case| {
                            let mut execute = || {
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
                                    fingerprint_logits,
                                )
                            };
                            let mut sample = if diagnostic {
                                diagnostic_case_with(&mut probe, case, execute, |event| {
                                    trace.borrow_mut().push(match event {
                                        DiagnosticEvent::Start(_) => "start_event",
                                        DiagnosticEvent::Finish(_) => "finish_event",
                                    });
                                    Ok(())
                                })?
                            } else {
                                execute()?
                            };
                            sample.sample_ns = 100;
                            Ok(sample)
                        },
                    )
                });
                assert_eq!(result.is_ok(), stop == 10);
                assert!(!resident_live.get());
                assert_eq!(states.get(), stop);
                assert_eq!(prefills.get(), stop);
                assert_eq!(assessments.get(), if stop == 10 { 10 } else { stop + 1 });
                assert_eq!(checkpoints.get(), if stop == 10 { 11 } else { stop + 1 });
                let mut expected = Vec::new();
                for _ in 0..stop {
                    expected.push("runtime");
                    if diagnostic {
                        expected.extend(["snapshot", "start_event", "scope"]);
                    }
                    expected.extend(["state", "prefill", "state_drop", "drain"]);
                    if diagnostic {
                        expected.extend(["finish", "snapshot_after", "finish_event"]);
                    }
                }
                if stop < 10 {
                    expected.push("runtime");
                }
                assert_eq!(*trace.borrow(), expected);
                if let Ok(evidence) = result {
                    assert_eq!(evidence.len(), 10);
                    assert!(
                        evidence
                            .iter()
                            .all(|(_, sample)| sample.instrumented == diagnostic)
                    );
                }
                drop(profile::Scope::begin().expect("runtime denial leaves no diagnostic scope"));
            }
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
            let case = schedule(PairSchedule::Interleaved, 2).unwrap()[0];
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
                            stream_diagnostic_event(
                                &mut BrokenWriter,
                                &measurement_fixture(false, true),
                                case,
                                event,
                            )
                        } else {
                            stream_diagnostic_event(
                                &mut bytes,
                                &measurement_fixture(false, true),
                                case,
                                event,
                            )
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
    fn recorded_2k_spread_completes_with_finite_continuation_and_full_chronology() {
        let seconds = [
            26.2563, 19.2572, 17.0546, 11.3095, 16.9522, 16.9579, 13.0913, 14.9084, 2.3451,
        ];
        let mut bench = v41_pair_test_bench(
            || Ok(()),
            |_, _, execution| Ok((vec![0.; 257], report(execution))),
        );
        let mut streamed = Vec::new();
        let mut calls = Vec::new();
        let evidence = collect(
            PairSchedule::Interleaved,
            2,
            || Ok(()),
            |case| {
                calls.push(case.case_index);
                // Case 9 was never recorded by v1; use a finite synthetic continuation.
                let seconds = seconds.get(case.case_index).copied().unwrap_or(12.0);
                let mut sample = native_case(&mut bench, &row_context(), 3, 1, case)?;
                sample.sample_ns = (seconds * 1e9) as u64;
                sample.instrumented = true;
                stream_diagnostic(
                    &mut streamed,
                    &measurement_fixture(false, true),
                    case,
                    "fixture_finish",
                    &sample.sample_ns,
                )?;
                Ok(sample)
            },
        )
        .unwrap();
        assert_eq!(calls, (0..10).collect::<Vec<_>>());
        let events: Vec<serde_json::Value> = std::str::from_utf8(&streamed)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(events.len(), 10);
        for (index, event) in events.iter().enumerate() {
            assert_eq!(event["case"]["case_index"], index);
        }
        assert!(
            events
                .iter()
                .all(|e| e["instrumented"] == true && e["performance_evidence"] == false)
        );
        let rows = aggregate(
            &row_context(),
            3,
            2,
            1,
            257,
            &serde_json::json!({}),
            &measurement_fixture(false, true),
            &evidence,
        )
        .unwrap();
        for row in &rows {
            let paired = &row.diagnostics.as_ref().unwrap()["deepseek_v41"]["paired_benchmark"];
            assert_eq!(paired["schedule_status"], "completed");
            assert_eq!(paired["pilot_status"], "completed");
            let chronology = paired["native_chronological_cases"].as_array().unwrap();
            assert_eq!(chronology.len(), 10);
            for (index, case) in chronology.iter().enumerate() {
                assert_eq!(case["case"]["case_index"], index);
                assert_eq!(case["sample_ns"], evidence[index].1.sample_ns);
            }
        }
        let full = &rows[0].diagnostics.as_ref().unwrap()["deepseek_v41"]["paired_benchmark"];
        let spread = &full["native_dispersion"];
        let min = evidence[8].1.sample_ns as f64;
        for (scope, max) in [
            ("pilot_plus_main", evidence[2].1.sample_ns as f64),
            ("retained_main", evidence[7].1.sample_ns as f64),
        ] {
            assert_eq!(spread[scope]["min_ns"], min);
            assert_eq!(spread[scope]["max_ns"], max);
            assert_eq!(spread[scope]["max_over_min"], max / min);
            assert!(spread[scope]["max_over_min"].as_f64().unwrap() > 1.5);
        }
    }

    #[test]
    fn diagnostic_rows_and_every_case_are_marked_without_retaining_profiles() {
        for (production, instrumented) in
            [(false, false), (false, true), (true, false), (true, true)]
        {
            let measurement = measurement_fixture(production, instrumented);
            let mut bench = v41_pair_test_bench(
                || Ok(()),
                |_, _, execution| Ok((vec![0.; 257], report(execution))),
            );
            let evidence = collect(
                PairSchedule::Interleaved,
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
                &measurement,
                &evidence,
            )
            .unwrap();
            for row in rows {
                let diagnostics = row.diagnostics.unwrap();
                let paired = &diagnostics["deepseek_v41"]["paired_benchmark"];
                assert_eq!(paired["protocol"], measurement.protocol());
                assert_eq!(
                    measurement.protocol(),
                    match (production, instrumented) {
                        (false, false) => "native_full_ced_abba_baab_v2",
                        (false, true) => "instrumented_full_ced_abba_baab_diagnostic_v2",
                        (true, false) => "native_full_ced_abba_baab_production_timing_v2",
                        (true, true) =>
                            "instrumented_full_ced_abba_baab_production_timing_diagnostic_v2",
                    }
                );
                assert_eq!(paired["measurement"], measurement.description());
                assert_eq!(paired["schedule_status"], "completed");
                assert_eq!(paired["pilot_status"], "completed");
                assert_eq!(
                    paired["timing_interpretation"],
                    "observational_not_automatic_performance_gain"
                );
                assert!(paired.get("max_within_policy_spread").is_none());
                assert!(paired.get("stability_baseline").is_none());
                assert_eq!(
                    paired["metal_debug_layer"],
                    serde_json::to_value(
                        measurement
                            .environment
                            .get("MTL_DEBUG_LAYER")
                            .and_then(|v| v.as_deref())
                    )
                    .unwrap()
                );
                assert_eq!(
                    paired["measurement"]["canonical_production_timing_requested"],
                    production && !instrumented
                );
                if instrumented {
                    assert_eq!(diagnostics["instrumented"], true);
                    assert_eq!(diagnostics["performance_evidence"], false);
                    assert_eq!(paired["instrumented"], true);
                    assert_eq!(paired["performance_evidence"], false);
                } else {
                    assert!(diagnostics.get("instrumented").is_none());
                    assert!(paired.get("instrumented").is_none());
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
                    assert!(case.get("measurement").is_none());
                }
            }
        }
    }

    #[test]
    fn diagnostic_gap_does_not_subtract_gpu_twice_or_sum_nested_host_spans() {
        let mut host = profile_operation();
        host.completed_submits = 0;
        host.gpu_ms = None;
        host.driver_scheduling_ms = None;
        host.encode_ms = 0.0;
        host.commit_wait_ms = 0.0;
        host.wall_ms = 40.0;
        let report = profile_report(vec![profile_operation(), host]);
        let totals = command_totals(&report);
        assert_eq!(totals.gpu_ms_valid_completed_only, Some(12.0));
        assert_eq!(totals.driver_scheduling_ms_valid_completed_only, Some(3.0));
        assert_eq!(totals.outer_wall_minus_command_encode_and_wait_ms, 75.0);
        assert_eq!(totals.completed_submits, 1);
        validate_diagnostic_report(&report).unwrap();
    }

    #[test]
    fn diagnostic_scheduling_totals_serialize_unknown_as_null_not_zero() {
        let mut op = profile_operation();
        op.driver_scheduling_ms = None;
        op.invalid_driver_scheduling_timestamps = 1;
        op.encode_ms = 0.0;
        let report = profile_report(vec![op]);
        let totals = command_totals(&report);
        let json = serde_json::to_value(totals).unwrap();
        assert!(json["driver_scheduling_ms_valid_completed_only"].is_null());
        assert_eq!(json["invalid_driver_scheduling_timestamps"], 1);
        assert_eq!(json["encode_ms"], 0.0);
        assert_eq!(json["gpu_ms_valid_completed_only"], 12.0);
        assert_eq!(json["outer_wall_minus_command_encode_and_wait_ms"], 80.0);
    }

    #[test]
    fn diagnostic_collection_and_stream_errors_stop_the_driver() {
        let mut report = profile_report(vec![]);
        report.process_pageins = None;
        let calls = Cell::new(0);
        assert!(
            drive(
                PairSchedule::Interleaved,
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
                PairSchedule::Interleaved,
                2,
                || Ok(()),
                |case| {
                    calls.set(calls.get() + 1);
                    stream_diagnostic(
                        &mut BrokenWriter,
                        &measurement_fixture(false, true),
                        case,
                        "fixture",
                        &(),
                    )?;
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
        let maximum_report = || {
            profile_report(
                (0..MAX_DIAGNOSTIC_OPERATIONS)
                    .map(|_| {
                        let mut op = profile_operation();
                        op.operation = "ced_encoder_engram_prepare";
                        op.dimensions = [usize::MAX; 4];
                        op.count = usize::MAX;
                        // Keep production totals finite/nonoverflowing while
                        // stressing long numeric encodings in every operation.
                        op.completed_submits = usize::MAX / MAX_DIAGNOSTIC_OPERATIONS;
                        op.invalid_gpu_timestamps = usize::MAX / MAX_DIAGNOSTIC_OPERATIONS;
                        op.gpu_ms = Some(f64::MAX / (2 * MAX_DIAGNOSTIC_OPERATIONS) as f64);
                        op.driver_scheduling_ms = op.gpu_ms;
                        op.invalid_driver_scheduling_timestamps = op.invalid_gpu_timestamps;
                        op.encode_ms = op.gpu_ms.unwrap();
                        op.commit_wait_ms = op.gpu_ms.unwrap();
                        op.wall_ms = f64::MAX;
                        op
                    })
                    .collect(),
            )
        };
        let report = maximum_report();
        validate_diagnostic_report(&report).unwrap();
        let operation_capacity = report.operations.capacity();
        let live_operations =
            report.operations.capacity() * std::mem::size_of::<profile::Operation>();
        drop(report);
        let cases = schedule(PairSchedule::Interleaved, 16).unwrap();
        assert_eq!(cases.len(), 38);
        struct MaximumDiagnosticProbe(Option<profile::Report>);
        impl DiagnosticProbe for MaximumDiagnosticProbe {
            type Scope = ();
            fn resources(&mut self) -> Result<DiagnosticResources> {
                Ok(DiagnosticResources {
                    process_pageins: u64::MAX,
                    process_disk_read_bytes: u64::MAX,
                    process_resident_size_bytes: u64::MAX,
                    process_phys_footprint_bytes: u64::MAX,
                    device_current_allocated_bytes: u64::MAX,
                })
            }
            fn pipelines(&self) -> MetalPipelineCacheMetrics {
                MetalPipelineCacheMetrics::default()
            }
            fn begin(&mut self) -> Result<()> {
                Ok(())
            }
            fn finish(&mut self, (): ()) -> profile::Report {
                self.0.take().unwrap()
            }
        }
        // Use the production finish-event serializer, including labels/totals,
        // without retaining JSON or profiles across cases (or in either row).
        let mut largest_event = 0;
        let mut largest_charged = 0;
        for case in &cases {
            let mut writer = CountBytes(0);
            let mut probe = MaximumDiagnosticProbe(Some(maximum_report()));
            diagnostic_case_with(
                &mut probe,
                *case,
                || Err(anyhow::anyhow!("CPU fixture")),
                |event| {
                    stream_diagnostic_event(
                        &mut writer,
                        &measurement_fixture(false, true),
                        *case,
                        event,
                    )
                },
            )
            .unwrap_err();
            largest_event = largest_event.max(writer.0);
            // Also charge serialized bytes conservatively although the real
            // writer streams without keeping a JSON string or parsed DOM.
            let fixed = std::mem::size_of::<DiagnosticObservation<'_>>()
                + std::mem::size_of::<profile::Report>()
                + 2 * std::mem::size_of::<DiagnosticResources>()
                + cases.capacity() * std::mem::size_of::<Case>();
            let charged = live_operations + writer.0 + fixed;
            largest_charged = largest_charged.max(charged);
            assert!(charged as u64 <= DIAGNOSTIC_CPU_BYTES);
        }
        println!(
            "attribution-v26-diagnostic-bound operation_bytes={} capacity={} live_operations_bytes={live_operations} largest_start_plus_finish_bytes={largest_event} charged_bytes={largest_charged} cases={} diagnostic_reserve_bytes={DIAGNOSTIC_CPU_BYTES} native_reserve_bytes={CALLER_CPU_BYTES}",
            std::mem::size_of::<profile::Operation>(),
            operation_capacity,
            cases.len()
        );
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
                        full_chunk_rows: usize::MAX,
                        retained_residual_bytes: u64::MAX,
                        retained_mix_bytes: u64::MAX,
                        retained_selected_bytes: u64::MAX,
                        retained_candidate_bytes: u64::MAX,
                        priced_retained_bytes: u64::MAX,
                        priced_chunk_bytes: u64::MAX,
                        priced_pipeline_chunk_bytes: u64::MAX,
                        priced_half_chunk_bytes: u64::MAX,
                        priced_publication_bytes: u64::MAX,
                        priced_seed_bytes: u64::MAX,
                        priced_batch_status_bytes: u64::MAX,
                        batch_cpu_bytes: u64::MAX,
                        priced_batch_diagnostic_bytes: u64::MAX,
                        batch_encoding_reserve_bytes: u64::MAX,
                        decoder_batch_layers: usize::MAX,
                        decoder_schedule_reason: Some("invalid test ledger"),
                        command_reserve_bytes: u64::MAX,
                        incremental_metal_bytes: u64::MAX,
                        incremental_cpu_bytes: u64::MAX,
                        priced_pipeline_cpu_bytes: u64::MAX,
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
            PairSchedule::Interleaved,
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
        let preweight_plan = staged_preweight(false);
        let mut preweight = with_load(Some(&preweight_plan), staged_signals, |report| {
            Ok(report.unwrap().clone())
        })
        .unwrap();
        fn saturate_numbers(value: &mut serde_json::Value) {
            match value {
                serde_json::Value::Number(_) => *value = u64::MAX.into(),
                serde_json::Value::Object(fields) => {
                    fields.values_mut().for_each(saturate_numbers);
                }
                serde_json::Value::Array(values) => values.iter_mut().for_each(saturate_numbers),
                _ => {}
            }
        }
        saturate_numbers(&mut preweight);
        preweight["prompt_rows"] = MAX_PP.into();
        preweight["capacity"] = MAX_PP.into();
        preweight["caller_cpu_bytes"] = CALLER_CPU_BYTES.into();
        // Stress JSON escaping and map overhead, not only a long unescaped value.
        let (measurement, provenance_env_bytes, provenance_keys) = escaped_many_key_measurement();
        let rows = aggregate(
            &row_context(),
            MAX_PP,
            RUNS,
            u64::MAX,
            VOCAB,
            &preweight,
            &measurement,
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
        assert_eq!(
            json.matches("\"canonical_production_timing_requested\"")
                .count(),
            2
        );
        assert_eq!(json.matches("\"layers\":[").count(), 2 * 19 + RUNS);
        std::hint::black_box((&head, &bits, &rows, &evidence, &json, &measurement));
        let result_peak_delta = caller_envelope_peak_rss_bytes()
            .checked_sub(before)
            .unwrap();
        let charged_bytes = head_and_bits_bytes.checked_add(result_peak_delta).unwrap();
        println!(
            "paired-caller-envelope-rss head_and_bits_bytes={head_and_bits_bytes} result_peak_delta_bytes={result_peak_delta} remaining_bytes={remaining_bytes} charged_bytes={charged_bytes} limit_bytes={CALLER_CPU_BYTES} json_bytes={} provenance_env_bytes={provenance_env_bytes} provenance_keys={provenance_keys} measurement=observational_not_heap_proof",
            json.len()
        );
        assert!(
            result_peak_delta <= remaining_bytes,
            "caller envelope exceeded: live head/bits {head_and_bits_bytes} + result RSS delta {result_peak_delta} = {charged_bytes} > {CALLER_CPU_BYTES}; no extra tolerance"
        );
        drop((head, bits, rows, evidence, json));
    }

    #[test]
    fn blocked_schedules_bound_all_counts_phases_and_policy_transitions() {
        for mode in [PairSchedule::FullBlocksFirst, PairSchedule::CedBlocksFirst] {
            let policies = if mode == PairSchedule::FullBlocksFirst {
                [Policy::A, Policy::B]
            } else {
                [Policy::B, Policy::A]
            };
            for runs in (4..=16).step_by(2) {
                let cases = schedule(mode, runs).unwrap();
                assert_eq!(cases.len(), 6 + 2 * runs);
                assert!(cases.len() <= 38);
                for (index, case) in cases.iter().enumerate() {
                    assert_eq!(case.case_index, index);
                    assert_eq!(case.retained, case.phase == Phase::Main);
                }
                for (block, &policy) in policies.iter().enumerate() {
                    let stream = &cases[block * (runs + 3)..(block + 1) * (runs + 3)];
                    assert!(
                        stream
                            .iter()
                            .all(|c| c.policy == policy && c.block == block)
                    );
                    assert_eq!(stream[0].phase, Phase::Warmup);
                    assert!(stream[1..3].iter().all(|c| c.phase == Phase::Pilot));
                    assert!(stream[3..].iter().all(|c| c.phase == Phase::Main));
                    assert_eq!(stream[1].order_in_block, 0);
                    assert_eq!(stream[2].order_in_block, 1);
                    assert_eq!(
                        stream[3..]
                            .iter()
                            .map(|c| c.order_in_block)
                            .collect::<Vec<_>>(),
                        (0..runs).collect::<Vec<_>>()
                    );
                }
                assert_eq!(
                    cases
                        .windows(2)
                        .filter(|w| w[0].policy != w[1].policy)
                        .count(),
                    1
                );
            }
            for runs in [0, 1, 2, 3, 5, 17, 18, usize::MAX] {
                assert!(schedule(mode, runs).is_err());
            }
            assert!(mode.validate(4, true).is_err());
        }
    }

    #[test]
    fn blocked_native_calls_keep_cold_input_raw_chronology_and_retained_statistics() {
        for mode in [PairSchedule::FullBlocksFirst, PairSchedule::CedBlocksFirst] {
            for runs in [4, 16] {
                let live = Rc::new(Cell::new(0));
                let calls = RefCell::new(Vec::new());
                let mut bench = v41_pair_test_bench(
                    || {
                        assert_eq!(live.replace(1), 0);
                        Ok(ColdState {
                            live: live.clone(),
                            rows: 0,
                        })
                    },
                    |state, ids, execution| {
                        assert_eq!(state.rows, 0);
                        state.rows += ids.len();
                        calls.borrow_mut().push((ids.to_vec(), execution));
                        Ok((vec![0.; 257], report(execution)))
                    },
                );
                let checkpoints = Cell::new(0);
                let assessments = Cell::new(0);
                let evidence = collect_assessed(
                    mode,
                    runs,
                    || {
                        assert_eq!(live.get(), 0);
                        checkpoints.set(checkpoints.get() + 1);
                        Ok(())
                    },
                    || {
                        assert_eq!(live.get(), 0);
                        assessments.set(assessments.get() + 1);
                        Ok(())
                    },
                    |case| {
                        let mut sample = native_case_with(
                            &mut bench,
                            &row_context(),
                            3,
                            42,
                            case,
                            |body| body(),
                            fingerprint_logits,
                        )?;
                        assert_eq!(live.get(), 0);
                        drop(
                            profile::Scope::begin()
                                .expect("blocked native case leaves no profiling scope"),
                        );
                        sample.sample_ns = (case.case_index as u64 + 1) * 1000;
                        Ok(sample)
                    },
                )
                .unwrap();
                assert_eq!(checkpoints.get(), evidence.len() + 1);
                assert_eq!(assessments.get(), evidence.len());
                let calls = calls.borrow();
                assert_eq!(calls.len(), evidence.len());
                assert!(calls.iter().all(|(ids, _)| ids == &calls[0].0));
                let measurement = measurement_fixture(true, false).with_schedule(mode);
                let rows = aggregate(
                    &row_context(),
                    3,
                    runs,
                    42,
                    257,
                    &serde_json::json!({}),
                    &measurement,
                    &evidence,
                )
                .unwrap();
                for (row, policy) in rows.iter().zip([Policy::A, Policy::B]) {
                    let expected = evidence
                        .iter()
                        .filter(|(c, _)| c.retained && c.policy == policy)
                        .map(|(_, s)| s.sample_ns)
                        .collect::<Vec<_>>();
                    assert_eq!(row.samples_ns, expected);
                    assert_eq!(row.n_repetitions, runs);
                    let paired =
                        &row.diagnostics.as_ref().unwrap()["deepseek_v41"]["paired_benchmark"];
                    assert_eq!(paired["schema"], "v41-ced-pair-blocked-v1");
                    assert_eq!(paired["schedule"], serde_json::json!(mode));
                    let raw = paired["native_chronological_cases"].as_array().unwrap();
                    assert_eq!(raw.len(), evidence.len());
                    for (entry, (case, sample)) in raw.iter().zip(&evidence) {
                        assert_eq!(entry["case"], serde_json::to_value(case).unwrap());
                        assert_eq!(entry["sample_ns"], sample.sample_ns);
                        assert_eq!(entry["endpoint_bitwise_matches_warm_full"], true);
                        if case.phase == Phase::Pilot && case.order_in_block == 1 {
                            assert_eq!(entry["pilot_status"], "completed");
                        }
                    }
                }
                assert!(
                    aggregate(
                        &row_context(),
                        3,
                        runs,
                        42,
                        257,
                        &serde_json::json!({}),
                        &measurement,
                        &evidence[..evidence.len() - 1]
                    )
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn blocked_invalid_calls_stop_without_retry_or_later_policy_execution() {
        for mode in [PairSchedule::FullBlocksFirst, PairSchedule::CedBlocksFirst] {
            for stop in 0..14 {
                let calls = Cell::new(0);
                let result = drive(
                    mode,
                    4,
                    || Ok(()),
                    |case| {
                        calls.set(calls.get() + 1);
                        ensure!(case.case_index != stop, "injected execution failure");
                        Ok((1., ()))
                    },
                );
                assert!(result.is_err());
                assert_eq!(calls.get(), stop + 1);
            }
            for invalid in [0., -1., f64::NAN, f64::INFINITY] {
                for bad in [1, 2, 3, 6, 8, 9, 10, 13] {
                    let calls = Cell::new(0);
                    assert!(
                        drive(
                            mode,
                            4,
                            || Ok(()),
                            |case| {
                                calls.set(calls.get() + 1);
                                Ok((if case.case_index == bad { invalid } else { 1. }, ()))
                            }
                        )
                        .is_err()
                    );
                    assert_eq!(calls.get(), bad + 1);
                }
            }
            let calls = Cell::new(0);
            assert!(
                drive(
                    mode,
                    2,
                    || Ok(()),
                    |_| {
                        calls.set(1);
                        Ok((1., ()))
                    }
                )
                .is_err()
            );
            assert_eq!(calls.get(), 0);
        }
    }

    #[test]
    fn blocked_launch_identity_and_invalid_arguments_precede_model_open() {
        use clap::Parser;
        for (text, mode) in [
            ("full-blocks-first", PairSchedule::FullBlocksFirst),
            ("ced-blocks-first", PairSchedule::CedBlocksFirst),
        ] {
            let argv = [
                "qwen-bench",
                "suite",
                "-m",
                "/nonexistent/blocked-fixture.gguf",
                "--pp",
                "128",
                "--runs",
                "4",
                "--v41-ced-pair",
                "--v41-ced-pair-schedule",
                text,
            ];
            let args = Args::try_parse_from(argv).unwrap();
            let mut startup = Vec::new();
            let measurement = crate::emit_pair_request(
                &args,
                |_, _| measurement_fixture(false, false),
                &mut startup,
            )
            .unwrap()
            .unwrap();
            let event: serde_json::Value = serde_json::from_slice(&startup).unwrap();
            assert_eq!(
                event["status"],
                "environment_check_pending_no_device_created"
            );
            assert_eq!(event["measurement"]["schedule"], text);
            assert_eq!(event["measurement"]["schema"], "v41-ced-pair-blocked-v1");
            assert!(
                measurement
                    .protocol()
                    .contains(text.replace('-', "_").as_str())
            );
            measurement.validate_schedule(mode, 4).unwrap();
            for (runs, diagnostic, pp, chunk, depth) in [
                (2, false, vec![128], None, vec![0]),
                (4, true, vec![128], None, vec![0]),
                (4, false, vec![8193], None, vec![0]),
                (4, false, vec![128], Some(128), vec![0]),
                (4, false, vec![128], None, vec![1]),
            ] {
                let Cmd::Suite(mut suite) = Args::try_parse_from(argv).unwrap().cmd else {
                    unreachable!()
                };
                suite.runs = runs;
                suite.v41_ced_pair_diagnostic = diagnostic;
                suite.pp = pp;
                suite.prefill_chunk = chunk;
                suite.depth = depth;
                let launch = measurement_fixture(false, diagnostic).with_schedule(mode);
                let error =
                    crate::suite::run_suite_with_measurement(suite, Some(launch)).unwrap_err();
                assert!(
                    !format!("{error:#}").contains("open /nonexistent"),
                    "{error:#}"
                );
            }
            let no_pair = [
                "qwen-bench",
                "suite",
                "-m",
                "fixture.gguf",
                "--v41-ced-pair-schedule",
                text,
            ];
            assert!(Args::try_parse_from(no_pair).is_err());
        }
        for value in ["", "full", "auto", "FullBlocksFirst", "ced-blocks-first "] {
            assert!(
                Args::try_parse_from([
                    "qwen-bench",
                    "suite",
                    "-m",
                    "fixture.gguf",
                    "--v41-ced-pair",
                    "--v41-ced-pair-schedule",
                    value
                ])
                .is_err()
            );
        }
        let default = measurement_fixture(false, false);
        assert_eq!(default.protocol(), "native_full_ced_abba_baab_v2");
        assert!(default.description().get("schedule").is_none());
    }

    #[test]
    fn bounded_schedule_is_two_policies_with_both_warm_before_pilot() {
        use Policy::{A, B};
        for runs in (2..=16).step_by(2) {
            let cases = schedule(PairSchedule::Interleaved, runs).unwrap();
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
            assert!(schedule(PairSchedule::Interleaved, runs).is_err());
        }
    }

    #[test]
    fn dispersion_is_finite_positive_without_a_spread_gate() {
        for samples in [[2., 3.], [1., 1.50001], [1., 100.]] {
            let spread = dispersion(&samples).unwrap();
            assert_eq!(spread.min_ns, samples[0]);
            assert_eq!(spread.max_ns, samples[1]);
            assert_eq!(spread.max_over_min, samples[1] / samples[0]);
        }
        for values in [
            vec![],
            vec![1.],
            vec![0., 1.],
            vec![-1., 1.],
            vec![1., f64::NAN],
            vec![1., f64::INFINITY],
            vec![1., f64::NEG_INFINITY],
        ] {
            assert!(dispersion(&values).is_err(), "{values:?}");
        }
        let evidence = drive(
            PairSchedule::Interleaved,
            2,
            || Ok(()),
            |case| Ok((if case.policy == Policy::A { 100. } else { 10. }, ())),
        )
        .unwrap();
        assert_eq!(
            evidence.len(),
            10,
            "cross-policy differences do not gate scheduling"
        );
    }

    #[test]
    fn pilot_and_main_spread_complete_but_invalid_samples_stop_immediately() {
        for varied_index in 2..14 {
            let calls = Cell::new(0);
            let result = drive(
                PairSchedule::Interleaved,
                4,
                || Ok(()),
                |case| {
                    calls.set(calls.get() + 1);
                    Ok((
                        if case.case_index == varied_index {
                            200.
                        } else {
                            100.
                        },
                        (),
                    ))
                },
            );
            assert_eq!(result.unwrap().len(), 14);
            assert_eq!(calls.get(), 14);
        }
        for invalid in [0., -1., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for bad_index in 2..14 {
                let calls = Cell::new(0);
                assert!(
                    drive(
                        PairSchedule::Interleaved,
                        4,
                        || Ok(()),
                        |case| {
                            calls.set(calls.get() + 1);
                            Ok((
                                if case.case_index == bad_index {
                                    invalid
                                } else {
                                    1.
                                },
                                (),
                            ))
                        }
                    )
                    .is_err()
                );
                assert_eq!(calls.get(), bad_index + 1);
            }
        }
    }

    #[test]
    fn cancellation_and_execution_error_never_schedule_later_cases() {
        for stop in 0..=10 {
            let checkpoints = Cell::new(0);
            let calls = Cell::new(0);
            let result = drive(
                PairSchedule::Interleaved,
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
                    PairSchedule::Interleaved,
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
            native_case(
                &mut bench,
                &row_context(),
                3,
                1,
                schedule(PairSchedule::Interleaved, 2).unwrap()[0],
            )
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
            PairSchedule::Interleaved,
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
            schedule(PairSchedule::Interleaved, 2)
                .unwrap()
                .iter()
                .map(|c| c.policy.execution())
                .collect::<Vec<_>>()
        );
        let preweight = serde_json::json!({"accepted": true, "caller_cpu_bytes": CALLER_CPU_BYTES});
        let rows = aggregate(
            &ctx,
            3,
            2,
            42,
            257,
            &preweight,
            &measurement_fixture(false, false),
            &evidence,
        )
        .unwrap();
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
            assert_eq!(paired["schedule_status"], "completed");
            assert_eq!(paired["pilot_status"], "completed");
            let spread = &paired["native_dispersion"];
            assert_eq!(
                spread["pilot_plus_main"]["min_ns"],
                (expected_ns / 11 * 10) as f64
            );
            assert_eq!(spread["pilot_plus_main"]["max_ns"], expected_ns as f64);
            assert_eq!(spread["pilot_plus_main"]["max_over_min"], 1.1);
            assert_eq!(spread["retained_main"]["min_ns"], expected_ns as f64);
            assert_eq!(spread["retained_main"]["max_ns"], expected_ns as f64);
            assert_eq!(spread["retained_main"]["max_over_min"], 1.0);
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
            assert_eq!(cases[0]["pilot_status"], "pending");
            assert_eq!(cases[2]["pilot_status"], "collecting");
            assert_eq!(cases[5]["pilot_status"], "completed");
            assert_eq!(cases[9]["pilot_status"], "completed");
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
        for mode in [
            PairSchedule::Interleaved,
            PairSchedule::FullBlocksFirst,
            PairSchedule::CedBlocksFirst,
        ] {
            let runs = if mode == PairSchedule::Interleaved {
                2
            } else {
                4
            };
            for stop in 0..schedule(mode, runs).unwrap().len() {
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
                                    "fallback" => {
                                        actual.fallback = Some("fixture CED refusal".into())
                                    }
                                    "wrong_policy" => {
                                        actual.execution =
                                            if execution == PrefillExecution::FullForward {
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
                        mode,
                        runs,
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
                    // A bad first warmup establishes the baseline; the next call rejects it.
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
    }

    #[test]
    fn malformed_ced_work_or_memory_stops_before_later_cases() {
        for mode in [
            PairSchedule::Interleaved,
            PairSchedule::FullBlocksFirst,
            PairSchedule::CedBlocksFirst,
        ] {
            let runs = if mode == PairSchedule::Interleaved {
                2
            } else {
                4
            };
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
                        mode,
                        runs,
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
                assert_eq!(
                    calls.get(),
                    schedule(mode, runs)
                        .unwrap()
                        .iter()
                        .position(|c| c.policy == Policy::B)
                        .unwrap()
                        + 1,
                    "fault={fault}"
                );
                assert_eq!(live.get(), 0);
            }
        }
    }

    #[test]
    fn p4_cli_requires_pair_and_launch_result_identity_preserves_selected_plan() {
        use clap::Parser;

        let base = [
            "qwen-bench",
            "suite",
            "-m",
            "/nonexistent/p4-fixture.gguf",
            "--pp",
            "8192",
            "--runs",
            "4",
        ];
        let error =
            Args::try_parse_from(base.into_iter().chain(["--v41-ced-pair-p4"])).unwrap_err();
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
        assert!(error.to_string().contains("--v41-ced-pair"));
        let Cmd::Suite(mut invalid) = Args::try_parse_from(base).unwrap().cmd else {
            panic!("suite")
        };
        assert!(!invalid.v41_ced_pair_p4);
        invalid.v41_ced_pair_p4 = true;
        let error = crate::suite::run_suite_with_measurement(invalid, None).unwrap_err();
        assert!(error.to_string().contains("--v41-ced-pair-p4 requires"));

        for p4 in [false, true] {
            for mode in [
                PairSchedule::Interleaved,
                PairSchedule::FullBlocksFirst,
                PairSchedule::CedBlocksFirst,
            ] {
                let mut args = Args::try_parse_from(
                    base.into_iter()
                        .chain(["--v41-ced-pair"])
                        .chain(p4.then_some("--v41-ced-pair-p4")),
                )
                .unwrap();
                let Cmd::Suite(suite) = &mut args.cmd else {
                    panic!("suite")
                };
                assert_eq!(suite.v41_ced_pair_p4, p4);
                suite.v41_ced_pair_schedule = Some(mode);
                let mut launch = Vec::new();
                let measurement = crate::emit_pair_request(
                    &args,
                    |_, _| measurement_fixture(false, false),
                    &mut launch,
                )
                .unwrap()
                .unwrap();
                measurement.validate_p4(p4).unwrap();
                assert!(measurement.validate_p4(!p4).is_err());
                let event: serde_json::Value = serde_json::from_slice(&launch).unwrap();
                let target = serde_json::to_value(p4.then_some(P4_SOFT_TARGET)).unwrap();
                assert_eq!(
                    event["measurement"]["resident_soft_coalescing_target_bytes"],
                    target
                );
                let mut preweight = staged_preweight(false);
                preweight.plan.resident_storage.soft_coalescing_target =
                    measurement.soft_coalescing_target;
                preweight.measurement = measurement;
                preweight.report = preweight_report(
                    &preweight.plan,
                    FamilyBenchExtent {
                        forwards: 8192,
                        prefill_endpoint: 8192,
                    },
                )
                .unwrap();
                let evidence = collect(
                    mode,
                    4,
                    || Ok(()),
                    |case| {
                        let mut bench = v41_pair_test_bench(
                            || Ok(()),
                            |_, _, execution| Ok((vec![0.; 257], report(execution))),
                        );
                        let mut sample = native_case(&mut bench, &row_context(), 3, 1, case)?;
                        sample.sample_ns = 100;
                        Ok(sample)
                    },
                )
                .unwrap();
                with_load_plan(Some(&preweight), staged_signals, |report, selected| {
                    assert_eq!(
                        selected.unwrap().resident_storage.soft_coalescing_target,
                        p4.then_some(P4_SOFT_TARGET)
                    );
                    assert_eq!(
                        report.unwrap()["resident_soft_coalescing_target_bytes"],
                        target
                    );
                    let rows = aggregate(
                        &row_context(),
                        3,
                        4,
                        1,
                        257,
                        report.unwrap(),
                        &preweight.measurement,
                        &evidence,
                    )?;
                    assert_eq!(rows.len(), 2);
                    for row in rows {
                        let diagnostics = row.diagnostics.unwrap();
                        let paired = &diagnostics["deepseek_v41"]["paired_benchmark"];
                        assert_eq!(paired["measurement"], event["measurement"]);
                        assert_eq!(
                            paired["preweight_plan"]["resident_soft_coalescing_target_bytes"],
                            target
                        );
                    }
                    Ok(())
                })
                .unwrap();
                let Cmd::Suite(suite) = args.cmd else {
                    panic!("suite")
                };
                let error = crate::suite::run_suite_with_measurement(
                    suite,
                    Some(
                        measurement_fixture(false, false)
                            .with_schedule(mode)
                            .with_p4(!p4),
                    ),
                )
                .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("mapping target differs from request")
                );
                preweight.plan.resident_storage.soft_coalescing_target =
                    (!p4).then_some(P4_SOFT_TARGET);
                let error =
                    with_load_plan(Some(&preweight), staged_signals, |_, _| -> Result<()> {
                        panic!("mismatched mapping must not load")
                    })
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("mapping target differs from launch")
                );
            }
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
        assert!(!args.v41_ced_pair_production_timing);
        assert_eq!(args.runs, 1, "ordinary defaults unchanged");
        assert!(
            measurement_fixture(false, false)
                .validate_environment()
                .is_ok()
        );
        assert!(
            measurement_vars(false, false, &[])
                .validate_environment()
                .is_err()
        );
        for name in [
            "QWEN_V41_PREFILL_EXECUTION",
            "QWEN_V41_SINK_TENSOR",
            "QWEN_V41_PREFILL_STAGES",
            "QWEN_V41_CENSUS",
            "QWEN_MATMAT_F16_HALF_ACT",
            "QWEN_MATVEC_IQ3_XXS_FAST",
        ] {
            assert!(
                measurement_vars(false, false, &[("MTL_DEBUG_LAYER", "1"), (name, "0")])
                    .validate_environment()
                    .is_err()
            );
        }
        assert!(
            measurement_vars(
                false,
                false,
                &[
                    ("MTL_DEBUG_LAYER", "1"),
                    ("QWEN_METAL_LEASE_SKIP_WIRED_GATE", "1")
                ]
            )
            .validate_environment()
            .is_err()
        );
    }

    #[test]
    fn adaptive_transition_cli_args_keep_selected_geometry_and_caller_bounds() {
        use clap::Parser;
        assert_eq!(MAX_DIAGNOSTIC_OPERATIONS, 4096);
        for n in [2047, 2048, 2049, 4095, 4096, 4097, 8192] {
            for diagnostic in [false, true] {
                for production in [false, true] {
                    let n_text = n.to_string();
                    let mut argv = vec![
                        "qwen-bench",
                        "suite",
                        "-m",
                        "fixture.gguf",
                        "--pp",
                        &n_text,
                        "--runs",
                        "2",
                        "--v41-ced-pair",
                    ];
                    if diagnostic {
                        argv.push("--v41-ced-pair-diagnostic");
                    }
                    if production {
                        argv.push("--v41-ced-pair-production-timing");
                    }
                    let Cmd::Suite(args) = Args::try_parse_from(argv).unwrap().cmd else {
                        panic!("suite")
                    };
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
                    assert_eq!(args.pp, [n]);
                    assert_eq!(args.v41_ced_pair_diagnostic, diagnostic);
                    assert_eq!(args.v41_ced_pair_production_timing, production);
                    let extent =
                        FamilyBenchExtent::for_shapes(&args.pp, &args.tg, &args.depth, (0, 0))
                            .unwrap();
                    assert_eq!(extent.prefill_endpoint, n);
                    assert_eq!(extent.forwards, n);
                    assert!(diagnostic_operation_bound(n, MAX_LAYERS) <= MAX_DIAGNOSTIC_OPERATIONS);
                    assert_eq!(
                        caller_cpu_bytes(diagnostic),
                        CALLER_CPU_BYTES + if diagnostic { DIAGNOSTIC_CPU_BYTES } else { 0 }
                    );
                    for width in [2048, 4096] {
                        // This synthetic ledger tests report plumbing; core geometry
                        // tests independently compare actual per-allocation prices.
                        let mut staged = staged_preweight(diagnostic);
                        staged.plan.full_chunk_rows = width;
                        staged.plan.ced.as_mut().unwrap().full_chunk_rows = width;
                        let report = preweight_report(&staged.plan, extent).unwrap();
                        assert_eq!(report["prompt_rows"], n);
                        assert_eq!(report["capacity"], n);
                        assert_eq!(report["full_chunk_rows"], width);
                        assert_eq!(report["caller_cpu_bytes"], caller_cpu_bytes(diagnostic));
                    }
                }
            }
        }
    }
}
