use anyhow::{Context, Result, ensure};
use clap::Args;
use objc2_metal::MTLDevice;
use qwen_llm::checkpoint_identity::{
    CheckpointIdentityCache, IdentityCacheOutcome, checkpoint_content_identity,
};
use qwen_llm::deepseek_v4_metal::{
    DeepSeekV4MemoryPlan, DeepSeekV4MetalResidency, DeepSeekV4MhcBufferRole,
    DeepSeekV4MhcCommandKind, DeepSeekV4MhcDeleteArm, DeepSeekV4MhcDeleteProfile,
    DeepSeekV4MhcEndpointEvidence, DeepSeekV4MhcExecutionKind, DeepSeekV4MhcOracle,
    DeepSeekV4MhcOracleIdentity, DeepSeekV4MhcSiteKind, DeepSeekV4MhcTimedEndpoint,
    DeepSeekV4MhcVerifiedCapture, DeepSeekV4ModelContentId, DeepSeekV4Session,
    DeepSeekV4SessionCapacity, deepseek_v4_diagnostics_metallib_sha256,
    seal_mhc_delete_oracle_pair,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::{
    DispatchCensusRow, KernelTraceCounters, MetalAllocationCensusRow, MetalContext,
    MetalPipelineCacheMetrics, allocation_census_begin, allocation_census_take,
    diagnostics_observer_active_counts, dispatch_census_begin, dispatch_census_take,
    kernel_trace_begin, kernel_trace_snapshot,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::host_validity::{HostSnapshot, VmCounters, VmDelta};

const TOKENS: usize = 2_048;
const CONTINUATION_TOKEN: u32 = 35;
const K160_TENSORS: usize = 1_328;
const K160_SOURCE_BYTES: u64 = 89_920_886_108;
const K160_EXPERTS: usize = 160;
const T_CRITICAL_ONE_SIDED_95_DF5: f64 = 2.015_048;
const GATE: f64 = 0.02;
const STATIONARITY_GATE: f64 = 0.05;
const FROZEN_AC_POWER_MODE: i64 = 2;
const PARENT_COMMIT: &str = "bbfcca84f10dc4ca095cba37608ea7804025bb7e";
const PARENT_ENDPOINT_FILE_SHA256: &str =
    "61fbbeec26a9cf026f88102674e0dd20054c1d91eb0c9bf4d0c8b813a37081bb";
#[cfg(test)]
const PARENT_ENDPOINT_ARTIFACT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/bench/2026-08-09-dsv4-k160-mhc-delete-v1/parent-endpoint.json"
));
const MHC_SITE_COUNT: usize = 86;
const DELETED_DISPATCH_COUNT: usize = MHC_SITE_COUNT * 2;
const MHC_RMS_KERNEL: &str = "kernel_rms_norm_batched_f32";
const MHC_FUNCTION_KERNEL: &str = "kernel_mat_vec_q8_0_f32_lcpp_batch";
const MHC_CONTROLS_KERNEL: &str = "kernel_deepseek_v4_hc_controls_batch";
const ARM_SEQUENCE: [Arm; 36] = [
    Arm::A,
    Arm::P,
    Arm::Z,
    Arm::Z,
    Arm::P,
    Arm::A,
    Arm::P,
    Arm::Z,
    Arm::A,
    Arm::A,
    Arm::Z,
    Arm::P,
    Arm::Z,
    Arm::A,
    Arm::P,
    Arm::P,
    Arm::A,
    Arm::Z,
    Arm::Z,
    Arm::A,
    Arm::P,
    Arm::P,
    Arm::A,
    Arm::Z,
    Arm::P,
    Arm::Z,
    Arm::A,
    Arm::A,
    Arm::Z,
    Arm::P,
    Arm::A,
    Arm::P,
    Arm::Z,
    Arm::Z,
    Arm::P,
    Arm::A,
];

#[derive(Args, Debug)]
pub struct Dsv4MhcDeleteArgs {
    /// First shard of the exact DeepSeek V4 K160 cohort.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Strong model-content identity cache.
    #[arg(long, value_name = "DIR")]
    identity_cache: PathBuf,
    /// Independently acquired bbfcca8 endpoint record.
    #[arg(long, value_name = "JSON")]
    parent_endpoint: PathBuf,
    /// Canonical packet report destination.
    #[arg(long, value_name = "JSON")]
    json_out: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Arm {
    A,
    P,
    Z,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::P => "P",
            Self::Z => "Z",
        }
    }

    fn engine(self) -> DeepSeekV4MhcDeleteArm {
        match self {
            Self::A => DeepSeekV4MhcDeleteArm::Current,
            Self::P => DeepSeekV4MhcDeleteArm::Producer,
            Self::Z => DeepSeekV4MhcDeleteArm::Zero,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ParentEndpoint {
    schema_version: u32,
    payload_sha256: String,
    payload: ParentEndpointPayload,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ParentEndpointPayload {
    report_kind: String,
    acquired_at_utc: String,
    source_commit: String,
    build: Value,
    build_identity_sha256: String,
    binary_sha256: String,
    harness_sha256: String,
    metallib_sha256: String,
    release_build: bool,
    device_name: String,
    device_registry_id: u64,
    os_version: String,
    metal_system_profile_sha256: String,
    qwen_env: BTreeMap<String, String>,
    model_content_id: String,
    token_sha256: String,
    endpoint_digest_domain: String,
    endpoint: TimedEndpoint,
}

#[derive(Clone, Debug, Serialize)]
struct SessionAllocation {
    contract: SessionContractRecord,
    realized: Vec<RealizedAllocationRecord>,
    construction_wall_ms: f64,
    before_bytes: u64,
    after_bytes: u64,
    delta_bytes: i128,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SessionContractRecord {
    sha256: String,
    memory_plan_sha256: String,
    forward_limit: usize,
    csa_physical_rows: usize,
    hca_physical_rows: usize,
    session_logical_bytes: u64,
    session_priced_upper_bytes: u64,
    session_storage_mode: &'static str,
    oracle_payload_sha256: Option<String>,
    oracle_payload_bytes: u64,
    oracle_storage_mode: Option<&'static str>,
    allocations: Vec<MemoryAllocationRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct RealizedAllocationRecord {
    requested_bytes: u64,
    buffer_length: u64,
    storage_mode: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct CommandInterval {
    submission_ordinal: usize,
    layer: usize,
    kind: &'static str,
    gpu_start_seconds: f64,
    gpu_end_seconds: f64,
    duration_ms: f64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SiteRecord {
    ordinal: usize,
    layer: usize,
    site: &'static str,
    producer_runs: bool,
    producer_output_role: &'static str,
    producer_output_offset: Option<u64>,
    controls_input_role: &'static str,
    controls_input_offset: u64,
    site_bytes: u64,
    shape: [u64; 2],
}

#[derive(Clone, Debug, Serialize)]
struct ProfileRecord {
    execution: &'static str,
    queue_identity: u64,
    wall_ms: f64,
    raw_gpu_ms: f64,
    union_gpu_ms: f64,
    outside_gpu_ms: f64,
    sites: Vec<SiteRecord>,
    command_intervals: Vec<CommandInterval>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DispatchRow {
    family: String,
    tag: Option<String>,
    encoder_ordinal: u64,
    encoder_concurrent: bool,
    kernel: String,
    grid: [u64; 3],
    threads: [u64; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct TraceRecord {
    encoders: u64,
    concurrent_encoders: u64,
    dispatches: u64,
}

#[derive(Clone, Serialize)]
struct PreflightRecord {
    arm: &'static str,
    trace: TraceRecord,
    dispatch_sha256: String,
    dispatches: Vec<DispatchRow>,
    profile: Option<ProfileRecord>,
    allocation: SessionAllocation,
}

#[derive(Clone, Serialize)]
struct CorrectnessRecord {
    arm: &'static str,
    endpoint_sha256: String,
    endpoint_position: u32,
    continuation_position: u32,
    endpoint_causal_digest: String,
    continuation_causal_digest: String,
    matches_capture: bool,
    allocation: SessionAllocation,
}

#[derive(Clone, Serialize)]
struct WarmupRecord {
    arm: &'static str,
    profile: ProfileRecord,
    allocation: SessionAllocation,
}

#[derive(Clone, Serialize)]
struct PipelineDelta {
    misses: u64,
    miss_wall_ns: u64,
    compiler_wall_ns: u64,
}

#[derive(Clone, Serialize)]
struct TimedObservation {
    ordinal: usize,
    block: usize,
    sextet: usize,
    position_in_sextet: usize,
    arm: &'static str,
    profile: ProfileRecord,
    endpoint: TimedEndpoint,
    endpoint_matches_capture: bool,
    allocation: SessionAllocation,
    pipeline_delta: PipelineDelta,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct TimedEndpoint {
    sha256: String,
    position: u32,
    token_count: usize,
    logits_count: usize,
    hidden_count: usize,
}

#[derive(Clone, Debug, Serialize)]
struct CaptureRecord {
    manifest_sha256: String,
    manifest: CaptureManifest,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    command_intervals: Vec<CommandInterval>,
    wall_ms: f64,
    raw_gpu_ms: f64,
    union_gpu_ms: f64,
    allocation: SessionAllocation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CaptureManifest {
    schema_version: u32,
    arithmetic_parent: &'static str,
    experimental_commit: String,
    build_source_state: String,
    build_identity_sha256: String,
    binary_sha256: String,
    metallib_sha256: String,
    identity: OracleIdentity,
    payload_sha256: String,
    payload_bytes: u64,
    endpoint_evidence_sha256: String,
    dtype: &'static str,
    storage_mode: &'static str,
    shape: [u64; 3],
    queue_identity: u64,
    sites: Vec<SiteRecord>,
    commands: Vec<CommandTopology>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct CommandTopology {
    submission_ordinal: usize,
    layer: usize,
    kind: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct OracleIdentity {
    model_content_id: String,
    compatibility_id: String,
    token_sha256: String,
    policy_sha256: String,
    policy_manifest: String,
    start_position: u32,
    n_tokens: usize,
    rms_epsilon_bits: u32,
    hc_epsilon_bits: u32,
    device_registry_id: u64,
    residency_tensor_count: usize,
    residency_source_bytes: u64,
    expert_count: usize,
}

#[derive(Clone, Serialize)]
struct MemoryPlanRecord {
    sha256: String,
    residency_buffer_count: usize,
    residency_logical_bytes: u64,
    residency_priced_upper_bytes: u64,
    session_logical_bytes: u64,
    session_priced_upper_bytes: u64,
    total_priced_upper_bytes: u64,
    forward_limit: usize,
    csa_physical_rows: usize,
    hca_physical_rows: usize,
    session_storage_mode: &'static str,
    allocations: Vec<MemoryAllocationRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MemoryAllocationRecord {
    name: String,
    logical_bytes: u64,
    priced_bytes: u64,
    alignment: u64,
    storage_mode: &'static str,
}

#[derive(Serialize)]
struct AcquisitionRecord {
    schema_version: u32,
    report_kind: &'static str,
    acquired_at_utc: String,
    build: Value,
    build_identity_sha256: String,
    binary_sha256: String,
    metallib_sha256: String,
    release_build: bool,
    source_before: SourceStateRecord,
    source_after: SourceStateRecord,
    device_name: String,
    device_registry_id: u64,
    os_version: String,
    metal: MetalRecord,
    power_before_timed: Value,
    power_after_timed: Value,
    qwen_env: BTreeMap<String, String>,
    model_path: String,
    model_content_id: String,
    identity_cache_outcome: String,
    identity_hashed_bytes: u64,
    tensor_count: usize,
    source_bytes: u64,
    expert_count: usize,
    token_ids: Vec<u32>,
    token_sha256: String,
    continuation_token: u32,
    load_ms: f64,
    memory_plan: MemoryPlanRecord,
    parent_endpoint_file_sha256: &'static str,
    parent_endpoint: ParentEndpoint,
    parent_endpoint_match: bool,
    captures: Vec<CaptureRecord>,
    oracle_pre_sha256: String,
    oracle_post_sha256: String,
    ordinary_preflight: PreflightRecord,
    preflights: Vec<PreflightRecord>,
    correctness: Vec<CorrectnessRecord>,
    warmups: Vec<WarmupRecord>,
    observer_counts_before_timed: [usize; 3],
    observer_counts_after_timed: [usize; 3],
    observations: Vec<TimedObservation>,
    host_before: Value,
    host_after: Value,
    vm_before: Value,
    vm_after: Value,
    vm_delta: Value,
    host_before_valid: bool,
    host_after_valid: bool,
    vm_valid: bool,
}

#[derive(Clone, Serialize)]
struct MetalRecord {
    max_buffer_length: usize,
    recommended_max_working_set_bytes: u64,
    initial_allocated_bytes: u64,
    process_limit_remaining_bytes: Option<u64>,
    system_profile: Value,
    system_profile_sha256: String,
}

#[derive(Clone, Serialize)]
struct SourceStateRecord {
    commit: String,
    dirty: bool,
    source_state: String,
}

struct CampaignIdentity {
    experimental_commit: String,
    build_source_state: String,
    build_identity_sha256: String,
    binary_sha256: String,
    metallib_sha256: String,
}

#[derive(Serialize)]
struct Bound {
    mean: f64,
    sd: f64,
    se: f64,
    lower95: f64,
    upper95: f64,
    zero_variance: bool,
}

#[derive(Serialize)]
struct SextetEffect {
    sextet: usize,
    block: usize,
    a_wall_ms: f64,
    p_wall_ms: f64,
    z_wall_ms: f64,
    q_wall: f64,
    s_wall: f64,
    r_wall: f64,
    kill_wall: f64,
    clear_wall: f64,
    q_gpu: f64,
    s_gpu: f64,
    kill_gpu: f64,
    clear_gpu: f64,
}

#[derive(Serialize)]
struct Decision {
    verdict: &'static str,
    authority: &'static str,
    validity_failures: Vec<String>,
    stationarity: BTreeMap<&'static str, f64>,
    sextets: Vec<SextetEffect>,
    kill_wall: Bound,
    clear_wall: Bound,
    kill_gpu: Bound,
    clear_gpu: Bound,
    block_medians: BTreeMap<&'static str, [f64; 2]>,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    acquisition_sha256: String,
    acquisition: AcquisitionRecord,
    decision: Decision,
}

#[derive(Serialize)]
struct AcquisitionHoldReport<'a> {
    schema_version: u32,
    report_kind: &'static str,
    verdict: &'static str,
    authority: &'static str,
    acquisition_sha256: &'a str,
    acquisition: &'a AcquisitionRecord,
}

#[derive(Serialize)]
struct FailureReport {
    schema_version: u32,
    report_kind: &'static str,
    acquired_at_utc: String,
    verdict: &'static str,
    authority: &'static str,
    error: String,
    build: Value,
    release_build: bool,
    qwen_env: BTreeMap<String, String>,
    model_path: String,
    identity_cache_path: String,
    parent_endpoint_path: String,
}

#[derive(Serialize)]
struct ReservationReport {
    schema_version: u32,
    report_kind: &'static str,
    reserved_at_utc: String,
    verdict: &'static str,
    authority: &'static str,
    pid: u32,
}

struct ReportSink {
    file: Option<File>,
    temporary: PathBuf,
    target: PathBuf,
    parent: PathBuf,
}

impl ReportSink {
    fn reserve(path: &Path) -> Result<Self> {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        ensure!(
            parent.is_dir(),
            "report parent {} is not a directory",
            parent.display()
        );
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .context("report path has no UTF-8 file name")?;
        let reservation = ReservationReport {
            schema_version: 1,
            report_kind: "dsv4_k160_mhc_delete_reservation",
            reserved_at_utc: super::utc_iso8601_now(),
            verdict: "HOLD",
            authority: "none",
            pid: std::process::id(),
        };
        let mut target = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .with_context(|| format!("reserve canonical report target {}", path.display()))?;
        serde_json::to_writer_pretty(&mut target, &reservation)
            .context("serialize mHC report reservation")?;
        target
            .write_all(b"\n")
            .context("terminate mHC report reservation")?;
        target.sync_all().context("sync mHC report reservation")?;
        File::open(parent)
            .with_context(|| format!("open report directory {}", parent.display()))?
            .sync_all()
            .context("sync report directory after reservation")?;
        drop(target);
        let temporary = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .with_context(|| format!("reserve report staging file {}", temporary.display()))?;
        Ok(Self {
            file: Some(file),
            temporary,
            target: path.to_path_buf(),
            parent: parent.to_path_buf(),
        })
    }

    fn checkpoint(&mut self, report: &impl Serialize) -> Result<()> {
        let mut file = self
            .file
            .take()
            .context("report sink was already consumed")?;
        serde_json::to_writer_pretty(&mut file, report)
            .context("serialize mHC acquisition checkpoint")?;
        file.write_all(b"\n")
            .context("terminate mHC acquisition checkpoint")?;
        file.sync_all().context("sync mHC acquisition checkpoint")?;
        drop(file);
        std::fs::rename(&self.temporary, &self.target).with_context(|| {
            format!(
                "publish mHC acquisition checkpoint {} -> {}",
                self.temporary.display(),
                self.target.display()
            )
        })?;
        File::open(&self.parent)
            .with_context(|| format!("open report directory {}", self.parent.display()))?
            .sync_all()
            .context("sync report directory after acquisition checkpoint")?;
        self.file = Some(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&self.temporary)
                .with_context(|| {
                    format!(
                        "rearm report staging file after acquisition {}",
                        self.temporary.display()
                    )
                })?,
        );
        Ok(())
    }

    fn publish(mut self, report: &impl Serialize) -> Result<()> {
        let mut file = self
            .file
            .take()
            .context("report sink was already consumed")?;
        serde_json::to_writer_pretty(&mut file, report).context("serialize mHC report")?;
        file.write_all(b"\n").context("terminate mHC report")?;
        file.sync_all().context("sync mHC report staging file")?;
        drop(file);
        std::fs::rename(&self.temporary, &self.target).with_context(|| {
            format!(
                "publish mHC report {} -> {}",
                self.temporary.display(),
                self.target.display()
            )
        })?;
        File::open(&self.parent)
            .with_context(|| format!("open report directory {}", self.parent.display()))?
            .sync_all()
            .context("sync report directory after publication")?;
        Ok(())
    }
}

impl Drop for ReportSink {
    fn drop(&mut self) {
        if self.file.take().is_some() || self.temporary.exists() {
            let _ = std::fs::remove_file(&self.temporary);
        }
    }
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn parse_hex_32(value: &str) -> Result<[u8; 32]> {
    ensure!(value.len() == 64, "expected 64 hex characters");
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .with_context(|| format!("invalid hex byte {index}"))?;
    }
    Ok(out)
}

fn os_version() -> Result<String> {
    let output = std::process::Command::new("/usr/bin/sw_vers")
        .arg("-productVersion")
        .output()
        .context("run sw_vers for mHC campaign")?;
    ensure!(output.status.success(), "sw_vers failed");
    let version = String::from_utf8(output.stdout).context("decode sw_vers output")?;
    let version = version.trim().to_string();
    ensure!(!version.is_empty(), "sw_vers returned an empty version");
    Ok(version)
}

fn capture_metal_system_profile() -> Result<(Value, String)> {
    let output = std::process::Command::new("/usr/sbin/system_profiler")
        .args(["SPDisplaysDataType", "-json", "-detailLevel", "mini"])
        .output()
        .context("capture Metal system profile")?;
    ensure!(output.status.success(), "system_profiler failed");
    let profile: Value =
        serde_json::from_slice(&output.stdout).context("parse Metal system profile")?;
    let encoded = serde_json::to_vec(&profile)?;
    ensure!(
        String::from_utf8_lossy(&encoded)
            .to_ascii_lowercase()
            .contains("metal"),
        "system profile lacks Metal capability evidence"
    );
    let digest = hex(Sha256::digest(&encoded));
    Ok((profile, digest))
}

fn power_evidence_is_complete(power: &Value) -> bool {
    power.get("source").and_then(Value::as_str) == Some("AC Power")
        && power.get("powermode_ac").and_then(Value::as_i64) == Some(FROZEN_AC_POWER_MODE)
        && power
            .get("thermal_warning_recorded")
            .and_then(Value::as_bool)
            == Some(false)
        && power
            .get("performance_warning_recorded")
            .and_then(Value::as_bool)
            == Some(false)
}

fn metal_evidence_is_complete(metal: &MetalRecord) -> bool {
    serde_json::to_vec(&metal.system_profile)
        .ok()
        .is_some_and(|encoded| {
            metal.system_profile_sha256 == hex(Sha256::digest(&encoded))
                && String::from_utf8_lossy(&encoded)
                    .to_ascii_lowercase()
                    .contains("metal")
        })
}

fn token_ids() -> Vec<u32> {
    (0..TOKENS)
        .map(|index| [35, 201, 200, 34][index % 4])
        .collect()
}

fn oracle_identity(identity: &DeepSeekV4MhcOracleIdentity) -> OracleIdentity {
    OracleIdentity {
        model_content_id: hex(identity.model_content_id),
        compatibility_id: hex(identity.compatibility_id),
        token_sha256: hex(identity.token_sha256),
        policy_sha256: hex(identity.policy_sha256),
        policy_manifest: identity.policy_manifest.clone(),
        start_position: identity.start_position,
        n_tokens: identity.n_tokens,
        rms_epsilon_bits: identity.rms_epsilon_bits,
        hc_epsilon_bits: identity.hc_epsilon_bits,
        device_registry_id: identity.device_registry_id,
        residency_tensor_count: identity.residency_tensor_count,
        residency_source_bytes: identity.residency_source_bytes,
        expert_count: identity.expert_count,
    }
}

fn command_kind(kind: DeepSeekV4MhcCommandKind) -> &'static str {
    match kind {
        DeepSeekV4MhcCommandKind::PreExpert => "pre_expert",
        DeepSeekV4MhcCommandKind::Expert => "expert",
        DeepSeekV4MhcCommandKind::SharedOverlap => "shared_overlap",
        DeepSeekV4MhcCommandKind::MergedPreExpert => "merged_pre_expert",
    }
}

fn execution_kind(kind: DeepSeekV4MhcExecutionKind) -> &'static str {
    match kind {
        DeepSeekV4MhcExecutionKind::Capture => "C",
        DeepSeekV4MhcExecutionKind::Current => "A",
        DeepSeekV4MhcExecutionKind::Producer => "P",
        DeepSeekV4MhcExecutionKind::Zero => "Z",
    }
}

fn site_kind(kind: DeepSeekV4MhcSiteKind) -> &'static str {
    match kind {
        DeepSeekV4MhcSiteKind::Attention => "attention",
        DeepSeekV4MhcSiteKind::Ffn => "ffn",
    }
}

fn buffer_role(role: DeepSeekV4MhcBufferRole) -> &'static str {
    match role {
        DeepSeekV4MhcBufferRole::None => "none",
        DeepSeekV4MhcBufferRole::OrdinaryMixes => "ordinary_mixes",
        DeepSeekV4MhcBufferRole::Oracle => "oracle",
    }
}

fn profile_record(profile: DeepSeekV4MhcDeleteProfile) -> ProfileRecord {
    let raw_gpu_ms = profile.raw_gpu_ms();
    let union_gpu_ms = profile.union_gpu_ms();
    ProfileRecord {
        execution: execution_kind(profile.execution),
        queue_identity: profile.queue_identity,
        wall_ms: profile.wall_ms,
        raw_gpu_ms,
        union_gpu_ms,
        outside_gpu_ms: profile.wall_ms - union_gpu_ms,
        sites: profile
            .sites
            .into_iter()
            .map(|site| SiteRecord {
                ordinal: site.ordinal,
                layer: site.layer,
                site: site_kind(site.site),
                producer_runs: site.producer_runs,
                producer_output_role: buffer_role(site.producer_output_role),
                producer_output_offset: site.producer_output_offset,
                controls_input_role: buffer_role(site.controls_input_role),
                controls_input_offset: site.controls_input_offset,
                site_bytes: site.site_bytes,
                shape: site.shape,
            })
            .collect(),
        command_intervals: profile
            .command_intervals
            .into_iter()
            .map(|interval| CommandInterval {
                submission_ordinal: interval.submission_ordinal,
                layer: interval.layer,
                kind: command_kind(interval.kind),
                gpu_start_seconds: interval.gpu_start_seconds,
                gpu_end_seconds: interval.gpu_end_seconds,
                duration_ms: interval.duration_ms(),
            })
            .collect(),
    }
}

fn timed_endpoint(endpoint: DeepSeekV4MhcTimedEndpoint) -> TimedEndpoint {
    TimedEndpoint {
        sha256: hex(endpoint.sha256),
        position: endpoint.position,
        token_count: endpoint.token_count,
        logits_count: endpoint.logits_count,
        hidden_count: endpoint.hidden_count,
    }
}

fn pipeline_delta(
    current: MetalPipelineCacheMetrics,
    previous: MetalPipelineCacheMetrics,
) -> PipelineDelta {
    let delta = current.saturating_delta_since(previous);
    PipelineDelta {
        misses: delta.misses,
        miss_wall_ns: delta.miss_wall_ns,
        compiler_wall_ns: delta.compiler_wall_ns,
    }
}

fn dispatch_rows(rows: &[DispatchCensusRow]) -> Vec<DispatchRow> {
    rows.iter()
        .map(|row| DispatchRow {
            family: row.family.to_string(),
            tag: row.tag.clone(),
            encoder_ordinal: row.encoder_ordinal,
            encoder_concurrent: row.encoder_concurrent,
            kernel: row.kernel.clone(),
            grid: [row.grid_width, row.grid_height, row.grid_depth],
            threads: [row.threads_width, row.threads_height, row.threads_depth],
        })
        .collect()
}

fn dispatch_digest(rows: &[DispatchCensusRow]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"qwen.dsv4.mhc-delete-dispatch.v1\0");
    for row in rows {
        digest.update((row.family.len() as u64).to_le_bytes());
        digest.update(row.family.as_bytes());
        if let Some(tag) = &row.tag {
            digest.update([1]);
            digest.update((tag.len() as u64).to_le_bytes());
            digest.update(tag.as_bytes());
        } else {
            digest.update([0]);
        }
        digest.update(row.encoder_ordinal.to_le_bytes());
        digest.update([u8::from(row.encoder_concurrent)]);
        digest.update((row.kernel.len() as u64).to_le_bytes());
        digest.update(row.kernel.as_bytes());
        for value in [
            row.grid_width,
            row.grid_height,
            row.grid_depth,
            row.threads_width,
            row.threads_height,
            row.threads_depth,
        ] {
            digest.update(value.to_le_bytes());
        }
    }
    digest.finalize().into()
}

fn trace_record(trace: KernelTraceCounters) -> TraceRecord {
    TraceRecord {
        encoders: trace.encoders,
        concurrent_encoders: trace.concurrent_encoders,
        dispatches: trace.dispatches,
    }
}

fn memory_plan_record(
    plan: &DeepSeekV4MemoryPlan,
    capacity: DeepSeekV4SessionCapacity,
) -> MemoryPlanRecord {
    let allocations = plan
        .session_allocations()
        .iter()
        .map(|allocation| MemoryAllocationRecord {
            name: allocation.name.clone(),
            logical_bytes: allocation.logical_bytes,
            priced_bytes: allocation.priced_bytes,
            alignment: allocation.alignment,
            storage_mode: "shared",
        })
        .collect::<Vec<_>>();
    let mut digest = Sha256::new();
    digest.update(b"qwen.dsv4.mhc-delete-memory-plan.v1\0");
    for value in [
        plan.residency_buffer_count() as u64,
        plan.residency_logical_bytes(),
        plan.residency_priced_upper_bytes(),
        plan.session_logical_bytes(),
        plan.session_priced_upper_bytes(),
        plan.total_priced_upper_bytes(),
        capacity.forward_limit() as u64,
        capacity.csa_physical_rows() as u64,
        capacity.hca_physical_rows() as u64,
    ] {
        digest.update(value.to_le_bytes());
    }
    digest.update(b"shared\0");
    for allocation in plan.session_allocations() {
        digest.update((allocation.name.len() as u64).to_le_bytes());
        digest.update(allocation.name.as_bytes());
        digest.update(allocation.logical_bytes.to_le_bytes());
        digest.update(allocation.priced_bytes.to_le_bytes());
        digest.update(allocation.alignment.to_le_bytes());
    }
    MemoryPlanRecord {
        sha256: hex(digest.finalize()),
        residency_buffer_count: plan.residency_buffer_count(),
        residency_logical_bytes: plan.residency_logical_bytes(),
        residency_priced_upper_bytes: plan.residency_priced_upper_bytes(),
        session_logical_bytes: plan.session_logical_bytes(),
        session_priced_upper_bytes: plan.session_priced_upper_bytes(),
        total_priced_upper_bytes: plan.total_priced_upper_bytes(),
        forward_limit: capacity.forward_limit(),
        csa_physical_rows: capacity.csa_physical_rows(),
        hca_physical_rows: capacity.hca_physical_rows(),
        session_storage_mode: "shared",
        allocations,
    }
}

fn session_contract(
    memory: &MemoryPlanRecord,
    oracle: Option<&DeepSeekV4MhcOracle>,
) -> Result<SessionContractRecord> {
    let mut record = SessionContractRecord {
        sha256: String::new(),
        memory_plan_sha256: memory.sha256.clone(),
        forward_limit: memory.forward_limit,
        csa_physical_rows: memory.csa_physical_rows,
        hca_physical_rows: memory.hca_physical_rows,
        session_logical_bytes: memory.session_logical_bytes,
        session_priced_upper_bytes: memory.session_priced_upper_bytes,
        session_storage_mode: memory.session_storage_mode,
        oracle_payload_sha256: oracle.map(|oracle| hex(oracle.sealed_sha256())),
        oracle_payload_bytes: oracle.map_or(0, DeepSeekV4MhcOracle::payload_bytes),
        oracle_storage_mode: oracle.map(|_| "shared"),
        allocations: memory.allocations.clone(),
    };
    record.sha256 = hex(Sha256::digest(serde_json::to_vec(&record)?));
    Ok(record)
}

fn session_contract_digest_is_valid(contract: &SessionContractRecord) -> bool {
    let mut unhashed = contract.clone();
    unhashed.sha256.clear();
    serde_json::to_vec(&unhashed)
        .ok()
        .is_some_and(|bytes| contract.sha256 == hex(Sha256::digest(bytes)))
}

fn identity_cache_outcome(outcome: IdentityCacheOutcome) -> &'static str {
    match outcome {
        IdentityCacheOutcome::Hit => "hit",
        IdentityCacheOutcome::ComputedAndStored => "computed_and_stored",
        IdentityCacheOutcome::ComputedAndRepaired => "computed_and_repaired",
        IdentityCacheOutcome::ComputedUncached => "computed_uncached",
    }
}

fn build_packet_is_canonical(build: &Value) -> bool {
    build.get("status").and_then(Value::as_str) == Some("match")
        && build
            .get("problems")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
        && build
            .get("overrides")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
        && build.get("build_dirty").and_then(Value::as_bool) == Some(false)
        && build.get("runtime_dirty").and_then(Value::as_bool) == Some(false)
}

fn capture_source_state() -> Result<SourceStateRecord> {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    Ok(SourceStateRecord {
        commit: super::source_identity::git_text(repo, &["rev-parse", "HEAD"])
            .context("resolve live mHC source commit")?,
        dirty: super::source_identity::git_dirty(repo)
            .context("inspect live mHC worktree state")?,
        source_state: super::source_identity::tracked_source_state(repo)
            .context("hash live mHC source state")?,
    })
}

fn source_state_matches_build(source: &SourceStateRecord, build: &Value) -> bool {
    !source.dirty
        && build.get("build_commit").and_then(Value::as_str) == Some(source.commit.as_str())
        && build.get("runtime_commit").and_then(Value::as_str) == Some(source.commit.as_str())
        && build.get("build_source_state").and_then(Value::as_str)
            == Some(source.source_state.as_str())
        && build.get("runtime_source_state").and_then(Value::as_str)
            == Some(source.source_state.as_str())
}

fn validate_campaign_build(build: &Value) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "canonical mHC acquisition requires a release binary"
    );
    ensure!(
        build_packet_is_canonical(build),
        "canonical mHC acquisition requires a clean matching build without overrides"
    );
    ensure!(
        build
            .get("build_commit")
            .and_then(Value::as_str)
            .is_some_and(|commit| commit.len() == 40),
        "canonical mHC acquisition requires a full source commit"
    );
    Ok(())
}

fn file_sha256(path: &Path) -> Result<[u8; 32]> {
    let bytes =
        std::fs::read(path).with_context(|| format!("read {} for SHA-256", path.display()))?;
    Ok(Sha256::digest(bytes).into())
}

fn campaign_identity(build: &Value) -> Result<CampaignIdentity> {
    let executable = std::env::current_exe().context("resolve mHC campaign executable")?;
    Ok(CampaignIdentity {
        experimental_commit: build
            .get("build_commit")
            .and_then(Value::as_str)
            .context("build identity lacks experimental commit")?
            .to_string(),
        build_source_state: build
            .get("build_source_state")
            .and_then(Value::as_str)
            .context("build identity lacks source-state digest")?
            .to_string(),
        build_identity_sha256: hex(Sha256::digest(serde_json::to_vec(build)?)),
        binary_sha256: hex(file_sha256(&executable)?),
        metallib_sha256: hex(deepseek_v4_diagnostics_metallib_sha256()),
    })
}

fn new_session(
    ctx: &MetalContext,
    residency: &mut Option<DeepSeekV4MetalResidency>,
    model_content_id: DeepSeekV4ModelContentId,
    contract: &SessionContractRecord,
) -> Result<(DeepSeekV4Session, SessionAllocation)> {
    let weights = residency
        .take()
        .context("mHC campaign lost immutable model residency")?;
    let before_bytes = ctx.current_allocated_size();
    let started = Instant::now();
    allocation_census_begin();
    let session = DeepSeekV4Session::new_with_model_content_id(ctx, weights, model_content_id);
    let realized = allocation_census_take()
        .into_iter()
        .map(realized_allocation_record)
        .collect::<Vec<_>>();
    let session = session.context("construct fresh mHC campaign session")?;
    let capacity = session.capacity();
    ensure!(
        capacity.forward_limit() == contract.forward_limit
            && capacity.csa_physical_rows() == contract.csa_physical_rows
            && capacity.hca_physical_rows() == contract.hca_physical_rows,
        "fresh session capacity differs from the frozen allocation contract"
    );
    ensure!(
        realized.len() == contract.allocations.len()
            && realized
                .iter()
                .zip(&contract.allocations)
                .all(|(realized, planned)| {
                    realized.requested_bytes == planned.logical_bytes.max(1)
                        && realized.buffer_length == planned.logical_bytes.max(1)
                        && realized.storage_mode == planned.storage_mode
                }),
        "fresh session realized allocation census differs from the frozen plan"
    );
    let construction_wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let after_bytes = ctx.current_allocated_size();
    Ok((
        session,
        SessionAllocation {
            contract: contract.clone(),
            realized,
            construction_wall_ms,
            before_bytes,
            after_bytes,
            delta_bytes: i128::from(after_bytes) - i128::from(before_bytes),
        },
    ))
}

fn realized_allocation_record(row: MetalAllocationCensusRow) -> RealizedAllocationRecord {
    RealizedAllocationRecord {
        requested_bytes: row.requested_bytes,
        buffer_length: row.buffer_length,
        storage_mode: row.storage_mode,
    }
}

fn allocation_identity_is_valid(allocation: &SessionAllocation) -> bool {
    allocation.construction_wall_ms.is_finite()
        && allocation.construction_wall_ms > 0.0
        && session_contract_digest_is_valid(&allocation.contract)
}

fn allocation_plan_matches(expected: &SessionAllocation, observed: &SessionAllocation) -> bool {
    allocation_identity_is_valid(observed)
        && observed.contract == expected.contract
        && observed.realized == expected.realized
}

fn allocation_identity_matches(expected: &SessionAllocation, observed: &SessionAllocation) -> bool {
    allocation_plan_matches(expected, observed)
        && observed.before_bytes == expected.before_bytes
        && observed.after_bytes == expected.after_bytes
        && observed.delta_bytes == expected.delta_bytes
}

fn return_residency(residency: &mut Option<DeepSeekV4MetalResidency>, session: DeepSeekV4Session) {
    debug_assert!(residency.is_none());
    *residency = Some(session.into_residency());
}

fn capture_record(
    capture: &DeepSeekV4MhcVerifiedCapture,
    profile: DeepSeekV4MhcDeleteProfile,
    allocation: SessionAllocation,
    campaign: &CampaignIdentity,
) -> Result<CaptureRecord> {
    let profile = profile_record(profile);
    let commands = command_topology(&profile);
    let manifest = CaptureManifest {
        schema_version: 1,
        arithmetic_parent: PARENT_COMMIT,
        experimental_commit: campaign.experimental_commit.clone(),
        build_source_state: campaign.build_source_state.clone(),
        build_identity_sha256: campaign.build_identity_sha256.clone(),
        binary_sha256: campaign.binary_sha256.clone(),
        metallib_sha256: campaign.metallib_sha256.clone(),
        identity: oracle_identity(capture.identity()),
        payload_sha256: hex(capture.payload_sha256()),
        payload_bytes: capture.payload_bytes(),
        endpoint_evidence_sha256: hex(capture.evidence().sha256()),
        dtype: "f32",
        storage_mode: "shared",
        shape: [24, TOKENS as u64, MHC_SITE_COUNT as u64],
        queue_identity: profile.queue_identity,
        sites: profile.sites,
        commands,
    };
    let manifest_sha256 = hex(Sha256::digest(serde_json::to_vec(&manifest)?));
    Ok(CaptureRecord {
        manifest_sha256,
        manifest,
        command_intervals: profile.command_intervals,
        wall_ms: profile.wall_ms,
        raw_gpu_ms: profile.raw_gpu_ms,
        union_gpu_ms: profile.union_gpu_ms,
        allocation,
    })
}

fn command_topology(profile: &ProfileRecord) -> Vec<CommandTopology> {
    profile
        .command_intervals
        .iter()
        .map(|command| CommandTopology {
            submission_ordinal: command.submission_ordinal,
            layer: command.layer,
            kind: command.kind,
        })
        .collect()
}

fn acquire_capture(
    ctx: &MetalContext,
    residency: &mut Option<DeepSeekV4MetalResidency>,
    model_content_id: DeepSeekV4ModelContentId,
    tokens: &[u32],
    campaign: &CampaignIdentity,
    contract: &SessionContractRecord,
) -> Result<(DeepSeekV4MhcVerifiedCapture, CaptureRecord)> {
    let (mut session, allocation) = new_session(ctx, residency, model_content_id, contract)?;
    let result = session.capture_mhc_delete_oracle(ctx, tokens, CONTINUATION_TOKEN);
    return_residency(residency, session);
    let (capture, profile) = result.context("capture and verify mHC oracle")?;
    let record = capture_record(&capture, profile, allocation, campaign)?;
    Ok((capture, record))
}

fn run_ordinary_preflight(
    ctx: &MetalContext,
    residency: &mut Option<DeepSeekV4MetalResidency>,
    model_content_id: DeepSeekV4ModelContentId,
    tokens: &[u32],
    contract: &SessionContractRecord,
) -> Result<PreflightRecord> {
    let (mut session, allocation) = new_session(ctx, residency, model_content_id, contract)?;
    dispatch_census_begin();
    let trace_guard = kernel_trace_begin();
    let result = session.prefill_tokens(ctx, tokens).map(|_| ());
    let trace = kernel_trace_snapshot();
    drop(trace_guard);
    let rows = dispatch_census_take();
    return_residency(residency, session);
    result.context("execute ordinary mHC collector-off preflight")?;
    Ok(PreflightRecord {
        arm: "ordinary",
        trace: trace_record(trace),
        dispatch_sha256: hex(dispatch_digest(&rows)),
        dispatches: dispatch_rows(&rows),
        profile: None,
        allocation,
    })
}

fn run_capture_preflight(
    ctx: &MetalContext,
    residency: &mut Option<DeepSeekV4MetalResidency>,
    model_content_id: DeepSeekV4ModelContentId,
    tokens: &[u32],
    contract: &SessionContractRecord,
) -> Result<PreflightRecord> {
    let (mut session, allocation) = new_session(ctx, residency, model_content_id, contract)?;
    dispatch_census_begin();
    let trace_guard = kernel_trace_begin();
    let result = session.preflight_mhc_delete_capture(ctx, tokens);
    let trace = kernel_trace_snapshot();
    drop(trace_guard);
    let rows = dispatch_census_take();
    return_residency(residency, session);
    let profile = result.context("execute structural C preflight")?;
    Ok(PreflightRecord {
        arm: "C",
        trace: trace_record(trace),
        dispatch_sha256: hex(dispatch_digest(&rows)),
        dispatches: dispatch_rows(&rows),
        profile: Some(profile_record(profile)),
        allocation,
    })
}

fn run_arm_preflight(
    ctx: &MetalContext,
    residency: &mut Option<DeepSeekV4MetalResidency>,
    model_content_id: DeepSeekV4ModelContentId,
    tokens: &[u32],
    arm: Arm,
    oracle: &DeepSeekV4MhcOracle,
    contract: &SessionContractRecord,
) -> Result<PreflightRecord> {
    let (mut session, allocation) = new_session(ctx, residency, model_content_id, contract)?;
    dispatch_census_begin();
    let trace_guard = kernel_trace_begin();
    let result = session.preflight_mhc_delete_arm(ctx, tokens, arm.engine(), oracle);
    let trace = kernel_trace_snapshot();
    drop(trace_guard);
    let rows = dispatch_census_take();
    return_residency(residency, session);
    let profile =
        result.with_context(|| format!("execute structural {} preflight", arm.label()))?;
    Ok(PreflightRecord {
        arm: arm.label(),
        trace: trace_record(trace),
        dispatch_sha256: hex(dispatch_digest(&rows)),
        dispatches: dispatch_rows(&rows),
        profile: Some(profile_record(profile)),
        allocation,
    })
}

fn run_correctness_arm(
    ctx: &MetalContext,
    residency: &mut Option<DeepSeekV4MetalResidency>,
    model_content_id: DeepSeekV4ModelContentId,
    tokens: &[u32],
    arm: Arm,
    oracle: &DeepSeekV4MhcOracle,
    reference: &DeepSeekV4MhcEndpointEvidence,
    contract: &SessionContractRecord,
) -> Result<CorrectnessRecord> {
    let (mut session, allocation) = new_session(ctx, residency, model_content_id, contract)?;
    let execution = session.execute_mhc_delete_arm(ctx, tokens, arm.engine(), oracle);
    let evidence = match execution {
        Ok(_) => session.capture_mhc_delete_endpoint_evidence(ctx, CONTINUATION_TOKEN),
        Err(error) => Err(error),
    };
    return_residency(residency, session);
    let evidence = evidence.with_context(|| format!("capture {} correctness", arm.label()))?;
    let matches_capture = evidence == *reference;
    Ok(CorrectnessRecord {
        arm: arm.label(),
        endpoint_sha256: hex(evidence.sha256()),
        endpoint_position: evidence.endpoint_position(),
        continuation_position: evidence.continuation_position(),
        endpoint_causal_digest: hex(evidence.endpoint_causal_digest()),
        continuation_causal_digest: hex(evidence.continuation_causal_digest()),
        matches_capture,
        allocation,
    })
}

fn run_warmup_arm(
    ctx: &MetalContext,
    residency: &mut Option<DeepSeekV4MetalResidency>,
    model_content_id: DeepSeekV4ModelContentId,
    tokens: &[u32],
    arm: Arm,
    oracle: &DeepSeekV4MhcOracle,
    contract: &SessionContractRecord,
) -> Result<WarmupRecord> {
    let (mut session, allocation) = new_session(ctx, residency, model_content_id, contract)?;
    let result = session.execute_mhc_delete_arm(ctx, tokens, arm.engine(), oracle);
    return_residency(residency, session);
    let profile = result
        .map(profile_record)
        .with_context(|| format!("execute {} pipeline warmup", arm.label()))?;
    Ok(WarmupRecord {
        arm: arm.label(),
        profile,
        allocation,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_timed_arm(
    ctx: &MetalContext,
    residency: &mut Option<DeepSeekV4MetalResidency>,
    model_content_id: DeepSeekV4ModelContentId,
    tokens: &[u32],
    arm: Arm,
    oracle: &DeepSeekV4MhcOracle,
    expected_endpoint: DeepSeekV4MhcTimedEndpoint,
    ordinal: usize,
    contract: &SessionContractRecord,
) -> Result<TimedObservation> {
    let (mut session, allocation) = new_session(ctx, residency, model_content_id, contract)?;
    let pipeline_before = ctx.pipeline_cache_metrics();
    let profile = session.execute_mhc_delete_arm(ctx, tokens, arm.engine(), oracle);
    let pipeline_after = ctx.pipeline_cache_metrics();
    let endpoint = match profile {
        Ok(profile) => session
            .capture_mhc_delete_timed_endpoint()
            .map(|endpoint| (profile, endpoint)),
        Err(error) => Err(error),
    };
    return_residency(residency, session);
    let (profile, endpoint) =
        endpoint.with_context(|| format!("execute timed {} arm {ordinal}", arm.label()))?;
    Ok(TimedObservation {
        ordinal,
        block: ordinal / 18 + 1,
        sextet: ordinal / 6 + 1,
        position_in_sextet: ordinal % 6,
        arm: arm.label(),
        profile: profile_record(profile),
        endpoint: timed_endpoint(endpoint),
        endpoint_matches_capture: endpoint == expected_endpoint,
        allocation,
        pipeline_delta: pipeline_delta(pipeline_after, pipeline_before),
    })
}

fn command_signature(profile: &ProfileRecord) -> Vec<(usize, &'static str)> {
    profile
        .command_intervals
        .iter()
        .map(|command| (command.layer, command.kind))
        .collect()
}

fn expected_site(index: usize) -> (usize, &'static str) {
    (
        index / 2,
        if index.is_multiple_of(2) {
            "attention"
        } else {
            "ffn"
        },
    )
}

fn validate_site_ledger(profile: &ProfileRecord) -> Result<()> {
    validate_site_records(profile.execution, &profile.sites)
}

fn validate_site_records(execution: &str, sites: &[SiteRecord]) -> Result<()> {
    ensure!(
        sites.len() == MHC_SITE_COUNT,
        "{} preflight recorded {} mHC sites, expected {MHC_SITE_COUNT}",
        execution,
        sites.len()
    );
    let site_bytes = u64::try_from(24 * TOKENS * std::mem::size_of::<f32>())
        .context("mHC site byte count exceeds u64")?;
    for (index, site) in sites.iter().enumerate() {
        let (layer, kind) = expected_site(index);
        ensure!(
            site.ordinal == index && site.layer == layer && site.site == kind,
            "{} site {index} has ordinal/layer/kind {}/{}/{}, expected {index}/{layer}/{kind}",
            execution,
            site.ordinal,
            site.layer,
            site.site
        );
        ensure!(
            site.shape == [24, TOKENS as u64] && site.site_bytes == site_bytes,
            "{} site {index} has shape/bytes {:?}/{}, expected [24,{TOKENS}]/{site_bytes}",
            execution,
            site.shape,
            site.site_bytes
        );
        let oracle_offset = u64::try_from(index)
            .context("mHC site index exceeds u64")?
            .checked_mul(site_bytes)
            .context("mHC oracle offset overflow")?;
        match execution {
            "C" => ensure!(
                site.producer_runs
                    && site.producer_output_role == "oracle"
                    && site.producer_output_offset == Some(oracle_offset)
                    && site.controls_input_role == "oracle"
                    && site.controls_input_offset == oracle_offset,
                "capture site {index} does not bind its exact oracle slice"
            ),
            "A" => ensure!(
                site.producer_runs
                    && site.producer_output_role == "ordinary_mixes"
                    && site.producer_output_offset.is_some()
                    && site.controls_input_role == "ordinary_mixes"
                    && site.controls_input_offset == site.producer_output_offset.unwrap(),
                "A site {index} does not retain the ordinary producer/consumer path"
            ),
            "P" => ensure!(
                site.producer_runs
                    && site.producer_output_role == "ordinary_mixes"
                    && site.producer_output_offset.is_some()
                    && site.controls_input_role == "oracle"
                    && site.controls_input_offset == oracle_offset,
                "P site {index} does not isolate producer charge from oracle consumption"
            ),
            "Z" => ensure!(
                !site.producer_runs
                    && site.producer_output_role == "none"
                    && site.producer_output_offset.is_none()
                    && site.controls_input_role == "oracle"
                    && site.controls_input_offset == oracle_offset,
                "Z site {index} does not delete the producer and consume the oracle"
            ),
            execution => anyhow::bail!("unknown mHC site-ledger execution {execution}"),
        }
    }
    Ok(())
}

fn removed_dispatches<'a>(
    current: &'a [DispatchRow],
    zero: &[DispatchRow],
) -> Result<Vec<&'a DispatchRow>> {
    let mut zero_index = 0;
    let mut removed = Vec::new();
    for row in current {
        if zero.get(zero_index) == Some(row) {
            zero_index += 1;
        } else {
            removed.push(row);
        }
    }
    ensure!(
        zero_index == zero.len(),
        "Z dispatch rows are not an ordered subsequence of A"
    );
    Ok(removed)
}

fn profile_for<'a>(preflights: &'a [PreflightRecord], arm: &str) -> Result<&'a ProfileRecord> {
    preflights
        .iter()
        .find(|preflight| preflight.arm == arm)
        .and_then(|preflight| preflight.profile.as_ref())
        .with_context(|| format!("missing {arm} structural profile"))
}

fn preflight_for<'a>(preflights: &'a [PreflightRecord], arm: &str) -> Result<&'a PreflightRecord> {
    preflights
        .iter()
        .find(|preflight| preflight.arm == arm)
        .with_context(|| format!("missing {arm} preflight"))
}

fn validate_encoder_evidence(preflight: &PreflightRecord) -> Result<()> {
    ensure!(
        preflight
            .dispatches
            .iter()
            .all(|row| row.encoder_ordinal != u64::MAX),
        "{} preflight contains a dispatch outside an identified encoder",
        preflight.arm
    );
    let encoders = preflight
        .dispatches
        .iter()
        .map(|row| row.encoder_ordinal)
        .collect::<BTreeSet<_>>();
    let concurrent = preflight
        .dispatches
        .iter()
        .filter(|row| row.encoder_concurrent)
        .map(|row| row.encoder_ordinal)
        .collect::<BTreeSet<_>>();
    ensure!(
        encoders.len() as u64 == preflight.trace.encoders
            && concurrent.len() as u64 == preflight.trace.concurrent_encoders,
        "{} dispatch-to-encoder evidence differs from trace counts",
        preflight.arm
    );
    Ok(())
}

fn tagged_dispatch<'a>(preflight: &'a PreflightRecord, tag: &str) -> Result<&'a DispatchRow> {
    let matches = preflight
        .dispatches
        .iter()
        .filter(|row| row.tag.as_deref() == Some(tag))
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "{} preflight has {} rows tagged {tag}, expected one",
        preflight.arm,
        matches.len()
    );
    Ok(matches[0])
}

fn validate_site_encoder_placement(preflight: &PreflightRecord) -> Result<()> {
    let producer_runs = preflight.arm != "Z";
    for layer in 0..43 {
        for site in ["attention", "ffn"] {
            let prefix = format!("dsv4_mhc:{layer}:{site}");
            let controls = tagged_dispatch(preflight, &format!("{prefix}:controls"))?;
            ensure!(
                controls.kernel == MHC_CONTROLS_KERNEL && !controls.encoder_concurrent,
                "{} {prefix} controls use the wrong kernel or a concurrent encoder",
                preflight.arm
            );
            if producer_runs {
                let rms = tagged_dispatch(preflight, &format!("{prefix}:rms"))?;
                let function = tagged_dispatch(preflight, &format!("{prefix}:function"))?;
                ensure!(
                    rms.kernel == MHC_RMS_KERNEL
                        && function.kernel == MHC_FUNCTION_KERNEL
                        && rms.encoder_ordinal == controls.encoder_ordinal
                        && function.encoder_ordinal == controls.encoder_ordinal
                        && !rms.encoder_concurrent
                        && !function.encoder_concurrent,
                    "{} {prefix} producer and controls do not share one non-concurrent encoder",
                    preflight.arm
                );
            } else {
                let rms_tag = format!("{prefix}:rms");
                let function_tag = format!("{prefix}:function");
                ensure!(
                    !preflight.dispatches.iter().any(|row| {
                        row.tag.as_deref() == Some(rms_tag.as_str())
                            || row.tag.as_deref() == Some(function_tag.as_str())
                    }),
                    "Z retained a tagged producer at {prefix}"
                );
            }
        }
    }
    Ok(())
}

fn validate_preflights(ordinary: &PreflightRecord, preflights: &[PreflightRecord]) -> Result<()> {
    ensure!(ordinary.arm == "ordinary" && ordinary.profile.is_none());
    validate_encoder_evidence(ordinary)?;
    validate_site_encoder_placement(ordinary)?;
    for preflight in preflights {
        validate_encoder_evidence(preflight)?;
        validate_site_encoder_placement(preflight)?;
    }
    let capture = preflight_for(preflights, "C")?;
    let current = preflight_for(preflights, "A")?;
    let producer = preflight_for(preflights, "P")?;
    let zero = preflight_for(preflights, "Z")?;

    for peer in [capture, current, producer, zero] {
        ensure!(
            allocation_plan_matches(&current.allocation, &peer.allocation),
            "{} preflight allocation plan differs from A",
            peer.arm
        );
    }
    ensure!(
        allocation_plan_matches(&current.allocation, &ordinary.allocation),
        "ordinary preflight allocation plan differs from A"
    );

    ensure!(
        ordinary.trace == current.trace && ordinary.dispatches == current.dispatches,
        "passive mHC collector changes ordinary encoder/dispatch topology"
    );
    for peer in [capture, producer] {
        ensure!(
            peer.trace == current.trace && peer.dispatches == current.dispatches,
            "{} structural topology differs from A",
            peer.arm
        );
    }
    ensure!(
        zero.trace.encoders == current.trace.encoders
            && zero.trace.concurrent_encoders == current.trace.concurrent_encoders
            && zero.trace.dispatches + DELETED_DISPATCH_COUNT as u64 == current.trace.dispatches,
        "Z trace does not delete exactly {DELETED_DISPATCH_COUNT} dispatches while preserving encoders"
    );
    let removed = removed_dispatches(&current.dispatches, &zero.dispatches)?;
    ensure!(
        removed.len() == DELETED_DISPATCH_COUNT,
        "Z dispatch subsequence removed {} rows, expected {DELETED_DISPATCH_COUNT}",
        removed.len()
    );
    let rms = removed
        .iter()
        .filter(|row| row.kernel == MHC_RMS_KERNEL)
        .count();
    let function = removed
        .iter()
        .filter(|row| row.kernel == MHC_FUNCTION_KERNEL)
        .count();
    ensure!(
        rms == MHC_SITE_COUNT && function == MHC_SITE_COUNT,
        "Z removed {rms} mHC RMS and {function} function rows, expected {MHC_SITE_COUNT} each"
    );
    ensure!(
        removed
            .iter()
            .all(|row| row.kernel == MHC_RMS_KERNEL || row.kernel == MHC_FUNCTION_KERNEL),
        "Z removed a dispatch outside the two frozen mHC producer families"
    );
    let removed_tags = removed
        .iter()
        .map(|row| {
            row.tag
                .clone()
                .context("removed mHC dispatch lacks an explicit site tag")
        })
        .collect::<Result<BTreeSet<_>>>()?;
    ensure!(
        removed_tags.len() == DELETED_DISPATCH_COUNT,
        "removed mHC dispatch tags are missing or duplicated"
    );
    let expected_tags = (0..43)
        .flat_map(|layer| {
            ["attention", "ffn"].into_iter().flat_map(move |site| {
                ["rms", "function"]
                    .into_iter()
                    .map(move |operation| format!("dsv4_mhc:{layer}:{site}:{operation}"))
            })
        })
        .collect::<BTreeSet<_>>();
    ensure!(
        removed_tags == expected_tags,
        "Z deletion does not cover each tagged mHC producer exactly once"
    );
    for row in &removed {
        let tag = row
            .tag
            .as_deref()
            .context("removed mHC dispatch lacks tag")?;
        ensure!(
            (tag.ends_with(":rms") && row.kernel == MHC_RMS_KERNEL)
                || (tag.ends_with(":function") && row.kernel == MHC_FUNCTION_KERNEL),
            "tagged mHC dispatch {tag} uses unexpected kernel {}",
            row.kernel
        );
    }

    let expected_commands = command_signature(profile_for(preflights, "A")?);
    let expected_queue = profile_for(preflights, "A")?.queue_identity;
    for arm in ["C", "P", "Z"] {
        let profile = profile_for(preflights, arm)?;
        ensure!(
            command_signature(profile) == expected_commands
                && profile.queue_identity == expected_queue,
            "{arm} command/layer/queue topology differs from A"
        );
    }
    for arm in ["C", "A", "P", "Z"] {
        validate_site_ledger(profile_for(preflights, arm)?)?;
    }
    Ok(())
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn decision_bound(values: &[f64]) -> Result<Bound> {
    ensure!(values.len() == 6, "decision bound requires six sextets");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "decision bound contains non-finite input"
    );
    let mean = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / 5.0;
    let sd = variance.sqrt();
    let se = sd / 6.0f64.sqrt();
    Ok(Bound {
        mean,
        sd,
        se,
        lower95: mean - T_CRITICAL_ONE_SIDED_95_DF5 * se,
        upper95: mean + T_CRITICAL_ONE_SIDED_95_DF5 * se,
        zero_variance: sd == 0.0,
    })
}

fn symmetric_relative_difference(left: f64, right: f64) -> f64 {
    2.0 * (left - right).abs() / (left + right)
}

fn observation_value(observation: &TimedObservation, gpu: bool) -> f64 {
    if gpu {
        observation.profile.union_gpu_ms
    } else {
        observation.profile.wall_ms
    }
}

fn arm_mean(observations: &[TimedObservation], arm: &str, gpu: bool) -> Result<f64> {
    let values = observations
        .iter()
        .filter(|observation| observation.arm == arm)
        .map(|observation| observation_value(observation, gpu))
        .collect::<Vec<_>>();
    ensure!(
        values.len() == 2,
        "sextet contains {} {arm} observations, expected two",
        values.len()
    );
    Ok(mean(&values))
}

fn sextet_effects(observations: &[TimedObservation]) -> Result<Vec<SextetEffect>> {
    ensure!(observations.len() == ARM_SEQUENCE.len());
    let mut effects = Vec::with_capacity(6);
    for (index, sextet) in observations.chunks_exact(6).enumerate() {
        let a_wall_ms = arm_mean(sextet, "A", false)?;
        let p_wall_ms = arm_mean(sextet, "P", false)?;
        let z_wall_ms = arm_mean(sextet, "Z", false)?;
        ensure!(
            a_wall_ms.is_finite() && a_wall_ms > 0.0,
            "sextet {} has invalid A wall denominator",
            index + 1
        );
        let a_gpu_ms = arm_mean(sextet, "A", true)?;
        let p_gpu_ms = arm_mean(sextet, "P", true)?;
        let z_gpu_ms = arm_mean(sextet, "Z", true)?;
        let q_wall = (p_wall_ms - z_wall_ms) / a_wall_ms;
        let s_wall = (a_wall_ms - z_wall_ms) / a_wall_ms;
        let r_wall = (p_wall_ms - a_wall_ms) / a_wall_ms;
        let q_gpu = (p_gpu_ms - z_gpu_ms) / a_wall_ms;
        let s_gpu = (a_gpu_ms - z_gpu_ms) / a_wall_ms;
        effects.push(SextetEffect {
            sextet: index + 1,
            block: index / 3 + 1,
            a_wall_ms,
            p_wall_ms,
            z_wall_ms,
            q_wall,
            s_wall,
            r_wall,
            kill_wall: q_wall.max(s_wall),
            clear_wall: q_wall.min(s_wall),
            q_gpu,
            s_gpu,
            kill_gpu: q_gpu.max(s_gpu),
            clear_gpu: q_gpu.min(s_gpu),
        });
    }
    Ok(effects)
}

fn block_effect_medians(
    effects: &[SextetEffect],
    value: impl Fn(&SextetEffect) -> f64,
) -> [f64; 2] {
    [
        median(&effects[..3].iter().map(&value).collect::<Vec<_>>()),
        median(&effects[3..].iter().map(value).collect::<Vec<_>>()),
    ]
}

fn raw_stationarity(observations: &[TimedObservation], arm: &'static str) -> Result<f64> {
    let mut block = [Vec::new(), Vec::new()];
    for observation in observations.iter().filter(|row| row.arm == arm) {
        ensure!((1..=2).contains(&observation.block));
        block[observation.block - 1].push(observation.profile.wall_ms);
    }
    ensure!(
        block[0].len() == 6 && block[1].len() == 6,
        "stationarity requires six {arm} observations per block"
    );
    let left = median(&block[0]);
    let right = median(&block[1]);
    ensure!(
        left.is_finite() && left > 0.0 && right.is_finite() && right > 0.0,
        "{arm} stationarity medians are invalid"
    );
    Ok(symmetric_relative_difference(left, right))
}

fn push_failure(failures: &mut Vec<String>, condition: bool, label: impl Into<String>) {
    if !condition {
        failures.push(label.into());
    }
}

#[allow(clippy::too_many_arguments)]
fn classify(
    failures: &[String],
    kill_wall: &Bound,
    clear_wall: &Bound,
    kill_gpu: &Bound,
    clear_gpu: &Bound,
    kill_wall_blocks: [f64; 2],
    clear_wall_blocks: [f64; 2],
    kill_gpu_blocks: [f64; 2],
    clear_gpu_blocks: [f64; 2],
) -> (&'static str, &'static str) {
    let kill = failures.is_empty()
        && kill_wall.upper95 < GATE
        && kill_gpu.upper95 < GATE
        && kill_wall_blocks.iter().all(|value| *value < GATE)
        && kill_gpu_blocks.iter().all(|value| *value < GATE);
    let clear = failures.is_empty()
        && clear_wall.lower95 >= GATE
        && clear_wall_blocks.iter().all(|value| *value >= GATE)
        && clear_gpu.lower95 > 0.0
        && clear_gpu_blocks.iter().all(|value| *value > 0.0);
    if kill {
        (
            "KILL",
            "close frozen zero-producer substitution premise only",
        )
    } else if clear {
        (
            "IMPOSSIBLE CEILING CLEAR",
            "authorize separately preregistered exact chargeback only",
        )
    } else {
        ("HOLD", "none")
    }
}

fn reduce(acquisition: &AcquisitionRecord) -> Result<Decision> {
    let observations = &acquisition.observations;
    let mut failures = Vec::new();
    let live_campaign = campaign_identity(&acquisition.build).ok();
    push_failure(
        &mut failures,
        acquisition.release_build
            && acquisition.device_name == "Apple M4 Max"
            && !acquisition.os_version.is_empty()
            && power_evidence_is_complete(&acquisition.power_before_timed)
            && power_evidence_is_complete(&acquisition.power_after_timed)
            && metal_evidence_is_complete(&acquisition.metal)
            && build_packet_is_canonical(&acquisition.build)
            && source_state_matches_build(&acquisition.source_before, &acquisition.build)
            && source_state_matches_build(&acquisition.source_after, &acquisition.build)
            && acquisition.source_before.commit == acquisition.source_after.commit
            && acquisition.source_before.source_state == acquisition.source_after.source_state
            && acquisition.build_identity_sha256
                == hex(Sha256::digest(
                    serde_json::to_vec(&acquisition.build).unwrap_or_default(),
                ))
            && parse_hex_32(&acquisition.binary_sha256).is_ok()
            && acquisition.metallib_sha256 == hex(deepseek_v4_diagnostics_metallib_sha256())
            && live_campaign.is_some_and(|identity| {
                identity.build_identity_sha256 == acquisition.build_identity_sha256
                    && identity.binary_sha256 == acquisition.binary_sha256
                    && identity.metallib_sha256 == acquisition.metallib_sha256
            }),
        "build_hardware_identity",
    );
    push_failure(
        &mut failures,
        acquisition.observer_counts_before_timed == [0, 0, 0]
            && acquisition.observer_counts_after_timed == [0, 0, 0]
            && diagnostics_observer_active_counts() == [0, 0, 0],
        "structural_observer_inactive",
    );
    let expected_commands = profile_for(&acquisition.preflights, "A")
        .map(command_signature)
        .unwrap_or_default();
    let expected_queue = profile_for(&acquisition.preflights, "A")
        .map(|profile| profile.queue_identity)
        .unwrap_or_default();
    push_failure(
        &mut failures,
        !expected_commands.is_empty(),
        "timed_command_reference",
    );
    push_failure(
        &mut failures,
        validate_preflights(&acquisition.ordinary_preflight, &acquisition.preflights).is_ok(),
        "serialized_structural_preflight",
    );
    push_failure(
        &mut failures,
        observations.len() == ARM_SEQUENCE.len(),
        "observation_count",
    );
    for (index, observation) in observations.iter().enumerate() {
        let expected = ARM_SEQUENCE.get(index).copied();
        push_failure(
            &mut failures,
            expected.is_some_and(|arm| arm.label() == observation.arm)
                && observation.ordinal == index
                && observation.block == index / 18 + 1
                && observation.sextet == index / 6 + 1
                && observation.position_in_sextet == index % 6,
            format!("observation_schedule_{index}"),
        );
        let profile = &observation.profile;
        push_failure(
            &mut failures,
            profile.execution == observation.arm,
            format!("execution_label_{index}"),
        );
        push_failure(
            &mut failures,
            profile.sites.is_empty()
                && command_signature(profile) == expected_commands
                && profile.queue_identity == expected_queue,
            format!("command_topology_{index}"),
        );
        push_failure(
            &mut failures,
            profile.wall_ms.is_finite()
                && profile.wall_ms > 0.0
                && profile.union_gpu_ms.is_finite()
                && profile.union_gpu_ms > 0.0
                && profile.raw_gpu_ms.is_finite()
                && profile.raw_gpu_ms > 0.0,
            format!("timing_{index}"),
        );
        push_failure(
            &mut failures,
            profile.outside_gpu_ms >= -0.25,
            format!("outside_gpu_{index}"),
        );
        push_failure(
            &mut failures,
            observation.endpoint_matches_capture,
            format!("endpoint_{index}"),
        );
        push_failure(
            &mut failures,
            observation.pipeline_delta.misses == 0
                && observation.pipeline_delta.miss_wall_ns == 0
                && observation.pipeline_delta.compiler_wall_ns == 0,
            format!("pipeline_cache_{index}"),
        );
    }

    if let Some(first) = observations.first() {
        push_failure(
            &mut failures,
            allocation_identity_is_valid(&first.allocation)
                && first.allocation.contract.oracle_payload_sha256
                    == Some(acquisition.oracle_pre_sha256.clone())
                && first.allocation.contract.oracle_payload_bytes == 16_908_288
                && first.allocation.contract.oracle_storage_mode == Some("shared"),
            "session_allocation_contract",
        );
        for (index, observation) in observations.iter().enumerate().skip(1) {
            push_failure(
                &mut failures,
                allocation_identity_matches(&first.allocation, &observation.allocation),
                format!("session_allocation_{index}"),
            );
        }
        for (label, allocation) in std::iter::once((
            acquisition.ordinary_preflight.arm,
            &acquisition.ordinary_preflight.allocation,
        ))
        .chain(
            acquisition
                .preflights
                .iter()
                .map(|record| (record.arm, &record.allocation)),
        )
        .chain(
            acquisition
                .correctness
                .iter()
                .map(|record| (record.arm, &record.allocation)),
        )
        .chain(
            acquisition
                .warmups
                .iter()
                .map(|record| (record.arm, &record.allocation)),
        ) {
            push_failure(
                &mut failures,
                allocation_plan_matches(&first.allocation, allocation),
                format!("session_allocation_plan_{label}"),
            );
        }
    }
    push_failure(
        &mut failures,
        acquisition.correctness.len() == 3
            && acquisition
                .correctness
                .iter()
                .map(|record| record.arm)
                .collect::<BTreeSet<_>>()
                == BTreeSet::from(["A", "P", "Z"])
            && acquisition.correctness.iter().all(|record| {
                record.matches_capture
                    && record.endpoint_position == TOKENS as u32
                    && record.continuation_position == TOKENS as u32 + 1
            }),
        "untimed_correctness",
    );
    push_failure(
        &mut failures,
        acquisition.warmups.len() == 3
            && acquisition
                .warmups
                .iter()
                .map(|record| record.arm)
                .collect::<BTreeSet<_>>()
                == BTreeSet::from(["A", "P", "Z"])
            && acquisition.warmups.iter().all(|record| {
                record.profile.execution == record.arm
                    && record.profile.sites.is_empty()
                    && command_signature(&record.profile) == expected_commands
                    && record.profile.queue_identity == expected_queue
            }),
        "pipeline_warmups",
    );
    let capture_preflight = profile_for(&acquisition.preflights, "C").ok();
    let current_preflight = profile_for(&acquisition.preflights, "A").ok();
    let captures_valid = acquisition.captures.len() == 2
        && allocation_identity_is_valid(&acquisition.captures[0].allocation)
        && allocation_plan_matches(
            &acquisition.captures[0].allocation,
            &acquisition.captures[1].allocation,
        )
        && acquisition.captures[0].manifest == acquisition.captures[1].manifest
        && acquisition.captures[0].manifest_sha256 == acquisition.captures[1].manifest_sha256
        && acquisition.captures.iter().all(|capture| {
            serde_json::to_vec(&capture.manifest)
                .ok()
                .is_some_and(|bytes| capture.manifest_sha256 == hex(Sha256::digest(bytes)))
        })
        && acquisition.captures.first().is_some_and(|capture| {
            let manifest = &capture.manifest;
            validate_site_records("C", &manifest.sites).is_ok()
                && manifest.arithmetic_parent == PARENT_COMMIT
                && manifest.experimental_commit
                    == acquisition
                        .build
                        .get("build_commit")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                && manifest.build_source_state == acquisition.source_before.source_state
                && manifest.build_identity_sha256 == acquisition.build_identity_sha256
                && manifest.binary_sha256 == acquisition.binary_sha256
                && manifest.metallib_sha256 == acquisition.metallib_sha256
                && manifest.identity.model_content_id == acquisition.model_content_id
                && manifest.identity.token_sha256 == acquisition.token_sha256
                && manifest.payload_bytes == 16_908_288
                && manifest.dtype == "f32"
                && manifest.storage_mode == "shared"
                && manifest.shape == [24, TOKENS as u64, MHC_SITE_COUNT as u64]
                && capture_preflight.is_some_and(|profile| {
                    manifest.sites == profile.sites
                        && manifest.commands == command_topology(profile)
                        && manifest.queue_identity == profile.queue_identity
                })
                && current_preflight.is_some_and(|profile| {
                    manifest.commands == command_topology(profile)
                        && manifest.queue_identity == profile.queue_identity
                })
        });
    push_failure(&mut failures, captures_valid, "capture_manifest");
    push_failure(
        &mut failures,
        acquisition.oracle_pre_sha256 == acquisition.oracle_post_sha256
            && acquisition.captures.first().is_some_and(|capture| {
                capture.manifest.payload_sha256 == acquisition.oracle_pre_sha256
            }),
        "oracle_seal",
    );
    push_failure(
        &mut failures,
        acquisition.parent_endpoint_match
            && acquisition.parent_endpoint_file_sha256 == PARENT_ENDPOINT_FILE_SHA256
            && validate_parent_endpoint(&acquisition.parent_endpoint).is_ok()
            && acquisition.parent_endpoint.payload.acquired_at_utc <= acquisition.acquired_at_utc
            && acquisition.parent_endpoint.payload.device_name == acquisition.device_name
            && acquisition.parent_endpoint.payload.device_registry_id
                == acquisition.device_registry_id
            && acquisition.parent_endpoint.payload.os_version == acquisition.os_version
            && acquisition
                .parent_endpoint
                .payload
                .metal_system_profile_sha256
                == acquisition.metal.system_profile_sha256
            && acquisition.parent_endpoint.payload.qwen_env == acquisition.qwen_env
            && acquisition.parent_endpoint.payload.model_content_id == acquisition.model_content_id
            && acquisition.parent_endpoint.payload.token_sha256 == acquisition.token_sha256
            && acquisition.parent_endpoint.payload.metallib_sha256 == acquisition.metallib_sha256
            && acquisition.observations.iter().all(|observation| {
                observation.endpoint == acquisition.parent_endpoint.payload.endpoint
            }),
        "parent_endpoint",
    );
    push_failure(&mut failures, acquisition.host_before_valid, "host_before");
    push_failure(&mut failures, acquisition.host_after_valid, "host_after");
    push_failure(&mut failures, acquisition.vm_valid, "vm_pressure");

    ensure!(
        observations.len() == ARM_SEQUENCE.len(),
        "cannot reduce an incomplete acquisition"
    );
    let effects = sextet_effects(observations)?;
    let kill_wall_values = effects
        .iter()
        .map(|effect| effect.kill_wall)
        .collect::<Vec<_>>();
    let clear_wall_values = effects
        .iter()
        .map(|effect| effect.clear_wall)
        .collect::<Vec<_>>();
    let kill_gpu_values = effects
        .iter()
        .map(|effect| effect.kill_gpu)
        .collect::<Vec<_>>();
    let clear_gpu_values = effects
        .iter()
        .map(|effect| effect.clear_gpu)
        .collect::<Vec<_>>();
    let kill_wall = decision_bound(&kill_wall_values)?;
    let clear_wall = decision_bound(&clear_wall_values)?;
    let kill_gpu = decision_bound(&kill_gpu_values)?;
    let clear_gpu = decision_bound(&clear_gpu_values)?;
    for (label, bound) in [
        ("kill_wall", &kill_wall),
        ("clear_wall", &clear_wall),
        ("kill_gpu", &kill_gpu),
        ("clear_gpu", &clear_gpu),
    ] {
        push_failure(
            &mut failures,
            !bound.zero_variance,
            format!("zero_variance_{label}"),
        );
    }

    let mut stationarity = BTreeMap::new();
    for arm in ["A", "P", "Z"] {
        let value = raw_stationarity(observations, arm)?;
        push_failure(
            &mut failures,
            value <= STATIONARITY_GATE,
            format!("stationarity_{arm}"),
        );
        stationarity.insert(arm, value);
    }

    let kill_wall_blocks = block_effect_medians(&effects, |effect| effect.kill_wall);
    let clear_wall_blocks = block_effect_medians(&effects, |effect| effect.clear_wall);
    let kill_gpu_blocks = block_effect_medians(&effects, |effect| effect.kill_gpu);
    let clear_gpu_blocks = block_effect_medians(&effects, |effect| effect.clear_gpu);
    let mut block_medians = BTreeMap::new();
    block_medians.insert("kill_wall", kill_wall_blocks);
    block_medians.insert("clear_wall", clear_wall_blocks);
    block_medians.insert("kill_gpu", kill_gpu_blocks);
    block_medians.insert("clear_gpu", clear_gpu_blocks);

    let (verdict, authority) = classify(
        &failures,
        &kill_wall,
        &clear_wall,
        &kill_gpu,
        &clear_gpu,
        kill_wall_blocks,
        clear_wall_blocks,
        kill_gpu_blocks,
        clear_gpu_blocks,
    );
    Ok(Decision {
        verdict,
        authority,
        validity_failures: failures,
        stationarity,
        sextets: effects,
        kill_wall,
        clear_wall,
        kill_gpu,
        clear_gpu,
        block_medians,
    })
}

fn read_parent_endpoint(path: &Path) -> Result<ParentEndpoint> {
    let bytes =
        std::fs::read(path).with_context(|| format!("read parent endpoint {}", path.display()))?;
    ensure!(
        hex(Sha256::digest(&bytes)) == PARENT_ENDPOINT_FILE_SHA256,
        "parent endpoint file does not match the frozen artifact"
    );
    let endpoint: ParentEndpoint = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse parent endpoint {}", path.display()))?;
    validate_parent_endpoint(&endpoint)?;
    Ok(endpoint)
}

fn validate_parent_endpoint(endpoint: &ParentEndpoint) -> Result<()> {
    ensure!(
        endpoint.schema_version == 2,
        "unsupported parent endpoint schema"
    );
    ensure!(
        endpoint.payload_sha256 == hex(Sha256::digest(serde_json::to_vec(&endpoint.payload)?)),
        "parent endpoint payload digest is invalid"
    );
    ensure!(
        endpoint.payload.source_commit == PARENT_COMMIT,
        "parent endpoint source {} != frozen {PARENT_COMMIT}",
        endpoint.payload.source_commit
    );
    ensure!(
        endpoint.payload.report_kind == "dsv4_k160_mhc_parent_endpoint",
        "parent endpoint has wrong report kind"
    );
    ensure!(
        endpoint.payload.release_build,
        "parent endpoint was not acquired from a release binary"
    );
    let parent_build = &endpoint.payload.build;
    ensure!(
        parent_build.get("build_commit").and_then(Value::as_str) == Some(PARENT_COMMIT)
            && parent_build.get("runtime_commit").and_then(Value::as_str) == Some(PARENT_COMMIT)
            && build_packet_is_canonical(parent_build),
        "parent endpoint build identity is not a clean bbfcca8 match"
    );
    ensure!(
        endpoint.payload.build_identity_sha256
            == hex(Sha256::digest(serde_json::to_vec(parent_build)?)),
        "parent endpoint build-identity digest is invalid"
    );
    parse_hex_32(&endpoint.payload.binary_sha256)?;
    parse_hex_32(&endpoint.payload.harness_sha256)?;
    parse_hex_32(&endpoint.payload.metallib_sha256)?;
    parse_hex_32(&endpoint.payload.metal_system_profile_sha256)?;
    Ok(())
}

fn token_digest(tokens: &[u32]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"qwen.dsv4.mhc-delete-tokens.v1\0");
    digest.update(bytemuck::cast_slice(tokens));
    digest.finalize().into()
}

fn run_campaign(
    args: &Dsv4MhcDeleteArgs,
    build: Value,
    qwen_env: BTreeMap<String, String>,
) -> Result<AcquisitionRecord> {
    validate_campaign_build(&build)?;
    let source_before = capture_source_state()?;
    ensure!(
        source_state_matches_build(&source_before, &build),
        "live source state differs from the compiled campaign"
    );
    let campaign = campaign_identity(&build)?;
    let acquired_at_utc = super::utc_iso8601_now();
    let parent_endpoint = read_parent_endpoint(&args.parent_endpoint)?;
    let parent = &parent_endpoint.payload;
    let tokens = token_ids();
    let expected_token_sha256 = token_digest(&tokens);
    ensure!(
        parse_hex_32(&parent.token_sha256)? == expected_token_sha256,
        "parent endpoint token digest differs from the frozen input"
    );
    ensure!(
        parent.qwen_env == qwen_env,
        "parent endpoint QWEN environment differs from this campaign"
    );
    ensure!(
        parent.endpoint_digest_domain == "qwen.dsv4.mhc-delete-timed-endpoint.v1",
        "parent endpoint uses an unknown digest domain"
    );

    let host_before = HostSnapshot::capture("dsv4_mhc_before")?;
    host_before.validate()?;
    let host_before_valid = true;
    let vm_before = VmCounters::capture()?;

    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open K160 model {}", args.model.display()))?;
    let identity_cache = CheckpointIdentityCache::new(&args.identity_cache);
    let content = checkpoint_content_identity(&gguf, &identity_cache)
        .context("derive strong K160 model-content identity")?;
    let model_content_id = DeepSeekV4ModelContentId::new(content.content_id);
    ensure!(
        parse_hex_32(&parent.model_content_id)? == content.content_id,
        "parent endpoint model-content identity differs from the loaded model"
    );

    let ctx = MetalContext::new().context("create mHC campaign Metal context")?;
    let device_name = ctx.device.name().to_string();
    let device_registry_id = ctx.device.registryID();
    let current_os_version = os_version()?;
    let (metal_system_profile, metal_system_profile_sha256) = capture_metal_system_profile()?;
    ensure!(
        device_name == "Apple M4 Max",
        "canonical mHC campaign requires Apple M4 Max, got {device_name}"
    );
    ensure!(
        parent.device_name == device_name
            && parent.device_registry_id == device_registry_id
            && parent.os_version == current_os_version
            && parent.metal_system_profile_sha256 == metal_system_profile_sha256,
        "parent endpoint hardware/OS identity differs from this campaign"
    );
    ensure!(
        parent.metallib_sha256 == campaign.metallib_sha256,
        "parent and experimental metallib identities differ"
    );
    let initial_signals = ctx.memory_signals();
    let metal = MetalRecord {
        max_buffer_length: ctx.max_buffer_length(),
        recommended_max_working_set_bytes: initial_signals.recommended_max_bytes,
        initial_allocated_bytes: initial_signals.current_allocated_bytes,
        process_limit_remaining_bytes: initial_signals.process_limit_remaining_bytes,
        system_profile: metal_system_profile,
        system_profile_sha256: metal_system_profile_sha256,
    };
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, TOKENS + 1)
        .context("plan K160 mHC campaign residency")?;
    let tensor_count = plan.residency_report().tensor_count;
    let source_bytes = plan.residency_report().source_bytes;
    let expert_count = plan.config().expert_count as usize;
    ensure!(
        tensor_count == K160_TENSORS,
        "K160 tensor count {tensor_count} != {K160_TENSORS}"
    );
    ensure!(
        source_bytes == K160_SOURCE_BYTES,
        "K160 source bytes {source_bytes} != {K160_SOURCE_BYTES}"
    );
    ensure!(
        expert_count == K160_EXPERTS,
        "K160 expert count {expert_count} != {K160_EXPERTS}"
    );
    let memory_plan = memory_plan_record(plan.memory_plan(), plan.session_capacity());
    let capture_contract = session_contract(&memory_plan, None)?;
    let admitted = plan
        .admit(ctx.memory_signals())
        .context("admit K160 mHC campaign")?;
    let load_started = Instant::now();
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
        .context("load K160 mHC campaign residency")?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1e3;
    let mut residency = Some(realized.into_residency());
    ctx.set_pipeline_cache_metrics_enabled(true);

    let (capture_zero, capture_zero_record) = acquire_capture(
        &ctx,
        &mut residency,
        model_content_id,
        &tokens,
        &campaign,
        &capture_contract,
    )?;
    let reference_evidence = capture_zero.evidence().clone();
    let expected_endpoint = reference_evidence.timed_endpoint();
    let expected_endpoint_record = timed_endpoint(expected_endpoint);
    let (capture_one, capture_one_record) = acquire_capture(
        &ctx,
        &mut residency,
        model_content_id,
        &tokens,
        &campaign,
        &capture_contract,
    )?;
    validate_site_records("C", &capture_zero_record.manifest.sites)?;
    validate_site_records("C", &capture_one_record.manifest.sites)?;
    ensure!(
        capture_zero_record.manifest == capture_one_record.manifest
            && capture_zero_record.manifest_sha256 == capture_one_record.manifest_sha256,
        "independent C0/C1 capture manifests differ"
    );
    let oracle = seal_mhc_delete_oracle_pair(capture_zero, capture_one)
        .context("seal independent mHC captures")?;
    ensure!(oracle.n_tokens() == TOKENS);
    ensure!(oracle.payload_bytes() == 16_908_288);
    ensure!(oracle.identity().token_sha256 == expected_token_sha256);
    ensure!(oracle.identity().model_content_id == content.content_id);
    let parent_endpoint_match = parent.endpoint == expected_endpoint_record;
    ensure!(
        parent_endpoint_match,
        "instrumented capture endpoint differs from the independent parent"
    );
    let session_contract = session_contract(&memory_plan, Some(&oracle))?;

    let ordinary_preflight = run_ordinary_preflight(
        &ctx,
        &mut residency,
        model_content_id,
        &tokens,
        &session_contract,
    )?;
    let preflights = vec![
        run_capture_preflight(
            &ctx,
            &mut residency,
            model_content_id,
            &tokens,
            &session_contract,
        )?,
        run_arm_preflight(
            &ctx,
            &mut residency,
            model_content_id,
            &tokens,
            Arm::A,
            &oracle,
            &session_contract,
        )?,
        run_arm_preflight(
            &ctx,
            &mut residency,
            model_content_id,
            &tokens,
            Arm::P,
            &oracle,
            &session_contract,
        )?,
        run_arm_preflight(
            &ctx,
            &mut residency,
            model_content_id,
            &tokens,
            Arm::Z,
            &oracle,
            &session_contract,
        )?,
    ];
    validate_preflights(&ordinary_preflight, &preflights)?;
    let capture_preflight = profile_for(&preflights, "C")?;
    let current_preflight = profile_for(&preflights, "A")?;
    ensure!(
        capture_zero_record.manifest.sites == capture_preflight.sites
            && capture_zero_record.manifest.commands == command_topology(capture_preflight)
            && capture_zero_record.manifest.commands == command_topology(current_preflight)
            && capture_zero_record.manifest.queue_identity == capture_preflight.queue_identity
            && capture_zero_record.manifest.queue_identity == current_preflight.queue_identity,
        "C0/C1 capture manifests differ from structural C/A preflight"
    );

    let correctness = [Arm::A, Arm::P, Arm::Z]
        .into_iter()
        .map(|arm| {
            run_correctness_arm(
                &ctx,
                &mut residency,
                model_content_id,
                &tokens,
                arm,
                &oracle,
                &reference_evidence,
                &session_contract,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        correctness.iter().all(|record| record.matches_capture),
        "untimed A/P/Z correctness differs from the sealed capture"
    );

    let warmups = [Arm::A, Arm::P, Arm::Z]
        .into_iter()
        .map(|arm| {
            run_warmup_arm(
                &ctx,
                &mut residency,
                model_content_id,
                &tokens,
                arm,
                &oracle,
                &session_contract,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let oracle_pre_sha256 = hex(oracle
        .current_payload_sha256()
        .context("hash sealed oracle after warmup")?);
    ensure!(oracle_pre_sha256 == hex(oracle.sealed_sha256()));
    let observer_counts_before_timed = diagnostics_observer_active_counts();
    ensure!(
        observer_counts_before_timed == [0, 0, 0],
        "structural observers remain active before timed acquisition"
    );
    let power_before_timed = serde_json::to_value(super::capture_power_snapshot())?;
    ensure!(
        power_evidence_is_complete(&power_before_timed),
        "pre-timing power-mode evidence is incomplete or non-canonical"
    );

    let mut observations = Vec::with_capacity(ARM_SEQUENCE.len());
    for (ordinal, arm) in ARM_SEQUENCE.into_iter().enumerate() {
        observations.push(run_timed_arm(
            &ctx,
            &mut residency,
            model_content_id,
            &tokens,
            arm,
            &oracle,
            expected_endpoint,
            ordinal,
            &session_contract,
        )?);
    }
    let observer_counts_after_timed = diagnostics_observer_active_counts();
    ensure!(
        observer_counts_after_timed == [0, 0, 0],
        "structural observers became active during timed acquisition"
    );
    let power_after_timed = serde_json::to_value(super::capture_power_snapshot())?;
    ensure!(
        power_evidence_is_complete(&power_after_timed),
        "post-timing power-mode evidence is incomplete or non-canonical"
    );
    let oracle_post_sha256 = hex(oracle
        .current_payload_sha256()
        .context("hash sealed oracle after acquisition")?);

    let vm_after = VmCounters::capture()?;
    let vm_delta = VmDelta::between(&vm_before, &vm_after);
    let vm_valid = vm_delta.is_valid();
    let host_after = HostSnapshot::capture("dsv4_mhc_after")?;
    let host_after_valid = host_after.validate().is_ok();
    let source_after = capture_source_state()?;
    ensure!(
        source_state_matches_build(&source_after, &build)
            && source_after.commit == source_before.commit
            && source_after.source_state == source_before.source_state,
        "source/worktree identity changed during mHC acquisition"
    );
    let acquisition = AcquisitionRecord {
        schema_version: 1,
        report_kind: "dsv4_k160_mhc_delete_acquisition",
        acquired_at_utc,
        build,
        build_identity_sha256: campaign.build_identity_sha256,
        binary_sha256: campaign.binary_sha256,
        metallib_sha256: campaign.metallib_sha256,
        release_build: !cfg!(debug_assertions),
        source_before,
        source_after,
        device_name,
        device_registry_id,
        os_version: current_os_version,
        metal,
        power_before_timed,
        power_after_timed,
        qwen_env,
        model_path: args.model.display().to_string(),
        model_content_id: hex(content.content_id),
        identity_cache_outcome: identity_cache_outcome(content.outcome).to_string(),
        identity_hashed_bytes: content.bytes_hashed,
        tensor_count,
        source_bytes,
        expert_count,
        token_ids: tokens,
        token_sha256: hex(expected_token_sha256),
        continuation_token: CONTINUATION_TOKEN,
        load_ms,
        memory_plan,
        parent_endpoint_file_sha256: PARENT_ENDPOINT_FILE_SHA256,
        parent_endpoint,
        parent_endpoint_match,
        captures: vec![capture_zero_record, capture_one_record],
        oracle_pre_sha256,
        oracle_post_sha256,
        ordinary_preflight,
        preflights,
        correctness,
        warmups,
        observer_counts_before_timed,
        observer_counts_after_timed,
        observations,
        host_before: serde_json::to_value(&host_before)?,
        host_after: serde_json::to_value(&host_after)?,
        vm_before: serde_json::to_value(&vm_before)?,
        vm_after: serde_json::to_value(&vm_after)?,
        vm_delta: serde_json::to_value(&vm_delta)?,
        host_before_valid,
        host_after_valid,
        vm_valid,
    };
    Ok(acquisition)
}

pub fn run(
    args: Dsv4MhcDeleteArgs,
    build: Value,
    qwen_env: BTreeMap<String, String>,
) -> Result<()> {
    let mut sink = ReportSink::reserve(&args.json_out)?;
    let failure_build = build.clone();
    let failure_env = qwen_env.clone();
    match run_campaign(&args, build, qwen_env) {
        Ok(acquisition) => {
            let acquisition_bytes = serde_json::to_vec(&acquisition)?;
            let acquisition_sha256 = hex(Sha256::digest(&acquisition_bytes));
            sink.checkpoint(&AcquisitionHoldReport {
                schema_version: 1,
                report_kind: "dsv4_k160_mhc_delete_acquired_hold",
                verdict: "HOLD",
                authority: "none",
                acquisition_sha256: &acquisition_sha256,
                acquisition: &acquisition,
            })?;
            let decision = reduce(&acquisition)?;
            let report = Report {
                schema_version: 1,
                acquisition_sha256,
                acquisition,
                decision,
            };
            let decision_json = serde_json::to_string_pretty(&report.decision)?;
            sink.publish(&report)?;
            println!("{decision_json}");
            Ok(())
        }
        Err(error) => {
            let failure = FailureReport {
                schema_version: 1,
                report_kind: "dsv4_k160_mhc_delete_failure",
                acquired_at_utc: super::utc_iso8601_now(),
                verdict: "HOLD",
                authority: "none",
                error: format!("{error:#}"),
                build: failure_build,
                release_build: !cfg!(debug_assertions),
                qwen_env: failure_env,
                model_path: args.model.display().to_string(),
                identity_cache_path: args.identity_cache.display().to_string(),
                parent_endpoint_path: args.parent_endpoint.display().to_string(),
            };
            sink.publish(&failure)
                .context("publish durable mHC HOLD after campaign failure")?;
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_build(commit: &str) -> Value {
        serde_json::json!({
            "schema_version": 2,
            "build_commit": commit,
            "build_commit_short": &commit[..9],
            "build_dirty": false,
            "build_source_state": "git-source-sha256-v2:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "stamp_source": "test",
            "stamp_error": null,
            "runtime_commit": commit,
            "runtime_dirty": false,
            "runtime_source_state": "git-source-sha256-v2:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "status": "match",
            "problems": [],
            "overrides": []
        })
    }

    fn temporary_path(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("qwen-{label}-{}-{nonce}.json", std::process::id()))
    }

    fn row(kernel: &str) -> DispatchRow {
        DispatchRow {
            family: "test".to_string(),
            tag: None,
            encoder_ordinal: 0,
            encoder_concurrent: false,
            kernel: kernel.to_string(),
            grid: [1, 1, 1],
            threads: [1, 1, 1],
        }
    }

    fn observation(ordinal: usize, arm: Arm, wall_ms: f64, gpu_ms: f64) -> TimedObservation {
        let mut contract = SessionContractRecord {
            sha256: String::new(),
            memory_plan_sha256: "11".repeat(32),
            forward_limit: TOKENS + 1,
            csa_physical_rows: 768,
            hca_physical_rows: 512,
            session_logical_bytes: 1,
            session_priced_upper_bytes: 1,
            session_storage_mode: "shared",
            oracle_payload_sha256: Some("22".repeat(32)),
            oracle_payload_bytes: 16_908_288,
            oracle_storage_mode: Some("shared"),
            allocations: vec![MemoryAllocationRecord {
                name: "test".to_string(),
                logical_bytes: 1,
                priced_bytes: 1,
                alignment: 1,
                storage_mode: "shared",
            }],
        };
        contract.sha256 = hex(Sha256::digest(serde_json::to_vec(&contract).unwrap()));
        TimedObservation {
            ordinal,
            block: ordinal / 18 + 1,
            sextet: ordinal / 6 + 1,
            position_in_sextet: ordinal % 6,
            arm: arm.label(),
            profile: ProfileRecord {
                execution: arm.label(),
                queue_identity: 1,
                wall_ms,
                raw_gpu_ms: gpu_ms,
                union_gpu_ms: gpu_ms,
                outside_gpu_ms: wall_ms - gpu_ms,
                sites: Vec::new(),
                command_intervals: Vec::new(),
            },
            endpoint: TimedEndpoint {
                sha256: "00".repeat(32),
                position: TOKENS as u32,
                token_count: TOKENS,
                logits_count: 129_280,
                hidden_count: 4_096,
            },
            endpoint_matches_capture: true,
            allocation: SessionAllocation {
                contract,
                realized: vec![RealizedAllocationRecord {
                    requested_bytes: 1,
                    buffer_length: 1,
                    storage_mode: "shared",
                }],
                construction_wall_ms: 1.0,
                before_bytes: 10,
                after_bytes: 20,
                delta_bytes: 10,
            },
            pipeline_delta: PipelineDelta {
                misses: 0,
                miss_wall_ns: 0,
                compiler_wall_ns: 0,
            },
        }
    }

    fn bound(lower95: f64, upper95: f64) -> Bound {
        Bound {
            mean: (lower95 + upper95) * 0.5,
            sd: 0.001,
            se: 0.001,
            lower95,
            upper95,
            zero_variance: false,
        }
    }

    #[test]
    fn frozen_schedule_balances_every_sextet_and_position() {
        for sextet in ARM_SEQUENCE.chunks_exact(6) {
            for arm in [Arm::A, Arm::P, Arm::Z] {
                assert_eq!(sextet.iter().filter(|&&value| value == arm).count(), 2);
            }
        }
        for position in 0..6 {
            for arm in [Arm::A, Arm::P, Arm::Z] {
                assert_eq!(
                    ARM_SEQUENCE
                        .chunks_exact(6)
                        .filter(|sextet| sextet[position] == arm)
                        .count(),
                    2
                );
            }
        }
    }

    #[test]
    fn ordered_subsequence_identifies_only_deleted_producers() {
        let mut rms = row(MHC_RMS_KERNEL);
        rms.tag = Some("dsv4_mhc:0:attention:rms".to_string());
        let mut function = row(MHC_FUNCTION_KERNEL);
        function.tag = Some("dsv4_mhc:0:attention:function".to_string());
        let current = vec![row("before"), rms, function, row("after")];
        let zero = vec![row("before"), row("after")];
        let removed = removed_dispatches(&current, &zero).unwrap();
        assert_eq!(removed.len(), 2);
        assert_eq!(removed[0].kernel, MHC_RMS_KERNEL);
        assert_eq!(removed[1].kernel, MHC_FUNCTION_KERNEL);
    }

    #[test]
    fn explicit_tags_disambiguate_identical_kernel_families() {
        let mut unrelated = row(MHC_RMS_KERNEL);
        unrelated.tag = Some("other:rms".to_string());
        let mut deleted = row(MHC_RMS_KERNEL);
        deleted.tag = Some("dsv4_mhc:0:attention:rms".to_string());
        let current = vec![row("before"), unrelated.clone(), deleted, row("after")];
        let zero = vec![row("before"), unrelated, row("after")];
        let removed = removed_dispatches(&current, &zero).unwrap();
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].tag.as_deref(), Some("dsv4_mhc:0:attention:rms"));
    }

    #[test]
    fn sextet_uses_candidate_favorable_and_conservative_envelopes() {
        let arms = [Arm::A, Arm::P, Arm::Z, Arm::Z, Arm::P, Arm::A];
        let observations = arms
            .into_iter()
            .enumerate()
            .map(|(index, arm)| match arm {
                Arm::A => observation(index, arm, 100.0, 80.0),
                Arm::P => observation(index, arm, 103.0, 82.0),
                Arm::Z => observation(index, arm, 98.0, 78.0),
            })
            .chain((6..ARM_SEQUENCE.len()).map(|index| {
                let arm = ARM_SEQUENCE[index];
                observation(index, arm, 100.0, 80.0)
            }))
            .collect::<Vec<_>>();
        let effects = sextet_effects(&observations).unwrap();
        let first = &effects[0];
        assert!((first.q_wall - 0.05).abs() < 1e-12);
        assert!((first.s_wall - 0.02).abs() < 1e-12);
        assert!((first.r_wall - 0.03).abs() < 1e-12);
        assert!((first.kill_wall - 0.05).abs() < 1e-12);
        assert!((first.clear_wall - 0.02).abs() < 1e-12);
        assert!((first.kill_gpu - 0.04).abs() < 1e-12);
        assert!((first.clear_gpu - 0.02).abs() < 1e-12);
    }

    #[test]
    fn classifier_preserves_frozen_authority_boundaries() {
        let below = bound(0.005, 0.015);
        let above = bound(0.021, 0.030);
        let positive = bound(0.001, 0.010);
        assert_eq!(
            classify(
                &[],
                &below,
                &below,
                &below,
                &below,
                [0.01, 0.01],
                [0.01, 0.01],
                [0.01, 0.01],
                [0.01, 0.01],
            )
            .0,
            "KILL"
        );
        assert_eq!(
            classify(
                &[],
                &above,
                &above,
                &positive,
                &positive,
                [0.03, 0.03],
                [0.025, 0.025],
                [0.005, 0.005],
                [0.005, 0.005],
            )
            .0,
            "IMPOSSIBLE CEILING CLEAR"
        );
        assert_eq!(
            classify(
                &["invalid".to_string()],
                &below,
                &above,
                &below,
                &positive,
                [0.01, 0.01],
                [0.025, 0.025],
                [0.01, 0.01],
                [0.005, 0.005],
            )
            .0,
            "HOLD"
        );
    }

    #[test]
    fn build_overrides_cannot_enter_the_authority_path() {
        let mut build = clean_build(PARENT_COMMIT);
        assert!(build_packet_is_canonical(&build));
        build["overrides"] = serde_json::json!(["allow_dirty"]);
        assert!(!build_packet_is_canonical(&build));
    }

    #[test]
    fn report_sink_publishes_without_overwriting() {
        let path = temporary_path("mhc-report-sink");
        let sink = ReportSink::reserve(&path).unwrap();
        sink.publish(&serde_json::json!({"verdict": "HOLD"}))
            .unwrap();
        let report: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(report["verdict"], "HOLD");
        assert!(ReportSink::reserve(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn report_reservation_is_already_a_durable_hold() {
        let path = temporary_path("mhc-report-reservation");
        let sink = ReportSink::reserve(&path).unwrap();
        let reservation: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(reservation["verdict"], "HOLD");
        assert_eq!(reservation["authority"], "none");
        drop(sink);
        assert!(path.exists());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn acquisition_checkpoint_survives_until_final_publication() {
        let path = temporary_path("mhc-acquisition-checkpoint");
        let mut sink = ReportSink::reserve(&path).unwrap();
        sink.checkpoint(&serde_json::json!({
            "report_kind": "acquired_hold",
            "verdict": "HOLD",
            "authority": "none"
        }))
        .unwrap();
        let acquired: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(acquired["report_kind"], "acquired_hold");
        sink.publish(&serde_json::json!({"verdict": "KILL"}))
            .unwrap();
        let final_report: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(final_report["verdict"], "KILL");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn parent_endpoint_payload_is_self_authenticating() {
        let build = clean_build(PARENT_COMMIT);
        let payload = ParentEndpointPayload {
            report_kind: "dsv4_k160_mhc_parent_endpoint".to_string(),
            acquired_at_utc: "2026-08-09T00:00:00Z".to_string(),
            source_commit: PARENT_COMMIT.to_string(),
            build_identity_sha256: hex(Sha256::digest(serde_json::to_vec(&build).unwrap())),
            build,
            binary_sha256: "11".repeat(32),
            harness_sha256: "22".repeat(32),
            metallib_sha256: "33".repeat(32),
            release_build: true,
            device_name: "Apple M4 Max".to_string(),
            device_registry_id: 1,
            os_version: "26.0".to_string(),
            metal_system_profile_sha256: "99".repeat(32),
            qwen_env: BTreeMap::new(),
            model_content_id: "44".repeat(32),
            token_sha256: "66".repeat(32),
            endpoint_digest_domain: "qwen.dsv4.mhc-delete-timed-endpoint.v1".to_string(),
            endpoint: TimedEndpoint {
                sha256: "88".repeat(32),
                position: TOKENS as u32,
                token_count: TOKENS,
                logits_count: 129_280,
                hidden_count: 4_096,
            },
        };
        let mut endpoint = ParentEndpoint {
            schema_version: 2,
            payload_sha256: hex(Sha256::digest(serde_json::to_vec(&payload).unwrap())),
            payload,
        };
        validate_parent_endpoint(&endpoint).unwrap();
        endpoint.payload.device_registry_id += 1;
        assert!(validate_parent_endpoint(&endpoint).is_err());
    }

    #[test]
    fn committed_parent_endpoint_matches_the_frozen_digest() {
        assert_eq!(
            hex(Sha256::digest(PARENT_ENDPOINT_ARTIFACT)),
            PARENT_ENDPOINT_FILE_SHA256
        );
        let endpoint: ParentEndpoint = serde_json::from_slice(PARENT_ENDPOINT_ARTIFACT).unwrap();
        validate_parent_endpoint(&endpoint).unwrap();
    }
}
