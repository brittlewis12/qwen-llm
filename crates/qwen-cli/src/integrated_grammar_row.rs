use crate::grammar_row_runtime::{
    BankKey, CandidateBankPlan, FrozenBranchManifest, FrozenGrammarRuntime, GrammarState,
    PreparedCandidateBanks,
};
use crate::host_validity::{HostSnapshot, VmCounters, VmDelta};
use crate::messages::{ChatMessage, render_qwen_messages_prompt};
use crate::response_shape_runtime::RuntimeStateKind;
use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use objc2::rc::autoreleasepool;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};
use qwen_llm::{
    loader::Model,
    metal::{Buffer, KernelEncoder, MetalContext, MetalTensor, encode_mat_vec_q6_k_f32},
    metal_dflash::{MetalDFlashLayerMajorScratch, prefill_tokens_profiled_with_tail},
    metal_forward::{
        LmHeadTail, LmHeadTailEvidence, LmHeadTailKind, MetalSession, SessionSnapshot,
    },
    model::ArchKind,
    pid_metrics::PidSnapshot,
    runtime::{LoadedModel, Runtime, Sequence, SequenceConfig},
    sampling::{SAMPLER_ALGORITHM_VERSION, Sampler, SamplingConfig, SamplingError},
    tensor::GgmlType,
    tokenizer::NativeTokenizer,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

const RESULT_SCHEMA: &str = "qwen-a3b-integrated-grammar-row/v1";
const PROMPT_SCHEMA: &str = "qwen-integrated-grammar-prompt/v1";
const PROMPT_FILE_SHA256: &str = "c668685d855eea9270ab18a5f64a479b25c5efe85e17dc51391b92dfe98a59d9";
const PREREQUISITE_SHA256: &str =
    "1e37f4882be856fc8043a98a00bfeaa3a7ff8a726ab0f73a99a9e98d707c689c";
const MODEL_PATH: &str = "/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
const MODEL_BYTES: u64 = 22_134_528_992;
const MODEL_SHA256: &str = "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61";
const OUTPUT_WEIGHT_SHA256: &str =
    "122386a599833ffec3a266e48dd8a90aedc65e24c31323ce2388f40d7cc7b30c";
const MANIFEST_SHA256: &str = "2a349e612d9cbec271b25c2afdc82f28d79b29015f755b5efc4a98af7f3846d0";
const RENDERED_SHA256: &str = "59f6494df194686aec3493a496ebd970caee488d13a0e706626fe63c9351d945";
const TOKEN_IDS_SHA256: &str = "6e8203c02198324845eb104f5805adcb7eb0de27bcfb597dfcc2b2708a7f6814";
const PRIOR_COMMIT: &str = "c0b8289630b04390b7688174faac79dffb47f4ab";
const PRIOR_SOURCE_STATE: &str =
    "git-source-sha256-v2:4499acefdf7a548048b2dfb58b3e0f364ce52eaec9137c488a39af5af7dcf334";
const PROMPT_TOKENS: usize = 156;
const MAX_GENERATED_TOKENS: usize = 84;
const SEQUENCE_CAPACITY: usize = PROMPT_TOKENS + MAX_GENERATED_TOKENS;
const HIDDEN: usize = 2_048;
const VOCAB: usize = 248_320;
const OUTPUT_WEIGHT_BYTES: usize = 417_177_600;
const ROW_BYTES: usize = 1_680;
const OUTPUT_GUARD_ELEMENTS: usize = 64;
const OUTPUT_MAX_WIDTH: usize = 17;
const OUTPUT_GUARD_VALUE: f32 = -1234.5;
const OUTPUT_POISON_BITS: u32 = 0x7fc0_1234;
const BANK_POISON: u8 = 0xa5;
const HIDDEN_SEEDS: [u64; 4] = [1, 2, 3, 4];
const CONFORMANCE_SEEDS: [u64; 4] = [0, 1, 42, u64::MAX];
const WARMUP_PAIRS: usize = 5;
const SCORED_PAIRS: usize = 12;

#[derive(Debug)]
struct ArtifactInputFailure(String);

impl std::fmt::Display for ArtifactInputFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ArtifactInputFailure {}

#[derive(Parser, Debug)]
pub struct IntegratedGrammarRowArgs {
    #[arg(short = 'm', long)]
    model: PathBuf,
    #[arg(long)]
    prompt_fixture: PathBuf,
    #[arg(long)]
    runtime_artifact: PathBuf,
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    a3b_result: PathBuf,
    #[arg(long)]
    attest_no_other_user_gpu_workload: bool,
    #[arg(long, value_enum)]
    phase: ExecutionPhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ExecutionPhase {
    ConformanceOnly,
    Acquire,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptFixture {
    schema: String,
    fixture_id: String,
    source: PromptSource,
    messages: Vec<StrictChatMessage>,
    render: PromptRender,
    tokenizer: PromptTokenizer,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptSource {
    prompt_builder_path: String,
    prompt_builder_sha256: String,
    item_file_path: String,
    item_file_sha256: String,
    item_id: String,
    branch: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptRender {
    renderer: String,
    preserve_thinking: bool,
    append_generation_prompt: bool,
    bytes: usize,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptTokenizer {
    model_path: String,
    add_special: bool,
    token_count: usize,
    token_ids: Vec<i32>,
    token_ids_digest_schema: String,
    token_ids_sha256: String,
    qwen_tok_binary_sha256: String,
}

#[derive(Debug)]
struct ParsedPrompt {
    fixture: PromptFixture,
    rendered: String,
    token_ids: Vec<i32>,
}

#[derive(Clone, Debug, Serialize)]
struct PrerequisiteIdentity {
    path: PathBuf,
    sha256: String,
    profile: String,
    disposition: String,
    authority: String,
    build_commit: String,
    source_state: String,
}

#[derive(Clone, Debug, Serialize)]
struct ModelIdentity {
    path: PathBuf,
    file_bytes: u64,
    sha256: String,
    output_weight_bytes: usize,
    output_weight_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct PromptArtifactIdentity {
    path: PathBuf,
    complete_file_sha256: String,
    fixture_id: String,
    rendered_bytes: usize,
    rendered_sha256: String,
    token_count: usize,
    token_ids_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeArtifactIdentity {
    path: PathBuf,
    complete_file_sha256: String,
    semantic_sha256: String,
    states: u32,
    branch_states: u32,
    singleton_states: u32,
    terminal_states: u32,
    edges: u32,
    canonical_paths: u32,
    minimum_tokens: u32,
    maximum_tokens: u32,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestArtifactIdentity {
    path: PathBuf,
    complete_file_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct SamplerIdentity {
    algorithm_version: u32,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    scored_seed: u64,
}

#[derive(Clone, Debug, Serialize)]
struct ArtifactIdentity {
    prompt: PromptArtifactIdentity,
    runtime: RuntimeArtifactIdentity,
    manifest: ManifestArtifactIdentity,
    sampler: SamplerIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Arm {
    A,
    B,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::B => "b",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct TailEvidenceRecord {
    state_index: u32,
    state_kind: String,
    bank_kind: Option<String>,
    bank_state_index: Option<u32>,
    token_ids: Vec<i32>,
    tail_kind: String,
    n_in: usize,
    n_out: usize,
    weight_offset: u64,
    output_offset: u64,
    tail_dispatches: u32,
    full_head_dispatches: u32,
    command_completed: bool,
    command_error_none: bool,
    command_gpu_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
struct RequestTiming {
    prompt_read_parse_ms: f64,
    render_tokenize_validate_ms: f64,
    runtime_read_parse_validate_ms: f64,
    request_allocation_ms: f64,
    candidate_manifest_ms: f64,
    candidate_bank_prepare_ms: f64,
    candidate_upload_bind_ms: f64,
    prefill_command_ms: f64,
    prefill_gpu_ms: f64,
    root_readback_ms: f64,
    root_sample_grammar_callback_ms: f64,
    transition_command_ms: f64,
    transition_gpu_ms: f64,
    transition_readback_ms: f64,
    transition_sample_grammar_callback_ms: f64,
    commit_ms: f64,
    teardown_ms: f64,
    autorelease_pool_exit_ms: f64,
    phase_residual_ms: f64,
    total_request_ms: f64,
    ttft_ms: f64,
    generation_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
struct MemoryRecord {
    metal_allocated_bytes: u64,
    resident_bytes: u64,
    physical_footprint_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
struct RequestResult {
    arm: Arm,
    seed: u64,
    order_index: usize,
    scored: bool,
    timing: RequestTiming,
    output_token_ids: Vec<i32>,
    output_bytes: Vec<u8>,
    output_hex: String,
    state_indices: Vec<u32>,
    state_kinds: Vec<String>,
    pending_token: i32,
    generated_tokens: usize,
    transitions: usize,
    sampler_draws: usize,
    callback_count: usize,
    full_heads: u32,
    branch_heads: u32,
    singleton_heads: u32,
    terminal_heads: u32,
    bank_payload_bytes: usize,
    bank_physical_bytes: usize,
    branch_bank_sha256: Option<String>,
    singleton_bank_sha256: Option<String>,
    branch_upload_sha256: Option<String>,
    singleton_upload_sha256: Option<String>,
    branch_after_sha256: Option<String>,
    singleton_after_sha256: Option<String>,
    bank_upload_authenticated: Option<bool>,
    tail_records: Vec<TailEvidenceRecord>,
    memory: MemoryRecord,
}

#[derive(Clone, Debug, Serialize)]
struct PairResult {
    pair_index: usize,
    order: String,
    a: RequestResult,
    b: RequestResult,
    request_saving_ms: f64,
    ttft_delta_ms: f64,
    generation_saving_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
struct BankSeedEvidence {
    seed: u64,
    selected_logits_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct BankValidationEvidence {
    hidden_seeds: Vec<u64>,
    branch_states: usize,
    singleton_states: usize,
    compared_bank_logits: usize,
    full_dispatches: usize,
    compact_dispatches: usize,
    kernel: String,
    command_completed: bool,
    command_error_none: bool,
    per_seed: Vec<BankSeedEvidence>,
    branch_bank_sha256: String,
    singleton_bank_sha256: String,
    branch_upload_sha256: String,
    singleton_upload_sha256: String,
    branch_after_sha256: String,
    singleton_after_sha256: String,
    branch_source_rows_sha256: String,
    singleton_source_rows_sha256: String,
    bit_exact_selected_logits: bool,
    guards_intact: bool,
    padding_intact: bool,
    banks_unchanged: bool,
}

#[derive(Clone, Debug, Serialize)]
struct LockstepPositionEvidence {
    position_index: usize,
    state_index: u32,
    state_kind: String,
    token_ids: Vec<i32>,
    selected_logits_sha256: String,
    post_norm_hidden_sha256: String,
    model_state_sha256: String,
    sampled_local_token: usize,
    sampled_original_token: i32,
    candidate_index: usize,
    draws_before: usize,
    draws_after: usize,
    successor_state_index: u32,
    piece_sha256: String,
    callback_sha256: String,
    a_tail: TailEvidenceRecord,
    b_tail: TailEvidenceRecord,
}

#[derive(Clone, Debug, Serialize)]
struct LockstepSeedEvidence {
    seed: u64,
    positions: Vec<LockstepPositionEvidence>,
    output_token_ids: Vec<i32>,
    output_sha256: String,
    commit_state_sha256: String,
    pending_token: i32,
    generated_tokens: usize,
    transitions: usize,
    sampler_draws: usize,
    callback_count: usize,
    bit_exact: bool,
}

struct StateEqualityEvidence {
    post_norm_hidden_sha256: String,
    model_state_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct CorrectnessSummary {
    bank_validation: BankValidationEvidence,
    lockstep: Vec<LockstepSeedEvidence>,
}

#[derive(Clone, Debug, Serialize)]
struct Reduction {
    median_request_saving_ms: f64,
    positive_request_pairs: usize,
    median_ttft_delta_ms: f64,
    median_generation_saving_ms: f64,
    ab_median_request_saving_ms: f64,
    ba_median_request_saving_ms: f64,
    first_half_median_request_saving_ms: f64,
    second_half_median_request_saving_ms: f64,
    leave_one_out_request_medians_ms: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct Decision {
    disposition: String,
    authority: String,
    all_correctness: bool,
    environment_valid: bool,
    request_gate: bool,
    pair_win_gate: bool,
    generation_gate: bool,
    ttft_gate: bool,
}

#[derive(Debug, Serialize)]
struct IntegratedResult {
    schema: &'static str,
    test: &'static str,
    phase: &'static str,
    claim_scope: &'static str,
    build_identity: Value,
    qwen_env: BTreeMap<String, String>,
    user_gpu_attestation: bool,
    prerequisite: PrerequisiteIdentity,
    model_identity: ModelIdentity,
    artifact_identity: ArtifactIdentity,
    device: String,
    protocol: Value,
    correctness: CorrectnessSummary,
    host_before_warmup: HostSnapshot,
    host_before_scored: HostSnapshot,
    host_after_scored: HostSnapshot,
    vm_before_warmup: VmCounters,
    vm_after_scored: VmCounters,
    vm_delta: VmDelta,
    progress: AcquisitionProgress,
    warmup_pairs: Vec<PairResult>,
    scored_pairs: Vec<PairResult>,
    reduction: Reduction,
    decision: Decision,
}

#[derive(Debug, Serialize)]
struct ConformanceOnlyResult<'a> {
    schema: &'static str,
    test: &'static str,
    phase: &'static str,
    disposition: &'static str,
    authority: &'static str,
    build_identity: &'a Value,
    qwen_env: &'a BTreeMap<String, String>,
    user_gpu_attestation: bool,
    prerequisite: &'a PrerequisiteIdentity,
    model_identity: &'a ModelIdentity,
    artifact_identity: &'a ArtifactIdentity,
    device: String,
    correctness: &'a CorrectnessSummary,
}

#[derive(Clone, Debug, Default, Serialize)]
struct AcquisitionProgress {
    scored_b_started: bool,
    completed_scored_arms: usize,
    completed_scored_pairs: usize,
}

#[derive(Debug, Serialize)]
struct TerminalFailure<'a> {
    schema: &'static str,
    test: &'static str,
    phase: &'static str,
    disposition: &'static str,
    authority: &'static str,
    failure_class: &'static str,
    error: String,
    progress: &'a AcquisitionProgress,
    build_identity: &'a Value,
    qwen_env: &'a BTreeMap<String, String>,
    user_gpu_attestation: bool,
    prerequisite: &'a PrerequisiteIdentity,
    model_identity: &'a ModelIdentity,
    artifact_identity: &'a ArtifactIdentity,
    correctness: &'a CorrectnessSummary,
    warmup_pairs: &'a [PairResult],
    scored_pairs: &'a [PairResult],
    host_before_warmup: &'a HostSnapshot,
    host_before_scored: &'a HostSnapshot,
    vm_before_warmup: &'a VmCounters,
    host_after_scored: Option<&'a HostSnapshot>,
    vm_after_scored: Option<&'a VmCounters>,
    vm_delta: Option<&'a VmDelta>,
}

struct CandidateState {
    weight: MetalTensor,
    output: MetalTensor,
}

struct CandidateArtifacts {
    _manifest_bytes: Vec<u8>,
    _manifest: FrozenBranchManifest,
    plan: CandidateBankPlan,
    prepared: PreparedCandidateBanks,
    branch_buffer: Buffer,
    singleton_buffer: Buffer,
    branch_upload_sha256: String,
    singleton_upload_sha256: String,
    compact_storage: MetalTensor,
    states: Vec<Option<CandidateState>>,
}

#[derive(Default)]
struct CandidateSetupTiming {
    manifest_ms: f64,
    prepare_ms: f64,
    upload_bind_ms: f64,
}

struct CallbackSink {
    bytes: Vec<u8>,
    calls: usize,
}

impl CallbackSink {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            calls: 0,
        }
    }

    fn emit(&mut self, piece: &[u8]) {
        self.bytes.extend_from_slice(piece);
        self.calls += 1;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SampleOutcome {
    original_token: i32,
    local_token: usize,
    candidate_index: usize,
    draws_before: usize,
    draws_after: usize,
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn artifact_read(path: &Path, label: &str) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|error| {
        ArtifactInputFailure(format!(
            "{label} read failed for {}: {error}",
            path.display()
        ))
        .into()
    })
}

fn artifact_parse<T>(result: Result<T>, label: &str) -> Result<T> {
    match result {
        Ok(value) => Ok(value),
        Err(error)
            if error
                .chain()
                .any(|cause| cause.downcast_ref::<serde_json::Error>().is_some()) =>
        {
            Err(ArtifactInputFailure(format!("{label} parse failed: {error:#}")).into())
        }
        Err(error) => Err(error),
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    let mut hasher = Sha256::new();
    loop {
        let count = reader
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn token_ids_digest(tokens: &[i32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"qwen-token-ids-i32le/v1\0");
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing into String cannot fail");
    }
    output
}

fn strict_pointer<'a>(value: &'a Value, pointer: &str) -> Result<&'a Value> {
    value
        .pointer(pointer)
        .with_context(|| format!("prerequisite lacks {pointer}"))
}

fn validate_prerequisite(path: &Path) -> Result<PrerequisiteIdentity> {
    let bytes = artifact_read(path, "prerequisite result")?;
    let digest = sha256_hex(&bytes);
    ensure!(
        digest == PREREQUISITE_SHA256,
        "prerequisite result hash mismatch"
    );
    let value: Value = artifact_parse(
        serde_json::from_slice(&bytes)
            .map_err(anyhow::Error::from)
            .context("parse prerequisite result"),
        "prerequisite result",
    )?;
    ensure!(
        strict_pointer(&value, "/schema")?.as_str() == Some("qwen-grammar-lm-head-row-floor/v1")
    );
    ensure!(strict_pointer(&value, "/profile/id")?.as_str() == Some("a3b"));
    ensure!(strict_pointer(&value, "/profile/architecture")?.as_str() == Some("moe"));
    ensure!(strict_pointer(&value, "/profile/file_bytes")?.as_u64() == Some(MODEL_BYTES));
    ensure!(strict_pointer(&value, "/profile/model_sha256")?.as_str() == Some(MODEL_SHA256));
    ensure!(strict_pointer(&value, "/profile/layers")?.as_u64() == Some(40));
    ensure!(strict_pointer(&value, "/profile/hidden")?.as_u64() == Some(HIDDEN as u64));
    ensure!(strict_pointer(&value, "/profile/vocab")?.as_u64() == Some(VOCAB as u64));
    ensure!(strict_pointer(&value, "/tensor_identity/name")?.as_str() == Some("output.weight"));
    ensure!(strict_pointer(&value, "/tensor_identity/dtype")?.as_str() == Some("Q6_K"));
    ensure!(
        strict_pointer(&value, "/tensor_identity/sha256")?.as_str() == Some(OUTPUT_WEIGHT_SHA256)
    );
    ensure!(strict_pointer(&value, "/manifest_identity/sha256")?.as_str() == Some(MANIFEST_SHA256));
    ensure!(strict_pointer(&value, "/disposition")?.as_str() == Some("go"));
    ensure!(
        strict_pointer(&value, "/authority")?.as_str() == Some("one_later_a3b_integrated_packet")
    );
    ensure!(strict_pointer(&value, "/gates/all_go_gates")?.as_bool() == Some(true));
    ensure!(strict_pointer(&value, "/correctness/bit_exact")?.as_bool() == Some(true));
    ensure!(
        strict_pointer(&value, "/post_timing_validation/bank_unchanged")?.as_bool() == Some(true)
    );
    ensure!(
        strict_pointer(&value, "/qwen_env")?
            .as_object()
            .is_some_and(serde_json::Map::is_empty),
        "prerequisite QWEN environment is not empty"
    );
    ensure!(strict_pointer(&value, "/protocol/warmup_rounds")?.as_u64() == Some(5));
    ensure!(strict_pointer(&value, "/protocol/screen_rounds")?.as_u64() == Some(12));
    ensure!(strict_pointer(&value, "/protocol/paired_rounds")?.as_u64() == Some(12));
    ensure!(strict_pointer(&value, "/build_identity/status")?.as_str() == Some("match"));
    ensure!(strict_pointer(&value, "/build_identity/build_commit")?.as_str() == Some(PRIOR_COMMIT));
    ensure!(
        strict_pointer(&value, "/build_identity/runtime_commit")?.as_str() == Some(PRIOR_COMMIT)
    );
    ensure!(
        strict_pointer(&value, "/build_identity/build_source_state")?.as_str()
            == Some(PRIOR_SOURCE_STATE)
    );
    ensure!(
        strict_pointer(&value, "/build_identity/runtime_source_state")?.as_str()
            == Some(PRIOR_SOURCE_STATE)
    );
    ensure!(strict_pointer(&value, "/build_identity/build_dirty")?.as_bool() == Some(false));
    ensure!(strict_pointer(&value, "/build_identity/runtime_dirty")?.as_bool() == Some(false));
    ensure!(
        strict_pointer(&value, "/build_identity/problems")?
            .as_array()
            .is_some_and(Vec::is_empty)
    );
    ensure!(
        strict_pointer(&value, "/build_identity/overrides")?
            .as_array()
            .is_some_and(Vec::is_empty)
    );
    Ok(PrerequisiteIdentity {
        path: path.to_path_buf(),
        sha256: digest,
        profile: "a3b".to_string(),
        disposition: "go".to_string(),
        authority: "one_later_a3b_integrated_packet".to_string(),
        build_commit: PRIOR_COMMIT.to_string(),
        source_state: PRIOR_SOURCE_STATE.to_string(),
    })
}

fn parse_prompt_fixture(path: &Path) -> Result<PromptFixture> {
    let bytes = artifact_read(path, "prompt fixture")?;
    ensure!(
        sha256_hex(&bytes) == PROMPT_FILE_SHA256,
        "prompt fixture hash mismatch"
    );
    let fixture: PromptFixture = artifact_parse(
        serde_json::from_slice(&bytes)
            .map_err(anyhow::Error::from)
            .context("parse prompt fixture"),
        "prompt fixture",
    )?;
    ensure!(fixture.schema == PROMPT_SCHEMA);
    ensure!(fixture.fixture_id == "response-shape-self-pred-rb005-a3b-v1");
    ensure!(fixture.source.item_id == "0.5-rb-005");
    ensure!(fixture.source.branch == "Q-self-pred");
    ensure!(fixture.source.prompt_builder_path == "/Users/tito/code/llm/self_traj_spike_rf.py");
    ensure!(
        fixture.source.prompt_builder_sha256
            == "1c7e3595a938aeb13676e85f4e7ff02431e667b3db466c8ced766f9c371a4740"
    );
    ensure!(
        fixture.source.item_file_path
            == "/Users/tito/code/llm/self_trajectory/spike_rf/prompts/spike_items.json"
    );
    ensure!(
        fixture.source.item_file_sha256
            == "16f0d08867198a00a1935fb58275552e40282e1197aca168626c4018a2fc6ed4"
    );
    ensure!(fixture.messages.len() == 2);
    ensure!(fixture.messages[0].role == "system");
    ensure!(fixture.messages[0].content == "You are a helpful assistant.");
    ensure!(fixture.messages[1].role == "user");
    ensure!(fixture.render.renderer == "qwen-cli/messages.rs render_qwen_messages_prompt");
    ensure!(!fixture.render.preserve_thinking);
    ensure!(fixture.render.append_generation_prompt);
    ensure!(fixture.render.bytes == 684);
    ensure!(fixture.render.sha256 == RENDERED_SHA256);
    ensure!(fixture.tokenizer.model_path == MODEL_PATH);
    ensure!(!fixture.tokenizer.add_special);
    ensure!(fixture.tokenizer.token_count == PROMPT_TOKENS);
    ensure!(fixture.tokenizer.token_ids.len() == PROMPT_TOKENS);
    ensure!(fixture.tokenizer.token_ids_digest_schema == "qwen-token-ids-i32le/v1");
    ensure!(fixture.tokenizer.token_ids_sha256 == TOKEN_IDS_SHA256);
    ensure!(
        fixture.tokenizer.qwen_tok_binary_sha256
            == "0ea250bf58c2640fbe44b5e35474b7cad8c61cb943867f6345169aa7275a4952"
    );

    Ok(fixture)
}

fn render_tokenize_prompt(
    fixture: PromptFixture,
    tokenizer: &NativeTokenizer,
) -> Result<ParsedPrompt> {
    let messages: Vec<ChatMessage> = fixture
        .messages
        .iter()
        .map(|message| ChatMessage {
            role: message.role.clone(),
            content: message.content.clone(),
            ..Default::default()
        })
        .collect();
    let rendered = render_qwen_messages_prompt(&messages, false, true);
    ensure!(rendered.len() == fixture.render.bytes);
    ensure!(sha256_hex(rendered.as_bytes()) == RENDERED_SHA256);
    let token_ids = tokenizer
        .encode(&rendered, false)
        .context("tokenize frozen prompt")?;
    ensure!(token_ids == fixture.tokenizer.token_ids);
    ensure!(token_ids_digest(&token_ids) == TOKEN_IDS_SHA256);
    Ok(ParsedPrompt {
        fixture,
        rendered,
        token_ids,
    })
}

fn parse_prompt(path: &Path, tokenizer: &NativeTokenizer) -> Result<ParsedPrompt> {
    render_tokenize_prompt(parse_prompt_fixture(path)?, tokenizer)
}

fn authenticate_model_path(path: &Path) -> Result<(PathBuf, String)> {
    let actual =
        std::fs::canonicalize(path).with_context(|| format!("canonicalize {}", path.display()))?;
    let expected =
        std::fs::canonicalize(MODEL_PATH).with_context(|| format!("canonicalize {MODEL_PATH}"))?;
    ensure!(
        actual == expected,
        "model path does not match frozen A3B asset"
    );
    let metadata =
        std::fs::metadata(&actual).with_context(|| format!("stat {}", actual.display()))?;
    ensure!(metadata.len() == MODEL_BYTES, "model byte length mismatch");
    let sha256 = sha256_file(&actual)?;
    ensure!(sha256 == MODEL_SHA256, "model SHA-256 mismatch");
    Ok((actual, sha256))
}

fn validate_loaded_model<'a>(loaded: &'a LoadedModel) -> Result<(&'a [u8], String)> {
    let bound = Model::from_gguf(loaded.gguf()).context("bind loaded A3B model")?;
    ensure!(loaded.gguf().shard_count() == 1);
    ensure!(
        bound.arch.kind == ArchKind::Moe
            && bound.arch.n_layer == 40
            && bound.arch.hidden_size as usize == HIDDEN
            && bound.arch.vocab_size as usize == VOCAB
    );
    ensure!(!bound.tied_embeddings);
    let descriptor = bound.lm_head;
    ensure!(
        descriptor.name == "output.weight"
            && descriptor.dtype == GgmlType::Q6_K
            && descriptor.shape.as_slice() == [HIDDEN as u64, VOCAB as u64]
            && descriptor.n_bytes as usize == OUTPUT_WEIGHT_BYTES
    );
    let source = loaded
        .gguf()
        .try_slice(descriptor)
        .context("read retained output.weight bytes")?;
    ensure!(source.len() == OUTPUT_WEIGHT_BYTES);
    let digest = sha256_hex(source);
    ensure!(
        digest == OUTPUT_WEIGHT_SHA256,
        "output.weight SHA-256 mismatch"
    );
    ensure!(HIDDEN / 256 * 210 == ROW_BYTES);
    Ok((source, digest))
}

fn authenticate_artifacts(
    prompt_path: &Path,
    runtime_path: &Path,
    manifest_path: &Path,
    tokenizer: &NativeTokenizer,
) -> Result<ArtifactIdentity> {
    let prompt = parse_prompt(prompt_path, tokenizer)?;
    let prompt_identity = PromptArtifactIdentity {
        path: prompt_path.to_path_buf(),
        complete_file_sha256: PROMPT_FILE_SHA256.to_string(),
        fixture_id: prompt.fixture.fixture_id.clone(),
        rendered_bytes: prompt.rendered.len(),
        rendered_sha256: sha256_hex(prompt.rendered.as_bytes()),
        token_count: prompt.token_ids.len(),
        token_ids_sha256: token_ids_digest(&prompt.token_ids),
    };

    let runtime_bytes = artifact_read(runtime_path, "runtime artifact")?;
    let runtime = artifact_parse(
        FrozenGrammarRuntime::parse_authenticated(&runtime_bytes),
        "runtime artifact",
    )?;
    let document = runtime.document();
    let runtime_identity = RuntimeArtifactIdentity {
        path: runtime_path.to_path_buf(),
        complete_file_sha256: sha256_hex(&runtime_bytes),
        semantic_sha256: document.semantic_runtime_table_sha256.clone(),
        states: document.counts.states,
        branch_states: document.counts.branch_states,
        singleton_states: document.counts.singleton_states,
        terminal_states: document.counts.terminal_states,
        edges: document.counts.edges,
        canonical_paths: document.counts.canonical_paths,
        minimum_tokens: document.path_bounds.minimum_root_to_terminal_tokens,
        maximum_tokens: document.path_bounds.maximum_root_to_terminal_tokens,
    };

    let manifest_bytes = artifact_read(manifest_path, "branch manifest")?;
    artifact_parse(
        FrozenBranchManifest::parse_authenticated(&manifest_bytes),
        "branch manifest",
    )?;
    let manifest_identity = ManifestArtifactIdentity {
        path: manifest_path.to_path_buf(),
        complete_file_sha256: sha256_hex(&manifest_bytes),
    };
    let sampler = SamplingConfig::qwen_chat(0);
    Ok(ArtifactIdentity {
        prompt: prompt_identity,
        runtime: runtime_identity,
        manifest: manifest_identity,
        sampler: SamplerIdentity {
            algorithm_version: SAMPLER_ALGORITHM_VERSION,
            temperature: sampler.temperature,
            top_k: sampler.top_k,
            top_p: sampler.top_p,
            min_p: sampler.min_p,
            scored_seed: sampler.seed,
        },
    })
}

fn tensor_byte_range(tensor: &MetalTensor) -> Result<(usize, usize)> {
    let start = usize::try_from(tensor.offset).context("tensor offset does not fit usize")?;
    let len = usize::try_from(tensor.n_bytes()).context("tensor size does not fit usize")?;
    let end = start.checked_add(len).context("tensor endpoint overflow")?;
    ensure!(
        end <= tensor.buffer.length(),
        "tensor exceeds its Metal buffer"
    );
    Ok((start, end))
}

fn write_f32_tensor(tensor: &MetalTensor, values: &[f32]) -> Result<()> {
    ensure!(tensor.is_writable() && tensor.dtype == GgmlType::F32);
    ensure!(usize::try_from(tensor.n_elements())? == values.len());
    let (start, end) = tensor_byte_range(tensor)?;
    ensure!(end - start == std::mem::size_of_val(values));
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            tensor.buffer.contents().as_ptr().cast::<u8>().add(start),
            end - start,
        );
    }
    Ok(())
}

fn read_f32_tensor(tensor: &MetalTensor) -> Result<Vec<f32>> {
    ensure!(tensor.dtype == GgmlType::F32);
    let count = usize::try_from(tensor.n_elements())?;
    let (start, end) = tensor_byte_range(tensor)?;
    ensure!(end - start == count * std::mem::size_of::<f32>());
    let values = unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(start)
                .cast::<f32>(),
            count,
        )
    };
    Ok(values.to_vec())
}

fn read_buffer_bytes(buffer: &Buffer, length: usize) -> Result<Vec<u8>> {
    ensure!(length <= buffer.length());
    let bytes =
        unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), length) };
    Ok(bytes.to_vec())
}

fn read_selected_logits(session: &MetalSession, token_ids: &[i32]) -> Result<Vec<f32>> {
    ensure!(session.logits.dtype == GgmlType::F32);
    let count = usize::try_from(session.logits.n_elements())?;
    ensure!(count == VOCAB);
    let (start, end) = tensor_byte_range(&session.logits)?;
    ensure!(end - start == VOCAB * std::mem::size_of::<f32>());
    let base = unsafe {
        session
            .logits
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(start)
            .cast::<f32>()
    };
    token_ids
        .iter()
        .map(|&token| {
            let index = usize::try_from(token).context("negative selected token")?;
            ensure!(index < VOCAB, "selected token exceeds vocabulary");
            Ok(unsafe { *base.add(index) })
        })
        .collect()
}

fn sample_local(sampler: &mut Sampler, logits: &[f32], token_ids: &[i32]) -> Result<SampleOutcome> {
    ensure!(!logits.is_empty() && logits.len() == token_ids.len());
    ensure!(token_ids.windows(2).all(|pair| pair[0] < pair[1]));
    ensure!(
        !logits.iter().all(|&value| value == f32::NEG_INFINITY),
        "all selected logits are negative infinity"
    );
    let draws_before = sampler.draws();
    let sampled = sampler.sample(logits).map_err(|error| match error {
        SamplingError::NanLogit { token } => {
            let original = token_ids.get(token).copied();
            anyhow::anyhow!("NaN selected logit local={token} original={original:?}")
        }
        other => anyhow::anyhow!(other),
    })?;
    let local_token = usize::try_from(sampled.token).context("negative local sampled token")?;
    let original_token = *token_ids
        .get(local_token)
        .context("local sampled token exceeds grammar candidates")?;
    let draws_after = sampler.draws();
    ensure!(
        draws_after == draws_before + 1,
        "sample did not consume one draw"
    );
    Ok(SampleOutcome {
        original_token,
        local_token,
        candidate_index: sampled.candidate_index,
        draws_before,
        draws_after,
    })
}

fn state_kind_label(kind: RuntimeStateKind) -> String {
    match kind {
        RuntimeStateKind::Branch => "branch",
        RuntimeStateKind::Singleton => "singleton",
        RuntimeStateKind::Terminal => "terminal",
    }
    .to_string()
}

fn bank_key_fields(key: Option<BankKey>) -> (Option<String>, Option<u32>) {
    match key {
        Some(BankKey::Branch(index)) => (Some("branch".to_string()), Some(index)),
        Some(BankKey::Singleton(index)) => (Some("singleton".to_string()), Some(index)),
        None => (None, None),
    }
}

fn tail_record(
    runtime: &FrozenGrammarRuntime,
    state: GrammarState,
    evidence: LmHeadTailEvidence,
    gpu_ms: f64,
) -> Result<TailEvidenceRecord> {
    let state_ref = runtime.state(state)?;
    let (bank_kind, bank_state_index) = bank_key_fields(state_ref.bank_key());
    Ok(TailEvidenceRecord {
        state_index: state.index(),
        state_kind: state_kind_label(state_ref.kind()),
        bank_kind,
        bank_state_index,
        token_ids: state_ref
            .candidates()
            .iter()
            .map(|edge| edge.token_id)
            .collect(),
        tail_kind: evidence.kind.label().to_string(),
        n_in: evidence.n_in,
        n_out: evidence.n_out,
        weight_offset: evidence.weight_offset,
        output_offset: evidence.output_offset,
        tail_dispatches: evidence.tail_dispatches,
        full_head_dispatches: evidence.full_head_dispatches,
        command_completed: evidence.command_completed,
        command_error_none: evidence.command_error_none,
        command_gpu_ms: gpu_ms,
    })
}

impl CandidateArtifacts {
    fn build(
        ctx: &MetalContext,
        runtime: &FrozenGrammarRuntime,
        manifest_path: &Path,
        output_weight: &[u8],
    ) -> Result<(Self, CandidateSetupTiming)> {
        let mut timing = CandidateSetupTiming::default();
        let phase = Instant::now();
        let manifest_bytes = artifact_read(manifest_path, "branch manifest")?;
        let manifest = artifact_parse(
            FrozenBranchManifest::parse_authenticated(&manifest_bytes),
            "branch manifest",
        )?;
        timing.manifest_ms = elapsed_ms(phase);

        let phase = Instant::now();
        let plan = CandidateBankPlan::build(runtime, &manifest)?;
        let prepared = plan.prepare_from_output_weight(output_weight)?;
        timing.prepare_ms = elapsed_ms(phase);

        let phase = Instant::now();
        let branch_buffer = ctx.buffer_from(&prepared.branch.bytes)?;
        let singleton_buffer = ctx.buffer_from(&prepared.singleton.bytes)?;
        let branch_uploaded = read_buffer_bytes(&branch_buffer, prepared.branch.bytes.len())?;
        let singleton_uploaded =
            read_buffer_bytes(&singleton_buffer, prepared.singleton.bytes.len())?;
        ensure!(branch_uploaded == prepared.branch.bytes);
        ensure!(singleton_uploaded == prepared.singleton.bytes);
        let branch_upload_sha256 = sha256_hex(&branch_uploaded);
        let singleton_upload_sha256 = sha256_hex(&singleton_uploaded);
        ensure!(branch_upload_sha256 == prepared.branch.sha256);
        ensure!(singleton_upload_sha256 == prepared.singleton.sha256);
        let mut initial = vec![OUTPUT_GUARD_VALUE; 2 * OUTPUT_GUARD_ELEMENTS + OUTPUT_MAX_WIDTH];
        initial[OUTPUT_GUARD_ELEMENTS..OUTPUT_GUARD_ELEMENTS + OUTPUT_MAX_WIDTH].fill(0.0);
        let compact_storage = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&initial),
            vec![initial.len() as u64],
            GgmlType::F32,
        )?;
        let mut states: Vec<Option<CandidateState>> = std::iter::repeat_with(|| None)
            .take(runtime.document().states.len())
            .collect();
        for runtime_state in &runtime.document().states {
            let state = GrammarState::from_index(runtime_state.state_index);
            let state_ref = runtime.state(state)?;
            let Some(key) = state_ref.bank_key() else {
                ensure!(state_ref.is_terminal());
                continue;
            };
            let state_plan = plan.state(key)?;
            ensure!(state_plan.runtime_state_index == state.index());
            ensure!(state_plan.token_ids.len() == state_ref.candidates().len());
            ensure!(state_plan.token_ids.len() <= OUTPUT_MAX_WIDTH);
            let buffer = match key {
                BankKey::Branch(_) => branch_buffer.clone(),
                BankKey::Singleton(_) => singleton_buffer.clone(),
            };
            let weight = MetalTensor::q6_k_row_bank_weight_view(
                buffer,
                u64::try_from(state_plan.offset)?,
                HIDDEN,
                state_plan.token_ids.len(),
            )?;
            let output = compact_storage.view_subrange(
                OUTPUT_GUARD_ELEMENTS as u64,
                vec![state_plan.token_ids.len() as u64],
            );
            ensure!(weight.n_bytes() as usize == state_plan.payload_bytes);
            ensure!(weight.offset as usize == state_plan.offset);
            ensure!(output.is_writable());
            let slot = states
                .get_mut(usize::try_from(state.index())?)
                .context("runtime state exceeds candidate table")?;
            ensure!(slot.is_none());
            *slot = Some(CandidateState { weight, output });
        }
        ensure!(states.iter().filter(|state| state.is_some()).count() == 580);
        timing.upload_bind_ms = elapsed_ms(phase);
        Ok((
            Self {
                _manifest_bytes: manifest_bytes,
                _manifest: manifest,
                plan,
                prepared,
                branch_buffer,
                singleton_buffer,
                branch_upload_sha256,
                singleton_upload_sha256,
                compact_storage,
                states,
            },
            timing,
        ))
    }

    fn state(&self, state: GrammarState) -> Result<&CandidateState> {
        self.states
            .get(usize::try_from(state.index())?)
            .and_then(Option::as_ref)
            .context("candidate state is absent")
    }

    fn tail(&self, state: GrammarState) -> Result<LmHeadTail<'_>> {
        let state = self.state(state)?;
        Ok(LmHeadTail::CompactQ6K {
            weight: &state.weight,
            output: &state.output,
        })
    }

    fn read_logits(&self, state: GrammarState) -> Result<Vec<f32>> {
        read_f32_tensor(&self.state(state)?.output)
    }

    fn validate_after(&self, output_weight: &[u8]) -> Result<(String, String)> {
        self.prepared
            .verify_output_weight_unchanged(output_weight)?;
        let branch_after =
            read_buffer_bytes(&self.branch_buffer, self.prepared.branch.bytes.len())?;
        let singleton_after =
            read_buffer_bytes(&self.singleton_buffer, self.prepared.singleton.bytes.len())?;
        ensure!(branch_after == self.prepared.branch.bytes);
        ensure!(singleton_after == self.prepared.singleton.bytes);
        let branch_after_sha256 = sha256_hex(&branch_after);
        let singleton_after_sha256 = sha256_hex(&singleton_after);
        ensure!(branch_after_sha256 == self.branch_upload_sha256);
        ensure!(singleton_after_sha256 == self.singleton_upload_sha256);
        let storage = read_f32_tensor(&self.compact_storage)?;
        let guard_bits = OUTPUT_GUARD_VALUE.to_bits();
        ensure!(
            storage[..OUTPUT_GUARD_ELEMENTS]
                .iter()
                .all(|value| value.to_bits() == guard_bits)
        );
        ensure!(
            storage[OUTPUT_GUARD_ELEMENTS + OUTPUT_MAX_WIDTH..]
                .iter()
                .all(|value| value.to_bits() == guard_bits)
        );
        ensure!(self.plan.branch.row_incidences + self.plan.singleton.row_incidences == 1_597);
        ensure!(
            self.prepared.branch.row_copies == self.plan.branch.row_incidences
                && self.prepared.singleton.row_copies == self.plan.singleton.row_incidences
        );
        ensure!(
            self.prepared.branch.source_rows_sha256.len() == 64
                && self.prepared.singleton.source_rows_sha256.len() == 64
        );
        for (plan, prepared) in [
            (&self.plan.branch, &self.prepared.branch),
            (&self.plan.singleton, &self.prepared.singleton),
        ] {
            for state in &plan.states {
                ensure!(
                    prepared.bytes
                        [state.offset + state.payload_bytes..state.offset + state.span_bytes]
                        .iter()
                        .all(|&byte| byte == BANK_POISON)
                );
            }
        }
        Ok((branch_after_sha256, singleton_after_sha256))
    }
}

struct AdvanceResult {
    successor: GrammarState,
    sampled: SampleOutcome,
    piece: Vec<u8>,
}

fn sample_advance_emit(
    runtime: &FrozenGrammarRuntime,
    tokenizer: &NativeTokenizer,
    state: GrammarState,
    logits: &[f32],
    sampler: &mut Sampler,
    sink: &mut CallbackSink,
    validate_piece: bool,
) -> Result<AdvanceResult> {
    let state_ref = runtime.state(state)?;
    ensure!(
        !state_ref.is_terminal(),
        "terminal state cannot execute a head"
    );
    let token_ids: Vec<i32> = state_ref
        .candidates()
        .iter()
        .map(|edge| edge.token_id)
        .collect();
    let sampled = sample_local(sampler, logits, &token_ids)?;
    ensure!(
        !runtime
            .document()
            .stop_token_ids
            .contains(&sampled.original_token),
        "grammar selected a declared stop token"
    );
    let (successor, piece) = runtime.advance(state, sampled.original_token)?;
    if validate_piece {
        ensure!(
            tokenizer.try_decode_piece_bytes_exact(sampled.original_token)? == piece,
            "runtime edge piece differs from tokenizer"
        );
    }
    sink.emit(piece);
    Ok(AdvanceResult {
        successor,
        sampled,
        piece: piece.to_vec(),
    })
}

fn request_tail<'a>(
    arm: Arm,
    state: GrammarState,
    candidate: Option<&'a CandidateArtifacts>,
) -> Result<LmHeadTail<'a>> {
    match arm {
        Arm::A => Ok(LmHeadTail::Resident),
        Arm::B => candidate
            .context("candidate artifacts are absent")?
            .tail(state),
    }
}

fn read_request_logits(
    arm: Arm,
    state: GrammarState,
    runtime: &FrozenGrammarRuntime,
    sequence: &Sequence,
    candidate: Option<&CandidateArtifacts>,
) -> Result<Vec<f32>> {
    let state_ref = runtime.state(state)?;
    let token_ids: Vec<i32> = state_ref
        .candidates()
        .iter()
        .map(|edge| edge.token_id)
        .collect();
    match arm {
        Arm::A => read_selected_logits(sequence.metal_session(), &token_ids),
        Arm::B => candidate
            .context("candidate artifacts are absent")?
            .read_logits(state),
    }
}

fn validate_tail_for_arm(arm: Arm, record: &TailEvidenceRecord) -> Result<()> {
    ensure!(record.command_completed && record.command_error_none);
    ensure!(record.tail_dispatches == 1);
    match arm {
        Arm::A => {
            ensure!(record.tail_kind == LmHeadTailKind::Resident.label());
            ensure!(record.n_in == HIDDEN && record.n_out == VOCAB);
            ensure!(record.full_head_dispatches == 1);
        }
        Arm::B => {
            ensure!(record.tail_kind == LmHeadTailKind::CompactQ6K.label());
            ensure!(record.n_in == HIDDEN && record.n_out == record.token_ids.len());
            ensure!(record.full_head_dispatches == 0);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_request(
    arm: Arm,
    seed: u64,
    order_index: usize,
    scored: bool,
    loaded: &LoadedModel,
    tokenizer: &NativeTokenizer,
    output_weight: &[u8],
    prompt_path: &Path,
    runtime_path: &Path,
    manifest_path: &Path,
    validate_piece: bool,
) -> Result<RequestResult> {
    let request_start = Instant::now();
    let (mut result, inside_request_ms) = autoreleasepool(|_| -> Result<(RequestResult, f64)> {
        let mut timing = RequestTiming::default();

        let phase = Instant::now();
        let prompt_fixture = parse_prompt_fixture(prompt_path)?;
        timing.prompt_read_parse_ms = elapsed_ms(phase);
        let phase = Instant::now();
        let parsed_prompt = render_tokenize_prompt(prompt_fixture, tokenizer)?;
        ensure!(parsed_prompt.rendered.len() == 684);
        ensure!(parsed_prompt.token_ids.len() == PROMPT_TOKENS);
        ensure!(parsed_prompt.fixture.tokenizer.token_count == PROMPT_TOKENS);
        timing.render_tokenize_validate_ms = elapsed_ms(phase);

        let phase = Instant::now();
        let runtime_bytes = artifact_read(runtime_path, "runtime artifact")?;
        let runtime = artifact_parse(
            FrozenGrammarRuntime::parse_authenticated(&runtime_bytes),
            "runtime artifact",
        )?;
        ensure!(
            runtime
                .document()
                .path_bounds
                .maximum_root_to_terminal_tokens as usize
                == MAX_GENERATED_TOKENS
        );
        timing.runtime_read_parse_validate_ms = elapsed_ms(phase);

        let phase = Instant::now();
        let mut sequence = loaded.create_sequence(SequenceConfig::new(SEQUENCE_CAPACITY))?;
        let mut scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
            loaded.context(),
            loaded.metal_model(),
            PROMPT_TOKENS as u32,
            PROMPT_TOKENS,
        )?;
        ensure!(scratch.n as usize == PROMPT_TOKENS);
        let mut sampler = Sampler::new(SamplingConfig::qwen_chat(seed))?;
        let mut sink = CallbackSink::new();
        let mut state = runtime.root();
        let mut generated = Vec::new();
        let mut state_indices = vec![state.index()];
        let mut state_kinds = vec![state_kind_label(runtime.state(state)?.kind())];
        let mut tail_records = Vec::new();
        timing.request_allocation_ms = elapsed_ms(phase);

        let (candidate, candidate_timing) = match arm {
            Arm::A => (None, CandidateSetupTiming::default()),
            Arm::B => {
                let (candidate, timing) = CandidateArtifacts::build(
                    loaded.context(),
                    &runtime,
                    manifest_path,
                    output_weight,
                )?;
                (Some(candidate), timing)
            }
        };
        timing.candidate_manifest_ms = candidate_timing.manifest_ms;
        timing.candidate_bank_prepare_ms = candidate_timing.prepare_ms;
        timing.candidate_upload_bind_ms = candidate_timing.upload_bind_ms;

        let forward = loaded.forward();
        let phase = Instant::now();
        let root_tail = request_tail(arm, state, candidate.as_ref())?;
        let (prefill_gpu_ms, evidence) = unsafe {
            prefill_tokens_profiled_with_tail(
                &forward,
                &parsed_prompt.token_ids,
                0,
                sequence.metal_session_mut(),
                &mut scratch,
                root_tail,
            )
        }?;
        sequence.advance_by(PROMPT_TOKENS)?;
        timing.prefill_command_ms = elapsed_ms(phase);
        timing.prefill_gpu_ms = prefill_gpu_ms;
        let record = tail_record(&runtime, state, evidence, prefill_gpu_ms)?;
        validate_tail_for_arm(arm, &record)?;
        tail_records.push(record);

        let phase = Instant::now();
        let root_logits = read_request_logits(arm, state, &runtime, &sequence, candidate.as_ref())?;
        timing.root_readback_ms = elapsed_ms(phase);
        let phase = Instant::now();
        let first = sample_advance_emit(
            &runtime,
            tokenizer,
            state,
            &root_logits,
            &mut sampler,
            &mut sink,
            validate_piece,
        )?;
        generated.push(first.sampled.original_token);
        state = first.successor;
        state_indices.push(state.index());
        state_kinds.push(state_kind_label(runtime.state(state)?.kind()));
        timing.root_sample_grammar_callback_ms = elapsed_ms(phase);
        let first_callback_elapsed = request_start.elapsed();
        timing.ttft_ms = first_callback_elapsed.as_secs_f64() * 1e3;
        drop(root_logits);
        drop(first);

        let mut transitions = 0usize;
        while !runtime.state(state)?.is_terminal() {
            ensure!(generated.len() < MAX_GENERATED_TOKENS);
            let input_token = *generated.last().context("generated token list is empty")?;
            let position = u32::try_from(sequence.position())?;
            let tail = request_tail(arm, state, candidate.as_ref())?;
            let phase = Instant::now();
            let (profile, evidence) = unsafe {
                forward.single_token_profiled_concurrent_gdn_moe_with_tail(
                    input_token,
                    position,
                    sequence.metal_session_mut(),
                    tail,
                )
            }?;
            sequence.advance_by(1)?;
            transitions += 1;
            timing.transition_command_ms += elapsed_ms(phase);
            timing.transition_gpu_ms += profile.gpu_kernel_ms;
            let record = tail_record(&runtime, state, evidence, profile.gpu_kernel_ms)?;
            validate_tail_for_arm(arm, &record)?;
            tail_records.push(record);

            let phase = Instant::now();
            let logits = read_request_logits(arm, state, &runtime, &sequence, candidate.as_ref())?;
            timing.transition_readback_ms += elapsed_ms(phase);
            let phase = Instant::now();
            let next = sample_advance_emit(
                &runtime,
                tokenizer,
                state,
                &logits,
                &mut sampler,
                &mut sink,
                validate_piece,
            )?;
            generated.push(next.sampled.original_token);
            state = next.successor;
            state_indices.push(state.index());
            state_kinds.push(state_kind_label(runtime.state(state)?.kind()));
            timing.transition_sample_grammar_callback_ms += elapsed_ms(phase);
        }

        let commit_start = Instant::now();
        ensure!(runtime.state(state)?.is_terminal());
        ensure!(runtime.state(state)?.candidates().is_empty());
        ensure!(!generated.is_empty());
        ensure!(generated.len() == transitions + 1);
        ensure!(tail_records.len() == generated.len());
        ensure!(sampler.draws() == generated.len());
        ensure!(sink.calls == generated.len());
        ensure!(state_indices.len() == generated.len() + 1);
        ensure!(generated.len() <= MAX_GENERATED_TOKENS);
        ensure!(
            sequence.position() == PROMPT_TOKENS + transitions,
            "logical sequence position disagrees with consumed tokens"
        );
        let pending_token = *generated.last().context("missing pending token")?;
        let terminal_commit_elapsed = request_start.elapsed();
        timing.generation_ms = terminal_commit_elapsed
            .checked_sub(first_callback_elapsed)
            .context("generation clock underflow")?
            .as_secs_f64()
            * 1e3;
        let full_heads: u32 = tail_records
            .iter()
            .map(|record| record.full_head_dispatches)
            .sum();
        let branch_heads = tail_records
            .iter()
            .filter(|record| record.state_kind == "branch")
            .count() as u32;
        let singleton_heads = tail_records
            .iter()
            .filter(|record| record.state_kind == "singleton")
            .count() as u32;
        ensure!(branch_heads + singleton_heads == generated.len() as u32);
        match arm {
            Arm::A => ensure!(full_heads == generated.len() as u32),
            Arm::B => ensure!(full_heads == 0),
        }
        let (branch_after_sha256, singleton_after_sha256) = match candidate.as_ref() {
            Some(candidate) => {
                let (branch, singleton) = candidate.validate_after(output_weight)?;
                (Some(branch), Some(singleton))
            }
            None => (None, None),
        };

        let output_token_ids = generated.clone();
        let output_bytes = sink.bytes.clone();
        let output_hex = hex_bytes(&output_bytes);
        let callback_count = sink.calls;
        let sampler_draws = sampler.draws();
        let bank_payload_bytes = candidate.as_ref().map_or(0, |candidate| {
            candidate.plan.branch.payload_bytes + candidate.plan.singleton.payload_bytes
        });
        let bank_physical_bytes = candidate.as_ref().map_or(0, |candidate| {
            candidate.plan.branch.span_bytes + candidate.plan.singleton.span_bytes
        });
        let branch_bank_sha256 = candidate
            .as_ref()
            .map(|candidate| candidate.prepared.branch.sha256.clone());
        let singleton_bank_sha256 = candidate
            .as_ref()
            .map(|candidate| candidate.prepared.singleton.sha256.clone());
        let branch_upload_sha256 = candidate
            .as_ref()
            .map(|candidate| candidate.branch_upload_sha256.clone());
        let singleton_upload_sha256 = candidate
            .as_ref()
            .map(|candidate| candidate.singleton_upload_sha256.clone());
        let bank_upload_authenticated = candidate.as_ref().map(|candidate| {
            candidate.branch_upload_sha256 == candidate.prepared.branch.sha256
                && candidate.singleton_upload_sha256 == candidate.prepared.singleton.sha256
        });
        timing.commit_ms = elapsed_ms(commit_start);

        let teardown_start = Instant::now();
        drop(candidate);
        drop(sampler);
        drop(sink);
        drop(scratch);
        drop(sequence);
        drop(forward);
        drop(runtime);
        drop(runtime_bytes);
        drop(parsed_prompt);
        let generated_tokens = generated.len();
        drop(generated);
        timing.teardown_ms = elapsed_ms(teardown_start);

        let result = RequestResult {
            arm,
            seed,
            order_index,
            scored,
            timing,
            output_token_ids,
            output_bytes,
            output_hex,
            state_indices,
            state_kinds,
            pending_token,
            generated_tokens,
            transitions,
            sampler_draws,
            callback_count,
            full_heads,
            branch_heads,
            singleton_heads,
            terminal_heads: 0,
            bank_payload_bytes,
            bank_physical_bytes,
            branch_bank_sha256,
            singleton_bank_sha256,
            branch_upload_sha256,
            singleton_upload_sha256,
            branch_after_sha256,
            singleton_after_sha256,
            bank_upload_authenticated,
            tail_records,
            memory: MemoryRecord {
                metal_allocated_bytes: 0,
                resident_bytes: 0,
                physical_footprint_bytes: 0,
            },
        };
        Ok((result, elapsed_ms(request_start)))
    })?;
    result.timing.total_request_ms = elapsed_ms(request_start);
    result.timing.autorelease_pool_exit_ms = result.timing.total_request_ms - inside_request_ms;
    let additive_phase_ms = result.timing.prompt_read_parse_ms
        + result.timing.render_tokenize_validate_ms
        + result.timing.runtime_read_parse_validate_ms
        + result.timing.request_allocation_ms
        + result.timing.candidate_manifest_ms
        + result.timing.candidate_bank_prepare_ms
        + result.timing.candidate_upload_bind_ms
        + result.timing.prefill_command_ms
        + result.timing.root_readback_ms
        + result.timing.root_sample_grammar_callback_ms
        + result.timing.transition_command_ms
        + result.timing.transition_readback_ms
        + result.timing.transition_sample_grammar_callback_ms
        + result.timing.commit_ms
        + result.timing.teardown_ms
        + result.timing.autorelease_pool_exit_ms;
    result.timing.phase_residual_ms = result.timing.total_request_ms - additive_phase_ms;
    let process = PidSnapshot::now().context("capture post-request process memory")?;
    result.memory = MemoryRecord {
        metal_allocated_bytes: loaded.context().current_allocated_size(),
        resident_bytes: process.resident_size,
        physical_footprint_bytes: process.phys_footprint,
    };
    Ok(result)
}

fn compare_request_outputs(a: &RequestResult, b: &RequestResult) -> Result<()> {
    ensure!(a.arm == Arm::A && b.arm == Arm::B);
    ensure!(a.seed == b.seed);
    ensure!(a.output_token_ids == b.output_token_ids);
    ensure!(a.output_bytes == b.output_bytes);
    ensure!(a.output_hex == b.output_hex);
    ensure!(a.state_indices == b.state_indices);
    ensure!(a.state_kinds == b.state_kinds);
    ensure!(a.pending_token == b.pending_token);
    ensure!(a.generated_tokens == b.generated_tokens);
    ensure!(a.transitions == b.transitions);
    ensure!(a.sampler_draws == b.sampler_draws);
    ensure!(a.callback_count == b.callback_count);
    ensure!(b.full_heads == 0 && b.terminal_heads == 0 && a.terminal_heads == 0);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_pair(
    pair_index: usize,
    scored: bool,
    loaded: &LoadedModel,
    tokenizer: &NativeTokenizer,
    output_weight: &[u8],
    prompt_path: &Path,
    runtime_path: &Path,
    manifest_path: &Path,
    mut progress: Option<&mut AcquisitionProgress>,
) -> Result<PairResult> {
    let order = pair_order(pair_index);
    let mut arms = Vec::with_capacity(2);
    for (order_index, arm) in order.into_iter().enumerate() {
        if scored
            && arm == Arm::B
            && let Some(progress) = progress.as_deref_mut()
        {
            progress.scored_b_started = true;
        }
        let result = run_request(
            arm,
            0,
            order_index,
            scored,
            loaded,
            tokenizer,
            output_weight,
            prompt_path,
            runtime_path,
            manifest_path,
            false,
        )?;
        if scored && let Some(progress) = progress.as_deref_mut() {
            progress.completed_scored_arms += 1;
        }
        arms.push(result);
    }
    let a_index = arms
        .iter()
        .position(|result| result.arm == Arm::A)
        .context("pair lacks A arm")?;
    let b_index = arms
        .iter()
        .position(|result| result.arm == Arm::B)
        .context("pair lacks B arm")?;
    let a = arms.remove(a_index);
    let adjusted_b_index = if b_index > a_index {
        b_index - 1
    } else {
        b_index
    };
    let b = arms.remove(adjusted_b_index);
    compare_request_outputs(&a, &b)?;
    if scored && let Some(progress) = progress.as_deref_mut() {
        progress.completed_scored_pairs += 1;
    }
    let request_saving_ms = a.timing.total_request_ms - b.timing.total_request_ms;
    let ttft_delta_ms = b.timing.ttft_ms - a.timing.ttft_ms;
    let generation_saving_ms = a.timing.generation_ms - b.timing.generation_ms;
    Ok(PairResult {
        pair_index,
        order: format!(
            "{}{}",
            order[0].label().to_ascii_uppercase(),
            order[1].label().to_ascii_uppercase()
        ),
        a,
        b,
        request_saving_ms,
        ttft_delta_ms,
        generation_saving_ms,
    })
}

fn pair_order(pair_index: usize) -> [Arm; 2] {
    if pair_index % 2 == 0 {
        [Arm::A, Arm::B]
    } else {
        [Arm::B, Arm::A]
    }
}

fn hidden_vector(seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..HIDDEN)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((((state >> 40) & 0xffff) as i32) - 32_768) as f32 / 32_768.0
        })
        .collect()
}

fn check_command(command: &ProtocolObject<dyn MTLCommandBuffer>, label: &str) -> Result<()> {
    let status = command.status();
    let error = command.error();
    ensure!(
        status == MTLCommandBufferStatus::Completed && error.is_none(),
        "{label} failed: status={status:?} error={error:?}"
    );
    Ok(())
}

fn run_q6_dispatch(
    ctx: &MetalContext,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_out: usize,
    label: &str,
) -> Result<()> {
    let command = ctx.queue.commandBuffer().context("create Q6_K command")?;
    let encoder = KernelEncoder::begin(&command);
    encode_mat_vec_q6_k_f32(ctx, &encoder, weight, input, output, HIDDEN, n_out)?;
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    check_command(&command, label)
}

fn write_compact_poison(storage: &MetalTensor, width: usize) -> Result<()> {
    ensure!(width <= OUTPUT_MAX_WIDTH);
    let mut values = vec![OUTPUT_GUARD_VALUE; 2 * OUTPUT_GUARD_ELEMENTS + OUTPUT_MAX_WIDTH];
    values[OUTPUT_GUARD_ELEMENTS..OUTPUT_GUARD_ELEMENTS + width]
        .fill(f32::from_bits(OUTPUT_POISON_BITS));
    write_f32_tensor(storage, &values)
}

fn validate_compact_output(storage: &MetalTensor, width: usize) -> Result<()> {
    let values = read_f32_tensor(storage)?;
    let guard_bits = OUTPUT_GUARD_VALUE.to_bits();
    ensure!(
        values[..OUTPUT_GUARD_ELEMENTS]
            .iter()
            .all(|value| value.to_bits() == guard_bits)
    );
    ensure!(
        values[OUTPUT_GUARD_ELEMENTS..OUTPUT_GUARD_ELEMENTS + width]
            .iter()
            .all(|value| value.to_bits() != OUTPUT_POISON_BITS)
    );
    ensure!(
        values
            [OUTPUT_GUARD_ELEMENTS + width..OUTPUT_GUARD_ELEMENTS + width + OUTPUT_GUARD_ELEMENTS]
            .iter()
            .all(|value| value.to_bits() == guard_bits)
    );
    Ok(())
}

fn selected_logits_digest(token_ids: &[i32], logits: &[f32]) -> Result<String> {
    ensure!(token_ids.len() == logits.len());
    let mut hasher = Sha256::new();
    hasher.update(b"qwen-selected-logits/v1\0");
    hasher.update((token_ids.len() as u64).to_le_bytes());
    for (&token, &logit) in token_ids.iter().zip(logits) {
        hasher.update(token.to_le_bytes());
        hasher.update(logit.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_all_bank_views(
    loaded: &LoadedModel,
    runtime_path: &Path,
    manifest_path: &Path,
    output_weight: &[u8],
) -> Result<BankValidationEvidence> {
    let runtime_bytes = artifact_read(runtime_path, "runtime artifact")?;
    let runtime = artifact_parse(
        FrozenGrammarRuntime::parse_authenticated(&runtime_bytes),
        "runtime artifact",
    )?;
    let (candidate, _) =
        CandidateArtifacts::build(loaded.context(), &runtime, manifest_path, output_weight)?;
    let full_output = MetalTensor::zeros_f32(loaded.context(), vec![VOCAB as u64])?;
    let full_poison = vec![f32::from_bits(OUTPUT_POISON_BITS); VOCAB];
    let mut compared = 0usize;
    let mut branch_states = 0usize;
    let mut singleton_states = 0usize;
    let mut per_seed = Vec::with_capacity(HIDDEN_SEEDS.len());

    for seed in HIDDEN_SEEDS {
        let mut seed_hasher = Sha256::new();
        seed_hasher.update(b"qwen-bank-selected-logits/v1\0");
        seed_hasher.update(seed.to_le_bytes());
        let hidden = MetalTensor::from_bytes(
            loaded.context(),
            bytemuck::cast_slice(&hidden_vector(seed)),
            vec![HIDDEN as u64],
            GgmlType::F32,
        )?;
        write_f32_tensor(&full_output, &full_poison)?;
        run_q6_dispatch(
            loaded.context(),
            &loaded.metal_model().lm_head,
            &hidden,
            &full_output,
            VOCAB,
            "full head bank validation",
        )?;
        let full_logits = read_f32_tensor(&full_output)?;
        ensure!(
            full_logits
                .iter()
                .all(|value| value.to_bits() != OUTPUT_POISON_BITS)
        );

        for runtime_state in &runtime.document().states {
            let state = GrammarState::from_index(runtime_state.state_index);
            let state_ref = runtime.state(state)?;
            if state_ref.is_terminal() {
                ensure!(candidate.states[usize::try_from(state.index())?].is_none());
                continue;
            }
            match state_ref.kind() {
                RuntimeStateKind::Branch if seed == HIDDEN_SEEDS[0] => branch_states += 1,
                RuntimeStateKind::Singleton if seed == HIDDEN_SEEDS[0] => singleton_states += 1,
                RuntimeStateKind::Terminal => unreachable!(),
                _ => {}
            }
            let candidate_state = candidate.state(state)?;
            let width = state_ref.candidates().len();
            write_compact_poison(&candidate.compact_storage, width)?;
            run_q6_dispatch(
                loaded.context(),
                &candidate_state.weight,
                &hidden,
                &candidate_state.output,
                width,
                "compact bank validation",
            )?;
            validate_compact_output(&candidate.compact_storage, width)?;
            let compact = read_f32_tensor(&candidate_state.output)?;
            ensure!(compact.len() == width);
            for (local, edge) in state_ref.candidates().iter().enumerate() {
                let token = usize::try_from(edge.token_id)?;
                ensure!(
                    compact[local].to_bits() == full_logits[token].to_bits(),
                    "bank logit mismatch seed={seed} state={} local={local}",
                    state.index()
                );
                seed_hasher.update(state.index().to_le_bytes());
                seed_hasher.update(edge.token_id.to_le_bytes());
                seed_hasher.update(full_logits[token].to_bits().to_le_bytes());
                seed_hasher.update(compact[local].to_bits().to_le_bytes());
                compared += 1;
            }
        }
        per_seed.push(BankSeedEvidence {
            seed,
            selected_logits_sha256: format!("{:x}", seed_hasher.finalize()),
        });
    }
    let branch_bank_sha256 = candidate.prepared.branch.sha256.clone();
    let singleton_bank_sha256 = candidate.prepared.singleton.sha256.clone();
    let branch_upload_sha256 = candidate.branch_upload_sha256.clone();
    let singleton_upload_sha256 = candidate.singleton_upload_sha256.clone();
    let branch_source_rows_sha256 = candidate.prepared.branch.source_rows_sha256.clone();
    let singleton_source_rows_sha256 = candidate.prepared.singleton.source_rows_sha256.clone();
    let (branch_after_sha256, singleton_after_sha256) = candidate.validate_after(output_weight)?;
    ensure!(branch_states == 451 && singleton_states == 129 && compared == 1_597 * 4);
    Ok(BankValidationEvidence {
        hidden_seeds: HIDDEN_SEEDS.to_vec(),
        branch_states,
        singleton_states,
        compared_bank_logits: compared,
        full_dispatches: HIDDEN_SEEDS.len(),
        compact_dispatches: (branch_states + singleton_states) * HIDDEN_SEEDS.len(),
        kernel: "kernel_mat_vec_q6_K_f32".to_string(),
        command_completed: true,
        command_error_none: true,
        per_seed,
        branch_bank_sha256,
        singleton_bank_sha256,
        branch_upload_sha256,
        singleton_upload_sha256,
        branch_after_sha256,
        singleton_after_sha256,
        branch_source_rows_sha256,
        singleton_source_rows_sha256,
        bit_exact_selected_logits: true,
        guards_intact: true,
        padding_intact: true,
        banks_unchanged: true,
    })
}

fn compare_f32_bits(left: &[f32], right: &[f32], label: &str) -> Result<()> {
    ensure!(left.len() == right.len(), "{label} length mismatch");
    ensure!(
        left.iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits()),
        "{label} differs"
    );
    Ok(())
}

fn snapshot_for_compare(
    sequence: &Sequence,
    consumed_tokens: &[i32],
    pending_token: Option<i32>,
) -> Result<SessionSnapshot> {
    ensure!(sequence.position() == consumed_tokens.len());
    let identity = sequence.metal_session().snapshot_identity(0x656, 0x656);
    let mut snapshot =
        sequence
            .metal_session()
            .snapshot(identity, consumed_tokens.to_vec(), None)?;
    snapshot.pending_token = pending_token;
    Ok(snapshot)
}

fn compare_snapshots(left: &SessionSnapshot, right: &SessionSnapshot) -> Result<()> {
    ensure!(left.identity == right.identity);
    ensure!(left.prefix_tokens == right.prefix_tokens);
    ensure!(left.pending_token == right.pending_token);
    ensure!(left.kv_n_pos == right.kv_n_pos);
    ensure!(left.kv_k_arena == right.kv_k_arena);
    ensure!(left.kv_v_arena == right.kv_v_arena);
    ensure!(left.gdn_conv_arena == right.gdn_conv_arena);
    ensure!(left.gdn_state_arena == right.gdn_state_arena);
    ensure!(left.final_logits.is_none() && right.final_logits.is_none());
    Ok(())
}

fn f32_bits_digest(domain: &[u8], values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn hash_section(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn snapshot_digest(snapshot: &SessionSnapshot) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"qwen-session-state/v1\0");
    let identity = &snapshot.identity;
    hasher.update(identity.model_id.to_le_bytes());
    hasher.update(identity.tokenizer_id.to_le_bytes());
    hasher.update(identity.layout_version.to_le_bytes());
    hasher.update(identity.n_attn_layers.to_le_bytes());
    hasher.update(identity.n_gdn_layers.to_le_bytes());
    hasher.update(identity.kv_dim_elements.to_le_bytes());
    hasher.update(identity.kv_bytes_per_token.to_le_bytes());
    hasher.update((identity.kv_storage_kind as u32).to_le_bytes());
    hasher.update(identity.gdn_state_elements_per_layer.to_le_bytes());
    hasher.update(identity.gdn_conv_elements_per_layer.to_le_bytes());
    hasher.update((snapshot.prefix_tokens.len() as u64).to_le_bytes());
    for token in &snapshot.prefix_tokens {
        hasher.update(token.to_le_bytes());
    }
    match snapshot.pending_token {
        None => hasher.update([0]),
        Some(token) => {
            hasher.update([1]);
            hasher.update(token.to_le_bytes());
        }
    }
    hasher.update((snapshot.kv_n_pos.len() as u64).to_le_bytes());
    for position in &snapshot.kv_n_pos {
        hasher.update((*position as u64).to_le_bytes());
    }
    hash_section(&mut hasher, &snapshot.kv_k_arena);
    hash_section(&mut hasher, &snapshot.kv_v_arena);
    hash_section(&mut hasher, &snapshot.gdn_conv_arena);
    hash_section(&mut hasher, &snapshot.gdn_state_arena);
    format!("{:x}", hasher.finalize())
}

fn compare_sequence_state(
    a: &Sequence,
    b: &Sequence,
    consumed_tokens: &[i32],
    pending_token: Option<i32>,
) -> Result<StateEqualityEvidence> {
    ensure!(a.position() == b.position());
    let a_hidden = read_f32_tensor(&a.metal_session().h)?;
    let b_hidden = read_f32_tensor(&b.metal_session().h)?;
    compare_f32_bits(&a_hidden, &b_hidden, "post-norm hidden")?;
    let a_hidden_sha256 = f32_bits_digest(b"qwen-post-norm-hidden/v1\0", &a_hidden);
    let b_hidden_sha256 = f32_bits_digest(b"qwen-post-norm-hidden/v1\0", &b_hidden);
    ensure!(a_hidden_sha256 == b_hidden_sha256);
    let a_snapshot = snapshot_for_compare(a, consumed_tokens, pending_token)?;
    let b_snapshot = snapshot_for_compare(b, consumed_tokens, pending_token)?;
    compare_snapshots(&a_snapshot, &b_snapshot)?;
    let a_state_sha256 = snapshot_digest(&a_snapshot);
    let b_state_sha256 = snapshot_digest(&b_snapshot);
    ensure!(a_state_sha256 == b_state_sha256);
    Ok(StateEqualityEvidence {
        post_norm_hidden_sha256: a_hidden_sha256,
        model_state_sha256: a_state_sha256,
    })
}

fn validate_evidence_pair(
    runtime: &FrozenGrammarRuntime,
    state: GrammarState,
    a: LmHeadTailEvidence,
    b: LmHeadTailEvidence,
    a_command_gpu_ms: f64,
    b_command_gpu_ms: f64,
) -> Result<(TailEvidenceRecord, TailEvidenceRecord)> {
    let a_record = tail_record(runtime, state, a, a_command_gpu_ms)?;
    let b_record = tail_record(runtime, state, b, b_command_gpu_ms)?;
    validate_tail_for_arm(Arm::A, &a_record)?;
    validate_tail_for_arm(Arm::B, &b_record)?;
    ensure!(a_record.token_ids == b_record.token_ids);
    Ok((a_record, b_record))
}

fn run_lockstep_seed(
    seed: u64,
    loaded: &LoadedModel,
    tokenizer: &NativeTokenizer,
    output_weight: &[u8],
    prompt_path: &Path,
    runtime_path: &Path,
    manifest_path: &Path,
) -> Result<LockstepSeedEvidence> {
    let prompt = parse_prompt(prompt_path, tokenizer)?;
    let runtime_bytes = artifact_read(runtime_path, "runtime artifact")?;
    let runtime = artifact_parse(
        FrozenGrammarRuntime::parse_authenticated(&runtime_bytes),
        "runtime artifact",
    )?;
    let (candidate, _) =
        CandidateArtifacts::build(loaded.context(), &runtime, manifest_path, output_weight)?;
    let mut a_sequence = loaded.create_sequence(SequenceConfig::new(SEQUENCE_CAPACITY))?;
    let mut b_sequence = loaded.create_sequence(SequenceConfig::new(SEQUENCE_CAPACITY))?;
    let mut a_scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        loaded.context(),
        loaded.metal_model(),
        PROMPT_TOKENS as u32,
        PROMPT_TOKENS,
    )?;
    let mut b_scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        loaded.context(),
        loaded.metal_model(),
        PROMPT_TOKENS as u32,
        PROMPT_TOKENS,
    )?;
    let forward = loaded.forward();
    let mut state_a = runtime.root();
    let mut state_b = runtime.root();
    let (a_prefill_gpu_ms, a_evidence) = unsafe {
        prefill_tokens_profiled_with_tail(
            &forward,
            &prompt.token_ids,
            0,
            a_sequence.metal_session_mut(),
            &mut a_scratch,
            LmHeadTail::Resident,
        )
    }?;
    let (b_prefill_gpu_ms, b_evidence) = unsafe {
        prefill_tokens_profiled_with_tail(
            &forward,
            &prompt.token_ids,
            0,
            b_sequence.metal_session_mut(),
            &mut b_scratch,
            candidate.tail(state_b)?,
        )
    }?;
    a_sequence.advance_by(PROMPT_TOKENS)?;
    b_sequence.advance_by(PROMPT_TOKENS)?;
    let mut current_tail_records = validate_evidence_pair(
        &runtime,
        state_a,
        a_evidence,
        b_evidence,
        a_prefill_gpu_ms,
        b_prefill_gpu_ms,
    )?;
    let mut consumed = prompt.token_ids.clone();
    let mut current_state_evidence =
        compare_sequence_state(&a_sequence, &b_sequence, &consumed, None)?;

    let mut a_sampler = Sampler::new(SamplingConfig::qwen_chat(seed))?;
    let mut b_sampler = Sampler::new(SamplingConfig::qwen_chat(seed))?;
    let mut a_sink = CallbackSink::new();
    let mut b_sink = CallbackSink::new();
    let mut generated_a = Vec::new();
    let mut generated_b = Vec::new();
    let mut transitions = 0usize;
    let mut positions = Vec::new();
    let commit_state_sha256;

    loop {
        ensure!(state_a.index() == state_b.index());
        ensure!(!runtime.state(state_a)?.is_terminal());
        let sampled_state = state_a;
        let state_ref = runtime.state(sampled_state)?;
        let token_ids: Vec<i32> = state_ref
            .candidates()
            .iter()
            .map(|edge| edge.token_id)
            .collect();
        let a_logits = read_request_logits(Arm::A, state_a, &runtime, &a_sequence, None)?;
        let b_logits =
            read_request_logits(Arm::B, state_b, &runtime, &b_sequence, Some(&candidate))?;
        compare_f32_bits(&a_logits, &b_logits, "selected logits")?;
        let selected_logits_sha256 = selected_logits_digest(&token_ids, &a_logits)?;
        let a_advance = sample_advance_emit(
            &runtime,
            tokenizer,
            state_a,
            &a_logits,
            &mut a_sampler,
            &mut a_sink,
            true,
        )?;
        let b_advance = sample_advance_emit(
            &runtime,
            tokenizer,
            state_b,
            &b_logits,
            &mut b_sampler,
            &mut b_sink,
            true,
        )?;
        ensure!(a_advance.sampled == b_advance.sampled);
        ensure!(a_advance.successor.index() == b_advance.successor.index());
        ensure!(a_advance.piece == b_advance.piece);
        generated_a.push(a_advance.sampled.original_token);
        generated_b.push(b_advance.sampled.original_token);
        state_a = a_advance.successor;
        state_b = b_advance.successor;
        ensure!(a_sink.bytes == b_sink.bytes && a_sink.calls == b_sink.calls);
        positions.push(LockstepPositionEvidence {
            position_index: positions.len(),
            state_index: sampled_state.index(),
            state_kind: state_kind_label(state_ref.kind()),
            token_ids,
            selected_logits_sha256,
            post_norm_hidden_sha256: current_state_evidence.post_norm_hidden_sha256.clone(),
            model_state_sha256: current_state_evidence.model_state_sha256.clone(),
            sampled_local_token: a_advance.sampled.local_token,
            sampled_original_token: a_advance.sampled.original_token,
            candidate_index: a_advance.sampled.candidate_index,
            draws_before: a_advance.sampled.draws_before,
            draws_after: a_advance.sampled.draws_after,
            successor_state_index: a_advance.successor.index(),
            piece_sha256: sha256_hex(&a_advance.piece),
            callback_sha256: sha256_hex(&a_sink.bytes),
            a_tail: current_tail_records.0.clone(),
            b_tail: current_tail_records.1.clone(),
        });

        if runtime.state(state_a)?.is_terminal() {
            let pending = *generated_a
                .last()
                .context("missing final generated token")?;
            commit_state_sha256 =
                compare_sequence_state(&a_sequence, &b_sequence, &consumed, Some(pending))?
                    .model_state_sha256;
            break;
        }

        let input = *generated_a
            .last()
            .context("missing generated transition input")?;
        ensure!(Some(&input) == generated_b.last());
        let position = u32::try_from(a_sequence.position())?;
        ensure!(b_sequence.position() == a_sequence.position());
        let (a_profile, a_evidence) = unsafe {
            forward.single_token_profiled_concurrent_gdn_moe_with_tail(
                input,
                position,
                a_sequence.metal_session_mut(),
                LmHeadTail::Resident,
            )
        }?;
        let (b_profile, b_evidence) = unsafe {
            forward.single_token_profiled_concurrent_gdn_moe_with_tail(
                input,
                position,
                b_sequence.metal_session_mut(),
                candidate.tail(state_b)?,
            )
        }?;
        a_sequence.advance_by(1)?;
        b_sequence.advance_by(1)?;
        consumed.push(input);
        transitions += 1;
        current_tail_records = validate_evidence_pair(
            &runtime,
            state_a,
            a_evidence,
            b_evidence,
            a_profile.gpu_kernel_ms,
            b_profile.gpu_kernel_ms,
        )?;
        current_state_evidence = compare_sequence_state(&a_sequence, &b_sequence, &consumed, None)?;
    }

    ensure!(generated_a == generated_b);
    ensure!(generated_a.len() == transitions + 1);
    ensure!(a_sampler.draws() == generated_a.len());
    ensure!(b_sampler.draws() == generated_b.len());
    ensure!(a_sink.calls == generated_a.len());
    ensure!(b_sink.calls == generated_b.len());
    candidate.validate_after(output_weight)?;
    let pending_token = *generated_a
        .last()
        .context("missing lockstep pending token")?;
    Ok(LockstepSeedEvidence {
        seed,
        positions,
        output_token_ids: generated_a.clone(),
        output_sha256: sha256_hex(&a_sink.bytes),
        commit_state_sha256,
        pending_token,
        generated_tokens: generated_a.len(),
        transitions,
        sampler_draws: a_sampler.draws(),
        callback_count: a_sink.calls,
        bit_exact: true,
    })
}

fn validate_lockstep(
    loaded: &LoadedModel,
    tokenizer: &NativeTokenizer,
    output_weight: &[u8],
    prompt_path: &Path,
    runtime_path: &Path,
    manifest_path: &Path,
) -> Result<Vec<LockstepSeedEvidence>> {
    let mut evidence = Vec::with_capacity(CONFORMANCE_SEEDS.len());
    for seed in CONFORMANCE_SEEDS {
        evidence.push(run_lockstep_seed(
            seed,
            loaded,
            tokenizer,
            output_weight,
            prompt_path,
            runtime_path,
            manifest_path,
        )?);
    }
    Ok(evidence)
}

fn median(values: &[f64]) -> Result<f64> {
    ensure!(!values.is_empty(), "median requires at least one value");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Ok(if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

fn reduce_pairs(pairs: &[PairResult]) -> Result<Reduction> {
    ensure!(pairs.len() == SCORED_PAIRS);
    let request: Vec<f64> = pairs.iter().map(|pair| pair.request_saving_ms).collect();
    let ttft: Vec<f64> = pairs.iter().map(|pair| pair.ttft_delta_ms).collect();
    let generation: Vec<f64> = pairs.iter().map(|pair| pair.generation_saving_ms).collect();
    let ab: Vec<f64> = pairs
        .iter()
        .filter(|pair| pair.order == "AB")
        .map(|pair| pair.request_saving_ms)
        .collect();
    let ba: Vec<f64> = pairs
        .iter()
        .filter(|pair| pair.order == "BA")
        .map(|pair| pair.request_saving_ms)
        .collect();
    ensure!(ab.len() == 6 && ba.len() == 6);
    let mut leave_one_out_request_medians_ms = Vec::with_capacity(pairs.len());
    for omitted in 0..pairs.len() {
        let retained: Vec<f64> = request
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (index != omitted).then_some(value))
            .collect();
        leave_one_out_request_medians_ms.push(median(&retained)?);
    }
    Ok(Reduction {
        median_request_saving_ms: median(&request)?,
        positive_request_pairs: request.iter().filter(|&&value| value > 0.0).count(),
        median_ttft_delta_ms: median(&ttft)?,
        median_generation_saving_ms: median(&generation)?,
        ab_median_request_saving_ms: median(&ab)?,
        ba_median_request_saving_ms: median(&ba)?,
        first_half_median_request_saving_ms: median(&request[..6])?,
        second_half_median_request_saving_ms: median(&request[6..])?,
        leave_one_out_request_medians_ms,
    })
}

fn decide(reduction: &Reduction, all_correctness: bool, environment_valid: bool) -> Decision {
    let request_gate = reduction.median_request_saving_ms >= 5.0;
    let pair_win_gate = reduction.positive_request_pairs >= 10;
    let generation_gate = reduction.median_generation_saving_ms > 0.0;
    let ttft_gate = reduction.median_ttft_delta_ms <= 10.0;
    let all_go = all_correctness
        && environment_valid
        && request_gate
        && pair_win_gate
        && generation_gate
        && ttft_gate;
    let disposition = if !environment_valid {
        "invalid"
    } else if !all_correctness {
        "kill"
    } else if all_go {
        "go"
    } else if reduction.median_request_saving_ms < 0.0
        || reduction.median_generation_saving_ms < 0.0
    {
        "kill"
    } else {
        "no-go-park"
    };
    Decision {
        disposition: disposition.to_string(),
        authority: if all_go {
            "one_later_exact_a3b_product_design_decision".to_string()
        } else {
            "none".to_string()
        },
        all_correctness,
        environment_valid,
        request_gate,
        pair_win_gate,
        generation_gate,
        ttft_gate,
    }
}

fn validate_current_identity(build_identity: &Value) -> Result<()> {
    ensure!(build_identity.get("status").and_then(Value::as_str) == Some("match"));
    ensure!(build_identity.get("build_dirty").and_then(Value::as_bool) == Some(false));
    ensure!(build_identity.get("runtime_dirty").and_then(Value::as_bool) == Some(false));
    ensure!(
        build_identity
            .get("problems")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
    );
    ensure!(
        build_identity
            .get("overrides")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
    );
    ensure!(
        build_identity.get("build_commit") == build_identity.get("runtime_commit"),
        "build/runtime commits differ"
    );
    ensure!(
        build_identity.get("build_source_state") == build_identity.get("runtime_source_state"),
        "build/runtime source identities differ"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_terminal_failure(
    disposition: &'static str,
    failure_class: &'static str,
    error: &anyhow::Error,
    progress: &AcquisitionProgress,
    build_identity: &Value,
    qwen_env: &BTreeMap<String, String>,
    user_gpu_attestation: bool,
    prerequisite: &PrerequisiteIdentity,
    model_identity: &ModelIdentity,
    artifact_identity: &ArtifactIdentity,
    correctness: &CorrectnessSummary,
    warmup_pairs: &[PairResult],
    scored_pairs: &[PairResult],
    host_before_warmup: &HostSnapshot,
    host_before_scored: &HostSnapshot,
    vm_before_warmup: &VmCounters,
    host_after_scored: Option<&HostSnapshot>,
    vm_after_scored: Option<&VmCounters>,
    vm_delta: Option<&VmDelta>,
) -> Result<()> {
    ensure!(progress.scored_b_started);
    let result = TerminalFailure {
        schema: RESULT_SCHEMA,
        test: "a3b_integrated_grammar_row",
        phase: "acquire",
        disposition,
        authority: "none",
        failure_class,
        error: format!("{error:#}"),
        progress,
        build_identity,
        qwen_env,
        user_gpu_attestation,
        prerequisite,
        model_identity,
        artifact_identity,
        correctness,
        warmup_pairs,
        scored_pairs,
        host_before_warmup,
        host_before_scored,
        vm_before_warmup,
        host_after_scored,
        vm_after_scored,
        vm_delta,
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

fn is_packet_input_failure(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<ArtifactInputFailure>().is_some())
}

pub fn run(
    args: IntegratedGrammarRowArgs,
    build_identity: Value,
    qwen_env: BTreeMap<String, String>,
) -> Result<()> {
    validate_current_identity(&build_identity)?;
    ensure!(qwen_env.is_empty(), "v0.656 forbids every QWEN_* override");
    ensure!(
        args.attest_no_other_user_gpu_workload,
        "v0.656 requires --attest-no-other-user-gpu-workload"
    );
    ensure!(
        SAMPLER_ALGORITHM_VERSION == 1,
        "sampler algorithm version changed"
    );
    ensure!(SEQUENCE_CAPACITY == 240);

    let prerequisite = validate_prerequisite(&args.a3b_result)?;
    let (canonical_model, model_sha256) = authenticate_model_path(&args.model)?;
    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_model(&canonical_model)
        .context("load frozen A3B model")?;
    let tokenizer = loaded.tokenizer().context("construct native tokenizer")?;
    ensure!(tokenizer.n_vocab() as usize == VOCAB);
    let artifact_identity = authenticate_artifacts(
        &args.prompt_fixture,
        &args.runtime_artifact,
        &args.manifest,
        &tokenizer,
    )?;
    let (output_weight, output_weight_sha256) = validate_loaded_model(&loaded)?;
    let model_identity = ModelIdentity {
        path: canonical_model,
        file_bytes: MODEL_BYTES,
        sha256: model_sha256,
        output_weight_bytes: output_weight.len(),
        output_weight_sha256,
    };

    let bank_validation = validate_all_bank_views(
        &loaded,
        &args.runtime_artifact,
        &args.manifest,
        output_weight,
    )?;
    let lockstep = validate_lockstep(
        &loaded,
        &tokenizer,
        output_weight,
        &args.prompt_fixture,
        &args.runtime_artifact,
        &args.manifest,
    )?;
    let all_correctness = bank_validation.bit_exact_selected_logits
        && bank_validation.guards_intact
        && bank_validation.padding_intact
        && bank_validation.banks_unchanged
        && bank_validation.command_completed
        && bank_validation.command_error_none
        && lockstep.len() == CONFORMANCE_SEEDS.len()
        && lockstep.iter().all(|seed| seed.bit_exact);
    let correctness = CorrectnessSummary {
        bank_validation,
        lockstep,
    };

    if !all_correctness {
        let result = ConformanceOnlyResult {
            schema: RESULT_SCHEMA,
            test: "a3b_integrated_grammar_row",
            phase: match args.phase {
                ExecutionPhase::ConformanceOnly => "conformance_only",
                ExecutionPhase::Acquire => "acquire_preflight",
            },
            disposition: "kill",
            authority: "none",
            build_identity: &build_identity,
            qwen_env: &qwen_env,
            user_gpu_attestation: args.attest_no_other_user_gpu_workload,
            prerequisite: &prerequisite,
            model_identity: &model_identity,
            artifact_identity: &artifact_identity,
            device: loaded.context().describe(),
            correctness: &correctness,
        };
        println!("{}", serde_json::to_string(&result)?);
        return Ok(());
    }

    if args.phase == ExecutionPhase::ConformanceOnly {
        let result = ConformanceOnlyResult {
            schema: RESULT_SCHEMA,
            test: "a3b_integrated_grammar_row",
            phase: "conformance_only",
            disposition: "conformance_pass",
            authority: "none",
            build_identity: &build_identity,
            qwen_env: &qwen_env,
            user_gpu_attestation: args.attest_no_other_user_gpu_workload,
            prerequisite: &prerequisite,
            model_identity: &model_identity,
            artifact_identity: &artifact_identity,
            device: loaded.context().describe(),
            correctness: &correctness,
        };
        println!("{}", serde_json::to_string(&result)?);
        return Ok(());
    }

    let host_before_warmup = HostSnapshot::capture("before_warmup")?;
    host_before_warmup.validate()?;
    let vm_before_warmup = VmCounters::capture()?;
    let mut warmup_pairs = Vec::with_capacity(WARMUP_PAIRS);
    for pair_index in 0..WARMUP_PAIRS {
        warmup_pairs.push(run_pair(
            pair_index,
            false,
            &loaded,
            &tokenizer,
            output_weight,
            &args.prompt_fixture,
            &args.runtime_artifact,
            &args.manifest,
            None,
        )?);
    }

    let host_before_scored = HostSnapshot::capture("before_scored")?;
    host_before_scored.validate()?;
    let mut progress = AcquisitionProgress::default();
    let mut scored_pairs = Vec::with_capacity(SCORED_PAIRS);
    for pair_index in 0..SCORED_PAIRS {
        let pair = run_pair(
            pair_index,
            true,
            &loaded,
            &tokenizer,
            output_weight,
            &args.prompt_fixture,
            &args.runtime_artifact,
            &args.manifest,
            Some(&mut progress),
        );
        match pair {
            Ok(pair) => scored_pairs.push(pair),
            Err(error) if !progress.scored_b_started => return Err(error),
            Err(error) => {
                let (disposition, failure_class) = if is_packet_input_failure(&error) {
                    ("packet-failure", "scored_input_or_parser")
                } else {
                    ("kill", "scored_correctness_or_execution")
                };
                return emit_terminal_failure(
                    disposition,
                    failure_class,
                    &error,
                    &progress,
                    &build_identity,
                    &qwen_env,
                    args.attest_no_other_user_gpu_workload,
                    &prerequisite,
                    &model_identity,
                    &artifact_identity,
                    &correctness,
                    &warmup_pairs,
                    &scored_pairs,
                    &host_before_warmup,
                    &host_before_scored,
                    &vm_before_warmup,
                    None,
                    None,
                    None,
                );
            }
        }
    }
    let vm_after_scored = match VmCounters::capture() {
        Ok(counters) => counters,
        Err(error) => {
            return emit_terminal_failure(
                "invalid",
                "post_scored_vm_capture",
                &error,
                &progress,
                &build_identity,
                &qwen_env,
                args.attest_no_other_user_gpu_workload,
                &prerequisite,
                &model_identity,
                &artifact_identity,
                &correctness,
                &warmup_pairs,
                &scored_pairs,
                &host_before_warmup,
                &host_before_scored,
                &vm_before_warmup,
                None,
                None,
                None,
            );
        }
    };
    let vm_delta = VmDelta::between(&vm_before_warmup, &vm_after_scored);
    let host_after_scored = match HostSnapshot::capture("after_scored") {
        Ok(snapshot) => snapshot,
        Err(error) => {
            return emit_terminal_failure(
                "invalid",
                "post_scored_host_capture",
                &error,
                &progress,
                &build_identity,
                &qwen_env,
                args.attest_no_other_user_gpu_workload,
                &prerequisite,
                &model_identity,
                &artifact_identity,
                &correctness,
                &warmup_pairs,
                &scored_pairs,
                &host_before_warmup,
                &host_before_scored,
                &vm_before_warmup,
                None,
                Some(&vm_after_scored),
                Some(&vm_delta),
            );
        }
    };
    let environment_valid = host_after_scored.validate().is_ok() && vm_delta.is_valid();
    let reduction = match reduce_pairs(&scored_pairs) {
        Ok(reduction) => reduction,
        Err(error) => {
            return emit_terminal_failure(
                if environment_valid { "kill" } else { "invalid" },
                "reduction",
                &error,
                &progress,
                &build_identity,
                &qwen_env,
                args.attest_no_other_user_gpu_workload,
                &prerequisite,
                &model_identity,
                &artifact_identity,
                &correctness,
                &warmup_pairs,
                &scored_pairs,
                &host_before_warmup,
                &host_before_scored,
                &vm_before_warmup,
                Some(&host_after_scored),
                Some(&vm_after_scored),
                Some(&vm_delta),
            );
        }
    };
    let decision = decide(&reduction, all_correctness, environment_valid);

    let result = IntegratedResult {
        schema: RESULT_SCHEMA,
        test: "a3b_integrated_grammar_row",
        phase: "acquire",
        claim_scope: "exact frozen A3B response-shape request; no general grammar authority",
        build_identity,
        qwen_env,
        user_gpu_attestation: args.attest_no_other_user_gpu_workload,
        prerequisite,
        model_identity,
        artifact_identity,
        device: loaded.context().describe(),
        protocol: json!({
            "warmup_pairs": WARMUP_PAIRS,
            "scored_pairs": SCORED_PAIRS,
            "seed": 0,
            "conformance_seeds": CONFORMANCE_SEEDS,
            "hidden_seeds": HIDDEN_SEEDS,
            "prompt_tokens": PROMPT_TOKENS,
            "maximum_generated_tokens": MAX_GENERATED_TOKENS,
            "sequence_capacity": SEQUENCE_CAPACITY,
            "prefill_chunk": PROMPT_TOKENS,
            "order": "AB/BA alternating from AB independently in warmup and scored",
        }),
        correctness,
        host_before_warmup,
        host_before_scored,
        host_after_scored,
        vm_before_warmup,
        vm_after_scored,
        vm_delta,
        progress,
        warmup_pairs,
        scored_pairs,
        reduction,
        decision,
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROMPT_BYTES: &[u8] =
        include_bytes!("../../../docs/bench/v0656-response-shape-prompt.json");

    #[test]
    fn prompt_fixture_is_strict_and_reproduces_non_model_hashes() {
        assert_eq!(sha256_hex(PROMPT_BYTES), PROMPT_FILE_SHA256);
        let fixture: PromptFixture = serde_json::from_slice(PROMPT_BYTES).unwrap();
        assert_eq!(fixture.schema, PROMPT_SCHEMA);
        assert_eq!(fixture.messages.len(), 2);
        let messages: Vec<ChatMessage> = fixture
            .messages
            .iter()
            .map(|message| ChatMessage {
                role: message.role.clone(),
                content: message.content.clone(),
                ..Default::default()
            })
            .collect();
        let rendered = render_qwen_messages_prompt(&messages, false, true);
        assert_eq!(rendered.len(), 684);
        assert_eq!(sha256_hex(rendered.as_bytes()), RENDERED_SHA256);
        assert_eq!(fixture.tokenizer.token_ids.len(), PROMPT_TOKENS);
        assert_eq!(
            token_ids_digest(&fixture.tokenizer.token_ids),
            TOKEN_IDS_SHA256
        );
    }

    #[test]
    fn local_sampler_maps_success_and_nan_to_original_tokens() {
        let tokens = [100, 200, 300];
        let mut sampler = Sampler::new(SamplingConfig::qwen_chat(0)).unwrap();
        let outcome = sample_local(&mut sampler, &[0.0, 4.0, 1.0], &tokens).unwrap();
        assert_eq!(outcome.original_token, 200);
        assert_eq!(outcome.draws_before, 0);
        assert_eq!(outcome.draws_after, 1);

        for position in 0..tokens.len() {
            let mut logits = [0.0, 1.0, 2.0];
            logits[position] = f32::from_bits(OUTPUT_POISON_BITS);
            let mut sampler = Sampler::new(SamplingConfig::qwen_chat(0)).unwrap();
            let error = sample_local(&mut sampler, &logits, &tokens)
                .unwrap_err()
                .to_string();
            assert!(error.contains(&format!("original=Some({})", tokens[position])));
            assert_eq!(sampler.draws(), 0);
        }
    }

    #[test]
    fn local_sampler_covers_ties_infinities_and_empty_support() {
        let tokens = [7, 11, 19];
        for (seed, original, local, candidate) in [
            (0, 7, 0, 0),
            (1, 19, 2, 2),
            (42, 19, 2, 2),
            (u64::MAX, 11, 1, 1),
        ] {
            let mut sampler = Sampler::new(SamplingConfig::qwen_chat(seed)).unwrap();
            let outcome = sample_local(&mut sampler, &[1.0, 1.0, 1.0], &tokens).unwrap();
            assert_eq!(outcome.original_token, original);
            assert_eq!(outcome.local_token, local);
            assert_eq!(outcome.candidate_index, candidate);
            assert_eq!(sampler.draws(), 1);
        }
        let mut sampler = Sampler::new(SamplingConfig::qwen_chat(0)).unwrap();
        let outcome =
            sample_local(&mut sampler, &[f32::INFINITY, 0.0, f32::INFINITY], &tokens).unwrap();
        assert_eq!(outcome.original_token, 7);
        assert_eq!(outcome.local_token, 0);
        assert_eq!(outcome.candidate_index, 0);

        let mut singleton = Sampler::new(SamplingConfig::qwen_chat(42)).unwrap();
        let outcome = sample_local(&mut singleton, &[-3.0], &[1234]).unwrap();
        assert_eq!(outcome.original_token, 1234);
        assert_eq!(outcome.local_token, 0);
        assert_eq!(outcome.candidate_index, 0);
        assert_eq!(singleton.draws(), 1);

        let mut sampler = Sampler::new(SamplingConfig::qwen_chat(0)).unwrap();
        assert!(
            sample_local(&mut sampler, &[f32::NEG_INFINITY; 3], &tokens,)
                .unwrap_err()
                .to_string()
                .contains("negative infinity")
        );
        assert_eq!(sampler.draws(), 0);
    }

    #[test]
    fn reduction_uses_paired_differences() {
        let values = [1.0, 9.0, 3.0, 7.0];
        assert_eq!(median(&values).unwrap(), 5.0);
        assert!(median(&[]).is_err());

        let pairs: Vec<PairResult> = (0..SCORED_PAIRS)
            .map(|index| PairResult {
                pair_index: index,
                order: if index % 2 == 0 { "AB" } else { "BA" }.to_string(),
                a: dummy_request(Arm::A),
                b: dummy_request(Arm::B),
                request_saving_ms: (index + 1) as f64,
                ttft_delta_ms: 10.0,
                generation_saving_ms: 0.25,
            })
            .collect();
        let reduction = reduce_pairs(&pairs).unwrap();
        assert_eq!(reduction.median_request_saving_ms, 6.5);
        assert_eq!(reduction.positive_request_pairs, 12);
        assert_eq!(reduction.median_ttft_delta_ms, 10.0);
        assert_eq!(reduction.median_generation_saving_ms, 0.25);
        assert_eq!(reduction.ab_median_request_saving_ms, 6.0);
        assert_eq!(reduction.ba_median_request_saving_ms, 7.0);
        assert_eq!(reduction.first_half_median_request_saving_ms, 3.5);
        assert_eq!(reduction.second_half_median_request_saving_ms, 9.5);
        assert_eq!(reduction.leave_one_out_request_medians_ms.len(), 12);
    }

    #[test]
    fn pair_schedules_are_exact_and_restart_from_ab() {
        let warmup: Vec<_> = (0..WARMUP_PAIRS).map(pair_order).collect();
        let scored: Vec<_> = (0..SCORED_PAIRS).map(pair_order).collect();
        assert_eq!(warmup[0], [Arm::A, Arm::B]);
        assert_eq!(warmup[1], [Arm::B, Arm::A]);
        assert_eq!(warmup[4], [Arm::A, Arm::B]);
        assert_eq!(scored[0], [Arm::A, Arm::B]);
        assert_eq!(scored[11], [Arm::B, Arm::A]);
        assert_eq!(
            scored
                .iter()
                .filter(|&&order| order == [Arm::A, Arm::B])
                .count(),
            6
        );
        assert_eq!(
            scored
                .iter()
                .filter(|&&order| order == [Arm::B, Arm::A])
                .count(),
            6
        );
    }

    #[test]
    fn decision_boundaries_follow_frozen_precedence() {
        let go = reduction_fixture(5.0, 10, 10.0, 0.001);
        assert_eq!(decide(&go, true, true).disposition, "go");
        assert_eq!(decide(&go, false, true).disposition, "kill");
        assert_eq!(decide(&go, true, false).disposition, "invalid");

        assert_eq!(
            decide(&reduction_fixture(5.0, 9, 10.0, 0.001), true, true).disposition,
            "no-go-park"
        );
        assert_eq!(
            decide(&reduction_fixture(5.0, 10, 10.001, 0.001), true, true).disposition,
            "no-go-park"
        );
        assert_eq!(
            decide(&reduction_fixture(0.0, 10, 0.0, 0.001), true, true).disposition,
            "no-go-park"
        );
        assert_eq!(
            decide(&reduction_fixture(5.0, 10, 0.0, 0.0), true, true).disposition,
            "no-go-park"
        );
        assert_eq!(
            decide(&reduction_fixture(-0.001, 0, 0.0, 1.0), true, true).disposition,
            "kill"
        );
        assert_eq!(
            decide(&reduction_fixture(5.0, 12, 0.0, -0.001), true, true).disposition,
            "kill"
        );
        assert_eq!(
            decide(&reduction_fixture(-1.0, 0, 0.0, -1.0), true, false).disposition,
            "invalid"
        );
    }

    #[test]
    fn only_explicit_artifact_failures_are_packet_input_failures() {
        let tagged = anyhow::Error::new(ArtifactInputFailure(
            "prompt fixture read failed".to_string(),
        ));
        assert!(is_packet_input_failure(&tagged));

        let parse_error: Result<Value> = serde_json::from_slice::<Value>(b"{")
            .map_err(anyhow::Error::from)
            .context("parse fixture");
        let tagged_parse = artifact_parse(parse_error, "fixture").unwrap_err();
        assert!(is_packet_input_failure(&tagged_parse));

        let accounting = anyhow::Error::new(std::io::Error::other("proc_pid_rusage failed"));
        assert!(!is_packet_input_failure(&accounting));
    }

    fn reduction_fixture(request: f64, wins: usize, ttft: f64, generation: f64) -> Reduction {
        Reduction {
            median_request_saving_ms: request,
            positive_request_pairs: wins,
            median_ttft_delta_ms: ttft,
            median_generation_saving_ms: generation,
            ab_median_request_saving_ms: request,
            ba_median_request_saving_ms: request,
            first_half_median_request_saving_ms: request,
            second_half_median_request_saving_ms: request,
            leave_one_out_request_medians_ms: vec![request; SCORED_PAIRS],
        }
    }

    fn dummy_request(arm: Arm) -> RequestResult {
        RequestResult {
            arm,
            seed: 0,
            order_index: usize::from(arm == Arm::B),
            scored: true,
            timing: RequestTiming::default(),
            output_token_ids: vec![1],
            output_bytes: vec![b'x'],
            output_hex: "78".to_string(),
            state_indices: vec![0, 1],
            state_kinds: vec!["branch".to_string(), "terminal".to_string()],
            pending_token: 1,
            generated_tokens: 1,
            transitions: 0,
            sampler_draws: 1,
            callback_count: 1,
            full_heads: u32::from(arm == Arm::A),
            branch_heads: 1,
            singleton_heads: 0,
            terminal_heads: 0,
            bank_payload_bytes: 0,
            bank_physical_bytes: 0,
            branch_bank_sha256: None,
            singleton_bank_sha256: None,
            branch_upload_sha256: None,
            singleton_upload_sha256: None,
            branch_after_sha256: None,
            singleton_after_sha256: None,
            bank_upload_authenticated: None,
            tail_records: Vec::new(),
            memory: MemoryRecord {
                metal_allocated_bytes: 0,
                resident_bytes: 0,
                physical_footprint_bytes: 0,
            },
        }
    }
}
