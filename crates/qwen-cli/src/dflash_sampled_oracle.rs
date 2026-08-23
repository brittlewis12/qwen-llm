//! Bench-only exact serial E1a sampled DFlash development oracle.
//!
//! This harness does not execute the separately preregistered E0 lockstep
//! hidden-capture experiment and therefore grants no product authority.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use qwen_llm::{
    gguf::GgufFile,
    loader::{Model, open_dflash_drafter},
    metal::{MetalContext, MetalTensor},
    metal_dflash::{
        DFlash2SelectorIssue, DFlashDecoder, MetalDFlashHead, MetalDFlashLayerMajorScratch,
        MetalDFlashSession, prefill_tokens_with_multi_hidden,
    },
    metal_forward::{MetalForward, MetalModel, MetalSession, SessionSnapshot, SnapshotIdentity},
    sampling::{
        SAMPLER_ALGORITHM_VERSION, Sampler, SamplingConfig, SamplingDistribution,
        SamplingRngDiagnostic,
    },
    tokenizer::{Tokenizer, token_ids_sha256_i32le},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const SCHEMA: &str = "qwen.dflash_sampled_oracle";
const SCHEMA_VERSION: u32 = 4;
const DEVELOPMENT_SEMANTICS: &str =
    "exact_serial_e1a_one_hot_non_performance_e0_unmeasured_product_aligned_packed_prompt_prefill";
const PROMPT_PREFILL_SEMANTICS: &str =
    "normative_product_aligned_packed_prefill_tokens_with_multi_hidden";

#[derive(Parser, Debug)]
pub struct DflashSampledOracleArgs {
    /// Path to the target GGUF.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Path to the DFlash 2 drafter GGUF.
    #[arg(long)]
    drafter: PathBuf,
    /// Prompt text (must tokenize to at least one token).
    #[arg(short = 'p', long)]
    prompt: String,
    /// Maximum generated token count.
    #[arg(long, default_value = "32")]
    tokens: usize,
    /// Stop tokens, comma-separated. Defaults to GGUF metadata.
    #[arg(long, value_delimiter = ',')]
    stop_tokens: Option<Vec<i32>>,
    #[arg(long, default_value = "0.7")]
    temperature: f32,
    #[arg(long, default_value = "200")]
    top_k: usize,
    #[arg(long, default_value = "1.0")]
    top_p: f32,
    #[arg(long, default_value = "0.05")]
    min_p: f32,
    #[arg(long, default_value = "0")]
    seed: u64,
    /// Append-only JSONL evidence path.
    #[arg(short = 'o', long)]
    output: PathBuf,
    /// Skip the target warmup call.
    #[arg(long)]
    no_warmup: bool,
    /// Stable development fixture identifier. This is descriptive only.
    #[arg(long, default_value = "development-unclassified")]
    fixture_id: String,
    /// Fixture role (for example `development-sentinel`).
    #[arg(long, default_value = "development-unclassified")]
    fixture_role: String,
    /// Explicit target arm label. The reducer never infers this from a path.
    #[arg(long, default_value = "development-unclassified")]
    target_arm: String,
    /// Explicit drafter arm label. The reducer never infers this from a path.
    #[arg(long, default_value = "development-unclassified")]
    drafter_arm: String,
}

struct JsonlAppender {
    file: File,
}

impl JsonlAppender {
    fn open(path: &Path) -> Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path)
            .with_context(|| format!("open append-only JSONL {}", path.display()))?;
        let len = file.metadata()?.len();
        if len != 0 {
            file.seek(SeekFrom::End(-1))?;
            let mut last = [0u8; 1];
            file.read_exact(&mut last)?;
            ensure!(
                last[0] == b'\n',
                "refusing to append to JSONL without a terminating newline: {}",
                path.display()
            );
        }
        Ok(Self { file })
    }

    fn write(&mut self, value: &Value) -> Result<()> {
        let mut bytes = serde_json::to_vec(value)?;
        bytes.push(b'\n');
        self.file.write_all(&bytes)?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file.sync_all()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OneHotDecision {
    accepted: bool,
    emitted_now: bool,
    terminal: bool,
    next_carry: Option<i32>,
}

fn decide_one_hot(
    proposal: i32,
    sampled_target: i32,
    emitted_before: usize,
    token_limit: usize,
    stop_tokens: &[i32],
) -> OneHotDecision {
    let terminal = emitted_before
        .checked_add(1)
        .is_none_or(|count| count >= token_limit)
        || stop_tokens.contains(&sampled_target);
    if proposal != sampled_target {
        return OneHotDecision {
            accepted: false,
            emitted_now: false,
            terminal,
            next_carry: Some(sampled_target),
        };
    }
    OneHotDecision {
        accepted: true,
        emitted_now: true,
        terminal,
        next_carry: None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StopOutcome {
    reason: &'static str,
    terminal_token: i32,
    eos_hit: bool,
    token_limit_hit: bool,
}

fn stop_outcome(
    terminal_token: i32,
    emitted_count: usize,
    limit: usize,
    stops: &[i32],
) -> StopOutcome {
    let eos_hit = stops.contains(&terminal_token);
    let token_limit_hit = emitted_count >= limit;
    StopOutcome {
        reason: if eos_hit { "eos" } else { "token_limit" },
        terminal_token,
        eos_hit,
        token_limit_hit,
    }
}

fn position_u32(position: usize, label: &str) -> Result<u32> {
    u32::try_from(position).with_context(|| format!("{label} position {position} exceeds u32"))
}

fn f32_bits(value: f32) -> String {
    format!("0x{:08x}", value.to_bits())
}

fn f64_bits(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

fn u64_hex(value: u64) -> String {
    format!("0x{value:016x}")
}

fn logits_sha256(logits: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in logits {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn bytes_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn digest_hex(digest: &[u8; 32]) -> String {
    let mut result = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

struct GgufAssetIdentity {
    digest: [u8; 32],
    json: Value,
}

fn gguf_asset_identity(gguf: &GgufFile) -> GgufAssetIdentity {
    let mut aggregate = Sha256::new();
    let mut shards = Vec::with_capacity(gguf.shards.len());
    for (index, shard) in gguf.shards.iter().enumerate() {
        let bytes = shard.mmap_bytes();
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        aggregate.update((index as u64).to_le_bytes());
        aggregate.update((bytes.len() as u64).to_le_bytes());
        aggregate.update(digest);
        shards.push(json!({
            "index": index,
            "path": canonical_or_original(&shard.path),
            "bytes": bytes.len(),
            "sha256": digest_hex(&digest),
        }));
    }
    let digest: [u8; 32] = aggregate.finalize().into();
    GgufAssetIdentity {
        digest,
        json: json!({
            "aggregate_sha256_index_size_digest_le": digest_hex(&digest),
            "shards": shards,
        }),
    }
}

fn regular_file_identity(path: &Path) -> Result<Value> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let mut file = File::open(&canonical)
        .with_context(|| format!("open identity input {}", canonical.display()))?;
    let bytes = file.metadata()?.len();
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(json!({
        "path": canonical.display().to_string(),
        "bytes": bytes,
        "sha256": format!("{:x}", digest.finalize()),
    }))
}

fn digest_identity_word(digest: &[u8; 32], offset: usize) -> u64 {
    let bytes: [u8; 8] = digest[offset..offset + 8]
        .try_into()
        .expect("fixed SHA-256 identity word");
    u64::from_le_bytes(bytes)
}

fn validate_development_label(value: &str, flag: &str) -> Result<()> {
    ensure!(!value.is_empty(), "{flag} must be nonempty");
    ensure!(
        value.len() <= 128
            && value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/')
            }),
        "{flag} must be at most 128 ASCII label characters"
    );
    Ok(())
}

fn event(common: &Value, kind: &str, payload: Value) -> Value {
    let mut object = common.as_object().expect("common event object").clone();
    object.insert("event".into(), Value::String(kind.into()));
    object.insert("payload".into(), payload);
    Value::Object(object)
}

fn distribution_json(distribution: &SamplingDistribution) -> Value {
    json!({
        "selected_token": distribution.sampled.token,
        "candidate_index": distribution.sampled.candidate_index,
        "ordered_support": distribution.candidates.iter().map(|candidate| json!({
            "token": candidate.token,
            "weight_f64_bits": u64_hex(candidate.weight.to_bits()),
        })).collect::<Vec<_>>(),
        "total_weight_f64_bits": u64_hex(distribution.total_weight.to_bits()),
    })
}

fn rng_json(rng: Option<SamplingRngDiagnostic>) -> Value {
    match rng {
        Some(rng) => json!({
            "draws_before": rng.draws_before,
            "draws_after": rng.draws_before + 1,
            "state_before": rng.state_before.map(u64_hex),
            "state_after": rng.state_after.map(u64_hex),
            "raw_u64": u64_hex(rng.raw_u64),
            "raw_uniform_f64_bits": u64_hex(rng.unit_f64_bits),
        }),
        None => Value::Null,
    }
}

struct OracleSample {
    token: i32,
    logits_digest: String,
    distribution: SamplingDistribution,
    draw_index: usize,
}

fn sample_with_evidence(
    sampler: &mut Sampler,
    logits: &[f32],
    writer: &mut JsonlAppender,
    common: &Value,
    sample_index: usize,
    frontier: &str,
    target_position: usize,
    event_kind: &str,
    diagnostic_label: &str,
) -> Result<OracleSample> {
    let digest = logits_sha256(logits);
    let draw_index = sampler.draws();
    let (distribution, rng) = sampler.diagnose_next(logits)?;
    let sampled = sampler.sample(logits)?;
    ensure!(
        distribution.sampled == sampled,
        "{diagnostic_label}: diagnose_next selected {:?}, live sampler selected {:?}",
        distribution.sampled,
        sampled
    );
    writer.write(&event(
        common,
        event_kind,
        json!({
            "sample_index": sample_index,
            "frontier": frontier,
            "target_position": target_position,
            "logits_sha256_f32le": digest,
            "distribution": distribution_json(&distribution),
            "rng": rng_json(rng),
            "live_draws_after": sampler.draws(),
        }),
    ))?;
    Ok(OracleSample {
        token: sampled.token,
        logits_digest: digest,
        distribution,
        draw_index,
    })
}

fn sample_oracle(
    sampler: &mut Sampler,
    logits: &[f32],
    writer: &mut JsonlAppender,
    common: &Value,
    sample_index: usize,
    frontier: &str,
    target_position: usize,
) -> Result<OracleSample> {
    sample_with_evidence(
        sampler,
        logits,
        writer,
        common,
        sample_index,
        frontier,
        target_position,
        "sample_decision",
        "oracle sampler",
    )
}

fn sample_reference(
    sampler: &mut Sampler,
    logits: &[f32],
    writer: &mut JsonlAppender,
    common: &Value,
    sample_index: usize,
    frontier: &str,
    target_position: usize,
) -> Result<OracleSample> {
    sample_with_evidence(
        sampler,
        logits,
        writer,
        common,
        sample_index,
        frontier,
        target_position,
        "reference_sample_decision",
        "serial reference sampler",
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ProposalProbability {
    present_in_support: bool,
    numerator_weight: f64,
    total_weight: f64,
    normalized: f64,
}

fn proposal_probability(distribution: &SamplingDistribution, proposal: i32) -> ProposalProbability {
    let weight = distribution
        .candidates
        .iter()
        .find(|candidate| candidate.token == proposal)
        .map(|candidate| candidate.weight);
    let numerator_weight = weight.unwrap_or(0.0);
    ProposalProbability {
        present_in_support: weight.is_some(),
        numerator_weight,
        total_weight: distribution.total_weight,
        normalized: numerator_weight / distribution.total_weight,
    }
}

fn issue_json(issue: &DFlash2SelectorIssue) -> Value {
    match issue {
        DFlash2SelectorIssue::Sentinel {
            candidate_index,
            token_id,
        } => json!({"kind": "sentinel", "candidate_index": candidate_index, "token_id": token_id}),
        DFlash2SelectorIssue::DuplicateId {
            candidate_index,
            first_index,
            token_id,
        } => json!({
            "kind": "duplicate_id", "candidate_index": candidate_index,
            "first_index": first_index, "token_id": token_id,
        }),
        DFlash2SelectorIssue::NonFiniteScore {
            candidate_index,
            token_id,
            score,
        } => json!({
            "kind": "nonfinite_score", "candidate_index": candidate_index,
            "token_id": token_id, "score_f32_bits": f32_bits(*score),
        }),
        DFlash2SelectorIssue::NoValidChoice => json!({"kind": "no_valid_choice"}),
    }
}

fn selector_depth_json(depth: &qwen_llm::metal_dflash::DFlash2SelectorDepthDiagnostic) -> Value {
    json!({
        "depth": depth.depth,
        "predecessor_token": depth.predecessor_token,
        "predecessor_choice_index": depth.predecessor_choice_index,
        "top_k_ids": depth.top_k_ids,
        "unary_logits_f32_bits": depth.unary_logits.iter().copied().map(f32_bits).collect::<Vec<_>>(),
        "final_scores_f32_bits": depth.final_scores.iter().map(|score| score.map(f32_bits)).collect::<Vec<_>>(),
        "greedy_score_f32_bits": depth.greedy_score.map(f32_bits),
        "greedy_index": depth.greedy_index,
        "greedy_token": depth.greedy_token,
        "issues": depth.issues.iter().map(issue_json).collect::<Vec<_>>(),
    })
}

fn canonical_or_original(path: &Path) -> String {
    path.canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn kv_positions(session: &MetalSession) -> Vec<usize> {
    session.kv_n_pos.clone()
}

fn snapshot_identity_json(identity: &SnapshotIdentity) -> Value {
    json!({
        "model_id": identity.model_id,
        "tokenizer_id": identity.tokenizer_id,
        "layout_version": identity.layout_version,
        "n_attn_layers": identity.n_attn_layers,
        "n_gdn_layers": identity.n_gdn_layers,
        "kv_dim_elements": identity.kv_dim_elements,
        "kv_bytes_per_token": identity.kv_bytes_per_token,
        "kv_storage_kind": format!("{:?}", identity.kv_storage_kind),
        "gdn_state_elements_per_layer": identity.gdn_state_elements_per_layer,
        "gdn_conv_elements_per_layer": identity.gdn_conv_elements_per_layer,
    })
}

fn snapshot_identity_sha256(identity: &SnapshotIdentity) -> String {
    let mut hash = Sha256::new();
    hash.update(identity.model_id.to_le_bytes());
    hash.update(identity.tokenizer_id.to_le_bytes());
    hash.update(identity.layout_version.to_le_bytes());
    hash.update(identity.n_attn_layers.to_le_bytes());
    hash.update(identity.n_gdn_layers.to_le_bytes());
    hash.update(identity.kv_dim_elements.to_le_bytes());
    hash.update(identity.kv_bytes_per_token.to_le_bytes());
    hash.update((identity.kv_storage_kind as u32).to_le_bytes());
    hash.update(identity.gdn_state_elements_per_layer.to_le_bytes());
    hash.update(identity.gdn_conv_elements_per_layer.to_le_bytes());
    format!("{:x}", hash.finalize())
}

fn positions_sha256(positions: &[usize]) -> String {
    let mut hash = Sha256::new();
    for &position in positions {
        hash.update((position as u64).to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn pending_token_sha256(pending: Option<i32>) -> String {
    let mut hash = Sha256::new();
    match pending {
        Some(token) => {
            hash.update([1]);
            hash.update(token.to_le_bytes());
        }
        None => hash.update([0]),
    }
    format!("{:x}", hash.finalize())
}

fn snapshot_json(snapshot: &SessionSnapshot) -> Value {
    json!({
        "identity": snapshot_identity_json(&snapshot.identity),
        "identity_sha256_canonical_le": snapshot_identity_sha256(&snapshot.identity),
        "prefix_len": snapshot.prefix_tokens.len(),
        "prefix_token_ids_sha256_i32le": token_ids_sha256_i32le(&snapshot.prefix_tokens),
        "pending_token": snapshot.pending_token,
        "pending_token_sha256_tagged_i32le": pending_token_sha256(snapshot.pending_token),
        "kv_positions": snapshot.kv_n_pos,
        "kv_positions_sha256_u64le": positions_sha256(&snapshot.kv_n_pos),
        "sections": {
            "kv_k": {"bytes": snapshot.kv_k_arena.len(), "sha256": bytes_sha256(&snapshot.kv_k_arena)},
            "kv_v": {"bytes": snapshot.kv_v_arena.len(), "sha256": bytes_sha256(&snapshot.kv_v_arena)},
            "gdn_conv": {"bytes": snapshot.gdn_conv_arena.len(), "sha256": bytes_sha256(&snapshot.gdn_conv_arena)},
            "gdn_state": {"bytes": snapshot.gdn_state_arena.len(), "sha256": bytes_sha256(&snapshot.gdn_state_arena)},
        },
        "final_logits_present": snapshot.final_logits.is_some(),
        "capture_tail_present": snapshot.capture_tail.is_some(),
    })
}

#[derive(Clone, Copy)]
struct SnapshotComparisons {
    identity: bool,
    prefix: bool,
    pending_token: bool,
    kv_positions: bool,
    kv_k: bool,
    kv_v: bool,
    gdn_conv: bool,
    gdn_state: bool,
}

impl SnapshotComparisons {
    fn compare(oracle: &SessionSnapshot, reference: &SessionSnapshot) -> Self {
        Self {
            identity: oracle.identity == reference.identity,
            prefix: oracle.prefix_tokens == reference.prefix_tokens,
            pending_token: oracle.pending_token == reference.pending_token,
            kv_positions: oracle.kv_n_pos == reference.kv_n_pos,
            kv_k: oracle.kv_k_arena == reference.kv_k_arena,
            kv_v: oracle.kv_v_arena == reference.kv_v_arena,
            gdn_conv: oracle.gdn_conv_arena == reference.gdn_conv_arena,
            gdn_state: oracle.gdn_state_arena == reference.gdn_state_arena,
        }
    }

    fn all(self) -> bool {
        self.identity
            && self.prefix
            && self.pending_token
            && self.kv_positions
            && self.kv_k
            && self.kv_v
            && self.gdn_conv
            && self.gdn_state
    }

    fn json(self) -> Value {
        json!({
            "identity": self.identity,
            "prefix": self.prefix,
            "pending_token": self.pending_token,
            "kv_positions": self.kv_positions,
            "kv_k_bytes": self.kv_k,
            "kv_v_bytes": self.kv_v,
            "gdn_conv_bytes": self.gdn_conv,
            "gdn_state_bytes": self.gdn_state,
            "all": self.all(),
        })
    }
}

pub fn run(
    args: DflashSampledOracleArgs,
    build_identity: Value,
    lease_env: BTreeMap<String, String>,
) -> Result<()> {
    ensure!(!args.prompt.is_empty(), "--prompt must be nonempty");
    ensure!(args.tokens > 0, "--tokens must be positive");
    ensure!(
        (1..=200).contains(&args.top_k),
        "--top-k must be in 1..=200 for the bounded E1a development trace"
    );
    ensure!(
        args.temperature.is_finite() && args.temperature > 0.0,
        "--temperature must be finite and positive"
    );
    ensure!(
        std::env::var("QWEN_METAL_LEASE_WAIT").as_deref() == Ok("1"),
        "dflash-sampled-oracle requires QWEN_METAL_LEASE_WAIT=1 before Metal initialization"
    );
    validate_development_label(&args.fixture_id, "--fixture-id")?;
    validate_development_label(&args.fixture_role, "--fixture-role")?;
    validate_development_label(&args.target_arm, "--target-arm")?;
    validate_development_label(&args.drafter_arm, "--drafter-arm")?;

    let target_g = GgufFile::open(&args.model)
        .with_context(|| format!("open target {}", args.model.display()))?;
    let stops = super::resolve_stop_tokens(&target_g, args.stop_tokens.clone())?;
    let target_m = Model::from_gguf(&target_g).context("parse target arch")?;
    let drafter_g = GgufFile::open(&args.drafter)
        .with_context(|| format!("open drafter {}", args.drafter.display()))?;
    let head = open_dflash_drafter(&drafter_g, &target_m).context("bind drafter")?;
    ensure!(
        head.config.block_size > 1,
        "DFlash block must contain proposal slots"
    );
    let tokenizer = Tokenizer::from_gguf(&target_g).context("open tokenizer")?;
    let prompt_ids = tokenizer
        .encode(&args.prompt, false)
        .context("tokenize prompt")?;
    ensure!(
        !prompt_ids.is_empty(),
        "--prompt must tokenize to at least one token"
    );

    let sampling = SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
    }
    .validate()
    .context("invalid sampling configuration")?;
    let generated_boundary = prompt_ids
        .len()
        .checked_add(args.tokens)
        .context("prompt plus token limit overflows context capacity")?;
    let capacity = generated_boundary
        .checked_add(32)
        .context("context capacity plus safety margin overflows")?;
    ensure!(
        capacity <= u32::MAX as usize,
        "requested context capacity {capacity} exceeds bounded u32 position scope"
    );
    let h = target_m.arch.hidden_size as usize;
    let vocab = target_m.arch.vocab_size as usize;
    let feature_count = head
        .target_layer_ids
        .len()
        .checked_mul(h)
        .context("DFlash target hidden feature count overflows")?;
    let prefill_hidden_elements = prompt_ids
        .len()
        .checked_mul(feature_count)
        .context("packed prompt hidden allocation size overflows")?;
    let feature_count_u64 =
        u64::try_from(feature_count).context("hidden feature count exceeds u64")?;
    let prefill_hidden_elements_u64 = u64::try_from(prefill_hidden_elements)
        .context("packed prompt hidden allocation exceeds u64")?;
    let prompt_digest = token_ids_sha256_i32le(&prompt_ids);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let run_id = format!(
        "dflash-e1a-{}-{}-{}",
        now.as_nanos(),
        std::process::id(),
        &prompt_digest[..16]
    );
    // This is an integrity-sensitive, explicitly non-performance oracle. Hash
    // the exact mapped shards and executable before any evidence row is emitted.
    let target_asset = gguf_asset_identity(&target_g);
    let drafter_asset = gguf_asset_identity(&drafter_g);
    let snapshot_model_id = digest_identity_word(&target_asset.digest, 0);
    let snapshot_tokenizer_id = digest_identity_word(&target_asset.digest, 8);
    let executable = std::env::current_exe().context("resolve current executable")?;
    let executable_identity = regular_file_identity(&executable)?;
    let command = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let ctx = MetalContext::new().context("init MetalContext")?;
    let device = ctx.describe();
    eprintln!("[dflash-sampled-oracle] device: {device}");
    let common = json!({
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "build_identity": build_identity,
        "lease_env": lease_env,
        "classification": {
            "evidence_role": "development",
            "fixture_id": args.fixture_id,
            "fixture_role": args.fixture_role,
            "target_arm": args.target_arm,
            "drafter_arm": args.drafter_arm,
            "e0_status": "not_measured",
        },
        "config": {
            "tokens": args.tokens,
            "stop_tokens": stops,
            "temperature_f32_bits": f32_bits(sampling.temperature),
            "top_k": sampling.top_k,
            "top_p_f32_bits": f32_bits(sampling.top_p),
            "min_p_f32_bits": f32_bits(sampling.min_p),
            "seed": sampling.seed,
            "sampler_algorithm_version": SAMPLER_ALGORITHM_VERSION,
            "no_warmup": args.no_warmup,
            "semantics": DEVELOPMENT_SEMANTICS,
            "prompt_prefill": PROMPT_PREFILL_SEMANTICS,
        },
        "assets": {
            "target": target_asset.json,
            "drafter": drafter_asset.json,
            "executable": executable_identity,
        },
        "binding": {
            "target_architecture": target_g.architecture(),
            "target": {
                "n_layer": target_m.arch.n_layer,
                "hidden_size": target_m.arch.hidden_size,
                "vocab_size": target_m.arch.vocab_size,
            },
            "drafter": {
                "n_layer": head.config.n_layer,
                "hidden_size": head.config.hidden_size,
                "block_size": head.config.block_size,
                "swa_window": head.config.swa_window,
                "conv_kernel_size": head.config.conv_kernel_size,
                "conv_group_size": head.config.conv_group_size,
                "selector_rank": head.config.selector_rank,
                "selector_top_k": head.config.selector_top_k,
                "target_layer_ids": head.target_layer_ids,
            },
            "resolved_stop_tokens": stops,
        },
        "command": command,
        "host": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "metal_device": device,
        },
        "paths": {
            "model": canonical_or_original(&args.model),
            "drafter": canonical_or_original(&args.drafter),
            "output": canonical_or_original(&args.output),
        },
        "prompt": {
            "utf8_len": args.prompt.len(),
            "utf8_sha256": bytes_sha256(args.prompt.as_bytes()),
            "token_count": prompt_ids.len(),
            "token_ids_sha256_i32le": prompt_digest,
        },
        "sessions": {
            "oracle_target": format!("{run_id}/oracle-target"),
            "oracle_drafter": format!("{run_id}/oracle-drafter"),
            "serial_reference_target": format!("{run_id}/serial-reference-target"),
        },
    });
    let mut writer = JsonlAppender::open(&args.output)?;
    writer.write(&event(
        &common,
        "run_start",
        json!({"started_utc": super::utc_iso8601_now()}),
    ))?;
    writer.finish().context("sync run_start evidence")?;

    // The lease assertion above deliberately precedes the first Metal object.
    let prefetch_cfg = qwen_llm::runtime::LoadedModelConfig::default();
    let _ = qwen_llm::runtime::prefetch_opened_gguf(&target_g, &prefetch_cfg);
    let _ = qwen_llm::runtime::prefetch_opened_gguf(&drafter_g, &prefetch_cfg);
    let mm = MetalModel::load(&ctx, &target_g, &target_m).context("metal-load target")?;
    let mhead = MetalDFlashHead::load(&ctx, &drafter_g, &head).context("metal-load drafter")?;
    ensure!(
        mhead.selector.is_some(),
        "selector diagnostics require a DFlash 2 drafter"
    );
    let mf = MetalForward::new(&ctx, &mm);
    if !args.no_warmup {
        let mut warm = MetalSession::fresh(&ctx, &mm, capacity)?;
        let _ = mf.single_token(prompt_ids[0], 0, &mut warm)?;
    }

    let mut target_session = MetalSession::fresh(&ctx, &mm, capacity)?;
    let h_u64 = u64::try_from(h).context("target hidden size exceeds u64")?;
    let vocab_u64 = u64::try_from(vocab).context("target vocabulary size exceeds u64")?;
    let mut dflash_session = MetalDFlashSession::fresh(&ctx, &mhead, h_u64, vocab_u64, capacity)?;
    let hidden = MetalTensor::zeros_f32(&ctx, vec![feature_count_u64])?;
    let prefill_hidden = MetalTensor::zeros_f32(&ctx, vec![prefill_hidden_elements_u64])?;
    let mut prefill_scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, head.config.block_size)?;
    let started = Instant::now();
    let prefill_logits = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut target_session,
        &mut prefill_scratch,
        &head.target_layer_ids,
        Some(&prefill_hidden),
    )?;
    dflash_session.append_target_ctx_columns_contiguous_now(
        &ctx,
        &prefill_hidden,
        0,
        prompt_ids.len(),
        feature_count,
    )?;
    ensure!(
        dflash_session.target_ctx_n == prompt_ids.len(),
        "DFlash target context length diverged after prompt append"
    );

    let mut decoder = DFlashDecoder::new(&mf, &mhead, dflash_session);
    let mut sampler = Sampler::new(sampling)?;
    let mut oracle_hashes = Vec::new();
    let initial_sample = sample_oracle(
        &mut sampler,
        &prefill_logits,
        &mut writer,
        &common,
        0,
        "prompt_prefill",
        prompt_ids.len() - 1,
    )?;
    let mut carry = initial_sample.token;
    oracle_hashes.push(initial_sample.logits_digest);
    let mut emitted = Vec::with_capacity(args.tokens);
    let depth_count = head.config.block_size as usize - 1;
    let mut attempts = vec![0usize; depth_count];
    let mut accepts = vec![0usize; depth_count];
    let mut blocks = 0usize;
    let mut processed_pos = prompt_ids.len() - 1;

    'generation: loop {
        if emitted.len() >= args.tokens {
            break;
        }
        emitted.push(carry);
        if stops.contains(&carry) || emitted.len() >= args.tokens {
            break;
        }

        let carry_in = carry;
        let noise_start_pos = processed_pos
            .checked_add(1)
            .context("oracle target position overflow")?;
        let noise_start_pos_u32 = position_u32(noise_start_pos, "oracle carry")?;
        let diagnostic = decoder
            .draft_block_with_selector_diagnostic(carry, noise_start_pos_u32)
            .context("draft block with selector diagnostic")?;
        let proposals = diagnostic.draft_tokens[1..].to_vec();
        ensure!(
            proposals.len() == depth_count,
            "unexpected DFlash proposal count"
        );
        let mut accepted_this_block = 0usize;
        let mut mismatch_depth = None;

        let logits = mf.single_token_with_multi_hidden(
            carry,
            noise_start_pos_u32,
            &mut target_session,
            &head.target_layer_ids,
            &hidden,
        )?;
        decoder.session.append_target_ctx_column_now(
            &ctx,
            &hidden,
            noise_start_pos_u32,
            feature_count,
        )?;
        processed_pos = noise_start_pos;
        ensure!(
            decoder.session.target_ctx_n
                == processed_pos
                    .checked_add(1)
                    .context("oracle consumed context length overflow")?,
            "DFlash target context length diverged after carry append"
        );
        let mut target_sample = sample_oracle(
            &mut sampler,
            &logits,
            &mut writer,
            &common,
            oracle_hashes.len(),
            "target_transition",
            processed_pos,
        )?;
        oracle_hashes.push(target_sample.logits_digest.clone());

        for (depth, &proposal) in proposals.iter().enumerate() {
            attempts[depth] += 1;
            let sampled_target = target_sample.token;
            let decision =
                decide_one_hot(proposal, sampled_target, emitted.len(), args.tokens, &stops);
            let probability = proposal_probability(&target_sample.distribution, proposal);
            let emitted_after_decision = emitted
                .len()
                .checked_add(1)
                .context("decision emitted count overflow")?;
            let decision_stop = decision
                .terminal
                .then(|| stop_outcome(sampled_target, emitted_after_decision, args.tokens, &stops));
            writer.write(&event(
                &common,
                "one_hot_decision",
                json!({
                    "block_index": blocks,
                    "depth": depth,
                    "carry_in": carry_in,
                    "proposal": proposal,
                    "sampled_target": sampled_target,
                    "accepted": decision.accepted,
                    "terminal": decision.terminal,
                    "stop_reason": decision_stop.map(|outcome| outcome.reason),
                    "terminal_token": decision_stop.map(|outcome| outcome.terminal_token),
                    "eos_hit": decision_stop.is_some_and(|outcome| outcome.eos_hit),
                    "token_limit_hit": decision_stop.is_some_and(|outcome| outcome.token_limit_hit),
                    "proposal_present_in_support": probability.present_in_support,
                    "proposal_weight_f64_bits": u64_hex(probability.numerator_weight.to_bits()),
                    "total_weight_f64_bits": u64_hex(probability.total_weight.to_bits()),
                    "proposal_probability_f64_bits": u64_hex(probability.normalized.to_bits()),
                    "draw_index": target_sample.draw_index,
                }),
            ))?;
            if !decision.accepted {
                mismatch_depth = Some(depth);
                debug_assert_eq!(decision.next_carry, Some(sampled_target));
                break;
            }
            accepts[depth] += 1;
            accepted_this_block += 1;
            debug_assert!(decision.emitted_now);
            emitted.push(sampled_target);
            if decision.terminal {
                writer.write(&event(
                    &common,
                    "proposal_block",
                    json!({
                        "block_index": blocks,
                        "noise_start_pos": noise_start_pos,
                        "carry_in": carry_in,
                        "proposals": proposals,
                        "accepted": accepted_this_block,
                        "mismatch_depth": mismatch_depth,
                        "selector_depths": diagnostic.depths.iter().map(selector_depth_json).collect::<Vec<_>>(),
                    }),
                ))?;
                blocks += 1;
                break 'generation;
            }

            let next_pos = processed_pos
                .checked_add(1)
                .context("accepted target position overflow")?;
            let next_pos_u32 = position_u32(next_pos, "accepted target")?;
            let logits = mf.single_token_with_multi_hidden(
                sampled_target,
                next_pos_u32,
                &mut target_session,
                &head.target_layer_ids,
                &hidden,
            )?;
            decoder.session.append_target_ctx_column_now(
                &ctx,
                &hidden,
                next_pos_u32,
                feature_count,
            )?;
            processed_pos = next_pos;
            ensure!(
                decoder.session.target_ctx_n
                    == processed_pos
                        .checked_add(1)
                        .context("accepted consumed context length overflow")?,
                "DFlash target context length diverged after accepted append"
            );
            target_sample = sample_oracle(
                &mut sampler,
                &logits,
                &mut writer,
                &common,
                oracle_hashes.len(),
                "target_transition",
                processed_pos,
            )?;
            oracle_hashes.push(target_sample.logits_digest.clone());
        }

        writer.write(&event(
            &common,
            "proposal_block",
            json!({
                "block_index": blocks,
                "noise_start_pos": noise_start_pos,
                "carry_in": carry_in,
                "proposals": proposals,
                "accepted": accepted_this_block,
                "mismatch_depth": mismatch_depth,
                "selector_depths": diagnostic.depths.iter().map(selector_depth_json).collect::<Vec<_>>(),
            }),
        ))?;
        blocks += 1;
        ensure!(
            decoder.session.target_ctx_n
                == processed_pos
                    .checked_add(1)
                    .context("block consumed context length overflow")?,
            "DFlash target context length diverged at block boundary"
        );
        carry = target_sample.token;
    }

    ensure!(
        sampler.draws() == emitted.len(),
        "oracle consumed one draw per emitted token invariant failed"
    );
    let final_target_ctx_n = decoder.session.target_ctx_n;
    let consumed_generated = emitted
        .len()
        .checked_sub(1)
        .context("oracle has no final pending token")?;
    let consumed_prefix_len = prompt_ids
        .len()
        .checked_add(consumed_generated)
        .context("oracle consumed prefix length overflow")?;
    ensure!(
        final_target_ctx_n == consumed_prefix_len,
        "DFlash target context has {final_target_ctx_n} columns, expected {consumed_prefix_len}"
    );
    ensure!(
        processed_pos
            .checked_add(1)
            .context("final processed position overflow")?
            == consumed_prefix_len,
        "processed target position disagrees with consumed prefix length"
    );
    let oracle_kv = kv_positions(&target_session);

    // Fresh target state and fresh request-local RNG: no oracle state is shared.
    let mut reference_session = MetalSession::fresh(&ctx, &mm, capacity)?;
    let mut reference_scratch =
        MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, head.config.block_size)?;
    let reference_prefill = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut reference_session,
        &mut reference_scratch,
        &[],
        None,
    )?;
    let mut reference_sampler = Sampler::new(sampling)?;
    let reference_initial = sample_reference(
        &mut reference_sampler,
        &reference_prefill,
        &mut writer,
        &common,
        0,
        "prompt_prefill",
        prompt_ids.len() - 1,
    )?;
    let mut reference_next = reference_initial.token;
    let mut reference_hashes = vec![reference_initial.logits_digest];
    let mut reference_ids = Vec::with_capacity(args.tokens);
    let mut reference_pos = prompt_ids.len() - 1;
    loop {
        reference_ids.push(reference_next);
        if stops.contains(&reference_next) || reference_ids.len() >= args.tokens {
            break;
        }
        reference_pos = reference_pos
            .checked_add(1)
            .context("serial reference target position overflow")?;
        let reference_pos_u32 = position_u32(reference_pos, "serial reference")?;
        let logits = mf.single_token(reference_next, reference_pos_u32, &mut reference_session)?;
        let sampled = sample_reference(
            &mut reference_sampler,
            &logits,
            &mut writer,
            &common,
            reference_hashes.len(),
            "target_transition",
            reference_pos,
        )?;
        reference_next = sampled.token;
        reference_hashes.push(sampled.logits_digest);
    }
    let reference_kv = kv_positions(&reference_session);

    let mut oracle_prefix = prompt_ids.clone();
    oracle_prefix.extend_from_slice(&emitted[..consumed_generated]);
    let reference_consumed_generated = reference_ids
        .len()
        .checked_sub(1)
        .context("serial reference has no final pending token")?;
    let mut reference_prefix = prompt_ids.clone();
    reference_prefix.extend_from_slice(&reference_ids[..reference_consumed_generated]);
    let oracle_pending = *emitted
        .last()
        .context("oracle final pending token missing")?;
    let reference_pending = *reference_ids
        .last()
        .context("serial reference final pending token missing")?;
    let oracle_identity =
        target_session.snapshot_identity(snapshot_model_id, snapshot_tokenizer_id);
    let reference_identity =
        reference_session.snapshot_identity(snapshot_model_id, snapshot_tokenizer_id);
    let mut oracle_snapshot = target_session
        .snapshot(oracle_identity, oracle_prefix, None)
        .context("snapshot exact oracle target state")?;
    oracle_snapshot.pending_token = Some(oracle_pending);
    let mut reference_snapshot = reference_session
        .snapshot(reference_identity, reference_prefix, None)
        .context("snapshot exact serial reference target state")?;
    reference_snapshot.pending_token = Some(reference_pending);
    ensure!(
        oracle_snapshot.final_logits.is_none() && reference_snapshot.final_logits.is_none(),
        "state exactness snapshots must exclude final logits"
    );
    let state_comparisons = SnapshotComparisons::compare(&oracle_snapshot, &reference_snapshot);

    let ids_equal = emitted == reference_ids;
    let draws_equal = sampler.draws() == reference_sampler.draws();
    let logits_equal = oracle_hashes == reference_hashes;
    let kv_equal = oracle_kv == reference_kv;
    let state_equal = state_comparisons.all();
    let boundary_equal = ids_equal && draws_equal && logits_equal && kv_equal && state_equal;

    let oracle_continuation_pos = prompt_ids
        .len()
        .checked_add(emitted.len())
        .and_then(|position| position.checked_sub(1))
        .context("oracle continuation position overflow")?;
    let reference_continuation_pos = prompt_ids
        .len()
        .checked_add(reference_ids.len())
        .and_then(|position| position.checked_sub(1))
        .context("reference continuation position overflow")?;
    let oracle_continuation = mf.single_token(
        oracle_pending,
        position_u32(oracle_continuation_pos, "oracle continuation")?,
        &mut target_session,
    )?;
    let reference_continuation = mf.single_token(
        reference_pending,
        position_u32(reference_continuation_pos, "reference continuation")?,
        &mut reference_session,
    )?;
    let oracle_continuation_hash = logits_sha256(&oracle_continuation);
    let reference_continuation_hash = logits_sha256(&reference_continuation);
    let continuation_equal = boundary_equal.then(|| {
        oracle_continuation
            .iter()
            .zip(&reference_continuation)
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && oracle_continuation.len() == reference_continuation.len()
    });
    let all_equal = boundary_equal && continuation_equal == Some(true);
    let oracle_stop = stop_outcome(oracle_pending, emitted.len(), args.tokens, &stops);
    let reference_stop = stop_outcome(reference_pending, reference_ids.len(), args.tokens, &stops);

    writer.write(&event(
        &common,
        "serial_reference_end",
        json!({
            "generated_ids": reference_ids,
            "generated_ids_sha256_i32le": token_ids_sha256_i32le(&reference_ids),
            "stop_reason": reference_stop.reason,
            "terminal_token": reference_stop.terminal_token,
            "eos_hit": reference_stop.eos_hit,
            "token_limit_hit": reference_stop.token_limit_hit,
            "sampler_draws": reference_sampler.draws(),
            "transition_logits_sha256_f32le": reference_hashes,
            "target_kv_positions": reference_kv,
            "target_state": snapshot_json(&reference_snapshot),
            "target_state_comparisons": state_comparisons.json(),
            "continuation_position": reference_continuation_pos,
            "continuation_logits_sha256_f32le": reference_continuation_hash,
            "comparisons": {
                "generated_ids": ids_equal,
                "sampler_draw_count": draws_equal,
                "aligned_transition_logits": logits_equal,
                "target_kv_positions": kv_equal,
                "target_state": state_equal,
                "continuation_boundary_equal": boundary_equal,
                "one_token_continuation_compared": continuation_equal.is_some(),
                "one_token_continuation_logits": continuation_equal,
            },
        }),
    ))?;
    writer.write(&event(
        &common,
        "run_end",
        json!({
            "status": if all_equal { "ok" } else { "mismatch" },
            "generated_ids": emitted,
            "generated_ids_sha256_i32le": token_ids_sha256_i32le(&emitted),
            "stop_reason": oracle_stop.reason,
            "terminal_token": oracle_stop.terminal_token,
            "eos_hit": oracle_stop.eos_hit,
            "token_limit_hit": oracle_stop.token_limit_hit,
            "sampler_draws": sampler.draws(),
            "transition_logits_sha256_f32le": oracle_hashes,
            "target_kv_positions": oracle_kv,
            "target_state": snapshot_json(&oracle_snapshot),
            "target_state_comparisons": state_comparisons.json(),
            "dflash_target_ctx_n": final_target_ctx_n,
            "processed_target_position": processed_pos,
            "consumed_prefix_len": consumed_prefix_len,
            "continuation_position": oracle_continuation_pos,
            "continuation_logits_sha256_f32le": oracle_continuation_hash,
            "continuation_boundary_equal": boundary_equal,
            "one_token_continuation_compared": continuation_equal.is_some(),
            "one_token_continuation_equal": continuation_equal,
            "blocks": blocks,
            "attempts_by_depth": attempts,
            "accepts_by_depth": accepts,
            "elapsed_seconds_f64_bits": f64_bits(started.elapsed().as_secs_f64()),
            "timing_semantics": "diagnostic_only_exact_serial_oracle_non_performance",
        }),
    ))?;
    writer.finish()?;

    eprintln!("[dflash-sampled-oracle] exact serial oracle (non-performance)");
    for depth in 0..depth_count {
        if attempts[depth] != 0 {
            eprintln!(
                "[dflash-sampled-oracle] depth {depth}: accepts {}/{}",
                accepts[depth], attempts[depth]
            );
        }
    }
    if blocks == 0 {
        eprintln!(
            "[dflash-sampled-oracle] blocks=0 emitted={} mean_emitted_per_block=n/a",
            emitted.len()
        );
    } else {
        eprintln!(
            "[dflash-sampled-oracle] blocks={blocks} emitted={} mean_emitted_per_block={:.3}",
            emitted.len(),
            emitted.len() as f64 / blocks as f64
        );
    }
    ensure!(
        all_equal,
        "oracle and fresh plain serial reference differ; see JSONL comparisons"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_hot_rejects_and_carries_sampled_target() {
        assert_eq!(
            decide_one_hot(7, 9, 2, 8, &[]),
            OneHotDecision {
                accepted: false,
                emitted_now: false,
                terminal: false,
                next_carry: Some(9),
            }
        );
        assert!(decide_one_hot(7, 9, 7, 8, &[]).terminal);
    }

    #[test]
    fn one_hot_accepts_and_accounts_for_limit_and_eos() {
        let ordinary = decide_one_hot(7, 7, 2, 8, &[]);
        assert!(ordinary.accepted && ordinary.emitted_now && !ordinary.terminal);
        assert!(decide_one_hot(7, 7, 7, 8, &[]).terminal);
        assert!(decide_one_hot(7, 7, 2, 8, &[7]).terminal);
    }

    #[test]
    fn stop_outcomes_cover_accepted_and_rejected_eos() {
        let accepted = decide_one_hot(9, 9, 2, 8, &[9]);
        assert!(accepted.accepted && accepted.terminal);
        assert_eq!(
            stop_outcome(9, 3, 8, &[9]),
            StopOutcome {
                reason: "eos",
                terminal_token: 9,
                eos_hit: true,
                token_limit_hit: false,
            }
        );

        let rejected = decide_one_hot(7, 9, 2, 8, &[9]);
        assert!(!rejected.accepted && rejected.terminal);
        assert_eq!(rejected.next_carry, Some(9));
    }

    #[test]
    fn stop_outcomes_cover_exact_limit_and_eos_precedence_at_limit() {
        let limit = decide_one_hot(7, 7, 7, 8, &[]);
        assert!(limit.accepted && limit.terminal);
        assert_eq!(
            stop_outcome(7, 8, 8, &[]),
            StopOutcome {
                reason: "token_limit",
                terminal_token: 7,
                eos_hit: false,
                token_limit_hit: true,
            }
        );

        let eos_at_limit = stop_outcome(7, 8, 8, &[7]);
        assert_eq!(eos_at_limit.reason, "eos");
        assert!(eos_at_limit.eos_hit && eos_at_limit.token_limit_hit);
    }

    fn distribution() -> SamplingDistribution {
        SamplingDistribution {
            candidates: vec![
                qwen_llm::sampling::WeightedCandidate {
                    token: 4,
                    weight: 1.0,
                },
                qwen_llm::sampling::WeightedCandidate {
                    token: 8,
                    weight: 3.0,
                },
            ],
            total_weight: 4.0,
            sampled: qwen_llm::sampling::SampledToken {
                token: 8,
                candidate_index: 1,
            },
        }
    }

    #[test]
    fn proposal_probability_preserves_present_numerator_and_normalization() {
        let probability = proposal_probability(&distribution(), 8);
        assert!(probability.present_in_support);
        assert_eq!(probability.numerator_weight.to_bits(), 3.0f64.to_bits());
        assert_eq!(probability.total_weight.to_bits(), 4.0f64.to_bits());
        assert_eq!(probability.normalized.to_bits(), 0.75f64.to_bits());
    }

    #[test]
    fn proposal_probability_records_absent_support_as_exact_zero() {
        let probability = proposal_probability(&distribution(), 9);
        assert!(!probability.present_in_support);
        assert_eq!(probability.numerator_weight.to_bits(), 0.0f64.to_bits());
        assert_eq!(probability.total_weight.to_bits(), 4.0f64.to_bits());
        assert_eq!(probability.normalized.to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn jsonl_writer_appends_whole_schema_events() {
        let path = std::env::temp_dir().join(format!(
            "qwen-dflash-oracle-{}-{}.jsonl",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let common = json!({
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "run_id": "test",
            "build_identity": {}, "lease_env": {}, "config": {}, "paths": {}, "prompt": {},
        });
        {
            let mut out = JsonlAppender::open(&path).unwrap();
            out.write(&event(&common, "run_start", json!({}))).unwrap();
            out.finish().unwrap();
        }
        {
            let mut out = JsonlAppender::open(&path).unwrap();
            out.write(&event(&common, "run_end", json!({}))).unwrap();
            out.finish().unwrap();
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let rows = text
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["event"], "run_start");
        assert_eq!(rows[1]["event"], "run_end");
        assert!(
            rows.iter()
                .all(|row| row["schema_version"] == SCHEMA_VERSION)
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn clap_parses_sampled_oracle_surface() {
        let args = DflashSampledOracleArgs::try_parse_from([
            "dflash-sampled-oracle",
            "--model",
            "target.gguf",
            "--drafter",
            "draft.gguf",
            "--prompt",
            "hello",
            "--tokens",
            "9",
            "--stop-tokens",
            "1,2",
            "--temperature",
            "0.8",
            "--top-k",
            "40",
            "--top-p",
            "0.9",
            "--min-p",
            "0.1",
            "--seed",
            "17",
            "--output",
            "evidence.jsonl",
            "--no-warmup",
        ])
        .unwrap();
        assert_eq!(args.tokens, 9);
        assert_eq!(args.stop_tokens, Some(vec![1, 2]));
        assert_eq!(args.seed, 17);
        assert!(args.no_warmup);
    }
}
