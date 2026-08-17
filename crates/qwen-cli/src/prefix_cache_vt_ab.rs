use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use objc2_metal::MTLDevice;
use qwen_llm::{
    metal::{
        AttnMatrixVtDispatchCapture, capture_attn_matrix_vt_dispatches,
        with_attn_matrix_vt_compact_dispatch_override,
    },
    metal_dflash::{
        MetalDFlashLayerMajorScratch, PrefillScratchConfig, PrefillScratchPlan,
        plan_prefill_scratch_with_matrix_max_pos_configured,
        prefill_tokens_with_multi_hidden_profiled,
    },
    metal_forward::{MetalBlock, MetalForward, SessionSnapshot, SnapshotIdentity},
    runtime::{LoadedModel, LoadedModelConfig, PrefetchPolicy, Runtime, Sequence, SequenceConfig},
    tokenizer::token_ids_sha256_i32le,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    time::Instant,
};

const SCHEMA_VERSION: u32 = 1;
const EXPECTED_MODEL_BYTES: u64 = 17_106_773_984;
const EXPECTED_MODEL_SHA256: &str =
    "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b";
const PREFIX_BLOCK_SIZE: u32 = 1024;
const CONTINUATION_STEPS: usize = 8;
const EXPECTED_ATTN_LAYERS: u64 = 16;
const EXPECTED_N_Q: u32 = 24;
const EXPECTED_N_KV: u32 = 4;
const EXPECTED_HEAD_DIM: u32 = 256;
const ALLOCATOR_ALLOWANCE_BYTES: u64 = 2 << 30;
const HEADROOM_RESERVE_BYTES: u64 = 16 << 30;

#[derive(Parser, Debug)]
pub struct PrefixCacheVtAbArgs {
    #[arg(short = 'm', long)]
    model: PathBuf,
    #[arg(long)]
    prefix_len: usize,
    #[arg(long)]
    chunk_len: usize,
    #[arg(long)]
    physical_memory_bytes: u64,
    #[arg(long)]
    preflight_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SectionDigest {
    name: &'static str,
    bytes: usize,
    sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SnapshotDigest {
    fingerprint: String,
    snapshot_bytes: u64,
    prefix_len: usize,
    pending_token: Option<i32>,
    kv_n_pos: Vec<usize>,
    sections: Vec<SectionDigest>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ContinuationDigest {
    token_ids: Vec<i32>,
    logits_sha256: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct FileIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
}

struct ArmOutcome {
    sequence: Sequence,
    logits: Vec<f32>,
    record: Value,
}

fn emit(record: Value) -> Result<()> {
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "VT_PRODUCT_JSON {record}")?;
    stdout.flush()?;
    Ok(())
}

fn checked_add(name: &'static str, lhs: u64, rhs: u64) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| anyhow!("{name} addition overflow: {lhs} + {rhs}"))
}

fn checked_mul(name: &'static str, lhs: u64, rhs: u64) -> Result<u64> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| anyhow!("{name} multiplication overflow: {lhs} * {rhs}"))
}

fn post_suffix_snapshot_bytes(
    prefix_snapshot_bytes: u64,
    identity: &SnapshotIdentity,
    chunk_len: usize,
) -> Result<u64> {
    let chunk = u64::try_from(chunk_len).context("chunk length does not fit u64")?;
    let kv_extra = checked_mul(
        "post-suffix KV rows",
        checked_mul(
            "post-suffix KV layers",
            u64::from(identity.n_attn_layers),
            chunk,
        )?,
        u64::from(identity.kv_bytes_per_token),
    )?;
    let kv_extra = checked_mul("post-suffix K+V", kv_extra, 2)?;
    let token_extra = checked_mul("post-suffix token IDs", chunk, 4)?;
    checked_add(
        "post-suffix snapshot bytes",
        checked_add("post-suffix snapshot KV", prefix_snapshot_bytes, kv_extra)?,
        token_extra,
    )
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_f32(values: &[f32]) -> String {
    sha256_bytes(bytemuck::cast_slice(values))
}

fn file_identity(file: &File) -> Result<FileIdentity> {
    let metadata = file.metadata().context("stat authenticated model handle")?;
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        bytes: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
    })
}

fn path_identity(path: &Path) -> Result<FileIdentity> {
    let file = File::open(path).with_context(|| format!("open {} for identity", path.display()))?;
    file_identity(&file)
}

fn sha256_open_file(file: &mut File) -> Result<String> {
    file.seek(SeekFrom::Start(0))
        .context("seek authenticated model handle")?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let read = file.read(&mut buffer).context("read authenticated model")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn digest_word(digest: &str, start: usize) -> Result<u64> {
    let end = start
        .checked_add(16)
        .ok_or_else(|| anyhow!("digest word range overflow"))?;
    let word = digest
        .get(start..end)
        .ok_or_else(|| anyhow!("digest is too short for word at {start}"))?;
    u64::from_str_radix(word, 16).with_context(|| format!("parse digest word {word}"))
}

fn identity_bytes(identity: &SnapshotIdentity) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(48);
    bytes.extend_from_slice(&identity.model_id.to_le_bytes());
    bytes.extend_from_slice(&identity.tokenizer_id.to_le_bytes());
    bytes.extend_from_slice(&identity.layout_version.to_le_bytes());
    bytes.extend_from_slice(&identity.n_attn_layers.to_le_bytes());
    bytes.extend_from_slice(&identity.n_gdn_layers.to_le_bytes());
    bytes.extend_from_slice(&identity.kv_dim_elements.to_le_bytes());
    bytes.extend_from_slice(&identity.kv_bytes_per_token.to_le_bytes());
    bytes.extend_from_slice(&(identity.kv_storage_kind as u32).to_le_bytes());
    bytes.extend_from_slice(&identity.gdn_state_elements_per_layer.to_le_bytes());
    bytes.extend_from_slice(&identity.gdn_conv_elements_per_layer.to_le_bytes());
    bytes
}

fn digest_snapshot(snapshot: &SessionSnapshot) -> SnapshotDigest {
    let mut kv_n_pos_bytes = Vec::with_capacity(snapshot.kv_n_pos.len() * 8);
    for &value in &snapshot.kv_n_pos {
        kv_n_pos_bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
    let pending_bytes = snapshot.pending_token.map(i32::to_le_bytes);
    let pending_slice = pending_bytes
        .as_ref()
        .map(<[u8; 4]>::as_slice)
        .unwrap_or_default();
    let final_logits_bytes = snapshot
        .final_logits
        .as_deref()
        .map(bytemuck::cast_slice::<f32, u8>)
        .unwrap_or_default();
    let sections = vec![
        SectionDigest {
            name: "identity",
            bytes: identity_bytes(&snapshot.identity).len(),
            sha256: sha256_bytes(&identity_bytes(&snapshot.identity)),
        },
        SectionDigest {
            name: "prefix_tokens",
            bytes: snapshot.prefix_tokens.len() * std::mem::size_of::<i32>(),
            sha256: sha256_bytes(bytemuck::cast_slice(&snapshot.prefix_tokens)),
        },
        SectionDigest {
            name: "pending_token",
            bytes: pending_slice.len(),
            sha256: sha256_bytes(pending_slice),
        },
        SectionDigest {
            name: "kv_n_pos",
            bytes: kv_n_pos_bytes.len(),
            sha256: sha256_bytes(&kv_n_pos_bytes),
        },
        SectionDigest {
            name: "kv_k_arena",
            bytes: snapshot.kv_k_arena.len(),
            sha256: sha256_bytes(&snapshot.kv_k_arena),
        },
        SectionDigest {
            name: "kv_v_arena",
            bytes: snapshot.kv_v_arena.len(),
            sha256: sha256_bytes(&snapshot.kv_v_arena),
        },
        SectionDigest {
            name: "gdn_conv_arena",
            bytes: snapshot.gdn_conv_arena.len(),
            sha256: sha256_bytes(&snapshot.gdn_conv_arena),
        },
        SectionDigest {
            name: "gdn_state_arena",
            bytes: snapshot.gdn_state_arena.len(),
            sha256: sha256_bytes(&snapshot.gdn_state_arena),
        },
        SectionDigest {
            name: "final_logits",
            bytes: final_logits_bytes.len(),
            sha256: sha256_bytes(final_logits_bytes),
        },
    ];
    let mut fingerprint = Sha256::new();
    for section in &sections {
        fingerprint.update((section.name.len() as u64).to_le_bytes());
        fingerprint.update(section.name.as_bytes());
        fingerprint.update((section.bytes as u64).to_le_bytes());
        fingerprint.update(section.sha256.as_bytes());
    }
    SnapshotDigest {
        fingerprint: format!("{:x}", fingerprint.finalize()),
        snapshot_bytes: snapshot.n_bytes(),
        prefix_len: snapshot.prefix_len(),
        pending_token: snapshot.pending_token,
        kv_n_pos: snapshot.kv_n_pos.clone(),
        sections,
    }
}

fn exact_token_ids(initial: Vec<i32>, filler: &[i32], target: usize) -> Result<Vec<i32>> {
    if target == 0 {
        bail!("target token count must be positive");
    }
    if filler.is_empty() {
        bail!("tokenizer produced an empty filler sequence");
    }
    let mut ids = initial;
    while ids.len() < target {
        let remaining = target - ids.len();
        ids.extend_from_slice(&filler[..remaining.min(filler.len())]);
    }
    ids.truncate(target);
    Ok(ids)
}

fn identity_json(identity: &SnapshotIdentity) -> Value {
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

fn plan_json(plan: &PrefillScratchPlan) -> Result<Value> {
    let allocations = plan
        .allocations()
        .iter()
        .map(|allocation| {
            json!({
                "name": allocation.name(),
                "logical_bytes": allocation.logical_bytes(),
                "dtype": format!("{:?}", allocation.dtype()),
            })
        })
        .collect::<Vec<_>>();
    let deferred = plan
        .deferred_allocations()
        .iter()
        .map(|allocation| {
            json!({
                "name": allocation.name(),
                "logical_bytes": allocation.logical_bytes(),
                "dtype": format!("{:?}", allocation.dtype()),
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "block_size": plan.block_size(),
        "matrix_max_pos": plan.matrix_max_pos(),
        "matrix_query_rows": plan.matrix_query_rows(),
        "logical_bytes": plan.logical_bytes(),
        "maximum_logical_bytes": plan.maximum_logical_bytes()?,
        "allocation_count": plan.allocation_count(),
        "deferred_allocation_count": plan.deferred_allocations().len(),
        "allocations": allocations,
        "deferred_allocations": deferred,
    }))
}

fn validate_topology(
    capture: &AttnMatrixVtDispatchCapture,
    compact: bool,
    prefix_len: usize,
    chunk_len: usize,
) -> Result<()> {
    let prefix = u64::try_from(prefix_len).context("prefix length does not fit u64")?;
    let chunk = u64::try_from(chunk_len).context("chunk length does not fit u64")?;
    let kv_dim = checked_mul(
        "KV dimension",
        EXPECTED_N_KV as u64,
        EXPECTED_HEAD_DIM as u64,
    )?;
    let rows = checked_mul("V_T row sum", EXPECTED_ATTN_LAYERS, prefix)?;
    let elements = checked_mul("V_T element sum", rows, kv_dim)?;
    let n_pos_sum = checked_mul(
        "V_T n_pos sum",
        EXPECTED_ATTN_LAYERS,
        checked_add("P+C", prefix, chunk)?,
    )?;
    let compact_groups_per_layer = checked_mul("compact groups", EXPECTED_N_KV as u64, prefix)?;
    let expected_groups = if compact {
        checked_mul(
            "compact group sum",
            EXPECTED_ATTN_LAYERS,
            compact_groups_per_layer,
        )?
    } else {
        elements
    };
    let expected = qwen_llm::metal::AttnMatrixVtDispatchStats {
        calls: EXPECTED_ATTN_LAYERS,
        row_sum: rows,
        element_sum: elements,
        threadgroup_sum: expected_groups,
        compact_calls: if compact { EXPECTED_ATTN_LAYERS } else { 0 },
        legacy_calls: if compact { 0 } else { EXPECTED_ATTN_LAYERS },
        base_pos_sum: 0,
        n_pos_sum,
    };
    if capture.stats != expected {
        bail!(
            "V_T topology mismatch for compact={compact}: actual={:?} expected={expected:?}",
            capture.stats
        );
    }
    Ok(())
}

fn arm_name(compact: bool) -> &'static str {
    if compact { "B" } else { "A" }
}

#[allow(clippy::too_many_arguments)]
fn execute_arm(
    loaded: &LoadedModel,
    mf: &MetalForward<'_>,
    snapshot: &SessionSnapshot,
    identity: &SnapshotIdentity,
    suffix_ids: &[i32],
    prefix_len: usize,
    chunk_len: usize,
    capacity: usize,
    scratch: &mut MetalDFlashLayerMajorScratch,
    compact: bool,
    bank: &'static str,
    kind: &'static str,
    pair: usize,
    order: &'static str,
    sequence_index: usize,
    snapshot_digest: &SnapshotDigest,
    scratch_plan: &Value,
) -> Result<ArmOutcome> {
    let request_start = Instant::now();
    let create_start = Instant::now();
    let mut sequence = loaded.create_sequence(SequenceConfig::new(capacity))?;
    let sequence_create_ms = create_start.elapsed().as_secs_f64() * 1e3;

    let restore_start = Instant::now();
    unsafe {
        sequence
            .metal_session_mut()
            .restore_from(snapshot, identity)?;
    }
    let restore_ms = restore_start.elapsed().as_secs_f64() * 1e3;
    sequence.advance_by(prefix_len)?;

    let post_restore_start = Instant::now();
    let scoped = with_attn_matrix_vt_compact_dispatch_override(compact, || {
        capture_attn_matrix_vt_dispatches(|| {
            let suffix_start = Instant::now();
            let result = prefill_tokens_with_multi_hidden_profiled(
                mf,
                suffix_ids,
                u32::try_from(prefix_len).expect("validated prefix fits u32"),
                unsafe { sequence.metal_session_mut() },
                scratch,
                &[],
                None,
            );
            (result, suffix_start.elapsed().as_secs_f64() * 1e3)
        })
    })?;
    let ((prefill_result, suffix_wall_ms), capture) = scoped?;
    let (logits, suffix_gpu_ms) =
        prefill_result.with_context(|| format!("{} suffix prefill failed", arm_name(compact)))?;
    validate_topology(&capture, compact, prefix_len, chunk_len)?;
    sequence.advance_by(chunk_len)?;
    let first_token = logits
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(index, _)| index as i32)
        .ok_or_else(|| anyhow!("empty logits"))?;
    let post_restore_ttft_ms = post_restore_start.elapsed().as_secs_f64() * 1e3;
    let request_wall_ms = request_start.elapsed().as_secs_f64() * 1e3;
    let logits_sha256 = sha256_f32(&logits);

    Ok(ArmOutcome {
        sequence,
        logits,
        record: json!({
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "phase": kind,
            "cell": format!("P{prefix_len}/C{chunk_len}"),
            "prefix": prefix_len,
            "chunk": chunk_len,
            "role": arm_name(compact),
            "compact": compact,
            "bank": bank,
            "pair": pair,
            "order": order,
            "sequence": sequence_index,
            "capacity": capacity,
            "sequence_create_ms": sequence_create_ms,
            "restore_ms": restore_ms,
            "suffix_wall_ms": suffix_wall_ms,
            "suffix_gpu_ms": suffix_gpu_ms,
            "post_restore_ttft_ms": post_restore_ttft_ms,
            "request_wall_ms": request_wall_ms,
            "first_token": first_token,
            "logits_sha256": logits_sha256,
            "snapshot_fingerprint": snapshot_digest.fingerprint,
            "snapshot_bytes": snapshot_digest.snapshot_bytes,
            "snapshot_identity": identity_json(identity),
            "snapshot": snapshot_digest,
            "scratch_plan": scratch_plan,
            "capture_owner_thread": capture.owner_thread,
            "vt_dispatch": {
                "calls": capture.stats.calls,
                "row_sum": capture.stats.row_sum,
                "element_sum": capture.stats.element_sum,
                "threadgroup_sum": capture.stats.threadgroup_sum,
                "compact_calls": capture.stats.compact_calls,
                "legacy_calls": capture.stats.legacy_calls,
                "base_pos_sum": capture.stats.base_pos_sum,
                "n_pos_sum": capture.stats.n_pos_sum,
            },
        }),
    })
}

fn post_suffix_snapshot(
    outcome: &ArmOutcome,
    identity: &SnapshotIdentity,
    full_ids: &[i32],
) -> Result<SnapshotDigest> {
    let snapshot = outcome.sequence.metal_session().snapshot(
        identity.clone(),
        full_ids.to_vec(),
        Some(outcome.logits.clone()),
    )?;
    Ok(digest_snapshot(&snapshot))
}

fn continue_greedy(
    mf: &MetalForward<'_>,
    outcome: &mut ArmOutcome,
    start_position: usize,
) -> Result<ContinuationDigest> {
    let mut logits = outcome.logits.clone();
    let mut token_ids = Vec::with_capacity(CONTINUATION_STEPS);
    let mut logits_sha256 = Vec::with_capacity(CONTINUATION_STEPS);
    for step in 0..CONTINUATION_STEPS {
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(index, _)| index as i32)
            .ok_or_else(|| anyhow!("empty continuation logits"))?;
        token_ids.push(token);
        logits = mf.single_token(
            token,
            u32::try_from(start_position + step)
                .context("continuation position does not fit u32")?,
            unsafe { outcome.sequence.metal_session_mut() },
        )?;
        outcome.sequence.advance_by(1)?;
        logits_sha256.push(sha256_f32(&logits));
    }
    Ok(ContinuationDigest {
        token_ids,
        logits_sha256,
    })
}

fn run_correctness(
    loaded: &LoadedModel,
    mf: &MetalForward<'_>,
    snapshot: &SessionSnapshot,
    identity: &SnapshotIdentity,
    prefix_ids: &[i32],
    suffix_ids: &[i32],
    capacity: usize,
    scratch_x: &mut MetalDFlashLayerMajorScratch,
    scratch_y: &mut MetalDFlashLayerMajorScratch,
    snapshot_digest: &SnapshotDigest,
    plan: &Value,
) -> Result<String> {
    let prefix_len = prefix_ids.len();
    let chunk_len = suffix_ids.len();
    let full_ids = prefix_ids
        .iter()
        .chain(suffix_ids)
        .copied()
        .collect::<Vec<_>>();
    let mut arm_a = execute_arm(
        loaded,
        mf,
        snapshot,
        identity,
        suffix_ids,
        prefix_len,
        chunk_len,
        capacity,
        scratch_x,
        false,
        "X",
        "correctness_arm",
        0,
        "AB",
        1,
        snapshot_digest,
        plan,
    )?;
    let state_a = post_suffix_snapshot(&arm_a, identity, &full_ids)?;
    let continuation_a = continue_greedy(mf, &mut arm_a, prefix_len + chunk_len)?;
    let record_a = arm_a.record.clone();
    drop(arm_a);

    let mut arm_b = execute_arm(
        loaded,
        mf,
        snapshot,
        identity,
        suffix_ids,
        prefix_len,
        chunk_len,
        capacity,
        scratch_y,
        true,
        "Y",
        "correctness_arm",
        0,
        "AB",
        2,
        snapshot_digest,
        plan,
    )?;
    let state_b = post_suffix_snapshot(&arm_b, identity, &full_ids)?;
    let continuation_b = continue_greedy(mf, &mut arm_b, prefix_len + chunk_len)?;
    let record_b = arm_b.record.clone();
    let logits_digest = sha256_f32(&arm_b.logits);
    drop(arm_b);

    if record_a["logits_sha256"] != record_b["logits_sha256"]
        || record_a["first_token"] != record_b["first_token"]
        || state_a != state_b
        || continuation_a != continuation_b
    {
        bail!("legacy/compact model-backed correctness mismatch");
    }
    emit(record_a)?;
    emit(record_b)?;
    emit(json!({
        "schema_version": SCHEMA_VERSION,
        "kind": "correctness",
        "prefix": prefix_len,
        "chunk": chunk_len,
        "match": true,
        "suffix_logits_sha256": logits_digest,
        "state": state_a,
        "continuation": continuation_a,
    }))?;
    Ok(logits_digest)
}

fn run_scheduled_arm(
    loaded: &LoadedModel,
    mf: &MetalForward<'_>,
    snapshot: &SessionSnapshot,
    identity: &SnapshotIdentity,
    suffix_ids: &[i32],
    prefix_len: usize,
    capacity: usize,
    scratch_x: &mut MetalDFlashLayerMajorScratch,
    scratch_y: &mut MetalDFlashLayerMajorScratch,
    compact: bool,
    bank: &'static str,
    kind: &'static str,
    pair: usize,
    order: &'static str,
    sequence_index: usize,
    snapshot_digest: &SnapshotDigest,
    plan: &Value,
    expected_logits_sha256: &str,
) -> Result<()> {
    let scratch = match bank {
        "X" => scratch_x,
        "Y" => scratch_y,
        _ => bail!("invalid scratch bank {bank}"),
    };
    let outcome = execute_arm(
        loaded,
        mf,
        snapshot,
        identity,
        suffix_ids,
        prefix_len,
        suffix_ids.len(),
        capacity,
        scratch,
        compact,
        bank,
        kind,
        pair,
        order,
        sequence_index,
        snapshot_digest,
        plan,
    )?;
    if outcome.record["logits_sha256"] != expected_logits_sha256 {
        bail!("{kind} {} logits digest drifted", arm_name(compact));
    }
    emit(outcome.record)?;
    Ok(())
}

pub fn run(args: PrefixCacheVtAbArgs, build_identity: Value) -> Result<()> {
    let PrefixCacheVtAbArgs {
        model,
        prefix_len,
        chunk_len,
        physical_memory_bytes,
        preflight_only,
    } = args;
    if !matches!(
        (prefix_len, chunk_len),
        (8192, 128) | (16_384, 128) | (16_384, 1024)
    ) {
        bail!("geometry P{prefix_len}/C{chunk_len} is outside the frozen campaign");
    }
    let forbidden_environment = std::env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .filter(|name| {
            name == "RUST_LOG"
                || name.starts_with("QWEN")
                || name.starts_with("MTL")
                || name.starts_with("METAL")
        })
        .collect::<Vec<_>>();
    if !forbidden_environment.is_empty() {
        bail!("forbidden diagnostic/treatment environment is present: {forbidden_environment:?}");
    }
    let mut model_handle =
        File::open(&model).with_context(|| format!("open {}", model.display()))?;
    let model_identity = file_identity(&model_handle)?;
    let model_bytes = model_identity.bytes;
    if model_bytes != EXPECTED_MODEL_BYTES {
        bail!("model bytes {model_bytes} != pinned {EXPECTED_MODEL_BYTES}");
    }
    let model_sha256 = sha256_open_file(&mut model_handle)?;
    if model_sha256 != EXPECTED_MODEL_SHA256 {
        bail!("model SHA-256 {model_sha256} != pinned {EXPECTED_MODEL_SHA256}");
    }
    let snapshot_model_id = digest_word(&model_sha256, 0)?;
    let snapshot_tokenizer_id = digest_word(&model_sha256, 16)?;
    let capacity = prefix_len
        .checked_add(chunk_len)
        .and_then(|value| value.checked_add(CONTINUATION_STEPS))
        .ok_or_else(|| anyhow!("session capacity overflow"))?;

    let runtime = Runtime::metal()?;
    let loaded = runtime.load_model_with_config(
        &model,
        LoadedModelConfig {
            prefetch_policy: PrefetchPolicy::Off,
            ..LoadedModelConfig::default()
        },
    )?;
    let model_path_identity_after_load = path_identity(&model)?;
    if model_path_identity_after_load != model_identity {
        bail!(
            "model path identity changed across load: before={model_identity:?} after={model_path_identity_after_load:?}"
        );
    }
    let model_sha256_after_load = sha256_open_file(&mut model_handle)?;
    if model_sha256_after_load != model_sha256 {
        bail!("authenticated model handle changed across load");
    }
    let ctx = loaded.context();
    let mm = loaded.metal_model();
    let n_attn = mm
        .blocks
        .iter()
        .filter(|block| matches!(block, MetalBlock::Attn(_)))
        .count();
    if n_attn as u64 != EXPECTED_ATTN_LAYERS
        || mm.arch.n_q_heads != EXPECTED_N_Q
        || mm.arch.n_kv_heads != EXPECTED_N_KV
        || mm.arch.attn_head_dim != EXPECTED_HEAD_DIM
    {
        bail!(
            "unexpected model topology: attn={n_attn} n_q={} n_kv={} head_dim={}",
            mm.arch.n_q_heads,
            mm.arch.n_kv_heads,
            mm.arch.attn_head_dim
        );
    }
    let tokenizer = loaded.tokenizer()?;
    let prefix_ids = exact_token_ids(
        tokenizer.encode("You are a precise systems assistant.", false)?,
        &tokenizer.encode(
            " lorem ipsum dolor sit amet consectetur adipiscing elit",
            false,
        )?,
        prefix_len,
    )?;
    let suffix_ids = exact_token_ids(
        tokenizer.encode(
            "\n\nUser: Analyze the following deterministic trace.",
            false,
        )?,
        &tokenizer.encode(" packet sequence invariant latency evidence", false)?,
        chunk_len,
    )?;
    let prefix_token_sha256 = token_ids_sha256_i32le(&prefix_ids);
    let suffix_token_sha256 = token_ids_sha256_i32le(&suffix_ids);
    let mf = MetalForward::new(ctx, mm);

    let allocated_before_prefix = ctx.device.currentAllocatedSize() as u64;
    let mut prefix_scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        ctx,
        mm,
        PREFIX_BLOCK_SIZE,
        prefix_len,
    )?;
    let prefix_plan = prefix_scratch.prefill_scratch_plan();
    if prefix_plan.block_size() != PREFIX_BLOCK_SIZE
        || prefix_plan.matrix_max_pos() != prefix_len as u64
        || prefix_plan.matrix_query_rows() != PREFIX_BLOCK_SIZE
    {
        bail!(
            "prefix scratch geometry drifted: {}",
            plan_json(prefix_plan)?
        );
    }
    let prefix_plan_json = plan_json(prefix_plan)?;

    let session_alloc_before = ctx.device.currentAllocatedSize() as u64;
    let mut warm_sequence = loaded.create_sequence(SequenceConfig::new(capacity))?;
    let session_alloc_after = ctx.device.currentAllocatedSize() as u64;
    if session_alloc_after <= session_alloc_before {
        bail!(
            "Metal allocation counter did not increase across sequence creation: {session_alloc_before} -> {session_alloc_after}"
        );
    }
    let session_allocation_delta = session_alloc_after - session_alloc_before;
    let _ = mf.single_token(prefix_ids[0], 0, unsafe {
        warm_sequence.metal_session_mut()
    })?;
    drop(warm_sequence);

    let mut builder = loaded.create_sequence(SequenceConfig::new(capacity))?;
    let (prefix_logits, prefix_gpu_ms) = prefill_tokens_with_multi_hidden_profiled(
        &mf,
        &prefix_ids,
        0,
        unsafe { builder.metal_session_mut() },
        &mut prefix_scratch,
        &[],
        None,
    )?;
    builder.advance_by(prefix_len)?;
    let identity = builder
        .metal_session()
        .snapshot_identity(snapshot_model_id, snapshot_tokenizer_id);
    let snapshot = builder.metal_session().snapshot(
        identity.clone(),
        prefix_ids.clone(),
        Some(prefix_logits),
    )?;
    let snapshot_digest = digest_snapshot(&snapshot);
    drop(builder);
    drop(prefix_scratch);
    let snapshot_digest_after_drop = digest_snapshot(&snapshot);
    if snapshot_digest_after_drop != snapshot_digest {
        bail!("prefix snapshot changed after builder teardown");
    }

    let suffix_plan = plan_prefill_scratch_with_matrix_max_pos_configured(
        mm,
        u32::try_from(chunk_len).context("chunk length does not fit u32")?,
        prefix_len + chunk_len,
        PrefillScratchConfig::default(),
    )?;
    if suffix_plan.block_size() != chunk_len as u32
        || suffix_plan.matrix_max_pos() != (prefix_len + chunk_len) as u64
        || suffix_plan.matrix_query_rows() != chunk_len as u32
    {
        bail!(
            "suffix scratch geometry drifted: {}",
            plan_json(&suffix_plan)?
        );
    }
    let max_buffer_length = ctx.device.maxBufferLength() as u64;
    for allocation in suffix_plan
        .allocations()
        .iter()
        .chain(suffix_plan.deferred_allocations())
    {
        if allocation.logical_bytes() > max_buffer_length {
            bail!(
                "planned allocation {}={} exceeds maxBufferLength={max_buffer_length}",
                allocation.name(),
                allocation.logical_bytes()
            );
        }
    }
    let plan_max = suffix_plan.maximum_logical_bytes()?;
    let correctness_snapshot_bytes =
        post_suffix_snapshot_bytes(snapshot_digest.snapshot_bytes, &identity, chunk_len)?;
    let allocated_after_builder_drop = ctx.device.currentAllocatedSize() as u64;
    let mut peak_bound = allocated_after_builder_drop;
    for (name, bytes) in [
        ("model file", model_bytes),
        ("scratch X", plan_max),
        ("scratch Y", plan_max),
        ("session", session_allocation_delta),
        ("retained snapshot", snapshot_digest.snapshot_bytes),
        ("temporary snapshot", correctness_snapshot_bytes),
        ("allocator allowance", ALLOCATOR_ALLOWANCE_BYTES),
        ("headroom reserve", HEADROOM_RESERVE_BYTES),
    ] {
        peak_bound = checked_add(name, peak_bound, bytes)?;
    }
    if peak_bound > physical_memory_bytes {
        bail!("peak bound {peak_bound} exceeds physical memory {physical_memory_bytes}");
    }

    let allocated_before_x = ctx.device.currentAllocatedSize() as u64;
    let mut scratch_x =
        MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(ctx, mm, suffix_plan.clone())?;
    let allocated_after_x = ctx.device.currentAllocatedSize() as u64;
    let mut scratch_y =
        MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(ctx, mm, suffix_plan.clone())?;
    let allocated_after_y = ctx.device.currentAllocatedSize() as u64;
    if scratch_x.prefill_scratch_plan() != scratch_y.prefill_scratch_plan()
        || scratch_x.prefill_scratch_plan() != &suffix_plan
        || scratch_x.aliases_scratch(&scratch_y)
    {
        bail!("suffix scratch banks are not equal and disjoint");
    }
    let scratch_x_ids = scratch_x.mutable_buffer_ids();
    let scratch_y_ids = scratch_y.mutable_buffer_ids();
    let suffix_plan_json = plan_json(&suffix_plan)?;

    emit(json!({
        "schema_version": SCHEMA_VERSION,
        "kind": "setup",
        "build_identity": build_identity,
        "device": ctx.device.name().to_string(),
        "device_registry_id": ctx.device.registryID(),
        "max_buffer_length": max_buffer_length,
        "model": model,
        "model_bytes": model_bytes,
        "model_sha256": model_sha256,
        "model_sha256_after_load": model_sha256_after_load,
        "model_file_identity": model_identity,
        "model_path_identity_after_load": model_path_identity_after_load,
        "prefix": prefix_len,
        "chunk": chunk_len,
        "capacity": capacity,
        "prefix_token_sha256": prefix_token_sha256,
        "suffix_token_sha256": suffix_token_sha256,
        "prefix_gpu_ms": prefix_gpu_ms,
        "prefix_plan": prefix_plan_json,
        "suffix_plan": suffix_plan_json,
        "snapshot": snapshot_digest,
        "correctness_snapshot_bytes": correctness_snapshot_bytes,
        "session_allocation_delta": session_allocation_delta,
        "allocated_before_prefix": allocated_before_prefix,
        "allocated_after_builder_drop": allocated_after_builder_drop,
        "peak_bound": peak_bound,
        "physical_memory_bytes": physical_memory_bytes,
        "preflight_only": preflight_only,
        "allocated_before_x": allocated_before_x,
        "allocated_after_x": allocated_after_x,
        "allocated_after_y": allocated_after_y,
        "scratch_x_buffer_ids": scratch_x_ids,
        "scratch_y_buffer_ids": scratch_y_ids,
    }))?;

    let expected_logits_sha256 = run_correctness(
        &loaded,
        &mf,
        &snapshot,
        &identity,
        &prefix_ids,
        &suffix_ids,
        capacity,
        &mut scratch_x,
        &mut scratch_y,
        &snapshot_digest_after_drop,
        &plan_json(&suffix_plan)?,
    )?;
    if preflight_only {
        emit(json!({
            "schema_version": SCHEMA_VERSION,
            "kind": "preflight_complete",
            "prefix": prefix_len,
            "chunk": chunk_len,
            "expected_logits_sha256": expected_logits_sha256,
        }))?;
        return Ok(());
    }

    for (sequence_index, (compact, bank)) in [(false, "X"), (true, "Y"), (false, "Y"), (true, "X")]
        .into_iter()
        .enumerate()
    {
        run_scheduled_arm(
            &loaded,
            &mf,
            &snapshot,
            &identity,
            &suffix_ids,
            prefix_len,
            capacity,
            &mut scratch_x,
            &mut scratch_y,
            compact,
            bank,
            "warmup",
            0,
            "W",
            sequence_index + 3,
            &snapshot_digest_after_drop,
            &plan_json(&suffix_plan)?,
            &expected_logits_sha256,
        )?;
    }

    let schedule = [
        ((false, "X"), (true, "Y"), "AB"),
        ((true, "X"), (false, "Y"), "BA"),
        ((true, "Y"), (false, "X"), "BA"),
        ((false, "Y"), (true, "X"), "AB"),
        ((false, "X"), (true, "Y"), "AB"),
        ((true, "X"), (false, "Y"), "BA"),
    ];
    for (pair_index, (first, second, order)) in schedule.into_iter().enumerate() {
        for (sequence_index, (compact, bank)) in [first, second].into_iter().enumerate() {
            run_scheduled_arm(
                &loaded,
                &mf,
                &snapshot,
                &identity,
                &suffix_ids,
                prefix_len,
                capacity,
                &mut scratch_x,
                &mut scratch_y,
                compact,
                bank,
                "arm",
                pair_index + 1,
                order,
                7 + pair_index * 2 + sequence_index,
                &snapshot_digest_after_drop,
                &plan_json(&suffix_plan)?,
                &expected_logits_sha256,
            )?;
        }
    }
    emit(json!({
        "schema_version": SCHEMA_VERSION,
        "kind": "complete",
        "prefix": prefix_len,
        "chunk": chunk_len,
        "expected_logits_sha256": expected_logits_sha256,
    }))?;
    Ok(())
}
