//! Bench-only E0 lockstep development harness.
//!
//! This isolates `single_token` from `single_token_with_multi_hidden` and
//! grants no packed-verifier, product, serve, or performance authority.

use super::dflash_sampled_oracle::{
    JsonlAppender, SnapshotComparisons, bytes_sha256, digest_identity_word, distribution_json,
    event, f32_bits, f64_bits, gguf_asset_identity, logits_sha256, position_u32,
    regular_file_identity, rng_json, snapshot_json, u64_hex, validate_development_label,
};
use anyhow::{Context, Result, anyhow, ensure};
use clap::{Parser, ValueEnum};
use objc2_metal::MTLBuffer;
use qwen_llm::{
    gguf::GgufFile,
    loader::{Model, open_dflash_drafter},
    metal::{MetalContext, MetalTensor},
    metal_dflash::{MetalDFlashHead, MetalDFlashSession},
    metal_forward::{MetalForward, MetalModel, MetalSession, SessionSnapshot, SnapshotIdentity},
    sampling::{
        SAMPLER_ALGORITHM_VERSION, SampledToken, Sampler, SamplingConfig, SamplingDistribution,
        SamplingRngDiagnostic,
    },
    tokenizer::{Tokenizer, token_ids_sha256_i32le},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fmt::Write as _,
    fs::{File, OpenOptions},
    io::Write as _,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const SCHEMA: &str = "qwen.dflash_e0_lockstep";
const SCHEMA_VERSION: u32 = 1;
const DEVELOPMENT_SEMANTICS: &str = "token_major_serial_vs_multi_hidden_lockstep_development_only";
const HIDDEN_TRANSFER_SEMANTICS: &str =
    "poison_then_capture_exact_source_to_active_dflash_context_row";
const BINDING_MANIFEST_SCHEMA: &str = "qwen.dflash_e0_binding_manifest";
const BINDING_MANIFEST_VERSION_V1: u64 = 1;
const BINDING_MANIFEST_VERSION_V2: u64 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ArmOrder {
    SerialThenCapture,
    CaptureThenSerial,
}

impl ArmOrder {
    fn as_str(self) -> &'static str {
        match self {
            Self::SerialThenCapture => "serial_then_capture",
            Self::CaptureThenSerial => "capture_then_serial",
        }
    }
}

fn validate_e0_build_identity(value: &Value) -> Result<()> {
    let object = value
        .as_object()
        .context("E0 build identity must be an object")?;
    let build_commit = object
        .get("build_commit")
        .and_then(Value::as_str)
        .context("E0 requires a concrete build commit")?;
    let runtime_commit = object
        .get("runtime_commit")
        .and_then(Value::as_str)
        .context("E0 requires a concrete runtime commit")?;
    ensure!(
        build_commit.len() == 40
            && build_commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            && build_commit == runtime_commit,
        "E0 build/runtime commit identity is invalid or mismatched"
    );
    let build_dirty = object
        .get("build_dirty")
        .and_then(Value::as_bool)
        .context("E0 requires a concrete build dirty state")?;
    let runtime_dirty = object
        .get("runtime_dirty")
        .and_then(Value::as_bool)
        .context("E0 requires a concrete runtime dirty state")?;
    ensure!(
        build_dirty == runtime_dirty,
        "E0 build/runtime dirty identity mismatch"
    );
    let build_source = object
        .get("build_source_state")
        .and_then(Value::as_str)
        .context("E0 requires a concrete build source state")?;
    let runtime_source = object
        .get("runtime_source_state")
        .and_then(Value::as_str)
        .context("E0 requires a concrete runtime source state")?;
    ensure!(
        build_source.starts_with("git-source-sha256-v2:") && build_source == runtime_source,
        "E0 build/runtime source identity mismatch"
    );
    let problems = object
        .get("problems")
        .and_then(Value::as_array)
        .context("E0 requires build-identity problems")?;
    let overrides = object
        .get("overrides")
        .and_then(Value::as_array)
        .context("E0 requires build-identity overrides")?;
    ensure!(
        object.get("status").and_then(Value::as_str) == Some("match")
            && !build_dirty
            && !runtime_dirty
            && object.get("stamp_error").is_some_and(Value::is_null)
            && problems.is_empty()
            && overrides.is_empty(),
        "E0 acquisition requires a clean, matching build without overrides"
    );
    Ok(())
}

fn validate_distinct_artifact_paths(paths: &[(&str, &str)]) -> Result<()> {
    for (index, (left_label, left_path)) in paths.iter().enumerate() {
        for (right_label, right_path) in &paths[index + 1..] {
            ensure!(
                left_path != right_path,
                "E0 artifact paths must be distinct: {left_label} aliases {right_label}"
            );
        }
    }
    Ok(())
}

fn evidence_paths_json(
    model: &str,
    drafter: &str,
    binding_manifest: &str,
    output: &str,
    state_sidecar: &str,
    executable: &str,
) -> Value {
    json!({
        "model": model,
        "drafter": drafter,
        "binding_manifest": binding_manifest,
        "output": output,
        "state_sidecar": state_sidecar,
        "executable": executable,
    })
}

fn load_binding_manifest(path: &std::path::Path) -> Result<(Value, Value)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read E0 binding manifest {}", path.display()))?;
    let manifest: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse E0 binding manifest {}", path.display()))?;
    let identity = json!({
        "path": stable_path_identity(path),
        "bytes": bytes.len(),
        "sha256": bytes_sha256(&bytes),
    });
    Ok((manifest, identity))
}

fn validate_binding_manifest_arm_order(
    manifest: &Value,
    version: u64,
    arm_order: ArmOrder,
) -> Result<()> {
    let allowed = manifest["allowed_arm_orders"]
        .as_array()
        .context("E0 binding manifest arm orders must be an array")?;
    ensure!(
        allowed.len() == 2
            && allowed[0] == "serial_then_capture"
            && allowed[1] == "capture_then_serial"
            && allowed.iter().any(|value| value == arm_order.as_str()),
        "E0 binding manifest arm-order contract mismatch"
    );
    if version == BINDING_MANIFEST_VERSION_V2 {
        ensure!(
            manifest["required_arm_order"] == arm_order.as_str(),
            "E0 binding manifest required arm order mismatch"
        );
    }
    Ok(())
}

fn validate_binding_manifest_static(
    manifest: &Value,
    target_asset: &Value,
    drafter_asset: &Value,
    target_g: &GgufFile,
    target_m: &Model<'_>,
    head: &qwen_llm::loader::DFlashHead<'_>,
    args: &DflashE0LockstepArgs,
) -> Result<()> {
    let object = manifest
        .as_object()
        .context("E0 binding manifest must be an object")?;
    let version = manifest["schema_version"]
        .as_u64()
        .context("E0 binding manifest schema version must be an integer")?;
    ensure!(
        matches!(
            version,
            BINDING_MANIFEST_VERSION_V1 | BINDING_MANIFEST_VERSION_V2
        ),
        "unsupported E0 binding manifest schema version"
    );
    let mut expected_keys = vec![
        "schema",
        "schema_version",
        "evidence_role",
        "fixture_id",
        "fixture_role",
        "target_asset_sha256",
        "drafter_asset_sha256",
        "target_arm",
        "drafter_arm",
        "target",
        "drafter",
        "snapshot_abi",
        "allowed_arm_orders",
    ];
    if version == BINDING_MANIFEST_VERSION_V2 {
        expected_keys.push("required_arm_order");
    }
    ensure!(
        object.len() == expected_keys.len()
            && expected_keys.iter().all(|key| object.contains_key(*key)),
        "E0 binding manifest keys do not match its schema version"
    );
    ensure!(
        manifest["schema"] == BINDING_MANIFEST_SCHEMA && manifest["evidence_role"] == "development",
        "E0 binding manifest schema or authority is invalid"
    );
    ensure!(
        manifest["target_asset_sha256"] == target_asset["aggregate_sha256_index_size_digest_le"]
            && manifest["drafter_asset_sha256"]
                == drafter_asset["aggregate_sha256_index_size_digest_le"],
        "E0 binding manifest asset identity mismatch"
    );
    ensure!(
        manifest["fixture_id"] == args.fixture_id
            && manifest["fixture_role"] == args.fixture_role
            && manifest["target_arm"] == args.target_arm
            && manifest["drafter_arm"] == args.drafter_arm,
        "E0 binding manifest fixture or arm label mismatch"
    );
    let target = json!({
        "architecture": target_g.architecture(),
        "n_layer": target_m.arch.n_layer,
        "hidden_size": target_m.arch.hidden_size,
        "vocab_size": target_m.arch.vocab_size,
    });
    let drafter = json!({
        "n_layer": head.config.n_layer,
        "hidden_size": head.config.hidden_size,
        "block_size": head.config.block_size,
        "swa_window": head.config.swa_window,
        "conv_kernel_size": head.config.conv_kernel_size,
        "conv_group_size": head.config.conv_group_size,
        "selector_rank": head.config.selector_rank,
        "selector_top_k": head.config.selector_top_k,
        "target_layer_ids": head.target_layer_ids,
    });
    ensure!(
        manifest["target"] == target && manifest["drafter"] == drafter,
        "E0 binding manifest target/drafter geometry mismatch"
    );
    validate_binding_manifest_arm_order(manifest, version, args.arm_order)?;
    Ok(())
}

fn validate_binding_manifest_snapshot(manifest: &Value, identity: &SnapshotIdentity) -> Result<()> {
    let observed = json!({
        "layout_version": identity.layout_version,
        "n_attn_layers": identity.n_attn_layers,
        "n_gdn_layers": identity.n_gdn_layers,
        "kv_dim_elements": identity.kv_dim_elements,
        "kv_bytes_per_token": identity.kv_bytes_per_token,
        "kv_storage_kind": format!("{:?}", identity.kv_storage_kind),
        "gdn_state_elements_per_layer": identity.gdn_state_elements_per_layer,
        "gdn_conv_elements_per_layer": identity.gdn_conv_elements_per_layer,
    });
    ensure!(
        manifest["snapshot_abi"] == observed,
        "E0 binding manifest snapshot ABI mismatch"
    );
    Ok(())
}

#[derive(Parser, Debug)]
pub struct DflashE0LockstepArgs {
    /// Path to the target GGUF.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Path to the DFlash 2 drafter GGUF that defines capture layers.
    #[arg(long)]
    drafter: PathBuf,
    /// Prospective binding manifest frozen before this E0 acquisition.
    #[arg(long)]
    binding_manifest: PathBuf,
    /// Prompt text (must tokenize to at least one token).
    #[arg(short = 'p', long)]
    prompt: String,
    /// Maximum generated token count.
    #[arg(long, default_value = "4")]
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
    /// Explicit A/B execution order; reverse order is a development sentinel.
    #[arg(long, value_enum, default_value = "serial-then-capture")]
    arm_order: ArmOrder,
    /// Append-only JSONL evidence path.
    #[arg(short = 'o', long)]
    output: PathBuf,
    /// Exclusive-create binary sidecar for exact per-step target state bytes.
    #[arg(long)]
    state_sidecar: PathBuf,
    /// Skip the isolated target warmup calls.
    #[arg(long)]
    no_warmup: bool,
    #[arg(long, default_value = "development-unclassified")]
    fixture_id: String,
    #[arg(long, default_value = "development-unclassified")]
    fixture_role: String,
    #[arg(long, default_value = "development-unclassified")]
    target_arm: String,
    #[arg(long, default_value = "development-unclassified")]
    drafter_arm: String,
}

struct StateSidecar {
    file: File,
    path: PathBuf,
    bytes: u64,
    digest: Sha256,
}

impl StateSidecar {
    fn create(path: PathBuf) -> Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("exclusive-create E0 sidecar {}", path.display()))?;
        Ok(Self {
            file,
            path,
            bytes: 0,
            digest: Sha256::new(),
        })
    }

    fn append(&mut self, bytes: &[u8]) -> Result<Value> {
        let offset = self.bytes;
        self.file.write_all(bytes)?;
        self.digest.update(bytes);
        self.bytes = self
            .bytes
            .checked_add(u64::try_from(bytes.len()).context("sidecar write length exceeds u64")?)
            .context("sidecar length overflows")?;
        Ok(json!({
            "offset": offset,
            "bytes": bytes.len(),
            "sha256": bytes_sha256(bytes),
        }))
    }

    fn finish(&mut self) -> Result<Value> {
        self.file.flush()?;
        self.file.sync_all()?;
        Ok(json!({
            "path": stable_path_identity(&self.path),
            "bytes": self.bytes,
            "sha256": format!("{:x}", self.digest.clone().finalize()),
        }))
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

fn bytes_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn stable_path_identity(path: &std::path::Path) -> String {
    if let Ok(canonical) = path.canonicalize() {
        return canonical.display().to_string();
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    if let (Some(parent), Some(name)) = (absolute.parent(), absolute.file_name())
        && let Ok(canonical_parent) = parent.canonicalize()
    {
        return canonical_parent.join(name).display().to_string();
    }
    absolute.display().to_string()
}

fn f32le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

#[cfg(test)]
fn f32le_hex(values: &[f32]) -> String {
    bytes_hex(&f32le_bytes(values))
}

fn first_byte_mismatch(a: &[u8], b: &[u8]) -> Option<Value> {
    let common = a.len().min(b.len());
    for index in 0..common {
        if a[index] != b[index] {
            return Some(json!({
                "offset": index,
                "serial_byte": a[index],
                "capture_byte": b[index],
            }));
        }
    }
    (a.len() != b.len()).then(|| {
        json!({
            "offset": common,
            "serial_len": a.len(),
            "capture_len": b.len(),
        })
    })
}

fn first_logit_mismatch(a: &[f32], b: &[f32]) -> Option<Value> {
    let common = a.len().min(b.len());
    for index in 0..common {
        if a[index].to_bits() != b[index].to_bits() {
            return Some(json!({
                "index": index,
                "serial_f32_bits": f32_bits(a[index]),
                "capture_f32_bits": f32_bits(b[index]),
            }));
        }
    }
    (a.len() != b.len()).then(|| {
        json!({
            "index": common,
            "serial_len": a.len(),
            "capture_len": b.len(),
        })
    })
}

fn logits_bit_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(serial, capture)| serial.to_bits() == capture.to_bits())
}

fn distribution_bit_equal(a: &SamplingDistribution, b: &SamplingDistribution) -> bool {
    a.sampled == b.sampled
        && a.total_weight.to_bits() == b.total_weight.to_bits()
        && a.candidates.len() == b.candidates.len()
        && a.candidates
            .iter()
            .zip(&b.candidates)
            .all(|(a, b)| a.token == b.token && a.weight.to_bits() == b.weight.to_bits())
}

fn rng_bit_equal(a: Option<SamplingRngDiagnostic>, b: Option<SamplingRngDiagnostic>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.draws_before == b.draws_before
                && a.state_before == b.state_before
                && a.state_after == b.state_after
                && a.raw_u64 == b.raw_u64
                && a.unit_f64_bits == b.unit_f64_bits
                && a.unit_f64.to_bits() == b.unit_f64.to_bits()
        }
        _ => false,
    }
}

fn first_distribution_mismatch(
    a: &SamplingDistribution,
    b: &SamplingDistribution,
) -> Option<Value> {
    if a.sampled != b.sampled {
        return Some(json!({
            "field": "sampled",
            "serial": {"token": a.sampled.token, "candidate_index": a.sampled.candidate_index},
            "capture": {"token": b.sampled.token, "candidate_index": b.sampled.candidate_index},
        }));
    }
    if a.total_weight.to_bits() != b.total_weight.to_bits() {
        return Some(json!({
            "field": "total_weight_f64_bits",
            "serial": u64_hex(a.total_weight.to_bits()),
            "capture": u64_hex(b.total_weight.to_bits()),
        }));
    }
    let common = a.candidates.len().min(b.candidates.len());
    for index in 0..common {
        let serial = a.candidates[index];
        let capture = b.candidates[index];
        if serial.token != capture.token || serial.weight.to_bits() != capture.weight.to_bits() {
            return Some(json!({
                "field": "candidate",
                "index": index,
                "serial": {"token": serial.token, "weight_f64_bits": u64_hex(serial.weight.to_bits())},
                "capture": {"token": capture.token, "weight_f64_bits": u64_hex(capture.weight.to_bits())},
            }));
        }
    }
    (a.candidates.len() != b.candidates.len()).then(|| {
        json!({
            "field": "candidate_count",
            "serial": a.candidates.len(),
            "capture": b.candidates.len(),
        })
    })
}

fn read_tensor_range(
    tensor: &MetalTensor,
    local_byte_offset: usize,
    bytes: usize,
) -> Result<Vec<u8>> {
    let tensor_bytes =
        usize::try_from(tensor.n_bytes()).context("tensor byte length exceeds usize")?;
    let local_end = local_byte_offset
        .checked_add(bytes)
        .context("tensor local byte range overflows")?;
    ensure!(
        local_end <= tensor_bytes,
        "tensor local byte range {local_byte_offset}..{local_end} exceeds logical length {tensor_bytes}"
    );
    let base = usize::try_from(tensor.offset).context("tensor offset exceeds usize")?;
    let start = base
        .checked_add(local_byte_offset)
        .context("tensor absolute byte offset overflows")?;
    let end = start
        .checked_add(bytes)
        .context("tensor absolute byte range overflows")?;
    ensure!(
        end <= tensor.buffer.length(),
        "tensor absolute byte range {start}..{end} exceeds buffer length {}",
        tensor.buffer.length()
    );
    let mut out = vec![0u8; bytes];
    unsafe {
        let source = (tensor.buffer.contents().as_ptr() as *const u8).add(start);
        std::ptr::copy_nonoverlapping(source, out.as_mut_ptr(), bytes);
    }
    Ok(out)
}

fn poison_f32_tensor(tensor: &MetalTensor, poison_bits: u32) -> Result<()> {
    ensure!(
        tensor.is_writable(),
        "hidden capture tensor is not writable"
    );
    ensure!(
        tensor.dtype == qwen_llm::tensor::GgmlType::F32,
        "hidden capture tensor is not F32"
    );
    let count =
        usize::try_from(tensor.n_elements()).context("hidden element count exceeds usize")?;
    let base = usize::try_from(tensor.offset).context("hidden tensor offset exceeds usize")?;
    let bytes = count
        .checked_mul(4)
        .context("hidden poison byte count overflows")?;
    let end = base
        .checked_add(bytes)
        .context("hidden poison range overflows")?;
    ensure!(
        end <= tensor.buffer.length(),
        "hidden poison range exceeds backing buffer"
    );
    unsafe {
        let destination = (tensor.buffer.contents().as_ptr() as *mut u8).add(base);
        for index in 0..count {
            std::ptr::copy_nonoverlapping(
                poison_bits.to_le_bytes().as_ptr(),
                destination.add(index * 4),
                4,
            );
        }
    }
    Ok(())
}

fn snapshot_first_mismatch(a: &SessionSnapshot, b: &SessionSnapshot) -> Option<Value> {
    if a.identity != b.identity {
        return Some(json!({"section": "identity"}));
    }
    if a.prefix_tokens != b.prefix_tokens {
        return Some(json!({"section": "prefix_tokens"}));
    }
    if a.pending_token != b.pending_token {
        return Some(json!({
            "section": "pending_token",
            "serial": a.pending_token,
            "capture": b.pending_token,
        }));
    }
    if a.kv_n_pos != b.kv_n_pos {
        let common = a.kv_n_pos.len().min(b.kv_n_pos.len());
        let index = (0..common)
            .find(|&index| a.kv_n_pos[index] != b.kv_n_pos[index])
            .unwrap_or(common);
        return Some(json!({
            "section": "kv_positions",
            "index": index,
            "serial": a.kv_n_pos.get(index),
            "capture": b.kv_n_pos.get(index),
        }));
    }
    for (section, serial, capture) in [
        ("kv_k", &a.kv_k_arena, &b.kv_k_arena),
        ("kv_v", &a.kv_v_arena, &b.kv_v_arena),
        ("gdn_conv", &a.gdn_conv_arena, &b.gdn_conv_arena),
        ("gdn_state", &a.gdn_state_arena, &b.gdn_state_arena),
    ] {
        if let Some(mut mismatch) = first_byte_mismatch(serial, capture) {
            mismatch
                .as_object_mut()
                .expect("mismatch object")
                .insert("section".into(), Value::String(section.into()));
            return Some(mismatch);
        }
    }
    None
}

fn snapshot_evidence_json(snapshot: &SessionSnapshot, sidecar: &mut StateSidecar) -> Result<Value> {
    let mut value = snapshot_json(snapshot);
    let kv_k = sidecar.append(&snapshot.kv_k_arena)?;
    let kv_v = sidecar.append(&snapshot.kv_v_arena)?;
    let gdn_conv = sidecar.append(&snapshot.gdn_conv_arena)?;
    let gdn_state = sidecar.append(&snapshot.gdn_state_arena)?;
    value.as_object_mut().expect("snapshot JSON object").insert(
        "section_sidecars".into(),
        json!({
            "kv_k": kv_k,
            "kv_v": kv_v,
            "gdn_conv": gdn_conv,
            "gdn_state": gdn_state,
        }),
    );
    Ok(value)
}

struct TransitionObservation {
    serial_logits: Vec<f32>,
    capture_logits: Vec<f32>,
    payload: Value,
    all_equal: bool,
}

#[allow(clippy::too_many_arguments)]
fn transition(
    mf: &MetalForward<'_>,
    ctx: &MetalContext,
    serial_session: &mut MetalSession,
    capture_session: &mut MetalSession,
    dflash_session: &mut MetalDFlashSession,
    sidecar: &mut StateSidecar,
    expected_hidden_context: &mut Vec<u8>,
    expected_context_positions: &mut Vec<i32>,
    hidden: &MetalTensor,
    target_layer_ids: &[u32],
    feature_count: usize,
    expected_vocab: usize,
    serial_identity: &SnapshotIdentity,
    capture_identity: &SnapshotIdentity,
    token: i32,
    position: usize,
    consumed_prefix: &[i32],
    phase: &str,
    step_index: usize,
    order: ArmOrder,
) -> Result<TransitionObservation> {
    let position_u32 = position_u32(position, phase)?;
    let position_i32 = i32::try_from(position).context("E0 position exceeds i32")?;
    let poison_bits = 0x7fa5_a5a5;
    poison_f32_tensor(hidden, poison_bits)?;
    let (serial_logits, capture_logits) = match order {
        ArmOrder::SerialThenCapture => {
            let serial = mf
                .single_token(token, position_u32, serial_session)
                .with_context(|| format!("{phase} serial transition at position {position}"))?;
            let capture = mf
                .single_token_with_multi_hidden(
                    token,
                    position_u32,
                    capture_session,
                    target_layer_ids,
                    hidden,
                )
                .with_context(|| format!("{phase} capture transition at position {position}"))?;
            (serial, capture)
        }
        ArmOrder::CaptureThenSerial => {
            let capture = mf
                .single_token_with_multi_hidden(
                    token,
                    position_u32,
                    capture_session,
                    target_layer_ids,
                    hidden,
                )
                .with_context(|| format!("{phase} capture transition at position {position}"))?;
            let serial = mf
                .single_token(token, position_u32, serial_session)
                .with_context(|| format!("{phase} serial transition at position {position}"))?;
            (serial, capture)
        }
    };

    let hidden_bytes_len = feature_count
        .checked_mul(4)
        .context("hidden capture byte count overflows")?;
    let hidden_source = read_tensor_range(hidden, 0, hidden_bytes_len)?;
    let hidden_source_bits = hidden_source
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four hidden bytes")))
        .collect::<Vec<_>>();
    let first_retained_poison = hidden_source_bits
        .iter()
        .position(|&bits| bits == poison_bits);
    let first_nonfinite_hidden = hidden_source_bits
        .iter()
        .position(|&bits| !f32::from_bits(bits).is_finite());
    let hidden_overwrite_complete = first_retained_poison.is_none();
    let hidden_finite = first_nonfinite_hidden.is_none();
    let context_before = dflash_session.target_ctx_n;
    dflash_session
        .append_target_ctx_column_now(ctx, hidden, position_u32, feature_count)
        .with_context(|| format!("append E0 hidden at position {position}"))?;
    let expected_context_n = context_before
        .checked_add(1)
        .context("expected DFlash context length overflows")?;
    expected_hidden_context.extend_from_slice(&hidden_source);
    expected_context_positions.push(position_i32);
    let active_context_bytes = dflash_session
        .target_ctx_n
        .checked_mul(hidden_bytes_len)
        .context("active DFlash context byte count overflows")?;
    let active_hidden_context =
        read_tensor_range(&dflash_session.target_ctx_stacked, 0, active_context_bytes)?;
    let row_start = context_before
        .checked_mul(hidden_bytes_len)
        .context("DFlash context row offset overflows")?;
    let row_end = row_start
        .checked_add(hidden_bytes_len)
        .context("DFlash context row end overflows")?;
    let hidden_destination = active_hidden_context
        .get(row_start..row_end)
        .map(|row| row.to_vec())
        .unwrap_or_default();
    let active_position_bytes = read_tensor_range(
        &dflash_session.pos_ctx,
        0,
        dflash_session
            .target_ctx_n
            .checked_mul(4)
            .context("active DFlash position byte count overflows")?,
    )?;
    let active_positions = active_position_bytes
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four position bytes")))
        .collect::<Vec<_>>();
    let captured_position = active_positions.get(context_before).copied();

    let serial_snapshot = serial_session
        .snapshot(serial_identity.clone(), consumed_prefix.to_vec(), None)
        .with_context(|| format!("snapshot serial E0 state at position {position}"))?;
    let capture_snapshot = capture_session
        .snapshot(capture_identity.clone(), consumed_prefix.to_vec(), None)
        .with_context(|| format!("snapshot capture E0 state at position {position}"))?;
    let state = SnapshotComparisons::compare(&serial_snapshot, &capture_snapshot);
    let logits_equal = serial_logits.len() == expected_vocab
        && capture_logits.len() == expected_vocab
        && logits_bit_equal(&serial_logits, &capture_logits);
    let hidden_transfer_equal = hidden_source == hidden_destination;
    let context_history_equal = active_hidden_context == *expected_hidden_context;
    let position_history_equal = active_positions == *expected_context_positions;
    let context_length_equal = dflash_session.target_ctx_n == expected_context_n
        && dflash_session.target_ctx_n == consumed_prefix.len();
    let position_equal = captured_position == Some(position_i32);
    let watermarks_valid = dflash_session.kv_ctx_ready_n <= dflash_session.ctx_h_ready_n
        && dflash_session.ctx_h_ready_n <= dflash_session.target_ctx_n
        && dflash_session.kv_ctx_ready_n == 0
        && dflash_session.ctx_h_ready_n == 0;
    let all_equal = logits_equal
        && state.all()
        && hidden_overwrite_complete
        && hidden_finite
        && hidden_transfer_equal
        && context_history_equal
        && position_history_equal
        && context_length_equal
        && position_equal
        && watermarks_valid;
    let serial_logits_bytes = f32le_bytes(&serial_logits);
    let capture_logits_bytes = f32le_bytes(&capture_logits);
    let serial_state_json = snapshot_evidence_json(&serial_snapshot, sidecar)?;
    let capture_state_json = snapshot_evidence_json(&capture_snapshot, sidecar)?;
    let payload = json!({
        "phase": phase,
        "step_index": step_index,
        "token": token,
        "position": position,
        "consumed_prefix_len": consumed_prefix.len(),
        "consumed_prefix_sha256_i32le": token_ids_sha256_i32le(consumed_prefix),
        "arm_order": order.as_str(),
        "serial": {
            "api": "single_token",
            "logits_count": serial_logits.len(),
            "logits_sha256_f32le": bytes_sha256(&serial_logits_bytes),
            "logits_f32le_hex": bytes_hex(&serial_logits_bytes),
            "state": serial_state_json,
        },
        "capture": {
            "api": "single_token_with_multi_hidden",
            "logits_count": capture_logits.len(),
            "logits_sha256_f32le": bytes_sha256(&capture_logits_bytes),
            "logits_f32le_hex": bytes_hex(&capture_logits_bytes),
            "state": capture_state_json,
        },
        "hidden_transfer": {
            "semantics": HIDDEN_TRANSFER_SEMANTICS,
            "target_layer_ids": target_layer_ids,
            "shape": [target_layer_ids.len(), feature_count / target_layer_ids.len()],
            "source_bytes": hidden_source.len(),
            "source_sha256_f32le": bytes_sha256(&hidden_source),
            "source_f32le_hex": bytes_hex(&hidden_source),
            "poison_f32_bits": format!("0x{poison_bits:08x}"),
            "first_retained_poison_index": first_retained_poison,
            "first_nonfinite_index": first_nonfinite_hidden,
            "destination_bytes": hidden_destination.len(),
            "destination_sha256_f32le": bytes_sha256(&hidden_destination),
            "destination_f32le_hex": bytes_hex(&hidden_destination),
            "active_context_bytes": active_hidden_context.len(),
            "active_context_sha256_f32le": bytes_sha256(&active_hidden_context),
            "active_context_f32le_hex": bytes_hex(&active_hidden_context),
            "expected_active_context_sha256_f32le": bytes_sha256(expected_hidden_context),
            "active_positions": active_positions,
            "expected_active_positions": expected_context_positions,
            "target_ctx_index": context_before,
            "target_ctx_n_after": dflash_session.target_ctx_n,
            "captured_position": captured_position,
            "ctx_h_ready_n": dflash_session.ctx_h_ready_n,
            "kv_ctx_ready_n": dflash_session.kv_ctx_ready_n,
        },
        "comparisons": {
            "logits_bits": logits_equal,
            "state": state.json(),
            "hidden_overwrite_complete": hidden_overwrite_complete,
            "hidden_values_finite": hidden_finite,
            "hidden_transfer_bytes": hidden_transfer_equal,
            "active_context_history": context_history_equal,
            "active_position_history": position_history_equal,
            "target_ctx_length": context_length_equal,
            "target_ctx_position": position_equal,
            "target_ctx_watermarks": watermarks_valid,
            "all": all_equal,
        },
        "first_mismatch": {
            "logits": first_logit_mismatch(&serial_logits, &capture_logits),
            "state": snapshot_first_mismatch(&serial_snapshot, &capture_snapshot),
            "hidden_transfer": first_byte_mismatch(&hidden_source, &hidden_destination),
            "active_context": first_byte_mismatch(expected_hidden_context, &active_hidden_context),
        },
    });
    Ok(TransitionObservation {
        serial_logits,
        capture_logits,
        payload,
        all_equal,
    })
}

struct SampleObservation {
    committed: Option<SampledToken>,
    payload: Value,
    all_equal: bool,
}

#[allow(clippy::too_many_arguments)]
fn sample_frontier(
    serial_sampler: &mut Sampler,
    capture_sampler: &mut Sampler,
    serial_logits: &[f32],
    capture_logits: &[f32],
    sample_index: usize,
    target_position: usize,
    consumed_prefix: &[i32],
    token_limit: usize,
    stop_tokens: &[i32],
) -> Result<SampleObservation> {
    let serial_draws_before = serial_sampler.draws();
    let capture_draws_before = capture_sampler.draws();
    let (serial_distribution, serial_rng) = serial_sampler
        .diagnose_next(serial_logits)
        .context("diagnose serial E0 sampler")?;
    let (capture_distribution, capture_rng) = capture_sampler
        .diagnose_next(capture_logits)
        .context("diagnose capture E0 sampler")?;
    let serial_live = serial_sampler
        .sample(serial_logits)
        .context("draw serial E0 sampler")?;
    let capture_live = capture_sampler
        .sample(capture_logits)
        .context("draw capture E0 sampler")?;
    let distribution_equal = distribution_bit_equal(&serial_distribution, &capture_distribution);
    let rng_equal = rng_bit_equal(serial_rng, capture_rng);
    let diagnosed_live_equal =
        serial_distribution.sampled == serial_live && capture_distribution.sampled == capture_live;
    let live_equal = serial_live == capture_live;
    let draw_counts_equal = serial_draws_before == capture_draws_before
        && serial_sampler.draws() == capture_sampler.draws()
        && serial_sampler.draws() == serial_draws_before + 1;
    let all_equal =
        distribution_equal && rng_equal && diagnosed_live_equal && live_equal && draw_counts_equal;
    let committed = all_equal.then_some(serial_live);
    let emitted_count = sample_index + 1;
    let stop = committed.and_then(|sample| {
        (emitted_count >= token_limit || stop_tokens.contains(&sample.token))
            .then(|| stop_outcome(sample.token, emitted_count, token_limit, stop_tokens))
    });
    Ok(SampleObservation {
        committed,
        all_equal,
        payload: json!({
            "sample_index": sample_index,
            "target_position": target_position,
            "consumed_prefix_len": consumed_prefix.len(),
            "consumed_prefix_sha256_i32le": token_ids_sha256_i32le(consumed_prefix),
            "serial_logits_sha256_f32le": logits_sha256(serial_logits),
            "capture_logits_sha256_f32le": logits_sha256(capture_logits),
            "serial": {
                "distribution": distribution_json(&serial_distribution),
                "rng": rng_json(serial_rng),
                "live_sample": {
                    "token": serial_live.token,
                    "candidate_index": serial_live.candidate_index,
                    "draws_after": serial_sampler.draws(),
                },
            },
            "capture": {
                "distribution": distribution_json(&capture_distribution),
                "rng": rng_json(capture_rng),
                "live_sample": {
                    "token": capture_live.token,
                    "candidate_index": capture_live.candidate_index,
                    "draws_after": capture_sampler.draws(),
                },
            },
            "committed_token": committed.map(|sample| sample.token),
            "terminal": stop.is_some(),
            "stop_reason": stop.map(|outcome| outcome.reason),
            "eos_hit": stop.is_some_and(|outcome| outcome.eos_hit),
            "token_limit_hit": stop.is_some_and(|outcome| outcome.token_limit_hit),
            "comparisons": {
                "distribution_bits": distribution_equal,
                "rng_transition": rng_equal,
                "diagnosed_vs_live": diagnosed_live_equal,
                "live_sample": live_equal,
                "draw_counts": draw_counts_equal,
                "all": all_equal,
            },
            "first_distribution_mismatch": first_distribution_mismatch(
                &serial_distribution,
                &capture_distribution,
            ),
        }),
    })
}

fn continuation_distribution(
    serial_sampler: &Sampler,
    capture_sampler: &Sampler,
    serial_logits: &[f32],
    capture_logits: &[f32],
) -> Result<(Value, bool)> {
    let serial_draws = serial_sampler.draws();
    let capture_draws = capture_sampler.draws();
    let (serial_distribution, serial_rng) = serial_sampler
        .diagnose_next(serial_logits)
        .context("diagnose serial continuation sampler")?;
    let (capture_distribution, capture_rng) = capture_sampler
        .diagnose_next(capture_logits)
        .context("diagnose capture continuation sampler")?;
    let distribution_equal = distribution_bit_equal(&serial_distribution, &capture_distribution);
    let rng_equal = rng_bit_equal(serial_rng, capture_rng);
    let nonadvancing = serial_sampler.draws() == serial_draws
        && capture_sampler.draws() == capture_draws
        && serial_draws == capture_draws;
    let all = distribution_equal && rng_equal && nonadvancing;
    Ok((
        json!({
            "serial": {
                "distribution": distribution_json(&serial_distribution),
                "rng": rng_json(serial_rng),
                "live_draws_unchanged": serial_sampler.draws(),
            },
            "capture": {
                "distribution": distribution_json(&capture_distribution),
                "rng": rng_json(capture_rng),
                "live_draws_unchanged": capture_sampler.draws(),
            },
            "comparisons": {
                "distribution_bits": distribution_equal,
                "rng_transition": rng_equal,
                "diagnostics_nonadvancing": nonadvancing,
                "all": all,
            },
            "first_distribution_mismatch": first_distribution_mismatch(
                &serial_distribution,
                &capture_distribution,
            ),
        }),
        all,
    ))
}

struct RunOutcome {
    payload: Value,
    passed: bool,
}

struct Execution<'a> {
    args: &'a DflashE0LockstepArgs,
    common: &'a Value,
    writer: &'a mut JsonlAppender,
    sidecar: &'a mut StateSidecar,
    binding_manifest: &'a Value,
    ctx: &'a MetalContext,
    target_g: &'a GgufFile,
    target_m: &'a Model<'a>,
    drafter_g: &'a GgufFile,
    head: &'a qwen_llm::loader::DFlashHead<'a>,
    prompt_ids: &'a [i32],
    stops: &'a [i32],
    sampling: SamplingConfig,
    capacity: usize,
    snapshot_model_id: u64,
    snapshot_tokenizer_id: u64,
}

impl Execution<'_> {
    fn mismatch(
        &self,
        phase: &str,
        prompt_steps: usize,
        transitions: usize,
        samples: usize,
        generated: &[i32],
        serial_draws: usize,
        capture_draws: usize,
        started: Instant,
    ) -> RunOutcome {
        RunOutcome {
            passed: false,
            payload: json!({
                "status": "mismatch",
                "e0_status": "development_failed",
                "authority": "development_only_no_product_authority",
                "failed_phase": phase,
                "prompt_steps": prompt_steps,
                "target_transitions": transitions,
                "sample_frontiers": samples,
                "generated_ids": generated,
                "generated_ids_sha256_i32le": token_ids_sha256_i32le(generated),
                "serial_sampler_draws": serial_draws,
                "capture_sampler_draws": capture_draws,
                "continuation_compared": false,
                "elapsed_seconds_f64_bits": f64_bits(started.elapsed().as_secs_f64()),
                "timing_semantics": "diagnostic_only_non_performance",
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn observed_failure(
        &mut self,
        phase: &str,
        step_index: usize,
        token: Option<i32>,
        position: Option<usize>,
        error: &anyhow::Error,
        prompt_steps: usize,
        transitions: usize,
        samples: usize,
        generated: &[i32],
        serial_draws: usize,
        capture_draws: usize,
        started: Instant,
    ) -> Result<RunOutcome> {
        self.writer.write(&event(
            self.common,
            "observed_failure",
            json!({
                "phase": phase,
                "step_index": step_index,
                "token": token,
                "position": position,
                "error": format!("{error:#}"),
                "observed_e0_failure": true,
                "replaceable_infrastructure_failure": false,
            }),
        ))?;
        Ok(self.mismatch(
            phase,
            prompt_steps,
            transitions,
            samples,
            generated,
            serial_draws,
            capture_draws,
            started,
        ))
    }

    fn run(mut self) -> Result<RunOutcome> {
        let started = Instant::now();
        let mm = MetalModel::load(self.ctx, self.target_g, self.target_m)
            .context("metal-load E0 target")?;
        let mhead = MetalDFlashHead::load(self.ctx, self.drafter_g, self.head)
            .context("metal-load E0 drafter")?;
        ensure!(
            mhead.selector.is_some(),
            "E0 requires a DFlash 2 drafter binding"
        );
        let mf = MetalForward::new(self.ctx, &mm);
        let h = usize::try_from(self.target_m.arch.hidden_size)
            .context("target hidden size exceeds usize")?;
        let vocab = usize::try_from(self.target_m.arch.vocab_size)
            .context("target vocabulary exceeds usize")?;
        let feature_count = self
            .head
            .target_layer_ids
            .len()
            .checked_mul(h)
            .context("E0 capture feature count overflows")?;
        ensure!(
            feature_count > 0,
            "E0 requires at least one target capture layer"
        );
        if !self.args.no_warmup {
            let warm_hidden = MetalTensor::zeros_f32(self.ctx, vec![feature_count as u64])?;
            let mut warm_serial = MetalSession::fresh(self.ctx, &mm, self.capacity)?;
            let mut warm_capture = MetalSession::fresh(self.ctx, &mm, self.capacity)?;
            let _ = mf.single_token(self.prompt_ids[0], 0, &mut warm_serial)?;
            let _ = mf.single_token_with_multi_hidden(
                self.prompt_ids[0],
                0,
                &mut warm_capture,
                &self.head.target_layer_ids,
                &warm_hidden,
            )?;
        }

        let mut serial_session = MetalSession::fresh(self.ctx, &mm, self.capacity)?;
        let mut capture_session = MetalSession::fresh(self.ctx, &mm, self.capacity)?;
        let hidden = MetalTensor::zeros_f32(self.ctx, vec![feature_count as u64])?;
        let mut dflash_session =
            MetalDFlashSession::fresh(self.ctx, &mhead, h as u64, vocab as u64, self.capacity)?;
        let serial_identity =
            serial_session.snapshot_identity(self.snapshot_model_id, self.snapshot_tokenizer_id);
        let capture_identity =
            capture_session.snapshot_identity(self.snapshot_model_id, self.snapshot_tokenizer_id);
        ensure!(
            serial_identity == capture_identity,
            "fresh E0 target sessions have different snapshot identities"
        );
        validate_binding_manifest_snapshot(self.binding_manifest, &serial_identity)?;
        let mut serial_sampler = Sampler::new(self.sampling)?;
        let mut capture_sampler = Sampler::new(self.sampling)?;
        let mut prefix = Vec::with_capacity(self.prompt_ids.len() + self.args.tokens + 1);
        let mut serial_logits = Vec::new();
        let mut capture_logits = Vec::new();
        let mut expected_hidden_context = Vec::new();
        let mut expected_context_positions = Vec::new();
        let mut prompt_steps = 0usize;
        let mut transitions = 0usize;
        let mut sample_count = 0usize;

        for (position, &token) in self.prompt_ids.iter().enumerate() {
            prefix.push(token);
            let observed = match transition(
                &mf,
                self.ctx,
                &mut serial_session,
                &mut capture_session,
                &mut dflash_session,
                self.sidecar,
                &mut expected_hidden_context,
                &mut expected_context_positions,
                &hidden,
                &self.head.target_layer_ids,
                feature_count,
                vocab,
                &serial_identity,
                &capture_identity,
                token,
                position,
                &prefix,
                "prompt",
                position,
                self.args.arm_order,
            ) {
                Ok(observed) => observed,
                Err(error) => {
                    return self.observed_failure(
                        "prompt_step",
                        position,
                        Some(token),
                        Some(position),
                        &error,
                        prompt_steps,
                        transitions,
                        sample_count,
                        &[],
                        serial_sampler.draws(),
                        capture_sampler.draws(),
                        started,
                    );
                }
            };
            self.writer
                .write(&event(self.common, "prompt_step", observed.payload))?;
            prompt_steps += 1;
            serial_logits = observed.serial_logits;
            capture_logits = observed.capture_logits;
            if !observed.all_equal {
                return Ok(self.mismatch(
                    "prompt_step",
                    prompt_steps,
                    transitions,
                    sample_count,
                    &[],
                    serial_sampler.draws(),
                    capture_sampler.draws(),
                    started,
                ));
            }
        }

        let mut generated = Vec::with_capacity(self.args.tokens);
        let pending = loop {
            let sampled = match sample_frontier(
                &mut serial_sampler,
                &mut capture_sampler,
                &serial_logits,
                &capture_logits,
                sample_count,
                prefix.len() - 1,
                &prefix,
                self.args.tokens,
                self.stops,
            ) {
                Ok(sampled) => sampled,
                Err(error) => {
                    return self.observed_failure(
                        "sample_frontier",
                        sample_count,
                        None,
                        Some(prefix.len() - 1),
                        &error,
                        prompt_steps,
                        transitions,
                        sample_count,
                        &generated,
                        serial_sampler.draws(),
                        capture_sampler.draws(),
                        started,
                    );
                }
            };
            self.writer
                .write(&event(self.common, "sample_frontier", sampled.payload))?;
            sample_count += 1;
            let Some(committed) = sampled.committed else {
                return Ok(self.mismatch(
                    "sample_frontier",
                    prompt_steps,
                    transitions,
                    sample_count,
                    &generated,
                    serial_sampler.draws(),
                    capture_sampler.draws(),
                    started,
                ));
            };
            debug_assert!(sampled.all_equal);
            generated.push(committed.token);
            if generated.len() >= self.args.tokens || self.stops.contains(&committed.token) {
                break committed.token;
            }

            let position = prefix.len();
            prefix.push(committed.token);
            let observed = match transition(
                &mf,
                self.ctx,
                &mut serial_session,
                &mut capture_session,
                &mut dflash_session,
                self.sidecar,
                &mut expected_hidden_context,
                &mut expected_context_positions,
                &hidden,
                &self.head.target_layer_ids,
                feature_count,
                vocab,
                &serial_identity,
                &capture_identity,
                committed.token,
                position,
                &prefix,
                "generated",
                transitions,
                self.args.arm_order,
            ) {
                Ok(observed) => observed,
                Err(error) => {
                    return self.observed_failure(
                        "target_transition",
                        transitions,
                        Some(committed.token),
                        Some(position),
                        &error,
                        prompt_steps,
                        transitions,
                        sample_count,
                        &generated,
                        serial_sampler.draws(),
                        capture_sampler.draws(),
                        started,
                    );
                }
            };
            self.writer
                .write(&event(self.common, "target_transition", observed.payload))?;
            transitions += 1;
            serial_logits = observed.serial_logits;
            capture_logits = observed.capture_logits;
            if !observed.all_equal {
                return Ok(self.mismatch(
                    "target_transition",
                    prompt_steps,
                    transitions,
                    sample_count,
                    &generated,
                    serial_sampler.draws(),
                    capture_sampler.draws(),
                    started,
                ));
            }
        };

        if serial_sampler.draws() != generated.len() || capture_sampler.draws() != generated.len() {
            let error = anyhow!("E0 sampler draw count diverged from emitted count");
            return self.observed_failure(
                "sampler_accounting",
                sample_count,
                None,
                Some(prefix.len() - 1),
                &error,
                prompt_steps,
                transitions,
                sample_count,
                &generated,
                serial_sampler.draws(),
                capture_sampler.draws(),
                started,
            );
        }
        let stop = stop_outcome(pending, generated.len(), self.args.tokens, self.stops);
        let boundary_snapshots = (|| -> Result<_> {
            let serial = serial_session.snapshot(serial_identity.clone(), prefix.clone(), None)?;
            let capture =
                capture_session.snapshot(capture_identity.clone(), prefix.clone(), None)?;
            let serial_json = snapshot_evidence_json(&serial, self.sidecar)?;
            let capture_json = snapshot_evidence_json(&capture, self.sidecar)?;
            Ok((serial, capture, serial_json, capture_json))
        })();
        let (
            mut serial_boundary,
            mut capture_boundary,
            mut serial_boundary_json,
            mut capture_boundary_json,
        ) = match boundary_snapshots {
            Ok(snapshots) => snapshots,
            Err(error) => {
                return self.observed_failure(
                    "terminal_boundary",
                    0,
                    Some(pending),
                    Some(prefix.len()),
                    &error,
                    prompt_steps,
                    transitions,
                    sample_count,
                    &generated,
                    serial_sampler.draws(),
                    capture_sampler.draws(),
                    started,
                );
            }
        };
        serial_boundary.pending_token = Some(pending);
        capture_boundary.pending_token = Some(pending);
        for value in [&mut serial_boundary_json, &mut capture_boundary_json] {
            let object = value.as_object_mut().expect("boundary snapshot object");
            object.insert("pending_token".into(), json!(pending));
            object.insert(
                "pending_token_sha256_tagged_i32le".into(),
                json!(super::dflash_sampled_oracle::pending_token_sha256(Some(
                    pending,
                ))),
            );
        }
        let boundary_state = SnapshotComparisons::compare(&serial_boundary, &capture_boundary);
        let boundary_context_equal = dflash_session.target_ctx_n == prefix.len();
        let boundary_all = boundary_state.all() && boundary_context_equal;
        self.writer.write(&event(
            self.common,
            "terminal_boundary",
            json!({
                "generated_ids": generated,
                "generated_ids_sha256_i32le": token_ids_sha256_i32le(&generated),
                "stop_reason": stop.reason,
                "terminal_token": stop.terminal_token,
                "eos_hit": stop.eos_hit,
                "token_limit_hit": stop.token_limit_hit,
                "serial_sampler_draws": serial_sampler.draws(),
                "capture_sampler_draws": capture_sampler.draws(),
                "consumed_prefix_len": prefix.len(),
                "pending_token": pending,
                "serial_state": serial_boundary_json,
                "capture_state": capture_boundary_json,
                "dflash_target_ctx_n": dflash_session.target_ctx_n,
                "comparisons": {
                    "state": boundary_state.json(),
                    "target_ctx_length": boundary_context_equal,
                    "all": boundary_all,
                },
                "first_state_mismatch": snapshot_first_mismatch(
                    &serial_boundary,
                    &capture_boundary,
                ),
            }),
        ))?;
        if !boundary_all {
            return Ok(self.mismatch(
                "terminal_boundary",
                prompt_steps,
                transitions,
                sample_count,
                &generated,
                serial_sampler.draws(),
                capture_sampler.draws(),
                started,
            ));
        }

        let continuation_position = prefix.len();
        prefix.push(pending);
        let continuation = match transition(
            &mf,
            self.ctx,
            &mut serial_session,
            &mut capture_session,
            &mut dflash_session,
            self.sidecar,
            &mut expected_hidden_context,
            &mut expected_context_positions,
            &hidden,
            &self.head.target_layer_ids,
            feature_count,
            vocab,
            &serial_identity,
            &capture_identity,
            pending,
            continuation_position,
            &prefix,
            "continuation",
            0,
            self.args.arm_order,
        ) {
            Ok(continuation) => continuation,
            Err(error) => {
                return self.observed_failure(
                    "continuation",
                    0,
                    Some(pending),
                    Some(continuation_position),
                    &error,
                    prompt_steps,
                    transitions,
                    sample_count,
                    &generated,
                    serial_sampler.draws(),
                    capture_sampler.draws(),
                    started,
                );
            }
        };
        let (continuation_distribution, continuation_distribution_equal) =
            match continuation_distribution(
                &serial_sampler,
                &capture_sampler,
                &continuation.serial_logits,
                &continuation.capture_logits,
            ) {
                Ok(distribution) => distribution,
                Err(error) => {
                    return self.observed_failure(
                        "continuation_frontier",
                        0,
                        None,
                        Some(continuation_position),
                        &error,
                        prompt_steps,
                        transitions,
                        sample_count,
                        &generated,
                        serial_sampler.draws(),
                        capture_sampler.draws(),
                        started,
                    );
                }
            };
        let continuation_all = continuation.all_equal && continuation_distribution_equal;
        self.writer.write(&event(
            self.common,
            "continuation",
            json!({
                "transition": continuation.payload,
                "next_frontier": continuation_distribution,
                "comparisons": {
                    "transition": continuation.all_equal,
                    "next_frontier": continuation_distribution_equal,
                    "all": continuation_all,
                },
            }),
        ))?;
        if !continuation_all {
            return Ok(self.mismatch(
                "continuation",
                prompt_steps,
                transitions,
                sample_count,
                &generated,
                serial_sampler.draws(),
                capture_sampler.draws(),
                started,
            ));
        }

        Ok(RunOutcome {
            passed: true,
            payload: json!({
                "status": "ok",
                "e0_status": "development_lockstep_passed",
                "authority": "development_only_no_product_authority",
                "packed_verifier_status": "not_measured",
                "generated_ids": generated,
                "generated_ids_sha256_i32le": token_ids_sha256_i32le(&generated),
                "stop_reason": stop.reason,
                "terminal_token": stop.terminal_token,
                "eos_hit": stop.eos_hit,
                "token_limit_hit": stop.token_limit_hit,
                "prompt_steps": prompt_steps,
                "target_transitions": transitions,
                "sample_frontiers": sample_count,
                "serial_sampler_draws": serial_sampler.draws(),
                "capture_sampler_draws": capture_sampler.draws(),
                "continuation_compared": true,
                "continuation_equal": true,
                "final_consumed_prefix_len": prefix.len(),
                "final_dflash_target_ctx_n": dflash_session.target_ctx_n,
                "all_comparisons": true,
                "elapsed_seconds_f64_bits": f64_bits(started.elapsed().as_secs_f64()),
                "timing_semantics": "diagnostic_only_non_performance",
            }),
        })
    }
}

pub fn run(
    args: DflashE0LockstepArgs,
    build_identity: Value,
    lease_env: BTreeMap<String, String>,
) -> Result<()> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let run_id = format!("dflash-e0-{}-{}", now.as_nanos(), std::process::id());
    let command = std::env::args_os()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let model_path_identity = stable_path_identity(&args.model);
    let drafter_path_identity = stable_path_identity(&args.drafter);
    let manifest_path_identity = stable_path_identity(&args.binding_manifest);
    let output_path_identity = stable_path_identity(&args.output);
    let sidecar_path_identity = stable_path_identity(&args.state_sidecar);
    let executable = std::env::current_exe().context("resolve E0 executable")?;
    let executable_path_identity = stable_path_identity(&executable);
    validate_distinct_artifact_paths(&[
        ("model", &model_path_identity),
        ("drafter", &drafter_path_identity),
        ("binding_manifest", &manifest_path_identity),
        ("output", &output_path_identity),
        ("state_sidecar", &sidecar_path_identity),
        ("executable", &executable_path_identity),
    ])?;
    let evidence_paths = evidence_paths_json(
        &model_path_identity,
        &drafter_path_identity,
        &manifest_path_identity,
        &output_path_identity,
        &sidecar_path_identity,
        &executable_path_identity,
    );
    let bootstrap_common = json!({
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "build_identity": build_identity,
        "lease_env": lease_env,
        "command": command,
        "paths": evidence_paths.clone(),
    });
    let mut writer = JsonlAppender::create_exclusive(&args.output)?;
    writer.write(&event(
        &bootstrap_common,
        "bootstrap_start",
        json!({"started_utc": super::utc_iso8601_now()}),
    ))?;
    writer
        .finish()
        .context("sync E0 bootstrap_start evidence")?;

    macro_rules! bootstrap_fail {
        ($error:expr) => {{
            let error: anyhow::Error = $error;
            writer.write(&event(
                &bootstrap_common,
                "bootstrap_end",
                json!({
                    "status": "infrastructure_error",
                    "observed": false,
                    "replaceable": false,
                    "error": format!("{error:#}"),
                }),
            ))?;
            writer.finish()?;
            return Err(error);
        }};
    }
    macro_rules! bootstrap_try {
        ($expression:expr) => {{
            match $expression {
                Ok(value) => value,
                Err(error) => bootstrap_fail!(anyhow::Error::from(error)),
            }
        }};
    }
    macro_rules! bootstrap_ensure {
        ($condition:expr, $message:literal) => {{
            if !$condition {
                bootstrap_fail!(anyhow!($message));
            }
        }};
    }

    bootstrap_ensure!(!args.prompt.is_empty(), "--prompt must be nonempty");
    bootstrap_ensure!(args.tokens > 0, "--tokens must be positive");
    bootstrap_ensure!(
        (1..=200).contains(&args.top_k),
        "--top-k must be in 1..=200 for bounded E0 traces"
    );
    bootstrap_ensure!(
        args.temperature.is_finite() && args.temperature > 0.0,
        "--temperature must be finite and positive"
    );
    bootstrap_ensure!(
        std::env::var("QWEN_METAL_LEASE_WAIT").as_deref() == Ok("1"),
        "dflash-e0-lockstep requires QWEN_METAL_LEASE_WAIT=1 before Metal initialization"
    );
    bootstrap_try!(validate_development_label(&args.fixture_id, "--fixture-id"));
    bootstrap_try!(validate_development_label(
        &args.fixture_role,
        "--fixture-role"
    ));
    bootstrap_try!(validate_development_label(&args.target_arm, "--target-arm"));
    bootstrap_try!(validate_development_label(
        &args.drafter_arm,
        "--drafter-arm"
    ));
    bootstrap_try!(validate_e0_build_identity(&build_identity));
    let mut sidecar = bootstrap_try!(StateSidecar::create(args.state_sidecar.clone()));
    let (binding_manifest, binding_manifest_identity) =
        bootstrap_try!(load_binding_manifest(&args.binding_manifest));

    let target_g = bootstrap_try!(
        GgufFile::open(&args.model)
            .with_context(|| format!("open target {}", args.model.display()))
    );
    let stops = bootstrap_try!(super::resolve_stop_tokens(
        &target_g,
        args.stop_tokens.clone()
    ));
    let target_m =
        bootstrap_try!(Model::from_gguf(&target_g).context("parse E0 target architecture"));
    let drafter_g = bootstrap_try!(
        GgufFile::open(&args.drafter)
            .with_context(|| format!("open drafter {}", args.drafter.display()))
    );
    let head =
        bootstrap_try!(open_dflash_drafter(&drafter_g, &target_m).context("bind E0 drafter"));
    bootstrap_ensure!(
        !head.target_layer_ids.is_empty(),
        "E0 requires DFlash target capture layers"
    );
    let tokenizer = bootstrap_try!(Tokenizer::from_gguf(&target_g).context("open E0 tokenizer"));
    let prompt_ids = bootstrap_try!(
        tokenizer
            .encode(&args.prompt, false)
            .context("tokenize E0 prompt")
    );
    bootstrap_ensure!(
        !prompt_ids.is_empty(),
        "--prompt must tokenize to at least one token"
    );
    let sampling = bootstrap_try!(
        SamplingConfig {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            min_p: args.min_p,
            seed: args.seed,
        }
        .validate()
        .context("invalid E0 sampling configuration")
    );
    let capacity = bootstrap_try!(
        prompt_ids
            .len()
            .checked_add(args.tokens)
            .and_then(|value| value.checked_add(33))
            .context("E0 context capacity overflows")
    );
    bootstrap_ensure!(
        capacity <= i32::MAX as usize,
        "requested E0 capacity exceeds DFlash i32 position scope"
    );
    let prompt_digest = token_ids_sha256_i32le(&prompt_ids);
    let target_asset = gguf_asset_identity(&target_g);
    let drafter_asset = gguf_asset_identity(&drafter_g);
    bootstrap_try!(validate_binding_manifest_static(
        &binding_manifest,
        &target_asset.json,
        &drafter_asset.json,
        &target_g,
        &target_m,
        &head,
        &args,
    ));
    let snapshot_model_id = digest_identity_word(&target_asset.digest, 0);
    let snapshot_tokenizer_id = digest_identity_word(&target_asset.digest, 8);
    let executable_identity = bootstrap_try!(regular_file_identity(&executable));
    let ctx = bootstrap_try!(MetalContext::new().context("init E0 MetalContext"));
    let device = ctx.describe();
    eprintln!("[dflash-e0-lockstep] device: {device}");
    writer.write(&event(
        &bootstrap_common,
        "bootstrap_end",
        json!({"status": "ok", "observed": false}),
    ))?;
    writer.finish().context("sync E0 bootstrap_end evidence")?;
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
            "authority": "development_only_no_product_authority",
        },
        "config": {
            "tokens": args.tokens,
            "context_capacity": capacity,
            "stop_tokens": stops,
            "temperature_f32_bits": f32_bits(sampling.temperature),
            "top_k": sampling.top_k,
            "top_p_f32_bits": f32_bits(sampling.top_p),
            "min_p_f32_bits": f32_bits(sampling.min_p),
            "seed": sampling.seed,
            "sampler_algorithm_version": SAMPLER_ALGORITHM_VERSION,
            "no_warmup": args.no_warmup,
            "arm_order": args.arm_order.as_str(),
            "semantics": DEVELOPMENT_SEMANTICS,
            "hidden_transfer_semantics": HIDDEN_TRANSFER_SEMANTICS,
        },
        "assets": {
            "target": target_asset.json,
            "drafter": drafter_asset.json,
            "executable": executable_identity,
        },
        "binding_manifest": binding_manifest_identity,
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
        "paths": evidence_paths,
        "prompt": {
            "utf8_len": args.prompt.len(),
            "utf8_sha256": bytes_sha256(args.prompt.as_bytes()),
            "token_count": prompt_ids.len(),
            "token_ids_sha256_i32le": prompt_digest,
        },
        "sessions": {
            "serial_target": format!("{run_id}/serial-target"),
            "capture_target": format!("{run_id}/capture-target"),
            "capture_drafter": format!("{run_id}/capture-drafter"),
        },
    });
    writer.write(&event(
        &common,
        "run_start",
        json!({
            "started_utc": super::utc_iso8601_now(),
            "protocol": "E0-CLARIFICATION.md",
            "packed_verifier": "excluded_different_distribution",
        }),
    ))?;
    writer.finish().context("sync E0 run_start evidence")?;

    let prefetch_cfg = qwen_llm::runtime::LoadedModelConfig::default();
    let _ = qwen_llm::runtime::prefetch_opened_gguf(&target_g, &prefetch_cfg);
    let _ = qwen_llm::runtime::prefetch_opened_gguf(&drafter_g, &prefetch_cfg);
    let outcome = Execution {
        args: &args,
        common: &common,
        writer: &mut writer,
        sidecar: &mut sidecar,
        binding_manifest: &binding_manifest,
        ctx: &ctx,
        target_g: &target_g,
        target_m: &target_m,
        drafter_g: &drafter_g,
        head: &head,
        prompt_ids: &prompt_ids,
        stops: &stops,
        sampling,
        capacity,
        snapshot_model_id,
        snapshot_tokenizer_id,
    }
    .run();
    let sidecar_identity = sidecar.finish().context("finish E0 state sidecar")?;

    let mut outcome = match outcome {
        Ok(outcome) => outcome,
        Err(error) => {
            writer.write(&event(
                &common,
                "run_end",
                json!({
                    "status": "infrastructure_error",
                    "e0_status": "invalid_not_observed",
                    "authority": "development_only_no_product_authority",
                    "error": format!("{error:#}"),
                    "state_sidecar": sidecar_identity,
                }),
            ))?;
            writer.finish()?;
            return Err(error);
        }
    };
    outcome
        .payload
        .as_object_mut()
        .expect("run_end payload object")
        .insert("state_sidecar".into(), sidecar_identity);
    writer.write(&event(&common, "run_end", outcome.payload))?;
    writer.finish()?;
    if !outcome.passed {
        return Err(anyhow!("E0 lockstep development row failed exact parity"));
    }
    eprintln!("[dflash-e0-lockstep] development E0 lockstep passed; no product authority");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen_llm::sampling::{SamplingUniform, WeightedCandidate};

    fn distribution(weight: f64) -> SamplingDistribution {
        SamplingDistribution {
            candidates: vec![WeightedCandidate { token: 7, weight }],
            total_weight: weight,
            sampled: SampledToken {
                token: 7,
                candidate_index: 0,
            },
        }
    }

    #[test]
    fn bit_serialization_preserves_signed_zero_and_nan_payloads() {
        let values = [0.0, -0.0, f32::from_bits(0x7fc0_1234)];
        assert_eq!(f32le_hex(&values), "00000000000000803412c07f");
        assert!(logits_bit_equal(&values, &values));
        assert!(!logits_bit_equal(&[0.0], &[-0.0]));
        assert_eq!(
            first_logit_mismatch(&[0.0], &[-0.0]).expect("signed-zero mismatch")["index"],
            0
        );
    }

    #[test]
    fn distribution_comparison_is_bit_exact() {
        assert!(distribution_bit_equal(
            &distribution(0.0),
            &distribution(0.0)
        ));
        assert!(!distribution_bit_equal(
            &distribution(0.0),
            &distribution(-0.0)
        ));
        assert!(first_distribution_mismatch(&distribution(1.0), &distribution(2.0)).is_some());
    }

    #[test]
    fn stop_precedence_and_arm_labels_are_frozen() {
        let stop = stop_outcome(9, 4, 4, &[9]);
        assert_eq!(stop.reason, "eos");
        assert!(stop.eos_hit && stop.token_limit_hit);
        assert_eq!(ArmOrder::SerialThenCapture.as_str(), "serial_then_capture");
        assert_eq!(ArmOrder::CaptureThenSerial.as_str(), "capture_then_serial");
    }

    #[test]
    fn sampling_uniform_boundary_assumption_is_half_open() {
        assert!(SamplingUniform::new(0.0).is_ok());
        assert!(SamplingUniform::new(1.0).is_err());
    }

    #[test]
    fn acquisition_build_identity_must_be_clean_and_unoverridden() {
        let source = format!("git-source-sha256-v2:{}", "a".repeat(64));
        let mut identity = json!({
            "build_commit": "1".repeat(40),
            "runtime_commit": "1".repeat(40),
            "build_dirty": false,
            "runtime_dirty": false,
            "build_source_state": source,
            "runtime_source_state": source,
            "stamp_error": null,
            "status": "match",
            "problems": [],
            "overrides": [],
        });
        assert!(validate_e0_build_identity(&identity).is_ok());
        identity["build_dirty"] = json!(true);
        identity["runtime_dirty"] = json!(true);
        identity["status"] = json!("dirty");
        identity["problems"] = json!(["dirty"]);
        assert!(validate_e0_build_identity(&identity).is_err());
    }

    #[test]
    fn acquisition_artifacts_are_distinct_and_trace_is_exclusive() {
        assert!(
            validate_distinct_artifact_paths(&[("output", "/tmp/a"), ("manifest", "/tmp/b")])
                .is_ok()
        );
        assert!(
            validate_distinct_artifact_paths(&[("output", "/tmp/a"), ("manifest", "/tmp/a")])
                .is_err()
        );
        let path = std::env::temp_dir().join(format!(
            "qwen-dflash-e0-exclusive-{}-{}.jsonl",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut writer = JsonlAppender::create_exclusive(&path).unwrap();
        writer.write(&json!({"event": "reservation"})).unwrap();
        writer.finish().unwrap();
        assert!(JsonlAppender::create_exclusive(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn bootstrap_and_run_metadata_share_complete_artifact_paths() {
        let paths = evidence_paths_json(
            "/model",
            "/drafter",
            "/manifest",
            "/trace",
            "/state",
            "/executable",
        );
        let bootstrap = json!({"paths": paths.clone()});
        let run = json!({"paths": paths});
        assert_eq!(bootstrap["paths"], run["paths"]);
        assert_eq!(bootstrap["paths"].as_object().unwrap().len(), 6);
        assert_eq!(bootstrap["paths"]["executable"], "/executable");
    }

    #[test]
    fn binding_manifest_v2_requires_the_exact_arm_order() {
        let manifest = json!({
            "allowed_arm_orders": ["serial_then_capture", "capture_then_serial"],
            "required_arm_order": "capture_then_serial",
        });
        assert!(
            validate_binding_manifest_arm_order(
                &manifest,
                BINDING_MANIFEST_VERSION_V2,
                ArmOrder::CaptureThenSerial,
            )
            .is_ok()
        );
        assert!(
            validate_binding_manifest_arm_order(
                &manifest,
                BINDING_MANIFEST_VERSION_V2,
                ArmOrder::SerialThenCapture,
            )
            .is_err()
        );
    }
}
