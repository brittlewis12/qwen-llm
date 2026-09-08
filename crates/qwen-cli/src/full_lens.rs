use anyhow::{Context, Result, ensure};
use blake3::Hasher as Blake3Hasher;
use clap::{ArgGroup, Args, ValueEnum};
use qwen_llm::checkpoint_identity::{CheckpointIdentityCache, checkpoint_content_identity};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::host_page_size_bytes;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::runtime::{LoadedModel, LoadedModelConfig, ModelLoadIntent, Runtime, SequenceConfig};
use qwen_llm::tokenizer::Tokenizer;
use qwen_llm::workspace_lens::{
    MAX_WORKSPACE_LENS_PACKED_READOUT_POSITIONS, WORKSPACE_LENS_IDENTITY_SCHEME,
    WorkspaceLensError, WorkspaceLensPackedPostBlockCapture, WorkspaceLensPackedTransportedVector,
    WorkspaceLensPackedVocabularyPosition,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fs::{DirBuilder, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use zip::ZipArchive;

use crate::lens_input::{
    LensCohortRequest, LensInputRendering, LensInputSpec, LensMessageMode,
    prepare_qwen_model_input, validate_lens_input_spec,
};

use super::published_pt::{
    ArchiveLayout, ArchiveSpec, ensure_finite_f16, hash_sha256, validate_archive,
};
use super::{
    ByteLimitedWriter, FitMethod, JSON_FILE_MAX_BYTES, ORIENTATION, SCHEMA_VERSION,
    TOKEN_ARTIFACT_MAX_BYTES, TOKEN_ID_ARGUMENT_MAX_COUNT, TOKEN_MANIFEST_NAME, TOKEN_ORIENTATION,
    TOKEN_PAYLOAD_NAME, TOKEN_READOUT_SCHEMA, TokenReadoutManifest, decode_f32_le, digest_json,
    hex, open_regular_file, publish_immutable, read_bounded_jsonl_record, read_json_file,
    resolve_output_file_path, resolve_output_path, serialize_json_pretty_bounded, sync_directory,
    token_covector_digest, validate_token_build_identity, validate_token_readout_spec,
    write_atomic_replace,
};

mod access;
mod compare;
pub(crate) use access::{BoundFullAccess, FullAccess, FullExecutionMode};
mod import;
mod readout;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use tests::generic_bound_trace_json;
#[cfg(test)]
pub(crate) use tests::write_cpu_gguf;
mod token_bank;
mod trace;
mod trace_batch;
#[allow(unused_imports)]
pub(crate) use compare::*;
#[allow(unused_imports)]
pub(crate) use import::*;
#[allow(unused_imports)]
pub(crate) use readout::*;
#[allow(unused_imports)]
pub(crate) use token_bank::*;
#[allow(unused_imports)]
pub(crate) use trace::*;
#[allow(unused_imports)]
pub(crate) use trace_batch::*;

const FULL_SCHEMA: &str = "qwen.workspace_lens_full_transport";
const FULL_SCHEMA_VERSION: u32 = 1;
const FITTED_CHECKPOINT_REVISION: &str = "32a8451f38193fc75b72146ac69afe12e8f6326d";
const HIDDEN_SIZE: usize = 5_120;
const SOURCE_LAYER_COUNT: usize = 63;
const N_LAYERS: u32 = 64;
const VOCAB_SIZE: u32 = 248_320;
const MATRIX_BYTES: u64 = (HIDDEN_SIZE as u64) * (HIDDEN_SIZE as u64) * 2;
const COPY_BUFFER_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Args)]
pub(crate) struct ReadFullArgs {
    /// Ordinary Qwen dense/MoE or Muse Glimmer GGUF used for capture and output.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,

    /// Data-only linear transport or a legacy imported/assembled full lens directory.
    #[arg(
        long,
        conflicts_with = "logit_lens",
        required_unless_present = "logit_lens"
    )]
    pub(crate) full_lens: Option<PathBuf>,

    /// Read native post-block residuals through the deployed output tail, without a fit.
    #[arg(long, conflicts_with = "full_lens")]
    pub(crate) logit_lens: bool,

    /// Text prompt. Exactly one of --prompt or --token-ids is required.
    #[arg(
        long,
        allow_hyphen_values = true,
        conflicts_with = "token_ids",
        required_unless_present = "token_ids"
    )]
    pub(crate) prompt: Option<String>,

    /// Literal prompt token IDs. Exactly one of --prompt or --token-ids is required.
    #[arg(
        long,
        value_delimiter = ',',
        conflicts_with = "prompt",
        required_unless_present = "prompt"
    )]
    pub(crate) token_ids: Vec<u32>,

    /// Disable tokenizer-configured BOS/EOS insertion for text prompts.
    #[arg(long)]
    pub(crate) no_special_tokens: bool,

    /// Input position to inspect; defaults to the final prompt token.
    #[arg(long)]
    pub(crate) position: Option<usize>,

    /// Layers in output order; defaults to all model blocks for plain lens, otherwise artifact sources.
    #[arg(long, value_delimiter = ',')]
    pub(crate) layers: Vec<u32>,

    /// Results per layer (plain maximum 1024; fitted Qwen 25, fitted Muse 32).
    #[arg(long, default_value_t = 10)]
    pub(crate) top_k: usize,

    /// Reject prompts above this bound instead of silently truncating them.
    #[arg(long, default_value_t = 256)]
    pub(crate) max_tokens: usize,

    /// Private cache directory for the strong ordered-GGUF content identity.
    #[arg(long)]
    pub(crate) identity_cache: PathBuf,

    /// Acknowledge unverified source/deployment equivalence; exact bindings still must match.
    #[arg(long)]
    pub(crate) allow_unvalidated_transfer: bool,

    /// Include each selected pre-output-norm transported hidden vector.
    #[arg(long)]
    pub(crate) include_vector: bool,

    /// Immutable full-vocabulary F32 LE bundle (metadata.json and logits.f32le).
    #[arg(long, conflicts_with = "output")]
    pub(crate) full_output: Option<PathBuf>,

    /// Optional immutable deterministic JSON result.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FullTransport {
    method: String,
    target_layer: u32,
    source_layers: Vec<u32>,
    capture_site: String,
    orientation: String,
    hidden_size: u32,
    bias: String,
    storage_dtype: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FullModel {
    base_model: String,
    fitted_checkpoint: String,
    fitted_checkpoint_revision: String,
    architecture: String,
    n_layers: u32,
    hidden_size: u32,
    vocab_size: u32,
    output_norm: String,
    unembedding: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PublishedFit {
    fitter: String,
    fitter_revision: String,
    dataset: String,
    split: String,
    n_prompts: u64,
    max_sequence_length: u32,
    skip_first: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    valid_positions_per_prompt: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dim_batch: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_execution_dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accumulator_dtype: Option<String>,
    serialized_dtype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    docs_consumed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    n_positions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    config_json: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    weighting: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    corpus_mode: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PublishedSource {
    repository: String,
    revision: String,
    filename: String,
    byte_length: u64,
    sha256: String,
    data_pickle_sha256: String,
    license: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct TransferPolicy {
    fitted_weight_precision: String,
    deployed_checkpoint_policy: String,
    validation_status: String,
}

#[derive(Debug, Serialize)]
struct TransferReport {
    schema: &'static str,
    schema_version: u32,
    comparison_status: &'static str,
    comparison_scope: &'static str,
    quantization_effect_isolated: bool,
    publisher_claims_basis: &'static str,
    comparison_factors_not_isolated: Vec<&'static str>,
    covector_semantics: &'static str,
    rms_denominator_applied: bool,
    token_selection: &'static str,
    full_lens: TransferFullLens,
    deployed_model: TransferModel,
    native_fit: TransferNativeFit,
    selected_token_ids: Vec<u32>,
    aggregate: TransferAggregate,
    layers: Vec<TransferLayerAggregate>,
    directions: Vec<TransferDirection>,
}

#[derive(Debug, Serialize)]
struct TransferFullLens {
    repository: String,
    revision: String,
    payload_blake3: String,
    fitted_checkpoint: String,
    fitted_checkpoint_revision: String,
    n_prompts: u64,
    max_sequence_length: u32,
    skip_first: u32,
    valid_positions_per_prompt: Option<u32>,
}

#[derive(Debug, Serialize)]
struct TransferModel {
    path: PathBuf,
    content_blake3: String,
    architecture: String,
    n_layers: u32,
    hidden_size: u32,
    vocab_size: u32,
    lm_head_dtype: String,
}

#[derive(Debug, Serialize)]
struct TransferNativeFit {
    artifact: PathBuf,
    config_blake3: String,
    n_prompts: u64,
    max_tokens: usize,
    skip_first: usize,
    estimator_version: String,
    orientation: String,
    corpus_blake3: String,
    add_special_tokens: bool,
    truncated_prompts: u64,
    accumulator_dtype: String,
    target_layer: u32,
    source_layers: Vec<u32>,
    payload_blake3: String,
}

#[derive(Debug, Serialize)]
struct TransferAggregate {
    direction_count: usize,
    mean_cosine_similarity: f64,
    minimum_cosine_similarity: f64,
    mean_relative_l2_error: f64,
    maximum_relative_l2_error: f64,
    mean_norm_ratio_public_over_native: f64,
}

#[derive(Debug, Serialize)]
struct TransferLayerAggregate {
    source_layer: u32,
    direction_count: usize,
    mean_cosine_similarity: f64,
    minimum_cosine_similarity: f64,
    mean_relative_l2_error: f64,
    mean_norm_ratio_public_over_native: f64,
}

#[derive(Debug, Serialize)]
struct TransferDirection {
    source_layer: u32,
    token_id: u32,
    cosine_similarity: f64,
    relative_l2_error: f64,
    norm_ratio_public_over_native: f64,
    public_norm: f64,
    native_norm: f64,
    maximum_absolute_difference: f32,
}

fn direction_metrics(
    source_layer: u32,
    token_id: u32,
    public: &[f32],
    native: &[f32],
) -> Result<TransferDirection> {
    ensure!(
        public.len() == native.len() && !public.is_empty(),
        "direction shape mismatch"
    );
    let mut dot = 0.0f64;
    let mut public_sq = 0.0f64;
    let mut native_sq = 0.0f64;
    let mut difference_sq = 0.0f64;
    let mut maximum_absolute_difference = 0.0f32;
    for (&public, &native) in public.iter().zip(native) {
        ensure!(
            public.is_finite() && native.is_finite(),
            "direction contains non-finite data"
        );
        let public = f64::from(public);
        let native = f64::from(native);
        let difference = public - native;
        dot += public * native;
        public_sq += public * public;
        native_sq += native * native;
        difference_sq += difference * difference;
        maximum_absolute_difference = maximum_absolute_difference.max(difference.abs() as f32);
    }
    ensure!(
        public_sq > 0.0 && native_sq > 0.0,
        "direction has zero norm"
    );
    let public_norm = public_sq.sqrt();
    let native_norm = native_sq.sqrt();
    Ok(TransferDirection {
        source_layer,
        token_id,
        cosine_similarity: dot / (public_norm * native_norm),
        relative_l2_error: difference_sq.sqrt() / native_norm,
        norm_ratio_public_over_native: public_norm / native_norm,
        public_norm,
        native_norm,
        maximum_absolute_difference,
    })
}

fn aggregate_metrics(directions: &[TransferDirection]) -> Result<TransferAggregate> {
    ensure!(
        !directions.is_empty(),
        "transfer comparison produced no directions"
    );
    let count = directions.len() as f64;
    Ok(TransferAggregate {
        direction_count: directions.len(),
        mean_cosine_similarity: directions
            .iter()
            .map(|direction| direction.cosine_similarity)
            .sum::<f64>()
            / count,
        minimum_cosine_similarity: directions
            .iter()
            .map(|direction| direction.cosine_similarity)
            .fold(f64::INFINITY, f64::min),
        mean_relative_l2_error: directions
            .iter()
            .map(|direction| direction.relative_l2_error)
            .sum::<f64>()
            / count,
        maximum_relative_l2_error: directions
            .iter()
            .map(|direction| direction.relative_l2_error)
            .fold(0.0, f64::max),
        mean_norm_ratio_public_over_native: directions
            .iter()
            .map(|direction| direction.norm_ratio_public_over_native)
            .sum::<f64>()
            / count,
    })
}

fn aggregate_layer_metrics(
    directions: &[TransferDirection],
) -> Result<Vec<TransferLayerAggregate>> {
    let mut by_layer: BTreeMap<u32, Vec<&TransferDirection>> = BTreeMap::new();
    for direction in directions {
        by_layer
            .entry(direction.source_layer)
            .or_default()
            .push(direction);
    }
    ensure!(
        !by_layer.is_empty(),
        "transfer comparison produced no layers"
    );
    Ok(by_layer
        .into_iter()
        .map(|(source_layer, layer)| {
            let count = layer.len() as f64;
            TransferLayerAggregate {
                source_layer,
                direction_count: layer.len(),
                mean_cosine_similarity: layer
                    .iter()
                    .map(|direction| direction.cosine_similarity)
                    .sum::<f64>()
                    / count,
                minimum_cosine_similarity: layer
                    .iter()
                    .map(|direction| direction.cosine_similarity)
                    .fold(f64::INFINITY, f64::min),
                mean_relative_l2_error: layer
                    .iter()
                    .map(|direction| direction.relative_l2_error)
                    .sum::<f64>()
                    / count,
                mean_norm_ratio_public_over_native: layer
                    .iter()
                    .map(|direction| direction.norm_ratio_public_over_native)
                    .sum::<f64>()
                    / count,
            }
        })
        .collect())
}

fn validate_artifact_directory(directory: &Path, name: &str) -> Result<()> {
    let metadata = std::fs::symlink_metadata(directory)
        .with_context(|| format!("inspect {name} directory {}", directory.display()))?;
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "{name} {} must be a real directory",
        directory.display()
    );
    Ok(())
}

pub(super) fn resolve_output_file(output: &Path) -> Result<PathBuf> {
    let leaf = output
        .file_name()
        .filter(|leaf| !leaf.is_empty() && *leaf != "." && *leaf != "..")
        .context("report output must name a non-root file")?;
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let canonical_parent = std::fs::canonicalize(parent)
        .with_context(|| format!("resolve report output parent {}", parent.display()))?;
    let metadata = std::fs::symlink_metadata(&canonical_parent).with_context(|| {
        format!(
            "inspect report output parent {}",
            canonical_parent.display()
        )
    })?;
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "report output parent {} must be a real directory",
        canonical_parent.display()
    );
    let resolved = canonical_parent.join(leaf);
    if let Ok(metadata) = std::fs::symlink_metadata(&resolved) {
        ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "report output {} must be a regular non-symlink file",
            resolved.display()
        );
    }
    Ok(resolved)
}

fn transport_matrix_bytes(manifest: &FullLensManifest) -> Result<u64> {
    ensure!(
        manifest.payload.dtype == "f16_le"
            && manifest.payload.shape[0] == manifest.transport.source_layers.len()
            && manifest.payload.shape[1] == manifest.transport.hidden_size as usize
            && manifest.payload.shape[2] == manifest.transport.hidden_size as usize,
        "full lens payload shape does not match its transport"
    );
    let matrix_bytes = u64::from(manifest.transport.hidden_size)
        .checked_mul(u64::from(manifest.transport.hidden_size))
        .and_then(|words| words.checked_mul(2))
        .context("full lens matrix byte count overflow")?;
    ensure!(
        matrix_bytes.checked_mul(manifest.transport.source_layers.len() as u64)
            == Some(manifest.payload.byte_length),
        "full lens payload length does not match its matrix inventory"
    );
    Ok(matrix_bytes)
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
