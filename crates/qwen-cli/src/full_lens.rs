use anyhow::{Context, Result, ensure};
use blake3::Hasher as Blake3Hasher;
use clap::{ArgGroup, Args, ValueEnum};
use qwen_llm::checkpoint_identity::{CheckpointIdentityCache, checkpoint_content_identity};
use qwen_llm::gguf::GgufFile;
use qwen_llm::research::{MAX_RESEARCH_PACKED_READOUT_POSITIONS, RESEARCH_IDENTITY_SCHEME};
use qwen_llm::runtime::{Runtime, SequenceConfig};
use qwen_llm::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{DirBuilder, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use zip::ZipArchive;

use crate::messages::{
    AnnotatedMessageRender, MessageRenderSpanKind, Qwen38GenerationMode, Qwen38ReasoningEffort,
    QwenGenerationMode, parse_strict_messages_input,
    render_qwen_messages_prompt_with_generation_annotated,
    render_qwen38_messages_prompt_with_generation_annotated,
};

use super::published_pt::{ArchiveSpec, ensure_finite_f16, hash_sha256, validate_archive};
use super::{
    FitMethod, ORIENTATION, SCHEMA_VERSION, TOKEN_ARTIFACT_MAX_BYTES, TOKEN_ID_ARGUMENT_MAX_COUNT,
    TOKEN_MANIFEST_NAME, TOKEN_ORIENTATION, TOKEN_PAYLOAD_NAME, TOKEN_READOUT_SCHEMA,
    TokenReadoutManifest, decode_f32_le, digest_json, hex, open_regular_file, publish_immutable,
    read_json_file, resolve_output_file_path, resolve_output_path, serialize_json_pretty_bounded,
    sync_directory, token_covector_digest, validate_token_build_identity,
    validate_token_readout_spec, write_atomic_replace,
};

const FULL_SCHEMA: &str = "qwen.workspace_lens_full_transport";
const FULL_SCHEMA_VERSION: u32 = 1;
const FULL_MANIFEST_NAME: &str = "lens.json";
const FULL_PAYLOAD_NAME: &str = "transport.f16le";
const FITTED_CHECKPOINT_REVISION: &str = "32a8451f38193fc75b72146ac69afe12e8f6326d";
const HIDDEN_SIZE: usize = 5_120;
const SOURCE_LAYER_COUNT: usize = 63;
const N_LAYERS: u32 = 64;
const VOCAB_SIZE: u32 = 248_320;
const MATRIX_BYTES: u64 = (HIDDEN_SIZE as u64) * (HIDDEN_SIZE as u64) * 2;
const PAYLOAD_BYTES: u64 = MATRIX_BYTES * (SOURCE_LAYER_COUNT as u64);
const COPY_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const MAX_PROJECTED_FULL_TOKENS: usize = 32;
const MAX_FULL_READOUT_PROMPT_TOKENS: usize = 4_096;
const MAX_FULL_READOUT_TOP_K: usize = 16;
const MAX_TRACE_FULL_VECTOR_CELLS: usize = 32;
const MAX_TRACE_DOCUMENT_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PublishedProfileId {
    Qwen38J,
    Qwen36J,
    Qwen36R,
}

#[derive(Clone, Copy, Debug)]
struct PublishedProfile {
    id: PublishedProfileId,
    method: &'static str,
    source_repository: &'static str,
    source_revision: &'static str,
    source_filename: &'static str,
    source_bytes: u64,
    source_sha256: &'static str,
    data_pickle_sha256: &'static str,
    expected_payload_blake3: &'static str,
    archive_root: &'static str,
    target_layer: u32,
    identity_anchor_layer: Option<u32>,
    base_model: &'static str,
    fitted_checkpoint: &'static str,
    fitted_checkpoint_revision: &'static str,
    model_name_fragment: &'static str,
    license: &'static str,
}

const PUBLISHED_PROFILES: [PublishedProfile; 3] = [
    PublishedProfile {
        id: PublishedProfileId::Qwen38J,
        method: "j",
        source_repository: "eyes-ml/Qwen3.8-27B_jacobian-lens",
        source_revision: "f8608c19b441f605d87ce46b80184f3774d75f2c",
        source_filename: "Qwen3.8-27B_jacobian_lens.pt",
        source_bytes: 3_303_033_664,
        source_sha256: "6b51f369e45a68b7eb775081ba5d195bb41360fb2abd15b0ab5b49881b638d49",
        data_pickle_sha256: "3e58341435e2178dc78af9689fab7dd872661063e5087848b0099f436aa7d448",
        expected_payload_blake3: "4a75d250d754d6e02f7d865bf84253a4804df3f49a7f5ccd6059c74f0f26b9e3",
        archive_root: "Qwen3.8-27B_jacobian_lens",
        target_layer: 63,
        identity_anchor_layer: None,
        base_model: "Qwen/Qwen3.8-27B",
        fitted_checkpoint: "eyes-ml/Qwen3.8-27B",
        fitted_checkpoint_revision: FITTED_CHECKPOINT_REVISION,
        model_name_fragment: "qwen3.8",
        license: "Apache-2.0",
    },
    PublishedProfile {
        id: PublishedProfileId::Qwen36J,
        method: "j",
        source_repository: "camilablank/workspace-lenses",
        source_revision: "d740106d1e0f95456dc8718fba2895e9c8ffd6ef",
        source_filename: "qwen3.6-27b/j-lens/lens.pt",
        source_bytes: 3_303_028_503,
        source_sha256: "a036b35843d389b6655df721711917436fb79c83358c1861a4d58ad103a02724",
        data_pickle_sha256: "4e0c9b7e0b2f362d3711813314073a66eb76657d84875f038d3688705a3a3f70",
        expected_payload_blake3: "a76c67d0c977696970511bbaf23baeb9e02f6a526c873069b4bb18c92317965e",
        archive_root: "lens",
        target_layer: 62,
        identity_anchor_layer: Some(62),
        base_model: "Qwen/Qwen3.6-27B",
        fitted_checkpoint: "Qwen/Qwen3.6-27B",
        fitted_checkpoint_revision: "not_recorded_in_published_artifact",
        model_name_fragment: "qwen3.6",
        license: "MIT",
    },
    PublishedProfile {
        id: PublishedProfileId::Qwen36R,
        method: "r",
        source_repository: "camilablank/workspace-lenses",
        source_revision: "d740106d1e0f95456dc8718fba2895e9c8ffd6ef",
        source_filename: "qwen3.6-27b/r-lens/lens.pt",
        source_bytes: 3_303_028_567,
        source_sha256: "fe4d0b6a17318760e67e7a5ed417fc5dbb8e66129c6944656f506b2f10ce6192",
        data_pickle_sha256: "fcb9a42587adf9069e1fe5189d88c37ff3f505c5e62825ad5ce2da764ad4e424",
        expected_payload_blake3: "0be52938f7a3b6f9e419017aa03036b70fe97e18d304193116b8263fac69f58e",
        archive_root: "lens",
        target_layer: 62,
        identity_anchor_layer: Some(62),
        base_model: "Qwen/Qwen3.6-27B",
        fitted_checkpoint: "Qwen/Qwen3.6-27B",
        fitted_checkpoint_revision: "not_recorded_in_published_artifact",
        model_name_fragment: "qwen3.6",
        license: "MIT",
    },
];

#[derive(Debug, Args)]
pub(crate) struct ImportFullArgs {
    /// One supported exact pinned published .pt transport asset.
    #[arg(long)]
    source: PathBuf,

    /// New immutable full-lens artifact directory.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct CompareTransferArgs {
    /// Dense Qwen3.8 GGUF model whose own output norm and LM head define readouts.
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Directory produced by `qwen-lens import-full`.
    #[arg(long)]
    full_lens: PathBuf,

    /// Native selected-token J-lens directory produced by `fit-tokens`.
    #[arg(long)]
    native_readouts: PathBuf,

    /// Private cache directory for the strong ordered-GGUF content identity.
    #[arg(long)]
    identity_cache: PathBuf,

    /// Optional immutable JSON report path; deterministic report is always printed.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct ReadFullArgs {
    /// Dense Qwen3.6, Qwen3.8, or Muse Glimmer GGUF used for capture and output.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,

    /// Directory produced by `import-full` or `assemble-muse-full`.
    #[arg(long)]
    pub(crate) full_lens: PathBuf,

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

    /// Source layers in output order; defaults to all 0..62.
    #[arg(long, value_delimiter = ',')]
    pub(crate) layers: Vec<u32>,

    /// Full-vocabulary results per layer (maximum 16).
    #[arg(long, default_value_t = 10)]
    pub(crate) top_k: usize,

    /// Reject prompts above this bound instead of silently truncating them.
    #[arg(long, default_value_t = 256)]
    pub(crate) max_tokens: usize,

    /// Private cache directory for the strong ordered-GGUF content identity.
    #[arg(long)]
    pub(crate) identity_cache: PathBuf,

    /// Acknowledge Qwen published-lens transfer; not needed for model-bound Muse assets.
    #[arg(long)]
    pub(crate) allow_unvalidated_transfer: bool,

    /// Include each selected pre-output-norm transported hidden vector.
    #[arg(long)]
    pub(crate) include_vector: bool,

    /// Optional immutable deterministic JSON result.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,
}

#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("trace_full_input")
        .required(true)
        .multiple(false)
        .args(["prompt", "token_ids", "messages"])
))]
pub(crate) struct TraceFullArgs {
    /// Matching dense Qwen3.6 or Qwen3.8 GGUF used for packed capture and readout.
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Directory produced by `qwen-lens import-full`.
    #[arg(long)]
    full_lens: PathBuf,

    /// Raw text prompt; tokenizer-configured specials are enabled by default.
    #[arg(long, allow_hyphen_values = true)]
    prompt: Option<String>,

    /// Literal comma-separated token IDs; no specials are added.
    #[arg(long, value_delimiter = ',')]
    token_ids: Option<Vec<i32>>,

    /// Strict system/user/assistant message array or wrapper JSON.
    #[arg(long)]
    messages: Option<PathBuf>,

    /// Generation transition for --messages; supported values depend on the lens model.
    #[arg(
        long,
        value_enum,
        requires = "messages",
        conflicts_with_all = ["prompt", "token_ids"]
    )]
    message_mode: Option<TraceFullMessageMode>,

    /// Disable tokenizer-configured special insertion for --prompt.
    #[arg(
        long,
        requires = "prompt",
        conflicts_with_all = ["token_ids", "messages"]
    )]
    no_special_tokens: bool,

    /// Unique source layers in caller output order; defaults to every artifact layer.
    #[arg(long, value_delimiter = ',')]
    layers: Vec<u32>,

    /// Full-vocabulary results per layer and position.
    #[arg(long, default_value_t = 8)]
    top_k: usize,

    /// Reject inputs above this bound without truncating them.
    #[arg(long, default_value_t = MAX_RESEARCH_PACKED_READOUT_POSITIONS)]
    max_tokens: usize,

    /// Transported target-space vectors to include as layer:position cells.
    #[arg(long = "vectors", value_delimiter = ',')]
    vectors: Vec<TraceFullVectorCell>,

    /// Replace this JSON result file atomically after a successful trace.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Human summary or the complete JSON document on stdout.
    #[arg(long, value_enum)]
    format: Option<TraceFullStdoutFormat>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum TraceFullStdoutFormat {
    Summary,
    Json,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum TraceFullMessageMode {
    Auto,
    Thinking,
    NoThinking,
    Low,
    Medium,
    Xhigh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResolvedTraceFullMessageMode {
    Qwen36(QwenGenerationMode),
    Qwen38(Qwen38GenerationMode),
}

impl ResolvedTraceFullMessageMode {
    const fn artifact_name(self) -> &'static str {
        match self {
            Self::Qwen36(QwenGenerationMode::Auto) => "auto",
            Self::Qwen36(QwenGenerationMode::Thinking) => "thinking",
            Self::Qwen36(QwenGenerationMode::NoThinking) => "no_thinking",
            Self::Qwen38(Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Low)) => {
                "thinking_low"
            }
            Self::Qwen38(Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Medium)) => {
                "thinking_medium"
            }
            Self::Qwen38(Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Xhigh)) => {
                "thinking_xhigh"
            }
            Self::Qwen38(Qwen38GenerationMode::NoThinking) => "no_thinking",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct TraceFullVectorCell {
    source_layer: u32,
    source_position: usize,
}

pub(crate) struct ProjectedFullTokenDirections {
    pub(crate) method: String,
    pub(crate) target_layer: u32,
    pub(crate) source_layers: Vec<u32>,
    pub(crate) token_ids: Vec<i32>,
    pub(crate) hidden_size: usize,
    pub(crate) values: Vec<f32>,
}

impl FromStr for TraceFullVectorCell {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let mut parts = value.split(':');
        let layer = parts.next().unwrap_or_default();
        let position = parts.next().unwrap_or_default();
        if parts.next().is_some()
            || layer.is_empty()
            || position.is_empty()
            || !layer.bytes().all(|byte| byte.is_ascii_digit())
            || !position.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(format!(
                "vector cell {value:?} must be exactly LAYER:POSITION using unsigned decimal integers"
            ));
        }
        Ok(Self {
            source_layer: layer
                .parse()
                .map_err(|_| format!("vector layer {layer:?} does not fit u32"))?,
            source_position: position
                .parse()
                .map_err(|_| format!("vector position {position:?} does not fit usize"))?,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FullLensManifest {
    schema: String,
    schema_version: u32,
    status: String,
    transport: FullTransport,
    model: FullModel,
    fit: PublishedFit,
    source: PublishedSource,
    payload: FullPayload,
    transfer: TransferPolicy,
    provenance: ImportProvenance,
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
struct FullPayload {
    path: String,
    dtype: String,
    shape: [usize; 3],
    byte_length: u64,
    blake3: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct TransferPolicy {
    fitted_weight_precision: String,
    deployed_checkpoint_policy: String,
    validation_status: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ImportProvenance {
    build_commit: String,
    build_dirty: String,
    build_source_state: String,
    build_stamp_source: String,
    build_stamp_error: String,
    pickle_execution: String,
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

#[derive(Debug, Serialize)]
struct FullReadoutDocument {
    schema: &'static str,
    schema_version: u32,
    readout: &'static str,
    scoring: &'static str,
    score_semantics: &'static str,
    ranking_scope: &'static str,
    source_site: &'static str,
    input: FullReadoutInput,
    artifact: FullReadoutArtifact,
    deployed_model: FullReadoutModel,
    transfer: FullReadoutTransfer,
    reader: FullReadoutReader,
    results: Vec<FullLayerReadout>,
}

#[derive(Debug, Serialize)]
struct FullReadoutInput {
    source: &'static str,
    add_special_tokens: Option<bool>,
    token_ids: Vec<i32>,
    selected_position: usize,
    captured_token_id: i32,
    predicts_position: usize,
}

#[derive(Debug, Serialize)]
struct FullReadoutArtifact {
    manifest: PathBuf,
    manifest_canonical_json_blake3: String,
    payload_blake3: String,
    method: String,
    target_layer: u32,
    orientation: String,
    source_repository: String,
    source_revision: String,
    fitted_checkpoint: String,
    fitted_checkpoint_revision: String,
    fit_n_prompts: u64,
    fit_max_sequence_length: u32,
    fit_skip_first: u32,
}

#[derive(Debug, Serialize)]
struct FullReadoutModel {
    path: PathBuf,
    content_blake3: String,
    model_locator_id: String,
    tokenizer_metadata_id: String,
    architecture_contract: &'static str,
    n_layers: u32,
    hidden_size: u32,
    vocab_size: u32,
    full_attention_interval: u32,
    lm_head_dtype: String,
}

#[derive(Debug, Serialize)]
struct FullReadoutTransfer {
    validation_status: String,
    override_policy: &'static str,
}

#[derive(Debug, Serialize)]
struct FullReadoutReader {
    build_commit: &'static str,
    build_dirty: &'static str,
    build_source_state: &'static str,
    build_stamp_source: &'static str,
    build_stamp_error: &'static str,
}

#[derive(Debug, Serialize)]
struct FullLayerReadout {
    source_layer: u32,
    source_position: usize,
    source_token_id: i32,
    predicts_position: usize,
    rms_denominator_f64_recomputed: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    transported_vector: Option<FullTransportedVector>,
    top_k: Vec<FullTokenScore>,
}

#[derive(Debug, Serialize)]
struct FullTransportedVector {
    operation: &'static str,
    stage: &'static str,
    value_dtype: &'static str,
    hidden_coordinate: &'static str,
    hidden_size: usize,
    shape: [usize; 1],
    values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct FullTokenScore {
    rank: usize,
    token_id: u32,
    token_display_lossy: String,
    token_piece_hex: String,
    logit: f32,
}

#[derive(Debug, Serialize)]
struct TraceFullDocument {
    schema: &'static str,
    schema_version: u32,
    producer: TraceFullProducer,
    deployed_model: TraceFullModel,
    tokenizer: TraceFullTokenizer,
    lens: TraceFullLens,
    score_semantics: TraceFullScoreSemantics,
    execution_mode: &'static str,
    input_source: &'static str,
    add_special_tokens: Option<bool>,
    input_token_ids: Vec<i32>,
    input_tokens: Vec<TraceFullInputToken>,
    rendering: TraceFullRendering,
    coordinates: TraceFullCoordinates,
    selected_layers: Vec<u32>,
    top_k: usize,
    occurrence_definition: &'static str,
    cells: Vec<TraceFullCell>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vectors: Option<TraceFullVectors>,
    timing: TraceFullTiming,
    occurrences: TraceFullOccurrences,
}

#[derive(Debug, Serialize)]
struct TraceFullProducer {
    build_commit: &'static str,
    build_dirty: &'static str,
    build_source_state: &'static str,
}

#[derive(Debug, Serialize)]
struct TraceFullModel {
    path: PathBuf,
    locator_scheme: &'static str,
    locator_id: String,
    content_authenticated: bool,
    architecture: Option<String>,
    name: Option<String>,
    base_model_name: Option<String>,
    n_layers: u32,
    hidden_size: u32,
    vocab_size: u32,
}

#[derive(Debug, Serialize)]
struct TraceFullTokenizer {
    metadata_id: String,
    model: Option<String>,
    pretokenizer: Option<String>,
}

#[derive(Debug, Serialize)]
struct TraceFullScoreSemantics {
    kind: &'static str,
    normalization: &'static str,
    candidate_universe: &'static str,
    softmax_applied: bool,
}

#[derive(Debug, Serialize)]
struct TraceFullRendering {
    renderer: &'static str,
    generation_mode: Option<&'static str>,
    spans: Vec<TraceFullRenderedSpan>,
}

#[derive(Debug, Serialize)]
struct TraceFullRenderedSpan {
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    message_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    channel: Option<&'static str>,
    byte_start: usize,
    byte_end: usize,
    token_start: Option<usize>,
    token_end: Option<usize>,
}

#[derive(Debug, Serialize)]
struct TraceFullLens {
    kind: &'static str,
    method: String,
    target_layer: u32,
    source_site: String,
    source_repository: String,
    source_revision: String,
    source_filename: String,
    payload_blake3: String,
    scoring: &'static str,
}

#[derive(Debug, Serialize)]
struct TraceFullInputToken {
    position: usize,
    token_id: i32,
    token_display_lossy: String,
    token_piece_hex: String,
}

#[derive(Debug, Serialize)]
struct TraceFullCoordinates {
    source_layer: &'static str,
    source_position: &'static str,
    predicts_position: &'static str,
    rank: &'static str,
}

#[derive(Debug, Serialize)]
struct TraceFullCell {
    source_layer: u32,
    source_position: usize,
    source_token_id: i32,
    predicts_position: usize,
    top_k: Vec<TraceFullTokenScore>,
}

#[derive(Debug, Serialize)]
struct TraceFullTokenScore {
    rank: usize,
    token_id: u32,
    token_display_lossy: String,
    token_piece_hex: String,
    logit: f32,
}

#[derive(Debug, Serialize)]
struct TraceFullVectors {
    operation: &'static str,
    stage: &'static str,
    value_dtype: &'static str,
    hidden_coordinate: &'static str,
    hidden_size: usize,
    shape: [usize; 2],
    cell_order: &'static str,
    cells: Vec<TraceFullVector>,
}

#[derive(Debug, Serialize)]
struct TraceFullVector {
    source_layer: u32,
    source_position: usize,
    source_token_id: i32,
    predicts_position: usize,
    values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct TraceFullTiming {
    packed_prefill_gpu_ms: f64,
    packed_prefill_wall_ms: f64,
    matrix_read_wall_ms: f64,
    readout_gpu_ms: f64,
    readout_command_wall_ms: f64,
    trace_execution_wall_ms: f64,
}

#[derive(Debug, Serialize)]
struct TraceFullOccurrences {
    global: Vec<TraceFullOccurrence>,
    per_layer: Vec<TraceFullLayerOccurrences>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct TraceFullOccurrence {
    token_id: u32,
    count: usize,
    top1_count: usize,
    best_rank: usize,
}

#[derive(Debug, Serialize)]
struct TraceFullLayerOccurrences {
    source_layer: u32,
    tokens: Vec<TraceFullOccurrence>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OccurrenceAccumulator {
    count: usize,
    top1_count: usize,
    best_rank: usize,
}

pub(crate) fn import_full(mut args: ImportFullArgs) -> Result<()> {
    validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    args.output = resolve_output_path(&args.output)?;
    let (mut source, source_length) = open_regular_file(&args.source)?;
    ensure!(
        PUBLISHED_PROFILES
            .iter()
            .any(|profile| profile.source_bytes == source_length as u64),
        "{} length {} does not match any supported pinned full-lens asset",
        args.source.display(),
        source_length,
    );
    let source_metadata = source
        .metadata()
        .with_context(|| format!("inspect opened {}", args.source.display()))?;
    let source_modified = source_metadata.modified().ok();
    let source_sha256 = hash_sha256(&mut source, &args.source)?;
    let profile = profile_for_source(source_length as u64, &source_sha256).with_context(|| {
        format!(
            "{} SHA-256 {} does not identify a supported pinned full-lens asset",
            args.source.display(),
            source_sha256
        )
    })?;

    prepare_output_directory(&args.output)?;
    let manifest_path = args.output.join(FULL_MANIFEST_NAME);
    if manifest_path.exists() {
        let manifest: FullLensManifest = read_json_file(&manifest_path)?;
        validate_manifest(&manifest)?;
        ensure!(
            profile_for_manifest(&manifest)?.id == profile.id,
            "existing output artifact was imported from a different published lens"
        );
        verify_payload(&args.output, &manifest.payload)?;
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }

    source
        .seek(SeekFrom::Start(0))
        .with_context(|| format!("rewind pinned source {}", args.source.display()))?;
    let mut archive = ZipArchive::new(source)
        .with_context(|| format!("open pinned torch ZIP {}", args.source.display()))?;
    let spec = ArchiveSpec {
        root: profile.archive_root,
        layer_count: SOURCE_LAYER_COUNT,
        hidden_size: HIDDEN_SIZE,
        matrix_bytes: MATRIX_BYTES,
        data_pickle_sha256: profile.data_pickle_sha256,
        serialization_id: None,
        identity_storage_index: profile
            .identity_anchor_layer
            .and_then(|layer| usize::try_from(layer).ok()),
    };
    validate_archive(&mut archive, spec)?;

    let staging = staging_path(&args.output, FULL_PAYLOAD_NAME)?;
    let import = (|| -> Result<FullPayload> {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&staging)
            .with_context(|| format!("create full-lens staging payload {}", staging.display()))?;
        let payload = extract_payload(&mut archive, spec, &mut output)?;
        output
            .sync_all()
            .with_context(|| format!("sync full-lens staging payload {}", staging.display()))?;
        drop(output);
        ensure!(
            payload.blake3 == profile.expected_payload_blake3,
            "imported transport payload BLAKE3 does not match the pinned source payload"
        );

        let source = archive.into_inner();
        let final_metadata = source
            .metadata()
            .with_context(|| format!("reinspect opened {}", args.source.display()))?;
        ensure!(
            final_metadata.len() == profile.source_bytes
                && final_metadata.modified().ok() == source_modified,
            "pinned source changed while it was being imported"
        );
        publish_streamed_payload(&staging, &args.output.join(FULL_PAYLOAD_NAME), &payload)?;
        Ok(payload)
    })();
    if import.is_err() {
        let _ = std::fs::remove_file(&staging);
    }
    let payload = import?;

    let manifest = published_manifest(profile, payload);
    validate_manifest(&manifest)?;
    publish_immutable(
        &manifest_path,
        &serialize_json_pretty_bounded(&manifest, "full lens manifest")?,
    )?;
    sync_directory(&args.output)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

pub(crate) fn compare_transfer(args: CompareTransferArgs) -> Result<()> {
    validate_artifact_directory(&args.full_lens, "full lens")?;
    validate_artifact_directory(&args.native_readouts, "native readouts")?;
    let full_manifest: FullLensManifest = read_json_file(&args.full_lens.join(FULL_MANIFEST_NAME))?;
    validate_manifest(&full_manifest)?;
    ensure!(
        profile_for_manifest(&full_manifest)?.id == PublishedProfileId::Qwen38J,
        "transfer comparison currently supports only the published Qwen3.8 J lens"
    );
    let (native_manifest, native_values) = load_native_j_readouts(&args.native_readouts)?;
    ensure!(
        native_manifest.readouts.token_ids.len() <= MAX_PROJECTED_FULL_TOKENS,
        "transfer comparison supports at most {} selected tokens, got {}",
        MAX_PROJECTED_FULL_TOKENS,
        native_manifest.readouts.token_ids.len()
    );
    ensure!(
        native_manifest.config.target_layer == full_manifest.transport.target_layer,
        "native target layer {} != published target layer {}",
        native_manifest.config.target_layer,
        full_manifest.transport.target_layer
    );
    let full_sources: BTreeSet<_> = full_manifest
        .transport
        .source_layers
        .iter()
        .copied()
        .collect();
    ensure!(
        native_manifest
            .config
            .source_layers
            .iter()
            .all(|layer| full_sources.contains(layer)),
        "native source layers are not a subset of the published lens"
    );

    let model_started = Instant::now();
    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_model(&args.model)
        .with_context(|| format!("load model {}", args.model.display()))?;
    let arch = loaded.arch();
    ensure!(
        arch.n_layer == full_manifest.model.n_layers
            && arch.hidden_size == full_manifest.model.hidden_size
            && arch.vocab_size == full_manifest.model.vocab_size,
        "deployed model geometry does not match the published full lens"
    );
    ensure!(
        arch.n_layer == native_manifest.config.n_layers
            && arch.hidden_size == native_manifest.config.hidden_size
            && arch.vocab_size == native_manifest.config.vocab_size
            && arch.full_attention_interval == native_manifest.config.full_attention_interval,
        "deployed model geometry does not match the native selected-token artifact"
    );
    let identity = loaded.research_identity();
    let content = checkpoint_content_identity(
        loaded.gguf(),
        &CheckpointIdentityCache::new(&args.identity_cache),
    )
    .with_context(|| {
        format!(
            "resolve strong model identity using {}",
            args.identity_cache.display()
        )
    })?;
    let model_content_blake3 = hex(&content.content_id);
    ensure!(
        native_manifest.config.model_content_blake3 == model_content_blake3
            && native_manifest.config.model_locator_id
                == format!("{:016x}", identity.model_locator_id)
            && native_manifest.config.tokenizer_metadata_id
                == format!("{:016x}", identity.tokenizer_metadata_id),
        "native selected-token artifact is not bound to the deployed GGUF"
    );
    let model_load_seconds = model_started.elapsed().as_secs_f64();

    let mut sequence = loaded
        .create_sequence(SequenceConfig::new(1))
        .context("create transfer-comparison research sequence")?;
    let research = loaded
        .research_session(&mut sequence)
        .context("open transfer-comparison research session")?;
    let selected = research
        .selected_token_readouts(&native_manifest.readouts.token_ids)
        .context("derive deployed-model selected-token covectors")?;
    ensure!(
        token_covector_digest(&selected.token_ids, selected.hidden_size, &selected.values)?
            == native_manifest.readouts.target_covectors_blake3,
        "deployed-model target covectors differ from the native fit artifact"
    );
    ensure!(
        format!("{:?}", selected.lm_head_dtype) == native_manifest.readouts.lm_head_dtype
            && selected.lm_head_shape == native_manifest.readouts.lm_head_shape
            && format!("{:?}", selected.output_norm_dtype)
                == native_manifest.readouts.output_norm_dtype
            && selected.output_norm_shape == native_manifest.readouts.output_norm_shape,
        "deployed-model readout metadata differs from the native fit artifact"
    );

    let payload_path = args.full_lens.join(&full_manifest.payload.path);
    let (mut payload_file, payload_length) = open_regular_file(&payload_path)?;
    ensure!(
        payload_length as u64 == full_manifest.payload.byte_length,
        "full lens payload length changed"
    );
    let hidden_size = selected.hidden_size;
    let token_count = selected.token_ids.len();
    let source_slots: BTreeMap<u32, usize> = native_manifest
        .config
        .source_layers
        .iter()
        .copied()
        .enumerate()
        .map(|(slot, layer)| (layer, slot))
        .collect();
    let matrix_bytes = transport_matrix_bytes(&full_manifest)?;
    let mut matrix =
        vec![0u8; usize::try_from(matrix_bytes).context("full-lens matrix byte count")?];
    let mut payload_hasher = Blake3Hasher::new();
    let mut directions = Vec::with_capacity(source_slots.len() * token_count);
    let projection_started = Instant::now();
    for &layer in &full_manifest.transport.source_layers {
        payload_file
            .read_exact(&mut matrix)
            .with_context(|| format!("read published transport layer {layer}"))?;
        ensure_finite_f16(&matrix, layer as usize, 0)?;
        payload_hasher.update(&matrix);
        let Some(&source_slot) = source_slots.get(&layer) else {
            continue;
        };
        eprintln!(
            "compare transfer source_layer={} tokens={}",
            layer, token_count
        );
        let projected = research
            .project_f16_transport_readouts(&matrix, &selected)
            .with_context(|| format!("project published transport layer {layer}"))?;
        ensure!(
            projected.len() == token_count * hidden_size,
            "published projection returned an invalid shape"
        );
        for (token_slot, &token_id) in selected.token_ids.iter().enumerate() {
            let public_start = token_slot * hidden_size;
            let native_start = (source_slot * token_count + token_slot) * hidden_size;
            directions.push(direction_metrics(
                layer,
                token_id,
                &projected[public_start..public_start + hidden_size],
                &native_values[native_start..native_start + hidden_size],
            )?);
        }
    }
    let mut extra = [0u8; 1];
    ensure!(
        payload_file.read(&mut extra).with_context(|| format!(
            "check end of full lens payload {}",
            payload_path.display()
        ))? == 0,
        "full lens payload contains trailing bytes"
    );
    ensure!(
        payload_hasher.finalize().to_hex().as_str() == full_manifest.payload.blake3,
        "full lens payload BLAKE3 mismatch"
    );
    let projection_seconds = projection_started.elapsed().as_secs_f64();
    let aggregate = aggregate_metrics(&directions)?;
    let layers = aggregate_layer_metrics(&directions)?;
    eprintln!(
        "transfer comparison timing model_load_seconds={model_load_seconds:.6} projection_seconds={projection_seconds:.6}"
    );
    let report = TransferReport {
        schema: "qwen.workspace_lens_transfer_comparison",
        schema_version: 1,
        comparison_status: "descriptive_unmatched_estimator",
        comparison_scope: "published_bf16_n1000_fit_vs_native_deployed_checkpoint_fit_includes_checkpoint_and_recipe_differences",
        quantization_effect_isolated: false,
        publisher_claims_basis: "pinned_hugging_face_repository_revision_and_model_card",
        comparison_factors_not_isolated: vec![
            "checkpoint_representation_and_execution_precision",
            "prompt_sample",
            "prompt_count",
            "maximum_sequence_length",
            "leading_position_skip",
            "tokenization_implementation",
            "fitter_implementation",
        ],
        covector_semantics: "deployed_lm_head_row_times_output_norm_gamma_logit_ranking_numerator",
        rms_denominator_applied: false,
        token_selection: "inherited_from_native_artifact_in_caller_order",
        full_lens: TransferFullLens {
            repository: full_manifest.source.repository,
            revision: full_manifest.source.revision,
            payload_blake3: full_manifest.payload.blake3,
            fitted_checkpoint: full_manifest.model.fitted_checkpoint,
            fitted_checkpoint_revision: full_manifest.model.fitted_checkpoint_revision,
            n_prompts: full_manifest.fit.n_prompts,
            max_sequence_length: full_manifest.fit.max_sequence_length,
            skip_first: full_manifest.fit.skip_first,
            valid_positions_per_prompt: full_manifest.fit.valid_positions_per_prompt,
        },
        deployed_model: TransferModel {
            path: args.model,
            content_blake3: model_content_blake3,
            architecture: native_manifest.config.architecture.clone(),
            n_layers: arch.n_layer,
            hidden_size: arch.hidden_size,
            vocab_size: arch.vocab_size,
            lm_head_dtype: native_manifest.readouts.lm_head_dtype.clone(),
        },
        native_fit: TransferNativeFit {
            artifact: args.native_readouts,
            config_blake3: native_manifest.config_blake3,
            n_prompts: native_manifest.corpus.used_prompts,
            max_tokens: native_manifest.config.max_tokens,
            skip_first: native_manifest.config.skip_first,
            estimator_version: native_manifest.config.estimator_version,
            orientation: native_manifest.config.orientation,
            corpus_blake3: native_manifest.config.corpus_blake3,
            add_special_tokens: native_manifest.config.add_special_tokens,
            truncated_prompts: native_manifest.corpus.truncated_prompts,
            accumulator_dtype: native_manifest.fit.accumulator_dtype,
            target_layer: native_manifest.config.target_layer,
            source_layers: native_manifest.config.source_layers,
            payload_blake3: native_manifest.payload.blake3,
        },
        selected_token_ids: selected.token_ids,
        aggregate,
        layers,
        directions,
    };
    let report_bytes = serialize_json_pretty_bounded(&report, "transfer comparison report")?;
    if let Some(output) = args.output {
        let output = resolve_output_file(&output)?;
        publish_immutable(&output, &report_bytes)?;
    }
    println!("{}", String::from_utf8(report_bytes).unwrap());
    Ok(())
}

pub(crate) fn project_full_token_directions(
    artifact: &Path,
    token_ids: &[u32],
    source_layers: &[u32],
    loaded: &qwen_llm::runtime::LoadedModel,
) -> Result<ProjectedFullTokenDirections> {
    validate_artifact_directory(artifact, "published full transport lens")?;
    let manifest: FullLensManifest = read_json_file(&artifact.join(FULL_MANIFEST_NAME))?;
    validate_manifest(&manifest)?;
    validate_deployed_model(&manifest, loaded)?;
    let arch = loaded.arch();
    ensure!(
        !token_ids.is_empty() && token_ids.len() <= MAX_PROJECTED_FULL_TOKENS,
        "published full transport lens requires 1..={} selected token IDs",
        MAX_PROJECTED_FULL_TOKENS
    );
    let mut unique_tokens = BTreeSet::new();
    ensure!(
        token_ids
            .iter()
            .all(|&token| token < arch.vocab_size && unique_tokens.insert(token)),
        "published full transport token IDs must be unique and inside the model vocabulary"
    );
    ensure!(
        !source_layers.is_empty()
            && source_layers.windows(2).all(|pair| pair[0] < pair[1])
            && source_layers
                .iter()
                .all(|layer| manifest.transport.source_layers.contains(layer)),
        "published full transport source layers must be nonempty, sorted, unique artifact layers"
    );

    let mut sequence = loaded
        .create_sequence(SequenceConfig::new(1))
        .context("create published full transport projection sequence")?;
    let research = loaded
        .research_session(&mut sequence)
        .context("open published full transport projection session")?;
    let selected = research
        .selected_token_readouts(token_ids)
        .context("derive deployed-model selected-token covectors")?;
    let hidden_size = selected.hidden_size;
    let projected_values = source_layers
        .len()
        .checked_mul(token_ids.len())
        .and_then(|value| value.checked_mul(hidden_size))
        .context("published full transport projected direction count overflow")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(projected_values)
        .context("allocate published full transport projected directions")?;

    let payload_path = artifact.join(&manifest.payload.path);
    let (mut payload, payload_length) = open_regular_file(&payload_path)?;
    ensure!(
        payload_length as u64 == manifest.payload.byte_length,
        "published full transport payload length does not match its manifest"
    );
    let matrix_bytes = transport_matrix_bytes(&manifest)?;
    let matrix_length =
        usize::try_from(matrix_bytes).context("full transport matrix byte count")?;
    let mut matrix = vec![0_u8; matrix_length];
    for &layer in source_layers {
        let layer_slot = manifest
            .transport
            .source_layers
            .iter()
            .position(|&candidate| candidate == layer)
            .context("published full transport layer disappeared after validation")?;
        let offset = u64::try_from(layer_slot)
            .context("published full transport layer slot does not fit u64")?
            .checked_mul(matrix_bytes)
            .context("published full transport matrix offset overflow")?;
        payload
            .seek(SeekFrom::Start(offset))
            .with_context(|| format!("seek published full transport source layer {layer}"))?;
        payload
            .read_exact(&mut matrix)
            .with_context(|| format!("read published full transport source layer {layer}"))?;
        ensure_finite_f16(&matrix, layer as usize, 0)?;
        let projected = research
            .project_f16_transport_readouts(&matrix, &selected)
            .with_context(|| format!("project published full transport source layer {layer}"))?;
        ensure!(
            projected.len() == token_ids.len() * hidden_size,
            "published full transport projection returned an invalid shape"
        );
        values.extend(projected);
    }
    ensure!(
        values.len() == projected_values && values.iter().all(|value| value.is_finite()),
        "published full transport projection returned invalid values"
    );

    Ok(ProjectedFullTokenDirections {
        method: format!(
            "published_{}_selected_token_numerator",
            manifest.transport.method
        ),
        target_layer: manifest.transport.target_layer,
        source_layers: source_layers.to_vec(),
        token_ids: selected
            .token_ids
            .into_iter()
            .map(|token| token as i32)
            .collect(),
        hidden_size,
        values,
    })
}

pub(crate) fn read_full(args: ReadFullArgs) -> Result<()> {
    validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    validate_read_full_args(&args)?;
    validate_artifact_directory(&args.full_lens, "full lens")?;
    let manifest_path = args.full_lens.join(FULL_MANIFEST_NAME);
    let manifest: FullLensManifest = read_json_file(&manifest_path)?;
    validate_manifest(&manifest)?;
    let manifest_canonical_json_blake3 = digest_json(&manifest)?;
    let layers = if args.layers.is_empty() {
        manifest.transport.source_layers.clone()
    } else {
        args.layers.clone()
    };
    let mut unique_layers = BTreeSet::new();
    ensure!(
        !layers.is_empty()
            && layers.iter().all(|layer| unique_layers.insert(*layer)
                && manifest.transport.source_layers.contains(layer)),
        "--layers must be unique source layers present in the full lens"
    );

    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_model(&args.model)
        .with_context(|| format!("load model {}", args.model.display()))?;
    validate_deployed_model(&manifest, &loaded)?;
    let arch = loaded.arch();
    let identity = loaded.research_identity();
    let content = checkpoint_content_identity(
        loaded.gguf(),
        &CheckpointIdentityCache::new(&args.identity_cache),
    )
    .with_context(|| {
        format!(
            "resolve strong model identity using {}",
            args.identity_cache.display()
        )
    })?;
    let tokenizer = loaded.tokenizer().context("load tokenizer from GGUF")?;
    let (input_source, add_special_tokens, token_ids) = if let Some(prompt) = &args.prompt {
        (
            "prompt",
            Some(!args.no_special_tokens),
            tokenizer
                .encode(prompt, !args.no_special_tokens)
                .context("tokenize full-lens prompt")?,
        )
    } else {
        let mut token_ids = Vec::new();
        token_ids
            .try_reserve_exact(args.token_ids.len())
            .context("allocate explicit full-lens token IDs")?;
        for &token_id in &args.token_ids {
            ensure!(
                token_id < arch.vocab_size && token_id <= i32::MAX as u32,
                "--token-ids entry {token_id} is outside vocabulary {}",
                arch.vocab_size
            );
            token_ids.push(token_id as i32);
        }
        ("token_ids", None, token_ids)
    };
    ensure!(
        !token_ids.is_empty(),
        "full-lens input tokenized to no tokens"
    );
    ensure!(
        token_ids.len() <= args.max_tokens,
        "full-lens input has {} tokens, exceeding --max-tokens {}",
        token_ids.len(),
        args.max_tokens
    );
    let selected_position = args.position.unwrap_or(token_ids.len() - 1);
    ensure!(
        selected_position < token_ids.len(),
        "--position {selected_position} is outside {} input tokens",
        token_ids.len()
    );
    let prefix = &token_ids[..=selected_position];
    let mut sequence = loaded
        .create_sequence(SequenceConfig::new(prefix.len()))
        .context("create full-lens prompt sequence")?;
    let mut research = loaded
        .research_session(&mut sequence)
        .context("open full-lens research session")?;
    let capture = research
        .forward_prompt_last_post_block_residuals(prefix, &layers)
        .context("capture full-lens source residuals")?;
    ensure!(
        capture.position == selected_position
            && capture.token_id == token_ids[selected_position]
            && capture.capture.layer_ids == layers
            && capture.capture.hidden_size == arch.hidden_size as usize,
        "full-lens prompt capture metadata is inconsistent"
    );

    let payload_path = args.full_lens.join(&manifest.payload.path);
    let (mut payload_file, payload_length) = open_regular_file(&payload_path)?;
    ensure!(
        payload_length as u64 == manifest.payload.byte_length,
        "full lens payload length does not match its manifest"
    );
    let hidden_size = arch.hidden_size as usize;
    let matrix_len = usize::try_from(transport_matrix_bytes(&manifest)?)
        .context("full-lens matrix byte count")?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(matrix_len)
        .context("allocate full-lens transport matrix")?;
    matrix.resize(matrix_len, 0);
    let selected_slots: BTreeMap<u32, usize> = layers
        .iter()
        .copied()
        .enumerate()
        .map(|(slot, layer)| (layer, slot))
        .collect();
    let mut indexed_results = Vec::new();
    indexed_results
        .try_reserve_exact(layers.len())
        .context("allocate full-lens layer results")?;
    let mut payload_hasher = Blake3Hasher::new();
    for &layer in &manifest.transport.source_layers {
        payload_file
            .read_exact(&mut matrix)
            .with_context(|| format!("read full-lens source layer {layer}"))?;
        payload_hasher.update(&matrix);
        ensure_finite_f16(&matrix, layer as usize, 0)?;
        let Some(&capture_slot) = selected_slots.get(&layer) else {
            continue;
        };
        let residual_start = capture_slot
            .checked_mul(hidden_size)
            .context("full-lens capture offset overflow")?;
        let residual_end = residual_start
            .checked_add(hidden_size)
            .context("full-lens capture endpoint overflow")?;
        let residual = capture
            .capture
            .values
            .get(residual_start..residual_end)
            .context("full-lens capture payload is too short")?;
        eprintln!(
            "read full lens source_layer={} position={} top_k={}",
            layer, selected_position, args.top_k
        );
        let readout = research
            .apply_f16_transport_topk_with_vector(&matrix, residual, args.top_k)
            .with_context(|| format!("apply full-lens source layer {layer}"))?;
        let mut top_k = Vec::new();
        top_k
            .try_reserve_exact(readout.readout.scores.len())
            .context("allocate decoded full-lens top-k")?;
        for (rank, score) in readout.readout.scores.into_iter().enumerate() {
            let token_id = i32::try_from(score.token_id).context("decode full-lens token ID")?;
            let piece = tokenizer
                .try_decode_piece_bytes_exact(token_id)
                .with_context(|| format!("decode full-lens token {}", score.token_id))?;
            top_k.push(FullTokenScore {
                rank,
                token_id: score.token_id,
                token_display_lossy: String::from_utf8_lossy(piece).into_owned(),
                token_piece_hex: hex(piece),
                logit: score.logit,
            });
        }
        indexed_results.push((
            capture_slot,
            FullLayerReadout {
                source_layer: layer,
                source_position: selected_position,
                source_token_id: capture.token_id,
                predicts_position: selected_position + 1,
                rms_denominator_f64_recomputed: readout.readout.rms_denominator_f64_recomputed,
                transported_vector: args.include_vector.then_some(FullTransportedVector {
                    operation: "row_major_f16_transport_times_source_residual",
                    stage: "before_output_rmsnorm",
                    value_dtype: "f32",
                    hidden_coordinate: "target_post_block_residual",
                    hidden_size,
                    shape: [hidden_size],
                    values: readout.transported_values,
                }),
                top_k,
            },
        ));
    }
    let mut extra = [0u8; 1];
    ensure!(
        payload_file
            .read(&mut extra)
            .with_context(|| format!("check full lens payload end {}", payload_path.display()))?
            == 0,
        "full lens payload contains trailing bytes"
    );
    ensure!(
        payload_hasher.finalize().to_hex().as_str() == manifest.payload.blake3,
        "full lens payload BLAKE3 mismatch"
    );
    indexed_results.sort_unstable_by_key(|(slot, _)| *slot);
    ensure!(
        indexed_results.len() == layers.len()
            && indexed_results
                .iter()
                .enumerate()
                .all(|(expected, (slot, _))| expected == *slot),
        "full-lens readout did not produce every requested layer"
    );
    let results = indexed_results
        .into_iter()
        .map(|(_, result)| result)
        .collect();

    let document = FullReadoutDocument {
        schema: "llm.lens.readout",
        schema_version: 1,
        readout: "qwen_published_full_vocabulary",
        scoring: "deployed_output_rmsnorm_and_lm_head",
        score_semantics: "full_vocabulary_next_token_logits_no_softmax_v1",
        ranking_scope: "full_vocabulary",
        source_site: "post_block_residual",
        input: FullReadoutInput {
            source: input_source,
            add_special_tokens,
            token_ids,
            selected_position,
            captured_token_id: capture.token_id,
            predicts_position: selected_position + 1,
        },
        artifact: FullReadoutArtifact {
            manifest: manifest_path,
            manifest_canonical_json_blake3,
            payload_blake3: manifest.payload.blake3,
            method: manifest.transport.method,
            target_layer: manifest.transport.target_layer,
            orientation: manifest.transport.orientation,
            source_repository: manifest.source.repository,
            source_revision: manifest.source.revision,
            fitted_checkpoint: manifest.model.fitted_checkpoint,
            fitted_checkpoint_revision: manifest.model.fitted_checkpoint_revision,
            fit_n_prompts: manifest.fit.n_prompts,
            fit_max_sequence_length: manifest.fit.max_sequence_length,
            fit_skip_first: manifest.fit.skip_first,
        },
        deployed_model: FullReadoutModel {
            path: args.model,
            content_blake3: hex(&content.content_id),
            model_locator_id: format!("{:016x}", identity.model_locator_id),
            tokenizer_metadata_id: format!("{:016x}", identity.tokenizer_metadata_id),
            architecture_contract: "exact_qwen3_hybrid_dense_27b_v1",
            n_layers: arch.n_layer,
            hidden_size: arch.hidden_size,
            vocab_size: arch.vocab_size,
            full_attention_interval: arch.full_attention_interval,
            lm_head_dtype: format!("{:?}", loaded.metal_model().lm_head.dtype),
        },
        transfer: FullReadoutTransfer {
            validation_status: manifest.transfer.validation_status,
            override_policy: "explicit_allow_unvalidated_transfer",
        },
        reader: FullReadoutReader {
            build_commit: env!("QWEN_BUILD_COMMIT"),
            build_dirty: env!("QWEN_BUILD_DIRTY"),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE"),
            build_stamp_source: env!("QWEN_BUILD_STAMP_SOURCE"),
            build_stamp_error: env!("QWEN_BUILD_STAMP_ERROR"),
        },
        results,
    };
    let bytes = serialize_json_pretty_bounded(&document, "full lens readout")?;
    if let Some(output) = args.output {
        let output = resolve_output_file(&output)?;
        publish_immutable(&output, &bytes)?;
    }
    println!("{}", String::from_utf8(bytes).unwrap());
    Ok(())
}

pub(crate) fn trace_full(args: TraceFullArgs) -> Result<()> {
    validate_trace_full_args(&args)?;
    let output = args
        .output
        .as_deref()
        .map(resolve_output_file_path)
        .transpose()?;
    let stdout_format = effective_trace_stdout_format(args.format, output.is_some());
    let manifest_path = args.full_lens.join(FULL_MANIFEST_NAME);
    let manifest: FullLensManifest = read_json_file(&manifest_path)?;
    validate_trace_full_manifest(&manifest)?;
    let profile = profile_for_manifest(&manifest)?;
    let layers = if args.layers.is_empty() {
        manifest.transport.source_layers.clone()
    } else {
        args.layers.clone()
    };
    let mut unique_layers = BTreeSet::new();
    ensure!(
        !layers.is_empty()
            && layers.iter().all(|layer| unique_layers.insert(*layer)
                && manifest.transport.source_layers.contains(layer)),
        "--layers must be unique source layers present in the full lens"
    );

    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_model(&args.model)
        .with_context(|| format!("load model {}", args.model.display()))?;
    validate_deployed_model(&manifest, &loaded)?;
    let arch = loaded.arch();
    let runtime_identity = loaded.research_identity();
    let deployed_model = TraceFullModel {
        path: args.model.clone(),
        locator_scheme: RESEARCH_IDENTITY_SCHEME,
        locator_id: format!("{:016x}", runtime_identity.model_locator_id),
        content_authenticated: runtime_identity.content_authenticated,
        architecture: loaded.gguf().architecture(),
        name: loaded.gguf().get_str("general.name").map(str::to_owned),
        base_model_name: loaded
            .gguf()
            .get_str("general.base_model.0.name")
            .map(str::to_owned),
        n_layers: arch.n_layer,
        hidden_size: arch.hidden_size,
        vocab_size: arch.vocab_size,
    };
    let tokenizer_summary = TraceFullTokenizer {
        metadata_id: format!("{:016x}", runtime_identity.tokenizer_metadata_id),
        model: loaded
            .gguf()
            .get_str("tokenizer.ggml.model")
            .map(str::to_owned),
        pretokenizer: loaded
            .gguf()
            .get_str("tokenizer.ggml.pre")
            .map(str::to_owned),
    };
    let tokenizer = loaded.tokenizer().context("load tokenizer from GGUF")?;
    let (input_source, add_special_tokens, token_ids, rendering) =
        match (&args.prompt, &args.token_ids, &args.messages) {
            (Some(prompt), None, None) => {
                let token_ids = tokenizer
                    .encode(prompt, !args.no_special_tokens)
                    .context("tokenize trace-full prompt")?;
                (
                    "prompt",
                    Some(!args.no_special_tokens),
                    token_ids,
                    TraceFullRendering {
                        renderer: "tokenizer_text",
                        generation_mode: None,
                        spans: Vec::new(),
                    },
                )
            }
            (None, Some(token_ids), None) => {
                ensure!(!token_ids.is_empty(), "--token-ids must not be empty");
                ensure!(
                    token_ids
                        .iter()
                        .all(|&token_id| token_id >= 0 && (token_id as u32) < arch.vocab_size),
                    "--token-ids contains an ID outside vocabulary {}",
                    arch.vocab_size
                );
                (
                    "token_ids",
                    None,
                    token_ids.clone(),
                    TraceFullRendering {
                        renderer: "literal_token_ids",
                        generation_mode: None,
                        spans: Vec::new(),
                    },
                )
            }
            (None, None, Some(path)) => {
                let raw = std::fs::read_to_string(path)
                    .with_context(|| format!("read messages {}", path.display()))?;
                let messages = parse_strict_messages_input(&raw, &path.display().to_string())?;
                let resolved_mode = resolve_trace_full_message_mode(profile.id, args.message_mode)?;
                let (rendered, renderer) = match resolved_mode {
                    ResolvedTraceFullMessageMode::Qwen38(mode) => (
                        render_qwen38_messages_prompt_with_generation_annotated(
                            &messages, true, mode,
                        ),
                        "qwen3.8_messages_v1",
                    ),
                    ResolvedTraceFullMessageMode::Qwen36(mode) => (
                        render_qwen_messages_prompt_with_generation_annotated(
                            &messages, false, true, mode,
                        ),
                        "qwen3.6_messages_v1",
                    ),
                };
                let token_ids = tokenizer
                    .encode(&rendered.text, false)
                    .context("tokenize exact rendered messages prompt")?;
                let spans = align_rendered_message_spans(&tokenizer, &rendered, &token_ids)?;
                (
                    "messages",
                    Some(false),
                    token_ids,
                    TraceFullRendering {
                        renderer,
                        generation_mode: Some(resolved_mode.artifact_name()),
                        spans,
                    },
                )
            }
            _ => anyhow::bail!("specify exactly one of --prompt, --token-ids, or --messages"),
        };
    ensure!(
        !token_ids.is_empty(),
        "trace-full input tokenized to no tokens"
    );
    ensure!(
        token_ids.len() <= args.max_tokens,
        "trace-full input has {} tokens, exceeding --max-tokens {}; input is not truncated",
        token_ids.len(),
        args.max_tokens
    );
    let vector_positions_by_layer =
        group_trace_full_vector_cells(&args.vectors, &layers, token_ids.len())?;
    let input_tokens = token_ids
        .iter()
        .enumerate()
        .map(|(position, &token_id)| {
            let piece = tokenizer
                .try_decode_piece_bytes_exact(token_id)
                .with_context(|| format!("decode trace-full input token {token_id}"))?;
            Ok(TraceFullInputToken {
                position,
                token_id,
                token_display_lossy: String::from_utf8_lossy(piece).into_owned(),
                token_piece_hex: hex(piece),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let payload_path = args.full_lens.join(&manifest.payload.path);
    let (mut payload_file, payload_length) = open_regular_file(&payload_path)?;
    ensure!(
        payload_length as u64 == manifest.payload.byte_length,
        "full lens payload length does not match its manifest"
    );
    let matrix_bytes = transport_matrix_bytes(&manifest)?;
    let matrix_len = usize::try_from(matrix_bytes).context("full-lens matrix byte count")?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(matrix_len)
        .context("allocate reusable full-lens transport matrix")?;
    matrix.resize(matrix_len, 0);

    let trace_started = Instant::now();
    let mut sequence = loaded
        .create_sequence(SequenceConfig::new(token_ids.len()))
        .context("create trace-full prompt sequence")?;
    let mut research = loaded
        .research_session(&mut sequence)
        .context("open trace-full research session")?;
    let capture = research
        .forward_packed_post_block_capture(&token_ids, &layers)
        .context("capture packed trace-full post-block residuals")?;
    ensure!(
        capture.start_position() == 0
            && capture.token_ids() == token_ids.as_slice()
            && capture.layer_ids() == layers.as_slice()
            && capture.hidden_size() == arch.hidden_size as usize,
        "packed trace-full capture metadata is inconsistent"
    );

    let mut cells = Vec::new();
    cells
        .try_reserve_exact(
            layers
                .len()
                .checked_mul(token_ids.len())
                .context("trace-full cell count overflow")?,
        )
        .context("allocate trace-full cells")?;
    let mut matrix_read_wall_ms = 0.0f64;
    let mut readout_gpu_ms = 0.0f64;
    let mut readout_wall_ms = 0.0f64;
    let mut transported_vectors = Vec::new();
    transported_vectors
        .try_reserve_exact(args.vectors.len())
        .context("allocate selected transported-vector results")?;
    for &layer in &layers {
        let layer_slot = manifest
            .transport
            .source_layers
            .iter()
            .position(|&candidate| candidate == layer)
            .with_context(|| format!("full lens has no payload slot for layer {layer}"))?;
        let matrix_offset = (layer_slot as u64)
            .checked_mul(matrix_bytes)
            .context("full-lens matrix offset overflow")?;
        let read_started = Instant::now();
        payload_file
            .seek(SeekFrom::Start(matrix_offset))
            .with_context(|| format!("seek full-lens source layer {layer}"))?;
        payload_file
            .read_exact(&mut matrix)
            .with_context(|| format!("read full-lens source layer {layer}"))?;
        matrix_read_wall_ms += read_started.elapsed().as_secs_f64() * 1e3;

        let vector_positions = vector_positions_by_layer
            .get(&layer)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let readout = research
            .apply_packed_capture_f16_transport_topk_with_vectors(
                &capture,
                layer,
                &matrix,
                args.top_k,
                vector_positions,
            )
            .with_context(|| format!("apply packed full-lens source layer {layer}"))?;
        ensure!(
            readout.source_layer == layer
                && readout.start_position == 0
                && readout.position_count == token_ids.len()
                && readout.top_k == args.top_k,
            "packed trace-full readout metadata is inconsistent for layer {layer}"
        );
        readout_gpu_ms += readout.readout_gpu_ms;
        readout_wall_ms += readout.readout_wall_ms;
        for vector in readout.transported_vectors {
            transported_vectors.push(TraceFullVector {
                source_layer: layer,
                source_position: vector.source_position,
                source_token_id: vector.source_token_id,
                predicts_position: vector.predicts_position,
                values: vector.values,
            });
        }
        for position in readout.positions {
            let mut top_k = Vec::new();
            top_k
                .try_reserve_exact(position.scores.len())
                .context("allocate decoded trace-full top-k")?;
            for (rank, score) in position.scores.into_iter().enumerate() {
                let token_id = i32::try_from(score.token_id)
                    .context("decode trace-full vocabulary token ID")?;
                let piece = tokenizer
                    .try_decode_piece_bytes_exact(token_id)
                    .with_context(|| format!("decode trace-full token {}", score.token_id))?;
                top_k.push(TraceFullTokenScore {
                    rank,
                    token_id: score.token_id,
                    token_display_lossy: String::from_utf8_lossy(piece).into_owned(),
                    token_piece_hex: hex(piece),
                    logit: score.logit,
                });
            }
            cells.push(TraceFullCell {
                source_layer: layer,
                source_position: position.source_position,
                source_token_id: position.source_token_id,
                predicts_position: position.predicts_position,
                top_k,
            });
        }
    }
    let occurrences = aggregate_trace_full_occurrences(&cells, &layers);
    let vectors = (!args.vectors.is_empty()).then(|| TraceFullVectors {
        operation: "row_major_f16_transport_times_f32_post_block_residual",
        stage: "transported_target_coordinate_before_output_rmsnorm_and_lm_head",
        value_dtype: "f32",
        hidden_coordinate: "zero_based_target_layer_residual_coordinate",
        hidden_size: arch.hidden_size as usize,
        shape: [transported_vectors.len(), arch.hidden_size as usize],
        cell_order: "selected_layers_order_then_source_position_ascending",
        cells: transported_vectors,
    });
    let document = TraceFullDocument {
        schema: "qwen.lens.trace",
        schema_version: 3,
        producer: TraceFullProducer {
            build_commit: env!("QWEN_BUILD_COMMIT"),
            build_dirty: env!("QWEN_BUILD_DIRTY"),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE"),
        },
        deployed_model,
        tokenizer: tokenizer_summary,
        lens: TraceFullLens {
            kind: "published_full_transport",
            method: manifest.transport.method,
            target_layer: manifest.transport.target_layer,
            source_site: manifest.transport.capture_site,
            source_repository: manifest.source.repository,
            source_revision: manifest.source.revision,
            source_filename: manifest.source.filename,
            payload_blake3: manifest.payload.blake3,
            scoring: "deployed_output_rmsnorm_and_lm_head_full_vocabulary_logits_no_softmax",
        },
        score_semantics: TraceFullScoreSemantics {
            kind: "logit",
            normalization: "deployed_output_rmsnorm",
            candidate_universe: "full_model_vocabulary",
            softmax_applied: false,
        },
        execution_mode: "passive_packed_prefill_no_interventions",
        input_source,
        add_special_tokens,
        input_token_ids: token_ids,
        input_tokens,
        rendering,
        coordinates: TraceFullCoordinates {
            source_layer: "zero_based_transformer_block_index_at_post_block_residual",
            source_position: "zero_based_input_token_position",
            predicts_position: "source_position_plus_one",
            rank: "zero_based_full_vocabulary_logit_rank",
        },
        selected_layers: layers,
        top_k: args.top_k,
        occurrence_definition: "one_token_id_appearing_in_one_returned_top_k_list",
        cells,
        vectors,
        timing: TraceFullTiming {
            packed_prefill_gpu_ms: capture.packed_prefill_gpu_ms(),
            packed_prefill_wall_ms: capture.packed_prefill_wall_ms(),
            matrix_read_wall_ms,
            readout_gpu_ms,
            readout_command_wall_ms: readout_wall_ms,
            trace_execution_wall_ms: trace_started.elapsed().as_secs_f64() * 1e3,
        },
        occurrences,
    };
    let artifact_bytes = if output.is_some() || stdout_format == TraceFullStdoutFormat::Json {
        let bytes = serde_json::to_vec(&document).context("serialize trace artifact")?;
        ensure!(
            bytes.len() <= MAX_TRACE_DOCUMENT_BYTES,
            "serialized trace artifact is {} bytes; limit is {MAX_TRACE_DOCUMENT_BYTES}",
            bytes.len()
        );
        Some(bytes)
    } else {
        None
    };
    if let (Some(path), Some(bytes)) = (&output, &artifact_bytes) {
        write_atomic_replace(path, bytes)?;
    }
    match stdout_format {
        TraceFullStdoutFormat::Summary => print_trace_full_summary(&document, output.as_deref()),
        TraceFullStdoutFormat::Json => {
            let stdout = std::io::stdout();
            let mut stdout = stdout.lock();
            stdout
                .write_all(
                    artifact_bytes
                        .as_deref()
                        .expect("JSON output was serialized"),
                )
                .context("write trace JSON")?;
            stdout.write_all(b"\n").context("finish trace JSON")?;
        }
    }
    Ok(())
}

fn effective_trace_stdout_format(
    explicit: Option<TraceFullStdoutFormat>,
    has_output: bool,
) -> TraceFullStdoutFormat {
    explicit.unwrap_or(if has_output {
        TraceFullStdoutFormat::Summary
    } else {
        TraceFullStdoutFormat::Json
    })
}

fn print_trace_full_summary(document: &TraceFullDocument, output: Option<&Path>) {
    println!(
        "{} {} | {} tokens x {} layers = {} cells | top-k {}",
        document.lens.method.to_uppercase(),
        document.lens.kind,
        document.input_token_ids.len(),
        document.selected_layers.len(),
        document.cells.len(),
        document.top_k,
    );
    println!(
        "score={} normalization={} candidates={} softmax={} | model={} | renderer={}",
        document.score_semantics.kind,
        document.score_semantics.normalization,
        document.score_semantics.candidate_universe,
        document.score_semantics.softmax_applied,
        document.deployed_model.name.as_deref().unwrap_or("unknown"),
        document.rendering.renderer,
    );
    println!(
        "trace {:.1} ms (prefill {:.1} ms GPU, readout {:.1} ms GPU)",
        document.timing.trace_execution_wall_ms,
        document.timing.packed_prefill_gpu_ms,
        document.timing.readout_gpu_ms,
    );
    if let Some(path) = output {
        println!("artifact {}", path.display());
    }
}

fn resolve_trace_full_message_mode(
    profile: PublishedProfileId,
    requested: Option<TraceFullMessageMode>,
) -> Result<ResolvedTraceFullMessageMode> {
    match profile {
        PublishedProfileId::Qwen36J | PublishedProfileId::Qwen36R => match requested {
            None | Some(TraceFullMessageMode::Auto) => Ok(ResolvedTraceFullMessageMode::Qwen36(
                QwenGenerationMode::Auto,
            )),
            Some(TraceFullMessageMode::Thinking) => Ok(ResolvedTraceFullMessageMode::Qwen36(
                QwenGenerationMode::Thinking,
            )),
            Some(TraceFullMessageMode::NoThinking) => Ok(ResolvedTraceFullMessageMode::Qwen36(
                QwenGenerationMode::NoThinking,
            )),
            Some(mode) => anyhow::bail!(
                "--message-mode {} is not supported by Qwen3.6 J/R; use auto, thinking, or no-thinking",
                mode.to_possible_value().unwrap().get_name()
            ),
        },
        PublishedProfileId::Qwen38J => match requested {
            None | Some(TraceFullMessageMode::Medium | TraceFullMessageMode::Thinking) => {
                Ok(ResolvedTraceFullMessageMode::Qwen38(
                    Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Medium),
                ))
            }
            Some(TraceFullMessageMode::Low) => Ok(ResolvedTraceFullMessageMode::Qwen38(
                Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Low),
            )),
            Some(TraceFullMessageMode::Xhigh) => Ok(ResolvedTraceFullMessageMode::Qwen38(
                Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Xhigh),
            )),
            Some(TraceFullMessageMode::NoThinking) => Ok(ResolvedTraceFullMessageMode::Qwen38(
                Qwen38GenerationMode::NoThinking,
            )),
            Some(TraceFullMessageMode::Auto) => anyhow::bail!(
                "--message-mode auto is not supported by Qwen3.8 J; use thinking, no-thinking, low, medium, or xhigh"
            ),
        },
    }
}

fn validate_trace_full_args(args: &TraceFullArgs) -> Result<()> {
    ensure!(
        (1..=MAX_FULL_READOUT_TOP_K).contains(&args.top_k),
        "--top-k must be in 1..={MAX_FULL_READOUT_TOP_K}"
    );
    ensure!(
        (1..=MAX_RESEARCH_PACKED_READOUT_POSITIONS).contains(&args.max_tokens),
        "--max-tokens must be in 1..={MAX_RESEARCH_PACKED_READOUT_POSITIONS}"
    );
    let input_count = usize::from(args.prompt.is_some())
        + usize::from(args.token_ids.is_some())
        + usize::from(args.messages.is_some());
    ensure!(
        input_count == 1,
        "specify exactly one of --prompt, --token-ids, or --messages"
    );
    ensure!(
        args.prompt.as_ref().is_none_or(|prompt| !prompt.is_empty()),
        "--prompt must not be empty"
    );
    ensure!(
        args.token_ids
            .as_ref()
            .is_none_or(|token_ids| !token_ids.is_empty() && token_ids.len() <= args.max_tokens),
        "--token-ids must be nonempty and contain at most --max-tokens entries"
    );
    ensure!(
        args.prompt.is_some() || !args.no_special_tokens,
        "--no-special-tokens only applies to --prompt"
    );
    ensure!(
        args.messages.is_some() || args.message_mode.is_none(),
        "--message-mode only applies to --messages"
    );
    ensure!(
        args.vectors.len() <= MAX_TRACE_FULL_VECTOR_CELLS,
        "--vectors selects {} cells, exceeding the limit {MAX_TRACE_FULL_VECTOR_CELLS}",
        args.vectors.len()
    );
    let mut vector_cells = BTreeSet::new();
    ensure!(
        args.vectors.iter().all(|cell| vector_cells.insert(*cell)),
        "--vectors cells must be unique"
    );
    Ok(())
}

fn group_trace_full_vector_cells(
    cells: &[TraceFullVectorCell],
    selected_layers: &[u32],
    position_count: usize,
) -> Result<BTreeMap<u32, Vec<usize>>> {
    let mut grouped = BTreeMap::<u32, Vec<usize>>::new();
    for cell in cells {
        ensure!(
            selected_layers.contains(&cell.source_layer),
            "--vectors layer {} is not present in the effective --layers selection",
            cell.source_layer
        );
        ensure!(
            cell.source_position < position_count,
            "--vectors position {} is outside the tokenized input length {position_count}",
            cell.source_position
        );
        grouped
            .entry(cell.source_layer)
            .or_default()
            .push(cell.source_position);
    }
    for positions in grouped.values_mut() {
        positions.sort_unstable();
    }
    Ok(grouped)
}

fn align_rendered_message_spans(
    tokenizer: &Tokenizer,
    rendered: &AnnotatedMessageRender,
    full_token_ids: &[i32],
) -> Result<Vec<TraceFullRenderedSpan>> {
    let token_pieces = full_token_ids
        .iter()
        .map(|&token_id| {
            tokenizer
                .try_decode_piece_bytes_exact(token_id)
                .with_context(|| format!("decode rendered token {token_id}"))
        })
        .collect::<Result<Vec<_>>>()?;
    map_rendered_message_spans(rendered, &token_pieces)
}

fn map_rendered_message_spans(
    rendered: &AnnotatedMessageRender,
    token_pieces: &[&[u8]],
) -> Result<Vec<TraceFullRenderedSpan>> {
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(rendered.text.len())
        .context("allocate rendered token byte validation")?;
    let mut token_boundaries = BTreeMap::from([(0usize, 0usize)]);
    for (token_index, piece) in token_pieces.iter().enumerate() {
        decoded.extend_from_slice(piece);
        token_boundaries.insert(decoded.len(), token_index + 1);
    }
    ensure!(
        decoded == rendered.text.as_bytes(),
        "decoded rendered tokens do not reproduce the renderer-authored bytes"
    );

    for span in &rendered.spans {
        ensure!(
            span.byte_start < span.byte_end
                && span.byte_end <= rendered.text.len()
                && rendered.text.is_char_boundary(span.byte_start)
                && rendered.text.is_char_boundary(span.byte_end),
            "renderer produced an invalid byte span {}..{} for {} bytes",
            span.byte_start,
            span.byte_end,
            rendered.text.len()
        );
    }

    rendered
        .spans
        .iter()
        .map(|span| {
            let token_range = token_boundaries
                .get(&span.byte_start)
                .zip(token_boundaries.get(&span.byte_end))
                .filter(|(start, end)| start < end)
                .map(|(&start, &end)| (start, end));
            if is_structural_render_span(span.kind) {
                ensure!(
                    token_range.is_some(),
                    "renderer structural {} span {}..{} is not a nonempty exact token range",
                    span.kind.as_str(),
                    span.byte_start,
                    span.byte_end
                );
            }
            Ok(TraceFullRenderedSpan {
                kind: span.kind.as_str(),
                message_index: span.message_index,
                role: span.role.clone(),
                channel: span.channel.map(|channel| channel.as_str()),
                byte_start: span.byte_start,
                byte_end: span.byte_end,
                token_start: token_range.map(|range| range.0),
                token_end: token_range.map(|range| range.1),
            })
        })
        .collect()
}

fn is_structural_render_span(kind: MessageRenderSpanKind) -> bool {
    matches!(
        kind,
        MessageRenderSpanKind::MessageStartMarker
            | MessageRenderSpanKind::MessageEndMarker
            | MessageRenderSpanKind::GeneratedAssistantStartMarker
            | MessageRenderSpanKind::ThinkingChannelStartMarker
            | MessageRenderSpanKind::ThinkingChannelEndMarker
    )
}

fn validate_trace_full_manifest(manifest: &FullLensManifest) -> Result<()> {
    ensure!(manifest.schema == FULL_SCHEMA, "unknown full lens schema");
    ensure!(
        manifest.schema_version == FULL_SCHEMA_VERSION,
        "unsupported full lens schema version"
    );
    ensure!(manifest.status == "complete", "full lens is not complete");
    let profile = profile_for_manifest(manifest)?;
    ensure!(
        manifest.transport == canonical_transport(profile),
        "full lens transport contract is not canonical"
    );
    ensure!(
        manifest.model == canonical_model(profile),
        "full lens model geometry is not canonical"
    );
    ensure!(
        manifest.payload.path == FULL_PAYLOAD_NAME
            && manifest.payload.dtype == "f16_le"
            && manifest.payload.shape == [SOURCE_LAYER_COUNT, HIDDEN_SIZE, HIDDEN_SIZE]
            && manifest.payload.byte_length == PAYLOAD_BYTES
            && manifest.payload.blake3 == profile.expected_payload_blake3,
        "full lens payload descriptor is not canonical"
    );
    Ok(())
}

fn model_metadata_matches_profile(gguf: &GgufFile, profile: PublishedProfile) -> bool {
    let named = [
        gguf.get_str("general.name"),
        gguf.get_str("general.base_model.0.name"),
    ]
    .into_iter()
    .flatten()
    .any(|value| {
        let value = value.to_ascii_lowercase();
        value.contains(profile.model_name_fragment) && value.contains("27b")
    });
    named
        && gguf.get_str("tokenizer.ggml.model") == Some("gpt2")
        && gguf.get_str("tokenizer.ggml.pre") == Some("qwen35")
}

fn validate_deployed_model(
    manifest: &FullLensManifest,
    loaded: &qwen_llm::runtime::LoadedModel,
) -> Result<()> {
    let profile = profile_for_manifest(manifest)?;
    let arch = loaded.arch();
    let mut expected = qwen_llm::model::QWEN3_27B;
    ensure!(
        arch.mtp_n_hidden_layers <= expected.mtp_n_hidden_layers,
        "deployed model has an unsupported MTP inventory"
    );
    expected.mtp_n_hidden_layers = arch.mtp_n_hidden_layers;
    ensure!(
        arch.n_layer == manifest.model.n_layers
            && arch.hidden_size == manifest.model.hidden_size
            && arch.vocab_size == manifest.model.vocab_size
            && arch == expected,
        "deployed model does not match the published full transport geometry"
    );
    ensure!(
        model_metadata_matches_profile(loaded.gguf(), profile),
        "deployed model metadata does not match published asset {}",
        profile.source_filename
    );
    Ok(())
}

fn aggregate_trace_full_occurrences(
    cells: &[TraceFullCell],
    selected_layers: &[u32],
) -> TraceFullOccurrences {
    let mut global = BTreeMap::<u32, OccurrenceAccumulator>::new();
    let mut per_layer = BTreeMap::<u32, BTreeMap<u32, OccurrenceAccumulator>>::new();
    for cell in cells {
        let mut cell_ranks = BTreeMap::<u32, usize>::new();
        for score in &cell.top_k {
            cell_ranks
                .entry(score.token_id)
                .and_modify(|rank| *rank = (*rank).min(score.rank))
                .or_insert(score.rank);
        }
        for (token_id, rank) in cell_ranks {
            update_occurrence(&mut global, token_id, rank);
            update_occurrence(
                per_layer.entry(cell.source_layer).or_default(),
                token_id,
                rank,
            );
        }
    }
    TraceFullOccurrences {
        global: sorted_occurrences(global),
        per_layer: selected_layers
            .iter()
            .map(|&source_layer| TraceFullLayerOccurrences {
                source_layer,
                tokens: sorted_occurrences(per_layer.remove(&source_layer).unwrap_or_default()),
            })
            .collect(),
    }
}

fn update_occurrence(
    occurrences: &mut BTreeMap<u32, OccurrenceAccumulator>,
    token_id: u32,
    rank: usize,
) {
    occurrences
        .entry(token_id)
        .and_modify(|occurrence| {
            occurrence.count += 1;
            occurrence.top1_count += usize::from(rank == 0);
            occurrence.best_rank = occurrence.best_rank.min(rank);
        })
        .or_insert(OccurrenceAccumulator {
            count: 1,
            top1_count: usize::from(rank == 0),
            best_rank: rank,
        });
}

fn sorted_occurrences(
    occurrences: BTreeMap<u32, OccurrenceAccumulator>,
) -> Vec<TraceFullOccurrence> {
    let mut occurrences = occurrences
        .into_iter()
        .map(|(token_id, occurrence)| TraceFullOccurrence {
            token_id,
            count: occurrence.count,
            top1_count: occurrence.top1_count,
            best_rank: occurrence.best_rank,
        })
        .collect::<Vec<_>>();
    occurrences.sort_unstable_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| right.top1_count.cmp(&left.top1_count))
            .then_with(|| left.best_rank.cmp(&right.best_rank))
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
    occurrences
}

fn validate_read_full_args(args: &ReadFullArgs) -> Result<()> {
    ensure!(
        args.allow_unvalidated_transfer,
        "published full-lens transfer is unvalidated; pass --allow-unvalidated-transfer to acknowledge this"
    );
    ensure!(
        (1..=MAX_FULL_READOUT_TOP_K).contains(&args.top_k),
        "--top-k must be in 1..={MAX_FULL_READOUT_TOP_K}"
    );
    ensure!(
        (1..=MAX_FULL_READOUT_PROMPT_TOKENS).contains(&args.max_tokens),
        "--max-tokens must be in 1..={MAX_FULL_READOUT_PROMPT_TOKENS}"
    );
    ensure!(
        args.prompt.is_some() ^ !args.token_ids.is_empty(),
        "specify exactly one of --prompt or nonempty --token-ids"
    );
    ensure!(
        args.prompt.as_ref().is_none_or(|prompt| !prompt.is_empty()),
        "--prompt must not be empty"
    );
    ensure!(
        args.prompt.is_some() || !args.no_special_tokens,
        "--no-special-tokens only applies to --prompt"
    );
    ensure!(
        args.position
            .is_none_or(|position| position < args.max_tokens),
        "--position must be below --max-tokens"
    );
    ensure!(
        args.token_ids.is_empty() || args.token_ids.len() <= args.max_tokens,
        "--token-ids count must not exceed --max-tokens"
    );
    ensure!(
        args.position
            .is_none_or(|position| args.token_ids.is_empty() || position < args.token_ids.len()),
        "--position is outside the literal --token-ids input"
    );
    Ok(())
}

fn load_native_j_readouts(directory: &Path) -> Result<(TokenReadoutManifest, Vec<f32>)> {
    let manifest: TokenReadoutManifest = read_json_file(&directory.join(TOKEN_MANIFEST_NAME))?;
    ensure!(
        manifest.schema == TOKEN_READOUT_SCHEMA && manifest.schema_version == SCHEMA_VERSION,
        "unknown native selected-token artifact schema"
    );
    ensure!(
        manifest.status == "complete",
        "native selected-token fit is incomplete"
    );
    ensure!(
        manifest.config.method == FitMethod::J,
        "native comparison artifact must be a J fit"
    );
    ensure!(
        manifest.config.orientation == TOKEN_ORIENTATION,
        "native selected-token orientation is unsupported"
    );
    ensure!(
        manifest.config.target_layer < manifest.config.n_layers,
        "native selected-token target layer is out of range"
    );
    ensure!(
        !manifest.config.source_layers.is_empty()
            && manifest
                .config
                .source_layers
                .windows(2)
                .all(|layers| layers[0] < layers[1])
            && manifest
                .config
                .source_layers
                .iter()
                .all(|&layer| layer < manifest.config.target_layer),
        "native selected-token source layers must be strictly increasing and below the target"
    );
    ensure!(
        !manifest.readouts.token_ids.is_empty()
            && manifest.readouts.token_ids.len() <= TOKEN_ID_ARGUMENT_MAX_COUNT,
        "native selected-token count is outside the supported range"
    );
    ensure!(
        manifest.config_blake3 == digest_json(&manifest.config)?,
        "native selected-token config digest mismatch"
    );
    validate_token_readout_spec(&manifest.config.readouts, &manifest.config)?;
    ensure!(
        manifest.readouts == manifest.config.readouts,
        "native selected-token readout metadata disagrees with its config"
    );
    let expected_shape = [
        manifest.config.source_layers.len(),
        manifest.readouts.token_ids.len(),
        manifest.config.hidden_size as usize,
    ];
    ensure!(
        manifest.payload.path == TOKEN_PAYLOAD_NAME
            && manifest.payload.dtype == "f32_le"
            && manifest.payload.shape == expected_shape,
        "native selected-token payload descriptor is not canonical"
    );
    let expected_bytes = expected_shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
        .and_then(|values| values.checked_mul(4))
        .context("native selected-token byte count overflow")?;
    ensure!(
        expected_bytes <= TOKEN_ARTIFACT_MAX_BYTES
            && manifest.payload.byte_length == expected_bytes as u64,
        "native selected-token payload length is inconsistent"
    );
    ensure!(
        manifest.model.content_blake3 == manifest.config.model_content_blake3
            && manifest.model.model_locator_id == manifest.config.model_locator_id
            && manifest.model.tokenizer_metadata_id == manifest.config.tokenizer_metadata_id
            && manifest.model.architecture == manifest.config.architecture
            && manifest.model.n_layers == manifest.config.n_layers
            && manifest.model.hidden_size == manifest.config.hidden_size
            && manifest.model.vocab_size == manifest.config.vocab_size
            && manifest.model.full_attention_interval == manifest.config.full_attention_interval
            && manifest.model.content_authenticated,
        "native selected-token model summary disagrees with its config"
    );
    ensure!(
        manifest.fit.method == FitMethod::J
            && manifest.fit.target_layer == manifest.config.target_layer
            && manifest.fit.source_layers == manifest.config.source_layers
            && manifest.fit.skip_first == manifest.config.skip_first
            && manifest.fit.orientation == manifest.config.orientation
            && manifest.fit.estimator_version == manifest.config.estimator_version
            && manifest.fit.rule_version == manifest.config.rule_version
            && manifest.fit.valid_position_denominator == "number_of_valid_source_positions"
            && manifest.fit.prompt_denominator == "number_of_used_prompts"
            && manifest.fit.accumulator_dtype == "f32"
            && manifest.fit.storage_dtype == "f32_le"
            && manifest.fit.forward_seconds.is_finite()
            && manifest.fit.forward_seconds >= 0.0
            && manifest.fit.vjp_seconds.is_finite()
            && manifest.fit.vjp_seconds >= 0.0,
        "native selected-token fit summary disagrees with its config"
    );
    let accounted_records = manifest
        .corpus
        .used_prompts
        .checked_add(
            u64::try_from(manifest.corpus.skipped_prompts.len())
                .context("native selected-token skipped prompt count")?,
        )
        .context("native selected-token accounted prompt count overflow")?;
    let selected_records = u64::try_from(manifest.corpus.selected_records)
        .context("native selected-token selected prompt count")?;
    ensure!(
        manifest.corpus.selected_records == manifest.config.selected_records
            && manifest.corpus.used_prompts > 0
            && manifest.corpus.ordered_token_ids_blake3 == manifest.config.corpus_blake3
            && manifest.corpus.add_special_tokens == manifest.config.add_special_tokens
            && manifest.corpus.max_tokens == manifest.config.max_tokens
            && accounted_records == selected_records
            && manifest.corpus.truncated_prompts <= manifest.corpus.used_prompts,
        "native selected-token corpus summary disagrees with its config"
    );
    ensure!(
        is_lower_hex(&manifest.provenance.build_commit, 40)
            && !manifest.provenance.build_dirty.is_empty()
            && manifest.provenance.build_source_state == manifest.config.build_source_state
            && !manifest.provenance.build_stamp_source.is_empty(),
        "native selected-token build provenance disagrees with its config"
    );
    validate_token_build_identity(
        &manifest.provenance.build_source_state,
        &manifest.provenance.build_stamp_error,
    )?;
    let payload_bytes = super::verify_payload(directory, &manifest.payload)?;
    let values = decode_f32_le(
        &payload_bytes,
        expected_shape[0] * expected_shape[1] * expected_shape[2],
    )?;
    Ok((manifest, values))
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

fn profile_for_source(byte_length: u64, sha256: &str) -> Option<PublishedProfile> {
    PUBLISHED_PROFILES
        .iter()
        .copied()
        .find(|profile| profile.source_bytes == byte_length && profile.source_sha256 == sha256)
}

fn profile_for_manifest(manifest: &FullLensManifest) -> Result<PublishedProfile> {
    PUBLISHED_PROFILES
        .iter()
        .copied()
        .find(|profile| {
            manifest.transport.method == profile.method
                && manifest.source.repository == profile.source_repository
                && manifest.source.revision == profile.source_revision
                && manifest.source.filename == profile.source_filename
                && manifest.source.byte_length == profile.source_bytes
                && manifest.source.sha256 == profile.source_sha256
                && manifest.source.data_pickle_sha256 == profile.data_pickle_sha256
        })
        .context("full lens manifest does not identify a supported pinned published asset")
}

fn published_manifest(profile: PublishedProfile, payload: FullPayload) -> FullLensManifest {
    FullLensManifest {
        schema: FULL_SCHEMA.into(),
        schema_version: FULL_SCHEMA_VERSION,
        status: "complete".into(),
        transport: canonical_transport(profile),
        model: canonical_model(profile),
        fit: canonical_fit(profile),
        source: canonical_source(profile),
        payload,
        transfer: canonical_transfer_policy(profile),
        provenance: ImportProvenance {
            build_commit: env!("QWEN_BUILD_COMMIT").into(),
            build_dirty: env!("QWEN_BUILD_DIRTY").into(),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
            build_stamp_source: env!("QWEN_BUILD_STAMP_SOURCE").into(),
            build_stamp_error: env!("QWEN_BUILD_STAMP_ERROR").into(),
            pickle_execution: "none_fixed_schema_pinned_zip_entries_only".into(),
        },
    }
}

fn canonical_transport(profile: PublishedProfile) -> FullTransport {
    FullTransport {
        method: profile.method.into(),
        target_layer: profile.target_layer,
        source_layers: (0..SOURCE_LAYER_COUNT as u32).collect(),
        capture_site: "post_block_residual".into(),
        orientation: ORIENTATION.into(),
        hidden_size: HIDDEN_SIZE as u32,
        bias: "none".into(),
        storage_dtype: "f16_le".into(),
    }
}

fn canonical_model(profile: PublishedProfile) -> FullModel {
    FullModel {
        base_model: profile.base_model.into(),
        fitted_checkpoint: profile.fitted_checkpoint.into(),
        fitted_checkpoint_revision: profile.fitted_checkpoint_revision.into(),
        architecture: "qwen3_hybrid_dense".into(),
        n_layers: N_LAYERS,
        hidden_size: HIDDEN_SIZE as u32,
        vocab_size: VOCAB_SIZE,
        output_norm: "final_rms_norm_epsilon_1e-6".into(),
        unembedding: "untied_bias_free_lm_head".into(),
    }
}

fn canonical_fit(profile: PublishedProfile) -> PublishedFit {
    match profile.id {
        PublishedProfileId::Qwen38J => PublishedFit {
            fitter: "neuronpedia_utils/jlens/fit_lens.py".into(),
            fitter_revision: "7724688596eb734a0662f911bf183151a5c66b2f".into(),
            dataset: "Salesforce/wikitext:wikitext-103-raw-v1".into(),
            split: "train".into(),
            n_prompts: 1_000,
            max_sequence_length: 128,
            skip_first: 16,
            valid_positions_per_prompt: Some(111),
            dim_batch: Some(8),
            model_execution_dtype: Some("bfloat16".into()),
            accumulator_dtype: Some("float32".into()),
            serialized_dtype: "float16".into(),
            docs_consumed: None,
            n_positions: None,
            config_json: None,
            weighting: None,
            corpus_mode: None,
        },
        PublishedProfileId::Qwen36J | PublishedProfileId::Qwen36R => PublishedFit {
            fitter: "jlens.fit".into(),
            fitter_revision: "modal".into(),
            dataset: "NeelNanda/pile-10k".into(),
            split: "not_recorded_in_published_artifact".into(),
            n_prompts: 25,
            max_sequence_length: 128,
            skip_first: 4,
            valid_positions_per_prompt: None,
            dim_batch: None,
            model_execution_dtype: Some("bfloat16".into()),
            accumulator_dtype: None,
            serialized_dtype: "float16".into(),
            docs_consumed: Some(25),
            n_positions: Some("0.0".into()),
            config_json: Some(match profile.id {
                PublishedProfileId::Qwen36J => r#"{"estimator": "standard"}"#.into(),
                PublishedProfileId::Qwen36R => r#"{"estimator": "relp", "rules": {"ln_rule": true, "identity_rule": true, "half_rule": true, "include_qk_norms": false}}"#.into(),
                PublishedProfileId::Qwen38J => unreachable!(),
            }),
            weighting: Some("uniform".into()),
            corpus_mode: Some("pretrain".into()),
        },
    }
}

fn canonical_source(profile: PublishedProfile) -> PublishedSource {
    PublishedSource {
        repository: profile.source_repository.into(),
        revision: profile.source_revision.into(),
        filename: profile.source_filename.into(),
        byte_length: profile.source_bytes,
        sha256: profile.source_sha256.into(),
        data_pickle_sha256: profile.data_pickle_sha256.into(),
        license: profile.license.into(),
    }
}

fn canonical_transfer_policy(profile: PublishedProfile) -> TransferPolicy {
    TransferPolicy {
        fitted_weight_precision: match profile.id {
            PublishedProfileId::Qwen38J => "bfloat16",
            PublishedProfileId::Qwen36J | PublishedProfileId::Qwen36R => {
                "bfloat16_model_float16_serialized_transport"
            }
        }
        .into(),
        deployed_checkpoint_policy: "geometry_preserving_transfer_requires_validation".into(),
        validation_status: "unvalidated".into(),
    }
}

fn validate_manifest(manifest: &FullLensManifest) -> Result<()> {
    ensure!(manifest.schema == FULL_SCHEMA, "unknown full lens schema");
    ensure!(
        manifest.schema_version == FULL_SCHEMA_VERSION,
        "unsupported full lens schema version"
    );
    ensure!(manifest.status == "complete", "full lens is not complete");
    let profile = profile_for_manifest(manifest)?;
    ensure!(
        manifest.transport == canonical_transport(profile),
        "full lens transport contract is not canonical"
    );
    ensure!(
        manifest.model == canonical_model(profile),
        "full lens model contract is not canonical"
    );
    ensure!(
        manifest.fit == canonical_fit(profile),
        "full lens fit contract is not canonical"
    );
    ensure!(
        manifest.source == canonical_source(profile),
        "full lens source provenance is not canonical"
    );
    ensure!(
        manifest.payload.path == FULL_PAYLOAD_NAME
            && manifest.payload.dtype == "f16_le"
            && manifest.payload.shape == [SOURCE_LAYER_COUNT, HIDDEN_SIZE, HIDDEN_SIZE]
            && manifest.payload.byte_length == PAYLOAD_BYTES
            && manifest.payload.blake3 == profile.expected_payload_blake3,
        "full lens payload descriptor is not canonical"
    );
    ensure!(
        manifest.transfer == canonical_transfer_policy(profile),
        "full lens transfer policy is not fail-closed"
    );
    validate_token_build_identity(
        &manifest.provenance.build_source_state,
        &manifest.provenance.build_stamp_error,
    )?;
    ensure!(
        !manifest.provenance.build_commit.is_empty()
            && matches!(manifest.provenance.build_dirty.as_str(), "0" | "1")
            && !manifest.provenance.build_stamp_source.is_empty()
            && manifest.provenance.pickle_execution == "none_fixed_schema_pinned_zip_entries_only",
        "full lens import provenance is incomplete or permits pickle execution"
    );
    Ok(())
}

fn extract_payload<R: Read + Seek, W: Write>(
    archive: &mut ZipArchive<R>,
    spec: ArchiveSpec<'_>,
    output: &mut W,
) -> Result<FullPayload> {
    let extracted = super::published_pt::extract_payload(archive, spec, output)?;
    Ok(FullPayload {
        path: FULL_PAYLOAD_NAME.into(),
        dtype: "f16_le".into(),
        shape: [spec.layer_count, spec.hidden_size, spec.hidden_size],
        byte_length: extracted.byte_length,
        blake3: extracted.blake3,
    })
}

fn prepare_output_directory(output: &Path) -> Result<()> {
    if output.exists() {
        let metadata = std::fs::symlink_metadata(output)
            .with_context(|| format!("inspect output {}", output.display()))?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "output {} must be a real directory",
            output.display()
        );
        return Ok(());
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    DirBuilder::new()
        .mode(0o700)
        .create(output)
        .with_context(|| format!("create output directory {}", output.display()))?;
    sync_directory(parent)
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

fn resolve_output_file(output: &Path) -> Result<PathBuf> {
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

fn staging_path(directory: &Path, name: &str) -> Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_nanos();
    Ok(directory.join(format!(".{name}.stage.{}.{}", std::process::id(), nonce)))
}

fn publish_streamed_payload(
    staging: &Path,
    destination: &Path,
    payload: &FullPayload,
) -> Result<()> {
    match std::fs::hard_link(staging, destination) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            verify_payload(
                destination.parent().unwrap_or_else(|| Path::new(".")),
                payload,
            )?;
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "publish staging payload {} to {}",
                    staging.display(),
                    destination.display()
                )
            });
        }
    }
    std::fs::remove_file(staging)
        .with_context(|| format!("remove staging payload {}", staging.display()))?;
    sync_directory(destination.parent().unwrap_or_else(|| Path::new(".")))
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

fn verify_payload(directory: &Path, payload: &FullPayload) -> Result<()> {
    ensure!(
        Path::new(&payload.path).components().count() == 1,
        "full lens payload path must be one relative filename"
    );
    let path = directory.join(&payload.path);
    let (mut file, length) = open_regular_file(&path)?;
    ensure!(
        length as u64 == payload.byte_length,
        "{} length {} != expected {}",
        path.display(),
        length,
        payload.byte_length
    );
    let mut hasher = Blake3Hasher::new();
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    let hidden_size = payload.shape[1];
    ensure!(
        payload.dtype == "f16_le" && hidden_size > 0 && payload.shape[2] == hidden_size,
        "full lens payload shape is invalid"
    );
    let matrix_bytes = (hidden_size as u64)
        .checked_mul(hidden_size as u64)
        .and_then(|words| words.checked_mul(2))
        .context("full lens payload matrix byte count overflow")?;
    ensure!(
        matrix_bytes.checked_mul(payload.shape[0] as u64) == Some(payload.byte_length),
        "full lens payload matrix inventory does not match its byte length"
    );
    for layer in 0..payload.shape[0] {
        let mut remaining = matrix_bytes;
        while remaining > 0 {
            let matrix_byte_offset = matrix_bytes - remaining;
            let read = usize::try_from(remaining.min(buffer.len() as u64))
                .context("full lens verification chunk")?;
            file.read_exact(&mut buffer[..read])
                .with_context(|| format!("read full lens payload {}", path.display()))?;
            hasher.update(&buffer[..read]);
            ensure_finite_f16(&buffer[..read], layer, matrix_byte_offset as usize / 2)?;
            remaining -= read as u64;
        }
    }
    let mut extra = [0u8; 1];
    ensure!(
        file.read(&mut extra)
            .with_context(|| format!("check full lens payload end {}", path.display()))?
            == 0,
        "full lens payload contains trailing bytes"
    );
    ensure!(
        hasher.finalize().to_hex().as_str() == payload.blake3,
        "full lens payload BLAKE3 mismatch"
    );
    Ok(())
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::io::Cursor;
    use zip::CompressionMethod;
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    fn test_archive(matrix: &[u8], data_pickle: &[u8]) -> Vec<u8> {
        let mut output = Cursor::new(Vec::new());
        {
            let mut writer = ZipWriter::new(&mut output);
            let options =
                SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
            for (name, bytes) in [
                ("lens/data.pkl", data_pickle),
                ("lens/.format_version", b"1" as &[u8]),
                ("lens/.storage_alignment", b"64" as &[u8]),
                ("lens/byteorder", b"little" as &[u8]),
                ("lens/version", b"3" as &[u8]),
                ("lens/.data/serialization_id", b"fixture" as &[u8]),
                ("lens/data/0", matrix),
            ] {
                writer.start_file(name, options).unwrap();
                writer.write_all(bytes).unwrap();
            }
            writer.finish().unwrap();
        }
        output.into_inner()
    }

    fn canonical_payload() -> FullPayload {
        FullPayload {
            path: FULL_PAYLOAD_NAME.into(),
            dtype: "f16_le".into(),
            shape: [SOURCE_LAYER_COUNT, HIDDEN_SIZE, HIDDEN_SIZE],
            byte_length: PAYLOAD_BYTES,
            blake3: PUBLISHED_PROFILES[0].expected_payload_blake3.into(),
        }
    }

    fn test_read_full_args() -> ReadFullArgs {
        ReadFullArgs {
            model: "model.gguf".into(),
            full_lens: "full-lens".into(),
            prompt: Some("hello".into()),
            token_ids: Vec::new(),
            no_special_tokens: false,
            position: None,
            layers: vec![0, 31, 62],
            top_k: 10,
            max_tokens: 256,
            identity_cache: "identity-cache".into(),
            allow_unvalidated_transfer: true,
            include_vector: false,
            output: None,
        }
    }

    fn test_trace_full_args() -> TraceFullArgs {
        TraceFullArgs {
            model: "model.gguf".into(),
            full_lens: "full-lens".into(),
            prompt: Some("hello".into()),
            token_ids: None,
            messages: None,
            message_mode: None,
            no_special_tokens: false,
            layers: vec![0, 31, 62],
            top_k: 8,
            max_tokens: MAX_RESEARCH_PACKED_READOUT_POSITIONS,
            vectors: Vec::new(),
            output: None,
            format: None,
        }
    }

    #[test]
    fn trace_stdout_format_defaults_follow_output_presence() {
        assert_eq!(
            effective_trace_stdout_format(None, false),
            TraceFullStdoutFormat::Json
        );
        assert_eq!(
            effective_trace_stdout_format(None, true),
            TraceFullStdoutFormat::Summary
        );
        assert_eq!(
            effective_trace_stdout_format(Some(TraceFullStdoutFormat::Summary), false),
            TraceFullStdoutFormat::Summary
        );
        assert_eq!(
            effective_trace_stdout_format(Some(TraceFullStdoutFormat::Json), true),
            TraceFullStdoutFormat::Json
        );
    }

    fn trace_score(rank: usize, token_id: u32) -> TraceFullTokenScore {
        TraceFullTokenScore {
            rank,
            token_id,
            token_display_lossy: format!("token-{token_id}"),
            token_piece_hex: format!("{token_id:02x}"),
            logit: -(rank as f32),
        }
    }

    #[test]
    fn rendered_span_mapping_keeps_newline_leading_content_with_null_bpe_boundaries() {
        let rendered = render_qwen_messages_prompt_with_generation_annotated(
            &[crate::messages::ChatMessage {
                role: "user".into(),
                content: "\n\nhello".into(),
                ..Default::default()
            }],
            false,
            true,
            QwenGenerationMode::Auto,
        );
        let mut owned_pieces = Vec::<Vec<u8>>::new();
        let mut span_index = 0;
        while span_index < rendered.spans.len() {
            let start = rendered.spans[span_index].byte_start;
            let mut end = rendered.spans[span_index].byte_end;
            if !is_structural_render_span(rendered.spans[span_index].kind) {
                while span_index + 1 < rendered.spans.len()
                    && !is_structural_render_span(rendered.spans[span_index + 1].kind)
                {
                    span_index += 1;
                    end = rendered.spans[span_index].byte_end;
                }
            }
            owned_pieces.push(rendered.text.as_bytes()[start..end].to_vec());
            span_index += 1;
        }
        let pieces = owned_pieces.iter().map(Vec::as_slice).collect::<Vec<_>>();

        let spans = map_rendered_message_spans(&rendered, &pieces).unwrap();
        let content = rendered
            .spans
            .iter()
            .position(|span| span.kind == MessageRenderSpanKind::MessageContent)
            .unwrap();
        assert_eq!(
            &rendered.text[rendered.spans[content].byte_start..rendered.spans[content].byte_end],
            "\n\nhello"
        );
        assert_eq!(
            (spans[content].token_start, spans[content].token_end),
            (None, None)
        );
        assert!(spans.iter().zip(&rendered.spans).all(|(mapped, authored)| {
            !is_structural_render_span(authored.kind)
                || matches!((mapped.token_start, mapped.token_end), (Some(start), Some(end)) if start < end)
        }));
    }

    #[test]
    fn extracts_pinned_inventory_without_executing_pickle() {
        let matrix = [0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c];
        let pickle = b"inert fixture";
        let bytes = test_archive(&matrix, pickle);
        let digest = hex(&Sha256::digest(pickle));
        let spec = ArchiveSpec {
            root: "lens",
            layer_count: 1,
            hidden_size: 2,
            matrix_bytes: matrix.len() as u64,
            data_pickle_sha256: &digest,
            serialization_id: None,
            identity_storage_index: Some(0),
        };
        let mut archive = ZipArchive::new(Cursor::new(bytes)).unwrap();
        validate_archive(&mut archive, spec).unwrap();
        let mut output = Vec::new();
        let payload = extract_payload(&mut archive, spec, &mut output).unwrap();
        assert_eq!(output, matrix);
        assert_eq!(payload.byte_length, matrix.len() as u64);
    }

    #[test]
    fn rejects_non_finite_half_storage() {
        let matrix = [0x00, 0x7c];
        let pickle = b"inert fixture";
        let bytes = test_archive(&matrix, pickle);
        let digest = hex(&Sha256::digest(pickle));
        let spec = ArchiveSpec {
            root: "lens",
            layer_count: 1,
            hidden_size: 1,
            matrix_bytes: matrix.len() as u64,
            data_pickle_sha256: &digest,
            serialization_id: None,
            identity_storage_index: None,
        };
        let mut archive = ZipArchive::new(Cursor::new(bytes)).unwrap();
        validate_archive(&mut archive, spec).unwrap();
        let error = extract_payload(&mut archive, spec, &mut Vec::new()).unwrap_err();
        assert!(error.to_string().contains("non-finite F16"));
    }

    #[test]
    fn full_manifest_binds_every_published_claim_and_payload_digest() {
        let profile = PUBLISHED_PROFILES[0];
        let manifest = published_manifest(profile, canonical_payload());
        validate_manifest(&manifest).unwrap();
        assert_eq!(manifest.fit.skip_first, 16);
        assert_eq!(manifest.fit.accumulator_dtype.as_deref(), Some("float32"));

        let mut wrong_fit = published_manifest(profile, canonical_payload());
        wrong_fit.fit.accumulator_dtype = Some("bfloat16".into());
        assert!(validate_manifest(&wrong_fit).is_err());

        let mut wrong_payload = published_manifest(profile, canonical_payload());
        wrong_payload.payload.blake3 = "00".repeat(32);
        assert!(validate_manifest(&wrong_payload).is_err());
    }

    #[test]
    fn released_qwen36_pair_has_matched_recipe_and_distinct_methods() {
        let j_profile = PUBLISHED_PROFILES[1];
        let r_profile = PUBLISHED_PROFILES[2];
        let j = published_manifest(
            j_profile,
            FullPayload {
                blake3: j_profile.expected_payload_blake3.into(),
                ..canonical_payload()
            },
        );
        let r = published_manifest(
            r_profile,
            FullPayload {
                blake3: r_profile.expected_payload_blake3.into(),
                ..canonical_payload()
            },
        );
        validate_manifest(&j).unwrap();
        validate_manifest(&r).unwrap();
        assert_eq!(j.transport.method, "j");
        assert_eq!(r.transport.method, "r");
        assert_eq!(j.transport.target_layer, 62);
        assert_eq!(j.fit.n_prompts, 25);
        assert_eq!(j.fit.max_sequence_length, 128);
        assert_eq!(j.fit.skip_first, 4);
        assert_eq!(j.fit.dataset, r.fit.dataset);
    }

    #[test]
    fn trace_manifest_validates_geometry_and_pinned_digest_without_rescanning_payload() {
        let mut manifest = published_manifest(PUBLISHED_PROFILES[0], canonical_payload());
        manifest.provenance.build_source_state = "not-used-by-trace-full".into();
        validate_trace_full_manifest(&manifest).unwrap();

        manifest.payload.blake3 = "00".repeat(32);
        assert!(validate_trace_full_manifest(&manifest).is_err());
        manifest.payload.blake3 = PUBLISHED_PROFILES[0].expected_payload_blake3.into();
        manifest.payload.byte_length -= 2;
        assert!(validate_trace_full_manifest(&manifest).is_err());
    }

    #[test]
    fn archive_validation_binds_pickle_digest() {
        let matrix = [0x00, 0x3c];
        let bytes = test_archive(&matrix, b"unexpected pickle");
        let digest = "00".repeat(32);
        let spec = ArchiveSpec {
            root: "lens",
            layer_count: 1,
            hidden_size: 1,
            matrix_bytes: matrix.len() as u64,
            data_pickle_sha256: &digest,
            serialization_id: None,
            identity_storage_index: None,
        };
        let mut archive = ZipArchive::new(Cursor::new(bytes)).unwrap();
        assert!(validate_archive(&mut archive, spec).is_err());
    }

    #[test]
    fn layer_aggregates_preserve_layer_heterogeneity() {
        let directions = vec![
            TransferDirection {
                source_layer: 0,
                token_id: 1,
                cosine_similarity: 0.25,
                relative_l2_error: 1.0,
                norm_ratio_public_over_native: 0.5,
                public_norm: 1.0,
                native_norm: 2.0,
                maximum_absolute_difference: 1.0,
            },
            TransferDirection {
                source_layer: 62,
                token_id: 1,
                cosine_similarity: 0.99,
                relative_l2_error: 0.1,
                norm_ratio_public_over_native: 1.0,
                public_norm: 2.0,
                native_norm: 2.0,
                maximum_absolute_difference: 0.1,
            },
        ];
        let layers = aggregate_layer_metrics(&directions).unwrap();
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].source_layer, 0);
        assert_eq!(layers[0].mean_cosine_similarity, 0.25);
        assert_eq!(layers[1].source_layer, 62);
        assert_eq!(layers[1].mean_cosine_similarity, 0.99);
    }

    #[test]
    fn full_readout_requires_explicit_safe_input_contract() {
        let mut args = test_read_full_args();
        validate_read_full_args(&args).unwrap();

        args.allow_unvalidated_transfer = false;
        assert!(
            validate_read_full_args(&args)
                .unwrap_err()
                .to_string()
                .contains("unvalidated")
        );
        args.allow_unvalidated_transfer = true;

        args.top_k = 17;
        assert!(validate_read_full_args(&args).is_err());
        args.top_k = 10;

        args.token_ids = vec![1];
        assert!(validate_read_full_args(&args).is_err());
        args.prompt = None;
        validate_read_full_args(&args).unwrap();

        args.no_special_tokens = true;
        assert!(validate_read_full_args(&args).is_err());
        args.no_special_tokens = false;
        args.token_ids.clear();
        assert!(validate_read_full_args(&args).is_err());

        args.prompt = Some(String::new());
        assert!(validate_read_full_args(&args).is_err());
        args.prompt = Some("hello".into());
        args.position = Some(args.max_tokens);
        assert!(validate_read_full_args(&args).is_err());
        args.prompt = None;
        args.token_ids = vec![1, 2];
        args.position = Some(2);
        assert!(validate_read_full_args(&args).is_err());
    }

    #[test]
    fn trace_full_arguments_enforce_the_bounded_exact_input_contract() {
        let mut args = test_trace_full_args();
        validate_trace_full_args(&args).unwrap();

        args.top_k = 17;
        assert!(validate_trace_full_args(&args).is_err());
        args.top_k = 8;
        args.max_tokens = MAX_RESEARCH_PACKED_READOUT_POSITIONS + 1;
        assert!(validate_trace_full_args(&args).is_err());
        args.max_tokens = MAX_RESEARCH_PACKED_READOUT_POSITIONS;

        args.prompt = None;
        args.messages = Some("messages.json".into());
        args.no_special_tokens = true;
        assert!(validate_trace_full_args(&args).is_err());
        args.no_special_tokens = false;
        validate_trace_full_args(&args).unwrap();

        args.messages = None;
        args.token_ids = Some(Vec::new());
        assert!(validate_trace_full_args(&args).is_err());
        args.token_ids = Some(vec![1, 2]);
        validate_trace_full_args(&args).unwrap();
        args.prompt = Some("also set".into());
        assert!(validate_trace_full_args(&args).is_err());
    }

    #[test]
    fn trace_message_modes_are_model_compatible_and_record_resolved_modes() {
        for profile in [PublishedProfileId::Qwen36J, PublishedProfileId::Qwen36R] {
            assert_eq!(
                resolve_trace_full_message_mode(profile, None)
                    .unwrap()
                    .artifact_name(),
                "auto"
            );
            assert_eq!(
                resolve_trace_full_message_mode(profile, Some(TraceFullMessageMode::Thinking))
                    .unwrap()
                    .artifact_name(),
                "thinking"
            );
            assert_eq!(
                resolve_trace_full_message_mode(profile, Some(TraceFullMessageMode::NoThinking))
                    .unwrap()
                    .artifact_name(),
                "no_thinking"
            );
            for invalid in [
                TraceFullMessageMode::Low,
                TraceFullMessageMode::Medium,
                TraceFullMessageMode::Xhigh,
            ] {
                assert!(resolve_trace_full_message_mode(profile, Some(invalid)).is_err());
            }
        }
        for (requested, expected) in [
            (None, "thinking_medium"),
            (Some(TraceFullMessageMode::Thinking), "thinking_medium"),
            (Some(TraceFullMessageMode::Medium), "thinking_medium"),
            (Some(TraceFullMessageMode::Low), "thinking_low"),
            (Some(TraceFullMessageMode::Xhigh), "thinking_xhigh"),
            (Some(TraceFullMessageMode::NoThinking), "no_thinking"),
        ] {
            assert_eq!(
                resolve_trace_full_message_mode(PublishedProfileId::Qwen38J, requested)
                    .unwrap()
                    .artifact_name(),
                expected
            );
        }
        assert!(
            resolve_trace_full_message_mode(
                PublishedProfileId::Qwen38J,
                Some(TraceFullMessageMode::Auto)
            )
            .is_err()
        );
    }

    #[test]
    fn trace_vector_cells_parse_strict_unsigned_coordinates() {
        assert_eq!(
            "31:127".parse::<TraceFullVectorCell>().unwrap(),
            TraceFullVectorCell {
                source_layer: 31,
                source_position: 127,
            }
        );
        for invalid in ["", "31", "31:", ":1", "31:1:2", "-1:2", "1:+2", "a:2"] {
            assert!(
                invalid.parse::<TraceFullVectorCell>().is_err(),
                "unexpectedly accepted {invalid:?}"
            );
        }
    }

    #[test]
    fn trace_vector_cells_require_selected_layers_and_valid_positions() {
        let cells = [
            TraceFullVectorCell {
                source_layer: 31,
                source_position: 4,
            },
            TraceFullVectorCell {
                source_layer: 0,
                source_position: 2,
            },
            TraceFullVectorCell {
                source_layer: 31,
                source_position: 1,
            },
        ];
        let grouped = group_trace_full_vector_cells(&cells, &[0, 31], 5).unwrap();
        assert_eq!(grouped[&0], [2]);
        assert_eq!(grouped[&31], [1, 4]);

        assert!(group_trace_full_vector_cells(&cells, &[0], 5).is_err());
        assert!(group_trace_full_vector_cells(&cells, &[0, 31], 4).is_err());
    }

    #[test]
    fn trace_vector_cells_are_bounded_and_unique() {
        let mut args = test_trace_full_args();
        args.vectors = vec![
            TraceFullVectorCell {
                source_layer: 31,
                source_position: 2,
            };
            2
        ];
        assert!(validate_trace_full_args(&args).is_err());

        args.vectors = (0..=MAX_TRACE_FULL_VECTOR_CELLS)
            .map(|source_position| TraceFullVectorCell {
                source_layer: 31,
                source_position,
            })
            .collect();
        assert!(validate_trace_full_args(&args).is_err());
    }

    #[test]
    fn trace_occurrences_count_each_token_once_per_cell_and_sort_deterministically() {
        let cells = vec![
            TraceFullCell {
                source_layer: 2,
                source_position: 0,
                source_token_id: 10,
                predicts_position: 1,
                top_k: vec![trace_score(0, 7), trace_score(1, 5)],
            },
            TraceFullCell {
                source_layer: 2,
                source_position: 1,
                source_token_id: 11,
                predicts_position: 2,
                top_k: vec![trace_score(0, 5), trace_score(1, 7), trace_score(2, 7)],
            },
            TraceFullCell {
                source_layer: 0,
                source_position: 0,
                source_token_id: 10,
                predicts_position: 1,
                top_k: vec![trace_score(0, 7), trace_score(1, 9)],
            },
        ];
        let occurrences = aggregate_trace_full_occurrences(&cells, &[2, 0]);
        assert_eq!(
            occurrences.global,
            [
                TraceFullOccurrence {
                    token_id: 7,
                    count: 3,
                    top1_count: 2,
                    best_rank: 0,
                },
                TraceFullOccurrence {
                    token_id: 5,
                    count: 2,
                    top1_count: 1,
                    best_rank: 0,
                },
                TraceFullOccurrence {
                    token_id: 9,
                    count: 1,
                    top1_count: 0,
                    best_rank: 1,
                },
            ]
        );
        assert_eq!(occurrences.per_layer[0].source_layer, 2);
        assert_eq!(
            occurrences.per_layer[0]
                .tokens
                .iter()
                .map(|occurrence| occurrence.token_id)
                .collect::<Vec<_>>(),
            [5, 7]
        );
        assert_eq!(occurrences.per_layer[1].source_layer, 0);
        assert_eq!(
            occurrences.per_layer[1]
                .tokens
                .iter()
                .map(|occurrence| occurrence.token_id)
                .collect::<Vec<_>>(),
            [7, 9]
        );
    }
}
