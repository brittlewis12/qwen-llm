use super::full_lens::{
    ReadFullArgs, TraceFullArgs, TraceFullStdoutFormat, ensure_trace_document_budget,
    trace_host_result_reserve_bytes, trace_position_tiles,
};
use super::lens_input::{LensInputRendering, prepare_muse_input, validate_lens_input_spec};
use super::muse_full_lens_artifact as artifact;
use super::muse_lens_artifact;
use super::muse_lens_rows_artifact as rows;
use super::muse_published_full_lens_artifact as published;
use anyhow::{Context, Result, ensure};
use blake3::Hasher;
use clap::Args;
use half::f16;
use qwen_llm::checkpoint_identity::{
    CheckpointIdentityCache, checkpoint_content_identity_without_weight_hashing,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::{
    MetalContext, MetalMemoryAdmission, evaluate_metal_memory_admission_with_cpu_bytes,
};
use qwen_llm::muse_glimmer::{ARCHITECTURE_NAME, MuseGlimmerModel};
use qwen_llm::muse_glimmer_runtime::MuseGlimmerLoadedModel;
use qwen_llm::muse_glimmer_text_session::{
    MUSE_GLIMMER_FULL_READOUT_MAX_ROWS, MUSE_GLIMMER_FULL_READOUT_MAX_TOP_K,
    MuseGlimmerFullReadoutWorkspacePlan,
};
use qwen_llm::tokenizer::LlamaCppTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{DirBuilder, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Args)]
pub(crate) struct AssembleMuseFullArgs {
    /// Directory whose immediate child directories are completed Muse row shards.
    #[arg(long)]
    shards_root: PathBuf,

    /// New assembled Muse full-transport directory, or an incomplete one with --resume.
    #[arg(long)]
    output: PathBuf,

    /// Reopen a complete artifact or restart an interrupted assembly.
    #[arg(long)]
    resume: bool,
}

struct ShardInput {
    directory: PathBuf,
    manifest: rows::Manifest,
}

#[derive(Debug, Deserialize)]
struct SchemaProbe {
    schema: String,
}

#[derive(Debug, Serialize)]
struct ReadoutDocument {
    schema: &'static str,
    schema_version: u32,
    readout: &'static str,
    score_semantics: &'static str,
    ranking_scope: &'static str,
    source_site: &'static str,
    input: ReadoutInput,
    artifact: ReadoutArtifact,
    deployed_model: ReadoutModel,
    reader: ReadoutReader,
    results: Vec<LayerReadout>,
}

#[derive(Debug, Serialize)]
struct PublishedReadoutDocument {
    schema: &'static str,
    schema_version: u32,
    readout: &'static str,
    score_semantics: &'static str,
    ranking_scope: &'static str,
    source_site: &'static str,
    input: ReadoutInput,
    artifact: PublishedReadoutArtifact,
    deployed_model: ReadoutModel,
    transfer: PublishedReadoutTransfer,
    reader: ReadoutReader,
    results: Vec<LayerReadout>,
}

#[derive(Debug, Serialize)]
struct ReadoutInput {
    source: &'static str,
    add_special_tokens: Option<bool>,
    token_ids: Vec<u32>,
    selected_position: usize,
    captured_token_id: u32,
    predicts_position: usize,
}

#[derive(Debug, Serialize)]
struct ReadoutArtifact {
    manifest: PathBuf,
    manifest_canonical_json_blake3: String,
    declared_payload_blake3: String,
    model_content_blake3: String,
    content_identity_policy: String,
    identity_input_outcomes: Vec<String>,
    artifact_profile: String,
    method: String,
    target_layer: u32,
    orientation: String,
    corpus_blake3: String,
    fit_used_prompts: u64,
    fit_max_tokens: usize,
    fit_skip_first: usize,
    query_batch_size: usize,
    storage_dtype: &'static str,
    conversion: String,
}

#[derive(Debug, Serialize)]
struct PublishedReadoutArtifact {
    binding: &'static str,
    manifest: PathBuf,
    manifest_canonical_json_blake3: String,
    profile: String,
    declared_payload_blake3: String,
    method: String,
    target_layer: u32,
    orientation: String,
    source_repository: String,
    source_revision: String,
    source_sha256: String,
    fitted_checkpoint: String,
    fitted_checkpoint_revision: String,
    claims_basis: String,
    fit_n_prompts: u64,
    fit_max_sequence_length: u32,
    fit_skip_first: u32,
    fit_modality: String,
    storage_dtype: String,
}

#[derive(Debug, Serialize)]
struct PublishedReadoutTransfer {
    validation_status: String,
    override_policy: &'static str,
    image_token_status: String,
}

#[derive(Debug, Serialize)]
struct ReadoutModel {
    path: PathBuf,
    content_blake3: String,
    content_identity_outcome: String,
    weight_bytes_hashed: u64,
    architecture: &'static str,
    artifact_profile: String,
    n_layers: u32,
    hidden_size: u32,
    vocab_size: u32,
    output_tail: &'static str,
}

#[derive(Debug, Serialize)]
struct ReadoutReader {
    build_commit: &'static str,
    build_dirty: &'static str,
    build_source_state: &'static str,
}

#[derive(Debug, Serialize)]
struct LayerReadout {
    source_layer: u32,
    source_position: usize,
    source_token_id: u32,
    predicts_position: usize,
    verified_matrix_blake3: String,
    rms_denominator_f64_recomputed: f32,
    matrix_read_wall_ms: f64,
    transport_wall_ms: f64,
    output_tail_wall_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    transported_vector: Option<TransportedVector>,
    top_k: Vec<TokenScore>,
}

#[derive(Debug, Serialize)]
struct TransportedVector {
    operation: &'static str,
    stage: &'static str,
    value_dtype: &'static str,
    hidden_coordinate: &'static str,
    hidden_size: usize,
    shape: [usize; 1],
    values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct TokenScore {
    rank: usize,
    token_id: u32,
    token_display_lossy: String,
    token_piece_hex: String,
    logit: f32,
}

enum ReadArtifact {
    Local {
        manifest: artifact::Manifest,
        canonical_json_blake3: String,
    },
    Published {
        manifest: published::Manifest,
        canonical_json_blake3: String,
    },
}

impl ReadArtifact {
    fn source_layers(&self) -> &[u32] {
        match self {
            Self::Local { manifest, .. } => &manifest.config.source_layers,
            Self::Published { manifest, .. } => &manifest.transport.source_layers,
        }
    }

    fn payload_path(&self) -> &str {
        match self {
            Self::Local { manifest, .. } => &manifest.payload.path,
            Self::Published { manifest, .. } => &manifest.payload.path,
        }
    }

    fn payload_byte_length(&self) -> u64 {
        match self {
            Self::Local { manifest, .. } => manifest.payload.byte_length,
            Self::Published { manifest, .. } => manifest.payload.byte_length,
        }
    }

    fn matrices(&self) -> &[artifact::MatrixDescriptor] {
        match self {
            Self::Local { manifest, .. } => &manifest.payload.matrices,
            Self::Published { manifest, .. } => &manifest.payload.matrices,
        }
    }
}

const MAX_MUSE_TRACE_VECTOR_CELLS: usize = 32;
const MAX_MUSE_TRACE_DOCUMENT_BYTES: usize = 256 * 1024 * 1024;

fn muse_trace_capture_bytes(
    layer_count: usize,
    token_count: usize,
    hidden_size: usize,
) -> Result<u64> {
    let elements = layer_count
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(hidden_size))
        .context("Muse trace capture size overflow")?;
    u64::try_from(
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .context("Muse trace capture byte count overflow")?,
    )
    .context("Muse trace capture byte count does not fit u64")
}

#[derive(Debug, Serialize)]
struct MuseTraceDocument {
    schema: &'static str,
    schema_version: u32,
    producer: MuseTraceProducer,
    deployed_model: MuseTraceModel,
    tokenizer: MuseTraceTokenizer,
    lens: MuseTraceLens,
    score_semantics: MuseTraceScoreSemantics,
    execution_mode: &'static str,
    input_source: &'static str,
    add_special_tokens: Option<bool>,
    input_token_ids: Vec<i32>,
    input_tokens: Vec<MuseTraceInputToken>,
    rendering: MuseTraceRendering,
    coordinates: MuseTraceCoordinates,
    selected_layers: Vec<u32>,
    top_k: usize,
    occurrence_definition: &'static str,
    cells: Vec<MuseTraceCell>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vectors: Option<MuseTraceVectors>,
    timing: BTreeMap<&'static str, f64>,
    occurrences: MuseTraceOccurrences,
}

#[derive(Debug, Serialize)]
struct MuseTraceProducer {
    build_commit: &'static str,
    build_dirty: &'static str,
    build_source_state: &'static str,
}

#[derive(Debug, Serialize)]
struct MuseTraceModel {
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
struct MuseTraceTokenizer {
    metadata_id: String,
    model: Option<String>,
    pretokenizer: Option<String>,
}

#[derive(Debug, Serialize)]
struct MuseTraceLens {
    kind: &'static str,
    method: String,
    target_layer: u32,
    source_site: String,
    source_repository: String,
    source_revision: String,
    source_filename: String,
    payload_blake3: String,
    fitted_checkpoint: String,
    fitted_checkpoint_revision: String,
    orientation: String,
    transfer_validation_status: String,
    transfer_override_policy: &'static str,
    image_token_status: String,
    scoring: &'static str,
}

#[derive(Debug, Serialize)]
struct MuseTraceScoreSemantics {
    kind: &'static str,
    normalization: &'static str,
    candidate_universe: &'static str,
    softmax_applied: bool,
}

#[derive(Debug, Serialize)]
struct MuseTraceInputToken {
    position: usize,
    token_id: i32,
    token_display_lossy: String,
    token_piece_hex: String,
}

type MuseTraceRendering = LensInputRendering;

#[derive(Debug, Serialize)]
struct MuseTraceCoordinates {
    source_layer: &'static str,
    source_position: &'static str,
    predicts_position: &'static str,
    rank: &'static str,
}

#[derive(Debug, Serialize)]
struct MuseTraceCell {
    source_layer: u32,
    source_position: usize,
    source_token_id: i32,
    predicts_position: usize,
    top_k: Vec<MuseTraceTokenScore>,
}

#[derive(Debug, Serialize)]
struct MuseTraceTokenScore {
    rank: usize,
    token_id: u32,
    token_display_lossy: String,
    token_piece_hex: String,
    logit: f32,
}

#[derive(Debug, Serialize)]
struct MuseTraceVectors {
    operation: &'static str,
    stage: &'static str,
    value_dtype: &'static str,
    hidden_coordinate: &'static str,
    hidden_size: usize,
    shape: [usize; 2],
    cell_order: &'static str,
    cells: Vec<MuseTraceVector>,
}

#[derive(Debug, Serialize)]
struct MuseTraceVector {
    source_layer: u32,
    source_position: usize,
    source_token_id: i32,
    predicts_position: usize,
    values: Vec<f32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MuseTraceOccurrence {
    token_id: u32,
    count: usize,
    top1_count: usize,
    best_rank: usize,
}

#[derive(Debug, Serialize)]
struct MuseTraceLayerOccurrences {
    source_layer: u32,
    tokens: Vec<MuseTraceOccurrence>,
}

#[derive(Debug, Serialize)]
struct MuseTraceOccurrences {
    global: Vec<MuseTraceOccurrence>,
    per_layer: Vec<MuseTraceLayerOccurrences>,
}

#[derive(Clone, Copy)]
struct MuseOccurrenceAccumulator {
    count: usize,
    top1_count: usize,
    best_rank: usize,
}

pub(crate) fn is_artifact(directory: &Path) -> Result<bool> {
    let path = directory.join(artifact::MANIFEST_NAME);
    if !path.exists() {
        return Ok(false);
    }
    let probe: SchemaProbe = super::read_json_file(&path)?;
    Ok(matches!(
        probe.schema.as_str(),
        artifact::SCHEMA | published::SCHEMA
    ))
}

pub(crate) fn read_full(args: ReadFullArgs) -> Result<()> {
    super::validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    validate_read_args(&args)?;
    let full_lens = canonical_real_directory(&args.full_lens, "Muse full-transport artifact")?;
    let manifest_path = full_lens.join(artifact::MANIFEST_NAME);
    let probe: SchemaProbe = super::read_json_file(&manifest_path)?;
    let read_artifact = match probe.schema.as_str() {
        artifact::SCHEMA => {
            let manifest: artifact::Manifest = super::read_json_file(&manifest_path)?;
            artifact::validate_manifest(&manifest)?;
            let canonical_json_blake3 = super::digest_json(&manifest)?;
            ReadArtifact::Local {
                manifest,
                canonical_json_blake3,
            }
        }
        published::SCHEMA => {
            ensure!(
                args.allow_unvalidated_transfer,
                "published Muse full transport requires --allow-unvalidated-transfer for BF16-to-GGUF use"
            );
            let manifest: published::Manifest = super::read_json_file(&manifest_path)?;
            published::validate_manifest(&manifest)?;
            let canonical_json_blake3 = super::digest_json(&manifest)?;
            ReadArtifact::Published {
                manifest,
                canonical_json_blake3,
            }
        }
        schema => anyhow::bail!("unsupported Muse full-transport schema {schema:?}"),
    };

    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open Muse model {}", args.model.display()))?;
    let bound =
        MuseGlimmerModel::from_gguf(&gguf).context("bind Muse model for full-transport readout")?;
    match &read_artifact {
        ReadArtifact::Local { manifest, .. } => ensure!(
            manifest.config.architecture == ARCHITECTURE_NAME
                && manifest.config.artifact_profile
                    == muse_lens_artifact::profile_name(bound.artifact_profile)
                && manifest.config.geometry == muse_lens_artifact::geometry(&bound.config),
            "Muse full transport does not match the deployed model profile or geometry"
        ),
        ReadArtifact::Published { manifest, .. } => ensure!(
            manifest.model.architecture == ARCHITECTURE_NAME
                && manifest.model.geometry == muse_lens_artifact::geometry(&bound.config),
            "Muse published full transport does not match the deployed release geometry"
        ),
    }
    let tokenizer = LlamaCppTokenizer::from_gguf(&gguf, &args.model)
        .context("load Muse tokenizer for full readout")?;
    muse_lens_artifact::validate_tokenizer(&tokenizer, &bound.config)?;
    let (input_source, add_special_tokens, token_ids) = prepare_read_input(&args, &tokenizer)?;
    let selected_position = args.position.unwrap_or(token_ids.len() - 1);
    ensure!(
        selected_position < token_ids.len(),
        "--position {selected_position} is outside {} input tokens",
        token_ids.len()
    );
    let prefix = &token_ids[..=selected_position];
    ensure!(
        prefix.len() <= bound.config.context_length as usize,
        "Muse full readout requires {} token forwards, exceeding model context {}",
        prefix.len(),
        bound.config.context_length,
    );
    let layers = select_layers(&args.layers, read_artifact.source_layers())?;
    let mut capture_layers = layers.clone();
    capture_layers.sort_unstable();
    let capture_slots = capture_layers
        .iter()
        .copied()
        .enumerate()
        .map(|(slot, layer)| (layer, slot))
        .collect::<BTreeMap<_, _>>();

    let content = checkpoint_content_identity_without_weight_hashing(
        &gguf,
        &CheckpointIdentityCache::new(&args.identity_cache),
    )
    .with_context(|| {
        format!(
            "resolve Muse model identity without hashing weights using {}",
            args.identity_cache.display()
        )
    })?;
    let content_id = super::hex(&content.content_id);
    ensure!(
        content.bytes_hashed == 0,
        "Muse full readout refuses model identities that hash weight bytes"
    );
    if let ReadArtifact::Local { manifest, .. } = &read_artifact {
        ensure!(
            content_id == manifest.config.model_content_blake3,
            "Muse full transport was fitted for a different GGUF content identity"
        );
    }

    let context = MetalContext::new().context("initialize Metal for Muse full readout")?;
    let mut loaded = MuseGlimmerLoadedModel::load(&context, &gguf, prefix.len())
        .context("load Muse model for full readout")?;
    let model_config = loaded.config().clone();
    let mut runner = loaded
        .create_runner(&context)
        .context("create Muse full-readout runner")?;
    for &token in &prefix[..prefix.len() - 1] {
        runner
            .forward_token(token)
            .context("forward Muse full-readout prefix")?;
    }
    let capture = runner
        .forward_token_capture_post_blocks(prefix[prefix.len() - 1], &capture_layers)
        .context("capture Muse full-readout source residuals")?;
    ensure!(
        capture.position == selected_position
            && capture.token_id == token_ids[selected_position]
            && capture.layer_ids == capture_layers
            && capture.hidden_size == model_config.hidden_size as usize,
        "Muse full-readout capture metadata is inconsistent"
    );

    let mut results = Vec::new();
    results
        .try_reserve_exact(layers.len())
        .context("allocate Muse full-readout layer results")?;
    for &layer in &layers {
        let descriptor = read_artifact
            .matrices()
            .iter()
            .find(|matrix| matrix.source_layer == layer)
            .context("Muse full transport omitted a selected source matrix")?;
        let started = Instant::now();
        let matrix = read_matrix(
            &full_lens,
            read_artifact.payload_path(),
            read_artifact.payload_byte_length(),
            descriptor,
        )?;
        let matrix_read_wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let capture_slot = *capture_slots
            .get(&layer)
            .context("Muse capture omitted a selected source layer")?;
        let source_residual = capture
            .layer_values(capture_slot)
            .context("Muse capture residual payload is too short")?;

        let started = Instant::now();
        let transported = runner
            .apply_f16_post_block_transport(&matrix, source_residual)
            .with_context(|| format!("apply Muse full transport at source layer {layer}"))?;
        let transport_wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let rms_denominator_f64_recomputed =
            rms_denominator(&transported, model_config.rms_epsilon);
        let started = Instant::now();
        let logits = runner
            .deployed_logits_from_post_block_residual(&transported)
            .with_context(|| format!("apply Muse deployed output tail at source layer {layer}"))?;
        let output_tail_wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let ranked = top_k_logits(&logits, args.top_k)?;
        let mut top_k = Vec::with_capacity(ranked.len());
        for (rank, (token_id, logit)) in ranked.into_iter().enumerate() {
            let piece = tokenizer
                .try_decode_piece_bytes_exact(token_id as i32)
                .with_context(|| format!("decode Muse readout token {token_id}"))?;
            top_k.push(TokenScore {
                rank,
                token_id,
                token_display_lossy: String::from_utf8_lossy(&piece).into_owned(),
                token_piece_hex: super::hex(&piece),
                logit,
            });
        }
        results.push(LayerReadout {
            source_layer: layer,
            source_position: selected_position,
            source_token_id: capture.token_id,
            predicts_position: selected_position + 1,
            verified_matrix_blake3: descriptor.blake3.clone(),
            rms_denominator_f64_recomputed,
            matrix_read_wall_ms,
            transport_wall_ms,
            output_tail_wall_ms,
            transported_vector: args.include_vector.then_some(TransportedVector {
                operation: "row_major_f16_transport_times_source_residual",
                stage: "before_output_rmsnorm",
                value_dtype: "f32",
                hidden_coordinate: "target_post_block_residual",
                hidden_size: model_config.hidden_size as usize,
                shape: [model_config.hidden_size as usize],
                values: transported,
            }),
            top_k,
        });
    }

    let input = ReadoutInput {
        source: input_source,
        add_special_tokens,
        token_ids,
        selected_position,
        captured_token_id: capture.token_id,
        predicts_position: selected_position + 1,
    };
    let content_identity_outcome = format!("{:?}", content.outcome);
    let runtime_profile = muse_lens_artifact::profile_name(bound.artifact_profile).to_owned();
    let bytes = match read_artifact {
        ReadArtifact::Local {
            manifest,
            canonical_json_blake3,
        } => {
            let document = ReadoutDocument {
                schema: "muse_glimmer.lens.full_readout",
                schema_version: 1,
                readout: "full_vocabulary",
                score_semantics: "deployed_output_rmsnorm_native_head_scale_softcap_no_softmax_v1",
                ranking_scope: "full_vocabulary",
                source_site: "post_block_residual",
                input,
                artifact: ReadoutArtifact {
                    manifest: manifest_path,
                    manifest_canonical_json_blake3: canonical_json_blake3,
                    declared_payload_blake3: manifest.payload.blake3.clone(),
                    model_content_blake3: manifest.config.model_content_blake3.clone(),
                    content_identity_policy: manifest.config.content_identity_policy.clone(),
                    identity_input_outcomes: manifest.identity.input_outcomes.clone(),
                    artifact_profile: manifest.config.artifact_profile.clone(),
                    method: manifest.config.method.clone(),
                    target_layer: manifest.config.target_layer,
                    orientation: manifest.config.orientation.clone(),
                    corpus_blake3: manifest.config.corpus_blake3.clone(),
                    fit_used_prompts: manifest.corpus.used_prompts,
                    fit_max_tokens: manifest.config.max_tokens,
                    fit_skip_first: manifest.config.skip_first,
                    query_batch_size: manifest.config.query_batch_size,
                    storage_dtype: "f16_le",
                    conversion: manifest.assembly.conversion.clone(),
                },
                deployed_model: ReadoutModel {
                    path: args.model.clone(),
                    content_blake3: content_id,
                    content_identity_outcome,
                    weight_bytes_hashed: content.bytes_hashed,
                    architecture: ARCHITECTURE_NAME,
                    artifact_profile: runtime_profile,
                    n_layers: model_config.layer_count,
                    hidden_size: model_config.hidden_size,
                    vocab_size: model_config.vocab_size,
                    output_tail: "rmsnorm_native_output_projection_logit_scale_final_softcap",
                },
                reader: ReadoutReader {
                    build_commit: env!("QWEN_BUILD_COMMIT"),
                    build_dirty: env!("QWEN_BUILD_DIRTY"),
                    build_source_state: env!("QWEN_BUILD_SOURCE_STATE"),
                },
                results,
            };
            super::serialize_json_pretty_bounded(&document, "Muse full readout")?
        }
        ReadArtifact::Published {
            manifest,
            canonical_json_blake3,
        } => {
            let document = PublishedReadoutDocument {
                schema: "muse_glimmer.lens.full_readout",
                schema_version: 2,
                readout: "published_full_vocabulary",
                score_semantics: "deployed_output_rmsnorm_native_head_scale_softcap_no_softmax_v1",
                ranking_scope: "full_vocabulary",
                source_site: "post_block_residual",
                input,
                artifact: PublishedReadoutArtifact {
                    binding: "published_checkpoint_geometry_transfer",
                    manifest: manifest_path,
                    manifest_canonical_json_blake3: canonical_json_blake3,
                    profile: manifest.profile,
                    declared_payload_blake3: manifest.payload.blake3,
                    method: manifest.transport.method,
                    target_layer: manifest.transport.target_layer,
                    orientation: manifest.transport.orientation,
                    source_repository: manifest.source.repository,
                    source_revision: manifest.source.revision,
                    source_sha256: manifest.source.sha256,
                    fitted_checkpoint: manifest.model.fitted_checkpoint,
                    fitted_checkpoint_revision: manifest.model.fitted_checkpoint_revision,
                    claims_basis: manifest.fit.claims_basis,
                    fit_n_prompts: manifest.fit.n_prompts,
                    fit_max_sequence_length: manifest.fit.max_sequence_length,
                    fit_skip_first: manifest.fit.skip_first,
                    fit_modality: manifest.fit.modality,
                    storage_dtype: manifest.payload.dtype,
                },
                deployed_model: ReadoutModel {
                    path: args.model.clone(),
                    content_blake3: content_id,
                    content_identity_outcome,
                    weight_bytes_hashed: content.bytes_hashed,
                    architecture: ARCHITECTURE_NAME,
                    artifact_profile: runtime_profile,
                    n_layers: model_config.layer_count,
                    hidden_size: model_config.hidden_size,
                    vocab_size: model_config.vocab_size,
                    output_tail: "rmsnorm_native_output_projection_logit_scale_final_softcap",
                },
                transfer: PublishedReadoutTransfer {
                    validation_status: manifest.transfer.validation_status,
                    override_policy: "explicit_allow_unvalidated_transfer",
                    image_token_status: manifest.transfer.image_token_status,
                },
                reader: ReadoutReader {
                    build_commit: env!("QWEN_BUILD_COMMIT"),
                    build_dirty: env!("QWEN_BUILD_DIRTY"),
                    build_source_state: env!("QWEN_BUILD_SOURCE_STATE"),
                },
                results,
            };
            super::serialize_json_pretty_bounded(&document, "Muse published full readout")?
        }
    };
    if let Some(output) = args.output {
        let output = super::resolve_output_path(&output)?;
        super::publish_immutable(&output, &bytes)?;
    }
    println!("{}", String::from_utf8(bytes).unwrap());
    Ok(())
}

pub(crate) fn trace_full(args: TraceFullArgs) -> Result<()> {
    super::validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    validate_trace_args(&args)?;
    let output = args
        .output
        .as_deref()
        .map(super::resolve_output_file_path)
        .transpose()?;
    let stdout_format = args.format.unwrap_or(if output.is_some() {
        TraceFullStdoutFormat::Summary
    } else {
        TraceFullStdoutFormat::Json
    });
    let trace_started = Instant::now();
    let full_lens =
        canonical_real_directory(&args.full_lens, "Muse published full-transport artifact")?;
    let manifest_path = full_lens.join(published::MANIFEST_NAME);
    let manifest: published::Manifest = super::read_json_file(&manifest_path)?;
    published::validate_manifest(&manifest)?;
    ensure!(
        args.allow_unvalidated_transfer,
        "published Muse trace requires --allow-unvalidated-transfer for BF16-to-GGUF use"
    );
    let layers = select_layers(&args.layers, &manifest.transport.source_layers)?;

    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open Muse model {}", args.model.display()))?;
    let bound = MuseGlimmerModel::from_gguf(&gguf).context("bind Muse model for full trace")?;
    ensure!(
        manifest.model.architecture == ARCHITECTURE_NAME
            && manifest.model.geometry == muse_lens_artifact::geometry(&bound.config),
        "Muse published trace artifact does not match deployed release geometry"
    );
    let tokenizer = LlamaCppTokenizer::from_gguf(&gguf, &args.model)
        .context("load Muse tokenizer for full trace")?;
    muse_lens_artifact::validate_tokenizer(&tokenizer, &bound.config)?;
    let prepared_input = prepare_muse_input(
        args.input_spec(),
        bound.config.chat_template_profile,
        &tokenizer,
        bound.config.vocab_size,
    )?;
    let input_source = prepared_input.source;
    let add_special_tokens = prepared_input.add_special_tokens;
    let token_ids = prepared_input.token_ids;
    let rendering = prepared_input.rendering;
    ensure!(!token_ids.is_empty(), "Muse trace input has no tokens");
    if let Some(max_tokens) = args.max_tokens {
        ensure!(
            token_ids.len() <= max_tokens,
            "Muse trace input has {} tokens, exceeding --max-tokens {max_tokens}",
            token_ids.len(),
        );
    }
    ensure!(
        token_ids.len() <= bound.config.context_length as usize,
        "Muse trace requires {} token forwards, exceeding model context {}",
        token_ids.len(),
        bound.config.context_length,
    );
    let vector_requests = validate_trace_vector_requests(&args, &layers, token_ids.len())?;
    let hidden_size = bound.config.hidden_size as usize;
    ensure_trace_document_budget(
        token_ids.len(),
        layers.len(),
        args.top_k,
        vector_requests.len(),
        hidden_size,
        MAX_MUSE_TRACE_DOCUMENT_BYTES,
        "Muse trace request",
    )?;
    let host_result_reserve_bytes =
        trace_host_result_reserve_bytes(1, MAX_MUSE_TRACE_DOCUMENT_BYTES)?;

    let identity_cache = args
        .identity_cache
        .as_ref()
        .context("Muse trace-full requires --identity-cache")?;
    let content = checkpoint_content_identity_without_weight_hashing(
        &gguf,
        &CheckpointIdentityCache::new(identity_cache),
    )
    .with_context(|| {
        format!(
            "resolve Muse model identity without hashing weights using {}",
            identity_cache.display()
        )
    })?;
    ensure!(
        content.bytes_hashed == 0,
        "Muse trace-full refuses model identities that hash weight bytes"
    );

    let mut capture_layers = layers.clone();
    capture_layers.sort_unstable();
    let capture_slots = capture_layers
        .iter()
        .copied()
        .enumerate()
        .map(|(slot, layer)| (layer, slot))
        .collect::<BTreeMap<_, _>>();
    let values_per_layer = token_ids
        .len()
        .checked_mul(hidden_size)
        .context("Muse trace capture size overflow")?;
    let capture_bytes = muse_trace_capture_bytes(layers.len(), token_ids.len(), hidden_size)?;

    let context = MetalContext::new().context("initialize Metal for Muse full trace")?;
    let model_load_started = Instant::now();
    let mut loaded = MuseGlimmerLoadedModel::load(&context, &gguf, token_ids.len())
        .context("load Muse model for full trace")?;
    let model_load_wall_ms = model_load_started.elapsed().as_secs_f64() * 1e3;
    let model_config = loaded.config().clone();
    let mut runner = loaded
        .create_runner(&context)
        .context("create Muse full-trace runner")?;
    let scalar_readout_requested =
        qwen_llm::env_flag::read_default_off("QWEN_MUSE_TRACE_FULL_SCALAR");
    let command_batch_supported = runner.supports_command_batched_full_readout();
    let workspace_rows = token_ids.len().min(MUSE_GLIMMER_FULL_READOUT_MAX_ROWS);
    let additional_host_bytes = capture_bytes
        .checked_add(host_result_reserve_bytes)
        .context("Muse trace host memory estimate overflow")?;
    let scalar_readout =
        use_scalar_muse_trace_readout(scalar_readout_requested, command_batch_supported);
    let (scalar_readout, admission_fallback, capture_admission) = if scalar_readout {
        let plan = runner
            .full_readout_workspace_plan(1)
            .context("price scalar Muse trace readout")?;
        (
            true,
            false,
            muse_trace_composite_admission(&context, &plan, additional_host_bytes)?,
        )
    } else {
        let plan = runner
            .full_readout_workspace_plan(workspace_rows)
            .context("price batched Muse trace readout")?;
        let batched = muse_trace_composite_admission(&context, &plan, additional_host_bytes)?;
        if batched.admitted {
            (false, false, batched)
        } else {
            let scalar_plan = runner
                .full_readout_workspace_plan(1)
                .context("price scalar Muse trace fallback")?;
            let scalar =
                muse_trace_composite_admission(&context, &scalar_plan, additional_host_bytes)?;
            if scalar.admitted {
                eprintln!(
                    "Muse trace batched readout memory admission denied (reason={}, required={:?}); using the independently admitted scalar readout",
                    batched.reason.as_str(),
                    batched.required_bytes,
                );
            }
            (true, true, scalar)
        }
    };
    ensure!(
        capture_admission.admitted,
        "Muse trace composite memory admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
        capture_admission.reason.as_str(),
        capture_admission.required_bytes,
        capture_admission.working_set_headroom_bytes,
        capture_admission.signals.process_limit_remaining_bytes,
    );
    let mut captured_by_layer = BTreeMap::<u32, Vec<f32>>::new();
    for &layer in &layers {
        let mut values = Vec::new();
        values
            .try_reserve_exact(values_per_layer)
            .context("allocate Muse trace captures")?;
        captured_by_layer.insert(layer, values);
    }
    let prefill_started = Instant::now();
    for (position, &token_id) in token_ids.iter().enumerate() {
        let capture = runner
            .forward_token_capture_post_blocks(token_id as u32, &capture_layers)
            .with_context(|| format!("capture Muse trace position {position}"))?;
        ensure!(
            capture.position == position
                && capture.token_id == token_id as u32
                && capture.layer_ids == capture_layers
                && capture.hidden_size == hidden_size,
            "Muse trace capture metadata is inconsistent at position {position}"
        );
        for &layer in &layers {
            let slot = capture_slots[&layer];
            captured_by_layer
                .get_mut(&layer)
                .expect("selected Muse trace layer has capture storage")
                .extend_from_slice(
                    capture
                        .layer_values(slot)
                        .context("Muse trace capture payload is too short")?,
                );
        }
    }
    let prefill_wall_ms = prefill_started.elapsed().as_secs_f64() * 1e3;
    ensure!(
        captured_by_layer
            .values()
            .all(|values| values.len() == values_per_layer),
        "Muse trace did not capture a complete layer-position grid"
    );

    let expected_cells = layers
        .len()
        .checked_mul(token_ids.len())
        .context("Muse trace cell count overflow")?;
    let mut cells = Vec::new();
    cells
        .try_reserve_exact(expected_cells)
        .context("allocate Muse trace cells")?;
    let mut vectors = Vec::new();
    vectors
        .try_reserve_exact(vector_requests.len())
        .context("allocate Muse trace vectors")?;
    let mut matrix_read_wall_ms = 0.0;
    let mut transport_prepare_wall_ms = 0.0;
    let mut scalar_transport_wall_ms = 0.0;
    let mut scalar_output_tail_wall_ms = 0.0;
    let mut batched_readout_gpu_ms = 0.0;
    let mut batched_readout_command_wall_ms = 0.0;
    let mut batched_transport_gpu_ms = 0.0;
    let mut batched_transport_wall_ms = 0.0;
    let mut batched_output_tail_gpu_ms = 0.0;
    let mut batched_output_tail_wall_ms = 0.0;
    let position_tiles = trace_position_tiles(token_ids.len(), MUSE_GLIMMER_FULL_READOUT_MAX_ROWS)?;
    let mut readout_workspace = if scalar_readout {
        None
    } else {
        Some(
            runner
                .create_full_readout_workspace(workspace_rows)
                .context("allocate pre-admitted Muse trace full-readout workspace")?,
        )
    };
    for &layer in &layers {
        let descriptor = manifest
            .payload
            .matrices
            .iter()
            .find(|matrix| matrix.source_layer == layer)
            .context("Muse published trace omitted a selected source matrix")?;
        let started = Instant::now();
        let matrix = read_matrix(
            &full_lens,
            &manifest.payload.path,
            manifest.payload.byte_length,
            descriptor,
        )?;
        matrix_read_wall_ms += started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let prepared = runner
            .prepare_f16_post_block_transport(&matrix)
            .with_context(|| format!("prepare Muse trace source layer {layer}"))?;
        transport_prepare_wall_ms += started.elapsed().as_secs_f64() * 1e3;
        let captures = &captured_by_layer[&layer];
        if let Some(workspace) = readout_workspace.as_mut() {
            for tile in &position_tiles {
                let tile_rows = tile.end - tile.start;
                let capture_start = tile
                    .start
                    .checked_mul(hidden_size)
                    .context("Muse trace tile capture offset overflow")?;
                let capture_end = tile
                    .end
                    .checked_mul(hidden_size)
                    .context("Muse trace tile capture endpoint overflow")?;
                let tile_captures = captures
                    .get(capture_start..capture_end)
                    .context("Muse trace tile capture is outside retained storage")?;
                let vector_positions = (tile.start..tile.end)
                    .filter(|&position| vector_requests.contains(&(layer, position)))
                    .map(|position| position - tile.start)
                    .collect::<Vec<_>>();
                let readout = runner
                    .apply_prepared_f16_transport_topk_rows(
                        workspace,
                        &prepared,
                        tile_captures,
                        args.top_k,
                        &vector_positions,
                    )
                    .with_context(|| {
                        format!(
                            "apply batched Muse trace source layer {layer} positions {}..{}",
                            tile.start, tile.end
                        )
                    })?;
                batched_readout_gpu_ms += readout.gpu_ms;
                batched_readout_command_wall_ms += readout.command_wall_ms;
                batched_transport_gpu_ms += readout.transport_gpu_ms;
                batched_transport_wall_ms += readout.transport_wall_ms;
                batched_output_tail_gpu_ms += readout.output_tail_gpu_ms;
                batched_output_tail_wall_ms += readout.output_tail_wall_ms;
                ensure!(
                    readout.row_count == tile_rows
                        && readout.top_k == args.top_k
                        && readout.rows.len() == tile_rows
                        && readout.transported_rows.len() == vector_positions.len(),
                    "batched Muse trace metadata is inconsistent for layer {layer} positions {}..{}",
                    tile.start,
                    tile.end,
                );
                for transported in readout.transported_rows {
                    ensure!(
                        transported.row < tile_rows,
                        "Muse transported row is outside its trace tile"
                    );
                    let position = tile
                        .start
                        .checked_add(transported.row)
                        .context("Muse transported trace position overflow")?;
                    vectors.push(MuseTraceVector {
                        source_layer: layer,
                        source_position: position,
                        source_token_id: token_ids[position],
                        predicts_position: position + 1,
                        values: transported.values,
                    });
                }
                for row in readout.rows {
                    ensure!(
                        row.row < tile_rows,
                        "Muse readout row is outside its trace tile"
                    );
                    let position = tile
                        .start
                        .checked_add(row.row)
                        .context("Muse readout trace position overflow")?;
                    let mut top_k = Vec::with_capacity(row.scores.len());
                    for (rank, score) in row.scores.into_iter().enumerate() {
                        let piece = tokenizer
                            .try_decode_piece_bytes_exact(score.token_id as i32)
                            .with_context(|| {
                                format!("decode Muse trace token {}", score.token_id)
                            })?;
                        top_k.push(MuseTraceTokenScore {
                            rank,
                            token_id: score.token_id,
                            token_display_lossy: String::from_utf8_lossy(&piece).into_owned(),
                            token_piece_hex: super::hex(&piece),
                            logit: score.logit,
                        });
                    }
                    cells.push(MuseTraceCell {
                        source_layer: layer,
                        source_position: position,
                        source_token_id: token_ids[position],
                        predicts_position: position + 1,
                        top_k,
                    });
                }
            }
            continue;
        }
        for (position, &source_token_id) in token_ids.iter().enumerate() {
            let start = position * hidden_size;
            let source_residual = &captures[start..start + hidden_size];
            let started = Instant::now();
            let transported = runner
                .apply_prepared_f16_post_block_transport(&prepared, source_residual)
                .with_context(|| format!("apply Muse trace layer {layer} position {position}"))?;
            scalar_transport_wall_ms += started.elapsed().as_secs_f64() * 1e3;
            let started = Instant::now();
            let logits = runner
                .deployed_logits_from_post_block_residual(&transported)
                .with_context(|| {
                    format!("apply Muse trace output tail at layer {layer} position {position}")
                })?;
            scalar_output_tail_wall_ms += started.elapsed().as_secs_f64() * 1e3;
            let ranked = top_k_logits(&logits, args.top_k)?;
            let mut top_k = Vec::with_capacity(ranked.len());
            for (rank, (token_id, logit)) in ranked.into_iter().enumerate() {
                let piece = tokenizer
                    .try_decode_piece_bytes_exact(token_id as i32)
                    .with_context(|| format!("decode Muse trace token {token_id}"))?;
                top_k.push(MuseTraceTokenScore {
                    rank,
                    token_id,
                    token_display_lossy: String::from_utf8_lossy(&piece).into_owned(),
                    token_piece_hex: super::hex(&piece),
                    logit,
                });
            }
            if vector_requests.contains(&(layer, position)) {
                vectors.push(MuseTraceVector {
                    source_layer: layer,
                    source_position: position,
                    source_token_id,
                    predicts_position: position + 1,
                    values: transported,
                });
            }
            cells.push(MuseTraceCell {
                source_layer: layer,
                source_position: position,
                source_token_id,
                predicts_position: position + 1,
                top_k,
            });
        }
    }
    ensure!(
        cells.len() == expected_cells && vectors.len() == vector_requests.len(),
        "Muse trace produced an incomplete result grid"
    );
    let occurrences = aggregate_muse_trace_occurrences(&cells, &layers);
    let mut input_tokens = Vec::new();
    input_tokens
        .try_reserve_exact(token_ids.len())
        .context("allocate Muse trace input-token records")?;
    for (position, &token_id) in token_ids.iter().enumerate() {
        let piece = tokenizer
            .try_decode_piece_bytes_exact(token_id)
            .with_context(|| format!("decode Muse input token {token_id}"))?;
        input_tokens.push(MuseTraceInputToken {
            position,
            token_id,
            token_display_lossy: String::from_utf8_lossy(&piece).into_owned(),
            token_piece_hex: super::hex(&piece),
        });
    }
    let mut timing = BTreeMap::new();
    timing.insert("model_load_wall_ms", model_load_wall_ms);
    timing.insert("scalar_prefill_capture_wall_ms", prefill_wall_ms);
    timing.insert("matrix_read_wall_ms", matrix_read_wall_ms);
    timing.insert("transport_prepare_wall_ms", transport_prepare_wall_ms);
    timing.insert(
        "transport_wall_ms",
        legacy_muse_trace_transport_wall_ms(
            transport_prepare_wall_ms,
            if scalar_readout {
                scalar_transport_wall_ms
            } else {
                batched_transport_wall_ms
            },
        ),
    );
    timing.insert(
        "output_tail_wall_ms",
        if scalar_readout {
            scalar_output_tail_wall_ms
        } else {
            batched_output_tail_wall_ms
        },
    );
    if scalar_readout {
        timing.insert("scalar_transport_wall_ms", scalar_transport_wall_ms);
        timing.insert("scalar_output_tail_wall_ms", scalar_output_tail_wall_ms);
    } else {
        timing.insert("batched_readout_gpu_ms", batched_readout_gpu_ms);
        timing.insert(
            "batched_readout_command_wall_ms",
            batched_readout_command_wall_ms,
        );
        timing.insert("batched_transport_gpu_ms", batched_transport_gpu_ms);
        timing.insert("batched_transport_wall_ms", batched_transport_wall_ms);
        timing.insert("batched_output_tail_gpu_ms", batched_output_tail_gpu_ms);
        timing.insert("batched_output_tail_wall_ms", batched_output_tail_wall_ms);
    }
    timing.insert(
        "trace_execution_wall_ms",
        trace_started.elapsed().as_secs_f64() * 1e3,
    );
    let model_name = args
        .model
        .file_name()
        .map(|name| name.to_string_lossy().into_owned());
    let document = MuseTraceDocument {
        schema: "qwen.lens.trace",
        schema_version: 3,
        producer: MuseTraceProducer {
            build_commit: env!("QWEN_BUILD_COMMIT"),
            build_dirty: env!("QWEN_BUILD_DIRTY"),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE"),
        },
        deployed_model: MuseTraceModel {
            path: args.model.clone(),
            locator_scheme: "ordered_gguf_declared_content_blake3_v1",
            locator_id: super::hex(&content.content_id),
            content_authenticated: true,
            architecture: Some(ARCHITECTURE_NAME.into()),
            name: model_name,
            base_model_name: None,
            n_layers: model_config.layer_count,
            hidden_size: model_config.hidden_size,
            vocab_size: model_config.vocab_size,
        },
        tokenizer: MuseTraceTokenizer {
            metadata_id: super::hex(&model_config.tokenizer_identity_sha256),
            model: Some(model_config.tokenizer_model.clone()),
            pretokenizer: Some(model_config.tokenizer_pre.clone()),
        },
        lens: MuseTraceLens {
            kind: "published_full_transport",
            method: manifest.transport.method.clone(),
            target_layer: manifest.transport.target_layer,
            source_site: manifest.transport.coordinate.clone(),
            source_repository: manifest.source.repository.clone(),
            source_revision: manifest.source.revision.clone(),
            source_filename: manifest.source.filename.clone(),
            payload_blake3: manifest.payload.blake3.clone(),
            fitted_checkpoint: manifest.model.fitted_checkpoint.clone(),
            fitted_checkpoint_revision: manifest.model.fitted_checkpoint_revision.clone(),
            orientation: manifest.transport.orientation.clone(),
            transfer_validation_status: manifest.transfer.validation_status.clone(),
            transfer_override_policy: "explicit_allow_unvalidated_transfer",
            image_token_status: manifest.transfer.image_token_status.clone(),
            scoring: "transport_then_deployed_output_tail",
        },
        score_semantics: MuseTraceScoreSemantics {
            kind: "logit",
            normalization: "deployed_output_rmsnorm_native_head_scale_softcap",
            candidate_universe: "full_model_vocabulary",
            softmax_applied: false,
        },
        execution_mode: if scalar_readout_requested {
            "passive_scalar_prefill_prepared_transport_no_interventions"
        } else if !command_batch_supported {
            "passive_scalar_prefill_automatic_scalar_readout_fallback_no_interventions"
        } else if admission_fallback {
            "passive_scalar_prefill_admission_scalar_readout_fallback_no_interventions"
        } else {
            "passive_scalar_prefill_position_batched_gpu_readout_no_interventions"
        },
        input_source,
        add_special_tokens,
        input_token_ids: token_ids,
        input_tokens,
        rendering,
        coordinates: MuseTraceCoordinates {
            source_layer: "zero_based_post_block_layer_id",
            source_position: "zero_based_tokenized_input_position",
            predicts_position: "source_position_plus_one",
            rank: "zero_based_descending_logit_with_token_id_tie_break",
        },
        selected_layers: layers,
        top_k: args.top_k,
        occurrence_definition: "one_token_id_appearing_in_one_returned_top_k_list",
        cells,
        vectors: (!vectors.is_empty()).then_some(MuseTraceVectors {
            operation: "row_major_f16_transport_times_source_residual",
            stage: "before_output_rmsnorm",
            value_dtype: "f32",
            hidden_coordinate: "target_post_block_residual",
            hidden_size,
            shape: [vectors.len(), hidden_size],
            cell_order: "selected_layer_order_then_source_position",
            cells: vectors,
        }),
        timing,
        occurrences,
    };
    let bytes = serde_json::to_vec(&document).context("serialize Muse trace artifact")?;
    ensure!(
        bytes.len() <= MAX_MUSE_TRACE_DOCUMENT_BYTES,
        "serialized Muse trace artifact is {} bytes; limit is {MAX_MUSE_TRACE_DOCUMENT_BYTES}",
        bytes.len()
    );
    super::lens_inspect::parse_trace_bytes(&bytes, Path::new("<generated Muse trace>"))
        .context("self-validate generated Muse trace artifact")?;
    if let Some(path) = &output {
        super::write_atomic_replace(path, &bytes)?;
    }
    match stdout_format {
        TraceFullStdoutFormat::Summary => print_muse_trace_summary(&document, output.as_deref()),
        TraceFullStdoutFormat::Json => {
            let stdout = std::io::stdout();
            let mut stdout = stdout.lock();
            stdout.write_all(&bytes).context("write Muse trace JSON")?;
            stdout.write_all(b"\n").context("finish Muse trace JSON")?;
        }
    }
    Ok(())
}

fn validate_trace_args(args: &TraceFullArgs) -> Result<()> {
    validate_lens_input_spec(args.input_spec())?;
    ensure!(
        args.prompt.as_ref().is_none_or(|prompt| !prompt.is_empty()),
        "--prompt must not be empty"
    );
    validate_muse_top_k(args.top_k)?;
    ensure!(
        args.max_tokens.is_none_or(|max_tokens| max_tokens > 0),
        "Muse --max-tokens must be positive"
    );
    ensure!(
        args.vectors.len() <= MAX_MUSE_TRACE_VECTOR_CELLS,
        "Muse trace supports at most {MAX_MUSE_TRACE_VECTOR_CELLS} vector cells"
    );
    ensure!(
        args.identity_cache.is_some(),
        "Muse trace-full requires --identity-cache"
    );
    Ok(())
}

fn validate_trace_vector_requests(
    args: &TraceFullArgs,
    layers: &[u32],
    token_count: usize,
) -> Result<BTreeSet<(u32, usize)>> {
    let mut requests = BTreeSet::new();
    ensure!(
        args.vectors.iter().all(|cell| {
            layers.contains(&cell.source_layer)
                && cell.source_position < token_count
                && requests.insert((cell.source_layer, cell.source_position))
        }),
        "Muse vector cells must be unique selected-layer positions inside the input"
    );
    Ok(requests)
}

fn aggregate_muse_trace_occurrences(
    cells: &[MuseTraceCell],
    selected_layers: &[u32],
) -> MuseTraceOccurrences {
    let mut global = BTreeMap::<u32, MuseOccurrenceAccumulator>::new();
    let mut per_layer = BTreeMap::<u32, BTreeMap<u32, MuseOccurrenceAccumulator>>::new();
    for cell in cells {
        let mut cell_ranks = BTreeMap::<u32, usize>::new();
        for score in &cell.top_k {
            cell_ranks
                .entry(score.token_id)
                .and_modify(|rank| *rank = (*rank).min(score.rank))
                .or_insert(score.rank);
        }
        for (token_id, rank) in cell_ranks {
            update_muse_occurrence(&mut global, token_id, rank);
            update_muse_occurrence(
                per_layer.entry(cell.source_layer).or_default(),
                token_id,
                rank,
            );
        }
    }
    MuseTraceOccurrences {
        global: sorted_muse_occurrences(global),
        per_layer: selected_layers
            .iter()
            .map(|&source_layer| MuseTraceLayerOccurrences {
                source_layer,
                tokens: sorted_muse_occurrences(
                    per_layer.remove(&source_layer).unwrap_or_default(),
                ),
            })
            .collect(),
    }
}

fn update_muse_occurrence(
    occurrences: &mut BTreeMap<u32, MuseOccurrenceAccumulator>,
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
        .or_insert(MuseOccurrenceAccumulator {
            count: 1,
            top1_count: usize::from(rank == 0),
            best_rank: rank,
        });
}

fn sorted_muse_occurrences(
    occurrences: BTreeMap<u32, MuseOccurrenceAccumulator>,
) -> Vec<MuseTraceOccurrence> {
    let mut occurrences = occurrences
        .into_iter()
        .map(|(token_id, occurrence)| MuseTraceOccurrence {
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

fn print_muse_trace_summary(document: &MuseTraceDocument, output: Option<&Path>) {
    println!(
        "{} {} | {} tokens x {} layers = {} cells | top-k {}",
        document.lens.method,
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
    println!("{}", muse_trace_timing_summary(&document.timing));
    if let Some(path) = output {
        println!("artifact {}", path.display());
    }
}

fn muse_trace_timing_summary(timing: &BTreeMap<&'static str, f64>) -> String {
    let trace = timing["trace_execution_wall_ms"];
    let prefill = timing["scalar_prefill_capture_wall_ms"];
    let prepare = timing["transport_prepare_wall_ms"];
    if let (Some(&command), Some(&gpu)) = (
        timing.get("batched_readout_command_wall_ms"),
        timing.get("batched_readout_gpu_ms"),
    ) {
        format!(
            "trace {trace:.1} ms (prefill {prefill:.1} ms, transport prepare {prepare:.1} ms, batched command {command:.1} ms, GPU {gpu:.1} ms)"
        )
    } else {
        format!(
            "trace {trace:.1} ms (prefill {prefill:.1} ms, transport prepare {prepare:.1} ms, scalar transport {:.1} ms, scalar output {:.1} ms)",
            timing["scalar_transport_wall_ms"], timing["scalar_output_tail_wall_ms"]
        )
    }
}

fn legacy_muse_trace_transport_wall_ms(
    transport_prepare_wall_ms: f64,
    transport_execution_wall_ms: f64,
) -> f64 {
    transport_prepare_wall_ms + transport_execution_wall_ms
}

fn use_scalar_muse_trace_readout(scalar_requested: bool, command_batch_supported: bool) -> bool {
    scalar_requested || !command_batch_supported
}

fn muse_trace_composite_cpu_bytes(
    host_transport_bytes: u64,
    additional_host_bytes: u64,
) -> Result<u64> {
    host_transport_bytes
        .checked_add(additional_host_bytes)
        .context("Muse trace composite host memory estimate overflow")
}

fn muse_trace_composite_admission(
    context: &MetalContext,
    plan: &MuseGlimmerFullReadoutWorkspacePlan,
    additional_host_bytes: u64,
) -> Result<MetalMemoryAdmission> {
    Ok(evaluate_metal_memory_admission_with_cpu_bytes(
        plan.priced_upper_bytes(),
        muse_trace_composite_cpu_bytes(plan.host_transport_reserve_bytes(), additional_host_bytes)?,
        plan.prepared_transport_reserve_bytes(),
        context.memory_signals(),
        true,
    ))
}

fn validate_read_args(args: &ReadFullArgs) -> Result<()> {
    ensure!(
        args.prompt.is_some() ^ !args.token_ids.is_empty(),
        "exactly one of --prompt or --token-ids is required"
    );
    validate_muse_top_k(args.top_k)?;
    ensure!(args.max_tokens > 0, "--max-tokens must be positive");
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

fn prepare_read_input(
    args: &ReadFullArgs,
    tokenizer: &LlamaCppTokenizer,
) -> Result<(&'static str, Option<bool>, Vec<u32>)> {
    let (source, add_special_tokens, signed) = if let Some(prompt) = &args.prompt {
        let add_special_tokens = !args.no_special_tokens;
        (
            "prompt",
            Some(add_special_tokens),
            tokenizer
                .encode(prompt, add_special_tokens)
                .context("tokenize Muse full-readout prompt")?,
        )
    } else {
        let mut signed = Vec::with_capacity(args.token_ids.len());
        for &token in &args.token_ids {
            ensure!(
                token < tokenizer.n_vocab() && token <= i32::MAX as u32,
                "--token-ids entry {token} is outside Muse vocabulary {}",
                tokenizer.n_vocab()
            );
            signed.push(token as i32);
        }
        ("token_ids", None, signed)
    };
    ensure!(!signed.is_empty(), "Muse full-readout input has no tokens");
    ensure!(
        signed.len() <= args.max_tokens,
        "Muse full-readout input has {} tokens, exceeding --max-tokens {}",
        signed.len(),
        args.max_tokens
    );
    let tokens = signed
        .into_iter()
        .map(|token| {
            ensure!(
                token >= 0 && (token as u32) < tokenizer.n_vocab(),
                "Muse tokenizer produced token {token} outside its vocabulary"
            );
            Ok(token as u32)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((source, add_special_tokens, tokens))
}

fn select_layers(requested: &[u32], available: &[u32]) -> Result<Vec<u32>> {
    let layers = if requested.is_empty() {
        available.to_vec()
    } else {
        requested.to_vec()
    };
    let mut unique = HashSet::new();
    ensure!(
        !layers.is_empty()
            && layers
                .iter()
                .all(|layer| unique.insert(*layer) && available.binary_search(layer).is_ok()),
        "--layers must be unique source layers present in the Muse full transport"
    );
    Ok(layers)
}

pub(crate) fn read_matrix(
    directory: &Path,
    payload_path: &str,
    payload_byte_length: u64,
    matrix: &artifact::MatrixDescriptor,
) -> Result<Vec<u8>> {
    ensure!(
        Path::new(payload_path).components().count() == 1,
        "Muse full-transport payload path must be one relative filename"
    );
    let path = directory.join(payload_path);
    let (mut file, length) = super::open_regular_file(&path)?;
    ensure!(
        length as u64 == payload_byte_length,
        "Muse full-transport payload length changed"
    );
    file.seek(SeekFrom::Start(matrix.byte_offset))
        .with_context(|| format!("seek Muse source matrix {}", matrix.source_layer))?;
    let matrix_len = usize::try_from(matrix.byte_length)
        .context("Muse source matrix byte length does not fit this platform")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(matrix_len)
        .context("allocate Muse source matrix")?;
    bytes.resize(matrix_len, 0);
    file.read_exact(&mut bytes)
        .with_context(|| format!("read Muse source matrix {}", matrix.source_layer))?;
    ensure!(
        blake3::hash(&bytes).to_hex().as_str() == matrix.blake3,
        "Muse source matrix {} digest mismatch",
        matrix.source_layer
    );
    Ok(bytes)
}

fn rms_denominator(values: &[f32], epsilon: f32) -> f32 {
    ((values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / values.len() as f64)
        + f64::from(epsilon))
    .sqrt() as f32
}

fn top_k_logits(logits: &[f32], top_k: usize) -> Result<Vec<(u32, f32)>> {
    validate_muse_top_k(top_k)?;
    let mut ranked = Vec::<(u32, f32)>::with_capacity(top_k);
    for (token_id, &logit) in logits.iter().enumerate() {
        ensure!(
            logit.is_finite(),
            "Muse output tail produced non-finite logits"
        );
        let token_id = u32::try_from(token_id).context("Muse logit token ID")?;
        if ranked.len() < top_k {
            ranked.push((token_id, logit));
            ranked.sort_by(compare_scores);
        } else if compare_scores(&(token_id, logit), ranked.last().unwrap()).is_lt() {
            *ranked.last_mut().unwrap() = (token_id, logit);
            ranked.sort_by(compare_scores);
        }
    }
    ensure!(
        ranked.len() == top_k,
        "Muse vocabulary is smaller than top-k"
    );
    Ok(ranked)
}

fn validate_muse_top_k(top_k: usize) -> Result<()> {
    ensure!(
        (1..=MUSE_GLIMMER_FULL_READOUT_MAX_TOP_K).contains(&top_k),
        "Muse --top-k must be in 1..={MUSE_GLIMMER_FULL_READOUT_MAX_TOP_K}"
    );
    Ok(())
}

fn compare_scores(left: &(u32, f32), right: &(u32, f32)) -> std::cmp::Ordering {
    right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
}

pub(crate) fn assemble(mut args: AssembleMuseFullArgs) -> Result<()> {
    super::validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    let shards_root = canonical_real_directory(&args.shards_root, "Muse row-shard root")?;
    args.output = super::resolve_output_path(&args.output)?;
    ensure!(
        args.output != shards_root,
        "Muse full-transport output must differ from its shard root"
    );
    let shards = discover_shards(&shards_root)?;
    let first = shards.first().context("Muse row-shard root is empty")?;
    let config = artifact::config_from_row(&first.manifest.config);
    artifact::validate_config(&config)?;
    validate_shard_set(&shards, &config)?;
    let assembly = assembly_descriptor(&shards, config.geometry.hidden_size);
    let corpus = first.manifest.corpus.clone();
    let identity = identity_summary(&shards)?;
    let replay = aggregate_replay(&shards)?;
    let config_blake3 = super::digest_json(&config)?;

    if let Some(manifest) = open_output(
        &args.output,
        args.resume,
        &config,
        &identity,
        &corpus,
        &assembly,
        &config_blake3,
    )? {
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }

    let partial_path = args.output.join(artifact::PARTIAL_PAYLOAD_NAME);
    let payload_path = args.output.join(artifact::PAYLOAD_NAME);
    remove_regular_if_present(&partial_path)?;
    remove_regular_if_present(&payload_path)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&partial_path)
        .with_context(|| {
            format!(
                "create Muse transport staging file {}",
                partial_path.display()
            )
        })?;

    let hidden = config.geometry.hidden_size as usize;
    let mut shard_hashers = (0..shards.len()).map(|_| Hasher::new()).collect::<Vec<_>>();
    let mut payload_hasher = Hasher::new();
    let matrix_bytes = artifact::matrix_bytes(&config)?;
    let mut matrices = Vec::with_capacity(config.source_layers.len());
    let mut f32_bytes = Vec::new();
    let mut f16_bytes = Vec::new();
    let mut written = 0u64;

    for (source_slot, &source_layer) in config.source_layers.iter().enumerate() {
        eprintln!(
            "assemble Muse transport source {}/{} layer={source_layer}",
            source_slot + 1,
            config.source_layers.len()
        );
        let matrix_offset = written;
        let mut matrix_hasher = Hasher::new();
        for (shard_slot, shard) in shards.iter().enumerate() {
            let row_count =
                usize::try_from(shard.manifest.config.row_end - shard.manifest.config.row_start)
                    .context("Muse input shard row count")?;
            let source_slice_bytes = row_count
                .checked_mul(hidden)
                .and_then(|values| values.checked_mul(4))
                .context("Muse input source slice byte count overflow")?;
            let source_offset = source_slot
                .checked_mul(source_slice_bytes)
                .and_then(|bytes| u64::try_from(bytes).ok())
                .context("Muse input source slice offset overflow")?;
            let payload_path = shard.directory.join(&shard.manifest.payload.path);
            let (mut input, input_length) = super::open_regular_file(&payload_path)?;
            ensure!(
                input_length as u64 == shard.manifest.payload.byte_length,
                "Muse input payload {} length changed",
                payload_path.display()
            );
            input
                .seek(SeekFrom::Start(source_offset))
                .with_context(|| format!("seek Muse input payload {}", payload_path.display()))?;
            f32_bytes.resize(source_slice_bytes, 0);
            input
                .read_exact(&mut f32_bytes)
                .with_context(|| format!("read Muse input payload {}", payload_path.display()))?;
            shard_hashers[shard_slot].update(&f32_bytes);
            convert_f32_slice_to_f16(&f32_bytes, &mut f16_bytes)?;
            output
                .write_all(&f16_bytes)
                .context("write assembled Muse transport")?;
            matrix_hasher.update(&f16_bytes);
            payload_hasher.update(&f16_bytes);
            written = written
                .checked_add(u64::try_from(f16_bytes.len()).context("Muse F16 slice length")?)
                .context("Muse assembled payload length overflow")?;
        }
        ensure!(
            written - matrix_offset == matrix_bytes,
            "assembled Muse source matrix has the wrong byte length"
        );
        matrices.push(artifact::MatrixDescriptor {
            source_layer,
            byte_offset: matrix_offset,
            byte_length: matrix_bytes,
            blake3: matrix_hasher.finalize().to_hex().to_string(),
        });
    }

    for (shard, hasher) in shards.iter().zip(shard_hashers) {
        ensure!(
            hasher.finalize().to_hex().as_str() == shard.manifest.payload.blake3,
            "Muse input row-shard payload digest mismatch for rows {}..{}",
            shard.manifest.config.row_start,
            shard.manifest.config.row_end
        );
    }
    let expected_bytes = matrix_bytes
        .checked_mul(u64::try_from(config.source_layers.len()).context("Muse source count")?)
        .context("Muse assembled payload size overflow")?;
    ensure!(
        written == expected_bytes,
        "assembled Muse payload length {written} differs from expected {expected_bytes}"
    );
    output.sync_all().context("sync assembled Muse transport")?;
    drop(output);
    std::fs::rename(&partial_path, &payload_path).with_context(|| {
        format!(
            "publish Muse transport {} to {}",
            partial_path.display(),
            payload_path.display()
        )
    })?;
    super::sync_directory(&args.output)?;

    let manifest = artifact::Manifest {
        schema: artifact::SCHEMA.into(),
        schema_version: artifact::SCHEMA_VERSION,
        status: "complete".into(),
        config_blake3,
        config,
        identity,
        corpus,
        assembly,
        payload: artifact::Payload {
            path: artifact::PAYLOAD_NAME.into(),
            dtype: "f16_le".into(),
            shape: [config_source_count(&shards)?, hidden, hidden],
            byte_length: written,
            blake3: payload_hasher.finalize().to_hex().to_string(),
            matrices,
        },
        replay,
        provenance: artifact::Provenance {
            build_commit: env!("QWEN_BUILD_COMMIT").into(),
            build_dirty: env!("QWEN_BUILD_DIRTY").into(),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
            build_stamp_source: env!("QWEN_BUILD_STAMP_SOURCE").into(),
            build_stamp_error: env!("QWEN_BUILD_STAMP_ERROR").into(),
        },
    };
    artifact::validate_manifest(&manifest)?;
    super::publish_immutable(
        &args.output.join(artifact::MANIFEST_NAME),
        &super::serialize_json_pretty_bounded(&manifest, "Muse full-transport manifest")?,
    )?;
    super::sync_directory(&args.output)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn canonical_real_directory(path: &Path, label: &str) -> Result<PathBuf> {
    let lexical = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect {label} {}", path.display()))?;
    ensure!(
        lexical.file_type().is_dir() && !lexical.file_type().is_symlink(),
        "{label} {} must be a real directory, not a symlink",
        path.display()
    );
    let canonical = std::fs::canonicalize(path)
        .with_context(|| format!("resolve {label} {}", path.display()))?;
    let metadata = std::fs::symlink_metadata(&canonical)
        .with_context(|| format!("inspect {label} {}", canonical.display()))?;
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "{label} {} must be a real directory",
        canonical.display()
    );
    Ok(canonical)
}

fn discover_shards(root: &Path) -> Result<Vec<ShardInput>> {
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(root)
        .with_context(|| format!("read Muse row-shard root {}", root.display()))?
    {
        let entry = entry.with_context(|| format!("read entry in {}", root.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("inspect Muse shard candidate {}", entry.path().display()))?;
        ensure!(
            !file_type.is_symlink(),
            "Muse shard root contains symlink {}",
            entry.path().display()
        );
        if !file_type.is_dir() {
            continue;
        }
        let manifest_path = entry.path().join(rows::MANIFEST_NAME);
        if !manifest_path.exists() {
            continue;
        }
        let manifest: rows::Manifest = super::read_json_file(&manifest_path)?;
        rows::validate_stored(&manifest)
            .with_context(|| format!("validate Muse row shard {}", entry.path().display()))?;
        shards.push(ShardInput {
            directory: entry.path(),
            manifest,
        });
    }
    shards.sort_by_key(|shard| shard.manifest.config.row_start);
    ensure!(
        !shards.is_empty(),
        "Muse row-shard root has no complete shards"
    );
    Ok(shards)
}

fn validate_shard_set(shards: &[ShardInput], config: &artifact::Config) -> Result<()> {
    let first = shards.first().context("Muse row-shard set is empty")?;
    let mut cursor = 0u32;
    for shard in shards {
        ensure!(
            artifact::config_from_row(&shard.manifest.config) == *config,
            "Muse row shard {} has incompatible fit configuration",
            shard.directory.display()
        );
        ensure!(
            shard.manifest.corpus == first.manifest.corpus,
            "Muse row shard {} has incompatible corpus summary",
            shard.directory.display()
        );
        ensure!(
            shard.manifest.config.row_start == cursor,
            "Muse row-shard coverage has a gap or overlap at row {cursor}"
        );
        cursor = shard.manifest.config.row_end;
    }
    ensure!(
        cursor == config.geometry.hidden_size,
        "Muse row-shard coverage ends at {cursor}, expected {}",
        config.geometry.hidden_size
    );
    Ok(())
}

fn assembly_descriptor(shards: &[ShardInput], hidden_size: u32) -> artifact::Assembly {
    artifact::Assembly {
        input_schema: rows::SCHEMA.into(),
        input_schema_version: rows::SCHEMA_VERSION,
        input_dtype: "f32_le".into(),
        conversion: artifact::CONVERSION.into(),
        row_coverage: [0, hidden_size],
        shards: shards
            .iter()
            .map(|shard| artifact::InputShard {
                row_start: shard.manifest.config.row_start,
                row_end: shard.manifest.config.row_end,
                config_blake3: shard.manifest.config_blake3.clone(),
                payload_blake3: shard.manifest.payload.blake3.clone(),
            })
            .collect(),
    }
}

fn aggregate_replay(shards: &[ShardInput]) -> Result<Vec<rows::ReplayDiagnostic>> {
    let mut aggregate = shards
        .first()
        .context("Muse row-shard set is empty")?
        .manifest
        .diagnostics
        .clone();
    for shard in &shards[1..] {
        ensure!(
            shard.manifest.diagnostics.len() == aggregate.len(),
            "Muse row-shard replay schedules differ"
        );
        for (aggregate, current) in aggregate.iter_mut().zip(&shard.manifest.diagnostics) {
            ensure!(
                aggregate.block == current.block && aggregate.kind == current.kind,
                "Muse row-shard replay schedules differ at block {}",
                current.block
            );
            aggregate.post_attention_replay_max_abs_error = aggregate
                .post_attention_replay_max_abs_error
                .max(current.post_attention_replay_max_abs_error);
            aggregate.post_block_replay_max_abs_error = aggregate
                .post_block_replay_max_abs_error
                .max(current.post_block_replay_max_abs_error);
        }
    }
    Ok(aggregate)
}

fn identity_summary(shards: &[ShardInput]) -> Result<artifact::IdentitySummary> {
    let first = shards.first().context("Muse row-shard set is empty")?;
    let content_blake3 = first.manifest.config.model_content_blake3.clone();
    let policy = first.manifest.config.content_identity_policy.clone();
    let mut input_outcomes = BTreeSet::new();
    let mut weight_bytes_hashed = 0u64;
    for shard in shards {
        ensure!(
            shard.manifest.model.content_blake3 == content_blake3
                && shard.manifest.model.content_authenticated
                && shard.manifest.config.content_identity_policy == policy,
            "Muse row-shard identity summaries differ"
        );
        input_outcomes.insert(shard.manifest.model.content_identity_outcome.clone());
        weight_bytes_hashed = weight_bytes_hashed
            .checked_add(shard.manifest.model.content_bytes_hashed)
            .context("Muse row-shard identity byte count overflow")?;
    }
    ensure!(
        weight_bytes_hashed == 0,
        "Muse full assembly refuses row shards that hashed model weights"
    );
    Ok(artifact::IdentitySummary {
        policy,
        content_blake3,
        content_authenticated: true,
        input_outcomes: input_outcomes.into_iter().collect(),
        weight_bytes_hashed,
    })
}

fn open_output(
    output: &Path,
    resume: bool,
    config: &artifact::Config,
    identity: &artifact::IdentitySummary,
    corpus: &rows::CorpusSummary,
    assembly: &artifact::Assembly,
    config_blake3: &str,
) -> Result<Option<artifact::Manifest>> {
    super::validate_output_leaf(output)?;
    let assembly_state = artifact::AssemblyState {
        schema: artifact::ASSEMBLY_STATE_SCHEMA.into(),
        schema_version: artifact::SCHEMA_VERSION,
        config_blake3: config_blake3.into(),
        assembly_blake3: super::digest_json(assembly)?,
    };
    let mut created = false;
    if output.exists() {
        let metadata = std::fs::symlink_metadata(output)
            .with_context(|| format!("inspect Muse full output {}", output.display()))?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "Muse full output {} must be a real directory",
            output.display()
        );
        ensure!(
            resume,
            "Muse full output {} already exists; pass --resume",
            output.display()
        );
    } else {
        let parent = output.parent().context("Muse full output has no parent")?;
        DirBuilder::new()
            .mode(0o700)
            .create(output)
            .with_context(|| format!("create Muse full output {}", output.display()))?;
        super::sync_directory(parent)?;
        created = true;
    }

    let manifest_path = output.join(artifact::MANIFEST_NAME);
    if !manifest_path.exists() {
        let state_path = output.join(artifact::ASSEMBLY_STATE_NAME);
        if created {
            super::publish_immutable(
                &state_path,
                &super::serialize_json_pretty_bounded(&assembly_state, "Muse full assembly state")?,
            )?;
        } else {
            let observed: artifact::AssemblyState = super::read_json_file(&state_path)
                .context("existing Muse output is not an owned interrupted assembly")?;
            ensure!(
                observed == assembly_state,
                "interrupted Muse assembly state differs from the requested shard set"
            );
        }
        return Ok(None);
    }
    let manifest: artifact::Manifest = super::read_json_file(&manifest_path)?;
    artifact::validate_manifest(&manifest)?;
    ensure!(
        &manifest.config == config
            && &manifest.identity == identity
            && &manifest.corpus == corpus
            && &manifest.assembly == assembly
            && manifest.config_blake3 == config_blake3,
        "completed Muse full transport differs from requested shard set"
    );
    verify_complete_payload(output, &manifest.payload)?;
    Ok(Some(manifest))
}

fn verify_complete_payload(output: &Path, payload: &artifact::Payload) -> Result<()> {
    const BUFFER_BYTES: usize = 8 * 1024 * 1024;
    let path = output.join(&payload.path);
    let (mut file, length) = super::open_regular_file(&path)?;
    ensure!(
        length as u64 == payload.byte_length,
        "completed Muse full-transport payload length changed"
    );
    let mut buffer = vec![0u8; BUFFER_BYTES];
    let mut whole = Hasher::new();
    for matrix in &payload.matrices {
        file.seek(SeekFrom::Start(matrix.byte_offset))
            .with_context(|| format!("seek completed Muse payload {}", path.display()))?;
        let mut remaining = matrix.byte_length;
        let mut matrix_hasher = Hasher::new();
        while remaining > 0 {
            let count = usize::try_from(remaining.min(BUFFER_BYTES as u64))
                .context("Muse payload verification chunk")?;
            file.read_exact(&mut buffer[..count])
                .with_context(|| format!("verify completed Muse payload {}", path.display()))?;
            matrix_hasher.update(&buffer[..count]);
            whole.update(&buffer[..count]);
            remaining -= count as u64;
        }
        ensure!(
            matrix_hasher.finalize().to_hex().as_str() == matrix.blake3,
            "completed Muse source matrix {} digest mismatch",
            matrix.source_layer
        );
    }
    ensure!(
        whole.finalize().to_hex().as_str() == payload.blake3,
        "completed Muse full-transport payload digest mismatch"
    );
    Ok(())
}

fn remove_regular_if_present(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "Muse assembly path {} must be a regular non-symlink file",
                path.display()
            );
            std::fs::remove_file(path)
                .with_context(|| format!("remove interrupted Muse assembly {}", path.display()))?;
            super::sync_directory(path.parent().unwrap_or_else(|| Path::new(".")))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspect Muse assembly path {}", path.display()));
        }
    }
    Ok(())
}

fn convert_f32_slice_to_f16(input: &[u8], output: &mut Vec<u8>) -> Result<()> {
    ensure!(
        input.len().is_multiple_of(4),
        "Muse F32 transport slice is not word-aligned"
    );
    output.clear();
    let expected = input
        .len()
        .checked_div(2)
        .context("Muse F16 transport slice length")?;
    output
        .try_reserve_exact(expected)
        .context("allocate Muse F16 conversion buffer")?;
    for chunk in input.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        ensure!(
            value.is_finite(),
            "Muse input transport contains non-finite F32"
        );
        let converted = f16::from_f32(value);
        ensure!(
            converted.is_finite(),
            "Muse input transport overflows finite F16 storage"
        );
        output.extend_from_slice(&converted.to_bits().to_le_bytes());
    }
    ensure!(
        output.len() == expected,
        "Muse F16 conversion produced the wrong byte count"
    );
    Ok(())
}

fn config_source_count(shards: &[ShardInput]) -> Result<usize> {
    Ok(shards
        .first()
        .context("Muse row-shard set is empty")?
        .manifest
        .config
        .source_layers
        .len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn trace_capture_pricing_scales_with_requested_work() {
        assert_eq!(
            muse_trace_capture_bytes(51, 129, 6_656).unwrap(),
            175_159_296
        );
        assert!(muse_trace_capture_bytes(usize::MAX, 2, 6_656).is_err());
    }

    #[test]
    fn trace_summary_uses_mode_specific_timing_keys() {
        let base = BTreeMap::from([
            ("trace_execution_wall_ms", 10.0),
            ("scalar_prefill_capture_wall_ms", 2.0),
            ("transport_prepare_wall_ms", 1.0),
            ("transport_wall_ms", 5.0),
            ("output_tail_wall_ms", 6.0),
        ]);
        let mut batched = base.clone();
        batched.insert("batched_readout_command_wall_ms", 4.0);
        batched.insert("batched_readout_gpu_ms", 3.0);
        let batched_summary = muse_trace_timing_summary(&batched);
        assert!(batched_summary.contains("batched command 4.0 ms"));
        assert!(batched_summary.contains("GPU 3.0 ms"));

        let mut scalar = base;
        scalar.insert("scalar_transport_wall_ms", 5.0);
        scalar.insert("scalar_output_tail_wall_ms", 6.0);
        let scalar_summary = muse_trace_timing_summary(&scalar);
        assert!(scalar_summary.contains("scalar transport 5.0 ms"));
        assert!(scalar_summary.contains("scalar output 6.0 ms"));
        assert_eq!(legacy_muse_trace_transport_wall_ms(1.0, 5.0), 6.0);
    }

    #[test]
    fn trace_readout_automatically_falls_back_without_blocking_scalar_models() {
        assert!(!use_scalar_muse_trace_readout(false, true));
        assert!(use_scalar_muse_trace_readout(false, false));
        assert!(use_scalar_muse_trace_readout(true, true));
    }

    #[test]
    fn trace_composite_admission_accounts_for_matrix_and_retained_results() {
        assert_eq!(muse_trace_composite_cpu_bytes(80, 120).unwrap(), 200);
        assert!(muse_trace_composite_cpu_bytes(u64::MAX, 1).is_err());
    }

    #[test]
    fn artifact_probe_recognizes_local_and_published_muse_schemas() {
        let root = std::env::temp_dir().join(format!(
            "qwen-muse-full-probe-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        for (schema, expected) in [
            (artifact::SCHEMA, true),
            (published::SCHEMA, true),
            ("qwen.workspace_lens_full_transport", false),
        ] {
            std::fs::write(
                root.join(artifact::MANIFEST_NAME),
                format!(r#"{{"schema":"{schema}"}}"#),
            )
            .unwrap();
            assert_eq!(is_artifact(&root).unwrap(), expected);
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn f32_to_f16_conversion_is_little_endian_and_fails_closed() {
        let input = [0.0f32, 1.5, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut output = Vec::new();
        convert_f32_slice_to_f16(&input, &mut output).unwrap();
        let observed = output
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(observed, [0.0, 1.5, -2.0]);

        let nan = f32::NAN.to_le_bytes();
        assert!(convert_f32_slice_to_f16(&nan, &mut output).is_err());
        let overflow = f32::MAX.to_le_bytes();
        assert!(convert_f32_slice_to_f16(&overflow, &mut output).is_err());
    }

    #[test]
    fn completed_payload_verification_checks_matrix_and_whole_digests() {
        let root = std::env::temp_dir().join(format!(
            "qwen-muse-full-verify-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let bytes = b"abcdefgh";
        std::fs::write(root.join(artifact::PAYLOAD_NAME), bytes).unwrap();
        let payload = artifact::Payload {
            path: artifact::PAYLOAD_NAME.into(),
            dtype: "f16_le".into(),
            shape: [2, 1, 2],
            byte_length: bytes.len() as u64,
            blake3: blake3::hash(bytes).to_hex().to_string(),
            matrices: vec![
                artifact::MatrixDescriptor {
                    source_layer: 0,
                    byte_offset: 0,
                    byte_length: 4,
                    blake3: blake3::hash(&bytes[..4]).to_hex().to_string(),
                },
                artifact::MatrixDescriptor {
                    source_layer: 1,
                    byte_offset: 4,
                    byte_length: 4,
                    blake3: blake3::hash(&bytes[4..]).to_hex().to_string(),
                },
            ],
        };
        verify_complete_payload(&root, &payload).unwrap();
        std::fs::write(root.join(artifact::PAYLOAD_NAME), b"abcdEfgh").unwrap();
        assert!(verify_complete_payload(&root, &payload).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn full_vocabulary_top_k_is_descending_with_stable_token_ties() {
        let logits = [0.5, 2.0, 2.0, -1.0, 1.5];
        assert_eq!(
            top_k_logits(&logits, 3).unwrap(),
            [(1, 2.0), (2, 2.0), (4, 1.5)]
        );
        let mut invalid = logits;
        invalid[3] = f32::NAN;
        assert!(top_k_logits(&invalid, 3).is_err());

        let tied = vec![1.0; 64];
        assert_eq!(
            top_k_logits(&tied, 32).unwrap(),
            (0..32).map(|token_id| (token_id, 1.0)).collect::<Vec<_>>()
        );
        assert!(validate_muse_top_k(0).is_err());
        assert!(validate_muse_top_k(1).is_ok());
        assert!(validate_muse_top_k(32).is_ok());
        assert!(validate_muse_top_k(33).is_err());
    }

    #[test]
    fn trace_occurrences_match_inspector_ordering_and_layer_schedule() {
        let score = |rank, token_id| MuseTraceTokenScore {
            rank,
            token_id,
            token_display_lossy: token_id.to_string(),
            token_piece_hex: format!("{token_id:02x}"),
            logit: -(rank as f32),
        };
        let cells = vec![
            MuseTraceCell {
                source_layer: 25,
                source_position: 0,
                source_token_id: 1,
                predicts_position: 1,
                top_k: vec![score(0, 7), score(1, 9)],
            },
            MuseTraceCell {
                source_layer: 50,
                source_position: 0,
                source_token_id: 1,
                predicts_position: 1,
                top_k: vec![score(0, 9), score(1, 7)],
            },
        ];
        let occurrences = aggregate_muse_trace_occurrences(&cells, &[50, 25]);
        assert_eq!(
            occurrences
                .global
                .iter()
                .map(|item| (item.token_id, item.count, item.top1_count))
                .collect::<Vec<_>>(),
            [(7, 2, 1), (9, 2, 1)]
        );
        assert_eq!(occurrences.per_layer[0].source_layer, 50);
        assert_eq!(occurrences.per_layer[1].source_layer, 25);
    }
}
