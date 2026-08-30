use super::full_lens::ReadFullArgs;
use super::muse_full_lens_artifact as artifact;
use super::muse_lens_artifact;
use super::muse_lens_rows_artifact as rows;
use anyhow::{Context, Result, ensure};
use blake3::Hasher;
use clap::Args;
use half::f16;
use qwen_llm::checkpoint_identity::{CheckpointIdentityCache, checkpoint_content_identity};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::muse_glimmer::{ARCHITECTURE_NAME, MuseGlimmerModel};
use qwen_llm::muse_glimmer_runtime::MuseGlimmerLoadedModel;
use qwen_llm::tokenizer::LlamaCppTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
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
struct ReadoutModel {
    path: PathBuf,
    content_blake3: String,
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

pub(crate) fn is_artifact(directory: &Path) -> Result<bool> {
    let path = directory.join(artifact::MANIFEST_NAME);
    if !path.exists() {
        return Ok(false);
    }
    let probe: SchemaProbe = super::read_json_file(&path)?;
    Ok(probe.schema == artifact::SCHEMA)
}

pub(crate) fn read_full(args: ReadFullArgs) -> Result<()> {
    super::validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    validate_read_args(&args)?;
    let full_lens = canonical_real_directory(&args.full_lens, "Muse full-transport artifact")?;
    let manifest_path = full_lens.join(artifact::MANIFEST_NAME);
    let manifest: artifact::Manifest = super::read_json_file(&manifest_path)?;
    artifact::validate_manifest(&manifest)?;
    let manifest_canonical_json_blake3 = super::digest_json(&manifest)?;

    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open Muse model {}", args.model.display()))?;
    let bound =
        MuseGlimmerModel::from_gguf(&gguf).context("bind Muse model for full-transport readout")?;
    ensure!(
        manifest.config.architecture == ARCHITECTURE_NAME
            && manifest.config.artifact_profile
                == muse_lens_artifact::profile_name(bound.artifact_profile)
            && manifest.config.geometry == muse_lens_artifact::geometry(&bound.config),
        "Muse full transport does not match the deployed model profile or geometry"
    );
    let content =
        checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(&args.identity_cache))
            .with_context(|| {
                format!(
                    "resolve Muse model identity using {}",
                    args.identity_cache.display()
                )
            })?;
    let content_id = super::hex(&content.content_id);
    ensure!(
        content_id == manifest.config.model_content_blake3,
        "Muse full transport was fitted for a different GGUF content identity"
    );

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
    let layers = select_layers(&args.layers, &manifest.config.source_layers)?;
    let mut capture_layers = layers.clone();
    capture_layers.sort_unstable();
    let capture_slots = capture_layers
        .iter()
        .copied()
        .enumerate()
        .map(|(slot, layer)| (layer, slot))
        .collect::<BTreeMap<_, _>>();

    let prefix = &token_ids[..=selected_position];
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
        let descriptor = manifest
            .payload
            .matrices
            .iter()
            .find(|matrix| matrix.source_layer == layer)
            .context("Muse full transport omitted a selected source matrix")?;
        let started = Instant::now();
        let matrix = read_matrix(&full_lens, &manifest.payload, descriptor)?;
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

    let document = ReadoutDocument {
        schema: "muse_glimmer.lens.full_readout",
        schema_version: 1,
        readout: "full_vocabulary",
        score_semantics: "deployed_output_rmsnorm_native_head_scale_softcap_no_softmax_v1",
        ranking_scope: "full_vocabulary",
        source_site: "post_block_residual",
        input: ReadoutInput {
            source: input_source,
            add_special_tokens,
            token_ids,
            selected_position,
            captured_token_id: capture.token_id,
            predicts_position: selected_position + 1,
        },
        artifact: ReadoutArtifact {
            manifest: manifest_path,
            manifest_canonical_json_blake3,
            declared_payload_blake3: manifest.payload.blake3.clone(),
            model_content_blake3: manifest.config.model_content_blake3.clone(),
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
            path: args.model,
            content_blake3: content_id,
            architecture: ARCHITECTURE_NAME,
            artifact_profile: manifest.config.artifact_profile,
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
    let bytes = super::serialize_json_pretty_bounded(&document, "Muse full readout")?;
    if let Some(output) = args.output {
        let output = super::resolve_output_path(&output)?;
        super::publish_immutable(&output, &bytes)?;
    }
    println!("{}", String::from_utf8(bytes).unwrap());
    Ok(())
}

fn validate_read_args(args: &ReadFullArgs) -> Result<()> {
    ensure!(
        args.prompt.is_some() ^ !args.token_ids.is_empty(),
        "exactly one of --prompt or --token-ids is required"
    );
    ensure!(
        args.top_k > 0 && args.top_k <= 16,
        "--top-k must be in 1..=16"
    );
    ensure!(
        args.max_tokens > 0 && args.max_tokens <= 4_096,
        "--max-tokens must be in 1..=4096"
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

fn read_matrix(
    directory: &Path,
    payload: &artifact::Payload,
    matrix: &artifact::MatrixDescriptor,
) -> Result<Vec<u8>> {
    let path = directory.join(&payload.path);
    let (mut file, length) = super::open_regular_file(&path)?;
    ensure!(
        length as u64 == payload.byte_length,
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
    ensure!(top_k > 0 && top_k <= 16, "invalid Muse top-k bound");
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
    let replay = aggregate_replay(&shards)?;
    let config_blake3 = super::digest_json(&config)?;

    if let Some(manifest) = open_output(
        &args.output,
        args.resume,
        &config,
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

fn open_output(
    output: &Path,
    resume: bool,
    config: &artifact::Config,
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
    }
}
