use super::muse_lens_artifact as artifact;
use super::{FitMethod, FitRowsArgs, PayloadDescriptor, PreparedPrompt, SkippedPrompt};
use anyhow::{Context, Result, ensure};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::muse_glimmer::{ARCHITECTURE_NAME, MuseGlimmerArtifactProfile, MuseGlimmerModel};
use qwen_llm::muse_glimmer_lens::MuseGlimmerLensRule;
use qwen_llm::muse_glimmer_lens_fit::MUSE_GLIMMER_FULL_R_MAX_DIM_BATCH;
use qwen_llm::muse_glimmer_runtime::MuseGlimmerLoadedModel;
use qwen_llm::tokenizer::LlamaCppTokenizer;
use serde::{Deserialize, Serialize};
use std::fs::DirBuilder;
use std::os::unix::fs::DirBuilderExt;
use std::path::Path;
use std::time::Instant;

const SHARD_SCHEMA: &str = "muse_glimmer.full_r_row_shard";
const CHECKPOINT_SCHEMA: &str = "muse_glimmer.full_r_row_checkpoint";
const SCHEMA_VERSION: u32 = 1;
const ESTIMATOR_VERSION: &str = "summed_causal_target_vjp_mean_source_positions_v1";
const ORIENTATION: &str = "source_layer_output_coordinate_source_coordinate";
const RULE_CONTRACT: &str = "muse_glimmer_batched_full_attention_block_r_v1";
const COORDINATE: &str = "post_block_residual_hugging_face_block_output";
const REPLAY_SEMANTICS: &str = "local_smooth_f32_block_vjp_at_production_f16_kv_trajectory_capture";
const PRODUCTION_SEMANTICS: &str = "scalar_forward_trajectory_with_f16_attention_kv_cache";
const MODEL_IDENTITY_SCHEME: &str = "muse_retained_file_stamps_tensor_table_v1";
const PAYLOAD_NAME: &str = "rows.f32le";
const MANIFEST_NAME: &str = "shard.json";
const CHECKPOINT_NAME: &str = "checkpoint.json";
const MAX_SHARD_ROWS: usize = 256;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct MuseRowConfig {
    estimator_version: String,
    orientation: String,
    rule_contract: String,
    coordinate: String,
    replay_semantics: String,
    production_semantics: String,
    method: FitMethod,
    model_locator_blake3: String,
    model_identity_scheme: String,
    architecture: String,
    artifact_profile: String,
    geometry: artifact::Geometry,
    tokenizer_identity_sha256: String,
    chat_template_sha256: String,
    target_layer: u32,
    source_layers: Vec<u32>,
    row_start: u32,
    row_end: u32,
    skip_first: usize,
    max_tokens: usize,
    add_special_tokens: bool,
    corpus_blake3: String,
    selected_records: usize,
    build_source_state: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct MuseReplayDiagnostic {
    block: u32,
    kind: String,
    post_attention_replay_max_abs_error: f32,
    post_block_replay_max_abs_error: f32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MuseRowCheckpoint {
    schema: String,
    schema_version: u32,
    config_blake3: String,
    generation: u64,
    next_record: usize,
    used_prompts: u64,
    truncated_prompts: u64,
    skipped_prompts: Vec<SkippedPrompt>,
    forward_seconds: f64,
    vjp_seconds: f64,
    sums: PayloadDescriptor,
    diagnostics: Vec<MuseReplayDiagnostic>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MuseModelSummary {
    path: String,
    locator_blake3: String,
    identity_scheme: String,
    content_authenticated: bool,
    locator_weight_bytes_hashed: u64,
    architecture: String,
    artifact_profile: String,
    geometry: artifact::Geometry,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MuseCorpusSummary {
    selected_records: usize,
    used_prompts: u64,
    skipped_prompts: Vec<SkippedPrompt>,
    truncated_prompts: u64,
    ordered_token_ids_blake3: String,
    add_special_tokens: bool,
    max_tokens: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MuseFitSummary {
    estimator_version: String,
    orientation: String,
    rule_contract: String,
    coordinate: String,
    replay_semantics: String,
    production_semantics: String,
    method: FitMethod,
    target_layer: u32,
    source_layers: Vec<u32>,
    row_start: u32,
    row_end: u32,
    dim_batch: usize,
    skip_first: usize,
    valid_position_denominator: String,
    prompt_denominator: String,
    accumulator_dtype: String,
    storage_dtype: String,
    forward_seconds: f64,
    vjp_seconds: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MuseProvenance {
    build_commit: String,
    build_dirty: String,
    build_source_state: String,
    build_stamp_source: String,
    build_stamp_error: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MuseRowManifest {
    schema: String,
    schema_version: u32,
    status: String,
    config_blake3: String,
    config: MuseRowConfig,
    model: MuseModelSummary,
    corpus: MuseCorpusSummary,
    fit: MuseFitSummary,
    payload: PayloadDescriptor,
    diagnostics: Vec<MuseReplayDiagnostic>,
    provenance: MuseProvenance,
}

struct ActiveState {
    generation: u64,
    next_record: usize,
    used_prompts: u64,
    truncated_prompts: u64,
    skipped_prompts: Vec<SkippedPrompt>,
    forward_seconds: f64,
    vjp_seconds: f64,
    sums: Vec<f32>,
    sums_path: Option<String>,
    diagnostics: Vec<MuseReplayDiagnostic>,
}

enum WorkerState {
    Active(ActiveState),
    Complete(Box<MuseRowManifest>),
}

pub(crate) fn fit_rows(mut args: FitRowsArgs, gguf: GgufFile) -> Result<()> {
    validate_args(&args)?;
    super::validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    let bound = MuseGlimmerModel::from_gguf(&gguf)
        .context("bind Muse Glimmer artifact profile and geometry")?;
    let config = bound.config.clone();
    let profile = bound.artifact_profile;
    drop(bound);
    ensure!(
        profile == MuseGlimmerArtifactProfile::UnslothQ8_0,
        "Muse full-R row fitting currently requires the released Q8_0 profile"
    );
    ensure!(
        args.target_layer > 0
            && args.target_layer < config.layer_count
            && args.source_layers == [args.target_layer - 1]
            && !config.sliding_layers[args.target_layer as usize],
        "Muse full-R row fitting requires one adjacent source below a full-attention target"
    );
    ensure!(
        args.target_layer == 51 && args.source_layers == [50],
        "the first Muse full-R production contract is target 51 to source 50 only"
    );
    ensure!(
        args.row_end <= config.hidden_size,
        "Muse row range {}..{} exceeds hidden size {}",
        args.row_start,
        args.row_end,
        config.hidden_size
    );

    args.output = super::resolve_output_path(&args.output)?;
    let requests = super::read_prompt_requests(&args.prompts, args.max_prompts)?;
    let tokenizer = LlamaCppTokenizer::from_gguf(&gguf, &args.model)
        .context("load Muse llama.cpp tokenizer")?;
    artifact::validate_tokenizer(&tokenizer, &config)?;
    let add_special_tokens = !args.no_special_tokens;
    let prompts = super::prepare_prompts(
        requests,
        &tokenizer,
        add_special_tokens,
        args.max_tokens,
        config.vocab_size,
    )?;
    let corpus_blake3 = super::corpus_digest(&prompts);
    let model_locator_blake3 = model_locator_digest(&gguf, &config, profile)?;
    let row_count = usize::try_from(args.row_end - args.row_start).context("Muse row count")?;
    let hidden_size = config.hidden_size as usize;
    let value_count = row_count
        .checked_mul(hidden_size)
        .context("Muse row accumulator size overflow")?;
    let geometry = artifact::geometry(&config);
    let fit_config = MuseRowConfig {
        estimator_version: ESTIMATOR_VERSION.into(),
        orientation: ORIENTATION.into(),
        rule_contract: RULE_CONTRACT.into(),
        coordinate: COORDINATE.into(),
        replay_semantics: REPLAY_SEMANTICS.into(),
        production_semantics: PRODUCTION_SEMANTICS.into(),
        method: args.method,
        model_locator_blake3: model_locator_blake3.clone(),
        model_identity_scheme: MODEL_IDENTITY_SCHEME.into(),
        architecture: ARCHITECTURE_NAME.into(),
        artifact_profile: artifact::profile_name(profile).into(),
        geometry: geometry.clone(),
        tokenizer_identity_sha256: super::hex(&config.tokenizer_identity_sha256),
        chat_template_sha256: super::hex(&config.chat_template_sha256),
        target_layer: args.target_layer,
        source_layers: args.source_layers.clone(),
        row_start: args.row_start,
        row_end: args.row_end,
        skip_first: args.skip_first,
        max_tokens: args.max_tokens,
        add_special_tokens,
        corpus_blake3: corpus_blake3.clone(),
        selected_records: prompts.len(),
        build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
    };
    let config_blake3 = super::digest_json(&fit_config)?;
    let shape = [1, row_count, hidden_size];
    let mut state = open_or_create_state(
        &args.output,
        args.resume,
        &fit_config,
        &prompts,
        &config_blake3,
        value_count,
        shape,
    )?;
    if let WorkerState::Complete(manifest) = state {
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }
    let WorkerState::Active(ref mut active) = state else {
        unreachable!();
    };
    validate_active_state(active, &prompts, &fit_config, value_count)?;

    let context = MetalContext::new().context("initialize Metal for Muse full-R fitting")?;
    let mut loaded = MuseGlimmerLoadedModel::load(&context, &gguf, args.max_tokens)
        .context("load Muse Glimmer full-R model")?;
    let mut runner = loaded
        .create_runner(&context)
        .context("create Muse full-R runner")?;

    for (record_index, prompt) in prompts.iter().enumerate().skip(active.next_record) {
        if let Some(skipped) = super::skipped_prompt(prompt, args.skip_first) {
            active.skipped_prompts.push(skipped);
            active.next_record = record_index + 1;
            checkpoint_active(&args.output, &config_blake3, active, shape)?;
            continue;
        }
        eprintln!(
            "fit Muse full-R prompt {}/{} id={} tokens={} rows={}..{} dim_batch={}",
            record_index + 1,
            prompts.len(),
            prompt.id,
            prompt.token_ids.len(),
            args.row_start,
            args.row_end,
            args.dim_batch
        );
        runner
            .reset()
            .with_context(|| format!("reset Muse session for prompt {}", prompt.id))?;
        let tokens = prompt
            .token_ids
            .iter()
            .map(|&token| token as u32)
            .collect::<Vec<_>>();
        let started = Instant::now();
        let capture = runner
            .capture_fresh_lens_prompt(&tokens, args.target_layer)
            .with_context(|| format!("capture Muse prompt {}", prompt.id))?;
        active.forward_seconds += started.elapsed().as_secs_f64();
        let started = Instant::now();
        let slab = runner
            .fit_adjacent_full_attention_rows_batched(
                &capture,
                args.row_start..args.row_end,
                args.skip_first,
                args.dim_batch,
                MuseGlimmerLensRule::R,
            )
            .with_context(|| format!("fit Muse full-R rows for prompt {}", prompt.id))?;
        active.vjp_seconds += started.elapsed().as_secs_f64();
        let expected_valid_positions = tokens
            .len()
            .checked_sub(args.skip_first)
            .and_then(|count| count.checked_sub(1))
            .context("Muse valid-position denominator underflow")?;
        ensure!(
            slab.source_block == args.source_layers[0]
                && slab.target_block == args.target_layer
                && slab.rule == MuseGlimmerLensRule::R
                && slab.row_start == args.row_start
                && slab.row_end == args.row_end
                && slab.n_tokens == tokens.len()
                && slab.n_valid_positions == expected_valid_positions
                && slab.hidden_size == hidden_size
                && slab.values.len() == active.sums.len(),
            "Muse full-R fit returned inconsistent metadata"
        );
        for (sum, value) in active.sums.iter_mut().zip(slab.values) {
            *sum += value;
        }
        ensure!(
            active.sums.iter().all(|value| value.is_finite()),
            "Muse full-R accumulator became non-finite"
        );
        merge_diagnostic(
            &mut active.diagnostics,
            MuseReplayDiagnostic {
                block: slab.target_block,
                kind: "full".into(),
                post_attention_replay_max_abs_error: slab.post_attention_replay_max_abs_error,
                post_block_replay_max_abs_error: slab.post_block_replay_max_abs_error,
            },
        )?;
        active.used_prompts += 1;
        active.truncated_prompts += u64::from(prompt.truncated);
        active.next_record = record_index + 1;
        checkpoint_active(&args.output, &config_blake3, active, shape)?;
    }
    ensure!(
        active.used_prompts > 0,
        "no Muse corpus prompt had valid fit positions"
    );

    let scale = (active.used_prompts as f32).recip();
    let mut averaged = active.sums.clone();
    averaged.iter_mut().for_each(|value| *value *= scale);
    ensure!(
        averaged.iter().all(|value| value.is_finite()),
        "averaged Muse full-R rows are non-finite"
    );
    let payload_bytes = super::encode_f32_le_fallible(&averaged)?;
    super::publish_immutable(&args.output.join(PAYLOAD_NAME), &payload_bytes)?;
    let payload = super::payload_descriptor(PAYLOAD_NAME, &payload_bytes, shape);
    let manifest = MuseRowManifest {
        schema: SHARD_SCHEMA.into(),
        schema_version: SCHEMA_VERSION,
        status: "complete".into(),
        config_blake3: config_blake3.clone(),
        config: fit_config.clone(),
        model: MuseModelSummary {
            path: args.model.display().to_string(),
            locator_blake3: model_locator_blake3,
            identity_scheme: MODEL_IDENTITY_SCHEME.into(),
            content_authenticated: false,
            locator_weight_bytes_hashed: 0,
            architecture: ARCHITECTURE_NAME.into(),
            artifact_profile: artifact::profile_name(profile).into(),
            geometry,
        },
        corpus: MuseCorpusSummary {
            selected_records: prompts.len(),
            used_prompts: active.used_prompts,
            skipped_prompts: active.skipped_prompts.clone(),
            truncated_prompts: active.truncated_prompts,
            ordered_token_ids_blake3: corpus_blake3,
            add_special_tokens,
            max_tokens: args.max_tokens,
        },
        fit: MuseFitSummary {
            estimator_version: ESTIMATOR_VERSION.into(),
            orientation: ORIENTATION.into(),
            rule_contract: RULE_CONTRACT.into(),
            coordinate: COORDINATE.into(),
            replay_semantics: REPLAY_SEMANTICS.into(),
            production_semantics: PRODUCTION_SEMANTICS.into(),
            method: args.method,
            target_layer: args.target_layer,
            source_layers: args.source_layers.clone(),
            row_start: args.row_start,
            row_end: args.row_end,
            dim_batch: args.dim_batch,
            skip_first: args.skip_first,
            valid_position_denominator: "number_of_valid_source_positions".into(),
            prompt_denominator: "number_of_used_prompts".into(),
            accumulator_dtype: "f32".into(),
            storage_dtype: "f32_le".into(),
            forward_seconds: active.forward_seconds,
            vjp_seconds: active.vjp_seconds,
        },
        payload,
        diagnostics: active.diagnostics.clone(),
        provenance: MuseProvenance {
            build_commit: env!("QWEN_BUILD_COMMIT").into(),
            build_dirty: env!("QWEN_BUILD_DIRTY").into(),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
            build_stamp_source: env!("QWEN_BUILD_STAMP_SOURCE").into(),
            build_stamp_error: env!("QWEN_BUILD_STAMP_ERROR").into(),
        },
    };
    validate_complete_manifest(&manifest, &fit_config, &prompts, &config_blake3, shape)?;
    let manifest_bytes =
        super::serialize_json_pretty_bounded(&manifest, "Muse full-R row manifest")?;
    super::publish_immutable(&args.output.join(MANIFEST_NAME), &manifest_bytes)?;
    super::sync_directory(&args.output)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn validate_args(args: &FitRowsArgs) -> Result<()> {
    ensure!(
        args.method == FitMethod::R,
        "Muse full-R rows require --method r"
    );
    ensure!(
        args.dim_batch == MUSE_GLIMMER_FULL_R_MAX_DIM_BATCH,
        "Muse full-R rows require --dim-batch {MUSE_GLIMMER_FULL_R_MAX_DIM_BATCH}"
    );
    ensure!(
        args.max_tokens > 0
            && args.max_tokens <= qwen_llm::muse_glimmer_lens::MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS,
        "Muse --max-tokens exceeds the bounded lens prompt contract"
    );
    let row_count =
        usize::try_from(args.row_end.saturating_sub(args.row_start)).context("Muse row count")?;
    ensure!(
        row_count > 0 && row_count <= MAX_SHARD_ROWS,
        "Muse full-R shards must contain 1..={MAX_SHARD_ROWS} rows"
    );
    Ok(())
}

fn model_locator_digest(
    gguf: &GgufFile,
    config: &qwen_llm::muse_glimmer::MuseGlimmerConfig,
    profile: MuseGlimmerArtifactProfile,
) -> Result<String> {
    let stamps = gguf
        .revalidate_retained_shard_stamps()
        .context("revalidate retained Muse GGUF descriptors")?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"muse-retained-file-stamps-tensor-table-v1\0");
    hasher.update(artifact::profile_name(profile).as_bytes());
    hasher.update(&config.tokenizer_identity_sha256);
    hasher.update(&config.chat_template_sha256);
    hasher.update(&(stamps.len() as u64).to_le_bytes());
    for stamp in stamps {
        hasher.update(&(stamp.shard_idx as u64).to_le_bytes());
        hasher.update(&stamp.device.to_le_bytes());
        hasher.update(&stamp.inode.to_le_bytes());
        hasher.update(&stamp.size.to_le_bytes());
        hasher.update(&stamp.mtime_sec.to_le_bytes());
        hasher.update(&stamp.mtime_nsec.to_le_bytes());
        hasher.update(&stamp.ctime_sec.to_le_bytes());
        hasher.update(&stamp.ctime_nsec.to_le_bytes());
    }
    hasher.update(&(gguf.tensors.len() as u64).to_le_bytes());
    for tensor in &gguf.tensors {
        hash_bytes(&mut hasher, tensor.name.as_bytes());
        hasher.update(&(tensor.shape.len() as u64).to_le_bytes());
        for &dimension in &tensor.shape {
            hasher.update(&dimension.to_le_bytes());
        }
        hasher.update(&(tensor.dtype as i32).to_le_bytes());
        hasher.update(&(tensor.shard_idx as u64).to_le_bytes());
        hasher.update(&tensor.data_offset.to_le_bytes());
        hasher.update(&tensor.n_bytes.to_le_bytes());
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn open_or_create_state(
    output: &Path,
    resume: bool,
    expected_config: &MuseRowConfig,
    prompts: &[PreparedPrompt],
    config_blake3: &str,
    value_count: usize,
    shape: [usize; 3],
) -> Result<WorkerState> {
    super::validate_output_leaf(output)?;
    if output.exists() {
        let metadata = std::fs::symlink_metadata(output)
            .with_context(|| format!("inspect output {}", output.display()))?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "output {} must be a real directory",
            output.display()
        );
        ensure!(
            resume,
            "output {} already exists; pass --resume",
            output.display()
        );
    } else {
        let parent = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        ensure!(
            parent.is_dir(),
            "output parent {} does not exist",
            parent.display()
        );
        DirBuilder::new()
            .mode(0o700)
            .create(output)
            .with_context(|| format!("create output directory {}", output.display()))?;
        super::sync_directory(parent)?;
    }

    let manifest_path = output.join(MANIFEST_NAME);
    if manifest_path.exists() {
        let manifest: MuseRowManifest = super::read_json_file(&manifest_path)?;
        validate_complete_manifest(&manifest, expected_config, prompts, config_blake3, shape)?;
        let bytes = super::verify_payload(output, &manifest.payload)?;
        super::decode_f32_le(&bytes, value_count)?;
        return Ok(WorkerState::Complete(Box::new(manifest)));
    }

    let checkpoint_path = output.join(CHECKPOINT_NAME);
    if !checkpoint_path.exists() {
        return Ok(WorkerState::Active(ActiveState {
            generation: 0,
            next_record: 0,
            used_prompts: 0,
            truncated_prompts: 0,
            skipped_prompts: Vec::new(),
            forward_seconds: 0.0,
            vjp_seconds: 0.0,
            sums: vec![0.0; value_count],
            sums_path: None,
            diagnostics: Vec::new(),
        }));
    }
    let checkpoint: MuseRowCheckpoint = super::read_json_file(&checkpoint_path)?;
    ensure!(
        checkpoint.schema == CHECKPOINT_SCHEMA
            && checkpoint.schema_version == SCHEMA_VERSION
            && checkpoint.config_blake3 == config_blake3,
        "Muse full-R checkpoint contract or config mismatch"
    );
    ensure!(
        checkpoint.sums.shape == shape
            && checkpoint.sums.dtype == "f32_le"
            && checkpoint.sums.path == format!("sums-{:08}.f32le", checkpoint.generation),
        "Muse full-R checkpoint payload descriptor is not canonical"
    );
    let expected_bytes = value_count
        .checked_mul(4)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .context("Muse checkpoint byte count overflow")?;
    ensure!(
        checkpoint.sums.byte_length == expected_bytes,
        "Muse full-R checkpoint byte length mismatch"
    );
    let bytes = super::verify_payload(output, &checkpoint.sums)?;
    let sums = super::decode_f32_le(&bytes, value_count)?;
    Ok(WorkerState::Active(ActiveState {
        generation: checkpoint.generation,
        next_record: checkpoint.next_record,
        used_prompts: checkpoint.used_prompts,
        truncated_prompts: checkpoint.truncated_prompts,
        skipped_prompts: checkpoint.skipped_prompts,
        forward_seconds: checkpoint.forward_seconds,
        vjp_seconds: checkpoint.vjp_seconds,
        sums,
        sums_path: Some(checkpoint.sums.path),
        diagnostics: checkpoint.diagnostics,
    }))
}

fn checkpoint_active(
    output: &Path,
    config_blake3: &str,
    state: &mut ActiveState,
    shape: [usize; 3],
) -> Result<()> {
    state.generation = state
        .generation
        .checked_add(1)
        .context("Muse checkpoint generation overflow")?;
    let sums_name = format!("sums-{:08}.f32le", state.generation);
    let bytes = super::encode_f32_le_fallible(&state.sums)?;
    super::publish_immutable(&output.join(&sums_name), &bytes)?;
    let checkpoint = MuseRowCheckpoint {
        schema: CHECKPOINT_SCHEMA.into(),
        schema_version: SCHEMA_VERSION,
        config_blake3: config_blake3.into(),
        generation: state.generation,
        next_record: state.next_record,
        used_prompts: state.used_prompts,
        truncated_prompts: state.truncated_prompts,
        skipped_prompts: state.skipped_prompts.clone(),
        forward_seconds: state.forward_seconds,
        vjp_seconds: state.vjp_seconds,
        sums: super::payload_descriptor(&sums_name, &bytes, shape),
        diagnostics: state.diagnostics.clone(),
    };
    super::write_atomic_replace(
        &output.join(CHECKPOINT_NAME),
        &super::serialize_json_pretty_bounded(&checkpoint, "Muse full-R checkpoint")?,
    )?;
    if let Some(previous) = state.sums_path.replace(sums_name) {
        let previous_path = output.join(previous);
        if previous_path.exists() {
            std::fs::remove_file(&previous_path)
                .with_context(|| format!("remove prior Muse sums {}", previous_path.display()))?;
            super::sync_directory(output)?;
        }
    }
    Ok(())
}

fn validate_active_state(
    state: &ActiveState,
    prompts: &[PreparedPrompt],
    config: &MuseRowConfig,
    expected_values: usize,
) -> Result<()> {
    ensure!(
        state.next_record <= prompts.len()
            && state.generation == u64::try_from(state.next_record).context("Muse cursor")?,
        "Muse checkpoint cursor or generation is invalid"
    );
    let processed = &prompts[..state.next_record];
    let expected_skipped = processed
        .iter()
        .filter_map(|prompt| super::skipped_prompt(prompt, config.skip_first))
        .collect::<Vec<_>>();
    let expected_used = u64::try_from(state.next_record - expected_skipped.len())
        .context("Muse used prompt count")?;
    let expected_truncated = u64::try_from(
        processed
            .iter()
            .filter(|prompt| {
                super::skipped_prompt(prompt, config.skip_first).is_none() && prompt.truncated
            })
            .count(),
    )
    .context("Muse truncated prompt count")?;
    ensure!(
        state.used_prompts == expected_used
            && state.skipped_prompts == expected_skipped
            && state.truncated_prompts == expected_truncated,
        "Muse checkpoint counters disagree with the corpus prefix"
    );
    ensure!(
        state.forward_seconds.is_finite()
            && state.forward_seconds >= 0.0
            && state.vjp_seconds.is_finite()
            && state.vjp_seconds >= 0.0
            && state.sums.len() == expected_values
            && state.sums.iter().all(|value| value.is_finite()),
        "Muse checkpoint timings or accumulator are invalid"
    );
    if state.used_prompts == 0 {
        ensure!(
            state.forward_seconds == 0.0 && state.vjp_seconds == 0.0,
            "Muse checkpoint without fitted prompts has fit timings"
        );
    }
    validate_diagnostics(&state.diagnostics, state.used_prompts, config.target_layer)
}

fn validate_diagnostics(
    diagnostics: &[MuseReplayDiagnostic],
    used_prompts: u64,
    target_layer: u32,
) -> Result<()> {
    if used_prompts == 0 {
        ensure!(
            diagnostics.is_empty(),
            "Muse checkpoint without fitted prompts has diagnostics"
        );
        return Ok(());
    }
    ensure!(
        diagnostics.len() == 1
            && diagnostics[0].block == target_layer
            && diagnostics[0].kind == "full"
            && diagnostics[0]
                .post_attention_replay_max_abs_error
                .is_finite()
            && diagnostics[0].post_attention_replay_max_abs_error >= 0.0
            && diagnostics[0].post_block_replay_max_abs_error.is_finite()
            && diagnostics[0].post_block_replay_max_abs_error >= 0.0,
        "Muse replay diagnostic is invalid"
    );
    Ok(())
}

fn merge_diagnostic(
    aggregate: &mut Vec<MuseReplayDiagnostic>,
    current: MuseReplayDiagnostic,
) -> Result<()> {
    if aggregate.is_empty() {
        aggregate.push(current);
        return Ok(());
    }
    ensure!(
        aggregate.len() == 1
            && aggregate[0].block == current.block
            && aggregate[0].kind == current.kind,
        "Muse replay diagnostic schedule changed"
    );
    aggregate[0].post_attention_replay_max_abs_error = aggregate[0]
        .post_attention_replay_max_abs_error
        .max(current.post_attention_replay_max_abs_error);
    aggregate[0].post_block_replay_max_abs_error = aggregate[0]
        .post_block_replay_max_abs_error
        .max(current.post_block_replay_max_abs_error);
    Ok(())
}

fn validate_complete_manifest(
    manifest: &MuseRowManifest,
    expected_config: &MuseRowConfig,
    prompts: &[PreparedPrompt],
    expected_config_blake3: &str,
    expected_shape: [usize; 3],
) -> Result<()> {
    ensure!(
        manifest.schema == SHARD_SCHEMA
            && manifest.schema_version == SCHEMA_VERSION
            && manifest.status == "complete"
            && &manifest.config == expected_config,
        "completed Muse full-R shard contract or config mismatch"
    );
    ensure!(
        manifest.config_blake3 == super::digest_json(&manifest.config)?
            && manifest.config_blake3 == expected_config_blake3,
        "completed Muse full-R config digest mismatch"
    );
    ensure!(
        manifest.payload.path == PAYLOAD_NAME
            && manifest.payload.dtype == "f32_le"
            && manifest.payload.shape == expected_shape,
        "completed Muse full-R payload descriptor is not canonical"
    );
    let expected_bytes = expected_shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
        .and_then(|values| values.checked_mul(4))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .context("completed Muse full-R byte count overflow")?;
    ensure!(
        manifest.payload.byte_length == expected_bytes,
        "completed Muse full-R payload byte length mismatch"
    );
    ensure!(
        manifest.model.locator_blake3 == expected_config.model_locator_blake3
            && manifest.model.identity_scheme == MODEL_IDENTITY_SCHEME
            && !manifest.model.content_authenticated
            && manifest.model.locator_weight_bytes_hashed == 0
            && manifest.model.architecture == expected_config.architecture
            && manifest.model.artifact_profile == expected_config.artifact_profile
            && manifest.model.geometry == expected_config.geometry,
        "completed Muse full-R model summary disagrees with its config"
    );
    let expected_skipped = prompts
        .iter()
        .filter_map(|prompt| super::skipped_prompt(prompt, expected_config.skip_first))
        .collect::<Vec<_>>();
    let expected_used = u64::try_from(prompts.len() - expected_skipped.len())
        .context("completed Muse used prompt count")?;
    let expected_truncated = u64::try_from(
        prompts
            .iter()
            .filter(|prompt| {
                super::skipped_prompt(prompt, expected_config.skip_first).is_none()
                    && prompt.truncated
            })
            .count(),
    )
    .context("completed Muse truncated prompt count")?;
    ensure!(
        manifest.corpus.selected_records == expected_config.selected_records
            && manifest.corpus.used_prompts == expected_used
            && manifest.corpus.skipped_prompts == expected_skipped
            && manifest.corpus.truncated_prompts == expected_truncated
            && manifest.corpus.ordered_token_ids_blake3 == expected_config.corpus_blake3
            && manifest.corpus.add_special_tokens == expected_config.add_special_tokens
            && manifest.corpus.max_tokens == expected_config.max_tokens,
        "completed Muse full-R corpus summary disagrees with its config"
    );
    ensure!(
        manifest.fit.estimator_version == expected_config.estimator_version
            && manifest.fit.orientation == expected_config.orientation
            && manifest.fit.rule_contract == expected_config.rule_contract
            && manifest.fit.coordinate == expected_config.coordinate
            && manifest.fit.replay_semantics == expected_config.replay_semantics
            && manifest.fit.production_semantics == expected_config.production_semantics
            && manifest.fit.method == FitMethod::R
            && manifest.fit.target_layer == expected_config.target_layer
            && manifest.fit.source_layers == expected_config.source_layers
            && manifest.fit.row_start == expected_config.row_start
            && manifest.fit.row_end == expected_config.row_end
            && manifest.fit.dim_batch == MUSE_GLIMMER_FULL_R_MAX_DIM_BATCH
            && manifest.fit.skip_first == expected_config.skip_first
            && manifest.fit.valid_position_denominator == "number_of_valid_source_positions"
            && manifest.fit.prompt_denominator == "number_of_used_prompts"
            && manifest.fit.accumulator_dtype == "f32"
            && manifest.fit.storage_dtype == "f32_le"
            && manifest.fit.forward_seconds.is_finite()
            && manifest.fit.forward_seconds >= 0.0
            && manifest.fit.vjp_seconds.is_finite()
            && manifest.fit.vjp_seconds >= 0.0,
        "completed Muse full-R fit summary disagrees with its config"
    );
    validate_diagnostics(
        &manifest.diagnostics,
        manifest.corpus.used_prompts,
        expected_config.target_layer,
    )?;
    ensure!(
        !manifest.provenance.build_commit.is_empty()
            && !manifest.provenance.build_dirty.is_empty()
            && manifest.provenance.build_source_state == expected_config.build_source_state
            && manifest.provenance.build_stamp_source == env!("QWEN_BUILD_STAMP_SOURCE")
            && manifest.provenance.build_stamp_error == env!("QWEN_BUILD_STAMP_ERROR"),
        "completed Muse full-R provenance is incomplete"
    );
    super::validate_token_build_identity(
        &manifest.provenance.build_source_state,
        &manifest.provenance.build_stamp_error,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn args() -> FitRowsArgs {
        FitRowsArgs {
            model: PathBuf::from("m.gguf"),
            prompts: PathBuf::from("p.jsonl"),
            output: PathBuf::from("out"),
            identity_cache: PathBuf::from("unused-for-muse"),
            method: FitMethod::R,
            target_layer: 51,
            source_layers: vec![50],
            row_start: 0,
            row_end: 256,
            dim_batch: 32,
            skip_first: 0,
            max_tokens: 16,
            max_prompts: 25,
            no_special_tokens: false,
            resume: false,
        }
    }

    fn row_config() -> MuseRowConfig {
        MuseRowConfig {
            estimator_version: ESTIMATOR_VERSION.into(),
            orientation: ORIENTATION.into(),
            rule_contract: RULE_CONTRACT.into(),
            coordinate: COORDINATE.into(),
            replay_semantics: REPLAY_SEMANTICS.into(),
            production_semantics: PRODUCTION_SEMANTICS.into(),
            method: FitMethod::R,
            model_locator_blake3: "11".repeat(32),
            model_identity_scheme: MODEL_IDENTITY_SCHEME.into(),
            architecture: ARCHITECTURE_NAME.into(),
            artifact_profile: "unsloth_q8_0".into(),
            geometry: artifact::Geometry {
                layer_count: 52,
                hidden_size: 2,
                vocab_size: 8,
                query_heads: 1,
                kv_heads: 1,
                head_dim: 2,
                sliding_window: 4,
                sliding_layers: vec![false; 52],
            },
            tokenizer_identity_sha256: "22".repeat(32),
            chat_template_sha256: "33".repeat(32),
            target_layer: 51,
            source_layers: vec![50],
            row_start: 0,
            row_end: 2,
            skip_first: 0,
            max_tokens: 2,
            add_special_tokens: false,
            corpus_blake3: "44".repeat(32),
            selected_records: 1,
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
        }
    }

    #[test]
    fn first_full_r_contract_rejects_unqualified_shapes() {
        validate_args(&args()).unwrap();
        let mut value = args();
        value.method = FitMethod::J;
        assert!(validate_args(&value).is_err());
        let mut value = args();
        value.dim_batch = 8;
        assert!(validate_args(&value).is_err());
        let mut value = args();
        value.row_end = 257;
        assert!(validate_args(&value).is_err());
    }

    #[test]
    fn replay_diagnostics_merge_only_matching_full_block() {
        let mut diagnostics = Vec::new();
        merge_diagnostic(
            &mut diagnostics,
            MuseReplayDiagnostic {
                block: 51,
                kind: "full".into(),
                post_attention_replay_max_abs_error: 0.1,
                post_block_replay_max_abs_error: 0.2,
            },
        )
        .unwrap();
        merge_diagnostic(
            &mut diagnostics,
            MuseReplayDiagnostic {
                block: 51,
                kind: "full".into(),
                post_attention_replay_max_abs_error: 0.3,
                post_block_replay_max_abs_error: 0.15,
            },
        )
        .unwrap();
        assert_eq!(diagnostics[0].post_attention_replay_max_abs_error, 0.3);
        assert_eq!(diagnostics[0].post_block_replay_max_abs_error, 0.2);
        assert!(validate_diagnostics(&diagnostics, 2, 51).is_ok());
    }

    #[test]
    fn checkpoint_round_trip_preserves_cursor_sums_and_diagnostics() {
        let root = std::env::temp_dir().join(format!(
            "qwen-muse-row-checkpoint-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        let config = row_config();
        let config_blake3 = super::super::digest_json(&config).unwrap();
        let prompts = [PreparedPrompt {
            id: "p0".into(),
            token_ids: vec![1, 2],
            original_token_count: 2,
            truncated: false,
        }];
        let diagnostics = vec![MuseReplayDiagnostic {
            block: 51,
            kind: "full".into(),
            post_attention_replay_max_abs_error: 0.1,
            post_block_replay_max_abs_error: 0.2,
        }];
        let mut active = ActiveState {
            generation: 0,
            next_record: 1,
            used_prompts: 1,
            truncated_prompts: 0,
            skipped_prompts: Vec::new(),
            forward_seconds: 0.5,
            vjp_seconds: 1.5,
            sums: vec![1.0, 2.0, 3.0, 4.0],
            sums_path: None,
            diagnostics: diagnostics.clone(),
        };
        checkpoint_active(&root, &config_blake3, &mut active, [1, 2, 2]).unwrap();
        let WorkerState::Active(restored) =
            open_or_create_state(&root, true, &config, &prompts, &config_blake3, 4, [1, 2, 2])
                .unwrap()
        else {
            panic!("expected active Muse row checkpoint");
        };
        assert_eq!(restored.generation, 1);
        assert_eq!(restored.next_record, 1);
        assert_eq!(restored.sums, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(restored.diagnostics, diagnostics);
        validate_active_state(&restored, &prompts, &config, 4).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }
}
