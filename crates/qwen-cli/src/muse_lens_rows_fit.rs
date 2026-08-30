use super::muse_lens_artifact;
use super::muse_lens_rows_artifact as artifact;
use super::{FitRowsArgs, PreparedPrompt};
use anyhow::{Context, Result, ensure};
use qwen_llm::checkpoint_identity::{
    CheckpointIdentityCache, checkpoint_content_identity_without_weight_hashing,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::muse_glimmer::MuseGlimmerModel;
use qwen_llm::muse_glimmer_lens::MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS;
use qwen_llm::muse_glimmer_lens_fit::{
    MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD, MUSE_GLIMMER_QUERY_BATCH_MAX,
    MuseGlimmerAttentionBlockKind, MuseGlimmerBlockReplayDiagnostic,
    MuseGlimmerQueryBatchVjpTimings,
};
use qwen_llm::muse_glimmer_runtime::MuseGlimmerLoadedModel;
use qwen_llm::tokenizer::LlamaCppTokenizer;
use std::fs::DirBuilder;
use std::os::unix::fs::DirBuilderExt;
use std::path::Path;
use std::time::Instant;

#[derive(Debug)]
struct ActiveState {
    generation: u64,
    next_record: usize,
    used_prompts: u64,
    truncated_prompts: u64,
    skipped_prompts: Vec<super::SkippedPrompt>,
    forward_seconds: f64,
    vjp_wall_seconds: f64,
    vjp_timings: artifact::VjpTimings,
    sums: Vec<f32>,
    sums_path: Option<String>,
    diagnostics: Vec<artifact::ReplayDiagnostic>,
}

#[derive(Debug)]
enum WorkerState {
    Active(ActiveState),
    Complete(Box<artifact::Manifest>),
}

pub(crate) fn fit_rows(mut args: FitRowsArgs, gguf: GgufFile) -> Result<()> {
    validate_args(&args)?;
    super::validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;

    let bound = MuseGlimmerModel::from_gguf(&gguf)
        .context("bind Muse Glimmer artifact profile and geometry")?;
    let model_config = bound.config.clone();
    let profile = bound.artifact_profile;
    drop(bound);
    ensure!(
        args.target_layer < model_config.layer_count,
        "--target-layer {} is outside Muse layer count {}",
        args.target_layer,
        model_config.layer_count
    );
    ensure!(
        args.row_end <= model_config.hidden_size,
        "row range {}..{} exceeds Muse hidden size {}",
        args.row_start,
        args.row_end,
        model_config.hidden_size
    );

    args.output = super::resolve_output_path(&args.output)?;
    let requests = super::read_prompt_requests(&args.prompts, args.max_prompts)?;
    let tokenizer = LlamaCppTokenizer::from_gguf(&gguf, &args.model)
        .context("load Muse llama.cpp tokenizer")?;
    muse_lens_artifact::validate_tokenizer(&tokenizer, &model_config)?;
    let add_special_tokens = !args.no_special_tokens;
    let prompts = super::prepare_prompts(
        requests,
        &tokenizer,
        add_special_tokens,
        args.max_tokens,
        model_config.vocab_size,
    )?;
    let corpus_blake3 = super::corpus_digest(&prompts);
    let content = checkpoint_content_identity_without_weight_hashing(
        &gguf,
        &CheckpointIdentityCache::new(&args.identity_cache),
    )
    .with_context(|| {
        format!(
            "resolve Muse GGUF identity without hashing weights using {}",
            args.identity_cache.display()
        )
    })?;
    ensure!(
        content.bytes_hashed == 0,
        "Muse row fitting must not hash model weights"
    );
    let content_id = super::hex(&content.content_id);
    let config = artifact::make_config(
        &model_config,
        profile,
        content_id.clone(),
        args.method,
        args.target_layer,
        args.source_layers.clone(),
        args.row_start,
        args.row_end,
        args.skip_first,
        args.dim_batch,
        args.max_tokens,
        add_special_tokens,
        corpus_blake3.clone(),
        prompts.len(),
        env!("QWEN_BUILD_SOURCE_STATE").into(),
    );
    artifact::validate_config(&config, &model_config, profile, &content_id)?;
    let config_blake3 = super::digest_json(&config)?;
    let shape = artifact::expected_shape(&config)?;
    let value_count = shape
        .into_iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(dimension))
        .context("Muse row-shard value count overflow")?;

    let mut state = open_or_create_state(
        &args.output,
        args.resume,
        &config,
        &prompts,
        &config_blake3,
        shape,
        value_count,
    )?;
    if let WorkerState::Complete(manifest) = state {
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }
    let WorkerState::Active(ref mut active) = state else {
        unreachable!();
    };
    validate_active_state(active, &prompts, &config, value_count)?;

    let context = MetalContext::new().context("initialize Metal for Muse row fitting")?;
    let mut loaded = MuseGlimmerLoadedModel::load(&context, &gguf, args.max_tokens)
        .context("load Muse Glimmer row-fitting model")?;
    let mut runner = loaded
        .create_runner(&context)
        .context("create Muse row-fitting runner")?;
    let output_rows = (args.row_start..args.row_end).collect::<Vec<_>>();
    let traversed_blocks = ((args.source_layers[0] + 1)..=args.target_layer).collect::<Vec<_>>();
    let rule = artifact::rule(args.method);

    let invocation_start = active.next_record;
    let record_range = fit_record_range(active.next_record, prompts.len(), args.records_this_run)?;
    for record_index in record_range {
        let prompt = &prompts[record_index];
        if let Some(skipped) = super::skipped_prompt(prompt, args.skip_first) {
            active.skipped_prompts.push(skipped);
            active.next_record = record_index + 1;
            checkpoint_active(
                &args.output,
                &config_blake3,
                active,
                &prompts,
                &config,
                shape,
            )?;
            continue;
        }
        eprintln!(
            "fit Muse row shard prompt {}/{} id={} tokens={} rows={}..{} query_batch={} sources={}",
            record_index + 1,
            prompts.len(),
            prompt.id,
            prompt.token_ids.len(),
            args.row_start,
            args.row_end,
            args.dim_batch,
            args.source_layers.len(),
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
        let captures = runner
            .capture_fresh_lens_prompt_blocks(&tokens, &traversed_blocks)
            .with_context(|| format!("capture Muse row-fit prompt {}", prompt.id))?;
        active.forward_seconds += started.elapsed().as_secs_f64();
        ensure!(
            artifact::valid_seconds(active.forward_seconds),
            "Muse row-fit forward timing overflow"
        );

        let started = Instant::now();
        let fit = runner
            .fit_full_transport_rows_to_sources_batched(
                &captures,
                args.target_layer,
                &args.source_layers,
                &output_rows,
                args.skip_first,
                args.dim_batch,
                rule,
            )
            .with_context(|| format!("fit Muse transport rows for prompt {}", prompt.id))?;
        active.vjp_wall_seconds += started.elapsed().as_secs_f64();
        ensure!(
            artifact::valid_seconds(active.vjp_wall_seconds),
            "Muse row-fit VJP wall timing overflow"
        );
        let expected_valid_positions = prompt.token_ids.len() - args.skip_first - 1;
        ensure!(
            fit.source_layers == args.source_layers
                && fit.target_block == args.target_layer
                && fit.method == rule
                && fit.output_row_ids == output_rows
                && fit.n_valid_positions == expected_valid_positions
                && fit.hidden_size == model_config.hidden_size as usize
                && fit.query_batch_size == args.dim_batch
                && fit.values.len() == active.sums.len()
                && fit.values.iter().all(|value| value.is_finite()),
            "Muse row fit returned inconsistent metadata, shape, or values"
        );
        for (sum, value) in active.sums.iter_mut().zip(fit.values) {
            *sum += value;
        }
        ensure!(
            active.sums.iter().all(|value| value.is_finite()),
            "Muse row-fit accumulator became non-finite"
        );
        merge_diagnostics(&mut active.diagnostics, &fit.diagnostics)?;
        let timings = vjp_timings(&fit.timings);
        ensure!(timings.is_valid(), "Muse row fit returned invalid timings");
        active.vjp_timings.add_assign(&timings);
        ensure!(
            active.vjp_timings.fits_within_wall(active.vjp_wall_seconds),
            "Muse row-fit timing accumulator overflow"
        );
        active.used_prompts = active
            .used_prompts
            .checked_add(1)
            .context("Muse used prompt count overflow")?;
        active.truncated_prompts = active
            .truncated_prompts
            .checked_add(u64::from(prompt.truncated))
            .context("Muse truncated prompt count overflow")?;
        active.next_record = record_index + 1;
        checkpoint_active(
            &args.output,
            &config_blake3,
            active,
            &prompts,
            &config,
            shape,
        )?;
    }
    if active.next_record < prompts.len() {
        eprintln!(
            "paused Muse row fit at record {}/{} after {} record(s) this invocation; checkpoint generation {} is resumable with --resume",
            active.next_record,
            prompts.len(),
            active.next_record - invocation_start,
            active.generation,
        );
        return Ok(());
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
        "averaged Muse full-transport rows are non-finite"
    );
    let payload_bytes = super::encode_f32_le_fallible(&averaged)?;
    let payload = artifact::payload(artifact::PAYLOAD_NAME, &payload_bytes, shape);
    super::publish_immutable(&args.output.join(artifact::PAYLOAD_NAME), &payload_bytes)?;

    let manifest = artifact::Manifest {
        schema: artifact::SCHEMA.into(),
        schema_version: artifact::SCHEMA_VERSION,
        status: "complete".into(),
        config_blake3: config_blake3.clone(),
        config: config.clone(),
        model: artifact::ModelSummary {
            path: args.model.display().to_string(),
            architecture: config.architecture.clone(),
            artifact_profile: config.artifact_profile.clone(),
            content_blake3: content_id,
            content_identity_outcome: format!("{:?}", content.outcome),
            content_bytes_hashed: content.bytes_hashed,
            content_authenticated: true,
        },
        corpus: artifact::CorpusSummary {
            selected_records: prompts.len(),
            used_prompts: active.used_prompts,
            skipped_prompts: active.skipped_prompts.clone(),
            truncated_prompts: active.truncated_prompts,
            ordered_token_ids_blake3: corpus_blake3,
            add_special_tokens,
            max_tokens: args.max_tokens,
            prompt_reduction: artifact::PROMPT_REDUCTION.into(),
        },
        fit: artifact::FitSummary {
            method: config.method.clone(),
            rule_contract: config.rule_contract.clone(),
            target_layer: args.target_layer,
            source_layers: args.source_layers.clone(),
            coordinate: config.coordinate.clone(),
            estimator: config.estimator.clone(),
            orientation: config.orientation.clone(),
            reduction: config.reduction.clone(),
            row_start: args.row_start,
            row_end: args.row_end,
            skip_first: args.skip_first,
            query_batch_size: args.dim_batch,
            valid_position_denominator: "number_of_valid_source_positions".into(),
            accumulator_dtype: "f32".into(),
            storage_dtype: "f32_le".into(),
            forward_seconds: active.forward_seconds,
            vjp_wall_seconds: active.vjp_wall_seconds,
            vjp_timings: active.vjp_timings.clone(),
        },
        payload,
        diagnostics: active.diagnostics.clone(),
        provenance: artifact::Provenance {
            build_commit: env!("QWEN_BUILD_COMMIT").into(),
            build_dirty: env!("QWEN_BUILD_DIRTY").into(),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
            build_stamp_source: env!("QWEN_BUILD_STAMP_SOURCE").into(),
            build_stamp_error: env!("QWEN_BUILD_STAMP_ERROR").into(),
        },
    };
    artifact::validate_complete(&manifest, &config, &prompts, &config_blake3)?;
    super::publish_immutable(
        &args.output.join(artifact::MANIFEST_NAME),
        &super::serialize_json_pretty_bounded(&manifest, "Muse row-shard manifest")?,
    )?;
    super::sync_directory(&args.output)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn validate_args(args: &FitRowsArgs) -> Result<()> {
    ensure!(
        args.max_prompts > 0 && args.max_prompts <= super::MAX_PROMPT_RECORDS,
        "Muse --max-prompts must be in 1..={}",
        super::MAX_PROMPT_RECORDS
    );
    validate_records_this_run(args.records_this_run)?;
    ensure!(
        args.dim_batch > 0 && args.dim_batch <= MUSE_GLIMMER_QUERY_BATCH_MAX,
        "Muse --dim-batch must be in 1..={MUSE_GLIMMER_QUERY_BATCH_MAX}"
    );
    ensure!(
        args.max_tokens > 0 && args.max_tokens <= MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS,
        "Muse --max-tokens must be in 1..={MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS}"
    );
    ensure!(
        args.skip_first
            .checked_add(2)
            .is_some_and(|minimum| minimum <= args.max_tokens),
        "Muse --max-tokens must be at least --skip-first + 2"
    );
    ensure!(
        args.target_layer > 0,
        "Muse row fitting requires a nonzero --target-layer"
    );
    ensure!(
        !args.source_layers.is_empty()
            && args.source_layers.windows(2).all(|pair| pair[0] < pair[1])
            && args
                .source_layers
                .iter()
                .all(|&source| source < args.target_layer),
        "Muse --source-layers must be nonempty, strictly increasing, and below --target-layer"
    );
    let row_count = args
        .row_end
        .checked_sub(args.row_start)
        .context("Muse row range must be nonempty and half-open")? as usize;
    ensure!(
        row_count > 0 && row_count <= MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD,
        "Muse row shard must contain 1..={MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD} rows"
    );
    ensure!(
        args.prompts != Path::new("-"),
        "Muse fitting requires a replayable prompt corpus file"
    );
    Ok(())
}

fn open_or_create_state(
    output: &Path,
    resume: bool,
    config: &artifact::Config,
    prompts: &[PreparedPrompt],
    config_blake3: &str,
    shape: [usize; 3],
    value_count: usize,
) -> Result<WorkerState> {
    super::validate_output_leaf(output)?;
    if output.exists() {
        let metadata = std::fs::symlink_metadata(output)
            .with_context(|| format!("inspect Muse row output {}", output.display()))?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "Muse row output {} must be a real directory",
            output.display()
        );
        ensure!(
            resume,
            "Muse row output {} already exists; pass --resume",
            output.display()
        );
    } else {
        let parent = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        ensure!(
            parent.is_dir(),
            "Muse row output parent {} does not exist",
            parent.display()
        );
        DirBuilder::new()
            .mode(0o700)
            .create(output)
            .with_context(|| format!("create Muse row output {}", output.display()))?;
        super::sync_directory(parent)?;
    }

    let manifest_path = output.join(artifact::MANIFEST_NAME);
    if manifest_path.exists() {
        let manifest: artifact::Manifest = super::read_json_file(&manifest_path)?;
        artifact::validate_complete(&manifest, config, prompts, config_blake3)?;
        let bytes = verify_payload(output, &manifest.payload)?;
        super::decode_f32_le(&bytes, value_count)?;
        return Ok(WorkerState::Complete(Box::new(manifest)));
    }
    ensure!(
        !output.join(muse_lens_artifact::MANIFEST_NAME).exists(),
        "output contains a Muse selected-token artifact, not a full-transport row shard"
    );

    let checkpoint_path = output.join(artifact::CHECKPOINT_NAME);
    if !checkpoint_path.exists() {
        return Ok(WorkerState::Active(ActiveState {
            generation: 0,
            next_record: 0,
            used_prompts: 0,
            truncated_prompts: 0,
            skipped_prompts: Vec::new(),
            forward_seconds: 0.0,
            vjp_wall_seconds: 0.0,
            vjp_timings: artifact::VjpTimings::default(),
            sums: super::try_zeroed_f32(value_count, "Muse row-shard accumulator")?,
            sums_path: None,
            diagnostics: Vec::new(),
        }));
    }

    let checkpoint: artifact::Checkpoint = super::read_json_file(&checkpoint_path)?;
    ensure!(
        checkpoint.schema == artifact::CHECKPOINT_SCHEMA
            && checkpoint.schema_version == artifact::SCHEMA_VERSION,
        "unknown or unsupported Muse row-shard checkpoint schema"
    );
    ensure!(
        checkpoint.config_blake3 == config_blake3,
        "Muse row-shard checkpoint config differs from the requested fit"
    );
    ensure!(
        checkpoint.sums.path == format!("muse-row-sums-{:08}.f32le", checkpoint.generation)
            && checkpoint.sums.dtype == "f32_le"
            && checkpoint.sums.shape == shape,
        "Muse row-shard checkpoint payload descriptor is not canonical"
    );
    let expected_bytes = value_count
        .checked_mul(4)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .context("Muse row-shard checkpoint byte count overflow")?;
    ensure!(
        checkpoint.sums.byte_length == expected_bytes,
        "Muse row-shard checkpoint byte count is inconsistent"
    );
    let bytes = verify_payload(output, &checkpoint.sums)?;
    let sums = super::decode_f32_le(&bytes, value_count)?;
    let state = ActiveState {
        generation: checkpoint.generation,
        next_record: checkpoint.next_record,
        used_prompts: checkpoint.used_prompts,
        truncated_prompts: checkpoint.truncated_prompts,
        skipped_prompts: checkpoint.skipped_prompts,
        forward_seconds: checkpoint.forward_seconds,
        vjp_wall_seconds: checkpoint.vjp_wall_seconds,
        vjp_timings: checkpoint.vjp_timings,
        sums,
        sums_path: Some(checkpoint.sums.path),
        diagnostics: checkpoint.diagnostics,
    };
    validate_active_state(&state, prompts, config, value_count)?;
    Ok(WorkerState::Active(state))
}

fn validate_active_state(
    state: &ActiveState,
    prompts: &[PreparedPrompt],
    config: &artifact::Config,
    expected_values: usize,
) -> Result<()> {
    ensure!(
        state.next_record <= prompts.len()
            && state.generation
                == u64::try_from(state.next_record).context("Muse row-shard checkpoint cursor")?,
        "Muse row-shard checkpoint generation or cursor is invalid"
    );
    let processed = &prompts[..state.next_record];
    let expected_skipped = processed
        .iter()
        .filter_map(|prompt| super::skipped_prompt(prompt, config.skip_first))
        .collect::<Vec<_>>();
    let expected_used = u64::try_from(processed.len() - expected_skipped.len())
        .context("Muse row-shard checkpoint used prompts")?;
    let expected_truncated = u64::try_from(
        processed
            .iter()
            .filter(|prompt| {
                super::skipped_prompt(prompt, config.skip_first).is_none() && prompt.truncated
            })
            .count(),
    )
    .context("Muse row-shard checkpoint truncated prompts")?;
    ensure!(
        state.used_prompts == expected_used
            && state.skipped_prompts == expected_skipped
            && state.truncated_prompts == expected_truncated,
        "Muse row-shard checkpoint counters disagree with the corpus prefix"
    );
    ensure!(
        artifact::valid_seconds(state.forward_seconds)
            && artifact::valid_seconds(state.vjp_wall_seconds)
            && state.vjp_timings.fits_within_wall(state.vjp_wall_seconds),
        "Muse row-shard checkpoint timings are invalid"
    );
    ensure!(
        state.sums.len() == expected_values && state.sums.iter().all(|value| value.is_finite()),
        "Muse row-shard checkpoint accumulator is malformed"
    );
    if state.used_prompts == 0 {
        ensure!(
            state.diagnostics.is_empty()
                && state.forward_seconds == 0.0
                && state.vjp_wall_seconds == 0.0
                && state.vjp_timings.is_zero(),
            "Muse row-shard checkpoint without fits has diagnostics or timings"
        );
    } else {
        artifact::validate_diagnostics(&state.diagnostics, config)?;
    }
    Ok(())
}

fn checkpoint_active(
    output: &Path,
    config_blake3: &str,
    state: &mut ActiveState,
    prompts: &[PreparedPrompt],
    config: &artifact::Config,
    shape: [usize; 3],
) -> Result<()> {
    let next_generation = state
        .generation
        .checked_add(1)
        .context("Muse row-shard checkpoint generation overflow")?;
    let next_record = u64::try_from(state.next_record)
        .context("Muse row-shard checkpoint cursor does not fit u64")?;
    ensure!(
        next_generation == next_record,
        "Muse row-shard checkpoint cursor did not advance exactly once"
    );
    state.generation = next_generation;
    let expected_values = shape
        .into_iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(dimension))
        .context("Muse row-shard checkpoint value count overflow")?;
    validate_active_state(state, prompts, config, expected_values)?;
    let sums_name = format!("muse-row-sums-{:08}.f32le", state.generation);
    ensure!(
        state.sums_path.as_deref() != Some(&sums_name),
        "Muse row-shard next generation is already committed"
    );
    let sums_path = output.join(&sums_name);
    remove_uncommitted_generation(&sums_path)?;
    let bytes = super::encode_f32_le_fallible(&state.sums)?;
    super::publish_immutable(&sums_path, &bytes)?;
    let checkpoint = artifact::Checkpoint {
        schema: artifact::CHECKPOINT_SCHEMA.into(),
        schema_version: artifact::SCHEMA_VERSION,
        config_blake3: config_blake3.into(),
        generation: state.generation,
        next_record: state.next_record,
        used_prompts: state.used_prompts,
        truncated_prompts: state.truncated_prompts,
        skipped_prompts: state.skipped_prompts.clone(),
        forward_seconds: state.forward_seconds,
        vjp_wall_seconds: state.vjp_wall_seconds,
        vjp_timings: state.vjp_timings.clone(),
        sums: artifact::payload(&sums_name, &bytes, shape),
        diagnostics: state.diagnostics.clone(),
    };
    super::write_atomic_replace(
        &output.join(artifact::CHECKPOINT_NAME),
        &super::serialize_json_pretty_bounded(&checkpoint, "Muse row-shard checkpoint")?,
    )?;
    if let Some(previous) = state.sums_path.replace(sums_name) {
        let previous_path = output.join(previous);
        if previous_path.exists() {
            std::fs::remove_file(&previous_path).with_context(|| {
                format!("remove prior Muse row sums {}", previous_path.display())
            })?;
            super::sync_directory(output)?;
        }
    }
    Ok(())
}

fn merge_diagnostics(
    aggregate: &mut Vec<artifact::ReplayDiagnostic>,
    current: &[MuseGlimmerBlockReplayDiagnostic],
) -> Result<()> {
    ensure!(
        current.iter().all(|diagnostic| {
            diagnostic.post_attention_replay_max_abs_error.is_finite()
                && diagnostic.post_attention_replay_max_abs_error >= 0.0
                && diagnostic.post_block_replay_max_abs_error.is_finite()
                && diagnostic.post_block_replay_max_abs_error >= 0.0
        }),
        "Muse row-shard replay diagnostics contain non-finite or negative values"
    );
    if aggregate.is_empty() {
        aggregate.extend(current.iter().map(|diagnostic| artifact::ReplayDiagnostic {
            block: diagnostic.block,
            kind: block_kind(diagnostic.kind).into(),
            post_attention_replay_max_abs_error: diagnostic.post_attention_replay_max_abs_error,
            post_block_replay_max_abs_error: diagnostic.post_block_replay_max_abs_error,
        }));
        return Ok(());
    }
    ensure!(
        aggregate.len() == current.len(),
        "Muse row-shard replay diagnostic schedule length changed"
    );
    for (aggregate, current) in aggregate.iter_mut().zip(current) {
        ensure!(
            aggregate.block == current.block && aggregate.kind == block_kind(current.kind),
            "Muse row-shard replay diagnostic schedule changed at block {}",
            current.block
        );
        aggregate.post_attention_replay_max_abs_error = aggregate
            .post_attention_replay_max_abs_error
            .max(current.post_attention_replay_max_abs_error);
        aggregate.post_block_replay_max_abs_error = aggregate
            .post_block_replay_max_abs_error
            .max(current.post_block_replay_max_abs_error);
    }
    Ok(())
}

fn remove_uncommitted_generation(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                "uncommitted Muse row generation {} must be a regular non-symlink file",
                path.display()
            );
            std::fs::remove_file(path).with_context(|| {
                format!("remove uncommitted Muse row generation {}", path.display())
            })?;
            super::sync_directory(path.parent().unwrap_or_else(|| Path::new(".")))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspect Muse row generation {}", path.display()));
        }
    }
    Ok(())
}

fn validate_records_this_run(records_this_run: Option<usize>) -> Result<()> {
    ensure!(
        records_this_run != Some(0),
        "--records-this-run must be nonzero"
    );
    Ok(())
}

fn fit_record_range(
    next_record: usize,
    total_records: usize,
    records_this_run: Option<usize>,
) -> Result<std::ops::Range<usize>> {
    ensure!(
        next_record <= total_records,
        "Muse row-fit checkpoint cursor exceeds selected corpus"
    );
    validate_records_this_run(records_this_run)?;
    let end = records_this_run
        .map(|count| next_record.saturating_add(count).min(total_records))
        .unwrap_or(total_records);
    Ok(next_record..end)
}

fn block_kind(kind: MuseGlimmerAttentionBlockKind) -> &'static str {
    match kind {
        MuseGlimmerAttentionBlockKind::Full => "full",
        MuseGlimmerAttentionBlockKind::Sliding => "sliding",
    }
}

fn vjp_timings(timings: &MuseGlimmerQueryBatchVjpTimings) -> artifact::VjpTimings {
    artifact::VjpTimings {
        replay_seconds: timings.replay.as_secs_f64(),
        full_attention_bank_command_seconds: timings.full_attention_bank_command.as_secs_f64(),
        sliding_attention_bank_command_seconds: timings
            .sliding_attention_bank_command
            .as_secs_f64(),
        feed_forward_reverse_seconds: timings.feed_forward_reverse.as_secs_f64(),
        attention_output_reverse_seconds: timings.attention_output_reverse.as_secs_f64(),
        attention_cpu_reverse_seconds: timings.attention_cpu_reverse.as_secs_f64(),
        attention_input_reverse_seconds: timings.attention_input_reverse.as_secs_f64(),
        total_seconds: timings.total.as_secs_f64(),
    }
}

fn verify_payload(directory: &Path, descriptor: &artifact::Payload) -> Result<Vec<u8>> {
    ensure!(
        Path::new(&descriptor.path).components().count() == 1,
        "Muse row-shard payload path must be one relative filename"
    );
    let length = usize::try_from(descriptor.byte_length)
        .context("Muse row-shard payload length does not fit this platform")?;
    let path = directory.join(&descriptor.path);
    let bytes = super::read_regular_file_exact(&path, length)?;
    ensure!(
        blake3::hash(&bytes).to_hex().as_str() == descriptor.blake3,
        "Muse row-shard payload {} digest mismatch",
        path.display()
    );
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::super::FitMethod;
    use super::*;
    use qwen_llm::muse_glimmer::{MuseGlimmerArtifactProfile, MuseGlimmerConfig};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn args() -> FitRowsArgs {
        FitRowsArgs {
            model: PathBuf::from("model.gguf"),
            prompts: PathBuf::from("prompts.jsonl"),
            output: PathBuf::from("rows"),
            identity_cache: PathBuf::from("identity-cache"),
            method: FitMethod::R,
            target_layer: 51,
            source_layers: vec![50],
            row_start: 0,
            row_end: 1,
            dim_batch: 1,
            skip_first: 0,
            max_tokens: 2,
            max_prompts: 1,
            records_this_run: None,
            no_special_tokens: true,
            resume: false,
        }
    }

    fn config() -> artifact::Config {
        artifact::make_config(
            &MuseGlimmerConfig::unsloth_release_reference(),
            MuseGlimmerArtifactProfile::UnslothQ8_0,
            "11".repeat(32),
            FitMethod::R,
            51,
            vec![50],
            0,
            1,
            0,
            1,
            2,
            false,
            "22".repeat(32),
            1,
            format!("git-source-sha256-v2:{}", "33".repeat(32)),
        )
    }

    fn prompt() -> PreparedPrompt {
        PreparedPrompt {
            id: "p0".into(),
            token_ids: vec![1, 2],
            original_token_count: 2,
            truncated: false,
        }
    }

    fn temporary_root() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "qwen-muse-row-fit-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        root
    }

    #[test]
    fn muse_row_args_enforce_bounded_shards_and_query_batches() {
        validate_args(&args()).unwrap();
        let mut value = args();
        value.row_end = 256;
        validate_args(&value).unwrap();
        value.row_end = 257;
        assert!(validate_args(&value).is_err());
        value = args();
        value.dim_batch = MUSE_GLIMMER_QUERY_BATCH_MAX;
        validate_args(&value).unwrap();
        value.dim_batch = MUSE_GLIMMER_QUERY_BATCH_MAX + 1;
        assert!(validate_args(&value).is_err());
        value = args();
        value.source_layers = vec![50, 49];
        assert!(validate_args(&value).is_err());
        value = args();
        value.resume = true;
        validate_args(&value).unwrap();
        value.records_this_run = Some(1);
        validate_args(&value).unwrap();
        value.records_this_run = Some(0);
        assert!(validate_args(&value).is_err());
    }

    #[test]
    fn fit_record_budget_advances_from_the_resume_cursor() {
        assert_eq!(fit_record_range(0, 25, None).unwrap(), 0..25);
        assert_eq!(fit_record_range(0, 25, Some(1)).unwrap(), 0..1);
        assert_eq!(fit_record_range(1, 25, Some(1)).unwrap(), 1..2);
        assert_eq!(fit_record_range(24, 25, Some(8)).unwrap(), 24..25);
        assert_eq!(fit_record_range(25, 25, Some(1)).unwrap(), 25..25);
        assert!(fit_record_range(0, 25, Some(0)).is_err());
        assert!(fit_record_range(26, 25, Some(1)).is_err());
    }

    #[test]
    fn checkpoint_round_trip_restores_exact_accumulator_and_metadata() {
        let root = temporary_root();
        let output = root.join("rows");
        let config = config();
        let prompts = vec![prompt()];
        let digest = super::super::digest_json(&config).unwrap();
        let shape = artifact::expected_shape(&config).unwrap();
        let count = shape.into_iter().product::<usize>();
        let WorkerState::Active(mut state) =
            open_or_create_state(&output, false, &config, &prompts, &digest, shape, count).unwrap()
        else {
            panic!("new Muse row state was complete");
        };
        state.sums[0] = 3.5;
        state.next_record = 1;
        state.used_prompts = 1;
        state.forward_seconds = 0.5;
        state.vjp_wall_seconds = 1.5;
        state.vjp_timings.total_seconds = 1.0;
        state.diagnostics.push(artifact::ReplayDiagnostic {
            block: 51,
            kind: "full".into(),
            post_attention_replay_max_abs_error: 0.01,
            post_block_replay_max_abs_error: 0.02,
        });
        super::super::publish_immutable(
            &output.join("muse-row-sums-00000001.f32le"),
            b"orphaned pre-checkpoint bytes",
        )
        .unwrap();
        checkpoint_active(&output, &digest, &mut state, &prompts, &config, shape).unwrap();

        let WorkerState::Active(restored) =
            open_or_create_state(&output, true, &config, &prompts, &digest, shape, count).unwrap()
        else {
            panic!("checkpoint unexpectedly completed");
        };
        assert_eq!(restored.generation, 1);
        assert_eq!(restored.next_record, 1);
        assert_eq!(restored.used_prompts, 1);
        assert_eq!(restored.sums[0].to_bits(), 3.5f32.to_bits());
        assert_eq!(restored.diagnostics, state.diagnostics);
        assert_eq!(restored.vjp_timings, state.vjp_timings);
        std::fs::remove_dir_all(root).unwrap();
    }
}
