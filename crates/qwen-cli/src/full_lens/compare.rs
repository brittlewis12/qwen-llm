//! Transfer comparison between transports.

use super::*;

pub(super) const MAX_TRANSFER_COMPARISON_TOKENS: usize = 32;

#[derive(Debug, Args)]
pub(crate) struct CompareTransferArgs {
    /// Dense Qwen3.8 GGUF model whose own output norm and LM head define readouts.
    #[arg(short = 'm', long)]
    pub(super) model: PathBuf,

    /// Directory produced by `qwen-lens import-full`.
    #[arg(long)]
    pub(super) full_lens: PathBuf,

    /// Native selected-token J-lens directory produced by `fit-tokens`.
    #[arg(long)]
    pub(super) native_readouts: PathBuf,

    /// Private cache directory for the strong ordered-GGUF content identity.
    #[arg(long)]
    pub(super) identity_cache: PathBuf,

    /// Optional immutable JSON report path; deterministic report is always printed.
    #[arg(long)]
    pub(super) output: Option<PathBuf>,
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
        native_manifest.readouts.token_ids.len() <= MAX_TRANSFER_COMPARISON_TOKENS,
        "transfer comparison supports at most {} selected tokens, got {}",
        MAX_TRANSFER_COMPARISON_TOKENS,
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
        .load_model_with_intent(
            &args.model,
            LoadedModelConfig::default(),
            ModelLoadIntent::SinglePassAnalysis,
        )
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
    let identity = loaded.workspace_lens_identity();
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
        .context("create transfer-comparison workspace-lens sequence")?;
    let workspace_lens = loaded
        .workspace_lens_session(&mut sequence)
        .context("open transfer-comparison workspace-lens session")?;
    let selected = workspace_lens
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
        let projected = workspace_lens
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
