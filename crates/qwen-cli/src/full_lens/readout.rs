//! Full-vocabulary readouts from imported transports.

use super::*;

pub(super) const MAX_FULL_READOUT_TOP_K: usize = 25;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FullTokenTargetCovector {
    DeployedLogitNumerator,
    RawLmHead,
}

#[derive(Debug, Serialize)]
pub(super) struct FullReadoutDocument {
    pub(super) schema: &'static str,
    pub(super) schema_version: u32,
    pub(super) readout: &'static str,
    pub(super) scoring: &'static str,
    pub(super) score_semantics: &'static str,
    pub(super) ranking_scope: &'static str,
    pub(super) source_site: &'static str,
    pub(super) input: FullReadoutInput,
    pub(super) artifact: FullReadoutArtifact,
    pub(super) deployed_model: FullReadoutModel,
    pub(super) transfer: FullReadoutTransfer,
    pub(super) reader: FullReadoutReader,
    pub(super) results: Vec<FullLayerReadout>,
}

#[derive(Debug, Serialize)]
pub(super) struct FullReadoutInput {
    pub(super) source: &'static str,
    pub(super) add_special_tokens: Option<bool>,
    pub(super) token_ids: Vec<i32>,
    pub(super) selected_position: usize,
    pub(super) captured_token_id: i32,
    pub(super) predicts_position: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct FullReadoutArtifact {
    pub(super) manifest: PathBuf,
    pub(super) manifest_canonical_json_blake3: String,
    pub(super) payload_blake3: String,
    pub(super) method: String,
    pub(super) target_layer: u32,
    pub(super) orientation: String,
    pub(super) source_repository: String,
    pub(super) source_revision: String,
    pub(super) fitted_checkpoint: String,
    pub(super) fitted_checkpoint_revision: String,
    pub(super) fit_n_prompts: u64,
    pub(super) fit_max_sequence_length: u32,
    pub(super) fit_skip_first: u32,
}

#[derive(Debug, Serialize)]
pub(super) struct FullReadoutModel {
    pub(super) path: PathBuf,
    pub(super) content_blake3: String,
    pub(super) model_locator_id: String,
    pub(super) tokenizer_metadata_id: String,
    pub(super) architecture_contract: &'static str,
    pub(super) n_layers: u32,
    pub(super) hidden_size: u32,
    pub(super) vocab_size: u32,
    pub(super) full_attention_interval: u32,
    pub(super) lm_head_dtype: String,
}

#[derive(Debug, Serialize)]
pub(super) struct FullReadoutTransfer {
    pub(super) validation_status: String,
    pub(super) override_policy: &'static str,
}

#[derive(Debug, Serialize)]
pub(super) struct FullReadoutReader {
    pub(super) build_commit: &'static str,
    pub(super) build_dirty: &'static str,
    pub(super) build_source_state: &'static str,
    pub(super) build_stamp_source: &'static str,
    pub(super) build_stamp_error: &'static str,
}

#[derive(Debug, Serialize)]
pub(super) struct FullLayerReadout {
    pub(super) source_layer: u32,
    pub(super) source_position: usize,
    pub(super) source_token_id: i32,
    pub(super) predicts_position: usize,
    pub(super) rms_denominator_f64_recomputed: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) transported_vector: Option<FullTransportedVector>,
    pub(super) top_k: Vec<FullTokenScore>,
}

#[derive(Debug, Serialize)]
pub(super) struct FullTransportedVector {
    pub(super) operation: &'static str,
    pub(super) stage: &'static str,
    pub(super) value_dtype: &'static str,
    pub(super) hidden_coordinate: &'static str,
    pub(super) hidden_size: usize,
    pub(super) shape: [usize; 1],
    pub(super) values: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub(super) struct FullTokenScore {
    pub(super) rank: usize,
    pub(super) token_id: u32,
    pub(super) token_display_lossy: String,
    pub(super) token_piece_hex: String,
    pub(super) logit: f32,
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

    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open model {}", args.model.display()))?;
    ensure!(
        ModelFamily::detect(&gguf) == Some(ModelFamily::Qwen35),
        "full-lens readout requires an ordinary dense Qwen model"
    );
    let model_context_tokens = gguf.declared_context_length()?;
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load tokenizer from GGUF")?;
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
                token_id < tokenizer.n_vocab() && token_id <= i32::MAX as u32,
                "--token-ids entry {token_id} is outside vocabulary {}",
                tokenizer.n_vocab()
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
    ensure!(
        prefix.len() <= model_context_tokens,
        "full-lens readout requires {} token forwards, exceeding model context {model_context_tokens}",
        prefix.len(),
    );

    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_opened_gguf_with_intent(
            gguf,
            args.model.clone(),
            LoadedModelConfig::default(),
            ModelLoadIntent::SinglePassAnalysis,
        )
        .with_context(|| format!("load model {}", args.model.display()))?;
    validate_deployed_model(&manifest, &loaded)?;
    let arch = loaded.arch();
    ensure!(
        token_ids
            .iter()
            .all(|&token| token >= 0 && (token as u32) < arch.vocab_size),
        "full-lens input contains a token outside the deployed vocabulary"
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
    crate::lens_run::ensure_qwen_sequence_admitted(&loaded, prefix.len())?;
    let mut sequence = loaded
        .create_sequence(SequenceConfig::new(prefix.len()))
        .context("create full-lens prompt sequence")?;
    let mut workspace_lens = loaded
        .workspace_lens_session(&mut sequence)
        .context("open full-lens workspace-lens session")?;
    let capture = workspace_lens
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
    let mut full_readout_workspace = match workspace_lens.full_readout_workspace(1) {
        Ok(workspace) => Some(workspace),
        Err(WorkspaceLensError::FullReadoutMemoryAdmissionDenied {
            reason,
            requested_bytes,
            working_set_headroom_bytes,
            process_remaining_bytes,
        }) => {
            eprintln!(
                "full-lens reusable GPU workspace unavailable ({reason:?}, requested={requested_bytes}, working_set_headroom={working_set_headroom_bytes:?}, process_remaining={process_remaining_bytes:?}); falling back to established per-layer allocations"
            );
            None
        }
        Err(error) => return Err(error).context("allocate reusable full-lens GPU workspace"),
    };
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
        let readout = if let Some(workspace) = &mut full_readout_workspace {
            workspace.apply_row_f16_transport_topk_with_vector(&matrix, residual, args.top_k)
        } else {
            workspace_lens.apply_f16_transport_topk_with_vector(&matrix, residual, args.top_k)
        }
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

pub(super) fn validate_deployed_model(
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

pub(super) fn validate_read_full_args(args: &ReadFullArgs) -> Result<()> {
    ensure!(
        args.allow_unvalidated_transfer,
        "published full-lens transfer is unvalidated; pass --allow-unvalidated-transfer to acknowledge this"
    );
    ensure!(
        (1..=MAX_FULL_READOUT_TOP_K).contains(&args.top_k),
        "--top-k must be in 1..={MAX_FULL_READOUT_TOP_K}"
    );
    ensure!(args.max_tokens > 0, "--max-tokens must be positive");
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

pub(super) fn load_native_j_readouts(directory: &Path) -> Result<(TokenReadoutManifest, Vec<f32>)> {
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
    let payload_bytes = crate::verify_payload(directory, &manifest.payload)?;
    let values = decode_f32_le(
        &payload_bytes,
        expected_shape[0] * expected_shape[1] * expected_shape[2],
    )?;
    Ok((manifest, values))
}
