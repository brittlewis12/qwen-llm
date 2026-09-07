//! Resident trace cohorts: bounded writers and atomic batch publication.

use super::*;

pub(super) const MAX_TRACE_FULL_BATCH_RECORD_BYTES: usize = 1024 * 1024;

pub(super) const MAX_TRACE_FULL_BATCH_INPUT_BYTES: usize = 8 * 1024 * 1024;

pub(super) const MAX_TRACE_FULL_BATCH_OUTPUT_BYTES: usize = 2 * 1024 * 1024 * 1024;

pub(super) const MAX_TRACE_FULL_BATCH_DOCUMENT_BYTES: usize =
    MAX_TRACE_FULL_BATCH_OUTPUT_BYTES - JSON_FILE_MAX_BYTES;

#[derive(Debug)]
pub(super) struct TraceFullBatchRequest {
    pub(super) input: LensCohortRequest,
    pub(super) vectors: Vec<TraceFullVectorCell>,
}

impl TraceFullBatchRequest {
    pub(super) fn input_spec(&self) -> LensInputSpec<'_> {
        self.input.input_spec()
    }
}

pub(super) struct PreparedTraceFullBatchPrompt<'model> {
    pub(super) line_number: usize,
    pub(super) request_id: String,
    pub(super) input_source: &'static str,
    pub(super) add_special_tokens: Option<bool>,
    pub(super) token_ids: Vec<i32>,
    pub(super) input_tokens: Vec<TraceFullInputToken>,
    pub(super) rendering: TraceFullRendering,
    pub(super) vector_positions_by_layer: BTreeMap<u32, Vec<usize>>,
    pub(super) captures: Vec<WorkspaceLensPackedPostBlockCapture<'model>>,
    pub(super) cells: Vec<TraceFullCell>,
    pub(super) transported_vectors: Vec<TraceFullVector>,
}

pub(super) struct PreparedTraceFullBatchInput {
    pub(super) line_number: usize,
    pub(super) request_id: String,
    pub(super) input_source: &'static str,
    pub(super) add_special_tokens: Option<bool>,
    pub(super) token_ids: Vec<i32>,
    pub(super) input_tokens: Vec<TraceFullInputToken>,
    pub(super) rendering: TraceFullRendering,
    pub(super) vector_positions_by_layer: BTreeMap<u32, Vec<usize>>,
    pub(super) vector_count: usize,
}

pub(super) struct TraceFullBatchPublication {
    pub(super) output: PathBuf,
    pub(super) parent: PathBuf,
    pub(super) staging: PathBuf,
    pub(super) staged_bytes: usize,
    pub(super) max_bytes: usize,
    pub(super) manifest_reserve_bytes: usize,
    pub(super) published: bool,
}

impl TraceFullBatchPublication {
    pub(super) fn create(
        output: &Path,
        max_bytes: usize,
        manifest_reserve_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            manifest_reserve_bytes > 0 && manifest_reserve_bytes < max_bytes,
            "trace-full batch manifest reserve must fit inside its output budget"
        );
        ensure!(
            !output.exists(),
            "trace-full batch output {} must not already exist",
            output.display()
        );
        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        let leaf = output
            .file_name()
            .context("trace-full batch output has no directory name")?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock before Unix epoch")?
            .as_nanos();
        let staging = parent.join(format!(
            ".{}.stage.{}.{}",
            leaf.to_string_lossy(),
            std::process::id(),
            nonce
        ));
        DirBuilder::new()
            .mode(0o700)
            .create(&staging)
            .with_context(|| format!("create trace-full batch staging {}", staging.display()))?;
        Ok(Self {
            output: output.to_path_buf(),
            parent: parent.to_path_buf(),
            staging,
            staged_bytes: 0,
            max_bytes,
            manifest_reserve_bytes,
            published: false,
        })
    }

    pub(super) fn stage_json_document(
        &mut self,
        path: &str,
        document: &impl Serialize,
    ) -> Result<usize> {
        ensure!(
            Path::new(path).components().count() == 1 && path != TRACE_FULL_BATCH_MANIFEST_NAME,
            "trace-full batch document path must be one non-manifest filename"
        );
        let document_bank_bytes = self
            .max_bytes
            .checked_sub(self.manifest_reserve_bytes)
            .context("trace-full batch document budget underflow")?;
        let aggregate_remaining = document_bank_bytes
            .checked_sub(self.staged_bytes)
            .context("trace-full batch document budget exhausted")?;
        let document_limit = aggregate_remaining.min(MAX_TRACE_DOCUMENT_BYTES);
        ensure!(
            document_limit > 0,
            "trace-full batch document budget is exhausted"
        );
        let target = self.staging.join(path);
        ensure!(
            !target.exists(),
            "trace-full batch document {path:?} is duplicated"
        );
        let temporary = self.staging.join(format!(".{path}.tmp"));
        let serialized: Result<usize> = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&temporary)
                .with_context(|| {
                    format!(
                        "create trace-full batch temporary document {}",
                        temporary.display()
                    )
                })?;
            let written = {
                let mut buffered = std::io::BufWriter::new(&mut file);
                let mut bounded = ByteLimitedWriter::new(&mut buffered, document_limit);
                serde_json::to_writer(&mut bounded, document)
                    .with_context(|| format!("serialize trace-full batch document {path:?}"))?;
                bounded
                    .flush()
                    .with_context(|| format!("flush trace-full batch document {path:?}"))?;
                bounded.written()
            };
            file.sync_all()
                .with_context(|| format!("sync trace-full batch document {path:?}"))?;
            Ok(written)
        })();
        if serialized.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        let written = serialized?;
        std::fs::rename(&temporary, &target)
            .with_context(|| format!("install trace-full batch document {}", target.display()))?;
        self.staged_bytes = self
            .staged_bytes
            .checked_add(written)
            .context("trace-full batch output byte count overflow")?;
        Ok(written)
    }

    pub(super) fn publish(mut self, manifest: &[u8]) -> Result<()> {
        ensure!(
            manifest.len() <= self.manifest_reserve_bytes,
            "trace-full batch manifest requires {} bytes, exceeding reserved {}",
            manifest.len(),
            self.manifest_reserve_bytes
        );
        let total_bytes = self
            .staged_bytes
            .checked_add(manifest.len())
            .context("trace-full batch publication byte count overflow")?;
        ensure!(
            total_bytes <= self.max_bytes,
            "trace-full batch publication requires {total_bytes} bytes, exceeding aggregate output budget {}",
            self.max_bytes
        );
        write_atomic_replace(&self.staging.join(TRACE_FULL_BATCH_MANIFEST_NAME), manifest)?;
        sync_directory(&self.staging)?;
        ensure!(
            !self.output.exists(),
            "trace-full batch output {} appeared during publication",
            self.output.display()
        );
        publish_trace_full_batch_directory_exclusive(&self.staging, &self.output)?;
        self.published = true;
        if let Err(error) = sync_directory(&self.parent) {
            eprintln!(
                "warning: trace-full batch {} is published, but its parent directory could not be synced: {error:#}",
                self.output.display()
            );
        }
        Ok(())
    }
}

impl Drop for TraceFullBatchPublication {
    fn drop(&mut self) {
        if !self.published && self.staging.exists() {
            let _ = std::fs::remove_dir_all(&self.staging);
        }
    }
}

pub(super) fn publish_trace_full_batch_directory_exclusive(
    staging: &Path,
    output: &Path,
) -> Result<()> {
    let old = CString::new(staging.as_os_str().as_bytes())?;
    let new = CString::new(output.as_os_str().as_bytes())?;
    let renamed = unsafe {
        libc::renameatx_np(
            libc::AT_FDCWD,
            old.as_ptr(),
            libc::AT_FDCWD,
            new.as_ptr(),
            libc::RENAME_EXCL,
        )
    };
    if renamed != 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "publish trace-full batch staging {} to {}",
                staging.display(),
                output.display()
            )
        });
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub(super) struct TraceFullBatchAttribution {
    pub(super) batch_schema: &'static str,
    pub(super) request_id: String,
    pub(super) request_index: usize,
    pub(super) request_count: usize,
    pub(super) aggregate_rows: usize,
    pub(super) shared_timing_fields: [&'static str; 4],
}

#[derive(Debug, Serialize)]
pub(super) struct TraceFullBatchArtifact {
    pub(super) request_id: String,
    pub(super) request_index: usize,
    pub(super) source_line: usize,
    pub(super) input_tokens: usize,
    pub(super) path: String,
}

#[derive(Debug, Serialize)]
pub(super) struct TraceFullBatchTiming {
    pub(super) model_load_wall_ms: f64,
    pub(super) packed_prefill_gpu_ms: f64,
    pub(super) packed_prefill_wall_ms: f64,
    pub(super) matrix_read_wall_ms: f64,
    pub(super) readout_gpu_ms: f64,
    pub(super) readout_command_wall_ms: f64,
    pub(super) batch_execution_wall_ms: f64,
    pub(super) document_staging_wall_ms: f64,
    pub(super) total_wall_ms: f64,
}

pub(crate) fn trace_full_batch(args: TraceFullArgs) -> Result<()> {
    validate_trace_full_args(&args)?;
    let requests_path = args
        .requests_jsonl
        .as_deref()
        .context("--requests-jsonl is required")?;
    let output_dir = resolve_output_path(
        args.output_dir
            .as_deref()
            .context("--output-dir is required")?,
    )?;
    ensure!(
        !output_dir.exists(),
        "trace-full batch output {} must not already exist",
        output_dir.display()
    );
    let requests = read_trace_full_batch_requests(requests_path)?;
    ensure!(
        requests.len() >= 2,
        "trace-full batching requires at least two requests"
    );

    let manifest_path = args.full_lens.join(FULL_MANIFEST_NAME);
    let manifest: FullLensManifest = read_json_file(&manifest_path)?;
    validate_trace_full_manifest(&manifest)?;
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

    let batch_started = Instant::now();
    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open model {}", args.model.display()))?;
    let family = ModelFamily::detect(&gguf).context("detect trace-full Qwen family")?;
    ensure!(
        family == ModelFamily::Qwen35,
        "trace-full requires an ordinary dense Qwen model"
    );
    let model_context_tokens = gguf.declared_context_length()?;
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load tokenizer from GGUF")?;

    let mut prepared_inputs = Vec::new();
    prepared_inputs
        .try_reserve_exact(requests.len())
        .context("allocate trace-full prepared inputs")?;
    let mut aggregate_rows = 0usize;
    let mut aggregate_minimum_output_bytes = 0usize;
    for (line_number, request) in requests {
        let vector_count = request.vectors.len();
        let prepared = prepare_qwen_model_input(request.input_spec(), family, &gguf, &tokenizer)
            .with_context(|| format!("prepare trace-full request on line {line_number}"))?;
        ensure!(
            !prepared.token_ids.is_empty(),
            "trace-full request on line {line_number} tokenized to no tokens"
        );
        if let Some(max_tokens) = args.max_tokens {
            ensure!(
                prepared.token_ids.len() <= max_tokens,
                "trace-full request on line {line_number} has {} tokens, exceeding --max-tokens {max_tokens}; input is not truncated",
                prepared.token_ids.len(),
            );
        }
        ensure!(
            prepared.token_ids.len() <= model_context_tokens,
            "trace-full request on line {line_number} requires {} token forwards, exceeding model context {}",
            prepared.token_ids.len(),
            model_context_tokens,
        );
        aggregate_rows = aggregate_rows
            .checked_add(prepared.token_ids.len())
            .context("trace-full aggregate row count overflow")?;
        let vector_positions_by_layer =
            group_trace_full_vector_cells(&request.vectors, &layers, prepared.token_ids.len())?;
        let minimum_output_bytes = ensure_trace_document_budget(
            prepared.token_ids.len(),
            layers.len(),
            args.top_k,
            request.vectors.len(),
            manifest.transport.hidden_size as usize,
            MAX_TRACE_DOCUMENT_BYTES,
            &format!("trace-full request on line {line_number}"),
        )?;
        aggregate_minimum_output_bytes = aggregate_minimum_output_bytes
            .checked_add(minimum_output_bytes)
            .context("trace-full aggregate output reserve overflow")?;
        ensure!(
            aggregate_minimum_output_bytes <= MAX_TRACE_FULL_BATCH_DOCUMENT_BYTES,
            "trace-full request cohort cannot fit the {MAX_TRACE_FULL_BATCH_DOCUMENT_BYTES}-byte aggregate document budget after reserving its manifest; select fewer requests, layers, top-k ranks, or vector cells"
        );
        let input_tokens = decode_trace_full_input_tokens(&tokenizer, &prepared.token_ids)?;
        prepared_inputs.push(PreparedTraceFullBatchInput {
            line_number,
            request_id: request.input.id,
            input_source: prepared.source,
            add_special_tokens: prepared.add_special_tokens,
            token_ids: prepared.token_ids,
            input_tokens,
            rendering: prepared.rendering,
            vector_positions_by_layer,
            vector_count,
        });
    }

    crate::shutdown::checkpoint()?;
    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let model_load_started = Instant::now();
    let loaded = runtime
        .load_opened_gguf_with_intent(
            gguf,
            args.model.clone(),
            LoadedModelConfig::default(),
            ModelLoadIntent::SinglePassAnalysis,
        )
        .with_context(|| format!("load model {}", args.model.display()))?;
    let model_load_wall_ms = model_load_started.elapsed().as_secs_f64() * 1e3;
    validate_deployed_model(&manifest, &loaded)?;
    let arch = loaded.arch();
    let (deployed_model, tokenizer_summary) = trace_full_runtime_summaries(&args.model, &loaded);
    let lens_summary = trace_full_lens_summary(&manifest);

    let execution_started = Instant::now();
    let max_prompt_rows = prepared_inputs
        .iter()
        .map(|input| input.token_ids.len())
        .max()
        .context("trace-full prompt cohort is empty")?;
    let host_result_reserve_bytes =
        trace_host_result_reserve_bytes(prepared_inputs.len(), MAX_TRACE_DOCUMENT_BYTES)?;
    let capture_priced_upper_bytes = prepared_inputs.iter().try_fold(0u64, |total, input| {
        let tiles = trace_position_tiles(
            input.token_ids.len(),
            MAX_WORKSPACE_LENS_PACKED_READOUT_POSITIONS,
        )?;
        let prompt_bytes = qwen_trace_capture_priced_upper_bytes(
            &loaded,
            &tiles,
            layers.len(),
            arch.hidden_size as usize,
        )?;
        total
            .checked_add(prompt_bytes)
            .context("trace-full cohort capture byte count overflow")
    })?;
    let prefill_scratch_upper_bytes = loaded
        .workspace_lens_packed_capture_scratch_upper_bytes(max_prompt_rows)
        .context("price trace-full cohort packed-prefill scratch")?;
    let admission = loaded
        .qwen_execution_memory_admission_with_additional_bytes(
            1,
            max_prompt_rows,
            prefill_scratch_upper_bytes,
            capture_priced_upper_bytes,
            host_result_reserve_bytes,
        )
        .context("price trace-full cohort sequence and retained captures")?;
    ensure!(
        admission.admitted,
        "trace-full cohort memory admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
        admission.reason.as_str(),
        admission.required_bytes,
        admission.working_set_headroom_bytes,
        admission.signals.process_limit_remaining_bytes,
    );
    let mut prompts = Vec::new();
    prompts
        .try_reserve_exact(prepared_inputs.len())
        .context("allocate trace-full captured prompts")?;
    for input in prepared_inputs {
        crate::shutdown::checkpoint()?;
        let mut sequence = loaded
            .create_sequence(SequenceConfig::new(input.token_ids.len()))
            .with_context(|| {
                format!(
                    "create trace-full sequence for request {:?}",
                    input.request_id
                )
            })?;
        let position_tiles = trace_position_tiles(
            input.token_ids.len(),
            MAX_WORKSPACE_LENS_PACKED_READOUT_POSITIONS,
        )?;
        let mut captures = Vec::new();
        captures
            .try_reserve_exact(position_tiles.len())
            .context("allocate trace-full request capture tiles")?;
        for tile in &position_tiles {
            crate::shutdown::checkpoint()?;
            let capture = {
                let mut workspace_lens = loaded
                    .workspace_lens_session(&mut sequence)
                    .with_context(|| {
                        format!("open trace-full session for request {:?}", input.request_id)
                    })?;
                workspace_lens
                    .forward_packed_post_block_capture(&input.token_ids[tile.clone()], &layers)
                    .with_context(|| {
                        format!(
                            "capture packed trace-full request {:?} positions {}..{}",
                            input.request_id, tile.start, tile.end
                        )
                    })?
            };
            ensure!(
                capture.start_position() == tile.start
                    && capture.end_position() == tile.end
                    && capture.token_ids() == &input.token_ids[tile.clone()]
                    && capture.layer_ids() == layers.as_slice()
                    && capture.hidden_size() == arch.hidden_size as usize,
                "packed trace-full capture metadata is inconsistent for request {:?} at {}..{}",
                input.request_id,
                tile.start,
                tile.end,
            );
            captures.push(capture);
        }
        let expected_cells = layers
            .len()
            .checked_mul(input.token_ids.len())
            .context("trace-full request cell count overflow")?;
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(expected_cells)
            .context("allocate trace-full request cells")?;
        let mut transported_vectors = Vec::new();
        transported_vectors
            .try_reserve_exact(input.vector_count)
            .context("allocate trace-full request vectors")?;
        prompts.push(PreparedTraceFullBatchPrompt {
            line_number: input.line_number,
            request_id: input.request_id,
            input_source: input.input_source,
            add_special_tokens: input.add_special_tokens,
            token_ids: input.token_ids,
            input_tokens: input.input_tokens,
            rendering: input.rendering,
            vector_positions_by_layer: input.vector_positions_by_layer,
            captures,
            cells,
            transported_vectors,
        });
    }
    let mut readout_owner = loaded
        .create_sequence(SequenceConfig::new(1))
        .context("create trace-full readout owner")?;

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

    let tiled = max_prompt_rows > MAX_WORKSPACE_LENS_PACKED_READOUT_POSITIONS;
    let workspace_rows = max_prompt_rows.min(MAX_WORKSPACE_LENS_PACKED_READOUT_POSITIONS);
    let execution_mode = if tiled {
        "resident_layer_major_prompt_local_tiled_readout_no_interventions"
    } else {
        "resident_layer_major_prompt_local_readout_no_interventions"
    };
    let mut prompt_workspace = {
        let workspace_lens = loaded
            .workspace_lens_session(&mut readout_owner)
            .context("open trace-full cohort workspace owner")?;
        workspace_lens
            .full_readout_workspace(workspace_rows)
            .context("allocate admitted prompt-local trace-full GPU workspace")?
    };

    let mut matrix_read_wall_ms = 0.0f64;
    let mut readout_gpu_ms = 0.0f64;
    let mut readout_wall_ms = 0.0f64;
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
        prompt_workspace
            .bind_f16_transport(&matrix)
            .with_context(|| format!("bind cohort full-lens source layer {layer}"))?;

        for prompt_index in 0..prompts.len() {
            let vector_positions = prompts[prompt_index]
                .vector_positions_by_layer
                .get(&layer)
                .cloned()
                .unwrap_or_default();
            let request_id = prompts[prompt_index].request_id.clone();
            let capture_count = prompts[prompt_index].captures.len();
            for capture_index in 0..capture_count {
                let readout = {
                    let capture = &prompts[prompt_index].captures[capture_index];
                    let capture_end = capture.end_position();
                    let tile_vector_positions = vector_positions
                        .iter()
                        .copied()
                        .filter(|position| {
                            *position >= capture.start_position() && *position < capture_end
                        })
                        .collect::<Vec<_>>();
                    prompt_workspace
                        .apply_packed_capture_bound_f16_transport_topk_with_vectors(
                            capture,
                            layer,
                            args.top_k,
                            &tile_vector_positions,
                        )
                    .with_context(|| {
                        format!(
                            "apply prompt-local full-lens source layer {layer} for request {request_id:?} positions {}..{capture_end}",
                            capture.start_position(),
                        )
                    })?
                };
                readout_gpu_ms += readout.readout_gpu_ms;
                readout_wall_ms += readout.readout_wall_ms;
                append_trace_full_prompt_readout(
                    &tokenizer,
                    layer,
                    readout.positions,
                    readout.transported_vectors,
                    &mut prompts[prompt_index],
                )?;
            }
        }
    }

    let packed_prefill_gpu_ms = prompts
        .iter()
        .flat_map(|prompt| prompt.captures.iter())
        .map(WorkspaceLensPackedPostBlockCapture::packed_prefill_gpu_ms)
        .sum::<f64>();
    let packed_prefill_wall_ms = prompts
        .iter()
        .flat_map(|prompt| prompt.captures.iter())
        .map(WorkspaceLensPackedPostBlockCapture::packed_prefill_wall_ms)
        .sum::<f64>();
    let batch_execution_wall_ms = execution_started.elapsed().as_secs_f64() * 1e3;
    let request_count = prompts.len();
    let document_staging_started = Instant::now();
    let mut publication = TraceFullBatchPublication::create(
        &output_dir,
        MAX_TRACE_FULL_BATCH_OUTPUT_BYTES,
        JSON_FILE_MAX_BYTES,
    )?;
    let mut artifacts = Vec::new();
    artifacts
        .try_reserve_exact(request_count)
        .context("allocate trace-full batch manifest entries")?;
    for (request_index, prompt) in prompts.into_iter().enumerate() {
        let prompt_packed_prefill_gpu_ms = prompt
            .captures
            .iter()
            .map(WorkspaceLensPackedPostBlockCapture::packed_prefill_gpu_ms)
            .sum::<f64>();
        let prompt_packed_prefill_wall_ms = prompt
            .captures
            .iter()
            .map(WorkspaceLensPackedPostBlockCapture::packed_prefill_wall_ms)
            .sum::<f64>();
        let expected_cells = layers
            .len()
            .checked_mul(prompt.token_ids.len())
            .context("trace-full request cell count overflow")?;
        ensure!(
            prompt.cells.len() == expected_cells,
            "trace-full request {:?} produced {} cells, expected {expected_cells}",
            prompt.request_id,
            prompt.cells.len()
        );
        let occurrences = aggregate_trace_full_occurrences(&prompt.cells, &layers);
        let vectors = (!prompt.transported_vectors.is_empty()).then(|| TraceFullVectors {
            operation: "row_major_f16_transport_times_f32_post_block_residual",
            stage: "transported_target_coordinate_before_output_rmsnorm_and_lm_head",
            value_dtype: "f32",
            hidden_coordinate: "zero_based_target_layer_residual_coordinate",
            hidden_size: arch.hidden_size as usize,
            shape: [prompt.transported_vectors.len(), arch.hidden_size as usize],
            cell_order: "selected_layers_order_then_source_position_ascending",
            cells: prompt.transported_vectors,
        });
        let input_tokens = prompt.token_ids.len();
        let request_id = prompt.request_id;
        let document = TraceFullDocument {
            schema: "qwen.lens.trace",
            schema_version: 3,
            producer: TraceFullProducer {
                build_commit: env!("QWEN_BUILD_COMMIT"),
                build_dirty: env!("QWEN_BUILD_DIRTY"),
                build_source_state: env!("QWEN_BUILD_SOURCE_STATE"),
            },
            deployed_model: deployed_model.clone(),
            tokenizer: tokenizer_summary.clone(),
            lens: lens_summary.clone(),
            score_semantics: TraceFullScoreSemantics {
                kind: "logit",
                normalization: "deployed_output_rmsnorm",
                candidate_universe: "full_model_vocabulary",
                softmax_applied: false,
            },
            execution_mode,
            input_source: prompt.input_source,
            add_special_tokens: prompt.add_special_tokens,
            input_token_ids: prompt.token_ids,
            input_tokens: prompt.input_tokens,
            rendering: prompt.rendering,
            coordinates: TraceFullCoordinates {
                source_layer: "zero_based_transformer_block_index_at_post_block_residual",
                source_position: "zero_based_input_token_position",
                predicts_position: "source_position_plus_one",
                rank: "zero_based_full_vocabulary_logit_rank",
            },
            selected_layers: layers.clone(),
            top_k: args.top_k,
            occurrence_definition: "one_token_id_appearing_in_one_returned_top_k_list",
            cells: prompt.cells,
            vectors,
            timing: TraceFullTiming {
                packed_prefill_gpu_ms: prompt_packed_prefill_gpu_ms,
                packed_prefill_wall_ms: prompt_packed_prefill_wall_ms,
                matrix_read_wall_ms,
                readout_gpu_ms,
                readout_command_wall_ms: readout_wall_ms,
                trace_execution_wall_ms: batch_execution_wall_ms,
            },
            occurrences,
            batch: Some(TraceFullBatchAttribution {
                batch_schema: "qwen.lens.trace_batch",
                request_id: request_id.clone(),
                request_index,
                request_count,
                aggregate_rows,
                shared_timing_fields: [
                    "matrix_read_wall_ms",
                    "readout_gpu_ms",
                    "readout_command_wall_ms",
                    "trace_execution_wall_ms",
                ],
            }),
        };
        let artifact_path = format!("trace-{request_index:04}.json");
        artifacts.push(TraceFullBatchArtifact {
            request_id,
            request_index,
            source_line: prompt.line_number,
            input_tokens,
            path: artifact_path.clone(),
        });
        publication.stage_json_document(&artifact_path, &document)?;
    }
    let document_staging_wall_ms = document_staging_started.elapsed().as_secs_f64() * 1e3;
    let total_wall_ms = batch_started.elapsed().as_secs_f64() * 1e3;

    let batch_manifest = TraceFullBatchManifest {
        schema: "qwen.lens.trace_batch",
        schema_version: 2,
        producer: TraceFullProducer {
            build_commit: env!("QWEN_BUILD_COMMIT"),
            build_dirty: env!("QWEN_BUILD_DIRTY"),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE"),
        },
        execution_mode,
        requests_jsonl: requests_path.to_path_buf(),
        deployed_model,
        lens: lens_summary,
        selected_layers: layers,
        top_k: args.top_k,
        request_count,
        aggregate_rows,
        artifacts,
        timing: TraceFullBatchTiming {
            model_load_wall_ms,
            packed_prefill_gpu_ms,
            packed_prefill_wall_ms,
            matrix_read_wall_ms,
            readout_gpu_ms,
            readout_command_wall_ms: readout_wall_ms,
            batch_execution_wall_ms,
            document_staging_wall_ms,
            total_wall_ms,
        },
    };
    let manifest_bytes =
        serialize_json_pretty_bounded(&batch_manifest, "trace-full batch manifest")?;
    publication.publish(&manifest_bytes)?;
    println!(
        "trace-full batch {} requests, {} aggregate rows, {} layers | {:.1} ms execution ({:.1} ms readout GPU) | {}",
        request_count,
        aggregate_rows,
        batch_manifest.selected_layers.len(),
        batch_execution_wall_ms,
        readout_gpu_ms,
        output_dir.display()
    );
    Ok(())
}

pub(super) fn read_trace_full_batch_requests(
    path: &Path,
) -> Result<Vec<(usize, TraceFullBatchRequest)>> {
    let (file, length) = open_regular_file(path)?;
    ensure!(
        length <= MAX_TRACE_FULL_BATCH_INPUT_BYTES,
        "trace-full request file exceeds {MAX_TRACE_FULL_BATCH_INPUT_BYTES} bytes"
    );
    let request_root = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut reader = BufReader::new(file);
    let mut line_bytes = Vec::new();
    let mut line_number = 0usize;
    let mut requests = Vec::new();
    let mut ids = BTreeSet::new();
    while read_bounded_jsonl_record(
        &mut reader,
        &mut line_bytes,
        MAX_TRACE_FULL_BATCH_RECORD_BYTES,
        path,
        line_number + 1,
    )? {
        line_number += 1;
        let line = std::str::from_utf8(&line_bytes)
            .with_context(|| format!("read {} line {line_number} as UTF-8", path.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut value: serde_json::Value = serde_json::from_str(trimmed)
            .with_context(|| format!("parse {} line {line_number}", path.display()))?;
        let object = value.as_object_mut().with_context(|| {
            format!("trace-full request on line {line_number} must be a JSON object")
        })?;
        let vectors = object
            .remove("vectors")
            .map(serde_json::from_value)
            .transpose()
            .with_context(|| format!("parse trace-full vectors on line {line_number}"))?
            .unwrap_or_default();
        let mut input: LensCohortRequest = serde_json::from_value(value)
            .with_context(|| format!("parse {} line {line_number}", path.display()))?;
        input
            .resolve_paths(request_root)
            .with_context(|| format!("resolve trace-full request on line {line_number}"))?;
        let request = TraceFullBatchRequest { input, vectors };
        ensure!(
            !request.input.id.is_empty() && request.input.id.len() <= 128,
            "trace-full request ID on line {line_number} must contain 1..=128 bytes"
        );
        ensure!(
            ids.insert(request.input.id.clone()),
            "trace-full request ID {:?} is duplicated",
            request.input.id
        );
        validate_lens_input_spec(request.input_spec())
            .with_context(|| format!("validate trace-full request on line {line_number}"))?;
        ensure!(
            request
                .input
                .prompt
                .as_ref()
                .is_none_or(|prompt| !prompt.is_empty()),
            "trace-full request prompt on line {line_number} is empty"
        );
        let mut vector_cells = BTreeSet::new();
        ensure!(
            request
                .vectors
                .iter()
                .all(|cell| vector_cells.insert(*cell)),
            "trace-full request on line {line_number} has duplicate vector cells"
        );
        requests.push((line_number, request));
    }
    ensure!(!requests.is_empty(), "trace-full request cohort is empty");
    Ok(requests)
}
