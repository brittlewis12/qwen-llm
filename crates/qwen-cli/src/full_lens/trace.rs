//! Full-context traces: tiling, budgets, and the trace document.

use super::*;

pub(super) const MAX_TRACE_DOCUMENT_BYTES: usize = 256 * 1024 * 1024;

pub(super) const TRACE_MIN_INPUT_TOKEN_JSON_BYTES: usize = 72;

pub(super) const TRACE_MIN_CELL_JSON_BYTES: usize = 90;

pub(super) const TRACE_DISTRIBUTION_SUMMARY_RESERVE_BYTES: usize = 512;

pub(super) const TRACE_MIN_SCORE_JSON_BYTES: usize = 80;

pub(super) const TRACE_VECTOR_JSON_METADATA_RESERVE_BYTES: usize = 256;

pub(super) const TRACE_VECTOR_VALUE_RESERVE_BYTES: usize = 32;

pub(crate) fn trace_vector_reserve_bytes(vector_count: usize, hidden_size: usize) -> Result<usize> {
    let vector_values = vector_count
        .checked_mul(hidden_size)
        .context("trace vector value count overflow")?;
    vector_count
        .checked_mul(TRACE_VECTOR_JSON_METADATA_RESERVE_BYTES)
        .and_then(|bytes| {
            bytes.checked_add(vector_values.checked_mul(TRACE_VECTOR_VALUE_RESERVE_BYTES)?)
        })
        .context("trace vector reserve overflow")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ensure_trace_document_budget(
    position_count: usize,
    layer_count: usize,
    top_k: usize,
    distribution_summaries: bool,
    vector_count: usize,
    hidden_size: usize,
    max_document_bytes: usize,
    label: &str,
) -> Result<usize> {
    let cell_count = position_count
        .checked_mul(layer_count)
        .context("trace cell count overflow")?;
    let score_count = cell_count
        .checked_mul(top_k)
        .context("trace score count overflow")?;
    let required_row_bytes = position_count
        .checked_mul(TRACE_MIN_INPUT_TOKEN_JSON_BYTES)
        .and_then(|bytes| {
            bytes.checked_add(cell_count.checked_mul(
                TRACE_MIN_CELL_JSON_BYTES
                    + if distribution_summaries {
                        TRACE_DISTRIBUTION_SUMMARY_RESERVE_BYTES
                    } else {
                        0
                    },
            )?)
        })
        .and_then(|bytes| bytes.checked_add(score_count.checked_mul(TRACE_MIN_SCORE_JSON_BYTES)?))
        .context("trace document size lower bound overflow")?;
    let vector_reserve_bytes = trace_vector_reserve_bytes(vector_count, hidden_size)?;
    let admitted_bytes = required_row_bytes
        .checked_add(vector_reserve_bytes)
        .context("trace document admission size overflow")?;
    ensure!(
        admitted_bytes <= max_document_bytes,
        "{label} cannot fit the {max_document_bytes}-byte trace artifact budget: required rows plus conservatively priced inline vectors reserve {admitted_bytes} bytes before token text and occurrence metadata; select fewer layers, a lower --top-k, or fewer vector cells"
    );
    Ok(admitted_bytes)
}

pub(crate) fn trace_host_result_reserve_bytes(
    document_count: usize,
    max_document_bytes: usize,
) -> Result<u64> {
    // Retain every prompt-local result while matrices are reused layer-major,
    // plus one active aggregation/serialization workspace.
    let bytes = document_count
        .checked_add(1)
        .and_then(|banks| banks.checked_mul(max_document_bytes))
        .context("trace host result reserve overflow")?;
    u64::try_from(bytes).context("trace host result reserve does not fit u64")
}

pub(crate) fn trace_position_tiles(
    position_count: usize,
    tile_capacity: usize,
) -> Result<Vec<Range<usize>>> {
    ensure!(position_count > 0, "trace requires at least one position");
    ensure!(tile_capacity > 0, "trace tile capacity must be positive");
    let tile_count = position_count
        .checked_add(tile_capacity - 1)
        .context("trace tile count overflow")?
        / tile_capacity;
    let mut tiles = Vec::new();
    tiles
        .try_reserve_exact(tile_count)
        .context("allocate trace tile plan")?;
    let mut start = 0usize;
    while start < position_count {
        let end = start.saturating_add(tile_capacity).min(position_count);
        tiles.push(start..end);
        start = end;
    }
    Ok(tiles)
}

pub(super) fn qwen_trace_capture_priced_upper_bytes(
    loaded: &LoadedModel,
    tiles: &[Range<usize>],
    layer_count: usize,
    hidden_size: usize,
) -> Result<u64> {
    tiles.iter().try_fold(0u64, |total, tile| {
        let logical_elements = (tile.end - tile.start)
            .checked_mul(layer_count)
            .and_then(|value| value.checked_mul(hidden_size))
            .context("trace capture size overflow")?;
        let logical_bytes = logical_elements
            .checked_mul(std::mem::size_of::<f32>())
            .context("trace capture byte count overflow")?;
        let priced = loaded
            .context()
            .shared_buffer_size_and_align(
                u64::try_from(logical_bytes).context("trace capture byte count")?,
            )?
            .size;
        total
            .checked_add(priced)
            .context("trace capture priced byte count overflow")
    })
}

#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("trace_full_input")
        .required(true)
        .multiple(false)
        .args([
            "prompt",
            "token_ids",
            "user",
            "messages",
            "open_responses",
            "requests_jsonl",
        ])
))]
pub(crate) struct TraceFullArgs {
    /// Matching Qwen3.6, Qwen3.8, or Muse Glimmer GGUF used for capture and readout.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,

    /// Data-only linear transport or a legacy imported full lens directory.
    #[arg(long)]
    pub(crate) full_lens: PathBuf,

    /// Raw untemplated text; tokenizer-configured specials are enabled by default.
    #[arg(long, visible_alias = "raw-prompt", allow_hyphen_values = true)]
    pub(crate) prompt: Option<String>,

    /// Literal comma-separated token IDs; no specials are added.
    #[arg(long, value_delimiter = ',')]
    pub(crate) token_ids: Option<Vec<i32>>,

    /// One user message rendered with the model-family template; '-' reads stdin.
    #[arg(long, value_name = "TEXT|-")]
    pub(crate) user: Option<String>,

    /// Add one system message before --user.
    #[arg(long, requires = "user")]
    pub(crate) system: Option<String>,

    /// Strict system/user/assistant message array or wrapper JSON.
    #[arg(long, value_name = "FILE|-")]
    pub(crate) messages: Option<PathBuf>,

    /// Open Responses request JSON rendered by the exact qwen serve prompt path.
    #[arg(long, visible_alias = "responses-input", value_name = "FILE|-")]
    pub(crate) open_responses: Option<PathBuf>,

    /// Strict JSONL prompt cohort; paths inside records are relative to this file.
    #[arg(long, value_name = "FILE", requires = "output_dir")]
    pub(crate) requests_jsonl: Option<PathBuf>,

    /// Generation transition for --user/--messages; supported values depend on the lens model.
    #[arg(
        long,
        value_enum,
        conflicts_with_all = ["prompt", "token_ids", "open_responses", "requests_jsonl"]
    )]
    pub(crate) message_mode: Option<LensMessageMode>,

    /// Disable tokenizer-configured special insertion for --prompt.
    #[arg(
        long,
        requires = "prompt",
        conflicts_with_all = ["token_ids", "user", "messages", "open_responses"]
    )]
    pub(crate) no_special_tokens: bool,

    /// Unique source layers in caller output order; defaults to every artifact layer.
    #[arg(long, value_delimiter = ',')]
    pub(crate) layers: Vec<u32>,

    /// Full-vocabulary results per layer and position.
    #[arg(long, default_value_t = 8)]
    pub(crate) top_k: usize,

    /// CPU F64 full-vocabulary statistics of restored F32 logits (native Qwen only).
    #[arg(long)]
    pub(crate) distribution_summaries: bool,

    /// Optional input-token budget below the model context; inputs are never truncated.
    #[arg(long)]
    pub(crate) max_tokens: Option<usize>,

    /// Transported target-space vectors to include as layer:position cells.
    #[arg(
        long = "vectors",
        value_delimiter = ',',
        conflicts_with = "requests_jsonl"
    )]
    pub(crate) vectors: Vec<TraceFullVectorCell>,

    /// Private native identity cache for legacy assets; data exact bindings always hash retained bytes.
    #[arg(long)]
    pub(crate) identity_cache: Option<PathBuf>,

    /// Acknowledge unverified source/deployment equivalence for unbound transports.
    #[arg(long)]
    pub(crate) allow_unvalidated_transfer: bool,

    /// Replace this JSON result file atomically after a successful trace.
    #[arg(long, conflicts_with = "requests_jsonl")]
    pub(crate) output: Option<PathBuf>,

    /// Cohort output directory containing one ordinary trace per request.
    #[arg(long, requires = "requests_jsonl")]
    pub(crate) output_dir: Option<PathBuf>,

    /// Human summary or the complete JSON document on stdout.
    #[arg(long, value_enum, conflicts_with = "requests_jsonl")]
    pub(crate) format: Option<TraceFullStdoutFormat>,
}

impl TraceFullArgs {
    pub(crate) fn input_spec(&self) -> LensInputSpec<'_> {
        LensInputSpec {
            prompt: self.prompt.as_deref(),
            token_ids: self.token_ids.as_deref(),
            user: self.user.as_deref(),
            system: self.system.as_deref(),
            messages: self.messages.as_deref(),
            open_responses: self.open_responses.as_deref(),
            no_special_tokens: self.no_special_tokens,
            message_mode: self.message_mode,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum TraceFullStdoutFormat {
    Summary,
    Json,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(deny_unknown_fields)]
pub(crate) struct TraceFullVectorCell {
    pub(crate) source_layer: u32,
    pub(crate) source_position: usize,
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

#[derive(Debug, Serialize)]
pub(super) struct TraceFullDocument {
    pub(super) schema: &'static str,
    pub(super) schema_version: u32,
    pub(super) producer: TraceFullProducer,
    pub(super) deployed_model: TraceFullModel,
    pub(super) tokenizer: TraceFullTokenizer,
    pub(super) lens: TraceFullLens,
    pub(super) score_semantics: TraceFullScoreSemantics,
    pub(super) execution_mode: &'static str,
    pub(super) input_source: &'static str,
    pub(super) add_special_tokens: Option<bool>,
    pub(super) input_token_ids: Vec<i32>,
    pub(super) input_tokens: Vec<TraceFullInputToken>,
    pub(super) rendering: TraceFullRendering,
    pub(super) coordinates: TraceFullCoordinates,
    pub(super) selected_layers: Vec<u32>,
    pub(super) top_k: usize,
    pub(super) occurrence_definition: &'static str,
    pub(super) cells: Vec<TraceFullCell>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) vectors: Option<TraceFullVectors>,
    pub(super) timing: TraceFullTiming,
    pub(super) occurrences: TraceFullOccurrences,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) batch: Option<TraceFullBatchAttribution>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TraceFullProducer {
    pub(super) build_commit: &'static str,
    pub(super) build_dirty: &'static str,
    pub(super) build_source_state: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TraceFullModel {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) content_blake3: Option<String>,
    pub(super) path: PathBuf,
    pub(super) locator_scheme: &'static str,
    pub(super) locator_id: String,
    pub(super) content_authenticated: bool,
    pub(super) architecture: Option<String>,
    pub(super) name: Option<String>,
    pub(super) base_model_name: Option<String>,
    pub(super) n_layers: u32,
    pub(super) hidden_size: u32,
    pub(super) vocab_size: u32,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TraceFullTokenizer {
    pub(super) metadata_id: String,
    pub(super) model: Option<String>,
    pub(super) pretokenizer: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TraceFullScoreSemantics {
    pub(super) kind: &'static str,
    pub(super) normalization: &'static str,
    pub(super) candidate_universe: &'static str,
    pub(super) softmax_applied: bool,
}

pub(super) type TraceFullRendering = LensInputRendering;

#[derive(Clone, Debug, Serialize)]
pub(super) struct TraceFullLens {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) producer_contract: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) runtime_binding: Option<serde_json::Value>,
    pub(super) kind: &'static str,
    pub(super) method: String,
    pub(super) target_layer: u32,
    pub(super) source_site: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub(super) source_repository: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub(super) source_revision: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub(super) source_filename: String,
    pub(super) payload_blake3: String,
    pub(super) scoring: &'static str,
}

#[derive(Debug, Serialize)]
pub(super) struct TraceFullInputToken {
    pub(super) position: usize,
    pub(super) token_id: i32,
    pub(super) token_display_lossy: String,
    pub(super) token_piece_hex: String,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct TraceFullCoordinates {
    pub(super) source_layer: &'static str,
    pub(super) source_position: &'static str,
    pub(super) predicts_position: &'static str,
    pub(super) rank: &'static str,
}

#[derive(Debug, Serialize)]
pub(super) struct TraceFullCell {
    pub(super) source_layer: u32,
    pub(super) source_position: usize,
    pub(super) source_token_id: i32,
    pub(super) predicts_position: usize,
    pub(super) top_k: Vec<TraceFullTokenScore>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) distribution_summary:
        Option<qwen_llm::workspace_lens::WorkspaceLensDistributionSummary>,
}

#[derive(Debug, Serialize)]
pub(super) struct TraceFullTokenScore {
    pub(super) rank: usize,
    pub(super) token_id: u32,
    pub(super) token_display_lossy: String,
    pub(super) token_piece_hex: String,
    pub(super) logit: f32,
}

#[derive(Debug, Serialize)]
pub(super) struct TraceFullVectors {
    pub(super) operation: &'static str,
    pub(super) stage: &'static str,
    pub(super) value_dtype: &'static str,
    pub(super) hidden_coordinate: &'static str,
    pub(super) hidden_size: usize,
    pub(super) shape: [usize; 2],
    pub(super) cell_order: &'static str,
    pub(super) cells: Vec<TraceFullVector>,
}

#[derive(Debug, Serialize)]
pub(super) struct TraceFullVector {
    pub(super) source_layer: u32,
    pub(super) source_position: usize,
    pub(super) source_token_id: i32,
    pub(super) predicts_position: usize,
    pub(super) values: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub(super) struct TraceFullTiming {
    pub(super) packed_prefill_gpu_ms: f64,
    pub(super) packed_prefill_wall_ms: f64,
    pub(super) matrix_read_wall_ms: f64,
    pub(super) readout_gpu_ms: f64,
    pub(super) readout_command_wall_ms: f64,
    pub(super) trace_execution_wall_ms: f64,
}

#[derive(Debug, Serialize)]
pub(super) struct TraceFullOccurrences {
    pub(super) global: Vec<TraceFullOccurrence>,
    pub(super) per_layer: Vec<TraceFullLayerOccurrences>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(super) struct TraceFullOccurrence {
    pub(super) token_id: u32,
    pub(super) count: usize,
    pub(super) top1_count: usize,
    pub(super) best_rank: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct TraceFullLayerOccurrences {
    pub(super) source_layer: u32,
    pub(super) tokens: Vec<TraceFullOccurrence>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct OccurrenceAccumulator {
    pub(super) count: usize,
    pub(super) top1_count: usize,
    pub(super) best_rank: usize,
}

pub(super) fn write_projected_full_token_tile(
    values: &mut [f32],
    layer_index: usize,
    total_token_count: usize,
    token_start: usize,
    hidden_size: usize,
    projected: &[f32],
) -> Result<()> {
    ensure!(
        hidden_size > 0 && projected.len().is_multiple_of(hidden_size),
        "published full transport projection tile has an invalid shape"
    );
    let tile_token_count = projected.len() / hidden_size;
    ensure!(
        token_start
            .checked_add(tile_token_count)
            .is_some_and(|end| end <= total_token_count),
        "published full transport projection tile exceeds its token dimension"
    );
    let destination_start = layer_index
        .checked_mul(total_token_count)
        .and_then(|value| value.checked_add(token_start))
        .and_then(|value| value.checked_mul(hidden_size))
        .context("published full transport projection destination overflow")?;
    let destination_end = destination_start
        .checked_add(projected.len())
        .context("published full transport projection destination overflow")?;
    values
        .get_mut(destination_start..destination_end)
        .context("published full transport projection destination is outside result")?
        .copy_from_slice(projected);
    Ok(())
}

pub(super) fn trace_full_runtime_summaries(
    model_path: &Path,
    loaded: &LoadedModel,
) -> (TraceFullModel, TraceFullTokenizer) {
    let arch = loaded.arch();
    let identity = loaded.workspace_lens_identity();
    (
        TraceFullModel {
            content_blake3: None,
            path: model_path.to_path_buf(),
            locator_scheme: WORKSPACE_LENS_IDENTITY_SCHEME,
            locator_id: format!("{:016x}", identity.model_locator_id),
            content_authenticated: identity.content_authenticated,
            architecture: loaded.gguf().architecture(),
            name: loaded.gguf().get_str("general.name").map(str::to_owned),
            base_model_name: loaded
                .gguf()
                .get_str("general.base_model.0.name")
                .map(str::to_owned),
            n_layers: arch.n_layer,
            hidden_size: arch.hidden_size,
            vocab_size: arch.vocab_size,
        },
        TraceFullTokenizer {
            metadata_id: format!("{:016x}", identity.tokenizer_metadata_id),
            model: loaded
                .gguf()
                .get_str("tokenizer.ggml.model")
                .map(str::to_owned),
            pretokenizer: loaded
                .gguf()
                .get_str("tokenizer.ggml.pre")
                .map(str::to_owned),
        },
    )
}

pub(super) fn trace_full_lens_summary(manifest: &FullLensManifest) -> TraceFullLens {
    TraceFullLens {
        producer_contract: None,
        runtime_binding: None,
        kind: "published_full_transport",
        method: manifest.transport.method.clone(),
        target_layer: manifest.transport.target_layer,
        source_site: manifest.transport.capture_site.clone(),
        source_repository: manifest.source.repository.clone(),
        source_revision: manifest.source.revision.clone(),
        source_filename: manifest.source.filename.clone(),
        payload_blake3: manifest.payload.blake3.clone(),
        scoring: "deployed_output_rmsnorm_and_lm_head_full_vocabulary_logits_no_softmax",
    }
}

pub(crate) fn trace_full(args: TraceFullArgs) -> Result<()> {
    validate_trace_full_args(&args)?;
    let output = args
        .output
        .as_deref()
        .map(resolve_output_file_path)
        .transpose()?;
    let stdout_format = effective_trace_stdout_format(args.format, output.is_some());
    let manifest = FullAccess::open(&args.full_lens, true)?;
    if manifest.is_data() {
        manifest.acknowledge_transfer(args.allow_unvalidated_transfer)?;
    }
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
    let family = ModelFamily::detect(&gguf).context("detect trace-full Qwen family")?;
    ensure!(
        family == ModelFamily::Qwen35,
        "packed trace-full requires the native dense Qwen capture capability; use scalar read-full for ordinary MoE"
    );
    let model_context_tokens = gguf.declared_context_length()?;
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load tokenizer from GGUF")?;
    let prepared_input = prepare_qwen_model_input(args.input_spec(), family, &gguf, &tokenizer)?;
    let input_source = prepared_input.source;
    let add_special_tokens = prepared_input.add_special_tokens;
    let token_ids = prepared_input.token_ids;
    let rendering = prepared_input.rendering;
    ensure!(
        !token_ids.is_empty(),
        "trace-full input tokenized to no tokens"
    );
    if let Some(max_tokens) = args.max_tokens {
        ensure!(
            token_ids.len() <= max_tokens,
            "trace-full input has {} tokens, exceeding --max-tokens {max_tokens}; input is not truncated",
            token_ids.len(),
        );
    }
    ensure!(
        token_ids.len() <= model_context_tokens,
        "trace-full requires {} token forwards, exceeding model context {model_context_tokens}",
        token_ids.len(),
    );
    let vector_positions_by_layer =
        group_trace_full_vector_cells(&args.vectors, &layers, token_ids.len())?;
    ensure_trace_document_budget(
        token_ids.len(),
        layers.len(),
        args.top_k,
        args.distribution_summaries,
        args.vectors.len(),
        manifest.transport.hidden_size as usize,
        MAX_TRACE_DOCUMENT_BYTES,
        "trace-full request",
    )?;
    let host_result_reserve_bytes = trace_host_result_reserve_bytes(1, MAX_TRACE_DOCUMENT_BYTES)?;
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

    crate::shutdown::checkpoint()?;
    let mut manifest = manifest.bind_opened(
        &gguf,
        FullExecutionMode::Packed,
        args.identity_cache.as_deref(),
        args.allow_unvalidated_transfer,
    )?;
    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_opened_gguf_with_intent(
            gguf,
            args.model.clone(),
            LoadedModelConfig::default(),
            ModelLoadIntent::SinglePassAnalysis,
        )
        .with_context(|| format!("load model {}", args.model.display()))?;
    manifest.validate_loaded(&loaded)?;
    let arch = loaded.arch();
    let (mut deployed_model, tokenizer_summary) =
        trace_full_runtime_summaries(&args.model, &loaded);
    manifest.authenticate_trace_model(&mut deployed_model);
    let trace_started = Instant::now();
    let position_tiles =
        trace_position_tiles(token_ids.len(), MAX_WORKSPACE_LENS_PACKED_READOUT_POSITIONS)?;
    let capture_priced_upper_bytes = qwen_trace_capture_priced_upper_bytes(
        &loaded,
        &position_tiles,
        layers.len(),
        arch.hidden_size as usize,
    )?;
    let prefill_scratch_upper_bytes = loaded
        .workspace_lens_packed_capture_scratch_upper_bytes(token_ids.len())
        .context("price trace-full packed-prefill scratch")?;
    let admission = loaded
        .qwen_execution_memory_admission_with_additional_bytes(
            1,
            token_ids.len(),
            prefill_scratch_upper_bytes,
            capture_priced_upper_bytes,
            host_result_reserve_bytes,
        )
        .context("price trace-full sequence and retained captures")?;
    ensure!(
        admission.admitted,
        "trace-full memory admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
        admission.reason.as_str(),
        admission.required_bytes,
        admission.working_set_headroom_bytes,
        admission.signals.process_limit_remaining_bytes,
    );
    let mut sequence = loaded
        .create_sequence(SequenceConfig::new(token_ids.len()))
        .context("create trace-full prompt sequence")?;
    let mut captures = Vec::new();
    captures
        .try_reserve_exact(position_tiles.len())
        .context("allocate trace-full capture tiles")?;
    let mut packed_prefill_gpu_ms = 0.0f64;
    let mut packed_prefill_wall_ms = 0.0f64;
    for tile in &position_tiles {
        crate::shutdown::checkpoint()?;
        let capture = {
            let mut workspace_lens = loaded
                .workspace_lens_session(&mut sequence)
                .context("open trace-full capture session")?;
            workspace_lens
                .forward_packed_post_block_capture(&token_ids[tile.clone()], &layers)
                .with_context(|| {
                    format!(
                        "capture packed trace-full post-block residuals at {}..{}",
                        tile.start, tile.end
                    )
                })?
        };
        ensure!(
            capture.start_position() == tile.start
                && capture.end_position() == tile.end
                && capture.token_ids() == &token_ids[tile.clone()]
                && capture.layer_ids() == layers.as_slice()
                && capture.hidden_size() == arch.hidden_size as usize,
            "packed trace-full capture metadata is inconsistent at {}..{}",
            tile.start,
            tile.end,
        );
        packed_prefill_gpu_ms += capture.packed_prefill_gpu_ms();
        packed_prefill_wall_ms += capture.packed_prefill_wall_ms();
        captures.push(capture);
    }
    let workspace_lens = loaded
        .workspace_lens_session(&mut sequence)
        .context("open trace-full workspace-lens session")?;
    let workspace_rows = token_ids
        .len()
        .min(MAX_WORKSPACE_LENS_PACKED_READOUT_POSITIONS);
    let mut full_readout_workspace = workspace_lens
        .full_readout_workspace(workspace_rows)
        .context("allocate admitted reusable trace-full GPU workspace")?;

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
        let read_started = Instant::now();
        let matrix = manifest.read_matrix(layer)?;
        matrix_read_wall_ms += read_started.elapsed().as_secs_f64() * 1e3;

        let vector_positions = vector_positions_by_layer
            .get(&layer)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        full_readout_workspace
            .bind_f16_transport(&matrix)
            .with_context(|| format!("bind full-lens source layer {layer}"))?;
        for capture in &captures {
            let capture_end = capture.end_position();
            let tile_vector_positions = vector_positions
                .iter()
                .copied()
                .filter(|position| *position >= capture.start_position() && *position < capture_end)
                .collect::<Vec<_>>();
            let readout = full_readout_workspace
                .apply_packed_capture_bound_f16_transport_topk_with_distribution_summaries(
                    capture,
                    layer,
                    args.top_k,
                    &tile_vector_positions,
                    args.distribution_summaries,
                )
                .with_context(|| {
                    format!(
                        "apply packed full-lens source layer {layer} positions {}..{capture_end}",
                        capture.start_position()
                    )
                })?;
            ensure!(
                readout.source_layer == layer
                    && readout.start_position == capture.start_position()
                    && readout.position_count == capture.position_count()
                    && readout.top_k == args.top_k,
                "packed trace-full readout metadata is inconsistent for layer {layer} positions {}..{capture_end}",
                capture.start_position(),
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
                    distribution_summary: position.distribution_summary,
                });
            }
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
        lens: manifest.trace_summary(),
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
            packed_prefill_gpu_ms,
            packed_prefill_wall_ms,
            matrix_read_wall_ms,
            readout_gpu_ms,
            readout_command_wall_ms: readout_wall_ms,
            trace_execution_wall_ms: trace_started.elapsed().as_secs_f64() * 1e3,
        },
        occurrences,
        batch: None,
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

pub(super) fn decode_trace_full_input_tokens(
    tokenizer: &Tokenizer,
    token_ids: &[i32],
) -> Result<Vec<TraceFullInputToken>> {
    token_ids
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
        .collect()
}

pub(super) fn append_trace_full_prompt_readout(
    tokenizer: &Tokenizer,
    source_layer: u32,
    positions: Vec<WorkspaceLensPackedVocabularyPosition>,
    vectors: Vec<WorkspaceLensPackedTransportedVector>,
    prompt: &mut PreparedTraceFullBatchPrompt<'_>,
) -> Result<()> {
    for vector in vectors {
        prompt.transported_vectors.push(TraceFullVector {
            source_layer,
            source_position: vector.source_position,
            source_token_id: vector.source_token_id,
            predicts_position: vector.predicts_position,
            values: vector.values,
        });
    }
    for position in positions {
        let mut top_k = Vec::new();
        top_k
            .try_reserve_exact(position.scores.len())
            .context("allocate decoded trace-full top-k")?;
        for (rank, score) in position.scores.into_iter().enumerate() {
            let token_id =
                i32::try_from(score.token_id).context("decode trace-full vocabulary token ID")?;
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
        prompt.cells.push(TraceFullCell {
            source_layer,
            source_position: position.source_position,
            source_token_id: position.source_token_id,
            predicts_position: position.predicts_position,
            top_k,
            distribution_summary: position.distribution_summary,
        });
    }
    Ok(())
}

pub(super) fn effective_trace_stdout_format(
    explicit: Option<TraceFullStdoutFormat>,
    has_output: bool,
) -> TraceFullStdoutFormat {
    explicit.unwrap_or(if has_output {
        TraceFullStdoutFormat::Summary
    } else {
        TraceFullStdoutFormat::Json
    })
}

pub(super) fn print_trace_full_summary(document: &TraceFullDocument, output: Option<&Path>) {
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

pub(super) fn validate_trace_full_args(args: &TraceFullArgs) -> Result<()> {
    ensure!(
        (1..=MAX_FULL_READOUT_TOP_K).contains(&args.top_k),
        "--top-k must be in 1..={MAX_FULL_READOUT_TOP_K}"
    );
    ensure!(
        args.max_tokens.is_none_or(|max_tokens| max_tokens > 0),
        "--max-tokens must be positive"
    );
    if args.requests_jsonl.is_some() {
        ensure!(
            args.output_dir.is_some()
                && args.output.is_none()
                && args.format.is_none()
                && args.vectors.is_empty(),
            "--requests-jsonl requires --output-dir and does not accept single-trace output options"
        );
        return Ok(());
    }
    ensure!(
        args.output_dir.is_none(),
        "--output-dir requires --requests-jsonl"
    );
    validate_lens_input_spec(args.input_spec())?;
    ensure!(
        args.prompt.as_ref().is_none_or(|prompt| !prompt.is_empty()),
        "--prompt must not be empty"
    );
    ensure!(
        args.token_ids.as_ref().is_none_or(|token_ids| {
            !token_ids.is_empty()
                && args
                    .max_tokens
                    .is_none_or(|max_tokens| token_ids.len() <= max_tokens)
        }),
        "--token-ids must be nonempty and fit the optional --max-tokens budget"
    );
    let mut vector_cells = BTreeSet::new();
    ensure!(
        args.vectors.iter().all(|cell| vector_cells.insert(*cell)),
        "--vectors cells must be unique"
    );
    Ok(())
}

pub(super) fn group_trace_full_vector_cells(
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

pub(super) fn aggregate_trace_full_occurrences(
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

pub(super) fn update_occurrence(
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

pub(super) fn sorted_occurrences(
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
