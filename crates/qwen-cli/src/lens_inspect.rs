use anyhow::{Context, Result, ensure};
use clap::{Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;

use super::lens_input::{
    LensInputRendering, LensRenderedSpan, is_known_lens_span, is_structural_lens_span,
    valid_lens_rendering_topology, valid_lens_span_metadata,
};
use super::read_regular_file_bounded;

pub(crate) const TRACE_MAX_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_AGGREGATE_LIMIT: usize = 25;

#[derive(Debug, Args)]
pub(crate) struct InspectArgs {
    /// Regular non-symlink qwen.lens.trace JSON artifact.
    trace: PathBuf,

    /// Render a human-readable view or a typed JSON result.
    #[arg(long, value_enum, default_value_t = InspectFormat::Text, global = true)]
    format: InspectFormat,

    #[command(subcommand)]
    view: InspectView,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum InspectFormat {
    Text,
    Json,
}

#[derive(Debug, Subcommand)]
enum InspectView {
    /// Show artifact identity, dimensions, rendering, vectors, and timings.
    Summary,
    /// Rank exact token-ID occurrences across all captured cells.
    Aggregate {
        #[arg(long, default_value_t = DEFAULT_AGGREGATE_LIMIT)]
        limit: usize,
        /// Captured layer IDs or inclusive numeric ranges, in captured order.
        #[arg(long)]
        layers: Option<String>,
        /// Restrict aggregation to this resolved position; may be repeated.
        #[arg(long = "position")]
        positions: Vec<String>,
    },
    /// List every input token and all exact renderer-authored anchors.
    Positions,
    /// Show captured top-k rows at one numeric or semantic position.
    Position {
        selector: String,
        #[arg(long)]
        top_k: Option<usize>,
        /// Captured layer IDs or inclusive numeric ranges, in captured order.
        #[arg(long)]
        layers: Option<String>,
    },
    /// Follow one exact vocabulary token ID.
    Token {
        #[arg(long)]
        id: u32,
        #[arg(long)]
        position: Option<String>,
        /// Captured layer IDs or inclusive numeric ranges, in captured order.
        #[arg(long)]
        layers: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
pub(crate) struct TraceDocument {
    pub(crate) schema: String,
    pub(crate) schema_version: u32,
    #[serde(default)]
    producer: Option<Producer>,
    #[serde(default)]
    pub(crate) deployed_model: Option<DeployedModel>,
    #[serde(default)]
    pub(crate) tokenizer: Option<TokenizerSummary>,
    pub(crate) lens: LensSummary,
    #[serde(default)]
    pub(crate) score_semantics: Option<ScoreSemantics>,
    #[serde(default)]
    execution_mode: Option<String>,
    #[serde(default)]
    pub(crate) input_source: Option<String>,
    #[serde(default)]
    pub(crate) add_special_tokens: Option<bool>,
    pub(crate) input_token_ids: Vec<i32>,
    input_tokens: Vec<InputToken>,
    #[serde(default)]
    pub(crate) rendering: Option<Rendering>,
    pub(crate) selected_layers: Vec<u32>,
    pub(crate) top_k: usize,
    pub(crate) cells: Vec<Cell>,
    #[serde(default)]
    pub(crate) vectors: Option<Vectors>,
    timing: BTreeMap<String, f64>,
    occurrences: Occurrences,
    #[serde(default)]
    batch: Option<TraceBatchAttribution>,
}

#[derive(Debug, Deserialize)]
struct TraceBatchAttribution {
    batch_schema: String,
    request_id: String,
    request_index: usize,
    request_count: usize,
    aggregate_rows: usize,
    shared_timing_fields: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Producer {
    #[serde(default)]
    build_commit: Option<String>,
    #[serde(default)]
    build_source_state: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DeployedModel {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    base_model_name: Option<String>,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    pub(crate) locator_id: Option<String>,
    #[serde(default)]
    vocab_size: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TokenizerSummary {
    #[serde(default)]
    pub(crate) metadata_id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    pretokenizer: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LensSummary {
    pub(crate) kind: String,
    pub(crate) method: String,
    #[serde(default)]
    pub(crate) target_layer: Option<u32>,
    #[serde(default)]
    pub(crate) source_repository: Option<String>,
    #[serde(default)]
    pub(crate) source_revision: Option<String>,
    #[serde(default)]
    pub(crate) source_filename: Option<String>,
    #[serde(default)]
    pub(crate) payload_blake3: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ScoreSemantics {
    pub(crate) kind: String,
    #[serde(default)]
    pub(crate) normalization: Option<String>,
    #[serde(default)]
    pub(crate) candidate_universe: Option<String>,
    pub(crate) softmax_applied: bool,
}

type Rendering = LensInputRendering;
type RenderedSpan = LensRenderedSpan;

#[derive(Debug, Deserialize)]
struct InputToken {
    position: usize,
    token_id: i32,
    token_display_lossy: String,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Cell {
    pub(crate) source_layer: u32,
    pub(crate) source_position: usize,
    source_token_id: i32,
    predicts_position: usize,
    pub(crate) top_k: Vec<TokenScore>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct TokenScore {
    pub(crate) rank: usize,
    pub(crate) token_id: u32,
    pub(crate) token_display_lossy: String,
    pub(crate) logit: f32,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Vectors {
    #[serde(default)]
    pub(crate) operation: Option<String>,
    #[serde(default)]
    pub(crate) stage: Option<String>,
    #[serde(default)]
    pub(crate) value_dtype: Option<String>,
    #[serde(default)]
    pub(crate) hidden_coordinate: Option<String>,
    #[serde(default)]
    pub(crate) hidden_size: Option<usize>,
    #[serde(default)]
    pub(crate) shape: Option<[usize; 2]>,
    #[serde(default)]
    pub(crate) cells: Vec<VectorCell>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct VectorCell {
    pub(crate) source_layer: u32,
    pub(crate) source_position: usize,
    source_token_id: i32,
    predicts_position: usize,
    pub(crate) values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct Occurrences {
    global: Vec<Occurrence>,
    per_layer: Vec<LayerOccurrences>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct Occurrence {
    token_id: u32,
    count: usize,
    top1_count: usize,
    best_rank: usize,
}

#[derive(Debug, Deserialize)]
struct LayerOccurrences {
    source_layer: u32,
    tokens: Vec<Occurrence>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "view", rename_all = "snake_case")]
enum ViewResult {
    Summary(SummaryView),
    Aggregate(AggregateView),
    Positions(PositionsView),
    Position(PositionView),
    Token(TokenView),
}

#[derive(Debug, Serialize)]
struct SummaryView {
    schema_version: u32,
    method: String,
    artifact: ArtifactView,
    model: Option<ModelView>,
    token_count: usize,
    layer_count: usize,
    cell_count: usize,
    captured_top_k: usize,
    score: ScoreView,
    rendering: Option<RenderingView>,
    vectors: Option<VectorView>,
    timings_ms: BTreeMap<String, f64>,
    producer: Option<ProducerView>,
    tokenizer: Option<TokenizerView>,
    execution_mode: Option<String>,
    input_source: Option<String>,
    add_special_tokens: Option<bool>,
    batch: Option<TraceBatchView>,
}

#[derive(Debug, Serialize)]
struct TraceBatchView {
    request_id: String,
    request_index: usize,
    request_count: usize,
    aggregate_rows: usize,
    shared_timing_fields: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ArtifactView {
    kind: String,
    source_repository: Option<String>,
    source_revision: Option<String>,
    source_filename: Option<String>,
    payload_blake3: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelView {
    name: Option<String>,
    base_model_name: Option<String>,
    architecture: Option<String>,
    locator_id: Option<String>,
    vocab_size: Option<u32>,
}

#[derive(Debug, Serialize)]
struct ScoreView {
    kind: Option<String>,
    normalization: Option<String>,
    candidate_universe: Option<String>,
    softmax_applied: Option<bool>,
    legacy_unknown: bool,
}

#[derive(Debug, Serialize)]
struct RenderingView {
    renderer: String,
    generation_mode: Option<String>,
    span_count: usize,
}

#[derive(Debug, Serialize)]
struct VectorView {
    operation: Option<String>,
    stage: Option<String>,
    value_dtype: Option<String>,
    hidden_size: Option<usize>,
    shape: Option<[usize; 2]>,
    cell_count: usize,
}

#[derive(Debug, Serialize)]
struct ProducerView {
    build_commit: Option<String>,
    build_source_state: Option<String>,
}

#[derive(Debug, Serialize)]
struct TokenizerView {
    metadata_id: Option<String>,
    model: Option<String>,
    pretokenizer: Option<String>,
}

#[derive(Debug, Serialize)]
struct AggregateView {
    captured_top_k: usize,
    top_k_censored: bool,
    selected_layers: Vec<u32>,
    selected_positions: Vec<usize>,
    rows: Vec<AggregateRow>,
}

#[derive(Debug, Serialize)]
struct PositionsView {
    positions: Vec<InputPositionView>,
    anchors: Vec<SemanticAnchorView>,
    rendering_spans: Vec<RenderedSpan>,
}

#[derive(Debug, Serialize)]
struct InputPositionView {
    position: usize,
    token_id: i32,
    display: String,
    structural_labels: Vec<String>,
    anchor_labels: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct SemanticAnchorView {
    selector: String,
    position: usize,
    message_index: Option<usize>,
    tool_call_index: Option<usize>,
    kind: String,
    label: Option<String>,
}

#[derive(Debug, Serialize)]
struct AggregateRow {
    token_id: u32,
    display: String,
    count: usize,
    top1_count: usize,
    best_rank: usize,
    layer_counts: Vec<LayerCount>,
    intensity_stripe: String,
}

#[derive(Debug, Serialize)]
struct LayerCount {
    source_layer: u32,
    count: usize,
}

#[derive(Debug, Serialize)]
struct PositionView {
    selector: String,
    position: usize,
    source_token_id: i32,
    source_display: String,
    requested_top_k: usize,
    captured_top_k: usize,
    top_k_censored: bool,
    selected_layers: Vec<u32>,
    layers: Vec<PositionLayer>,
}

#[derive(Debug, Serialize)]
struct PositionLayer {
    source_layer: u32,
    scores: Vec<ScoreViewRow>,
}

#[derive(Debug, Serialize)]
struct ScoreViewRow {
    rank: usize,
    token_id: u32,
    display: String,
    logit: f32,
}

#[derive(Debug, Serialize)]
struct TokenView {
    token_id: u32,
    display: Option<String>,
    captured_top_k: usize,
    top_k_censored: bool,
    selected_layers: Vec<u32>,
    global: OccurrenceView,
    global_outside_captured_top_k: bool,
    position: Option<TokenPositionView>,
    layers: Vec<TokenLayerView>,
}

#[derive(Debug, Serialize)]
struct OccurrenceView {
    count: usize,
    top1_count: usize,
    best_rank: Option<usize>,
}

#[derive(Debug, Serialize)]
struct TokenPositionView {
    selector: String,
    position: usize,
    source_token_id: i32,
    source_display: String,
}

#[derive(Debug, Serialize)]
struct TokenLayerView {
    source_layer: u32,
    count: Option<usize>,
    top1_count: Option<usize>,
    best_rank: Option<usize>,
    at_position: Option<TokenAtPosition>,
    outside_captured_top_k: bool,
}

#[derive(Debug, Serialize)]
struct TokenAtPosition {
    rank: usize,
    display: String,
    logit: f32,
}

pub(crate) fn run(args: InspectArgs) -> Result<()> {
    let document = parse_trace(&args.trace)?;
    let result = match args.view {
        InspectView::Summary => ViewResult::Summary(summary_view(&document)),
        InspectView::Aggregate {
            limit,
            layers,
            positions,
        } => ViewResult::Aggregate(aggregate_view(
            &document,
            limit,
            layers.as_deref(),
            &positions,
        )?),
        InspectView::Positions => ViewResult::Positions(positions_view(&document)?),
        InspectView::Position {
            selector,
            top_k,
            layers,
        } => ViewResult::Position(position_view(
            &document,
            &selector,
            top_k,
            layers.as_deref(),
        )?),
        InspectView::Token {
            id,
            position,
            layers,
        } => ViewResult::Token(token_view(
            &document,
            id,
            position.as_deref(),
            layers.as_deref(),
        )?),
    };
    match args.format {
        InspectFormat::Text => print_text(&result),
        InspectFormat::Json => println!("{}", serde_json::to_string_pretty(&result)?),
    }
    Ok(())
}

pub(crate) fn parse_trace(path: &std::path::Path) -> Result<TraceDocument> {
    let bytes = read_regular_file_bounded(path, TRACE_MAX_BYTES)?;
    parse_trace_bytes(&bytes, path)
}

pub(crate) fn parse_trace_bytes(bytes: &[u8], path: &std::path::Path) -> Result<TraceDocument> {
    let document: TraceDocument = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse trace JSON {}", path.display()))?;
    validate_trace(&document)?;
    Ok(document)
}

fn validate_trace(document: &TraceDocument) -> Result<()> {
    ensure!(
        document.schema == "qwen.lens.trace",
        "unknown trace schema {:?}",
        document.schema
    );
    ensure!(
        matches!(document.schema_version, 2 | 3),
        "unsupported trace schema version {}",
        document.schema_version
    );
    if document.schema_version == 3 {
        ensure!(document.producer.is_some(), "v3 trace is missing producer");
        ensure!(
            document.deployed_model.is_some(),
            "v3 trace is missing deployed_model"
        );
        ensure!(
            document.tokenizer.is_some(),
            "v3 trace is missing tokenizer"
        );
        ensure!(
            document.score_semantics.is_some(),
            "v3 trace is missing score_semantics"
        );
        ensure!(
            document.execution_mode.is_some(),
            "v3 trace is missing execution_mode"
        );
        ensure!(
            document.rendering.is_some(),
            "v3 trace is missing rendering"
        );
        ensure!(
            document.input_source.is_some(),
            "v3 trace is missing input_source"
        );
    }
    if let Some(batch) = &document.batch {
        ensure!(
            batch.batch_schema == "qwen.lens.trace_batch",
            "unknown trace batch schema {:?}",
            batch.batch_schema
        );
        ensure!(
            !batch.request_id.is_empty()
                && batch.request_count >= 2
                && batch.request_index < batch.request_count
                && batch.aggregate_rows >= document.input_token_ids.len(),
            "trace batch attribution is inconsistent"
        );
        let expected = [
            "matrix_read_wall_ms",
            "readout_gpu_ms",
            "readout_command_wall_ms",
            "trace_execution_wall_ms",
        ];
        ensure!(
            batch
                .shared_timing_fields
                .iter()
                .map(String::as_str)
                .eq(expected),
            "trace batch shared timing contract is not canonical"
        );
        ensure!(
            expected
                .iter()
                .all(|field| document.timing.contains_key(*field)),
            "trace batch omits one or more shared timing fields"
        );
    }
    ensure!(
        !document.selected_layers.is_empty(),
        "trace has no selected layers"
    );
    ensure!(document.top_k > 0, "trace captured top_k must be positive");
    let layer_set: HashSet<_> = document.selected_layers.iter().copied().collect();
    ensure!(
        layer_set.len() == document.selected_layers.len(),
        "selected layers must be unique"
    );
    ensure!(
        document.input_tokens.len() == document.input_token_ids.len(),
        "input token records do not match input token IDs"
    );
    for (position, token) in document.input_tokens.iter().enumerate() {
        ensure!(
            token.token_id >= 0,
            "input token at slot {position} has a negative token ID"
        );
        ensure!(
            token.position == position && token.token_id == document.input_token_ids[position],
            "input token record at slot {position} is inconsistent"
        );
        if let Some(vocab_size) = document
            .deployed_model
            .as_ref()
            .and_then(|model| model.vocab_size)
        {
            ensure!(
                (token.token_id as u32) < vocab_size,
                "input token ID {} is outside model vocabulary {vocab_size}",
                token.token_id
            );
        }
    }
    if document.schema_version == 2 {
        ensure!(
            document.rendering.is_none(),
            "v2 trace must not contain v3 rendering metadata"
        );
    }
    if let Some(rendering) = &document.rendering {
        ensure!(
            !rendering.renderer.is_empty(),
            "rendering identity is empty"
        );
        let mut previous_byte_end = 0;
        let mut previous_token_end = 0;
        for span in &rendering.spans {
            ensure!(
                is_known_lens_span(&span.kind)
                    && valid_lens_span_metadata(&rendering.renderer, span)
                    && span.role.as_ref().is_none_or(|role| matches!(
                        role.as_str(),
                        "system" | "user" | "assistant" | "tool"
                    ))
                    && span.channel.as_ref().is_none_or(|channel| matches!(
                        channel.as_str(),
                        "thinking" | "tool_call" | "tool_result"
                    )),
                "rendering span has invalid semantic metadata"
            );
            ensure!(
                span.byte_start < span.byte_end,
                "rendering span has an empty or reversed byte range"
            );
            ensure!(
                span.token_start.is_some() == span.token_end.is_some(),
                "rendering span has a partial token range"
            );
            if let (Some(token_start), Some(token_end)) = (span.token_start, span.token_end) {
                ensure!(
                    token_start < token_end && token_end <= document.input_tokens.len(),
                    "rendering span has an invalid token range"
                );
                ensure!(
                    token_start >= previous_token_end,
                    "aligned rendering spans have overlapping or reversed token ranges"
                );
                previous_token_end = token_end;
            }
            if is_structural_marker_kind(&span.kind) {
                ensure!(
                    span.token_start.is_some() && span.token_end.is_some(),
                    "structural rendering marker {} has no exact token range",
                    span.kind
                );
            }
            ensure!(
                span.byte_start >= previous_byte_end,
                "rendering spans are not in byte order"
            );
            if span.message_index.is_none() {
                ensure!(
                    span.kind == "bos_marker"
                        || span.kind.starts_with("generated_")
                        || span.role.is_some(),
                    "unattributed rendering span has no role"
                );
            }
            previous_byte_end = span.byte_end;
        }
        ensure!(
            valid_lens_rendering_topology(rendering),
            "rendering spans have an invalid record topology"
        );
        if document.schema_version == 3 {
            validate_trace_input_rendering(document, rendering)?;
        }
    }
    let expected_cells = document
        .selected_layers
        .len()
        .checked_mul(document.input_tokens.len())
        .context("trace cell count overflow")?;
    ensure!(
        document.cells.len() == expected_cells,
        "trace has {} cells, expected {expected_cells}",
        document.cells.len()
    );
    let mut coordinates = HashSet::new();
    for cell in &document.cells {
        ensure!(
            layer_set.contains(&cell.source_layer),
            "cell refers to unselected layer {}",
            cell.source_layer
        );
        ensure!(
            cell.source_position < document.input_tokens.len(),
            "cell position {} is outside the input",
            cell.source_position
        );
        ensure!(
            coordinates.insert((cell.source_layer, cell.source_position)),
            "duplicate cell at layer {} position {}",
            cell.source_layer,
            cell.source_position
        );
        ensure!(
            cell.source_token_id == document.input_token_ids[cell.source_position]
                && cell.predicts_position == cell.source_position + 1,
            "cell coordinates are inconsistent at layer {} position {}",
            cell.source_layer,
            cell.source_position
        );
        ensure!(
            cell.top_k.len() <= document.top_k,
            "cell top-k exceeds captured top_k"
        );
        let mut token_ids = HashSet::new();
        for (rank, score) in cell.top_k.iter().enumerate() {
            if let Some(vocab_size) = document
                .deployed_model
                .as_ref()
                .and_then(|model| model.vocab_size)
            {
                ensure!(
                    score.token_id < vocab_size,
                    "top-k token ID {} is outside model vocabulary {vocab_size}",
                    score.token_id
                );
            }
            ensure!(
                score.rank == rank,
                "cell top-k ranks are not zero-based and ordered"
            );
            ensure!(
                token_ids.insert(score.token_id),
                "cell top-k repeats token ID {}",
                score.token_id
            );
            ensure!(
                score.logit.is_finite(),
                "cell top-k contains a non-finite logit"
            );
        }
    }
    if let Some(vectors) = &document.vectors {
        let hidden_size = vectors
            .hidden_size
            .context("trace vectors are missing hidden_size")?;
        ensure!(hidden_size > 0, "trace vector hidden_size must be positive");
        ensure!(
            vectors.shape == Some([vectors.cells.len(), hidden_size]),
            "trace vector shape is inconsistent with cells and hidden_size"
        );
        let mut vector_coordinates = HashSet::new();
        for vector in &vectors.cells {
            ensure!(
                coordinates.contains(&(vector.source_layer, vector.source_position)),
                "vector refers to an uncaptured cell at layer {} position {}",
                vector.source_layer,
                vector.source_position
            );
            ensure!(
                vector_coordinates.insert((vector.source_layer, vector.source_position)),
                "duplicate vector at layer {} position {}",
                vector.source_layer,
                vector.source_position
            );
            ensure!(
                vector.source_token_id == document.input_token_ids[vector.source_position]
                    && vector.predicts_position == vector.source_position + 1,
                "vector coordinates are inconsistent at layer {} position {}",
                vector.source_layer,
                vector.source_position
            );
            ensure!(
                vector.values.len() == hidden_size,
                "vector at layer {} position {} has dimension {}, expected {hidden_size}",
                vector.source_layer,
                vector.source_position,
                vector.values.len()
            );
            ensure!(
                vector.values.iter().all(|value| value.is_finite()),
                "vector at layer {} position {} contains a non-finite value",
                vector.source_layer,
                vector.source_position
            );
        }
    }
    validate_occurrences(document)?;
    Ok(())
}

fn validate_occurrences(document: &TraceDocument) -> Result<()> {
    let mut layer_membership = HashSet::new();
    ensure!(
        document.occurrences.per_layer.len() == document.selected_layers.len(),
        "occurrence layers do not match selected layers"
    );
    for layer in &document.occurrences.per_layer {
        ensure!(
            document.selected_layers.contains(&layer.source_layer)
                && layer_membership.insert(layer.source_layer),
            "occurrences refer to a duplicate or unselected layer {}",
            layer.source_layer
        );
    }
    let (global, per_layer) = compute_occurrences(&document.cells, &document.selected_layers);
    ensure!(
        document.occurrences.global == global,
        "global occurrences are inconsistent with cells"
    );
    for layer in &document.occurrences.per_layer {
        ensure!(
            per_layer.get(&layer.source_layer) == Some(&layer.tokens),
            "layer {} occurrences are inconsistent with cells",
            layer.source_layer
        );
    }
    Ok(())
}

fn compute_occurrences(
    cells: &[Cell],
    layers: &[u32],
) -> (Vec<Occurrence>, BTreeMap<u32, Vec<Occurrence>>) {
    let mut global = BTreeMap::<u32, (usize, usize, usize)>::new();
    let mut per_layer = BTreeMap::<u32, BTreeMap<u32, (usize, usize, usize)>>::new();
    for cell in cells {
        for score in &cell.top_k {
            update_occurrence(&mut global, score.token_id, score.rank);
            update_occurrence(
                per_layer.entry(cell.source_layer).or_default(),
                score.token_id,
                score.rank,
            );
        }
    }
    let global = sort_occurrences(global);
    let per_layer = layers
        .iter()
        .map(|layer| {
            (
                *layer,
                sort_occurrences(per_layer.remove(layer).unwrap_or_default()),
            )
        })
        .collect();
    (global, per_layer)
}

fn update_occurrence(map: &mut BTreeMap<u32, (usize, usize, usize)>, token_id: u32, rank: usize) {
    map.entry(token_id)
        .and_modify(|entry| {
            entry.0 += 1;
            entry.1 += usize::from(rank == 0);
            entry.2 = entry.2.min(rank);
        })
        .or_insert((1, usize::from(rank == 0), rank));
}

fn sort_occurrences(map: BTreeMap<u32, (usize, usize, usize)>) -> Vec<Occurrence> {
    let mut rows: Vec<_> = map
        .into_iter()
        .map(|(token_id, (count, top1_count, best_rank))| Occurrence {
            token_id,
            count,
            top1_count,
            best_rank,
        })
        .collect();
    rows.sort_unstable_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| right.top1_count.cmp(&left.top1_count))
            .then_with(|| left.best_rank.cmp(&right.best_rank))
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
    rows
}

fn summary_view(document: &TraceDocument) -> SummaryView {
    SummaryView {
        schema_version: document.schema_version,
        method: document.lens.method.clone(),
        artifact: ArtifactView {
            kind: document.lens.kind.clone(),
            source_repository: document.lens.source_repository.clone(),
            source_revision: document.lens.source_revision.clone(),
            source_filename: document.lens.source_filename.clone(),
            payload_blake3: document.lens.payload_blake3.clone(),
        },
        model: document.deployed_model.as_ref().map(|model| ModelView {
            name: model.name.clone(),
            base_model_name: model.base_model_name.clone(),
            architecture: model.architecture.clone(),
            locator_id: model.locator_id.clone(),
            vocab_size: model.vocab_size,
        }),
        token_count: document.input_tokens.len(),
        layer_count: document.selected_layers.len(),
        cell_count: document.cells.len(),
        captured_top_k: document.top_k,
        score: document.score_semantics.as_ref().map_or(
            ScoreView {
                kind: None,
                normalization: None,
                candidate_universe: None,
                softmax_applied: None,
                legacy_unknown: true,
            },
            |score| ScoreView {
                kind: Some(score.kind.clone()),
                normalization: score.normalization.clone(),
                candidate_universe: score.candidate_universe.clone(),
                softmax_applied: Some(score.softmax_applied),
                legacy_unknown: false,
            },
        ),
        rendering: document.rendering.as_ref().map(|rendering| RenderingView {
            renderer: rendering.renderer.clone(),
            generation_mode: rendering.generation_mode.clone(),
            span_count: rendering.spans.len(),
        }),
        vectors: document.vectors.as_ref().map(|vectors| VectorView {
            operation: vectors.operation.clone(),
            stage: vectors.stage.clone(),
            value_dtype: vectors.value_dtype.clone(),
            hidden_size: vectors.hidden_size,
            shape: vectors.shape,
            cell_count: vectors.cells.len(),
        }),
        timings_ms: document.timing.clone(),
        producer: document.producer.as_ref().map(|producer| ProducerView {
            build_commit: producer.build_commit.clone(),
            build_source_state: producer.build_source_state.clone(),
        }),
        tokenizer: document.tokenizer.as_ref().map(|tokenizer| TokenizerView {
            metadata_id: tokenizer.metadata_id.clone(),
            model: tokenizer.model.clone(),
            pretokenizer: tokenizer.pretokenizer.clone(),
        }),
        execution_mode: document.execution_mode.clone(),
        input_source: document.input_source.clone(),
        add_special_tokens: document.add_special_tokens,
        batch: document.batch.as_ref().map(|batch| TraceBatchView {
            request_id: batch.request_id.clone(),
            request_index: batch.request_index,
            request_count: batch.request_count,
            aggregate_rows: batch.aggregate_rows,
            shared_timing_fields: batch.shared_timing_fields.clone(),
        }),
    }
}

fn validate_trace_input_rendering(document: &TraceDocument, rendering: &Rendering) -> Result<()> {
    let source = document
        .input_source
        .as_deref()
        .expect("v3 input source checked");
    let architecture = document
        .deployed_model
        .as_ref()
        .and_then(|model| model.architecture.as_deref());
    let ordinary_qwen = matches!(architecture, Some("qwen35" | "qwen35moe"));
    let muse = architecture == Some("muse-glimmer");
    let valid = match source {
        "prompt" => {
            document.add_special_tokens.is_some()
                && (ordinary_qwen && rendering.renderer == "tokenizer_text"
                    || muse && rendering.renderer == "muse_tokenizer_raw_prompt")
                && rendering.generation_mode.is_none()
                && rendering.spans.is_empty()
        }
        "token_ids" => {
            (ordinary_qwen || muse)
                && document.add_special_tokens.is_none()
                && rendering.renderer == "literal_token_ids"
                && rendering.generation_mode.is_none()
                && rendering.spans.is_empty()
        }
        "messages" => {
            document.add_special_tokens == Some(false)
                && match rendering.renderer.as_str() {
                    "qwen_chatml_messages_v1" if ordinary_qwen => {
                        rendering.generation_mode.as_deref() == Some("auto")
                    }
                    "qwen3.6_messages_v1" if ordinary_qwen => rendering
                        .generation_mode
                        .as_deref()
                        .is_some_and(|mode| matches!(mode, "auto" | "thinking" | "no_thinking")),
                    "qwen3.8_messages_v1" if ordinary_qwen => {
                        rendering.generation_mode.as_deref().is_some_and(|mode| {
                            matches!(
                                mode,
                                "thinking_low"
                                    | "thinking_medium"
                                    | "thinking_xhigh"
                                    | "no_thinking"
                            )
                        })
                    }
                    "muse_glimmer_atem_v1" if muse => {
                        rendering.generation_mode.as_deref().is_some_and(|mode| {
                            matches!(
                                mode,
                                "reasoning_low"
                                    | "reasoning_medium"
                                    | "reasoning_high"
                                    | "reasoning_xhigh"
                            )
                        }) && rendering.spans.is_empty()
                    }
                    "muse_glimmer_atem_annotated_v1" if muse => {
                        rendering.generation_mode.as_deref().is_some_and(|mode| {
                            matches!(
                                mode,
                                "reasoning_low"
                                    | "reasoning_medium"
                                    | "reasoning_high"
                                    | "reasoning_xhigh"
                            )
                        }) && !rendering.spans.is_empty()
                    }
                    _ => false,
                }
        }
        "open_responses" => {
            ordinary_qwen
                && document.add_special_tokens == Some(false)
                && rendering.renderer == "qwen_open_responses_annotated_v1"
                && rendering.generation_mode.as_deref().is_some_and(|mode| {
                    matches!(
                        mode,
                        "auto"
                            | "thinking_low"
                            | "thinking_medium"
                            | "thinking_xhigh"
                            | "no_thinking"
                    )
                })
                && !rendering.spans.is_empty()
        }
        _ => false,
    };
    ensure!(valid, "trace input rendering metadata is inconsistent");
    Ok(())
}

fn display_map(document: &TraceDocument) -> HashMap<u32, String> {
    let mut displays = HashMap::new();
    for cell in &document.cells {
        for score in &cell.top_k {
            displays
                .entry(score.token_id)
                .or_insert_with(|| score.token_display_lossy.clone());
        }
    }
    displays
}

fn select_layers(document: &TraceDocument, spec: Option<&str>) -> Result<Vec<u32>> {
    let Some(spec) = spec else {
        return Ok(document.selected_layers.clone());
    };
    ensure!(!spec.is_empty(), "--layers must not be empty");
    let captured: HashSet<_> = document.selected_layers.iter().copied().collect();
    let mut selected = HashSet::new();
    for item in spec.split(',') {
        ensure!(!item.is_empty(), "--layers contains an empty selection");
        if let Some((start, end)) = item.split_once("..") {
            ensure!(
                !start.is_empty() && !end.is_empty() && !end.contains(".."),
                "layer range {item:?} must be START..END"
            );
            let start: u32 = start
                .parse()
                .with_context(|| format!("invalid layer range start {start:?}"))?;
            let end: u32 = end
                .parse()
                .with_context(|| format!("invalid layer range end {end:?}"))?;
            ensure!(start <= end, "layer range {item:?} is reversed");
            let matches: Vec<_> = document
                .selected_layers
                .iter()
                .copied()
                .filter(|layer| (start..=end).contains(layer))
                .collect();
            ensure!(
                !matches.is_empty(),
                "layer range {item:?} selects no captured layers"
            );
            for layer in matches {
                ensure!(
                    selected.insert(layer),
                    "layer {layer} is selected more than once"
                );
            }
        } else {
            let layer: u32 = item
                .parse()
                .with_context(|| format!("invalid layer ID {item:?}"))?;
            ensure!(captured.contains(&layer), "layer {layer} was not captured");
            ensure!(
                selected.insert(layer),
                "layer {layer} is selected more than once"
            );
        }
    }
    ensure!(!selected.is_empty(), "--layers selects no captured layers");
    Ok(document
        .selected_layers
        .iter()
        .copied()
        .filter(|layer| selected.contains(layer))
        .collect())
}

fn select_positions(document: &TraceDocument, selectors: &[String]) -> Result<Vec<usize>> {
    if selectors.is_empty() {
        return Ok((0..document.input_tokens.len()).collect());
    }
    let mut selected = HashSet::new();
    selectors
        .iter()
        .map(|selector| {
            let position = resolve_position(document, selector)?;
            ensure!(
                selected.insert(position),
                "position {position} is selected more than once"
            );
            Ok(position)
        })
        .collect()
}

fn aggregate_view(
    document: &TraceDocument,
    limit: usize,
    layer_spec: Option<&str>,
    position_selectors: &[String],
) -> Result<AggregateView> {
    let selected_layers = select_layers(document, layer_spec)?;
    let selected_positions = select_positions(document, position_selectors)?;
    let layer_set: HashSet<_> = selected_layers.iter().copied().collect();
    let position_set: HashSet<_> = selected_positions.iter().copied().collect();
    let cells: Vec<_> = document
        .cells
        .iter()
        .filter(|cell| {
            layer_set.contains(&cell.source_layer) && position_set.contains(&cell.source_position)
        })
        .cloned()
        .collect();
    let (global, per_layer) = compute_occurrences(&cells, &selected_layers);
    let displays = display_map(document);
    let rows = global
        .iter()
        .take(limit)
        .map(|occurrence| {
            let layer_counts: Vec<_> = selected_layers
                .iter()
                .map(|layer| LayerCount {
                    source_layer: *layer,
                    count: per_layer
                        .get(layer)
                        .and_then(|tokens| {
                            tokens
                                .iter()
                                .find(|row| row.token_id == occurrence.token_id)
                        })
                        .map_or(0, |row| row.count),
                })
                .collect();
            let intensity_stripe = intensity_stripe(&layer_counts);
            AggregateRow {
                token_id: occurrence.token_id,
                display: displays
                    .get(&occurrence.token_id)
                    .cloned()
                    .unwrap_or_default(),
                count: occurrence.count,
                top1_count: occurrence.top1_count,
                best_rank: occurrence.best_rank,
                layer_counts,
                intensity_stripe,
            }
        })
        .collect();
    Ok(AggregateView {
        captured_top_k: document.top_k,
        top_k_censored: true,
        selected_layers,
        selected_positions,
        rows,
    })
}

fn intensity_stripe(counts: &[LayerCount]) -> String {
    const LEVELS: &[u8] = b" .:-=+*#%@";
    let maximum = counts.iter().map(|row| row.count).max().unwrap_or(0);
    counts
        .iter()
        .map(|row| {
            let index = if maximum == 0 {
                0
            } else {
                row.count * (LEVELS.len() - 1) / maximum
            };
            LEVELS[index] as char
        })
        .collect()
}

fn position_view(
    document: &TraceDocument,
    selector: &str,
    top_k: Option<usize>,
    layer_spec: Option<&str>,
) -> Result<PositionView> {
    let position = resolve_position(document, selector)?;
    let selected_layers = select_layers(document, layer_spec)?;
    let requested_top_k = top_k.unwrap_or(document.top_k);
    ensure!(requested_top_k > 0, "--top-k must be positive");
    let shown = requested_top_k.min(document.top_k);
    let layers = selected_layers
        .iter()
        .map(|layer| {
            let cell = cell_at(document, *layer, position).expect("validated cell grid");
            PositionLayer {
                source_layer: *layer,
                scores: cell.top_k.iter().take(shown).map(score_view_row).collect(),
            }
        })
        .collect();
    let source = &document.input_tokens[position];
    Ok(PositionView {
        selector: selector.to_owned(),
        position,
        source_token_id: source.token_id,
        source_display: source.token_display_lossy.clone(),
        requested_top_k,
        captured_top_k: document.top_k,
        top_k_censored: true,
        selected_layers,
        layers,
    })
}

fn score_view_row(score: &TokenScore) -> ScoreViewRow {
    ScoreViewRow {
        rank: score.rank,
        token_id: score.token_id,
        display: score.token_display_lossy.clone(),
        logit: score.logit,
    }
}

fn positions_view(document: &TraceDocument) -> Result<PositionsView> {
    let anchors = semantic_anchors(document)?;
    let mut structural = vec![Vec::new(); document.input_tokens.len()];
    if let Some(rendering) = &document.rendering {
        for span in rendering
            .spans
            .iter()
            .filter(|span| is_structural_marker_kind(&span.kind))
        {
            let (Some(start), Some(end)) = (span.token_start, span.token_end) else {
                continue;
            };
            let mut label = span.kind.clone();
            if let Some(message_index) = span.message_index {
                label.push_str(&format!("[message={message_index}]"));
            }
            if let Some(role) = &span.role {
                label.push_str(&format!("[role={role}]"));
            }
            if let Some(channel) = &span.channel {
                label.push_str(&format!("[channel={channel}]"));
            }
            if let Some(tool_call_index) = span.tool_call_index {
                label.push_str(&format!("[tool_call={tool_call_index}]"));
            }
            if let Some(value) = &span.label {
                label.push_str(&format!("[label={value:?}]"));
            }
            for labels in &mut structural[start..end] {
                labels.push(label.clone());
            }
        }
    }
    let mut anchor_labels = vec![Vec::new(); document.input_tokens.len()];
    for anchor in &anchors {
        anchor_labels[anchor.position].push(anchor.selector.clone());
    }
    let positions = document
        .input_tokens
        .iter()
        .map(|token| InputPositionView {
            position: token.position,
            token_id: token.token_id,
            display: token.token_display_lossy.clone(),
            structural_labels: std::mem::take(&mut structural[token.position]),
            anchor_labels: std::mem::take(&mut anchor_labels[token.position]),
        })
        .collect();
    Ok(PositionsView {
        positions,
        anchors,
        rendering_spans: document
            .rendering
            .as_ref()
            .map(|rendering| rendering.spans.clone())
            .unwrap_or_default(),
    })
}

fn semantic_anchors(document: &TraceDocument) -> Result<Vec<SemanticAnchorView>> {
    let mut anchors = Vec::new();
    if let Some(position) = document.input_tokens.len().checked_sub(1) {
        anchors.push(SemanticAnchorView {
            selector: "prefill:last".into(),
            position,
            message_index: None,
            tool_call_index: None,
            kind: "prefill_last".into(),
            label: None,
        });
    }
    let Some(rendering) = &document.rendering else {
        return Ok(anchors);
    };
    let mut message_edges =
        BTreeMap::<usize, (Option<&RenderedSpan>, Option<&RenderedSpan>)>::new();
    for span in &rendering.spans {
        let Some(message_index) = span.message_index else {
            continue;
        };
        let edges = message_edges.entry(message_index).or_default();
        match span.kind.as_str() {
            "message_start_marker" if edges.0.is_none() => edges.0 = Some(span),
            "message_end_marker" => edges.1 = Some(span),
            _ => {}
        }
    }
    for (message_index, (start, end)) in message_edges {
        for (edge, span) in [("start", start), ("end", end)] {
            let Some(span) = span else { continue };
            let Some(position) = span.token_start else {
                continue;
            };
            anchors.push(SemanticAnchorView {
                selector: format!("message:{message_index}:{edge}"),
                position,
                message_index: Some(message_index),
                tool_call_index: span.tool_call_index,
                kind: span.kind.clone(),
                label: span.label.clone(),
            });
        }
    }
    if let Some(span) = rendering
        .spans
        .iter()
        .find(|span| span.kind == "generated_assistant_start_marker")
        && let Some(position) = span.token_start
    {
        anchors.push(SemanticAnchorView {
            selector: "generated:assistant:start".into(),
            position,
            message_index: None,
            tool_call_index: span.tool_call_index,
            kind: span.kind.clone(),
            label: span.label.clone(),
        });
    }
    let roles = rendering
        .spans
        .iter()
        .filter_map(|span| span.role.clone())
        .collect::<BTreeSet<_>>();
    for role in roles {
        for edge in ["start", "end"] {
            let selector = format!("role:{role}:{edge}");
            if let Some(span) = resolved_structural_span(document, &selector)? {
                let Some(position) = span.token_start else {
                    continue;
                };
                anchors.push(SemanticAnchorView {
                    selector,
                    position,
                    message_index: span.message_index,
                    tool_call_index: span.tool_call_index,
                    kind: format!("role_{edge}"),
                    label: Some(role.clone()),
                });
            }
        }
    }
    let mut channels = BTreeSet::new();
    for span in &rendering.spans {
        if let Some(channel) = &span.channel {
            channels.insert(channel.clone());
        }
    }
    for channel in channels {
        for edge in ["start", "end"] {
            let selector = format!("channel:{channel}:{edge}");
            if let Ok(position) = resolve_position(document, &selector) {
                let span = resolved_structural_span(document, &selector)?;
                let message_index = span.and_then(|span| span.message_index);
                anchors.push(SemanticAnchorView {
                    selector,
                    position,
                    message_index,
                    tool_call_index: span.and_then(|span| span.tool_call_index),
                    kind: format!("channel_{edge}"),
                    label: Some(channel.clone()),
                });
            }
        }
    }
    Ok(anchors)
}

fn resolve_position(document: &TraceDocument, selector: &str) -> Result<usize> {
    if !selector.is_empty() && selector.bytes().all(|byte| byte.is_ascii_digit()) {
        let position: usize = selector
            .parse()
            .with_context(|| format!("position {selector:?} does not fit usize"))?;
        ensure!(
            position < document.input_tokens.len(),
            "position {position} is outside {} input tokens",
            document.input_tokens.len()
        );
        return Ok(position);
    }
    if selector == "prefill:last" {
        return document
            .input_tokens
            .len()
            .checked_sub(1)
            .context("prefill:last cannot resolve in an empty trace");
    }
    ensure!(
        document.schema_version == 3 && document.rendering.is_some(),
        "semantic selector {selector:?} requires v3 authored rendering spans"
    );
    let span = resolved_structural_span(document, selector)?.with_context(|| {
        format!("semantic selector {selector:?} matched no authored structural marker")
    })?;
    let (_, _, edge) = parse_semantic_selector(selector)?;
    if edge == "end"
        && !matches!(
            span.kind.as_str(),
            "message_end_marker" | "thinking_channel_end_marker"
        )
    {
        return span
            .token_end
            .and_then(|end| end.checked_sub(1))
            .with_context(|| {
                format!(
                    "semantic selector {selector:?} matched content without an exact token range"
                )
            });
    }
    span.token_start.with_context(|| {
        format!("semantic selector {selector:?} matched a marker without an exact token range")
    })
}

fn resolved_structural_span<'a>(
    document: &'a TraceDocument,
    selector: &str,
) -> Result<Option<&'a RenderedSpan>> {
    let rendering = document
        .rendering
        .as_ref()
        .expect("caller checked rendering");
    let (domain, value, edge) = parse_semantic_selector(selector)?;
    let span = match domain {
        "role" => find_structural_span(&rendering.spans, domain, value, edge),
        "message" => {
            let index: usize = value
                .parse()
                .with_context(|| format!("message index {value:?} does not fit usize"))?;
            if edge == "start" {
                rendering.spans.iter().find(|span| {
                    span.message_index == Some(index) && span.kind == "message_start_marker"
                })
            } else {
                rendering.spans.iter().rev().find(|span| {
                    span.message_index == Some(index) && span.kind == "message_end_marker"
                })
            }
        }
        "generated" if value == "assistant" && edge == "start" => rendering
            .spans
            .iter()
            .find(|span| span.kind == "generated_assistant_start_marker"),
        "channel" => {
            let generated_start = rendering.spans.iter().enumerate().rev().find(|(_, span)| {
                span.kind == "thinking_channel_start_marker"
                    && span.channel.as_deref() == Some(value)
                    && span.message_index.is_none()
            });
            if let Some((start_index, start)) = generated_start {
                if edge == "start" {
                    Some(start)
                } else {
                    Some(
                        rendering.spans[start_index + 1..]
                            .iter()
                            .find(|span| {
                                span.kind == "thinking_channel_end_marker"
                                    && span.channel.as_deref() == Some(value)
                                    && span.message_index.is_none()
                            })
                            .with_context(|| {
                                format!(
                                    "semantic selector {selector:?} has an open generated channel and no generated end marker"
                                )
                            })?,
                    )
                }
            } else {
                match edge {
                    "start" => rendering
                        .spans
                        .iter()
                        .find(|span| span.channel.as_deref() == Some(value)),
                    "end" => rendering
                        .spans
                        .iter()
                        .rev()
                        .find(|span| span.channel.as_deref() == Some(value)),
                    _ => None,
                }
            }
        }
        _ => None,
    };
    Ok(span)
}

fn is_structural_marker_kind(kind: &str) -> bool {
    is_structural_lens_span(kind)
}

fn parse_semantic_selector(selector: &str) -> Result<(&str, &str, &str)> {
    let mut parts = selector.split(':');
    let domain = parts.next().unwrap_or_default();
    let value = parts.next().unwrap_or_default();
    let edge = parts.next().unwrap_or_default();
    ensure!(
        parts.next().is_none()
            && matches!(domain, "role" | "channel" | "message" | "generated")
            && !value.is_empty()
            && matches!(edge, "start" | "end"),
        "selector must be an unsigned integer, prefill:last, message:N:start|end, role:ROLE:start|end, generated:assistant:start, or channel:CHANNEL:start|end"
    );
    Ok((domain, value, edge))
}

fn find_structural_span<'a>(
    spans: &'a [RenderedSpan],
    domain: &str,
    value: &str,
    edge: &str,
) -> Option<&'a RenderedSpan> {
    let message_index = spans
        .iter()
        .filter(|span| span.role.as_deref() == Some(value))
        .filter_map(|span| span.message_index)
        .max()?;
    match (domain, edge) {
        ("role", "start") => spans.iter().find(|span| {
            span.kind == "message_start_marker"
                && span.role.as_deref() == Some(value)
                && span.message_index == Some(message_index)
        }),
        ("role", "end") => spans.iter().rev().find(|span| {
            span.kind == "message_end_marker"
                && span.role.as_deref() == Some(value)
                && span.message_index == Some(message_index)
        }),
        _ => None,
    }
}

fn token_view(
    document: &TraceDocument,
    token_id: u32,
    selector: Option<&str>,
    layer_spec: Option<&str>,
) -> Result<TokenView> {
    if let Some(vocab_size) = document
        .deployed_model
        .as_ref()
        .and_then(|model| model.vocab_size)
    {
        ensure!(
            token_id < vocab_size,
            "token ID {token_id} is outside model vocabulary {vocab_size}"
        );
    }
    let selected_layers = select_layers(document, layer_spec)?;
    let layer_set: HashSet<_> = selected_layers.iter().copied().collect();
    let scoped_cells: Vec<_> = document
        .cells
        .iter()
        .filter(|cell| layer_set.contains(&cell.source_layer))
        .cloned()
        .collect();
    let (scoped_global, scoped_per_layer) = compute_occurrences(&scoped_cells, &selected_layers);
    let displays = display_map(document);
    let global_occurrence = scoped_global.iter().find(|row| row.token_id == token_id);
    let global = global_occurrence.map_or(
        OccurrenceView {
            count: 0,
            top1_count: 0,
            best_rank: None,
        },
        occurrence_view,
    );
    let mut layers = Vec::with_capacity(selected_layers.len());
    let position = selector
        .map(|selector| resolve_position(document, selector))
        .transpose()?;
    if let Some(position) = position {
        for layer in &selected_layers {
            let score = cell_at(document, *layer, position)
                .expect("validated cell grid")
                .top_k
                .iter()
                .find(|score| score.token_id == token_id);
            layers.push(TokenLayerView {
                source_layer: *layer,
                count: None,
                top1_count: None,
                best_rank: None,
                at_position: score.map(|score| TokenAtPosition {
                    rank: score.rank,
                    display: score.token_display_lossy.clone(),
                    logit: score.logit,
                }),
                outside_captured_top_k: score.is_none(),
            });
        }
    } else {
        for layer in &selected_layers {
            let occurrence = scoped_per_layer
                .get(layer)
                .and_then(|tokens| tokens.iter().find(|row| row.token_id == token_id));
            layers.push(TokenLayerView {
                source_layer: *layer,
                count: Some(occurrence.map_or(0, |row| row.count)),
                top1_count: Some(occurrence.map_or(0, |row| row.top1_count)),
                best_rank: occurrence.map(|row| row.best_rank),
                at_position: None,
                outside_captured_top_k: occurrence.is_none(),
            });
        }
    }
    let position_view = position.map(|position| TokenPositionView {
        selector: selector.unwrap().to_owned(),
        position,
        source_token_id: document.input_tokens[position].token_id,
        source_display: document.input_tokens[position].token_display_lossy.clone(),
    });
    Ok(TokenView {
        token_id,
        display: displays.get(&token_id).cloned(),
        captured_top_k: document.top_k,
        top_k_censored: true,
        selected_layers,
        global,
        global_outside_captured_top_k: global_occurrence.is_none(),
        position: position_view,
        layers,
    })
}

fn occurrence_view(row: &Occurrence) -> OccurrenceView {
    OccurrenceView {
        count: row.count,
        top1_count: row.top1_count,
        best_rank: Some(row.best_rank),
    }
}

fn cell_at(document: &TraceDocument, layer: u32, position: usize) -> Option<&Cell> {
    document
        .cells
        .iter()
        .find(|cell| cell.source_layer == layer && cell.source_position == position)
}

fn print_text(result: &ViewResult) {
    match result {
        ViewResult::Summary(view) => print_summary(view),
        ViewResult::Aggregate(view) => print_aggregate(view),
        ViewResult::Positions(view) => print_positions(view),
        ViewResult::Position(view) => print_position(view),
        ViewResult::Token(view) => print_token(view),
    }
}

fn print_summary(view: &SummaryView) {
    println!(
        "{} {} | schema v{}",
        view.method.to_uppercase(),
        view.artifact.kind,
        view.schema_version
    );
    if let Some(source) = view
        .artifact
        .source_filename
        .as_deref()
        .or(view.artifact.source_repository.as_deref())
    {
        println!("artifact {source}");
    }
    if let Some(model) = &view.model {
        if let Some(name) = model
            .name
            .as_deref()
            .or(model.base_model_name.as_deref())
            .or(model.architecture.as_deref())
        {
            println!("model {name}");
        }
    }
    println!(
        "{} tokens x {} layers = {} cells | captured top-k {}",
        view.token_count, view.layer_count, view.cell_count, view.captured_top_k
    );
    if let Some(source) = &view.input_source {
        println!(
            "input {source} | add_special_tokens {}",
            view.add_special_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".into())
        );
    }
    if view.score.legacy_unknown {
        println!("score legacy unknown | softmax unknown");
    } else {
        println!(
            "score {} | normalization {} | candidates {} | softmax {}",
            view.score.kind.as_deref().unwrap_or("unknown"),
            view.score.normalization.as_deref().unwrap_or("unknown"),
            view.score
                .candidate_universe
                .as_deref()
                .unwrap_or("unknown"),
            view.score.softmax_applied.unwrap()
        );
    }
    if let Some(rendering) = &view.rendering {
        println!(
            "renderer {} | spans {}{}",
            rendering.renderer,
            rendering.span_count,
            rendering
                .generation_mode
                .as_ref()
                .map(|mode| format!(" | generation {mode}"))
                .unwrap_or_default()
        );
    }
    if let Some(vectors) = &view.vectors {
        println!(
            "vectors {} cells{}",
            vectors.cell_count,
            vectors
                .shape
                .map(|shape| format!(" | shape {}x{}", shape[0], shape[1]))
                .unwrap_or_default()
        );
    }
    if let Some(batch) = &view.batch {
        println!(
            "batch request {} ({}/{}) | aggregate rows {} | shared timings {}",
            batch.request_id,
            batch.request_index + 1,
            batch.request_count,
            batch.aggregate_rows,
            batch.shared_timing_fields.join(",")
        );
    }
    if !view.timings_ms.is_empty() {
        println!(
            "timings {}",
            view.timings_ms
                .iter()
                .map(|(name, value)| format!("{name}={value:.1}ms"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}

fn print_aggregate(view: &AggregateView) {
    println!(
        "aggregate | captured top-k {} (top-k censored)",
        view.captured_top_k
    );
    println!(
        "layers {} | positions {}",
        view.selected_layers
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
        view.selected_positions
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("token_id  count  top1  best  stripe  layer_counts  display");
    for row in &view.rows {
        let counts = row
            .layer_counts
            .iter()
            .map(|entry| format!("{}:{}", entry.source_layer, entry.count))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{:<9} {:<6} {:<5} {:<5} |{}|  {:<12} {:?}",
            row.token_id,
            row.count,
            row.top1_count,
            row.best_rank,
            row.intensity_stripe,
            counts,
            row.display
        );
    }
}

fn print_positions(view: &PositionsView) {
    for position in &view.positions {
        let mut labels = position.structural_labels.clone();
        labels.extend(
            position
                .anchor_labels
                .iter()
                .map(|label| format!("@{label}")),
        );
        println!(
            "{} id={} display={:?}{}",
            position.position,
            position.token_id,
            position.display,
            if labels.is_empty() {
                String::new()
            } else {
                format!(" labels={}", labels.join(","))
            }
        );
    }
    println!("anchors");
    for anchor in &view.anchors {
        println!(
            "{}={}{}{} kind={}{}",
            anchor.selector,
            anchor.position,
            anchor
                .message_index
                .map(|index| format!(" message={index}"))
                .unwrap_or_default(),
            anchor
                .tool_call_index
                .map(|index| format!(" tool_call={index}"))
                .unwrap_or_default(),
            anchor.kind,
            anchor
                .label
                .as_ref()
                .map(|label| format!(" label={label:?}"))
                .unwrap_or_default(),
        );
    }
    println!("rendering_spans");
    for (index, span) in view.rendering_spans.iter().enumerate() {
        println!(
            "{} kind={} bytes={}..{} tokens={} role={} channel={} message={} tool_call={} label={}",
            index,
            span.kind,
            span.byte_start,
            span.byte_end,
            span.token_start
                .zip(span.token_end)
                .map_or_else(|| "-".into(), |(start, end)| format!("{start}..{end}")),
            span.role.as_deref().unwrap_or("-"),
            span.channel.as_deref().unwrap_or("-"),
            span.message_index
                .map_or_else(|| "-".into(), |value| value.to_string()),
            span.tool_call_index
                .map_or_else(|| "-".into(), |value| value.to_string()),
            span.label
                .as_ref()
                .map_or_else(|| "-".into(), |value| format!("{value:?}")),
        );
    }
}

fn print_position(view: &PositionView) {
    println!(
        "position {} ({}) | source token {} {:?} | showing up to {} of captured top-k {}",
        view.position,
        view.selector,
        view.source_token_id,
        view.source_display,
        view.requested_top_k.min(view.captured_top_k),
        view.captured_top_k
    );
    println!(
        "layers {} | top-k-censored={}",
        view.selected_layers
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
        view.top_k_censored
    );
    for layer in &view.layers {
        println!("layer {}", layer.source_layer);
        for score in &layer.scores {
            println!(
                "  rank {} token {} {:?} logit {}",
                score.rank, score.token_id, score.display, score.logit
            );
        }
    }
}

fn print_token(view: &TokenView) {
    println!(
        "token {}{} | captured top-k {} (top-k censored)",
        view.token_id,
        view.display
            .as_ref()
            .map(|display| format!(" {display:?}"))
            .unwrap_or_default(),
        view.captured_top_k
    );
    println!(
        "layers {}",
        view.selected_layers
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "global count={} top1={} best_rank={}{}",
        view.global.count,
        view.global.top1_count,
        view.global
            .best_rank
            .map(|rank| rank.to_string())
            .unwrap_or_else(|| "-".into()),
        if view.global_outside_captured_top_k {
            " | outside captured top-k"
        } else {
            ""
        }
    );
    if let Some(position) = &view.position {
        println!(
            "position {} ({}) | source token {} {:?}",
            position.position, position.selector, position.source_token_id, position.source_display
        );
    }
    for layer in &view.layers {
        if let Some(score) = &layer.at_position {
            println!(
                "layer {} rank={} logit={} display={:?}",
                layer.source_layer, score.rank, score.logit, score.display
            );
        } else if view.position.is_some() {
            println!("layer {} outside captured top-k", layer.source_layer);
        } else {
            println!(
                "layer {} count={} top1={} best_rank={}{}",
                layer.source_layer,
                layer.count.unwrap(),
                layer.top1_count.unwrap(),
                layer
                    .best_rank
                    .map(|rank| rank.to_string())
                    .unwrap_or_else(|| "-".into()),
                if layer.outside_captured_top_k {
                    " | outside captured top-k"
                } else {
                    ""
                }
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(version: u32, rendering: bool) -> TraceDocument {
        let rendering = rendering.then(|| Rendering {
            renderer: "qwen3.6_messages_v1".into(),
            generation_mode: Some("thinking".into()),
            spans: vec![
                RenderedSpan {
                    kind: "message_start_marker".into(),
                    message_index: Some(0),
                    tool_call_index: None,
                    role: Some("user".into()),
                    channel: None,
                    label: None,
                    byte_start: 0,
                    byte_end: 1,
                    token_start: Some(0),
                    token_end: Some(1),
                },
                RenderedSpan {
                    kind: "thinking_channel_start_marker".into(),
                    message_index: None,
                    tool_call_index: None,
                    role: Some("assistant".into()),
                    channel: Some("thinking".into()),
                    label: None,
                    byte_start: 1,
                    byte_end: 2,
                    token_start: Some(1),
                    token_end: Some(2),
                },
                RenderedSpan {
                    kind: "generated_assistant_start_marker".into(),
                    message_index: None,
                    tool_call_index: None,
                    role: Some("assistant".into()),
                    channel: None,
                    label: None,
                    byte_start: 2,
                    byte_end: 3,
                    token_start: Some(2),
                    token_end: Some(3),
                },
            ],
        });
        let scores = |position, layer| match (position, layer) {
            (0, 2) => vec![score(7, 0, "seven", 3.0), score(9, 1, "nine", 2.0)],
            (1, 2) => vec![score(7, 0, "seven", 4.0), score(8, 1, "eight", 1.0)],
            (2, 2) => vec![score(8, 0, "eight", 5.0), score(7, 1, "seven", 2.0)],
            (0, 5) => vec![score(8, 0, "eight", 3.0), score(9, 1, "nine", 1.0)],
            (1, 5) => vec![score(8, 0, "eight", 4.0), score(9, 1, "nine", 2.0)],
            _ => vec![score(8, 0, "eight", 5.0), score(9, 1, "nine", 2.0)],
        };
        let mut cells = Vec::new();
        for layer in [2, 5] {
            for position in 0..3 {
                cells.push(Cell {
                    source_layer: layer,
                    source_position: position,
                    source_token_id: 10 + position as i32,
                    predicts_position: position + 1,
                    top_k: scores(position, layer),
                });
            }
        }
        let (global, per_layer) = compute_occurrences(&cells, &[2, 5]);
        TraceDocument {
            schema: "qwen.lens.trace".into(),
            schema_version: version,
            producer: (version == 3).then(|| Producer {
                build_commit: Some("test".into()),
                build_source_state: Some("clean".into()),
            }),
            deployed_model: (version == 3).then(|| DeployedModel {
                name: Some("test".into()),
                base_model_name: None,
                architecture: Some("qwen35moe".into()),
                locator_id: Some("test".into()),
                vocab_size: Some(32),
            }),
            tokenizer: (version == 3).then(|| TokenizerSummary {
                metadata_id: Some("test".into()),
                model: Some("gpt2".into()),
                pretokenizer: Some("qwen35".into()),
            }),
            lens: LensSummary {
                kind: "test".into(),
                method: "j".into(),
                target_layer: Some(6),
                source_repository: None,
                source_revision: None,
                source_filename: None,
                payload_blake3: None,
            },
            score_semantics: (version == 3).then(|| ScoreSemantics {
                kind: "logit".into(),
                normalization: Some("rmsnorm".into()),
                candidate_universe: Some("full_model_vocabulary".into()),
                softmax_applied: false,
            }),
            execution_mode: (version == 3).then(|| "test".into()),
            input_source: (version == 3).then(|| "messages".into()),
            add_special_tokens: (version == 3).then_some(false),
            input_token_ids: vec![10, 11, 12],
            input_tokens: vec![input(0, 10, "a"), input(1, 11, "b"), input(2, 12, "c")],
            rendering,
            selected_layers: vec![2, 5],
            top_k: 2,
            cells,
            vectors: None,
            timing: BTreeMap::new(),
            occurrences: Occurrences {
                global,
                per_layer: [2, 5]
                    .into_iter()
                    .map(|source_layer| LayerOccurrences {
                        source_layer,
                        tokens: per_layer[&source_layer].clone(),
                    })
                    .collect(),
            },
            batch: None,
        }
    }

    fn score(token_id: u32, rank: usize, display: &str, logit: f32) -> TokenScore {
        TokenScore {
            rank,
            token_id,
            token_display_lossy: display.into(),
            logit,
        }
    }
    fn input(position: usize, token_id: i32, display: &str) -> InputToken {
        InputToken {
            position,
            token_id,
            token_display_lossy: display.into(),
        }
    }

    #[test]
    fn v2_aggregate_preserves_artifact_order_and_layer_timeline() {
        let document = fixture(2, false);
        validate_trace(&document).unwrap();
        let view = aggregate_view(&document, 3, None, &[]).unwrap();
        assert_eq!(
            view.rows.iter().map(|row| row.token_id).collect::<Vec<_>>(),
            vec![8, 9, 7]
        );
        assert_eq!(
            view.rows[2]
                .layer_counts
                .iter()
                .map(|row| row.count)
                .collect::<Vec<_>>(),
            vec![3, 0]
        );
        assert_eq!(view.rows[2].intensity_stripe, "@ ");
    }

    #[test]
    fn batch_attribution_scopes_every_shared_trace_timing() {
        let mut document = fixture(3, true);
        let shared = [
            "matrix_read_wall_ms",
            "readout_gpu_ms",
            "readout_command_wall_ms",
            "trace_execution_wall_ms",
        ];
        for field in shared {
            document.timing.insert(field.into(), 1.0);
        }
        document.batch = Some(TraceBatchAttribution {
            batch_schema: "qwen.lens.trace_batch".into(),
            request_id: "fixture".into(),
            request_index: 0,
            request_count: 2,
            aggregate_rows: 6,
            shared_timing_fields: shared.into_iter().map(str::to_owned).collect(),
        });
        validate_trace(&document).unwrap();
        let view = summary_view(&document).batch.unwrap();
        assert_eq!(view.request_id, "fixture");
        assert_eq!(view.shared_timing_fields.len(), 4);

        document.timing.remove("readout_gpu_ms");
        assert!(validate_trace(&document).is_err());
    }

    #[test]
    fn sparse_layer_ranges_preserve_capture_order_and_reject_ambiguity() {
        let mut document = fixture(3, true);
        document.selected_layers = vec![2, 5, 18, 31, 62];
        assert_eq!(
            select_layers(&document, Some("18..31,62")).unwrap(),
            [18, 31, 62]
        );
        assert!(select_layers(&document, Some("4")).is_err());
        assert!(select_layers(&document, Some("2..5,5")).is_err());
        assert!(select_layers(&document, Some("40..50")).is_err());
    }

    #[test]
    fn filtered_aggregate_recomputes_order_and_timeline() {
        let document = fixture(3, true);
        let view = aggregate_view(&document, 3, None, &["0".into()]).unwrap();
        assert_eq!(view.selected_positions, [0]);
        assert_eq!(
            view.rows.iter().map(|row| row.token_id).collect::<Vec<_>>(),
            [9, 7, 8]
        );
        assert_eq!(
            view.rows[0]
                .layer_counts
                .iter()
                .map(|row| row.count)
                .collect::<Vec<_>>(),
            [1, 1]
        );
    }

    #[test]
    fn v3_position_resolves_numeric_and_semantic_markers() {
        let document = fixture(3, true);
        validate_trace(&document).unwrap();
        assert_eq!(resolve_position(&document, "1").unwrap(), 1);
        assert_eq!(resolve_position(&document, "role:user:start").unwrap(), 0);
        assert_eq!(resolve_position(&document, "message:0:start").unwrap(), 0);
        assert_eq!(
            resolve_position(&document, "generated:assistant:start").unwrap(),
            2
        );
        assert_eq!(
            resolve_position(&document, "channel:thinking:start").unwrap(),
            1
        );
    }

    #[test]
    fn positions_expose_exact_anchors_and_open_generated_channel() {
        let mut document = fixture(3, true);
        document.rendering.as_mut().unwrap().spans.insert(
            1,
            RenderedSpan {
                kind: "thinking_channel_end_marker".into(),
                message_index: Some(0),
                tool_call_index: None,
                role: Some("assistant".into()),
                channel: Some("thinking".into()),
                label: None,
                byte_start: 1,
                byte_end: 2,
                token_start: Some(0),
                token_end: Some(1),
            },
        );
        assert!(
            resolve_position(&document, "channel:thinking:end")
                .unwrap_err()
                .to_string()
                .contains("open generated channel")
        );
        let view = positions_view(&document).unwrap();
        assert!(
            view.anchors
                .iter()
                .any(|anchor| anchor.selector == "prefill:last")
        );
        assert!(
            view.anchors
                .iter()
                .any(|anchor| anchor.selector == "message:0:start"
                    && anchor.message_index == Some(0))
        );
        assert!(
            view.anchors
                .iter()
                .any(|anchor| anchor.selector == "generated:assistant:start")
        );
        assert!(
            !view
                .anchors
                .iter()
                .any(|anchor| anchor.selector == "channel:thinking:end")
        );
    }

    #[test]
    fn repeated_role_aliases_report_last_message_boundary() {
        let mut document = fixture(3, true);
        let rendering = document.rendering.as_mut().unwrap();
        rendering.spans = vec![
            RenderedSpan {
                kind: "message_start_marker".into(),
                message_index: Some(0),
                tool_call_index: None,
                role: Some("user".into()),
                channel: None,
                label: None,
                byte_start: 0,
                byte_end: 1,
                token_start: Some(0),
                token_end: Some(1),
            },
            RenderedSpan {
                kind: "message_end_marker".into(),
                message_index: Some(0),
                tool_call_index: None,
                role: Some("user".into()),
                channel: None,
                label: None,
                byte_start: 1,
                byte_end: 2,
                token_start: Some(0),
                token_end: Some(1),
            },
            RenderedSpan {
                kind: "message_start_marker".into(),
                message_index: Some(2),
                tool_call_index: None,
                role: Some("user".into()),
                channel: None,
                label: None,
                byte_start: 2,
                byte_end: 3,
                token_start: Some(1),
                token_end: Some(2),
            },
            RenderedSpan {
                kind: "message_end_marker".into(),
                message_index: Some(2),
                tool_call_index: None,
                role: Some("user".into()),
                channel: None,
                label: None,
                byte_start: 3,
                byte_end: 4,
                token_start: Some(2),
                token_end: Some(3),
            },
        ];
        assert_eq!(resolve_position(&document, "role:user:start").unwrap(), 1);
        assert_eq!(resolve_position(&document, "role:user:end").unwrap(), 2);
        let view = positions_view(&document).unwrap();
        assert!(view.anchors.iter().any(|anchor| {
            anchor.selector == "role:user:start"
                && anchor.position == 1
                && anchor.message_index == Some(2)
        }));
    }

    #[test]
    fn token_position_marks_top_k_censored_layers() {
        let document = fixture(3, true);
        let view = token_view(&document, 7, Some("2"), None).unwrap();
        assert_eq!(view.layers[0].at_position.as_ref().unwrap().rank, 1);
        assert!(view.layers[1].outside_captured_top_k);
        assert!(view.layers[1].at_position.is_none());
    }

    #[test]
    fn v3_allows_null_nonstructural_ranges_but_requires_exact_structural_ranges() {
        let mut document = fixture(3, true);
        document.rendering.as_mut().unwrap().spans.insert(
            1,
            RenderedSpan {
                kind: "message_content".into(),
                message_index: Some(0),
                tool_call_index: None,
                role: Some("user".into()),
                channel: None,
                label: None,
                byte_start: 1,
                byte_end: 2,
                token_start: None,
                token_end: None,
            },
        );
        document.rendering.as_mut().unwrap().spans[2].byte_start = 2;
        document.rendering.as_mut().unwrap().spans[2].byte_end = 3;
        document.rendering.as_mut().unwrap().spans[3].byte_start = 3;
        document.rendering.as_mut().unwrap().spans[3].byte_end = 4;
        validate_trace(&document).unwrap();

        document.rendering.as_mut().unwrap().spans[0].token_start = None;
        document.rendering.as_mut().unwrap().spans[0].token_end = None;
        assert!(
            validate_trace(&document)
                .unwrap_err()
                .to_string()
                .contains("structural rendering marker")
        );
    }

    #[test]
    fn token_view_rejects_ids_outside_declared_vocabulary() {
        let document = fixture(3, true);
        assert!(
            token_view(&document, 32, None, None)
                .unwrap_err()
                .to_string()
                .contains("outside model vocabulary 32")
        );
    }

    #[test]
    fn v3_muse_message_records_resolve_role_and_channel_boundaries() {
        let mut document = fixture(3, true);
        document.deployed_model.as_mut().unwrap().architecture = Some("muse-glimmer".into());
        let rendering = document.rendering.as_mut().unwrap();
        rendering.renderer = "muse_glimmer_atem_annotated_v1".into();
        rendering.generation_mode = Some("reasoning_high".into());
        for position in 3..8 {
            let token_id = 10 + position as i32;
            document.input_token_ids.push(token_id);
            document.input_tokens.push(input(
                position,
                token_id,
                &char::from(b'a' + position as u8).to_string(),
            ));
            for layer in [2, 5] {
                document.cells.push(Cell {
                    source_layer: layer,
                    source_position: position,
                    source_token_id: token_id,
                    predicts_position: position + 1,
                    top_k: vec![score(8, 0, "eight", 1.0), score(9, 1, "nine", 0.5)],
                });
            }
        }
        let (global, per_layer) = compute_occurrences(&document.cells, &document.selected_layers);
        document.occurrences = Occurrences {
            global,
            per_layer: document
                .selected_layers
                .iter()
                .map(|source_layer| LayerOccurrences {
                    source_layer: *source_layer,
                    tokens: per_layer[source_layer].clone(),
                })
                .collect(),
        };
        document.rendering.as_mut().unwrap().spans = vec![
            RenderedSpan {
                kind: "bos_marker".into(),
                message_index: None,
                tool_call_index: None,
                role: None,
                channel: None,
                label: None,
                byte_start: 0,
                byte_end: 1,
                token_start: Some(0),
                token_end: Some(1),
            },
            RenderedSpan {
                kind: "message_start_marker".into(),
                message_index: Some(0),
                tool_call_index: None,
                role: Some("assistant".into()),
                channel: Some("thinking".into()),
                label: None,
                byte_start: 1,
                byte_end: 2,
                token_start: Some(1),
                token_end: Some(2),
            },
            RenderedSpan {
                kind: "role".into(),
                message_index: Some(0),
                tool_call_index: None,
                role: Some("assistant".into()),
                channel: Some("thinking".into()),
                label: None,
                byte_start: 2,
                byte_end: 3,
                token_start: None,
                token_end: None,
            },
            RenderedSpan {
                kind: "recipient".into(),
                message_index: Some(0),
                tool_call_index: None,
                role: Some("assistant".into()),
                channel: Some("thinking".into()),
                label: Some("self".into()),
                byte_start: 3,
                byte_end: 4,
                token_start: None,
                token_end: None,
            },
            RenderedSpan {
                kind: "message_marker".into(),
                message_index: Some(0),
                tool_call_index: None,
                role: Some("assistant".into()),
                channel: Some("thinking".into()),
                label: None,
                byte_start: 4,
                byte_end: 5,
                token_start: Some(2),
                token_end: Some(3),
            },
            RenderedSpan {
                kind: "assistant_reasoning_content".into(),
                message_index: Some(0),
                tool_call_index: None,
                role: Some("assistant".into()),
                channel: Some("thinking".into()),
                label: None,
                byte_start: 5,
                byte_end: 6,
                token_start: None,
                token_end: None,
            },
            RenderedSpan {
                kind: "message_end_marker".into(),
                message_index: Some(0),
                tool_call_index: None,
                role: Some("assistant".into()),
                channel: Some("thinking".into()),
                label: Some("<|eom|>".into()),
                byte_start: 6,
                byte_end: 7,
                token_start: Some(3),
                token_end: Some(4),
            },
            RenderedSpan {
                kind: "message_start_marker".into(),
                message_index: Some(0),
                tool_call_index: Some(0),
                role: Some("assistant".into()),
                channel: Some("tool_call".into()),
                label: None,
                byte_start: 7,
                byte_end: 8,
                token_start: Some(4),
                token_end: Some(5),
            },
            RenderedSpan {
                kind: "role".into(),
                message_index: Some(0),
                tool_call_index: Some(0),
                role: Some("assistant".into()),
                channel: Some("tool_call".into()),
                label: None,
                byte_start: 8,
                byte_end: 9,
                token_start: None,
                token_end: None,
            },
            RenderedSpan {
                kind: "recipient".into(),
                message_index: Some(0),
                tool_call_index: Some(0),
                role: Some("assistant".into()),
                channel: Some("tool_call".into()),
                label: Some("first".into()),
                byte_start: 9,
                byte_end: 10,
                token_start: None,
                token_end: None,
            },
            RenderedSpan {
                kind: "message_marker".into(),
                message_index: Some(0),
                tool_call_index: Some(0),
                role: Some("assistant".into()),
                channel: Some("tool_call".into()),
                label: None,
                byte_start: 10,
                byte_end: 11,
                token_start: Some(5),
                token_end: Some(6),
            },
            RenderedSpan {
                kind: "tool_call_content".into(),
                message_index: Some(0),
                tool_call_index: Some(0),
                role: Some("assistant".into()),
                channel: Some("tool_call".into()),
                label: None,
                byte_start: 11,
                byte_end: 12,
                token_start: None,
                token_end: None,
            },
            RenderedSpan {
                kind: "message_end_marker".into(),
                message_index: Some(0),
                tool_call_index: Some(0),
                role: Some("assistant".into()),
                channel: Some("tool_call".into()),
                label: Some("<|eot|>".into()),
                byte_start: 12,
                byte_end: 13,
                token_start: Some(6),
                token_end: Some(7),
            },
            RenderedSpan {
                kind: "generated_assistant_start_marker".into(),
                message_index: None,
                tool_call_index: None,
                role: Some("assistant".into()),
                channel: None,
                label: None,
                byte_start: 13,
                byte_end: 14,
                token_start: Some(7),
                token_end: Some(8),
            },
            RenderedSpan {
                kind: "generated_assistant_role".into(),
                message_index: None,
                tool_call_index: None,
                role: Some("assistant".into()),
                channel: None,
                label: None,
                byte_start: 14,
                byte_end: 15,
                token_start: None,
                token_end: None,
            },
        ];
        validate_trace(&document).unwrap();
        assert_eq!(
            resolve_position(&document, "role:assistant:start").unwrap(),
            1
        );
        assert_eq!(
            resolve_position(&document, "role:assistant:end").unwrap(),
            6
        );
        assert_eq!(resolve_position(&document, "message:0:start").unwrap(), 1);
        assert_eq!(resolve_position(&document, "message:0:end").unwrap(), 6);
        assert_eq!(
            resolve_position(&document, "channel:tool_call:start").unwrap(),
            4
        );
        assert_eq!(
            resolve_position(&document, "channel:tool_call:end").unwrap(),
            6
        );
        assert_eq!(
            resolve_position(&document, "channel:thinking:start").unwrap(),
            1
        );
        assert_eq!(
            resolve_position(&document, "channel:thinking:end").unwrap(),
            3
        );
        let view = positions_view(&document).unwrap();
        let message_end = view
            .anchors
            .iter()
            .filter(|anchor| anchor.selector == "message:0:end")
            .collect::<Vec<_>>();
        assert_eq!(message_end.len(), 1);
        assert_eq!(message_end[0].position, 6);
        assert_eq!(message_end[0].tool_call_index, Some(0));
        assert_eq!(message_end[0].label.as_deref(), Some("<|eot|>"));
        assert!(view.rendering_spans.iter().any(|span| {
            span.tool_call_index == Some(0)
                && span.channel.as_deref() == Some("tool_call")
                && span.label.as_deref() == Some("<|eot|>")
        }));

        let rendering = document.rendering.as_mut().unwrap();
        rendering.renderer = "qwen3.8_messages_v1".into();
        rendering.generation_mode = Some("thinking_xhigh".into());
        document.deployed_model.as_mut().unwrap().architecture = Some("qwen35".into());
        document.rendering.as_mut().unwrap().spans = vec![RenderedSpan {
            kind: "reasoning_instruction_content".into(),
            message_index: None,
            tool_call_index: None,
            role: Some("system".into()),
            channel: Some("thinking".into()),
            label: None,
            byte_start: 0,
            byte_end: 2,
            token_start: Some(0),
            token_end: Some(2),
        }];
        validate_trace(&document).unwrap();
        assert_eq!(
            resolve_position(&document, "channel:thinking:start").unwrap(),
            0
        );
        assert_eq!(
            resolve_position(&document, "channel:thinking:end").unwrap(),
            1
        );
    }

    #[test]
    fn v3_rejects_missing_required_metadata_while_v2_remains_legacy_readable() {
        let mut v3 = fixture(3, true);
        v3.producer = None;
        assert!(
            validate_trace(&v3)
                .unwrap_err()
                .to_string()
                .contains("missing producer")
        );

        validate_trace(&fixture(2, false)).unwrap();
    }

    #[test]
    fn v3_rejects_cross_family_rendering_metadata() {
        let mut document = fixture(3, true);
        validate_trace(&document).unwrap();
        document.deployed_model.as_mut().unwrap().architecture = Some("muse-glimmer".into());
        assert!(
            validate_trace(&document)
                .unwrap_err()
                .to_string()
                .contains("input rendering metadata")
        );
    }
}
