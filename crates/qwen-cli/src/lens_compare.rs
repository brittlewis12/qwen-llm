use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use super::lens_inspect::{self, Cell, TraceDocument, VectorCell};
use super::read_regular_file_bounded;

const COMPARE_MAX_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_LIMIT: usize = 25;

#[derive(Debug, Args)]
pub(crate) struct CompareArgs {
    /// Left regular non-symlink trace or run JSON artifact.
    left: PathBuf,
    /// Right regular non-symlink trace or run JSON artifact.
    right: PathBuf,
    /// Render concise text or a typed comparison result.
    #[arg(long, value_enum, default_value_t = CompareFormat::Text)]
    format: CompareFormat,
    /// Maximum detail rows or cells included in the result.
    #[arg(long, default_value_t = DEFAULT_LIMIT)]
    limit: usize,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CompareFormat {
    Text,
    Json,
}

#[derive(Deserialize)]
struct Envelope {
    schema: String,
    schema_version: u32,
}

#[derive(Debug, Serialize)]
#[serde(tag = "comparison_kind", rename_all = "snake_case")]
enum ComparisonResult {
    Trace(TraceComparison),
    Run(RunComparison),
}

pub(crate) fn run(args: CompareArgs) -> Result<()> {
    ensure!(args.limit > 0, "--limit must be positive");
    let left_bytes = read_regular_file_bounded(&args.left, COMPARE_MAX_BYTES)?;
    let right_bytes = read_regular_file_bounded(&args.right, COMPARE_MAX_BYTES)?;
    let left_envelope: Envelope = serde_json::from_slice(&left_bytes)
        .with_context(|| format!("parse artifact envelope {}", args.left.display()))?;
    let right_envelope: Envelope = serde_json::from_slice(&right_bytes)
        .with_context(|| format!("parse artifact envelope {}", args.right.display()))?;
    ensure!(
        left_envelope.schema == right_envelope.schema,
        "mixed schemas are not comparable: {:?} versus {:?}",
        left_envelope.schema,
        right_envelope.schema
    );
    let result = match left_envelope.schema.as_str() {
        "qwen.lens.trace" => {
            ensure!(
                matches!(left_envelope.schema_version, 2 | 3)
                    && matches!(right_envelope.schema_version, 2 | 3),
                "trace comparison supports schema versions 2 and 3 only"
            );
            let left = lens_inspect::parse_trace_bytes(&left_bytes, &args.left)?;
            let right = lens_inspect::parse_trace_bytes(&right_bytes, &args.right)?;
            ComparisonResult::Trace(compare_traces(&left, &right, args.limit)?)
        }
        "qwen.lens.run" => {
            ensure!(
                left_envelope.schema_version == 1 && right_envelope.schema_version == 1,
                "run comparison supports schema version 1 only"
            );
            let left: RunDocument = serde_json::from_slice(&left_bytes)
                .with_context(|| format!("parse run JSON {}", args.left.display()))?;
            let right: RunDocument = serde_json::from_slice(&right_bytes)
                .with_context(|| format!("parse run JSON {}", args.right.display()))?;
            left.validate()?;
            right.validate()?;
            ComparisonResult::Run(compare_runs(&left, &right, args.limit)?)
        }
        schema => bail!("unsupported comparison schema {schema:?}"),
    };
    match args.format {
        CompareFormat::Json => println!("{}", serde_json::to_string_pretty(&result)?),
        CompareFormat::Text => print_text(&result),
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct TraceComparison {
    alignment: &'static str,
    schema_version: u32,
    left_lens: LensIdentity,
    right_lens: LensIdentity,
    score_semantics: TraceScoreIdentity,
    cell_count: usize,
    top1_changed_cell_count: usize,
    first_changed: Option<CellCoordinate>,
    changed_cell_count: usize,
    changed_cells: Vec<TraceCellDifference>,
    aggregate_difference_count: usize,
    aggregate_differences: Vec<AggregateDifference>,
    vectors: VectorComparison,
    detail_limit: usize,
}

#[derive(Debug, Serialize)]
struct LensIdentity {
    method: String,
    artifact_kind: String,
    source_repository: Option<String>,
    source_revision: Option<String>,
    source_filename: Option<String>,
    payload_blake3: Option<String>,
}

#[derive(Debug, Serialize, PartialEq)]
struct TraceScoreIdentity {
    kind: Option<String>,
    normalization: Option<String>,
    candidate_universe: Option<String>,
    softmax_applied: Option<bool>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct CellCoordinate {
    source_layer: u32,
    source_position: usize,
}

#[derive(Debug, Serialize)]
struct TraceCellDifference {
    coordinate: CellCoordinate,
    top1_changed: bool,
    candidate_difference_count: usize,
    candidates: Vec<TopKCandidateDifference>,
}

#[derive(Debug, Serialize)]
struct TopKCandidateDifference {
    token_id: u32,
    display: String,
    status: CandidateStatus,
    left_rank: Option<usize>,
    right_rank: Option<usize>,
    left_logit: Option<f32>,
    right_logit: Option<f32>,
    logit_delta: Option<f32>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum CandidateStatus {
    PresentBoth,
    EnteredCapturedTopK,
    ExitedCapturedTopK,
}

impl CandidateStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::PresentBoth => "present_both",
            Self::EnteredCapturedTopK => "entered_captured_top_k",
            Self::ExitedCapturedTopK => "exited_captured_top_k",
        }
    }
}

#[derive(Debug, Serialize)]
struct AggregateDifference {
    token_id: u32,
    display: String,
    left_count: usize,
    right_count: usize,
    count_delta: i64,
    left_top1_count: usize,
    right_top1_count: usize,
    top1_count_delta: i64,
    left_best_rank: Option<usize>,
    right_best_rank: Option<usize>,
    best_rank_delta: Option<i64>,
}

#[derive(Debug, Default, Serialize)]
struct VectorComparison {
    metadata_compatible: bool,
    metadata_mismatch: Option<String>,
    metadata_incompatible_cell_count: usize,
    metadata_incompatible_cells: Vec<CellCoordinate>,
    matched_cell_count: usize,
    matched_cells: Vec<VectorMetrics>,
    unmatched_cell_count: usize,
    unmatched_cells: Vec<UnmatchedVectorCell>,
    incompatible_dimension_count: usize,
    incompatible_dimensions: Vec<IncompatibleVectorCell>,
}

#[derive(Debug, Serialize)]
struct VectorMetrics {
    coordinate: CellCoordinate,
    dimension: usize,
    cosine: Option<f64>,
    l2_distance: f64,
    left_norm: f64,
    right_norm: f64,
}

#[derive(Debug, Serialize)]
struct UnmatchedVectorCell {
    coordinate: CellCoordinate,
    side: &'static str,
    dimension: usize,
}

#[derive(Debug, Serialize)]
struct IncompatibleVectorCell {
    coordinate: CellCoordinate,
    left_dimension: usize,
    right_dimension: usize,
}

fn compare_traces(
    left: &TraceDocument,
    right: &TraceDocument,
    limit: usize,
) -> Result<TraceComparison> {
    ensure!(
        left.schema_version == right.schema_version,
        "trace schema versions differ"
    );
    ensure!(
        left.input_token_ids == right.input_token_ids,
        "trace input token IDs differ"
    );
    ensure!(
        left.selected_layers == right.selected_layers,
        "trace selected layers/order differ"
    );
    ensure!(left.top_k == right.top_k, "trace captured top-k differs");
    let left_coordinates = trace_coordinates(left);
    let right_coordinates = trace_coordinates(right);
    ensure!(
        left_coordinates == right_coordinates,
        "trace cell coordinates/order differ"
    );
    let left_score = trace_score_identity(left);
    let right_score = trace_score_identity(right);
    ensure!(left_score == right_score, "trace score semantics differ");
    if left.schema_version == 3 {
        ensure!(
            left.deployed_model
                .as_ref()
                .and_then(|model| model.locator_id.as_ref())
                == right
                    .deployed_model
                    .as_ref()
                    .and_then(|model| model.locator_id.as_ref()),
            "v3 deployed model locator IDs differ"
        );
        ensure!(
            left.deployed_model
                .as_ref()
                .and_then(|model| model.locator_id.as_ref())
                .is_some(),
            "v3 comparison requires a deployed model locator ID"
        );
        ensure!(
            left.tokenizer
                .as_ref()
                .and_then(|tokenizer| tokenizer.metadata_id.as_ref())
                == right
                    .tokenizer
                    .as_ref()
                    .and_then(|tokenizer| tokenizer.metadata_id.as_ref()),
            "v3 tokenizer metadata IDs differ"
        );
        ensure!(
            left.tokenizer
                .as_ref()
                .and_then(|tokenizer| tokenizer.metadata_id.as_ref())
                .is_some(),
            "v3 comparison requires a tokenizer metadata ID"
        );
    }

    let mut changed_cells = Vec::new();
    let mut top1_changed_cell_count = 0;
    let mut first_changed = None;
    for (left_cell, right_cell) in left.cells.iter().zip(&right.cells) {
        let top1_changed = left_cell.top_k.first().map(|score| score.token_id)
            != right_cell.top_k.first().map(|score| score.token_id);
        top1_changed_cell_count += usize::from(top1_changed);
        let candidates = compare_top_k(left_cell, right_cell);
        if !candidates.is_empty() {
            let coordinate = CellCoordinate {
                source_layer: left_cell.source_layer,
                source_position: left_cell.source_position,
            };
            first_changed.get_or_insert(coordinate);
            changed_cells.push(TraceCellDifference {
                coordinate,
                top1_changed,
                candidate_difference_count: candidates.len(),
                candidates: candidates.into_iter().take(limit).collect(),
            });
        }
    }
    let aggregate_differences = compare_aggregates(left, right);
    let vectors = compare_vectors(left, right, limit);
    Ok(TraceComparison {
        alignment: "exact layer/position and exact token ID; no inferred alignment",
        schema_version: left.schema_version,
        left_lens: lens_identity(left),
        right_lens: lens_identity(right),
        score_semantics: left_score,
        cell_count: left.cells.len(),
        top1_changed_cell_count,
        first_changed,
        changed_cell_count: changed_cells.len(),
        changed_cells: changed_cells.into_iter().take(limit).collect(),
        aggregate_difference_count: aggregate_differences.len(),
        aggregate_differences: aggregate_differences.into_iter().take(limit).collect(),
        vectors,
        detail_limit: limit,
    })
}

fn trace_coordinates(document: &TraceDocument) -> Vec<CellCoordinate> {
    document
        .cells
        .iter()
        .map(|cell| CellCoordinate {
            source_layer: cell.source_layer,
            source_position: cell.source_position,
        })
        .collect()
}

fn trace_score_identity(document: &TraceDocument) -> TraceScoreIdentity {
    document.score_semantics.as_ref().map_or(
        TraceScoreIdentity {
            kind: None,
            normalization: None,
            candidate_universe: None,
            softmax_applied: None,
        },
        |score| TraceScoreIdentity {
            kind: Some(score.kind.clone()),
            normalization: score.normalization.clone(),
            candidate_universe: score.candidate_universe.clone(),
            softmax_applied: Some(score.softmax_applied),
        },
    )
}

fn lens_identity(document: &TraceDocument) -> LensIdentity {
    LensIdentity {
        method: document.lens.method.clone(),
        artifact_kind: document.lens.kind.clone(),
        source_repository: document.lens.source_repository.clone(),
        source_revision: document.lens.source_revision.clone(),
        source_filename: document.lens.source_filename.clone(),
        payload_blake3: document.lens.payload_blake3.clone(),
    }
}

fn compare_top_k(left: &Cell, right: &Cell) -> Vec<TopKCandidateDifference> {
    let left_map: BTreeMap<_, _> = left
        .top_k
        .iter()
        .map(|score| (score.token_id, score))
        .collect();
    let right_map: BTreeMap<_, _> = right
        .top_k
        .iter()
        .map(|score| (score.token_id, score))
        .collect();
    left_map
        .keys()
        .chain(right_map.keys())
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|token_id| {
            let left_score = left_map.get(&token_id).copied();
            let right_score = right_map.get(&token_id).copied();
            if let (Some(a), Some(b)) = (left_score, right_score)
                && a.rank == b.rank
                && a.logit == b.logit
            {
                return None;
            }
            let status = match (left_score, right_score) {
                (Some(_), Some(_)) => CandidateStatus::PresentBoth,
                (None, Some(_)) => CandidateStatus::EnteredCapturedTopK,
                (Some(_), None) => CandidateStatus::ExitedCapturedTopK,
                (None, None) => unreachable!(),
            };
            Some(TopKCandidateDifference {
                token_id,
                display: right_score
                    .or(left_score)
                    .unwrap()
                    .token_display_lossy
                    .clone(),
                status,
                left_rank: left_score.map(|score| score.rank),
                right_rank: right_score.map(|score| score.rank),
                left_logit: left_score.map(|score| score.logit),
                right_logit: right_score.map(|score| score.logit),
                logit_delta: left_score.zip(right_score).map(|(a, b)| b.logit - a.logit),
            })
        })
        .collect()
}

#[derive(Clone, Copy, Default)]
struct AggregateValue {
    count: usize,
    top1: usize,
    best_rank: Option<usize>,
}

fn aggregate(document: &TraceDocument) -> BTreeMap<u32, AggregateValue> {
    let mut output = BTreeMap::new();
    for cell in &document.cells {
        for score in &cell.top_k {
            let row = output
                .entry(score.token_id)
                .or_insert(AggregateValue::default());
            row.count += 1;
            row.top1 += usize::from(score.rank == 0);
            row.best_rank = Some(
                row.best_rank
                    .map_or(score.rank, |rank| rank.min(score.rank)),
            );
        }
    }
    output
}

fn compare_aggregates(left: &TraceDocument, right: &TraceDocument) -> Vec<AggregateDifference> {
    let displays = trace_displays(left, right);
    let left_values = aggregate(left);
    let right_values = aggregate(right);
    let mut rows: Vec<_> = left_values
        .keys()
        .chain(right_values.keys())
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|token_id| {
            let a = left_values.get(&token_id).copied().unwrap_or_default();
            let b = right_values.get(&token_id).copied().unwrap_or_default();
            if a.count == b.count && a.top1 == b.top1 && a.best_rank == b.best_rank {
                return None;
            }
            Some(AggregateDifference {
                token_id,
                display: displays.get(&token_id).cloned().unwrap_or_default(),
                left_count: a.count,
                right_count: b.count,
                count_delta: b.count as i64 - a.count as i64,
                left_top1_count: a.top1,
                right_top1_count: b.top1,
                top1_count_delta: b.top1 as i64 - a.top1 as i64,
                left_best_rank: a.best_rank,
                right_best_rank: b.best_rank,
                best_rank_delta: a
                    .best_rank
                    .zip(b.best_rank)
                    .map(|(x, y)| y as i64 - x as i64),
            })
        })
        .collect();
    rows.sort_by_key(|row| (Reverse(row.count_delta.unsigned_abs()), row.token_id));
    rows
}

fn trace_displays(left: &TraceDocument, right: &TraceDocument) -> BTreeMap<u32, String> {
    left.cells
        .iter()
        .chain(&right.cells)
        .flat_map(|cell| &cell.top_k)
        .fold(BTreeMap::new(), |mut displays, score| {
            displays
                .entry(score.token_id)
                .or_insert_with(|| score.token_display_lossy.clone());
            displays
        })
}

fn vector_map(document: &TraceDocument) -> BTreeMap<CellCoordinate, &VectorCell> {
    document
        .vectors
        .as_ref()
        .into_iter()
        .flat_map(|vectors| &vectors.cells)
        .map(|cell| {
            (
                CellCoordinate {
                    source_layer: cell.source_layer,
                    source_position: cell.source_position,
                },
                cell,
            )
        })
        .collect()
}

fn compare_vectors(left: &TraceDocument, right: &TraceDocument, limit: usize) -> VectorComparison {
    let metadata_mismatch = vector_metadata_mismatch(left, right);
    let left = vector_map(left);
    let right = vector_map(right);
    let mut output = VectorComparison {
        metadata_compatible: metadata_mismatch.is_none(),
        metadata_mismatch,
        ..VectorComparison::default()
    };
    if !output.metadata_compatible {
        let coordinates = left
            .keys()
            .chain(right.keys())
            .copied()
            .collect::<BTreeSet<_>>();
        output.metadata_incompatible_cell_count = coordinates.len();
        output.metadata_incompatible_cells = coordinates.into_iter().take(limit).collect();
        return output;
    }
    for coordinate in left
        .keys()
        .chain(right.keys())
        .copied()
        .collect::<BTreeSet<_>>()
    {
        match (left.get(&coordinate), right.get(&coordinate)) {
            (Some(a), Some(b)) if a.values.len() == b.values.len() => {
                let (mut dot, mut left_sq, mut right_sq, mut distance_sq) = (0.0, 0.0, 0.0, 0.0);
                for (&x, &y) in a.values.iter().zip(&b.values) {
                    let (x, y) = (f64::from(x), f64::from(y));
                    dot += x * y;
                    left_sq += x * x;
                    right_sq += y * y;
                    distance_sq += (y - x) * (y - x);
                }
                let left_norm = left_sq.sqrt();
                let right_norm = right_sq.sqrt();
                output.matched_cell_count += 1;
                if output.matched_cells.len() < limit {
                    output.matched_cells.push(VectorMetrics {
                        coordinate,
                        dimension: a.values.len(),
                        cosine: (left_norm > 0.0 && right_norm > 0.0)
                            .then_some(dot / (left_norm * right_norm)),
                        l2_distance: distance_sq.sqrt(),
                        left_norm,
                        right_norm,
                    });
                }
            }
            (Some(a), Some(b)) => {
                output.incompatible_dimension_count += 1;
                if output.incompatible_dimensions.len() < limit {
                    output.incompatible_dimensions.push(IncompatibleVectorCell {
                        coordinate,
                        left_dimension: a.values.len(),
                        right_dimension: b.values.len(),
                    });
                }
            }
            (Some(a), None) | (None, Some(a)) => {
                let side = if left.contains_key(&coordinate) {
                    "left_only"
                } else {
                    "right_only"
                };
                output.unmatched_cell_count += 1;
                if output.unmatched_cells.len() < limit {
                    output.unmatched_cells.push(UnmatchedVectorCell {
                        coordinate,
                        side,
                        dimension: a.values.len(),
                    });
                }
            }
            (None, None) => unreachable!(),
        }
    }
    output
}

fn vector_metadata_mismatch(left: &TraceDocument, right: &TraceDocument) -> Option<String> {
    let left_vectors = left.vectors.as_ref();
    let right_vectors = right.vectors.as_ref();
    match (left_vectors, right_vectors) {
        (None, None) => None,
        (Some(_), None) | (None, Some(_)) => Some("only one trace contains vector metadata".into()),
        (Some(a), Some(b)) => {
            let left_metadata = (
                left.lens.target_layer,
                a.operation.as_deref(),
                a.stage.as_deref(),
                a.value_dtype.as_deref(),
                a.hidden_coordinate.as_deref(),
                a.hidden_size,
            );
            let right_metadata = (
                right.lens.target_layer,
                b.operation.as_deref(),
                b.stage.as_deref(),
                b.value_dtype.as_deref(),
                b.hidden_coordinate.as_deref(),
                b.hidden_size,
            );
            (left_metadata != right_metadata).then(|| {
                format!(
                    "vector coordinate metadata differs: left={left_metadata:?} right={right_metadata:?}"
                )
            })
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct RunSampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    seed: u64,
}

#[derive(Debug, Deserialize)]
struct RunDocument {
    schema: String,
    schema_version: u32,
    runtime_kind: String,
    model_path: PathBuf,
    #[serde(rename = "canonical_plan_path")]
    _canonical_plan_path: PathBuf,
    plan: serde_json::Value,
    #[serde(rename = "input_source")]
    _input_source: String,
    prompt_token_ids: Vec<i32>,
    generated_token_ids: Vec<i32>,
    sampler: RunSampler,
    #[serde(rename = "max_new_tokens")]
    _max_new_tokens: usize,
    decoded_text: String,
    stop_reason: String,
    operation_applications: Vec<RunOperationApplication>,
    #[serde(rename = "requested_live_readouts")]
    _requested_live_readouts: Vec<serde_json::Value>,
    live_readouts: Vec<RunReadout>,
    #[serde(default)]
    native_hyper_captures: Vec<serde_json::Value>,
}

impl RunDocument {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "qwen.lens.run" && self.schema_version == 1,
            "unsupported run schema/version"
        );
        ensure!(
            self.sampler.temperature.is_finite()
                && self.sampler.top_p.is_finite()
                && self.sampler.min_p.is_finite(),
            "run sampler contains non-finite settings"
        );
        for readout in &self.live_readouts {
            ensure!(
                !readout.score_kind.is_empty() && !readout.candidate_universe.is_empty(),
                "run readout is missing score semantics"
            );
            let mut identities = BTreeSet::new();
            for score in &readout.scores {
                ensure!(
                    score.score.is_finite(),
                    "run readout contains a non-finite score"
                );
                ensure!(
                    identities.insert(score.identity()),
                    "run readout repeats a candidate identity"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RunOperationApplication {
    id: String,
    layer: u32,
    phase: String,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct RunReadout {
    id: String,
    lens: String,
    method: String,
    score_kind: String,
    candidate_universe: String,
    source_layer: u32,
    target_layer: Option<u32>,
    phase: String,
    index: usize,
    scores: Vec<RunScore>,
}

#[derive(Debug, Deserialize)]
struct RunScore {
    token_id: Option<i32>,
    row_id: usize,
    word_id: Option<i64>,
    label: Option<String>,
    score: f32,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct ReadoutKey {
    id: String,
    lens: String,
    method: String,
    score_kind: String,
    candidate_universe: String,
    source_layer: u32,
    target_layer: Option<u32>,
    phase: String,
    index: usize,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct ScoreIdentity {
    token_id: Option<i32>,
    row_id: usize,
    word_id: Option<i64>,
    label: Option<String>,
}

impl RunReadout {
    fn key(&self) -> ReadoutKey {
        ReadoutKey {
            id: self.id.clone(),
            lens: self.lens.clone(),
            method: self.method.clone(),
            score_kind: self.score_kind.clone(),
            candidate_universe: self.candidate_universe.clone(),
            source_layer: self.source_layer,
            target_layer: self.target_layer,
            phase: self.phase.clone(),
            index: self.index,
        }
    }
}
impl RunScore {
    fn identity(&self) -> ScoreIdentity {
        ScoreIdentity {
            token_id: self.token_id,
            row_id: self.row_id,
            word_id: self.word_id,
            label: self.label.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
struct RunComparison {
    alignment: &'static str,
    runtime_kind: String,
    model_path: PathBuf,
    plans_equal: bool,
    left_generated_text: String,
    right_generated_text: String,
    left_generated_token_ids: Vec<i32>,
    right_generated_token_ids: Vec<i32>,
    first_generated_token_divergence: Option<GeneratedDivergence>,
    left_stop_reason: String,
    right_stop_reason: String,
    operation_applications: OperationComparison,
    native_capture_counts: SideCounts,
    matched_readout_count: usize,
    matched_readouts: Vec<MatchedReadout>,
    unmatched_readout_count: usize,
    unmatched_readouts: Vec<UnmatchedReadout>,
    detail_limit: usize,
}

#[derive(Debug, Serialize)]
struct GeneratedDivergence {
    index: usize,
    left_token_id: Option<i32>,
    right_token_id: Option<i32>,
    length_only: bool,
}

#[derive(Debug, Serialize)]
struct SideCounts {
    left: usize,
    right: usize,
}

#[derive(Debug, Serialize)]
struct OperationComparison {
    left_total_count: usize,
    right_total_count: usize,
    left_applications: Vec<RunOperationApplication>,
    right_applications: Vec<RunOperationApplication>,
}

#[derive(Debug, Serialize)]
struct MatchedReadout {
    key: ReadoutKey,
    candidate_difference_count: usize,
    candidate_differences: Vec<RunCandidateDifference>,
}

#[derive(Debug, Serialize)]
struct RunCandidateDifference {
    identity: ScoreIdentity,
    status: SelectedCandidateStatus,
    left_score: Option<f32>,
    right_score: Option<f32>,
    score_delta: Option<f32>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum SelectedCandidateStatus {
    PresentBoth,
    EnteredReadoutTopK,
    ExitedReadoutTopK,
}

impl SelectedCandidateStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::PresentBoth => "present_both",
            Self::EnteredReadoutTopK => "entered_readout_top_k",
            Self::ExitedReadoutTopK => "exited_readout_top_k",
        }
    }
}

#[derive(Debug, Serialize)]
struct UnmatchedReadout {
    key: ReadoutKey,
    side: &'static str,
}

fn compare_runs(left: &RunDocument, right: &RunDocument, limit: usize) -> Result<RunComparison> {
    ensure!(
        left.prompt_token_ids == right.prompt_token_ids,
        "run prompt token IDs differ"
    );
    ensure!(
        left.runtime_kind == right.runtime_kind,
        "run runtime kinds differ"
    );
    ensure!(
        left.model_path == right.model_path,
        "run model paths differ"
    );
    ensure!(
        left.sampler == right.sampler,
        "run sampler settings or seed differ"
    );
    let divergence = first_divergence(&left.generated_token_ids, &right.generated_token_ids);
    let left_readouts = readout_map(left)?;
    let right_readouts = readout_map(right)?;
    let mut matched_readouts = Vec::new();
    let mut unmatched_readouts = Vec::new();
    for key in left_readouts
        .keys()
        .chain(right_readouts.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
    {
        match (left_readouts.get(&key), right_readouts.get(&key)) {
            (Some(a), Some(b)) => matched_readouts.push(compare_readout(key, a, b, limit)),
            (Some(_), None) => unmatched_readouts.push(UnmatchedReadout {
                key,
                side: "left_only",
            }),
            (None, Some(_)) => unmatched_readouts.push(UnmatchedReadout {
                key,
                side: "right_only",
            }),
            (None, None) => unreachable!(),
        }
    }
    Ok(RunComparison {
        alignment: "exact readout key and exact candidate identity; no inferred alignment",
        runtime_kind: left.runtime_kind.clone(),
        model_path: left.model_path.clone(),
        plans_equal: left.plan == right.plan,
        left_generated_text: left.decoded_text.clone(),
        right_generated_text: right.decoded_text.clone(),
        left_generated_token_ids: left.generated_token_ids.clone(),
        right_generated_token_ids: right.generated_token_ids.clone(),
        first_generated_token_divergence: divergence,
        left_stop_reason: left.stop_reason.clone(),
        right_stop_reason: right.stop_reason.clone(),
        operation_applications: OperationComparison {
            left_total_count: left.operation_applications.len(),
            right_total_count: right.operation_applications.len(),
            left_applications: left
                .operation_applications
                .iter()
                .take(limit)
                .cloned()
                .collect(),
            right_applications: right
                .operation_applications
                .iter()
                .take(limit)
                .cloned()
                .collect(),
        },
        native_capture_counts: SideCounts {
            left: left.native_hyper_captures.len(),
            right: right.native_hyper_captures.len(),
        },
        matched_readout_count: matched_readouts.len(),
        matched_readouts: matched_readouts.into_iter().take(limit).collect(),
        unmatched_readout_count: unmatched_readouts.len(),
        unmatched_readouts: unmatched_readouts.into_iter().take(limit).collect(),
        detail_limit: limit,
    })
}

fn first_divergence(left: &[i32], right: &[i32]) -> Option<GeneratedDivergence> {
    let index = left
        .iter()
        .zip(right)
        .position(|(a, b)| a != b)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))?;
    Some(GeneratedDivergence {
        index,
        left_token_id: left.get(index).copied(),
        right_token_id: right.get(index).copied(),
        length_only: index == left.len().min(right.len()),
    })
}

fn readout_map(document: &RunDocument) -> Result<BTreeMap<ReadoutKey, &RunReadout>> {
    let mut map = BTreeMap::new();
    for readout in &document.live_readouts {
        ensure!(
            map.insert(readout.key(), readout).is_none(),
            "run repeats an exact readout key"
        );
    }
    Ok(map)
}

fn compare_readout(
    key: ReadoutKey,
    left: &RunReadout,
    right: &RunReadout,
    limit: usize,
) -> MatchedReadout {
    let left: BTreeMap<_, _> = left
        .scores
        .iter()
        .map(|score| (score.identity(), score.score))
        .collect();
    let right: BTreeMap<_, _> = right
        .scores
        .iter()
        .map(|score| (score.identity(), score.score))
        .collect();
    let differences: Vec<_> = left
        .keys()
        .chain(right.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|identity| {
            let a = left.get(&identity).copied();
            let b = right.get(&identity).copied();
            if a == b {
                return None;
            }
            Some(RunCandidateDifference {
                identity,
                status: match (a, b) {
                    (Some(_), Some(_)) => SelectedCandidateStatus::PresentBoth,
                    (None, Some(_)) => SelectedCandidateStatus::EnteredReadoutTopK,
                    (Some(_), None) => SelectedCandidateStatus::ExitedReadoutTopK,
                    (None, None) => unreachable!(),
                },
                left_score: a,
                right_score: b,
                score_delta: a.zip(b).map(|(x, y)| y - x),
            })
        })
        .collect();
    MatchedReadout {
        key,
        candidate_difference_count: differences.len(),
        candidate_differences: differences.into_iter().take(limit).collect(),
    }
}

fn print_text(result: &ComparisonResult) {
    match result {
        ComparisonResult::Trace(result) => {
            println!(
                "trace comparison: alignment=exact (layer,position) and token_id; no inference"
            );
            println!(
                "left method={} artifact={} file={:?} digest={:?} right method={} artifact={} file={:?} digest={:?}",
                result.left_lens.method,
                result.left_lens.artifact_kind,
                result.left_lens.source_filename,
                result.left_lens.payload_blake3,
                result.right_lens.method,
                result.right_lens.artifact_kind,
                result.right_lens.source_filename,
                result.right_lens.payload_blake3
            );
            println!(
                "score kind={:?} normalization={:?} universe={:?} softmax={:?}",
                result.score_semantics.kind,
                result.score_semantics.normalization,
                result.score_semantics.candidate_universe,
                result.score_semantics.softmax_applied
            );
            println!(
                "cells={} top1_changed={} changed={} first_changed={}",
                result.cell_count,
                result.top1_changed_cell_count,
                result.changed_cell_count,
                result.first_changed.map_or_else(
                    || "none".into(),
                    |c| format!("{}:{}", c.source_layer, c.source_position)
                )
            );
            for cell in &result.changed_cells {
                println!(
                    "cell {}:{} top1_changed={} candidate_differences={}",
                    cell.coordinate.source_layer,
                    cell.coordinate.source_position,
                    cell.top1_changed,
                    cell.candidate_difference_count
                );
                for candidate in &cell.candidates {
                    println!(
                        "  token={} status={} rank={:?}->{:?} logit={:?}->{:?} delta={:?}",
                        candidate.token_id,
                        candidate.status.as_str(),
                        candidate.left_rank,
                        candidate.right_rank,
                        candidate.left_logit,
                        candidate.right_logit,
                        candidate.logit_delta
                    );
                }
            }
            for row in &result.aggregate_differences {
                println!(
                    "aggregate token={} display={:?} count={}->{} delta={} top1={}->{} best_rank={:?}->{:?}",
                    row.token_id,
                    row.display,
                    row.left_count,
                    row.right_count,
                    row.count_delta,
                    row.left_top1_count,
                    row.right_top1_count,
                    row.left_best_rank,
                    row.right_best_rank
                );
            }
            println!(
                "aggregate_differences={} vectors_metadata_compatible={} vectors_matched={} vectors_unmatched={} vectors_dimension_incompatible={} vectors_metadata_incompatible={}",
                result.aggregate_difference_count,
                result.vectors.metadata_compatible,
                result.vectors.matched_cell_count,
                result.vectors.unmatched_cell_count,
                result.vectors.incompatible_dimension_count,
                result.vectors.metadata_incompatible_cell_count,
            );
            if let Some(mismatch) = &result.vectors.metadata_mismatch {
                println!("vector metadata mismatch: {mismatch}");
            }
            for vector in &result.vectors.matched_cells {
                println!(
                    "vector {}:{} dim={} cosine={:?} l2={} norms={}/{}",
                    vector.coordinate.source_layer,
                    vector.coordinate.source_position,
                    vector.dimension,
                    vector.cosine,
                    vector.l2_distance,
                    vector.left_norm,
                    vector.right_norm
                );
            }
            for vector in &result.vectors.unmatched_cells {
                println!(
                    "unmatched vector {}:{} side={} dim={}",
                    vector.coordinate.source_layer,
                    vector.coordinate.source_position,
                    vector.side,
                    vector.dimension
                );
            }
            for vector in &result.vectors.incompatible_dimensions {
                println!(
                    "incompatible vector {}:{} dimensions={}/{} (not compared)",
                    vector.coordinate.source_layer,
                    vector.coordinate.source_position,
                    vector.left_dimension,
                    vector.right_dimension
                );
            }
        }
        ComparisonResult::Run(result) => {
            println!(
                "run comparison: alignment=exact readout key and candidate identity; no inference"
            );
            println!(
                "runtime={} model={}",
                result.runtime_kind,
                result.model_path.display()
            );
            println!(
                "left generated_text={:?} token_ids={:?} stop={} right generated_text={:?} token_ids={:?} stop={}",
                result.left_generated_text,
                result.left_generated_token_ids,
                result.left_stop_reason,
                result.right_generated_text,
                result.right_generated_token_ids,
                result.right_stop_reason
            );
            println!(
                "first_generated_divergence={}",
                result
                    .first_generated_token_divergence
                    .as_ref()
                    .map_or_else(
                        || "none".into(),
                        |d| format!(
                            "index={} left={:?} right={:?} length_only={}",
                            d.index, d.left_token_id, d.right_token_id, d.length_only
                        )
                    )
            );
            println!(
                "operations left={} right={} native_captures left={} right={}",
                result.operation_applications.left_total_count,
                result.operation_applications.right_total_count,
                result.native_capture_counts.left,
                result.native_capture_counts.right
            );
            println!("plans_equal={}", result.plans_equal);
            for application in &result.operation_applications.left_applications {
                println!(
                    "left operation id={} layer={} phase={} index={}",
                    application.id, application.layer, application.phase, application.index
                );
            }
            for application in &result.operation_applications.right_applications {
                println!(
                    "right operation id={} layer={} phase={} index={}",
                    application.id, application.layer, application.phase, application.index
                );
            }
            println!(
                "readouts matched={} unmatched={}",
                result.matched_readout_count, result.unmatched_readout_count
            );
            for readout in &result.matched_readouts {
                println!(
                    "readout id={} lens={} method={} score_kind={} universe={} layer={} target={:?} phase={} index={} candidate_differences={}",
                    readout.key.id,
                    readout.key.lens,
                    readout.key.method,
                    readout.key.score_kind,
                    readout.key.candidate_universe,
                    readout.key.source_layer,
                    readout.key.target_layer,
                    readout.key.phase,
                    readout.key.index,
                    readout.candidate_difference_count
                );
                for candidate in &readout.candidate_differences {
                    println!(
                        "  candidate token={:?} row={} word={:?} label={:?} status={} score={:?}->{:?} delta={:?}",
                        candidate.identity.token_id,
                        candidate.identity.row_id,
                        candidate.identity.word_id,
                        candidate.identity.label,
                        candidate.status.as_str(),
                        candidate.left_score,
                        candidate.right_score,
                        candidate.score_delta
                    );
                }
            }
            for readout in &result.unmatched_readouts {
                println!(
                    "unmatched readout side={} id={} lens={} method={} score_kind={} universe={} layer={} target={:?} phase={} index={}",
                    readout.side,
                    readout.key.id,
                    readout.key.lens,
                    readout.key.method,
                    readout.key.score_kind,
                    readout.key.candidate_universe,
                    readout.key.source_layer,
                    readout.key.target_layer,
                    readout.key.phase,
                    readout.key.index
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn trace(top_k: serde_json::Value, vectors: Option<serde_json::Value>) -> TraceDocument {
        let scores = top_k.as_array().unwrap();
        let occurrences: Vec<_> = scores
            .iter()
            .map(|score| {
                json!({
                    "token_id": score["token_id"],
                    "count": 1,
                    "top1_count": usize::from(score["rank"] == 0),
                    "best_rank": score["rank"]
                })
            })
            .collect();
        let value = json!({
            "schema": "qwen.lens.trace",
            "schema_version": 3,
            "producer": {},
            "deployed_model": {"locator_id": "model", "vocab_size": 100},
            "tokenizer": {"metadata_id": "tokenizer"},
            "lens": {"kind": "published_full_transport", "method": "J", "payload_blake3": "artifact"},
            "score_semantics": {"kind": "logit", "normalization": "rmsnorm", "candidate_universe": "full_model_vocabulary", "softmax_applied": false},
            "execution_mode": "passive",
            "input_token_ids": [10],
            "input_tokens": [{"position": 0, "token_id": 10, "token_display_lossy": "a"}],
            "rendering": {"renderer": "test", "spans": []},
            "selected_layers": [2],
            "top_k": 2,
            "cells": [{"source_layer": 2, "source_position": 0, "source_token_id": 10, "predicts_position": 1, "top_k": top_k}],
            "vectors": vectors,
            "timing": {},
            "occurrences": {"global": occurrences, "per_layer": [{"source_layer": 2, "tokens": occurrences}]}
        });
        lens_inspect::parse_trace_bytes(
            &serde_json::to_vec(&value).unwrap(),
            std::path::Path::new("synthetic-trace.json"),
        )
        .unwrap()
    }

    fn scores(first: u32, second: u32, first_logit: f32) -> serde_json::Value {
        json!([
            {"rank": 0, "token_id": first, "token_display_lossy": first.to_string(), "logit": first_logit},
            {"rank": 1, "token_id": second, "token_display_lossy": second.to_string(), "logit": 1.0}
        ])
    }

    fn vectors(values: &[f32]) -> serde_json::Value {
        json!({
            "operation": "test", "stage": "test", "value_dtype": "f32",
            "hidden_size": values.len(), "shape": [1, values.len()],
            "cells": [{"source_layer": 2, "source_position": 0, "source_token_id": 10, "predicts_position": 1, "values": values}]
        })
    }

    #[test]
    fn rejects_incompatible_trace_coordinates() {
        let left = trace(scores(7, 8, 2.0), None);
        let mut right = trace(scores(7, 8, 2.0), None);
        right.selected_layers = vec![3];
        assert!(compare_traces(&left, &right, 10).is_err());
    }

    #[test]
    fn reports_top_k_entry_and_exit_without_invented_values() {
        let comparison = compare_traces(
            &trace(scores(7, 8, 2.0), None),
            &trace(scores(7, 9, 2.0), None),
            10,
        )
        .unwrap();
        let candidates = &comparison.changed_cells[0].candidates;
        let exited = candidates.iter().find(|row| row.token_id == 8).unwrap();
        let entered = candidates.iter().find(|row| row.token_id == 9).unwrap();
        assert!(matches!(exited.status, CandidateStatus::ExitedCapturedTopK));
        assert!(
            exited.right_rank.is_none()
                && exited.right_logit.is_none()
                && exited.logit_delta.is_none()
        );
        assert!(matches!(
            entered.status,
            CandidateStatus::EnteredCapturedTopK
        ));
        assert!(
            entered.left_rank.is_none()
                && entered.left_logit.is_none()
                && entered.logit_delta.is_none()
        );
    }

    #[test]
    fn computes_matching_vector_metrics() {
        let comparison = compare_traces(
            &trace(scores(7, 8, 2.0), Some(vectors(&[1.0, 0.0]))),
            &trace(scores(7, 8, 2.0), Some(vectors(&[0.0, 1.0]))),
            10,
        )
        .unwrap();
        let metrics = &comparison.vectors.matched_cells[0];
        assert_eq!(metrics.cosine, Some(0.0));
        assert!((metrics.l2_distance - 2.0_f64.sqrt()).abs() < 1e-12);
        assert_eq!((metrics.left_norm, metrics.right_norm), (1.0, 1.0));
    }

    fn run_document(generated: Vec<i32>, scores: Vec<RunScore>) -> RunDocument {
        RunDocument {
            schema: "qwen.lens.run".into(),
            schema_version: 1,
            runtime_kind: "ordinary_qwen".into(),
            model_path: "model.gguf".into(),
            _canonical_plan_path: "/plan.json".into(),
            plan: serde_json::from_value(json!({"version": 1, "lenses": [], "directions": [], "operations": [], "readouts": []})).unwrap(),
            _input_source: "token_ids".into(),
            prompt_token_ids: vec![1, 2],
            generated_token_ids: generated,
            sampler: RunSampler {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: 7,
            },
            _max_new_tokens: 2,
            decoded_text: "text".into(),
            stop_reason: "max_new_tokens".into(),
            operation_applications: Vec::new(),
            _requested_live_readouts: Vec::new(),
            live_readouts: vec![RunReadout {
                id: "readout".into(),
                lens: "lens".into(),
                method: "J".into(),
                score_kind: "selected_row_score".into(),
                candidate_universe: "selected_rows".into(),
                source_layer: 2,
                target_layer: Some(3),
                phase: "decode".into(),
                index: 0,
                scores,
            }],
            native_hyper_captures: Vec::new(),
        }
    }

    fn run_score(token_id: i32, score: f32) -> RunScore {
        RunScore {
            token_id: Some(token_id),
            row_id: token_id as usize,
            word_id: None,
            label: None,
            score,
        }
    }

    #[test]
    fn reports_generated_length_only_divergence() {
        let comparison = compare_runs(
            &run_document(vec![3], vec![]),
            &run_document(vec![3, 4], vec![]),
            10,
        )
        .unwrap();
        let divergence = comparison.first_generated_token_divergence.unwrap();
        assert_eq!(divergence.index, 1);
        assert!(divergence.length_only);
        assert_eq!(
            (divergence.left_token_id, divergence.right_token_id),
            (None, Some(4))
        );
    }

    #[test]
    fn aligns_readout_scores_only_by_exact_candidate_identity() {
        let comparison = compare_runs(
            &run_document(vec![3], vec![run_score(7, 1.0), run_score(8, 2.0)]),
            &run_document(vec![3], vec![run_score(7, 1.5), run_score(9, 3.0)]),
            10,
        )
        .unwrap();
        let rows = &comparison.matched_readouts[0].candidate_differences;
        let matched = rows
            .iter()
            .find(|row| row.identity.token_id == Some(7))
            .unwrap();
        assert_eq!(matched.score_delta, Some(0.5));
        let exited = rows
            .iter()
            .find(|row| row.identity.token_id == Some(8))
            .unwrap();
        assert!(matches!(
            exited.status,
            SelectedCandidateStatus::ExitedReadoutTopK
        ));
        assert!(exited.right_score.is_none() && exited.score_delta.is_none());
        let entered = rows
            .iter()
            .find(|row| row.identity.token_id == Some(9))
            .unwrap();
        assert!(matches!(
            entered.status,
            SelectedCandidateStatus::EnteredReadoutTopK
        ));
        assert!(entered.left_score.is_none() && entered.score_delta.is_none());
    }
}
