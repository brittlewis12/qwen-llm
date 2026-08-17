use anyhow::{Result, ensure};
use qwen_llm::deepseek_v4_metal::{DeepSeekV4CsaDecision, DeepSeekV4DecisionTranscript};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const CADENCES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
const CONE_BLOCK_ROWS: [usize; 5] = [16, 32, 64, 128, 256];
const LOCALITY_PAGE_ROWS: [usize; 4] = [8, 16, 32, 64];
const LOCALITY_MERGE_GAPS: [usize; 4] = [1, 3, 7, 15];

#[derive(Clone, Copy, Debug)]
struct CandidateSample {
    candidate_rows: usize,
    visible_rows: usize,
}

impl CandidateSample {
    fn fraction(self) -> f64 {
        self.candidate_rows as f64 / self.visible_rows as f64
    }

    fn full(visible_rows: usize) -> Self {
        Self {
            candidate_rows: visible_rows,
            visible_rows,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct AnchorObservation {
    oracle: CandidateSample,
    charged: CandidateSample,
    refresh_endpoint: Option<CandidateSample>,
}

#[derive(Debug)]
struct Anchor {
    cadence: usize,
    scores: Vec<f32>,
    age: usize,
}

impl Anchor {
    fn new(cadence: usize, scores: &[f32]) -> Self {
        Self {
            cadence,
            scores: scores.to_vec(),
            age: 0,
        }
    }

    fn observe(&mut self, current: &DeepSeekV4CsaDecision) -> Result<AnchorObservation> {
        ensure!(
            current.visible_scores.len() >= self.scores.len(),
            "CSA visible score history shrank from {} to {}",
            self.scores.len(),
            current.visible_scores.len()
        );
        let direct_upward_delta =
            max_upward_delta(&self.scores, &current.visible_scores[..self.scores.len()]);
        let cutoff = f64::from(current.rank_512.score);
        let prior_candidates = self
            .scores
            .iter()
            .filter(|&&score| upper_sum(f64::from(score), direct_upward_delta) >= cutoff)
            .count();
        let new_rows = current.visible_scores.len() - self.scores.len();
        let oracle = CandidateSample {
            candidate_rows: prior_candidates + new_rows,
            visible_rows: current.visible_scores.len(),
        };
        self.age += 1;
        let refresh = self.age >= self.cadence;
        if refresh {
            self.scores.clone_from(&current.visible_scores);
            self.age = 0;
        }
        Ok(AnchorObservation {
            oracle,
            charged: if refresh {
                CandidateSample::full(current.visible_scores.len())
            } else {
                oracle
            },
            refresh_endpoint: refresh.then_some(oracle),
        })
    }
}

fn max_upward_delta(previous: &[f32], current: &[f32]) -> f64 {
    previous
        .iter()
        .zip(current)
        .map(|(&previous, &current)| (f64::from(current) - f64::from(previous)).max(0.0))
        .fold(0.0f64, f64::max)
}

fn upper_sum(left: f64, right: f64) -> f64 {
    let value = left + right;
    if !value.is_finite() || value == f64::INFINITY {
        return value;
    }
    if value == -0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value >= 0.0 { bits + 1 } else { bits - 1 })
}

fn upper_product(left: f64, right: f64) -> f64 {
    let value = left * right;
    if !value.is_finite() || value == f64::INFINITY {
        return value;
    }
    f64::from_bits(value.to_bits() + 1)
}

#[derive(Debug)]
struct LayerState {
    previous_scores: Vec<f32>,
    previous_ids: Vec<u32>,
    anchors: Vec<Anchor>,
}

impl LayerState {
    fn new(decision: &DeepSeekV4CsaDecision) -> Self {
        Self {
            previous_scores: decision.visible_scores.clone(),
            previous_ids: decision.cache_order_selected_ids.clone(),
            anchors: CADENCES
                .into_iter()
                .map(|cadence| Anchor::new(cadence, &decision.visible_scores))
                .collect(),
        }
    }
}

#[derive(Debug, Default)]
struct LayerSamples {
    jaccard: Vec<f64>,
    step_max_upward_delta: Vec<f64>,
    cutoff_margin: Vec<f64>,
    head_weights: Vec<f64>,
    one_step_candidates: Vec<CandidateSample>,
    cadence_oracle_candidates: BTreeMap<usize, Vec<CandidateSample>>,
    cadence_charged_work: BTreeMap<usize, Vec<CandidateSample>>,
    cadence_refresh_endpoints: BTreeMap<usize, Vec<CandidateSample>>,
    cone_envelope: ConeEnvelopeSamples,
    selected_locality: SelectedLocalitySamples,
}

#[derive(Debug, Default)]
struct ConeEnvelopeSamples {
    bound_checks: u64,
    bound_violations: u64,
    max_violation: f64,
    nonpositive_cutoffs: usize,
    retained_work: BTreeMap<usize, Vec<CandidateSample>>,
}

impl ConeEnvelopeSamples {
    fn record(
        &mut self,
        decision: &DeepSeekV4CsaDecision,
        head_weights: &[f32],
        query_norms: &[f64],
        key_norms: &[f64],
    ) -> Result<()> {
        ensure!(
            head_weights.len() == 64,
            "Lightning capture requires 64 head weights"
        );
        ensure!(
            query_norms.len() == 64,
            "Lightning capture requires 64 query norms"
        );
        ensure!(
            key_norms.len() == decision.visible_scores.len(),
            "Lightning capture has {} key norms for {} visible scores",
            key_norms.len(),
            decision.visible_scores.len()
        );
        ensure!(
            query_norms
                .iter()
                .chain(key_norms)
                .all(|norm| norm.is_finite() && *norm >= 0.0),
            "Lightning capture contains an invalid norm"
        );

        let mut positive_scale = 0.0f64;
        for (&weight, &query_norm) in head_weights.iter().zip(query_norms) {
            if weight > 0.0 {
                positive_scale =
                    upper_sum(positive_scale, upper_product(f64::from(weight), query_norm));
            }
        }
        let row_bounds = key_norms
            .iter()
            .map(|&key_norm| upper_product(positive_scale, key_norm))
            .collect::<Vec<_>>();
        for (&score, &bound) in decision.visible_scores.iter().zip(&row_bounds) {
            self.bound_checks += 1;
            let violation = f64::from(score) - bound;
            if violation > 0.0 {
                self.bound_violations += 1;
                self.max_violation = self.max_violation.max(violation);
            }
        }

        let cutoff = f64::from(decision.rank_512.score);
        if cutoff <= 0.0 {
            self.nonpositive_cutoffs += 1;
        }
        for block_rows in CONE_BLOCK_ROWS {
            let candidate_rows = row_bounds
                .chunks(block_rows)
                .map(|block| {
                    let upper = block.iter().copied().fold(0.0f64, f64::max);
                    if upper >= cutoff { block.len() } else { 0 }
                })
                .sum();
            self.retained_work
                .entry(block_rows)
                .or_default()
                .push(CandidateSample {
                    candidate_rows,
                    visible_rows: row_bounds.len(),
                });
        }
        Ok(())
    }

    fn extend_from(&mut self, other: &Self) {
        self.bound_checks += other.bound_checks;
        self.bound_violations += other.bound_violations;
        self.max_violation = self.max_violation.max(other.max_violation);
        self.nonpositive_cutoffs += other.nonpositive_cutoffs;
        for (&rows, values) in &other.retained_work {
            self.retained_work
                .entry(rows)
                .or_default()
                .extend_from_slice(values);
        }
    }
}

#[derive(Debug, Default)]
struct SelectedLocalitySamples {
    runs: Vec<f64>,
    mean_run_length: Vec<f64>,
    max_run_length: Vec<f64>,
    contiguous_adjacency_fraction: Vec<f64>,
    run_descriptor_to_id_bytes: Vec<f64>,
    page_payload_amplification: BTreeMap<usize, Vec<f64>>,
    merged_span_count: BTreeMap<usize, Vec<f64>>,
    merged_span_payload_amplification: BTreeMap<usize, Vec<f64>>,
}

impl SelectedLocalitySamples {
    fn record(&mut self, selected_ids: &[u32], visible_rows: usize) -> Result<()> {
        ensure!(!selected_ids.is_empty(), "CSA selection is empty");
        ensure!(
            selected_ids.windows(2).all(|pair| pair[0] < pair[1]),
            "CSA cache-order IDs are not strictly ascending"
        );
        ensure!(
            selected_ids.last().copied().unwrap_or_default() < visible_rows as u32,
            "CSA selected ID exceeds visible rows"
        );

        let mut runs = 1usize;
        let mut current_run = 1usize;
        let mut max_run = 1usize;
        for pair in selected_ids.windows(2) {
            if pair[1] == pair[0] + 1 {
                current_run += 1;
                max_run = max_run.max(current_run);
            } else {
                runs += 1;
                current_run = 1;
            }
        }
        let selected_rows = selected_ids.len();
        self.runs.push(runs as f64);
        self.mean_run_length
            .push(selected_rows as f64 / runs as f64);
        self.max_run_length.push(max_run as f64);
        self.contiguous_adjacency_fraction.push(
            selected_rows.saturating_sub(runs) as f64
                / selected_rows.saturating_sub(1).max(1) as f64,
        );
        self.run_descriptor_to_id_bytes
            .push((runs * 8) as f64 / (selected_rows * 4) as f64);

        for page_rows in LOCALITY_PAGE_ROWS {
            let mut pages = Vec::new();
            for &row in selected_ids {
                let page = row as usize / page_rows;
                if pages.last().copied() != Some(page) {
                    pages.push(page);
                }
            }
            let loaded_rows = pages
                .into_iter()
                .map(|page| (visible_rows - page * page_rows).min(page_rows))
                .sum::<usize>();
            self.page_payload_amplification
                .entry(page_rows)
                .or_default()
                .push(loaded_rows as f64 / selected_rows as f64);
        }

        for max_gap in LOCALITY_MERGE_GAPS {
            let mut spans = 1usize;
            let mut loaded_rows = 1usize;
            for pair in selected_ids.windows(2) {
                let gap = (pair[1] - pair[0] - 1) as usize;
                if gap <= max_gap {
                    loaded_rows += (pair[1] - pair[0]) as usize;
                } else {
                    spans += 1;
                    loaded_rows += 1;
                }
            }
            self.merged_span_count
                .entry(max_gap)
                .or_default()
                .push(spans as f64);
            self.merged_span_payload_amplification
                .entry(max_gap)
                .or_default()
                .push(loaded_rows as f64 / selected_rows as f64);
        }
        Ok(())
    }

    fn extend_from(&mut self, other: &Self) {
        self.runs.extend_from_slice(&other.runs);
        self.mean_run_length
            .extend_from_slice(&other.mean_run_length);
        self.max_run_length.extend_from_slice(&other.max_run_length);
        self.contiguous_adjacency_fraction
            .extend_from_slice(&other.contiguous_adjacency_fraction);
        self.run_descriptor_to_id_bytes
            .extend_from_slice(&other.run_descriptor_to_id_bytes);
        for (&key, values) in &other.page_payload_amplification {
            self.page_payload_amplification
                .entry(key)
                .or_default()
                .extend_from_slice(values);
        }
        for (&key, values) in &other.merged_span_count {
            self.merged_span_count
                .entry(key)
                .or_default()
                .extend_from_slice(values);
        }
        for (&key, values) in &other.merged_span_payload_amplification {
            self.merged_span_payload_amplification
                .entry(key)
                .or_default()
                .extend_from_slice(values);
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MetricSummary {
    samples: usize,
    min: f64,
    mean: f64,
    median: f64,
    p95: f64,
    max: f64,
}

impl MetricSummary {
    fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                samples: 0,
                min: 0.0,
                mean: 0.0,
                median: 0.0,
                p95: 0.0,
                max: 0.0,
            };
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let p95_index = (sorted.len() * 95).div_ceil(100).saturating_sub(1);
        Self {
            samples: sorted.len(),
            min: sorted[0],
            mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
            median: sorted[sorted.len() / 2],
            p95: sorted[p95_index],
            max: sorted[sorted.len() - 1],
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CandidateSummary {
    samples: usize,
    candidate_rows: u64,
    visible_rows: u64,
    weighted_fraction: f64,
    distribution: MetricSummary,
}

impl CandidateSummary {
    fn from_samples(samples: &[CandidateSample]) -> Self {
        let candidate_rows = samples
            .iter()
            .map(|sample| sample.candidate_rows as u64)
            .sum::<u64>();
        let visible_rows = samples
            .iter()
            .map(|sample| sample.visible_rows as u64)
            .sum::<u64>();
        Self {
            samples: samples.len(),
            candidate_rows,
            visible_rows,
            weighted_fraction: if visible_rows == 0 {
                0.0
            } else {
                candidate_rows as f64 / visible_rows as f64
            },
            distribution: MetricSummary::from_values(
                &samples
                    .iter()
                    .map(|sample| sample.fraction())
                    .collect::<Vec<_>>(),
            ),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct HeadWeightSummary {
    values: MetricSummary,
    negative_values: usize,
    total_values: usize,
    negative_fraction: f64,
}

impl HeadWeightSummary {
    fn from_values(values: &[f64]) -> Self {
        let negative_values = values.iter().filter(|&&value| value < 0.0).count();
        Self {
            values: MetricSummary::from_values(values),
            negative_values,
            total_values: values.len(),
            negative_fraction: if values.is_empty() {
                0.0
            } else {
                negative_values as f64 / values.len() as f64
            },
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SelectedLocalityReport {
    runs: MetricSummary,
    mean_run_length: MetricSummary,
    max_run_length: MetricSummary,
    contiguous_adjacency_fraction: MetricSummary,
    run_descriptor_to_id_bytes: MetricSummary,
    page_payload_amplification: BTreeMap<usize, MetricSummary>,
    merged_span_count: BTreeMap<usize, MetricSummary>,
    merged_span_payload_amplification: BTreeMap<usize, MetricSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConeEnvelopeReport {
    bound_checks: u64,
    bound_violations: u64,
    max_violation: f64,
    nonpositive_cutoffs: usize,
    retained_work: BTreeMap<usize, CandidateSummary>,
}

impl ConeEnvelopeReport {
    fn from_samples(samples: &ConeEnvelopeSamples) -> Self {
        Self {
            bound_checks: samples.bound_checks,
            bound_violations: samples.bound_violations,
            max_violation: samples.max_violation,
            nonpositive_cutoffs: samples.nonpositive_cutoffs,
            retained_work: samples
                .retained_work
                .iter()
                .map(|(&rows, values)| (rows, CandidateSummary::from_samples(values)))
                .collect(),
        }
    }
}

impl SelectedLocalityReport {
    fn from_samples(samples: &SelectedLocalitySamples) -> Self {
        Self {
            runs: MetricSummary::from_values(&samples.runs),
            mean_run_length: MetricSummary::from_values(&samples.mean_run_length),
            max_run_length: MetricSummary::from_values(&samples.max_run_length),
            contiguous_adjacency_fraction: MetricSummary::from_values(
                &samples.contiguous_adjacency_fraction,
            ),
            run_descriptor_to_id_bytes: MetricSummary::from_values(
                &samples.run_descriptor_to_id_bytes,
            ),
            page_payload_amplification: samples
                .page_payload_amplification
                .iter()
                .map(|(&rows, values)| (rows, MetricSummary::from_values(values)))
                .collect(),
            merged_span_count: samples
                .merged_span_count
                .iter()
                .map(|(&gap, values)| (gap, MetricSummary::from_values(values)))
                .collect(),
            merged_span_payload_amplification: samples
                .merged_span_payload_amplification
                .iter()
                .map(|(&gap, values)| (gap, MetricSummary::from_values(values)))
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DecisionPoint {
    position: u32,
    layer: u32,
    visible_rows: usize,
    selected_ids_sha256: String,
    selected_count: u32,
    selection_status: i32,
    rank_512_row: u32,
    rank_512_score: f32,
    rank_513_row: u32,
    rank_513_score: f32,
    cutoff_margin: f64,
    head_weights_sha256: String,
    route_sha256: String,
}

#[derive(Debug, Serialize)]
pub struct LayerReport {
    layer: u32,
    comparisons: usize,
    jaccard: MetricSummary,
    step_max_upward_delta: MetricSummary,
    cutoff_margin: MetricSummary,
    head_weights: HeadWeightSummary,
    one_step_candidates: CandidateSummary,
    cadence_oracle_candidates: BTreeMap<usize, CandidateSummary>,
    cadence_charged_work: BTreeMap<usize, CandidateSummary>,
    cadence_refresh_endpoints: BTreeMap<usize, CandidateSummary>,
    cone_envelope: ConeEnvelopeReport,
    selected_locality: SelectedLocalityReport,
}

#[derive(Debug, Serialize)]
pub struct TemporalReport {
    schema_version: u32,
    interpretation: &'static str,
    requested_tokens: usize,
    captured_tokens: usize,
    first_position: Option<u32>,
    last_position: Option<u32>,
    aggregate_jaccard: MetricSummary,
    aggregate_head_weights: HeadWeightSummary,
    aggregate_one_step_candidates: CandidateSummary,
    aggregate_cadence_oracle_candidates: BTreeMap<usize, CandidateSummary>,
    aggregate_cadence_charged_work: BTreeMap<usize, CandidateSummary>,
    aggregate_cadence_refresh_endpoints: BTreeMap<usize, CandidateSummary>,
    aggregate_cone_envelope: ConeEnvelopeReport,
    aggregate_selected_locality: SelectedLocalityReport,
    selection_trace_sha256: String,
    route_trace_sha256: String,
    final_logits_sha256: Option<String>,
    final_causal_digest: Option<String>,
    decision_points: Vec<DecisionPoint>,
    layers: Vec<LayerReport>,
}

#[derive(Debug, Serialize)]
pub struct TemporalSummary {
    schema_version: u32,
    requested_tokens: usize,
    captured_tokens: usize,
    first_position: Option<u32>,
    last_position: Option<u32>,
    aggregate_jaccard: MetricSummary,
    aggregate_head_weights: HeadWeightSummary,
    aggregate_one_step_candidates: CandidateSummary,
    aggregate_cadence_oracle_candidates: BTreeMap<usize, CandidateSummary>,
    aggregate_cadence_charged_work: BTreeMap<usize, CandidateSummary>,
    aggregate_cone_envelope: ConeEnvelopeReport,
    aggregate_selected_locality: SelectedLocalityReport,
    selection_trace_sha256: String,
    route_trace_sha256: String,
    final_logits_sha256: Option<String>,
    final_causal_digest: Option<String>,
}

impl TemporalReport {
    pub fn summary(&self) -> TemporalSummary {
        TemporalSummary {
            schema_version: self.schema_version,
            requested_tokens: self.requested_tokens,
            captured_tokens: self.captured_tokens,
            first_position: self.first_position,
            last_position: self.last_position,
            aggregate_jaccard: self.aggregate_jaccard.clone(),
            aggregate_head_weights: self.aggregate_head_weights.clone(),
            aggregate_one_step_candidates: self.aggregate_one_step_candidates.clone(),
            aggregate_cadence_oracle_candidates: self.aggregate_cadence_oracle_candidates.clone(),
            aggregate_cadence_charged_work: self.aggregate_cadence_charged_work.clone(),
            aggregate_cone_envelope: self.aggregate_cone_envelope.clone(),
            aggregate_selected_locality: self.aggregate_selected_locality.clone(),
            selection_trace_sha256: self.selection_trace_sha256.clone(),
            route_trace_sha256: self.route_trace_sha256.clone(),
            final_logits_sha256: self.final_logits_sha256.clone(),
            final_causal_digest: self.final_causal_digest.clone(),
        }
    }

    pub fn attach_final_state(&mut self, logits_sha256: String, causal_digest: Option<String>) {
        self.final_logits_sha256 = Some(logits_sha256);
        self.final_causal_digest = causal_digest;
    }
}

pub struct TemporalCapture {
    requested_tokens: usize,
    positions: Vec<u32>,
    states: BTreeMap<u32, LayerState>,
    samples: BTreeMap<u32, LayerSamples>,
    selection_trace: Sha256,
    route_trace: Sha256,
    decision_points: Vec<DecisionPoint>,
}

impl TemporalCapture {
    pub fn new(requested_tokens: usize) -> Self {
        let mut selection_trace = Sha256::new();
        selection_trace.update(b"qwen-dsv4-temporal-selection-v1\0");
        let mut route_trace = Sha256::new();
        route_trace.update(b"qwen-dsv4-temporal-route-v1\0");
        Self {
            requested_tokens,
            positions: Vec::with_capacity(requested_tokens),
            states: BTreeMap::new(),
            samples: BTreeMap::new(),
            selection_trace,
            route_trace,
            decision_points: Vec::with_capacity(requested_tokens * 21),
        }
    }

    pub fn should_capture(&self) -> bool {
        self.positions.len() < self.requested_tokens
    }

    pub fn requested_tokens(&self) -> usize {
        self.requested_tokens
    }

    pub fn record(&mut self, transcript: DeepSeekV4DecisionTranscript) -> Result<()> {
        ensure!(
            transcript.position
                == self
                    .positions
                    .last()
                    .copied()
                    .map_or(transcript.position, |position| position + 1),
            "temporal decision positions are not consecutive"
        );
        self.positions.push(transcript.position);
        for layer in transcript.layers {
            let layer_id = layer.layer;
            let route_sha256 = digest_route(
                transcript.position,
                layer_id,
                &layer.route.expert_ids,
                &layer.route.normalized_scaled_weights,
                layer.route.routed_scale,
            );
            self.route_trace.update(transcript.position.to_le_bytes());
            self.route_trace.update(layer_id.to_le_bytes());
            self.route_trace
                .update((layer.route.expert_ids.len() as u32).to_le_bytes());
            for expert_id in &layer.route.expert_ids {
                self.route_trace.update(expert_id.to_le_bytes());
            }
            for weight in &layer.route.normalized_scaled_weights {
                self.route_trace.update(weight.to_bits().to_le_bytes());
            }
            self.route_trace
                .update(layer.route.routed_scale.to_bits().to_le_bytes());
            let head_weights = layer.indexer_head_weights;
            let query_norms = layer.indexer_query_norms;
            let key_norms = layer.indexer_key_norms;
            let Some(decision) = layer.csa else {
                continue;
            };
            let cutoff_margin =
                f64::from(decision.rank_512.score) - f64::from(decision.rank_513.score);
            self.decision_points.push(DecisionPoint {
                position: transcript.position,
                layer: layer_id,
                visible_rows: decision.visible_scores.len(),
                selected_ids_sha256: digest_u32s(
                    b"qwen-dsv4-temporal-selected-ids-v1\0",
                    &decision.cache_order_selected_ids,
                ),
                selected_count: decision.selected_count,
                selection_status: decision.selection_status,
                rank_512_row: decision.rank_512.row_id,
                rank_512_score: decision.rank_512.score,
                rank_513_row: decision.rank_513.row_id,
                rank_513_score: decision.rank_513.score,
                cutoff_margin,
                head_weights_sha256: digest_f32s(
                    b"qwen-dsv4-temporal-head-weights-v1\0",
                    &head_weights,
                ),
                route_sha256,
            });
            self.selection_trace
                .update(transcript.position.to_le_bytes());
            self.selection_trace.update(layer_id.to_le_bytes());
            self.selection_trace
                .update((decision.visible_scores.len() as u32).to_le_bytes());
            for row_id in &decision.cache_order_selected_ids {
                self.selection_trace.update(row_id.to_le_bytes());
            }
            ensure!(
                head_weights.len() == 64,
                "layer {layer_id} temporal capture has {} head weights, expected 64",
                head_weights.len()
            );
            let samples = self.samples.entry(layer_id).or_default();
            samples
                .cone_envelope
                .record(&decision, &head_weights, &query_norms, &key_norms)?;
            samples
                .head_weights
                .extend(head_weights.into_iter().map(f64::from));
            samples.selected_locality.record(
                &decision.cache_order_selected_ids,
                decision.visible_scores.len(),
            )?;
            let Some(state) = self.states.get_mut(&layer_id) else {
                self.states.insert(layer_id, LayerState::new(&decision));
                continue;
            };
            ensure!(
                decision.visible_scores.len() >= state.previous_scores.len(),
                "layer {} visible score history shrank",
                layer_id
            );
            let common_rows = state.previous_scores.len();
            let step_max_upward_delta = max_upward_delta(
                &state.previous_scores,
                &decision.visible_scores[..common_rows],
            );
            let intersection = ascending_intersection_count(
                &state.previous_ids,
                &decision.cache_order_selected_ids,
            );
            let union =
                state.previous_ids.len() + decision.cache_order_selected_ids.len() - intersection;
            let jaccard = intersection as f64 / union as f64;
            let cutoff = f64::from(decision.rank_512.score);
            let prior_candidates = state
                .previous_scores
                .iter()
                .filter(|&&score| upper_sum(f64::from(score), step_max_upward_delta) >= cutoff)
                .count();
            let new_rows = decision.visible_scores.len() - common_rows;
            let one_step_candidates = CandidateSample {
                candidate_rows: prior_candidates + new_rows,
                visible_rows: decision.visible_scores.len(),
            };

            samples.jaccard.push(jaccard);
            samples.step_max_upward_delta.push(step_max_upward_delta);
            samples.cutoff_margin.push(cutoff_margin);
            samples.one_step_candidates.push(one_step_candidates);
            for anchor in &mut state.anchors {
                let observation = anchor.observe(&decision)?;
                samples
                    .cadence_oracle_candidates
                    .entry(anchor.cadence)
                    .or_default()
                    .push(observation.oracle);
                samples
                    .cadence_charged_work
                    .entry(anchor.cadence)
                    .or_default()
                    .push(observation.charged);
                if let Some(endpoint) = observation.refresh_endpoint {
                    samples
                        .cadence_refresh_endpoints
                        .entry(anchor.cadence)
                        .or_default()
                        .push(endpoint);
                }
            }
            state.previous_scores = decision.visible_scores;
            state.previous_ids = decision.cache_order_selected_ids;
        }
        Ok(())
    }

    pub fn finish(self) -> TemporalReport {
        let selection_trace_sha256 = format!("{:x}", self.selection_trace.finalize());
        let route_trace_sha256 = format!("{:x}", self.route_trace.finalize());
        let mut aggregate_jaccard = Vec::new();
        let mut aggregate_head_weights = Vec::new();
        let mut aggregate_one_step = Vec::new();
        let mut aggregate_oracle: BTreeMap<usize, Vec<CandidateSample>> = BTreeMap::new();
        let mut aggregate_charged: BTreeMap<usize, Vec<CandidateSample>> = BTreeMap::new();
        let mut aggregate_endpoints: BTreeMap<usize, Vec<CandidateSample>> = BTreeMap::new();
        let mut aggregate_cone = ConeEnvelopeSamples::default();
        let mut aggregate_locality = SelectedLocalitySamples::default();
        let mut layers = Vec::with_capacity(self.samples.len());
        for (layer, samples) in self.samples {
            aggregate_jaccard.extend_from_slice(&samples.jaccard);
            aggregate_head_weights.extend_from_slice(&samples.head_weights);
            aggregate_one_step.extend_from_slice(&samples.one_step_candidates);
            for (&cadence, values) in &samples.cadence_oracle_candidates {
                aggregate_oracle
                    .entry(cadence)
                    .or_default()
                    .extend_from_slice(values);
            }
            for (&cadence, values) in &samples.cadence_charged_work {
                aggregate_charged
                    .entry(cadence)
                    .or_default()
                    .extend_from_slice(values);
            }
            for (&cadence, values) in &samples.cadence_refresh_endpoints {
                aggregate_endpoints
                    .entry(cadence)
                    .or_default()
                    .extend_from_slice(values);
            }
            aggregate_cone.extend_from(&samples.cone_envelope);
            aggregate_locality.extend_from(&samples.selected_locality);
            layers.push(LayerReport {
                layer,
                comparisons: samples.jaccard.len(),
                jaccard: MetricSummary::from_values(&samples.jaccard),
                step_max_upward_delta: MetricSummary::from_values(&samples.step_max_upward_delta),
                cutoff_margin: MetricSummary::from_values(&samples.cutoff_margin),
                head_weights: HeadWeightSummary::from_values(&samples.head_weights),
                one_step_candidates: CandidateSummary::from_samples(&samples.one_step_candidates),
                cadence_oracle_candidates: samples
                    .cadence_oracle_candidates
                    .into_iter()
                    .map(|(cadence, values)| (cadence, CandidateSummary::from_samples(&values)))
                    .collect(),
                cadence_charged_work: samples
                    .cadence_charged_work
                    .into_iter()
                    .map(|(cadence, values)| (cadence, CandidateSummary::from_samples(&values)))
                    .collect(),
                cadence_refresh_endpoints: samples
                    .cadence_refresh_endpoints
                    .into_iter()
                    .map(|(cadence, values)| (cadence, CandidateSummary::from_samples(&values)))
                    .collect(),
                cone_envelope: ConeEnvelopeReport::from_samples(&samples.cone_envelope),
                selected_locality: SelectedLocalityReport::from_samples(&samples.selected_locality),
            });
        }
        TemporalReport {
            schema_version: 5,
            interpretation: "direct upward-drift hindsight ceiling, real-arithmetic positive-weight Lightning norm envelope with observed-score validation, and exact cache-order selected-ID locality; neither temporal nor norm observations are an admissible future certificate",
            requested_tokens: self.requested_tokens,
            captured_tokens: self.positions.len(),
            first_position: self.positions.first().copied(),
            last_position: self.positions.last().copied(),
            aggregate_jaccard: MetricSummary::from_values(&aggregate_jaccard),
            aggregate_head_weights: HeadWeightSummary::from_values(&aggregate_head_weights),
            aggregate_one_step_candidates: CandidateSummary::from_samples(&aggregate_one_step),
            aggregate_cadence_oracle_candidates: aggregate_oracle
                .into_iter()
                .map(|(cadence, values)| (cadence, CandidateSummary::from_samples(&values)))
                .collect(),
            aggregate_cadence_charged_work: aggregate_charged
                .into_iter()
                .map(|(cadence, values)| (cadence, CandidateSummary::from_samples(&values)))
                .collect(),
            aggregate_cadence_refresh_endpoints: aggregate_endpoints
                .into_iter()
                .map(|(cadence, values)| (cadence, CandidateSummary::from_samples(&values)))
                .collect(),
            aggregate_cone_envelope: ConeEnvelopeReport::from_samples(&aggregate_cone),
            aggregate_selected_locality: SelectedLocalityReport::from_samples(&aggregate_locality),
            selection_trace_sha256,
            route_trace_sha256,
            final_logits_sha256: None,
            final_causal_digest: None,
            decision_points: self.decision_points,
            layers,
        }
    }
}

fn digest_u32s(domain: &[u8], values: &[u32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn digest_f32s(domain: &[u8], values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn digest_route(
    position: u32,
    layer: u32,
    expert_ids: &[u32],
    weights: &[f32],
    routed_scale: f32,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"qwen-dsv4-temporal-route-point-v1\0");
    hasher.update(position.to_le_bytes());
    hasher.update(layer.to_le_bytes());
    for expert_id in expert_ids {
        hasher.update(expert_id.to_le_bytes());
    }
    for weight in weights {
        hasher.update(weight.to_bits().to_le_bytes());
    }
    hasher.update(routed_scale.to_bits().to_le_bytes());
    format!("{:x}", hasher.finalize())
}

fn ascending_intersection_count(left: &[u32], right: &[u32]) -> usize {
    let (mut left_index, mut right_index, mut count) = (0usize, 0usize, 0usize);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen_llm::deepseek_v4_metal::{
        DeepSeekV4DecisionLayer, DeepSeekV4RankedCsaRow, DeepSeekV4RouteDecision,
    };

    fn transcript(position: u32, scores: Vec<f32>, ids: Vec<u32>) -> DeepSeekV4DecisionTranscript {
        let mut ranked = (0..scores.len()).collect::<Vec<_>>();
        ranked.sort_by(|&left, &right| {
            scores[right]
                .total_cmp(&scores[left])
                .then_with(|| left.cmp(&right))
        });
        let rank_512 = DeepSeekV4RankedCsaRow {
            row_id: ranked[511] as u32,
            score: scores[ranked[511]],
        };
        let rank_513 = DeepSeekV4RankedCsaRow {
            row_id: ranked[512] as u32,
            score: scores[ranked[512]],
        };
        let key_norms = vec![100.0; scores.len()];
        DeepSeekV4DecisionTranscript {
            schema_version: 3,
            position,
            layer_count: 1,
            csa_layer_count: 1,
            layers: vec![DeepSeekV4DecisionLayer {
                layer: 2,
                csa: Some(DeepSeekV4CsaDecision {
                    visible_scores: scores,
                    cache_order_selected_ids: ids,
                    selected_count: 512,
                    selection_status: 0,
                    rank_512_margin: rank_512.score - rank_513.score,
                    rank_512,
                    rank_513,
                }),
                indexer_head_weights: std::iter::once(-0.5)
                    .chain(std::iter::repeat_n(0.25, 63))
                    .collect(),
                indexer_query_norms: vec![1.0; 64],
                indexer_key_norms: key_norms,
                route: DeepSeekV4RouteDecision {
                    expert_ids: vec![0, 1, 2, 3, 4, 5],
                    normalized_scaled_weights: vec![1.0 / 6.0; 6],
                    routed_scale: 1.0,
                },
            }],
        }
    }

    #[test]
    fn temporal_capture_prices_direct_bands_and_refresh_work() {
        let scores0 = (0..520).map(|row| row as f32).collect::<Vec<_>>();
        let ids0 = (8..520).collect::<Vec<u32>>();
        let scores1 = scores0
            .iter()
            .enumerate()
            .map(|(row, &score)| score + (row % 3) as f32 * 0.01)
            .chain([520.0])
            .collect::<Vec<_>>();
        let ids1 = (9..521).collect::<Vec<u32>>();
        let mut capture = TemporalCapture::new(2);
        capture.record(transcript(4_095, scores0, ids0)).unwrap();
        capture.record(transcript(4_096, scores1, ids1)).unwrap();
        let mut report = capture.finish();
        report.attach_final_state("a".repeat(64), Some("b".repeat(64)));
        assert_eq!(report.captured_tokens, 2);
        assert_eq!(report.layers.len(), 1);
        assert_eq!(report.layers[0].comparisons, 1);
        assert!(report.layers[0].jaccard.min > 0.99);
        assert!(report.layers[0].one_step_candidates.distribution.max > 0.0);
        assert_eq!(
            report.layers[0].cadence_oracle_candidates.len(),
            CADENCES.len()
        );
        assert_eq!(
            report.layers[0].cadence_charged_work[&1].weighted_fraction,
            1.0
        );
        assert_eq!(report.layers[0].head_weights.negative_values, 2);
        assert_eq!(report.schema_version, 5);
        assert_eq!(report.aggregate_cone_envelope.bound_violations, 0);
        assert_eq!(
            report.aggregate_cone_envelope.retained_work[&16].weighted_fraction,
            1.0
        );
        assert_eq!(report.layers[0].selected_locality.runs.max, 1.0);
        assert_eq!(report.aggregate_selected_locality.runs.samples, 2);
        assert_eq!(
            report
                .aggregate_selected_locality
                .page_payload_amplification[&8]
                .max,
            513.0 / 512.0
        );
        assert_eq!(report.selection_trace_sha256.len(), 64);
        assert_eq!(report.route_trace_sha256.len(), 64);
        assert_eq!(report.decision_points.len(), 2);
        assert_eq!(report.decision_points[0].selected_ids_sha256.len(), 64);
        assert_eq!(
            report.final_logits_sha256.as_deref(),
            Some("a".repeat(64).as_str())
        );
    }

    #[test]
    fn upward_delta_converts_before_subtraction_and_ignores_decreases() {
        let previous = [1.0f32, 4.0];
        let next = f32::from_bits(1.0f32.to_bits() + 1);
        let current = [next, 2.0];
        assert_eq!(
            max_upward_delta(&previous, &current),
            f64::from(next) - f64::from(1.0f32)
        );
        assert_eq!(max_upward_delta(&[4.0], &[2.0]), 0.0);
        assert!(upper_sum(1.0, 0.0) > 1.0);
    }
}
