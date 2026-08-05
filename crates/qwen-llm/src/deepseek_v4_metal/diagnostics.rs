use serde::{Deserialize, Serialize};

#[cfg(test)]
const TEST_POSITION: u32 = 3070;
pub const LAYER_COUNT: usize = 43;
pub const CSA_LAYER_COUNT: usize = 21;
pub const CSA_TOP_K: usize = 512;
pub const ROUTE_TOP_K: usize = 6;
const SCHEMA_VERSION: u32 = 1;
const FP4_SHADOW_SCHEMA_VERSION: u32 = 1;
pub(crate) const FIRST_SPARSE_CSA_POSITION: u32 = (CSA_TOP_K as u32) * 4 + 3;

fn is_csa_layer(layer: usize) -> bool {
    (2..LAYER_COUNT).contains(&layer) && layer.is_multiple_of(2)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4RankedCsaRow {
    pub row_id: u32,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4CsaDecision {
    pub visible_scores: Vec<f32>,
    pub cache_order_selected_ids: Vec<u32>,
    pub selected_count: u32,
    pub selection_status: i32,
    pub rank_512: DeepSeekV4RankedCsaRow,
    pub rank_513: DeepSeekV4RankedCsaRow,
    pub rank_512_margin: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4RouteDecision {
    pub expert_ids: Vec<u32>,
    /// Final production weights: normalized, then multiplied by routed scale.
    pub normalized_scaled_weights: Vec<f32>,
    pub routed_scale: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4DecisionLayer {
    pub layer: u32,
    pub csa: Option<DeepSeekV4CsaDecision>,
    pub route: DeepSeekV4RouteDecision,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4DecisionTranscript {
    pub schema_version: u32,
    pub position: u32,
    pub layer_count: u32,
    pub csa_layer_count: u32,
    pub layers: Vec<DeepSeekV4DecisionLayer>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepSeekV4Fp4ShadowExecution {
    Packed,
    Singleton,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeepSeekV4Fp4ShadowEligibility {
    Ready,
    InvalidGeometry {
        visible_count: i32,
        expected_visible_count: i32,
        capacity: i32,
    },
    QueryStatus {
        head: u32,
        status: i32,
    },
    KeyStatus {
        row: u32,
        status: i32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4Fp4ShadowLayer {
    pub layer: u32,
    pub position: u32,
    pub eligibility: DeepSeekV4Fp4ShadowEligibility,
    pub query_statuses: Vec<i32>,
    pub visible_key_statuses: Vec<i32>,
    pub authoritative: DeepSeekV4CsaDecision,
    pub shadow: Option<DeepSeekV4CsaDecision>,
    pub shadow_selected_count: i32,
    pub shadow_selection_status: i32,
    pub selected_mask_exact: bool,
    pub max_abs_score_error: Option<f32>,
    pub relative_rms_score_error: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4Fp4ShadowReport {
    pub schema_version: u32,
    pub position: u32,
    pub execution: DeepSeekV4Fp4ShadowExecution,
    pub layers: Vec<DeepSeekV4Fp4ShadowLayer>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV4Fp4CounterfactualStateDigest {
    pub prefix_digest: [u8; 32],
    pub state_digest: [u8; 32],
    pub selection_trace_digest: [u8; 32],
    pub consumed_layer_count: u64,
    pub counterfactual_domain_digest: [u8; 32],
}

pub(crate) struct DeepSeekV4Fp4CounterfactualTrace {
    hasher: blake3::Hasher,
    consumed_layer_count: u64,
}

impl Default for DeepSeekV4Fp4CounterfactualTrace {
    fn default() -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"qwen-dsv4-fp4-selection-trace-v2\0");
        Self {
            hasher,
            consumed_layer_count: 0,
        }
    }
}

impl DeepSeekV4Fp4CounterfactualTrace {
    pub(crate) fn record(
        &mut self,
        execution: DeepSeekV4Fp4ShadowExecution,
        position: u32,
        layer: usize,
        visible_count: i32,
        ids: &[i32],
    ) -> Result<(), DeepSeekV4DiagnosticsError> {
        if !is_csa_layer(layer)
            || visible_count <= CSA_TOP_K as i32
            || ids.len() != CSA_TOP_K
            || ids.windows(2).any(|pair| pair[0] >= pair[1])
            || ids.iter().any(|&id| id < 0 || id >= visible_count)
        {
            return Err(DeepSeekV4DiagnosticsError::Shape(format!(
                "invalid consumed FP4 selection at position {position} layer {layer}"
            )));
        }
        self.hasher.update(&[match execution {
            DeepSeekV4Fp4ShadowExecution::Packed => 0,
            DeepSeekV4Fp4ShadowExecution::Singleton => 1,
        }]);
        self.hasher.update(&position.to_le_bytes());
        self.hasher.update(&(layer as u32).to_le_bytes());
        self.hasher.update(&visible_count.to_le_bytes());
        self.hasher.update(&(ids.len() as u32).to_le_bytes());
        for id in ids {
            self.hasher.update(&id.to_le_bytes());
        }
        self.consumed_layer_count = self.consumed_layer_count.checked_add(1).ok_or_else(|| {
            DeepSeekV4DiagnosticsError::Shape(
                "FP4 counterfactual consumed-layer count overflow".into(),
            )
        })?;
        Ok(())
    }

    pub(crate) fn digest(&self) -> ([u8; 32], u64) {
        (
            *self.hasher.clone().finalize().as_bytes(),
            self.consumed_layer_count,
        )
    }
}

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum DeepSeekV4DiagnosticsError {
    #[error(
        "DeepSeek V4 decision capture requires sparse CSA at or after position {minimum}, got {actual}"
    )]
    SparseSelectionUnavailable { minimum: u32, actual: u32 },
    #[error("DeepSeek V4 decision capture expected position {expected}, got {actual}")]
    WrongPosition { expected: u32, actual: u32 },
    #[error("DeepSeek V4 decision capture is already armed, complete, or consumed")]
    DuplicateCapture,
    #[error("DeepSeek V4 decision capture is not armed")]
    NotArmed,
    #[error("DeepSeek V4 decision capture is active and cannot {operation}")]
    ActiveCapture { operation: &'static str },
    #[error("DeepSeek V4 FP4 shadow lineage must be enabled before arming a report")]
    Fp4ShadowLineageDisabled,
    #[error("DeepSeek V4 FP4 shadow report is already armed or complete")]
    Fp4ShadowDuplicate,
    #[error("DeepSeek V4 FP4 packed report requires exactly one sparse query, got {actual}")]
    Fp4ShadowPackedQueryCount { actual: usize },
    #[error("DeepSeek V4 decision transcript is incomplete: captured {captured}/{expected} layers")]
    Incomplete { captured: usize, expected: usize },
    #[error("DeepSeek V4 decision capture expected layer {expected}, got {actual}")]
    WrongLayer { expected: usize, actual: usize },
    #[error("DeepSeek V4 decision transcript shape mismatch: {0}")]
    Shape(String),
    #[error("DeepSeek V4 decision transcript contains a non-finite {0}")]
    NonFinite(&'static str),
}

#[derive(Default)]
pub(crate) struct DeepSeekV4Fp4ShadowCapture {
    lineage_enabled: bool,
    state: Fp4ShadowCaptureState,
}

#[derive(Default)]
enum Fp4ShadowCaptureState {
    #[default]
    Idle,
    Armed {
        position: u32,
    },
    Capturing {
        position: u32,
        execution: DeepSeekV4Fp4ShadowExecution,
        layers: Vec<DeepSeekV4Fp4ShadowLayer>,
    },
    Complete(DeepSeekV4Fp4ShadowReport),
}

impl DeepSeekV4Fp4ShadowCapture {
    pub(crate) fn enable_lineage(&mut self) {
        self.lineage_enabled = true;
    }

    pub(crate) fn invalidate_lineage(&mut self) {
        self.lineage_enabled = false;
        self.state = Fp4ShadowCaptureState::Idle;
    }

    pub(crate) fn arm(
        &mut self,
        current_position: u32,
        position: u32,
    ) -> Result<(), DeepSeekV4DiagnosticsError> {
        if !self.lineage_enabled {
            return Err(DeepSeekV4DiagnosticsError::Fp4ShadowLineageDisabled);
        }
        if position < FIRST_SPARSE_CSA_POSITION {
            return Err(DeepSeekV4DiagnosticsError::SparseSelectionUnavailable {
                minimum: FIRST_SPARSE_CSA_POSITION,
                actual: position,
            });
        }
        if position < current_position {
            return Err(DeepSeekV4DiagnosticsError::WrongPosition {
                expected: current_position,
                actual: position,
            });
        }
        if !matches!(self.state, Fp4ShadowCaptureState::Idle) {
            return Err(DeepSeekV4DiagnosticsError::Fp4ShadowDuplicate);
        }
        self.state = Fp4ShadowCaptureState::Armed { position };
        Ok(())
    }

    pub(crate) fn begin_singleton(
        &mut self,
        actual_position: u32,
    ) -> Result<(), DeepSeekV4DiagnosticsError> {
        let Fp4ShadowCaptureState::Armed { position } = self.state else {
            return Ok(());
        };
        if actual_position != position {
            return Err(DeepSeekV4DiagnosticsError::WrongPosition {
                expected: position,
                actual: actual_position,
            });
        }
        self.state = Fp4ShadowCaptureState::Capturing {
            position,
            execution: DeepSeekV4Fp4ShadowExecution::Singleton,
            layers: Vec::with_capacity(CSA_LAYER_COUNT),
        };
        Ok(())
    }

    pub(crate) fn begin_packed(
        &mut self,
        final_position: u32,
        sparse_query_count: usize,
    ) -> Result<(), DeepSeekV4DiagnosticsError> {
        let Fp4ShadowCaptureState::Armed { position } = self.state else {
            return Ok(());
        };
        if final_position != position {
            return Err(DeepSeekV4DiagnosticsError::WrongPosition {
                expected: position,
                actual: final_position,
            });
        }
        if sparse_query_count != 1 {
            return Err(DeepSeekV4DiagnosticsError::Fp4ShadowPackedQueryCount {
                actual: sparse_query_count,
            });
        }
        self.state = Fp4ShadowCaptureState::Capturing {
            position,
            execution: DeepSeekV4Fp4ShadowExecution::Packed,
            layers: Vec::with_capacity(CSA_LAYER_COUNT),
        };
        Ok(())
    }

    pub(crate) fn is_capturing(&self) -> bool {
        matches!(self.state, Fp4ShadowCaptureState::Capturing { .. })
    }

    pub(crate) fn ensure_no_active_capture(
        &self,
        operation: &'static str,
    ) -> Result<(), DeepSeekV4DiagnosticsError> {
        if matches!(
            self.state,
            Fp4ShadowCaptureState::Armed { .. } | Fp4ShadowCaptureState::Capturing { .. }
        ) {
            return Err(DeepSeekV4DiagnosticsError::ActiveCapture { operation });
        }
        Ok(())
    }

    pub(crate) fn capture_layer(
        &mut self,
        report: DeepSeekV4Fp4ShadowLayer,
    ) -> Result<(), DeepSeekV4DiagnosticsError> {
        let Fp4ShadowCaptureState::Capturing {
            position, layers, ..
        } = &mut self.state
        else {
            return Ok(());
        };
        let expected_layer = 2 + layers.len() * 2;
        if report.position != *position || report.layer as usize != expected_layer {
            return Err(DeepSeekV4DiagnosticsError::WrongLayer {
                expected: expected_layer,
                actual: report.layer as usize,
            });
        }
        layers.push(report);
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> Result<(), DeepSeekV4DiagnosticsError> {
        if !matches!(self.state, Fp4ShadowCaptureState::Capturing { .. }) {
            return Ok(());
        }
        let Fp4ShadowCaptureState::Capturing {
            position,
            execution,
            layers,
        } = std::mem::take(&mut self.state)
        else {
            unreachable!("capturing state was checked before replacement");
        };
        if layers.len() != CSA_LAYER_COUNT {
            self.state = Fp4ShadowCaptureState::Capturing {
                position,
                execution,
                layers,
            };
            return Err(DeepSeekV4DiagnosticsError::Incomplete {
                captured: match &self.state {
                    Fp4ShadowCaptureState::Capturing { layers, .. } => layers.len(),
                    _ => 0,
                },
                expected: CSA_LAYER_COUNT,
            });
        }
        self.state = Fp4ShadowCaptureState::Complete(DeepSeekV4Fp4ShadowReport {
            schema_version: FP4_SHADOW_SCHEMA_VERSION,
            position,
            execution,
            layers,
        });
        Ok(())
    }

    pub(crate) fn take(&mut self) -> Result<DeepSeekV4Fp4ShadowReport, DeepSeekV4DiagnosticsError> {
        match std::mem::take(&mut self.state) {
            Fp4ShadowCaptureState::Complete(report) => Ok(report),
            state => {
                self.state = state;
                Err(DeepSeekV4DiagnosticsError::NotArmed)
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct DeepSeekV4DecisionCapture {
    state: CaptureState,
}

#[derive(Default)]
enum CaptureState {
    #[default]
    Idle,
    Armed {
        position: u32,
    },
    Capturing {
        position: u32,
        layers: Vec<DeepSeekV4DecisionLayer>,
    },
    Complete(DeepSeekV4DecisionTranscript),
    Taken,
}

impl DeepSeekV4DecisionCapture {
    pub(crate) fn is_capturing(&self) -> bool {
        matches!(self.state, CaptureState::Capturing { .. })
    }

    pub(crate) fn arm(&mut self, position: u32) -> Result<(), DeepSeekV4DiagnosticsError> {
        if position < FIRST_SPARSE_CSA_POSITION {
            return Err(DeepSeekV4DiagnosticsError::SparseSelectionUnavailable {
                minimum: FIRST_SPARSE_CSA_POSITION,
                actual: position,
            });
        }
        if !matches!(self.state, CaptureState::Idle) {
            return Err(DeepSeekV4DiagnosticsError::DuplicateCapture);
        }
        self.state = CaptureState::Armed { position };
        Ok(())
    }

    pub(crate) fn ensure_no_active_capture(
        &self,
        operation: &'static str,
    ) -> Result<(), DeepSeekV4DiagnosticsError> {
        if matches!(
            self.state,
            CaptureState::Armed { .. } | CaptureState::Capturing { .. }
        ) {
            return Err(DeepSeekV4DiagnosticsError::ActiveCapture { operation });
        }
        Ok(())
    }

    pub(crate) fn begin_forward(
        &mut self,
        actual_position: u32,
    ) -> Result<(), DeepSeekV4DiagnosticsError> {
        let CaptureState::Armed { position } = self.state else {
            return Ok(());
        };
        if actual_position != position {
            return Err(DeepSeekV4DiagnosticsError::WrongPosition {
                expected: position,
                actual: actual_position,
            });
        }
        self.state = CaptureState::Capturing {
            position,
            layers: Vec::with_capacity(LAYER_COUNT),
        };
        Ok(())
    }

    pub(crate) fn capture_layer(
        &mut self,
        layer: usize,
        csa: Option<DeepSeekV4CsaDecision>,
        route: DeepSeekV4RouteDecision,
    ) -> Result<(), DeepSeekV4DiagnosticsError> {
        let CaptureState::Capturing { layers, .. } = &mut self.state else {
            return Ok(());
        };
        if layer != layers.len() {
            return Err(DeepSeekV4DiagnosticsError::WrongLayer {
                expected: layers.len(),
                actual: layer,
            });
        }
        if csa.is_some() != is_csa_layer(layer) {
            return Err(DeepSeekV4DiagnosticsError::Shape(format!(
                "layer {layer} CSA decision presence does not match Flash-0731 geometry"
            )));
        }
        validate_csa(csa.as_ref())?;
        validate_route(&route)?;
        layers.push(DeepSeekV4DecisionLayer {
            layer: layer as u32,
            csa,
            route,
        });
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> Result<(), DeepSeekV4DiagnosticsError> {
        if !matches!(self.state, CaptureState::Capturing { .. }) {
            return Ok(());
        }
        let CaptureState::Capturing { position, layers } =
            std::mem::replace(&mut self.state, CaptureState::Taken)
        else {
            unreachable!("capturing state was checked before replacement");
        };
        if layers.len() != LAYER_COUNT {
            self.state = CaptureState::Capturing { position, layers };
            return Err(self.incomplete_error());
        }
        let csa_count = layers.iter().filter(|layer| layer.csa.is_some()).count();
        if csa_count != CSA_LAYER_COUNT {
            self.state = CaptureState::Capturing { position, layers };
            return Err(DeepSeekV4DiagnosticsError::Shape(format!(
                "captured {csa_count} CSA layers, expected {CSA_LAYER_COUNT}"
            )));
        }
        self.state = CaptureState::Complete(DeepSeekV4DecisionTranscript {
            schema_version: SCHEMA_VERSION,
            position,
            layer_count: LAYER_COUNT as u32,
            csa_layer_count: CSA_LAYER_COUNT as u32,
            layers,
        });
        Ok(())
    }

    pub(crate) fn take(
        &mut self,
    ) -> Result<DeepSeekV4DecisionTranscript, DeepSeekV4DiagnosticsError> {
        match std::mem::replace(&mut self.state, CaptureState::Taken) {
            CaptureState::Complete(transcript) => Ok(transcript),
            state @ CaptureState::Capturing { .. } => {
                self.state = state;
                Err(self.incomplete_error())
            }
            state @ CaptureState::Armed { .. } => {
                self.state = state;
                Err(self.incomplete_error())
            }
            CaptureState::Idle => {
                self.state = CaptureState::Idle;
                Err(DeepSeekV4DiagnosticsError::NotArmed)
            }
            CaptureState::Taken => Err(DeepSeekV4DiagnosticsError::DuplicateCapture),
        }
    }

    fn incomplete_error(&self) -> DeepSeekV4DiagnosticsError {
        let captured = match &self.state {
            CaptureState::Capturing { layers, .. } => layers.len(),
            _ => 0,
        };
        DeepSeekV4DiagnosticsError::Incomplete {
            captured,
            expected: LAYER_COUNT,
        }
    }
}

pub(crate) fn build_csa_decision(
    scores: Vec<f32>,
    visible_count: usize,
    selected_ids: Vec<i32>,
    selected_count: Vec<i32>,
    status: Vec<i32>,
) -> Result<DeepSeekV4CsaDecision, DeepSeekV4DiagnosticsError> {
    if visible_count <= CSA_TOP_K || visible_count > scores.len() {
        return Err(DeepSeekV4DiagnosticsError::Shape(format!(
            "visible CSA count {visible_count} is outside 513..={} scores",
            scores.len()
        )));
    }
    if selected_ids.len() != CSA_TOP_K || selected_count.len() != 1 || status.len() != 1 {
        return Err(DeepSeekV4DiagnosticsError::Shape(format!(
            "CSA IDs/count/status lengths are {}/{}/{}, expected {CSA_TOP_K}/1/1",
            selected_ids.len(),
            selected_count.len(),
            status.len()
        )));
    }
    if selected_count[0] != CSA_TOP_K as i32 || status[0] != 0 {
        return Err(DeepSeekV4DiagnosticsError::Shape(format!(
            "CSA selection count/status are {}/{}, expected {CSA_TOP_K}/0",
            selected_count[0], status[0]
        )));
    }
    let visible_scores = scores[..visible_count].to_vec();
    if visible_scores.iter().any(|score| !score.is_finite()) {
        return Err(DeepSeekV4DiagnosticsError::NonFinite("CSA score"));
    }
    let cache_order_selected_ids = selected_ids
        .into_iter()
        .map(|id| {
            u32::try_from(id)
                .map_err(|_| DeepSeekV4DiagnosticsError::Shape("negative CSA row ID".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if cache_order_selected_ids
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
        || cache_order_selected_ids
            .iter()
            .any(|&id| id as usize >= visible_count)
    {
        return Err(DeepSeekV4DiagnosticsError::Shape(
            "CSA selected IDs are not unique ascending visible rows".into(),
        ));
    }
    let ranked = stable_descending_ranks(&visible_scores)?;
    let mut expected_cache_order = ranked[..CSA_TOP_K]
        .iter()
        .map(|row| row.row_id)
        .collect::<Vec<_>>();
    expected_cache_order.sort_unstable();
    if cache_order_selected_ids != expected_cache_order {
        return Err(DeepSeekV4DiagnosticsError::Shape(
            "CSA selected IDs differ from stable top-512 scores".into(),
        ));
    }
    let rank_512 = ranked[CSA_TOP_K - 1].clone();
    let rank_513 = ranked[CSA_TOP_K].clone();
    let rank_512_margin = rank_512.score - rank_513.score;
    if !rank_512_margin.is_finite() {
        return Err(DeepSeekV4DiagnosticsError::NonFinite("CSA rank margin"));
    }
    if rank_512_margin < 0.0 {
        return Err(DeepSeekV4DiagnosticsError::Shape(
            "CSA rank-512 margin is negative".into(),
        ));
    }
    Ok(DeepSeekV4CsaDecision {
        visible_scores,
        cache_order_selected_ids,
        selected_count: selected_count[0] as u32,
        selection_status: status[0],
        rank_512,
        rank_513,
        rank_512_margin,
    })
}

pub(crate) struct DeepSeekV4Fp4ShadowLayerInputs {
    pub layer: usize,
    pub position: u32,
    pub visible_count: usize,
    pub capacity_rows: usize,
    pub query_statuses: Vec<i32>,
    pub visible_key_statuses: Vec<i32>,
    pub eligibility_record: Vec<i32>,
    pub authoritative_scores: Vec<f32>,
    pub authoritative_mask: Vec<i32>,
    pub authoritative_ids: Vec<i32>,
    pub authoritative_count: Vec<i32>,
    pub authoritative_status: Vec<i32>,
    pub shadow_scores: Vec<f32>,
    pub shadow_mask: Vec<i32>,
    pub shadow_ids: Vec<i32>,
    pub shadow_count: Vec<i32>,
    pub shadow_status: Vec<i32>,
}

pub(crate) fn build_fp4_shadow_layer(
    inputs: DeepSeekV4Fp4ShadowLayerInputs,
) -> Result<DeepSeekV4Fp4ShadowLayer, DeepSeekV4DiagnosticsError> {
    if inputs.query_statuses.len() != 64
        || inputs.visible_key_statuses.len() != inputs.visible_count
        || inputs.eligibility_record.len() != 3
        || inputs.authoritative_scores.len() != inputs.capacity_rows
        || inputs.authoritative_mask.len() != inputs.capacity_rows
        || inputs.shadow_scores.len() != inputs.capacity_rows
        || inputs.shadow_mask.len() != inputs.capacity_rows
        || inputs.shadow_count.len() != 1
        || inputs.shadow_status.len() != 1
    {
        return Err(DeepSeekV4DiagnosticsError::Shape(format!(
            "FP4 shadow layer {} has inconsistent report geometry",
            inputs.layer
        )));
    }
    let eligibility = match inputs.eligibility_record.as_slice() {
        [0, -1, 0] => DeepSeekV4Fp4ShadowEligibility::Ready,
        [1, visible_count, expected_visible_count] => {
            DeepSeekV4Fp4ShadowEligibility::InvalidGeometry {
                visible_count: *visible_count,
                expected_visible_count: *expected_visible_count,
                capacity: i32::try_from(inputs.capacity_rows).map_err(|_| {
                    DeepSeekV4DiagnosticsError::Shape(
                        "FP4 shadow capacity exceeds diagnostic i32 geometry".into(),
                    )
                })?,
            }
        }
        [2, head, status] => DeepSeekV4Fp4ShadowEligibility::QueryStatus {
            head: u32::try_from(*head).map_err(|_| {
                DeepSeekV4DiagnosticsError::Shape("negative FP4 query failure index".into())
            })?,
            status: *status,
        },
        [3, row, status] => DeepSeekV4Fp4ShadowEligibility::KeyStatus {
            row: u32::try_from(*row).map_err(|_| {
                DeepSeekV4DiagnosticsError::Shape("negative FP4 key failure index".into())
            })?,
            status: *status,
        },
        record => {
            return Err(DeepSeekV4DiagnosticsError::Shape(format!(
                "invalid FP4 eligibility record {record:?}"
            )));
        }
    };
    match eligibility {
        DeepSeekV4Fp4ShadowEligibility::Ready => {
            if inputs.query_statuses.iter().any(|&status| status != 0)
                || inputs
                    .visible_key_statuses
                    .iter()
                    .any(|&status| status != 0)
            {
                return Err(DeepSeekV4DiagnosticsError::Shape(
                    "READY FP4 eligibility contains a non-ready operand".into(),
                ));
            }
        }
        DeepSeekV4Fp4ShadowEligibility::QueryStatus { head, status } => {
            if inputs.query_statuses.get(head as usize).copied() != Some(status) || status == 0 {
                return Err(DeepSeekV4DiagnosticsError::Shape(
                    "FP4 query eligibility record does not match query statuses".into(),
                ));
            }
        }
        DeepSeekV4Fp4ShadowEligibility::KeyStatus { row, status } => {
            if inputs.visible_key_statuses.get(row as usize).copied() != Some(status) || status == 0
            {
                return Err(DeepSeekV4DiagnosticsError::Shape(
                    "FP4 key eligibility record does not match key statuses".into(),
                ));
            }
        }
        DeepSeekV4Fp4ShadowEligibility::InvalidGeometry { .. } => {}
    }

    let authoritative = build_csa_decision(
        inputs.authoritative_scores.clone(),
        inputs.visible_count,
        inputs.authoritative_ids,
        inputs.authoritative_count,
        inputs.authoritative_status,
    )?;
    let (shadow, selected_mask_exact, max_abs_score_error, relative_rms_score_error) =
        if eligibility == DeepSeekV4Fp4ShadowEligibility::Ready {
            let shadow = build_csa_decision(
                inputs.shadow_scores.clone(),
                inputs.visible_count,
                inputs.shadow_ids,
                inputs.shadow_count.clone(),
                inputs.shadow_status.clone(),
            )?;
            let mut squared_error = 0.0f64;
            let mut reference_norm = 0.0f64;
            let mut max_abs = 0.0f32;
            for (&actual, &reference) in inputs.shadow_scores[..inputs.visible_count]
                .iter()
                .zip(&inputs.authoritative_scores[..inputs.visible_count])
            {
                if !actual.is_finite() || !reference.is_finite() {
                    return Err(DeepSeekV4DiagnosticsError::NonFinite(
                        "FP4 shadow score differential",
                    ));
                }
                let error = actual - reference;
                squared_error += f64::from(error).powi(2);
                reference_norm += f64::from(reference).powi(2);
                max_abs = max_abs.max(error.abs());
            }
            let relative_rms = if reference_norm == 0.0 {
                if squared_error == 0.0 { 0.0 } else { f64::MAX }
            } else {
                (squared_error / reference_norm).sqrt()
            };
            (
                Some(shadow),
                inputs.shadow_mask == inputs.authoritative_mask,
                Some(max_abs),
                Some(relative_rms),
            )
        } else {
            if inputs.shadow_count[0] != 0
                || inputs.shadow_status[0] != 1
                || inputs.shadow_mask.iter().any(|&value| value != 0)
            {
                return Err(DeepSeekV4DiagnosticsError::Shape(
                    "ineligible FP4 shadow query did not fail closed".into(),
                ));
            }
            (None, false, None, None)
        };

    Ok(DeepSeekV4Fp4ShadowLayer {
        layer: inputs.layer as u32,
        position: inputs.position,
        eligibility,
        query_statuses: inputs.query_statuses,
        visible_key_statuses: inputs.visible_key_statuses,
        authoritative,
        shadow,
        shadow_selected_count: inputs.shadow_count[0],
        shadow_selection_status: inputs.shadow_status[0],
        selected_mask_exact,
        max_abs_score_error,
        relative_rms_score_error,
    })
}

fn stable_descending_ranks(
    scores: &[f32],
) -> Result<Vec<DeepSeekV4RankedCsaRow>, DeepSeekV4DiagnosticsError> {
    if scores.iter().any(|score| !score.is_finite()) {
        return Err(DeepSeekV4DiagnosticsError::NonFinite("CSA score"));
    }
    let mut rows = scores
        .iter()
        .enumerate()
        .map(|(row_id, &score)| DeepSeekV4RankedCsaRow {
            row_id: row_id as u32,
            score,
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .expect("finite CSA scores were validated")
            .then_with(|| left.row_id.cmp(&right.row_id))
    });
    Ok(rows)
}

pub(crate) fn build_route_decision(
    expert_ids: Vec<i32>,
    normalized_scaled_weights: Vec<f32>,
    expert_count: usize,
    routed_scale: f32,
) -> Result<DeepSeekV4RouteDecision, DeepSeekV4DiagnosticsError> {
    let expert_ids = expert_ids
        .into_iter()
        .map(|id| {
            u32::try_from(id)
                .map_err(|_| DeepSeekV4DiagnosticsError::Shape("negative routed expert ID".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let decision = DeepSeekV4RouteDecision {
        expert_ids,
        normalized_scaled_weights,
        routed_scale,
    };
    if decision
        .expert_ids
        .iter()
        .any(|&id| id as usize >= expert_count)
    {
        return Err(DeepSeekV4DiagnosticsError::Shape(
            "routed expert ID exceeds expert count".into(),
        ));
    }
    let mut unique_ids = decision.expert_ids.clone();
    unique_ids.sort_unstable();
    unique_ids.dedup();
    if unique_ids.len() != ROUTE_TOP_K {
        return Err(DeepSeekV4DiagnosticsError::Shape(
            "routed expert IDs are not unique".into(),
        ));
    }
    validate_route(&decision)?;
    Ok(decision)
}

fn validate_csa(csa: Option<&DeepSeekV4CsaDecision>) -> Result<(), DeepSeekV4DiagnosticsError> {
    if let Some(csa) = csa {
        if csa.visible_scores.len() <= CSA_TOP_K
            || csa.cache_order_selected_ids.len() != CSA_TOP_K
            || csa.selected_count != CSA_TOP_K as u32
            || csa.selection_status != 0
        {
            return Err(DeepSeekV4DiagnosticsError::Shape(
                "invalid CSA decision geometry".into(),
            ));
        }
        if csa.visible_scores.iter().any(|score| !score.is_finite())
            || !csa.rank_512.score.is_finite()
            || !csa.rank_513.score.is_finite()
            || !csa.rank_512_margin.is_finite()
        {
            return Err(DeepSeekV4DiagnosticsError::NonFinite("CSA decision value"));
        }
    }
    Ok(())
}

fn validate_route(route: &DeepSeekV4RouteDecision) -> Result<(), DeepSeekV4DiagnosticsError> {
    if route.expert_ids.len() != ROUTE_TOP_K || route.normalized_scaled_weights.len() != ROUTE_TOP_K
    {
        return Err(DeepSeekV4DiagnosticsError::Shape(format!(
            "route IDs/weights lengths are {}/{}, expected {ROUTE_TOP_K}/{ROUTE_TOP_K}",
            route.expert_ids.len(),
            route.normalized_scaled_weights.len()
        )));
    }
    if !route.routed_scale.is_finite()
        || route
            .normalized_scaled_weights
            .iter()
            .any(|weight| !weight.is_finite())
    {
        return Err(DeepSeekV4DiagnosticsError::NonFinite(
            "route weight or scale",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route() -> DeepSeekV4RouteDecision {
        DeepSeekV4RouteDecision {
            expert_ids: (0..ROUTE_TOP_K as u32).collect(),
            normalized_scaled_weights: vec![1.0 / ROUTE_TOP_K as f32; ROUTE_TOP_K],
            routed_scale: 1.0,
        }
    }

    fn csa() -> DeepSeekV4CsaDecision {
        build_csa_decision(
            (0..513).map(|row| row as f32).collect(),
            513,
            (1..513).collect(),
            vec![512],
            vec![0],
        )
        .unwrap()
    }

    fn fp4_layer(layer: usize, position: u32) -> DeepSeekV4Fp4ShadowLayer {
        let scores = (0..513).map(|row| row as f32).collect::<Vec<_>>();
        let mut mask = vec![1; 513];
        mask[0] = 0;
        build_fp4_shadow_layer(DeepSeekV4Fp4ShadowLayerInputs {
            layer,
            position,
            visible_count: 513,
            capacity_rows: 513,
            query_statuses: vec![0; 64],
            visible_key_statuses: vec![0; 513],
            eligibility_record: vec![0, -1, 0],
            authoritative_scores: scores.clone(),
            authoritative_mask: mask.clone(),
            authoritative_ids: (1..513).collect(),
            authoritative_count: vec![512],
            authoritative_status: vec![0],
            shadow_scores: scores,
            shadow_mask: mask,
            shadow_ids: (1..513).collect(),
            shadow_count: vec![512],
            shadow_status: vec![0],
        })
        .unwrap()
    }

    #[test]
    fn ranking_is_descending_and_breaks_ties_by_lower_row() {
        let mut scores = (0..513).map(|row| row as f32).collect::<Vec<_>>();
        scores[0] = -0.0;
        scores[1] = 0.0;
        scores[2] = 10_000.0;
        scores[3] = 10_000.0;
        let ranked = stable_descending_ranks(&scores).unwrap();
        assert_eq!((ranked[0].row_id, ranked[1].row_id), (2, 3));
        assert!(
            ranked.iter().position(|row| row.row_id == 0).unwrap()
                < ranked.iter().position(|row| row.row_id == 1).unwrap()
        );
        assert!(ranked.windows(2).all(|rows| rows[0].score >= rows[1].score));
    }

    #[test]
    fn arming_take_and_incomplete_states_fail_closed() {
        let mut capture = DeepSeekV4DecisionCapture::default();
        capture.finish().unwrap();
        capture
            .ensure_no_active_capture("restore a snapshot")
            .unwrap();
        assert_eq!(
            capture.arm(FIRST_SPARSE_CSA_POSITION - 1).unwrap_err(),
            DeepSeekV4DiagnosticsError::SparseSelectionUnavailable {
                minimum: FIRST_SPARSE_CSA_POSITION,
                actual: FIRST_SPARSE_CSA_POSITION - 1,
            }
        );
        capture
            .ensure_no_active_capture("restore a snapshot")
            .unwrap();
        assert_eq!(
            capture.take().unwrap_err(),
            DeepSeekV4DiagnosticsError::NotArmed
        );
        capture.arm(FIRST_SPARSE_CSA_POSITION).unwrap();
        assert_eq!(
            capture
                .ensure_no_active_capture("execute packed tokens")
                .unwrap_err(),
            DeepSeekV4DiagnosticsError::ActiveCapture {
                operation: "execute packed tokens"
            }
        );
        assert_eq!(
            capture
                .begin_forward(FIRST_SPARSE_CSA_POSITION + 1)
                .unwrap_err(),
            DeepSeekV4DiagnosticsError::WrongPosition {
                expected: FIRST_SPARSE_CSA_POSITION,
                actual: FIRST_SPARSE_CSA_POSITION + 1
            }
        );
        assert_eq!(
            capture.arm(FIRST_SPARSE_CSA_POSITION).unwrap_err(),
            DeepSeekV4DiagnosticsError::DuplicateCapture
        );
        assert_eq!(
            capture.take().unwrap_err(),
            DeepSeekV4DiagnosticsError::Incomplete {
                captured: 0,
                expected: LAYER_COUNT
            }
        );
        capture.begin_forward(FIRST_SPARSE_CSA_POSITION).unwrap();
        assert_eq!(
            capture
                .ensure_no_active_capture("restore a snapshot")
                .unwrap_err(),
            DeepSeekV4DiagnosticsError::ActiveCapture {
                operation: "restore a snapshot"
            }
        );
        capture.capture_layer(0, None, route()).unwrap();
        assert_eq!(
            capture.take().unwrap_err(),
            DeepSeekV4DiagnosticsError::Incomplete {
                captured: 1,
                expected: LAYER_COUNT
            }
        );
    }

    #[test]
    fn schema_geometry_is_serializable_and_requires_all_layers() {
        let mut capture = DeepSeekV4DecisionCapture::default();
        capture.arm(TEST_POSITION).unwrap();
        capture.begin_forward(TEST_POSITION).unwrap();
        for layer in 0..LAYER_COUNT {
            let csa = is_csa_layer(layer).then(csa);
            capture.capture_layer(layer, csa, route()).unwrap();
        }
        capture.finish().unwrap();
        capture
            .ensure_no_active_capture("restore a snapshot")
            .unwrap();
        let transcript = capture.take().unwrap();
        capture
            .ensure_no_active_capture("execute packed tokens")
            .unwrap();
        assert_eq!(transcript.layers.len(), LAYER_COUNT);
        assert_eq!(
            transcript
                .layers
                .iter()
                .filter(|layer| layer.csa.is_some())
                .count(),
            CSA_LAYER_COUNT
        );
        let value = serde_json::to_value(transcript).unwrap();
        assert_eq!(value["position"], TEST_POSITION);
        assert_eq!(value["layer_count"], LAYER_COUNT as u32);
        assert_eq!(
            capture.take().unwrap_err(),
            DeepSeekV4DiagnosticsError::DuplicateCapture
        );
    }

    #[test]
    fn csa_boundary_ranks_and_margin_follow_cpu_semantics() {
        let scores = (0..513).map(|row| row as f32).collect::<Vec<_>>();
        let selected_ids = (1..513).collect::<Vec<_>>();
        let decision = build_csa_decision(scores, 513, selected_ids, vec![512], vec![0]).unwrap();
        assert_eq!(decision.rank_512.row_id, 1);
        assert_eq!(decision.rank_513.row_id, 0);
        assert_eq!(decision.rank_512_margin, 1.0);
    }

    #[test]
    fn fp4_shadow_capture_is_repeatable_and_restricts_packed_geometry() {
        let mut capture = DeepSeekV4Fp4ShadowCapture::default();
        assert_eq!(
            capture.arm(2_048, 2_051).unwrap_err(),
            DeepSeekV4DiagnosticsError::Fp4ShadowLineageDisabled
        );
        capture.enable_lineage();
        capture.arm(2_048, 2_051).unwrap();
        assert_eq!(
            capture.begin_packed(2_051, 2).unwrap_err(),
            DeepSeekV4DiagnosticsError::Fp4ShadowPackedQueryCount { actual: 2 }
        );
        capture.begin_packed(2_051, 1).unwrap();
        for index in 0..CSA_LAYER_COUNT {
            capture
                .capture_layer(fp4_layer(2 + index * 2, 2_051))
                .unwrap();
        }
        capture.finish().unwrap();
        let packed = capture.take().unwrap();
        assert_eq!(packed.position, 2_051);
        assert_eq!(packed.execution, DeepSeekV4Fp4ShadowExecution::Packed);
        assert_eq!(packed.layers.len(), CSA_LAYER_COUNT);
        assert!(packed.layers.iter().all(|layer| {
            layer.eligibility == DeepSeekV4Fp4ShadowEligibility::Ready
                && layer.selected_mask_exact
                && layer.max_abs_score_error == Some(0.0)
                && layer.relative_rms_score_error == Some(0.0)
        }));

        capture.arm(2_052, 2_052).unwrap();
        capture.begin_singleton(2_052).unwrap();
        for index in 0..CSA_LAYER_COUNT {
            capture
                .capture_layer(fp4_layer(2 + index * 2, 2_052))
                .unwrap();
        }
        capture.finish().unwrap();
        let singleton = capture.take().unwrap();
        assert_eq!(singleton.execution, DeepSeekV4Fp4ShadowExecution::Singleton);
        assert_eq!(singleton.layers.len(), CSA_LAYER_COUNT);

        capture.arm(2_053, 2_053).unwrap();
        capture.begin_singleton(2_053).unwrap();
        for index in 0..CSA_LAYER_COUNT {
            let mut layer = fp4_layer(2 + index * 2, 2_053);
            if index == 0 {
                layer.eligibility = DeepSeekV4Fp4ShadowEligibility::KeyStatus {
                    row: 512,
                    status: i32::MIN,
                };
                layer.shadow = None;
                layer.shadow_selected_count = 0;
                layer.shadow_selection_status = 1;
                layer.selected_mask_exact = false;
                layer.max_abs_score_error = None;
                layer.relative_rms_score_error = None;
            }
            capture.capture_layer(layer).unwrap();
        }
        capture.finish().unwrap();
        let ineligible = capture.take().unwrap();
        assert!(matches!(
            ineligible.layers[0].eligibility,
            DeepSeekV4Fp4ShadowEligibility::KeyStatus { row: 512, status }
                if status == i32::MIN
        ));
        assert!(ineligible.layers[0].shadow.is_none());

        capture.arm(2_054, 2_054).unwrap();
        capture.begin_singleton(2_054).unwrap();
        for index in 0..CSA_LAYER_COUNT {
            capture
                .capture_layer(fp4_layer(2 + index * 2, 2_054))
                .unwrap();
        }
        capture.finish().unwrap();
        assert!(
            capture
                .take()
                .unwrap()
                .layers
                .iter()
                .all(|layer| layer.eligibility == DeepSeekV4Fp4ShadowEligibility::Ready)
        );
        capture.invalidate_lineage();
        assert_eq!(
            capture.arm(2_055, 2_055).unwrap_err(),
            DeepSeekV4DiagnosticsError::Fp4ShadowLineageDisabled
        );
    }

    #[test]
    fn fp4_shadow_report_preserves_ineligible_fail_closed_evidence() {
        let scores = (0..513).map(|row| row as f32).collect::<Vec<_>>();
        let mut authoritative_mask = vec![1; 513];
        authoritative_mask[0] = 0;
        let mut key_statuses = vec![0; 513];
        key_statuses[512] = i32::MIN;
        let report = build_fp4_shadow_layer(DeepSeekV4Fp4ShadowLayerInputs {
            layer: 2,
            position: 2_051,
            visible_count: 513,
            capacity_rows: 513,
            query_statuses: vec![0; 64],
            visible_key_statuses: key_statuses,
            eligibility_record: vec![3, 512, i32::MIN],
            authoritative_scores: scores,
            authoritative_mask,
            authoritative_ids: (1..513).collect(),
            authoritative_count: vec![512],
            authoritative_status: vec![0],
            shadow_scores: vec![f32::NEG_INFINITY; 513],
            shadow_mask: vec![0; 513],
            shadow_ids: vec![-1; 512],
            shadow_count: vec![0],
            shadow_status: vec![1],
        })
        .unwrap();
        assert_eq!(
            report.eligibility,
            DeepSeekV4Fp4ShadowEligibility::KeyStatus {
                row: 512,
                status: i32::MIN,
            }
        );
        assert!(report.shadow.is_none());
        assert_eq!(report.shadow_selected_count, 0);
        assert_eq!(report.shadow_selection_status, 1);
    }

    #[test]
    fn fp4_counterfactual_trace_binds_position_layer_visibility_and_ids() {
        let ids = (0..512).collect::<Vec<_>>();
        let mut first = DeepSeekV4Fp4CounterfactualTrace::default();
        first
            .record(DeepSeekV4Fp4ShadowExecution::Packed, 2_051, 2, 513, &ids)
            .unwrap();
        first
            .record(DeepSeekV4Fp4ShadowExecution::Packed, 2_051, 4, 513, &ids)
            .unwrap();
        let mut repeat = DeepSeekV4Fp4CounterfactualTrace::default();
        repeat
            .record(DeepSeekV4Fp4ShadowExecution::Packed, 2_051, 2, 513, &ids)
            .unwrap();
        repeat
            .record(DeepSeekV4Fp4ShadowExecution::Packed, 2_051, 4, 513, &ids)
            .unwrap();
        assert_eq!(first.digest(), repeat.digest());
        assert_eq!(first.digest().1, 2);

        let mut changed = DeepSeekV4Fp4CounterfactualTrace::default();
        changed
            .record(DeepSeekV4Fp4ShadowExecution::Packed, 2_052, 2, 513, &ids)
            .unwrap();
        changed
            .record(DeepSeekV4Fp4ShadowExecution::Packed, 2_051, 4, 513, &ids)
            .unwrap();
        assert_ne!(first.digest().0, changed.digest().0);

        let mut changed_execution = DeepSeekV4Fp4CounterfactualTrace::default();
        changed_execution
            .record(DeepSeekV4Fp4ShadowExecution::Singleton, 2_051, 2, 513, &ids)
            .unwrap();
        changed_execution
            .record(DeepSeekV4Fp4ShadowExecution::Packed, 2_051, 4, 513, &ids)
            .unwrap();
        assert_ne!(first.digest().0, changed_execution.digest().0);

        let mut invalid = DeepSeekV4Fp4CounterfactualTrace::default();
        let mut duplicate = ids;
        duplicate[511] = 510;
        assert!(
            invalid
                .record(
                    DeepSeekV4Fp4ShadowExecution::Packed,
                    2_051,
                    2,
                    513,
                    &duplicate,
                )
                .is_err()
        );
    }
}
