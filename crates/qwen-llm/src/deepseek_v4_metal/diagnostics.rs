use serde::{Deserialize, Serialize};

pub const TARGET_POSITION: u32 = 3070;
pub const LAYER_COUNT: usize = 43;
pub const CSA_LAYER_COUNT: usize = 21;
pub const CSA_TOP_K: usize = 512;
pub const ROUTE_TOP_K: usize = 6;
const SCHEMA_VERSION: u32 = 1;

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

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum DeepSeekV4DiagnosticsError {
    #[error("DeepSeek V4 decision capture only supports absolute position 3070, got {0}")]
    UnsupportedPosition(u32),
    #[error("DeepSeek V4 decision capture expected position {expected}, got {actual}")]
    WrongPosition { expected: u32, actual: u32 },
    #[error("DeepSeek V4 decision capture is already armed, complete, or consumed")]
    DuplicateCapture,
    #[error("DeepSeek V4 decision capture is not armed")]
    NotArmed,
    #[error("DeepSeek V4 decision capture is active and cannot {operation}")]
    ActiveCapture { operation: &'static str },
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
        if position != TARGET_POSITION {
            return Err(DeepSeekV4DiagnosticsError::UnsupportedPosition(position));
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
            capture.arm(TARGET_POSITION - 1).unwrap_err(),
            DeepSeekV4DiagnosticsError::UnsupportedPosition(TARGET_POSITION - 1)
        );
        assert_eq!(
            capture.take().unwrap_err(),
            DeepSeekV4DiagnosticsError::NotArmed
        );
        capture.arm(TARGET_POSITION).unwrap();
        assert_eq!(
            capture
                .ensure_no_active_capture("execute packed tokens")
                .unwrap_err(),
            DeepSeekV4DiagnosticsError::ActiveCapture {
                operation: "execute packed tokens"
            }
        );
        assert_eq!(
            capture.begin_forward(TARGET_POSITION + 1).unwrap_err(),
            DeepSeekV4DiagnosticsError::WrongPosition {
                expected: TARGET_POSITION,
                actual: TARGET_POSITION + 1
            }
        );
        assert_eq!(
            capture.arm(TARGET_POSITION).unwrap_err(),
            DeepSeekV4DiagnosticsError::DuplicateCapture
        );
        assert_eq!(
            capture.take().unwrap_err(),
            DeepSeekV4DiagnosticsError::Incomplete {
                captured: 0,
                expected: LAYER_COUNT
            }
        );
        capture.begin_forward(TARGET_POSITION).unwrap();
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
        capture.arm(TARGET_POSITION).unwrap();
        capture.begin_forward(TARGET_POSITION).unwrap();
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
        assert_eq!(value["position"], TARGET_POSITION);
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
}
