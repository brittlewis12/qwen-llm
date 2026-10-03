//! Transport-neutral intervention wire forms and direction normalization.

use crate::lens_scope::Scope;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub(crate) mod coefficient;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Normalization {
    AsStored,
    UnitL2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LensRowDirectionDefinition {
    pub(crate) id: String,
    pub(crate) lens: String,
    pub(crate) row: DirectionRow,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) target_covector: Option<DirectionTargetCovector>,
    pub(crate) normalization: Normalization,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DirectionTargetCovector {
    DeployedLogitNumerator,
    RawLmHead,
    RawLmHeadOrthogonalToDeployedLogitNumerator,
}

impl LensRowDirectionDefinition {
    pub(crate) fn effective_target_covector(&self) -> DirectionTargetCovector {
        self.target_covector
            .unwrap_or(DirectionTargetCovector::DeployedLogitNumerator)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum DirectionRow {
    TokenId { token_id: i32 },
    TemplateRowId { template_row_id: usize },
    Label { label: String },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OperationDefinition {
    pub(crate) id: String,
    pub(crate) scope: Scope,
    pub(crate) action: Action,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Action {
    FixedAdd {
        direction: String,
        #[serde(deserialize_with = "coefficient::deserialize_action")]
        coefficient: f32,
    },
    ResidualL2Fraction {
        direction: String,
        #[serde(deserialize_with = "coefficient::deserialize_action")]
        coefficient: f32,
    },
    ProjectionAblate {
        direction: String,
        #[serde(deserialize_with = "coefficient::deserialize_action")]
        coefficient: f32,
    },
    SourceToTarget {
        source: String,
        target: String,
        #[serde(deserialize_with = "coefficient::deserialize_action")]
        coefficient: f32,
    },
    CoordinateSwap {
        source: String,
        target: String,
        #[serde(deserialize_with = "coefficient::deserialize_action")]
        coefficient: f32,
    },
}

impl Action {
    pub(crate) fn coefficient(&self) -> f32 {
        match self {
            Self::FixedAdd { coefficient, .. }
            | Self::ResidualL2Fraction { coefficient, .. }
            | Self::ProjectionAblate { coefficient, .. }
            | Self::SourceToTarget { coefficient, .. }
            | Self::CoordinateSwap { coefficient, .. } => *coefficient,
        }
    }

    pub(crate) fn set_coefficient(&mut self, value: f32) {
        match self {
            Self::FixedAdd { coefficient, .. }
            | Self::ResidualL2Fraction { coefficient, .. }
            | Self::ProjectionAblate { coefficient, .. }
            | Self::SourceToTarget { coefficient, .. }
            | Self::CoordinateSwap { coefficient, .. } => *coefficient = value,
        }
    }

    pub(crate) fn direction_ids<'a>(&'a self) -> impl Iterator<Item = &'a str> + 'a {
        match self {
            Self::FixedAdd { direction, .. }
            | Self::ResidualL2Fraction { direction, .. }
            | Self::ProjectionAblate { direction, .. } => {
                EitherDirectionIds::One(std::iter::once(direction.as_str()))
            }
            Self::SourceToTarget { source, target, .. }
            | Self::CoordinateSwap { source, target, .. } => {
                EitherDirectionIds::Two([source.as_str(), target.as_str()].into_iter())
            }
        }
    }
}

pub(crate) enum EitherDirectionIds<'a> {
    One(std::iter::Once<&'a str>),
    Two(std::array::IntoIter<&'a str, 2>),
}

impl<'a> Iterator for EitherDirectionIds<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::One(iter) => iter.next(),
            Self::Two(iter) => iter.next(),
        }
    }
}

pub(crate) fn action_requires_unit_l2(action: &Action) -> bool {
    matches!(
        action,
        Action::ResidualL2Fraction { .. }
            | Action::ProjectionAblate { .. }
            | Action::SourceToTarget { .. }
            | Action::CoordinateSwap { .. }
    )
}

pub(crate) fn normalize_direction(
    mut row: Vec<f32>,
    normalization: Normalization,
    id: &str,
) -> Result<Vec<f32>> {
    ensure!(!row.is_empty(), "direction {id} is empty");
    ensure!(
        row.iter().all(|value| value.is_finite()),
        "direction {id} has a non-finite value"
    );
    let norm_squared = row
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    let norm = norm_squared.sqrt();
    ensure!(
        norm.is_finite() && norm > 0.0,
        "direction {id} has a zero or non-finite norm"
    );
    if normalization == Normalization::UnitL2 {
        for value in &mut row {
            *value = (f64::from(*value) / norm) as f32;
        }
    }
    ensure!(
        row.iter().all(|value| value.is_finite()),
        "direction {id} normalization overflowed"
    );
    Ok(row)
}

pub(crate) fn operation_enabled(operation: &OperationDefinition) -> bool {
    operation.action.coefficient() != 0.0
}
