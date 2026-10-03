//! Transport-neutral intervention wire forms and direction normalization.

use crate::lens_scope::Scope;
use anyhow::{Result, ensure};
use qwen_llm::{metal::MetalTensor, metal_forward::PostBlockIntervention};
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

pub(crate) fn validate_action(
    id: &str,
    action: &Action,
    mut direction: impl FnMut(&str) -> Result<Option<Normalization>>,
) -> Result<()> {
    ensure!(
        action.coefficient().is_finite(),
        "operation {id} coefficient must be finite"
    );
    if let Action::CoordinateSwap {
        source,
        target,
        coefficient,
    } = action
    {
        ensure!(
            source != target,
            "coordinate-swap operation {id} requires distinct source and target directions"
        );
        ensure!(
            (2.0 * coefficient).is_finite(),
            "coordinate-swap operation {id} coefficient overflows its reflection scale"
        );
    }
    // Resolve every reference before normalization, preserving CLI error precedence.
    let directions = action
        .direction_ids()
        .map(|name| direction(name).map(|normalization| (name, normalization)))
        .collect::<Result<Vec<_>>>()?;
    if action_requires_unit_l2(action) {
        for (name, normalization) in directions {
            if let Some(normalization) = normalization {
                ensure!(
                    normalization == Normalization::UnitL2,
                    "operation {id} requires unit_l2 direction {name}"
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn lower<'a>(
    action: &Action,
    layer: u32,
    mut direction: impl FnMut(&str) -> Result<&'a MetalTensor>,
    reflection: impl FnOnce() -> Result<&'a MetalTensor>,
) -> Result<PostBlockIntervention<'a>> {
    Ok(match action {
        Action::FixedAdd {
            direction: id,
            coefficient,
        } => PostBlockIntervention::Fixed {
            layer,
            direction: direction(id)?,
            coefficient: *coefficient,
        },
        Action::ResidualL2Fraction {
            direction: id,
            coefficient,
        } => PostBlockIntervention::ResidualL2Relative {
            layer,
            direction: direction(id)?,
            coefficient: *coefficient,
        },
        Action::ProjectionAblate {
            direction: id,
            coefficient,
        } => PostBlockIntervention::Projection {
            layer,
            direction: direction(id)?,
            coefficient: *coefficient,
        },
        Action::SourceToTarget {
            source,
            target,
            coefficient,
        } => PostBlockIntervention::SourceToTarget {
            layer,
            source: direction(source)?,
            target: direction(target)?,
            coefficient: *coefficient,
        },
        Action::CoordinateSwap { coefficient, .. } => PostBlockIntervention::Projection {
            layer,
            direction: reflection()?,
            coefficient: 2.0 * coefficient,
        },
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_validation_preserves_reference_precedence_and_deferred_native_normalization() {
        let action = Action::SourceToTarget {
            source: "a".into(),
            target: "missing".into(),
            coefficient: 0.0,
        };
        let mut calls = Vec::new();
        let error = validate_action("op", &action, |name| {
            calls.push(name.to_owned());
            ensure!(name != "missing", "unknown {name}");
            Ok(Some(Normalization::AsStored))
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "unknown missing");
        assert_eq!(calls, ["a", "missing"]);
        assert!(validate_action("op", &action, |_| Ok(None)).is_ok());
        assert_eq!(
            validate_action("op", &action, |_| Ok(Some(Normalization::AsStored)))
                .unwrap_err()
                .to_string(),
            "operation op requires unit_l2 direction a"
        );
        assert!(validate_action("op", &action, |_| Ok(Some(Normalization::UnitL2))).is_ok());
        let fixed = Action::FixedAdd {
            direction: "a".into(),
            coefficient: -0.0,
        };
        assert!(validate_action("op", &fixed, |_| Ok(Some(Normalization::AsStored))).is_ok());
        assert!(validate_action("op", &fixed, |_| anyhow::bail!("missing")).is_err());
    }

    #[test]
    fn shared_validation_refuses_invalid_scales_before_direction_lookup() {
        let lookup =
            |_: &str| -> Result<Option<Normalization>> { panic!("invalid action must fail first") };
        let fixed = Action::FixedAdd {
            direction: "a".into(),
            coefficient: f32::NAN,
        };
        assert_eq!(
            validate_action("op", &fixed, lookup)
                .unwrap_err()
                .to_string(),
            "operation op coefficient must be finite"
        );
        for (source, target, coefficient, expected) in [
            (
                "a",
                "a",
                1.0,
                "requires distinct source and target directions",
            ),
            (
                "a",
                "b",
                f32::MAX,
                "coefficient overflows its reflection scale",
            ),
        ] {
            let action = Action::CoordinateSwap {
                source: source.into(),
                target: target.into(),
                coefficient,
            };
            assert_eq!(
                validate_action("op", &action, lookup)
                    .unwrap_err()
                    .to_string(),
                format!("coordinate-swap operation op {expected}")
            );
        }
    }

    #[test]
    fn lowering_keeps_family_lookup_errors_and_reflections_lazy() {
        for action in [
            Action::FixedAdd {
                direction: "d".into(),
                coefficient: 1.0,
            },
            Action::ResidualL2Fraction {
                direction: "d".into(),
                coefficient: 1.0,
            },
            Action::ProjectionAblate {
                direction: "d".into(),
                coefficient: 1.0,
            },
            Action::SourceToTarget {
                source: "d".into(),
                target: "t".into(),
                coefficient: 1.0,
            },
        ] {
            let error = lower(
                &action,
                2,
                |id| anyhow::bail!("family row {id} missing"),
                || panic!("reflection not requested"),
            )
            .err()
            .unwrap();
            assert_eq!(error.to_string(), "family row d missing");
        }
        let action = Action::CoordinateSwap {
            source: "s".into(),
            target: "t".into(),
            coefficient: 0.5,
        };
        let error = lower(
            &action,
            2,
            |_| panic!("use prepared reflection only"),
            || anyhow::bail!("family reflection missing"),
        )
        .err()
        .unwrap();
        assert_eq!(error.to_string(), "family reflection missing");
    }
}
