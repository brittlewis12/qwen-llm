//! Bounded, authored-order operation admission; no model work during acceptance.

use super::registry::{MatrixKey, Registry};
use crate::lens_intervention::{
    self, Action, DirectionRow, DirectionTargetCovector, LensRowDirectionDefinition,
    OperationDefinition, OperationSite, operation_enabled,
};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

mod prepare;
pub(super) use prepare::PreparedDirections;

pub(crate) const MAX_DIRECTIONS: usize = 1024;
pub(crate) const MAX_OPERATIONS: usize = 1024;
pub(crate) const MAX_APPLICATIONS: usize = 16_384;
pub(crate) const MAX_DIRECTION_ROWS: usize = 4096;
pub(crate) const MAX_DIRECTION_BYTES: u64 = 256 * 1024 * 1024;
pub(crate) const MAX_PROJECTION_PRODUCTS: u64 = 1024 * 1024 * 1024;
pub(crate) const OPERATORS: [&str; 3] = ["fixed_add", "residual_l2_fraction", "projection_ablate"];

#[derive(Default)]
pub(crate) struct Plan {
    pub(crate) directions: BTreeMap<String, LensRowDirectionDefinition>,
    pub(crate) operations: Vec<OperationDefinition>,
    pub(crate) events: BTreeMap<u32, BTreeMap<u32, Vec<usize>>>,
    pub(crate) rows: BTreeMap<MatrixKey, BTreeMap<u32, Vec<String>>>,
    pub(crate) matrices: BTreeSet<MatrixKey>,
    pub(crate) assets: BTreeMap<String, Value>,
    pub(crate) applications: usize,
    pub(crate) direction_rows: usize,
    pub(crate) projected_rows: usize,
}

impl Plan {
    pub(super) fn applied_ids(&self, position: u32, layer: u32) -> Vec<&str> {
        self.events
            .get(&position)
            .and_then(|e| e.get(&layer))
            .map(|ids| {
                ids.iter()
                    .map(|&i| self.operations[i].id.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(super) fn publish_applications(
        &self,
        position: u32,
        counters: &crate::serve::jobs::state::Counters,
        sink: &super::Sink,
    ) -> Result<()> {
        use crate::serve::jobs::state::Phase;
        let phase = if u64::from(position) < counters.prompt_tokens {
            Phase::Prefill
        } else {
            Phase::Decode
        };
        let index = if phase == Phase::Prefill {
            u64::from(position)
        } else {
            u64::from(position) - counters.prompt_tokens
        };
        if let Some(event) = self.events.get(&position) {
            for (&layer, indices) in event {
                for &order in indices {
                    super::checkpoint(sink)?;
                    let operation = &self.operations[order];
                    sink.record(serde_json::json!({"kind":"operation_application","id":operation.id,"layer":layer,
                        "phase":phase,"index":index,"position":position,"order":order,"action":operation.action,
                        "provenance":"successful_original_forward"}),phase,counters);
                    super::checkpoint(sink)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn compile(
        directions: &[Value],
        operations: &[Value],
        layers: u32,
        prompt: usize,
        max_tokens: usize,
        vocab: usize,
        hidden: usize,
        registry: Option<&Registry>,
    ) -> Result<Self> {
        ensure!(
            directions.len() <= MAX_DIRECTIONS && operations.len() <= MAX_OPERATIONS,
            "direction/operation count exceeds native limits"
        );
        let mut plan = Self::default();
        for value in directions {
            let direction: LensRowDirectionDefinition = serde_json::from_value(value.clone())?;
            ensure!(
                !direction.id.is_empty()
                    && direction.id.len() <= 256
                    && !plan.directions.contains_key(&direction.id),
                "direction IDs must be unique, nonempty and bounded"
            );
            ensure!(
                direction.effective_target_covector()
                    == DirectionTargetCovector::DeployedLogitNumerator,
                "native interventions currently require deployed_logit_numerator covectors"
            );
            let DirectionRow::TokenId { token_id } = direction.row else {
                anyhow::bail!("native fitted directions require token_id rows");
            };
            ensure!(
                token_id >= 0 && (token_id as usize) < vocab,
                "direction token outside vocabulary"
            );
            let asset = registry
                .context("no fitted direction aliases registered")?
                .asset(&direction.lens)?;
            ensure!(
                asset["direction_rows"]
                    .as_array()
                    .is_some_and(|kinds| kinds.contains(&Value::from("token_id"))),
                "registered alias does not support token direction projection"
            );
            plan.assets.insert(direction.lens.clone(), asset.clone());
            plan.directions.insert(direction.id.clone(), direction);
        }
        ensure!(
            prompt > 0 && max_tokens > 0,
            "empty operation execution bounds"
        );
        let prompt = u32::try_from(prompt)?;
        let decode = u32::try_from(max_tokens - 1)?;
        prompt
            .checked_add(decode)
            .context("operation position overflow")?;
        let mut ids = BTreeSet::new();
        let mut used = BTreeSet::new();
        for value in operations {
            let operation: OperationDefinition = serde_json::from_value(value.clone())?;
            ensure!(
                !operation.id.is_empty()
                    && operation.id.len() <= 256
                    && ids.insert(operation.id.clone()),
                "operation IDs must be unique, nonempty and bounded"
            );
            ensure!(
                operation.site == OperationSite::PostBlock,
                "native operations support only the post_block site"
            );
            ensure!(
                matches!(
                    operation.action,
                    Action::FixedAdd { .. }
                        | Action::ResidualL2Fraction { .. }
                        | Action::ProjectionAblate { .. }
                ),
                "native operation kind is not supported"
            );
            lens_intervention::validate_action(&operation.id, &operation.action, |id| {
                Ok(Some(
                    plan.directions
                        .get(id)
                        .context("operation references unknown direction")?
                        .normalization,
                ))
            })?;
            operation.scope.validate(&operation.id, 1)?;
            ensure!(
                decode > 0 || operation.scope.decode.is_none(),
                "decode scope cannot reach a transition when max_new_tokens is 1"
            );
            let layer_count = operation
                .scope
                .layers
                .numeric_len(layers, "operation layers")?;
            ensure!(
                layer_count <= MAX_DIRECTION_ROWS,
                "operation layer count exceeds direction row bound"
            );
            let prefill = operation
                .scope
                .prefill
                .as_ref()
                .map(|s| s.numeric_len(prompt, "operation prefill"))
                .transpose()?
                .unwrap_or(0);
            let generated = operation
                .scope
                .decode
                .as_ref()
                .map(|s| s.numeric_len(decode, "operation decode"))
                .transpose()?
                .unwrap_or(0);
            ensure!(
                layer_count > 0 && prefill + generated > 0,
                "operation selects no reachable forwards"
            );
            let applications = prefill
                .checked_add(generated)
                .and_then(|n| n.checked_mul(layer_count))
                .context("operation applications overflow")?;
            if operation_enabled(&operation) {
                plan.applications = plan
                    .applications
                    .checked_add(applications)
                    .context("operation applications overflow")?;
                ensure!(
                    plan.applications <= MAX_APPLICATIONS,
                    "operation application budget exceeded"
                );
            }
            let selected = operation.scope.layers.expand(layers, "operation layers")?;
            // Validate even zero controls, without admitting their matrices or GPU rows.
            for id in operation.action.direction_ids() {
                let direction = &plan.directions[id];
                let asset = &plan.assets[&direction.lens];
                let available = asset["source_layers"]
                    .as_array()
                    .context("registered source layers")?;
                ensure!(
                    selected
                        .iter()
                        .all(|&layer| available.contains(&Value::from(layer))),
                    "operation selects an absent lens layer"
                );
            }
            let index = plan.operations.len();
            if operation_enabled(&operation) {
                for id in operation.action.direction_ids() {
                    let direction = &plan.directions[id];
                    let DirectionRow::TokenId { token_id } = direction.row else {
                        unreachable!()
                    };
                    for &layer in &selected {
                        if used.insert((id.to_owned(), layer)) {
                            let key = MatrixKey {
                                alias: direction.lens.clone(),
                                layer,
                            };
                            plan.matrices.insert(key.clone());
                            plan.rows
                                .entry(key)
                                .or_default()
                                .entry(token_id as u32)
                                .or_default()
                                .push(id.to_owned());
                            plan.direction_rows += 1;
                        }
                    }
                }
                ensure!(
                    plan.direction_rows <= MAX_DIRECTION_ROWS,
                    "direction row budget exceeded"
                );
                let prefill = operation
                    .scope
                    .prefill
                    .as_ref()
                    .map(|s| s.expand(prompt, "operation prefill"))
                    .transpose()?
                    .unwrap_or_default();
                let generated = operation
                    .scope
                    .decode
                    .as_ref()
                    .map(|s| s.expand(decode, "operation decode"))
                    .transpose()?
                    .unwrap_or_default();
                for position in prefill
                    .into_iter()
                    .chain(generated.into_iter().map(|index| prompt + index))
                {
                    for &layer in &selected {
                        plan.events
                            .entry(position)
                            .or_default()
                            .entry(layer)
                            .or_default()
                            .push(index);
                    }
                }
            }
            plan.operations.push(operation);
        }
        plan.projected_rows = plan.rows.values().map(|tokens| tokens.len()).sum();
        let retained = (plan.direction_rows as u64)
            .checked_mul(hidden as u64)
            .and_then(|n| n.checked_mul(4))
            .context("direction byte overflow")?;
        ensure!(
            retained <= MAX_DIRECTION_BYTES,
            "retained direction byte budget exceeded"
        );
        let products = (plan.projected_rows as u64)
            .checked_mul(hidden as u64)
            .and_then(|n| n.checked_mul(hidden as u64))
            .context("direction projection work overflow")?;
        ensure!(
            products <= MAX_PROJECTION_PRODUCTS,
            "direction projection work budget exceeded"
        );
        if !plan.matrices.is_empty() {
            registry
                .context("missing registry")?
                .matrix_bytes(&plan.matrices)?;
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests;
