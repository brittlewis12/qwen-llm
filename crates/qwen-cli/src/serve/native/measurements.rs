//! Independent, bounded before/after measurement scopes and observed-vector metrics.

use crate::lens_scope::{Scope, Selector as IndexSet};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const MAX_REQUESTS: usize = 1024;
pub(crate) const MAX_ROWS: usize = 16_384;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub(crate) id: String,
    pub(crate) scope: Scope,
}

#[derive(Debug, Default)]
pub(crate) struct Plan {
    pub(crate) requests: Vec<Request>,
    pub(crate) events: BTreeMap<u32, BTreeMap<u32, Vec<usize>>>,
    pub(crate) rows: usize,
    pub(crate) max_layers: usize,
}
impl Plan {
    pub(crate) fn compile(
        values: &[Value],
        layers: u32,
        prompt: usize,
        max_tokens: usize,
    ) -> Result<Self> {
        ensure!(prompt > 0, "empty prompt bound");
        ensure!(
            values.len() <= MAX_REQUESTS,
            "too many residual pair requests"
        );
        let prompt = u32::try_from(prompt)?;
        let decode = u32::try_from(
            max_tokens
                .checked_sub(1)
                .context("empty generation bound")?,
        )?;
        prompt
            .checked_add(decode)
            .context("pair position overflow")?;
        let mut plan = Self::default();
        let mut ids = BTreeSet::new();
        for value in values {
            let request: Request = serde_json::from_value(value.clone())?;
            ensure!(
                !request.id.is_empty() && request.id.len() <= 256 && ids.insert(request.id.clone()),
                "residual pair IDs must be bounded and unique"
            );
            request.scope.validate(&request.id, 1)?;
            let count = |phase: &Option<IndexSet>, bound, name| {
                phase
                    .as_ref()
                    .map(|s| s.numeric_len(bound, name))
                    .transpose()
                    .map(|n| n.unwrap_or(0))
            };
            let prefill = count(&request.scope.prefill, prompt, "pair prefill")?;
            let generated = count(&request.scope.decode, decode, "pair decode")?;
            let layer_count = request.scope.layers.numeric_len(layers, "pair layers")?;
            let rows = prefill
                .checked_add(generated)
                .and_then(|n| n.checked_mul(layer_count))
                .context("pair row overflow")?;
            ensure!(rows > 0, "pair selects no reachable forwards");
            plan.rows = plan.rows.checked_add(rows).context("pair rows overflow")?;
            ensure!(
                plan.rows <= MAX_ROWS,
                "residual pair publication exceeds row budget"
            );
            let selected_layers = request.scope.layers.expand(layers, "pair layers")?;
            let expand = |phase: &Option<IndexSet>, bound, name| {
                phase
                    .as_ref()
                    .map(|s| s.expand(bound, name))
                    .transpose()
                    .map(|n| n.unwrap_or_default())
            };
            let prefill = expand(&request.scope.prefill, prompt, "pair prefill")?;
            let generated = expand(&request.scope.decode, decode, "pair decode")?;
            for position in prefill
                .into_iter()
                .chain(generated.into_iter().map(|i| prompt + i))
            {
                let event = plan.events.entry(position).or_default();
                for &layer in &selected_layers {
                    event.entry(layer).or_default().push(plan.requests.len());
                }
                plan.max_layers = plan.max_layers.max(event.len());
            }
            plan.requests.push(request);
        }
        Ok(plan)
    }
    pub(crate) fn sites(&self) -> usize {
        self.events.values().map(BTreeMap::len).sum()
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct Metrics {
    pub(crate) norm_before: f64,
    pub(crate) norm_after: f64,
    pub(crate) delta_norm: f64,
    pub(crate) relative_delta: Option<f64>,
}
pub(crate) fn metrics(before: &[f32], after: &[f32]) -> Result<Metrics> {
    ensure!(
        !before.is_empty() && before.len() == after.len(),
        "pair dimensions differ or empty"
    );
    let (mut b, mut a, mut d) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in before.iter().zip(after) {
        ensure!(x.is_finite() && y.is_finite(), "nonfinite pair residual");
        let (x, y) = (f64::from(x), f64::from(y));
        b += x * x;
        a += y * y;
        d += (y - x) * (y - x);
    }
    let (norm_before, norm_after, delta_norm) = (b.sqrt(), a.sqrt(), d.sqrt());
    let relative_delta = (norm_before > 0.0).then(|| delta_norm / norm_before);
    ensure!(
        norm_before.is_finite()
            && norm_after.is_finite()
            && delta_norm.is_finite()
            && relative_delta.is_none_or(f64::is_finite),
        "nonfinite pair metrics"
    );
    Ok(Metrics {
        norm_before,
        norm_after,
        delta_norm,
        relative_delta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(id: &str) -> Value {
        json!({"id":id,"scope":{"layers":{"kind":"values","values":[1,3]},
            "prefill":{"kind":"values","values":[0]},"decode":{"kind":"all"}}})
    }

    #[test]
    fn shared_sites_keep_independent_rows_and_exclude_terminal_sample() {
        let plan = Plan::compile(&[request("a"), request("b")], 4, 3, 2).unwrap();
        assert_eq!(plan.rows, 8);
        assert_eq!(plan.sites(), 4);
        assert_eq!(plan.max_layers, 2);
        assert_eq!(plan.events.keys().copied().collect::<Vec<_>>(), [0, 3]);
        assert_eq!(plan.events[&3][&1], [0, 1]);
        assert_eq!(Plan::compile(&[request("a")], 4, 3, 1).unwrap().sites(), 2);
    }

    #[test]
    fn rejects_invalid_or_excessive_publication_before_expansion() {
        assert!(Plan::compile(&[request("a"), request("a")], 4, 3, 2).is_err());
        assert!(Plan::compile(&[request("")], 4, 3, 2).is_err());
        assert!(Plan::compile(&[request("a")], 3, 3, 2).is_err());
        assert!(Plan::compile(&vec![request("a"); MAX_REQUESTS + 1], 4, 3, 2).is_err());
        let huge = json!({"id":"a","scope":{"layers":{"kind":"all"},"prefill":{"kind":"all"}}});
        assert!(Plan::compile(&[huge], 64, MAX_ROWS, 1).is_err());
        let unreachable =
            json!({"id":"a","scope":{"layers":{"kind":"all"},"decode":{"kind":"all"}}});
        assert!(Plan::compile(&[unreachable], 4, 3, 1).is_err());
    }

    #[test]
    fn metrics_use_actual_vectors_and_define_zero_denominator() {
        let m = metrics(&[3., 4.], &[6., 8.]).unwrap();
        assert_eq!(
            (m.norm_before, m.norm_after, m.delta_norm, m.relative_delta),
            (5., 10., 5., Some(1.))
        );
        assert_eq!(
            metrics(&[3., 4.], &[3., 4.]).unwrap().relative_delta,
            Some(0.)
        );
        assert_eq!(metrics(&[0., 0.], &[3., 4.]).unwrap().relative_delta, None);
        assert!(metrics(&[], &[]).is_err());
        assert!(metrics(&[1.], &[1., 2.]).is_err());
        assert!(metrics(&[f32::NAN], &[1.]).is_err());
        assert!(metrics(&[1.], &[f32::INFINITY]).is_err());
        assert!(
            metrics(&[f32::MAX], &[-f32::MAX])
                .unwrap()
                .delta_norm
                .is_finite()
        );
    }
}
