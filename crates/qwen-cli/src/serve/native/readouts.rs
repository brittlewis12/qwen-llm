//! CPU planning for bounded original-forward plain readouts.

use crate::lens_scope::Scope;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const MAX_READOUTS: usize = 1024;
pub(crate) const MAX_TOP_K: usize = 1024;
pub(crate) const MAX_HEAD_EVALUATIONS: usize = 4096;
pub(crate) const MAX_SCORES: usize = 262_144;
pub(crate) const MAX_ROWS: usize = 16_384;
pub(crate) const MAX_LABEL_BYTES: usize = 64 * 1024;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Readout {
    pub(crate) id: String,
    pub(crate) lens: String,
    pub(crate) mode: String,
    pub(crate) scope: Scope,
    pub(crate) top_k: usize,
    #[serde(default)]
    retain: Option<String>,
}

#[derive(Debug, Default)]
pub(crate) struct Plan {
    pub(crate) readouts: Vec<Readout>,
    // Absolute consumed position -> layer -> caller-ordered readout indices.
    pub(crate) events: BTreeMap<u32, BTreeMap<u32, Vec<usize>>>,
    pub(crate) max_capture_layers: usize,
    pub(crate) head_evaluations: usize,
    pub(crate) output_rows: usize,
    pub(crate) output_scores: usize,
}

impl Plan {
    pub(crate) fn compile(
        values: &[Value],
        layers: u32,
        prompt: usize,
        max_tokens: usize,
        vocab: usize,
    ) -> Result<Self> {
        ensure!(values.len() <= MAX_READOUTS, "too many readout requests");
        ensure!(
            prompt > 0 && max_tokens > 0,
            "empty readout execution bounds"
        );
        let prompt = u32::try_from(prompt)?;
        let decode = u32::try_from(max_tokens - 1)?;
        prompt
            .checked_add(decode)
            .context("readout position overflow")?;
        let mut plan = Self::default();
        let mut ids = BTreeSet::new();
        for value in values {
            let readout: Readout = serde_json::from_value(value.clone())?;
            ensure!(
                readout.retain.is_none(),
                "readout retention production is not recovered"
            );
            ensure!(
                !readout.id.is_empty() && readout.id.len() <= 256 && ids.insert(readout.id.clone()),
                "readout IDs must be nonempty, bounded and unique"
            );
            ensure!(
                readout.lens == "plain" && readout.mode == "full_vocabulary",
                "only plain full_vocabulary readouts are recovered"
            );
            ensure!(
                readout.top_k > 0 && readout.top_k <= MAX_TOP_K.min(vocab),
                "readout top_k exceeds native bounds"
            );
            readout.scope.validate(&readout.id, 1)?;
            let layer_count = readout.scope.layers.numeric_len(layers, "readout layers")?;
            ensure!(layer_count > 0, "readout selects no model layers");
            let prefill_count = readout
                .scope
                .prefill
                .as_ref()
                .map(|s| s.numeric_len(prompt, "readout prefill"))
                .transpose()?
                .unwrap_or_default();
            let decode_count = readout
                .scope
                .decode
                .as_ref()
                .map(|s| s.numeric_len(decode, "readout decode"))
                .transpose()?
                .unwrap_or_default();
            ensure!(
                prefill_count > 0 || decode_count > 0,
                "readout selects no reachable forwards"
            );
            let rows = prefill_count
                .checked_add(decode_count)
                .and_then(|n| n.checked_mul(layer_count))
                .context("readout rows overflow")?;
            plan.output_rows = plan
                .output_rows
                .checked_add(rows)
                .context("readout rows overflow")?;
            plan.output_scores = rows
                .checked_mul(readout.top_k)
                .and_then(|n| plan.output_scores.checked_add(n))
                .context("readout scores overflow")?;
            ensure!(
                plan.output_rows <= MAX_ROWS && plan.output_scores <= MAX_SCORES,
                "aggregate readout publication exceeds row/score budget"
            );
            // A single request's sites are unique: refuse large All before expansion.
            ensure!(
                rows <= MAX_HEAD_EVALUATIONS,
                "aggregate full-vocabulary head evaluation budget exceeded"
            );
            let selected_layers = readout.scope.layers.expand(layers, "readout layers")?;
            let prefill = readout
                .scope
                .prefill
                .as_ref()
                .map(|s| s.expand(prompt, "readout prefill"))
                .transpose()?
                .unwrap_or_default();
            let generated = readout
                .scope
                .decode
                .as_ref()
                .map(|s| s.expand(decode, "readout decode"))
                .transpose()?
                .unwrap_or_default();
            let index = plan.readouts.len();
            for position in prefill
                .into_iter()
                .chain(generated.into_iter().map(|index| prompt + index))
            {
                let event = plan.events.entry(position).or_default();
                for &layer in &selected_layers {
                    if !event.contains_key(&layer) {
                        plan.head_evaluations += 1;
                        ensure!(
                            plan.head_evaluations <= MAX_HEAD_EVALUATIONS,
                            "aggregate full-vocabulary head evaluation budget exceeded"
                        );
                    }
                    event.entry(layer).or_default().push(index);
                }
                plan.max_capture_layers = plan.max_capture_layers.max(event.len());
            }
            plan.readouts.push(readout);
        }
        Ok(plan)
    }

    pub(crate) fn memory_bytes(
        &self,
        loaded: &qwen_llm::runtime::LoadedModel,
    ) -> Result<(u64, u64)> {
        self.price(
            loaded.arch().hidden_size as usize,
            loaded.arch().vocab_size as usize,
            |bytes| Ok(loaded.context().shared_buffer_size_and_align(bytes)?.size),
        )
    }

    fn price(
        &self,
        hidden: usize,
        vocab: usize,
        mut aligned: impl FnMut(u64) -> Result<u64>,
    ) -> Result<(u64, u64)> {
        if self.events.is_empty() {
            return Ok((0, 0));
        }
        let hidden = u64::try_from(hidden)?
            .checked_mul(4)
            .context("hidden bytes overflow")?;
        let vocab = u64::try_from(vocab)?
            .checked_mul(4)
            .context("vocabulary bytes overflow")?;
        let capture = hidden
            .checked_mul(u64::try_from(self.max_capture_layers)?)
            .context("capture bytes overflow")?;
        let mut gpu = 0u64;
        // Original-forward capture plus transported, normalized and head-logit tensors.
        for bytes in [capture, hidden, hidden, vocab] {
            gpu = gpu
                .checked_add(aligned(bytes)?)
                .context("readout GPU bytes overflow")?;
        }
        // Readback, transported vector, observer/new/previous logits, bounded top-k,
        // lossy labels and record construction. Ranking uses the existing O(k) helper.
        let cpu = capture
            .checked_add(hidden)
            .and_then(|n| vocab.checked_mul(3).and_then(|v| n.checked_add(v)))
            .and_then(|n| n.checked_add(2 * super::RECORD_BYTES as u64))
            .context("readout CPU bytes overflow")?;
        Ok((gpu, cpu))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn readout(id: &str) -> Value {
        json!({"id":id,"lens":"plain","mode":"full_vocabulary","top_k":2,
            "scope":{"layers":{"kind":"values","values":[0,3]},"prefill":{"kind":"values","values":[0,2]},"decode":{"kind":"all"}}})
    }
    #[test]
    fn shared_heads_keep_order_and_exclude_final_unconsumed_sample() {
        let plan = Plan::compile(&[readout("a"), readout("b")], 4, 3, 3, 10).unwrap();
        assert_eq!(
            (
                plan.head_evaluations,
                plan.output_rows,
                plan.output_scores,
                plan.max_capture_layers
            ),
            (8, 16, 32, 2)
        );
        assert_eq!(plan.events[&3][&0], [0, 1]);
        assert!(!plan.events.contains_key(&5));
    }
    #[test]
    fn impossible_and_excessive_scopes_fail_before_expansion() {
        for scope in [
            json!({"layers":{"kind":"values","values":[4]},"prefill":{"kind":"all"}}),
            json!({"layers":{"kind":"all"},"prefill":{"kind":"values","values":[3]}}),
            json!({"layers":{"kind":"all"},"decode":{"kind":"values","values":[2]}}),
            json!({"layers":{"kind":"values","values":[1,1]},"prefill":{"kind":"all"}}),
        ] {
            let mut value = readout("bad");
            value["scope"] = scope;
            assert!(Plan::compile(&[value], 4, 3, 3, 10).is_err());
        }
        assert!(Plan::compile(&[readout("a"), readout("a")], 4, 3, 3, 10).is_err());
        let all = json!({"id":"all","lens":"plain","mode":"full_vocabulary","top_k":1,"scope":{"layers":{"kind":"all"},"prefill":{"kind":"all"}}});
        assert!(Plan::compile(&[all], u32::MAX, u32::MAX as usize, 1, 10).is_err());
        let mut unreachable = readout("zero");
        unreachable["scope"] = json!({"layers":{"kind":"all"},"decode":{"kind":"all"}});
        assert!(Plan::compile(&[unreachable], 4, 3, 1, 10).is_err());
        for (field, value) in [
            ("retain", json!("scores_and_residual")),
            ("lens", json!("fitted")),
            ("top_k", json!(0)),
        ] {
            let mut bad = readout("bad");
            bad[field] = value;
            assert!(Plan::compile(&[bad], 4, 3, 3, 10).is_err());
        }
    }
    #[test]
    fn pricing_aligns_each_gpu_allocation_and_keeps_host_copies_separate() {
        let plan = Plan::compile(&[readout("a")], 4, 3, 3, 10).unwrap();
        let mut allocations = Vec::new();
        let (gpu, cpu) = plan
            .price(5, 10, |bytes| {
                allocations.push(bytes);
                Ok(bytes.div_ceil(256) * 256)
            })
            .unwrap();
        assert_eq!(allocations, [40, 20, 20, 40]);
        assert_eq!(gpu, 1024);
        assert_eq!(cpu, 40 + 20 + 120 + 2 * super::super::RECORD_BYTES as u64);
        assert_eq!(
            Plan::default()
                .price(usize::MAX, usize::MAX, |_| panic!("no capture"))
                .unwrap(),
            (0, 0)
        );
        assert!(plan.price(usize::MAX, usize::MAX, Ok).is_err());
    }
}
