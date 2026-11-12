//! Row-preserving, completion-gated prefill interleaving. All callbacks run
//! after submitted commands drain, while the appender remains unavailable.
use super::*;

/// Conservative M5 Full-append wall estimate for the production GSQ artifact.
/// The fixed host/weight-streaming cost makes tiny row cuts unsuitable.
pub fn predicted_ms(rows: usize) -> f64 {
    if rows == 0 {
        0.0
    } else {
        275.0 + 0.8 * rows as f64
    }
}

pub(super) fn ends(rows: usize, layers: usize, target_ms: usize, minimum_ms: usize) -> Vec<usize> {
    let predicted = predicted_ms(rows);
    if target_ms == 0 || layers < 2 || predicted <= minimum_ms as f64 {
        return Vec::new();
    }
    let slices = ((predicted / target_ms as f64).ceil() as usize).clamp(1, layers);
    if slices < 2 {
        return Vec::new();
    }
    (1..=slices).map(|i| i * layers / slices).collect()
}

impl Forward<'_> {
    /// Preserve the selected row geometry, yielding only with drained commands.
    /// A callback error/cancellation leaves the appender poisoned for disposal;
    /// it cannot be decoded or snapshotted before final validation succeeds.
    pub fn prefill_layer_interleaved(
        &self,
        state: &mut State,
        tokens: &[u32],
        cancel: Option<&AtomicBool>,
        with_logits: bool,
        target_ms: usize,
        minimum_ms: usize,
        between: &mut dyn FnMut() -> Result<()>,
    ) -> std::result::Result<Vec<f32>, StateError> {
        let ends = ends(
            tokens
                .len()
                .min(state.prefill_rows.unwrap_or(self.prefill_rows)),
            self.layers.len(),
            target_ms,
            minimum_ms,
        );
        if ends.is_empty() {
            return if with_logits {
                self.prefill(state, tokens, cancel)
            } else {
                self.advance_prefill(state, tokens, cancel)
                    .map(|_| Vec::new())
            };
        }
        let rows = self
            .select_prefill(
                state,
                tokens.len(),
                state.caller_cpu_reserve,
                state.prefill_rows,
            )?
            .selected_rows;
        let mut slicing = prefill::LayerInterleave {
            ends: &ends,
            between,
            commands: Vec::new(),
        };
        prefill::run_interleaved(
            self,
            state,
            tokens,
            cancel,
            rows,
            with_logits,
            None,
            Some(&mut slicing),
        )
    }

    /// CED keeps its original encoder rows, replay tiles and two-layer groups.
    /// The episode remains provisional through decoder validation and commit.
    pub fn cold_prefill_episode_layer_interleaved(
        &self,
        state: &mut State,
        tokens: &[u32],
        cancel: Option<&AtomicBool>,
        max_rows: usize,
        target_ms: usize,
        minimum_ms: usize,
        between: &mut dyn FnMut() -> Result<()>,
    ) -> std::result::Result<ColdPrefillSlice, StateError> {
        let ends = ends(
            tokens
                .len()
                .min(max_rows)
                .min(state.prefill_rows.unwrap_or(self.prefill_rows)),
            self.layers.len(),
            target_ms,
            minimum_ms,
        );
        if ends.is_empty() {
            return self.cold_prefill_episode_slice(state, tokens, cancel, max_rows);
        }
        let mut slicing = prefill::LayerInterleave {
            ends: &ends,
            between,
            commands: Vec::new(),
        };
        self.cold_prefill_episode_slice_interleaved(
            state,
            tokens,
            cancel,
            max_rows,
            Some(&mut slicing),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selective_cuts_preserve_small_appends_and_exact_zero_rollback() {
        for rows in [0, 1, 32, 128, 281] {
            assert!(ends(rows, 40, 200, 500).is_empty());
        }
        for rows in [342, 1509, 4096] {
            assert!(ends(rows, 40, 0, 500).is_empty());
            let cuts = ends(rows, 40, 200, 500);
            assert_eq!(cuts.last(), Some(&40));
            assert!(cuts[0] > 0 && cuts.windows(2).all(|x| x[0] < x[1]));
        }
        assert_eq!(ends(4096, 40, 1, 0), (1..=40).collect::<Vec<_>>());
    }
}
