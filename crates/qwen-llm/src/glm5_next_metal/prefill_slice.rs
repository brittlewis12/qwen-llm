//! Row-preserving layer quanta. The cost seed is the qualified M5 Fast/Half
//! geometry; observed slow slices reduce subsequent quanta without changing
//! any row, kernel or arithmetic choice.

/// Warm M5 Fast/Half wall-time seed for one original packed row forward at
/// roughly 48K depth. This prices the measured row regimes, not a hard bound;
/// it excludes idle residency, sampling and interleaved peer service.
pub fn predicted_prefill_ms(rows: usize) -> f64 {
    // Measured row regimes, including the ragged 513..1023 attention cliff.
    match rows {
        0 => 0.0,
        1..=8 => 20.0 + 14.0 * rows.saturating_sub(1) as f64,
        9..=112 => 212.0 + 3.0 * rows as f64,
        113..=512 => 224.0 + 0.97 * rows as f64,
        513..=1023 => 362.0 + 1.68 * rows as f64,
        _ => 204.0 + 0.90 * rows as f64,
    }
}

pub(super) struct Cursor {
    pub tokens: Vec<u32>,
    pub plan: LayerPlan,
}

pub(super) struct LayerPlan {
    layers: usize,
    next: usize,
    slices: usize,
    target_ms: f64,
    layer_cap: usize,
}
impl LayerPlan {
    pub fn new(layers: usize, rows: usize, target_ms: f64) -> Self {
        let total = predicted_prefill_ms(rows);
        let wanted = ((total / target_ms).ceil() as usize).clamp(1, layers);
        let slices = wanted.next_power_of_two().min(layers);
        Self {
            layers,
            next: 0,
            slices,
            target_ms,
            layer_cap: layers,
        }
    }
    pub fn next(&self) -> usize {
        self.next
    }
    pub fn end(&self, peers: bool) -> usize {
        if !peers {
            return self.layers;
        }
        let grid = (1..=self.slices)
            .map(|i| i * self.layers / self.slices)
            .find(|&end| end > self.next)
            .unwrap_or(self.layers);
        grid.min(self.next.saturating_add(self.layer_cap))
    }
    pub fn complete(&mut self, end: usize, gpu_ms: f64) {
        let count = end - self.next;
        // Allow the measured whole-layer skew (~127ms for a nominal 100ms).
        // Cold residency or an unseen geometry may be much slower: react to
        // actual GPU time, with at least one complete layer per submission.
        if gpu_ms > 1.5 * self.target_ms {
            self.layer_cap = self
                .layer_cap
                .min((self.target_ms * count as f64 / gpu_ms).floor().max(1.0) as usize);
        }
        self.next = end;
    }
    pub fn finished(&self) -> bool {
        self.next == self.layers
    }
}

/// A completed command of a row-preserving packed append. `logits` becomes
/// available only after every layer, head and status validation have finished.
pub struct Glm5NextPrefillSlice {
    pub layers: usize,
    pub gpu_ms: f64,
    pub logits: Option<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layer_quanta_keep_the_row_grid_and_cover_each_layer_once() {
        for (rows, slices) in [(32, 4), (342, 8), (1509, 16)] {
            let mut plan = LayerPlan::new(45, rows, 100.0);
            assert_eq!(plan.slices, slices);
            let mut seen = Vec::new();
            while !plan.finished() {
                let end = plan.end(true);
                seen.extend(plan.next()..end);
                plan.complete(end, 100.0);
            }
            assert_eq!(seen, (0..45).collect::<Vec<_>>());
        }
    }
    #[test]
    fn two_hundred_ms_target_keeps_large_append_row_geometry() {
        for (rows, slices) in [(342, 4), (951, 16), (1509, 8)] {
            let mut p = LayerPlan::new(45, rows, 200.0);
            assert_eq!(p.slices, slices);
            let mut layers = Vec::new();
            while !p.finished() {
                let end = p.end(true);
                layers.extend(p.next()..end);
                p.complete(end, 200.0);
            }
            assert_eq!(layers, (0..45).collect::<Vec<_>>());
        }
    }
    #[test]
    fn cold_slice_reduces_future_layers_and_no_peers_drains_the_same_chunk() {
        let mut plan = LayerPlan::new(45, 1509, 100.0);
        let first = plan.end(true);
        assert_eq!(first, 2);
        plan.complete(first, 500.0);
        assert_eq!(plan.end(true), 3);
        assert_eq!(plan.end(false), 45);
        plan.complete(45, 1000.0);
        assert!(plan.finished());
    }
    #[test]
    fn tiny_targets_still_make_progress_and_large_targets_do_not_split() {
        for target in [0.001, 1.0, 100.0, 10000.0] {
            let mut p = LayerPlan::new(45, 32, target);
            let mut commands = 0;
            while !p.finished() {
                let end = p.end(true);
                assert!(end > p.next());
                p.complete(end, 200.0);
                commands += 1;
            }
            assert!(commands <= 45);
            if target == 10000.0 {
                assert_eq!(commands, 1);
            }
        }
    }
}
