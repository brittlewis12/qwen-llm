//! Experimental independent-row KDA preparation for long packed chunks.
use super::packed::{Arith, FastPrecision, PackedLineage};
use std::cell::Cell;

thread_local! {
    static OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

crate::env_flag!(
    /// Default-off M5 Fast/Half GLM KDA convolution and normalization preparation in independent row/head groups at 128..=4096 rows. Reuses the dead dense gate/up and KDA output buffers to preserve projected q/k/v rows; adds no admitted scratch. Zero restores the serial preparation kernel.
    default_off enabled,
    "QWEN_GLM53_KDA_PARALLEL_PREP"
);

pub(super) fn selected(
    arith: Arith,
    heads: usize,
    rows: usize,
    raw_q_capacity: u64,
    raw_k_capacity: u64,
) -> bool {
    OVERRIDE.with(|override_value| override_value.get().unwrap_or_else(enabled))
        && arith.lineage == PackedLineage::Fast
        && arith.precision == FastPrecision::Half
        && heads == 64
        && (128..=4096).contains(&rows)
        && raw_q_capacity >= (heads * 128 * rows) as u64
        && raw_k_capacity >= (heads * 128 * rows) as u64
}

/// Temporary route override for the in-process paired benchmark. Production
/// callers use the cached environment default.
pub(super) struct Scope(Option<bool>);

impl Scope {
    pub(super) fn new(enabled: bool) -> Self {
        Self(OVERRIDE.with(|override_value| override_value.replace(Some(enabled))))
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        OVERRIDE.with(|override_value| override_value.set(self.0));
    }
}
