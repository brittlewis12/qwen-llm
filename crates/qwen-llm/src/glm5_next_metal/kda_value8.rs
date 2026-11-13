//! Wider independent-value panel for the GLM full-chunk KDA recurrence.
use super::packed::{Arith, FastPrecision, PackedLineage};

crate::env_flag!(
    /// Default-off M5 Fast/Half GLM KDA recurrence with eight independent value rows per SIMD group at 2,048 command rows. Each value row keeps the existing F32 recurrence order. Zero restores KDA value4.
    default_off enabled,
    "QWEN_GLM53_KDA_VALUE8"
);

#[cfg(feature = "perf-ablations")]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

fn requested() -> bool {
    #[cfg(feature = "perf-ablations")]
    if let Some(value) = OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    enabled()
}

pub(super) fn selected(arith: Arith, heads: usize, rows: usize) -> bool {
    requested()
        && arith.lineage == PackedLineage::Fast
        && arith.precision == FastPrecision::Half
        && heads == 64
        && rows == 2048
}

#[cfg(feature = "perf-ablations")]
pub fn with_prefill<R>(value: bool, body: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OVERRIDE.with(|state| state.set(self.0));
        }
    }
    let _restore = Restore(OVERRIDE.with(|state| state.replace(Some(value))));
    body()
}
