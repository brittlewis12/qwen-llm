//! Bounded recurrence scheduling change; all state and arithmetic remain F32.
use super::packed::{Arith, FastPrecision, PackedLineage};

crate::env_flag!(
    /// Opt-in M5 Fast/Half KDA prefill recurrence with four independent values per SIMD group at 64 heads and 1024/2048 rows. No new state format or scratch. Zero restores the incumbent recurrence; Exact and F32 modes retain it.
    default_off enabled,
    "QWEN_GLM53_KDA_VALUE4"
);

pub(super) fn selected(arith: Arith, heads: usize, rows: usize) -> bool {
    enabled()
        && arith.lineage == PackedLineage::Fast
        && arith.precision == FastPrecision::Half
        && heads == 64
        && matches!(rows, 1024 | 2048)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protects_exact_f32_and_untested_geometry() {
        for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
            for precision in FastPrecision::ALL {
                for heads in [1, 32, 64, 128] {
                    for rows in [0, 1, 8, 512, 1023, 1024, 1025, 2048, 2049, 4096] {
                        assert_eq!(
                            selected(Arith { lineage, precision }, heads, rows),
                            enabled()
                                && lineage == PackedLineage::Fast
                                && precision == FastPrecision::Half
                                && heads == 64
                                && matches!(rows, 1024 | 2048)
                        );
                    }
                }
            }
        }
    }
}
