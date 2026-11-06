//! Bounded recurrence scheduling change; all state and arithmetic remain F32.
use super::packed::{Arith, FastPrecision, PackedLineage};

crate::env_flag!(
    /// Default-on M5 Fast/Half KDA prefill recurrence with four independent values per SIMD group at 64 heads and 1024/2048 rows. No new state format or scratch. Zero restores the incumbent recurrence; Exact and F32 modes retain it.
    default_on enabled,
    "QWEN_GLM53_KDA_VALUE4"
);

pub(super) fn selected(arith: Arith, heads: usize, rows: usize) -> bool {
    enabled()
        && arith.lineage == PackedLineage::Fast
        && arith.precision == FastPrecision::Half
        && heads == 64
        && (matches!(rows, 1024 | 2048) || (rows == 512 && super::short_prefill::enabled()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protects_exact_f32_and_untested_geometry() {
        let half = Arith {
            lineage: PackedLineage::Fast,
            precision: FastPrecision::Half,
        };
        for rows in [1024, 2048] {
            assert_eq!(selected(half, 64, rows), enabled());
        }
        assert_eq!(
            selected(half, 64, 512),
            enabled() && crate::glm5_next_metal::short_prefill::enabled()
        );
        for rows in [0, 1, 8, 511, 513, 1023, 1025, 2047, 2049, 4096] {
            assert!(!selected(half, 64, rows));
        }
        for heads in [1, 32, 128] {
            assert!(!selected(half, heads, 2048));
        }
        for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
            for precision in FastPrecision::ALL {
                let arith = Arith { lineage, precision };
                if arith != half {
                    assert!(!selected(arith, 64, 2048));
                    assert!(!selected(arith, 64, 512));
                }
            }
        }
    }
}
