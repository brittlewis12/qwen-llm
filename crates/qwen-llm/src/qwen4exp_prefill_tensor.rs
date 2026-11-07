//! Additional screened row cohorts for the Flash-Next prompt routes.

pub(crate) mod small_f32;
mod small_f32_mma;

crate::env_flag!(
    /// Extend Flash-Next tensor dense and routed gate/up projections and parallel bucketing to 512..2047-row appends, plus BF16 HC at multiples of 128 rows, default on when their existing controls and hardware gates allow them. Set to 0 to restore the 2048-row-only qualification. No new operand precision or allocated buffers; routed down, wide expert gate/up and the tensor router retain their 2048-row qualification.
    default_on short_enabled,
    "QWEN_FN_SHORT_APPEND_TENSOR"
);

fn qualifies(short: bool, rows: usize) -> bool {
    rows == 2048 || short && (512..2048).contains(&rows)
}

pub(crate) fn rows_qualified(rows: usize) -> bool {
    qualifies(short_enabled(), rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_append_row_qualification_preserves_rollback_and_tails() {
        for on in [false, true] {
            assert!(qualifies(on, 2048));
            for m in [512, 513, 640, 769, 1009, 1024, 1152, 1536, 1920, 2047] {
                assert_eq!(qualifies(on, m), on);
            }
            for m in [0, 1, 32, 128, 256, 511, 2049, 4096] {
                assert!(!qualifies(on, m));
            }
        }
    }
}
