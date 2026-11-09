//! Additional screened row cohorts for the Flash-Next prompt routes.

mod hc_dot;
pub(crate) mod qsa_index_mma;
pub(crate) mod shapes;
pub(crate) mod small_f32;
pub(crate) mod small_f32_mma;

crate::env_flag!(
    /// Default-on Flash-Next short tensor dense/gate-up projections, parallel bucketing and aligned BF16 HC at M512..M2047. APPEND_SHAPES also admits M128..M511 on the tensor tier; separate shape controls complete ragged HC and routed down. Zero restores M2048-only qualification for these routes. No new precision or buffers. Wide experts and the tensor router retain M2048 qualification.
    default_on short_enabled,
    "QWEN_FN_SHORT_APPEND_TENSOR"
);

fn qualifies(short: bool, rows: usize) -> bool {
    rows == 2048 || short && (512..2048).contains(&rows)
}

pub(crate) fn rows_qualified(rows: usize) -> bool {
    qualifies(short_enabled(), rows) || (short_enabled() && shapes::selected(rows))
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
