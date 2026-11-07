//! Additional qualified command widths; never changes the admitted chunk size.
crate::env_flag!(
    /// Default-on M5 GLM Fast/Half KDA value4 at 128..=4096 command rows and wide Q6_K at 1025..=4096, including guarded tails. IQ and sparse attention retain their qualified row gates. Zero restores the previous 512/1024/2048 row gates. Existing per-stage, short-prefill, precision and tensor-tier rollbacks still apply; no memory-plan change.
    default_on requested,
    "QWEN_GLM53_PREFILL_ANY_ROWS"
);

#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

pub(crate) fn expanded(rows: usize) -> bool {
    #[cfg(test)]
    if let Some(value) = OVERRIDE.with(std::cell::Cell::get) {
        return value && (128..=4096).contains(&rows);
    }
    requested() && (128..=4096).contains(&rows)
}

#[cfg(test)]
pub(crate) struct Scope(Option<bool>, std::marker::PhantomData<std::rc::Rc<()>>);
#[cfg(test)]
impl Scope {
    pub(crate) fn new(value: bool) -> Self {
        Self(
            OVERRIDE.with(|v| v.replace(Some(value))),
            std::marker::PhantomData,
        )
    }
}
#[cfg(test)]
impl Drop for Scope {
    fn drop(&mut self) {
        OVERRIDE.with(|v| v.set(self.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm5_next_metal::{
        kda_value4,
        packed::{Arith, FastPrecision, PackedLineage, StageMode},
        short_prefill, tensor_prefill,
    };
    use crate::tensor::GgmlType;
    #[test]
    fn arbitrary_rows_are_bounded_and_rollback_is_nested() {
        let _on = Scope::new(true);
        for rows in 0..=8192 {
            assert_eq!(expanded(rows), (128..=4096).contains(&rows));
        }
        {
            let _off = Scope::new(false);
            for rows in 0..=8192 {
                assert!(!expanded(rows));
            }
        }
        assert!(expanded(1596));
    }
    #[test]
    fn irregular_rows_keep_precision_stage_and_short_prefill_contracts() {
        let _rows = Scope::new(true);
        let _dense = tensor_prefill::Scope::new(true);
        let half = Arith {
            lineage: PackedLineage::Fast,
            precision: FastPrecision::Half,
        };
        let kda_on = kda_value4::selected(half, 64, 1024);
        for rows in [
            128, 129, 511, 513, 1023, 1025, 1084, 1596, 1852, 2047, 2049, 4095, 4096,
        ] {
            for lineage in [PackedLineage::Exact, PackedLineage::Fast] {
                for precision in FastPrecision::ALL {
                    let a = Arith { lineage, precision };
                    assert_eq!(
                        kda_value4::selected(a, 64, rows),
                        a == half && kda_on && (rows >= 1024 || short_prefill::enabled())
                    );
                    assert!(!kda_value4::selected(a, 32, rows));
                    for mode in [StageMode::Exact, StageMode::FastF32, StageMode::Fast] {
                        assert_eq!(
                            tensor_prefill::eligible(a, mode, GgmlType::Q6_K, 4096, 8192, rows)
                                .unwrap(),
                            a == half && mode == StageMode::Fast && rows >= 1024
                        );
                    }
                }
            }
        }
    }
}
