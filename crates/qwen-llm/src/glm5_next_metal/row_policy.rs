//! Additional qualified command widths; never changes the admitted chunk size.
crate::env_flag!(
    /// M5 Fast/Half small-row matrix routes at 128..512 actual command rows. Zero restores the previous matrix gates; enclosing ANY_ROWS, SHORT_PREFILL, precision, tier and per-stage controls still apply. Sparse attention and workspace admission are unchanged.
    default_on small_requested,
    "QWEN_GLM53_PREFILL_SMALL_ROWS"
);

pub(crate) fn small_expanded(rows: usize) -> bool {
    #[cfg(test)]
    let requested = SMALL_OVERRIDE
        .with(std::cell::Cell::get)
        .unwrap_or_else(small_requested);
    #[cfg(not(test))]
    let requested = small_requested();
    requested && (128..512).contains(&rows) && expanded(rows) && super::short_prefill::enabled()
}

crate::env_flag!(
    /// Default-on M5 GLM Fast/Half KDA value4 at 128..=4096 command rows and wide Q6_K at 1025..=4096, including guarded tails. IQ gate/up and IQ3_S down additionally admit 1024..=4096 with PREFILL_IQ_ANY_ROWS; sparse attention retains its qualified row gates. Zero restores the previous 512/1024/2048 row gates. Existing per-stage, short-prefill, precision and tensor-tier rollbacks still apply; no memory-plan change.
    default_on requested,
    "QWEN_GLM53_PREFILL_ANY_ROWS"
);

crate::env_flag!(
    /// Default-on M5 GLM Fast/Half IQ2_S/IQ3_S gate/up and IQ3_S down at arbitrary 1024..=4096 command rows, within PREFILL_ANY_ROWS. Zero restores only the previous IQ 512/1024/2048 gates; preserves KDA/Q6, Exact/F32, existing per-stage controls and admitted workspace size.
    default_on iq_requested,
    "QWEN_GLM53_PREFILL_IQ_ANY_ROWS"
);

#[cfg(test)]
thread_local! {
    static SMALL_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static IQ_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) struct SmallScope(Option<bool>, std::marker::PhantomData<std::rc::Rc<()>>);
#[cfg(test)]
impl SmallScope {
    pub(crate) fn new(value: bool) -> Self {
        Self(
            SMALL_OVERRIDE.with(|v| v.replace(Some(value))),
            std::marker::PhantomData,
        )
    }
}
#[cfg(test)]
impl Drop for SmallScope {
    fn drop(&mut self) {
        SMALL_OVERRIDE.with(|v| v.set(self.0));
    }
}

pub(crate) fn expanded(rows: usize) -> bool {
    #[cfg(test)]
    if let Some(value) = OVERRIDE.with(std::cell::Cell::get) {
        return value && (128..=4096).contains(&rows);
    }
    requested() && (128..=4096).contains(&rows)
}

pub(crate) fn iq_expanded(rows: usize) -> bool {
    #[cfg(test)]
    if let Some(value) = IQ_OVERRIDE.with(std::cell::Cell::get) {
        return value && rows >= 1024 && expanded(rows);
    }
    iq_requested() && rows >= 1024 && expanded(rows)
}

#[cfg(test)]
pub(crate) struct IqScope(Option<bool>, std::marker::PhantomData<std::rc::Rc<()>>);
#[cfg(test)]
impl IqScope {
    pub(crate) fn new(value: bool) -> Self {
        Self(
            IQ_OVERRIDE.with(|v| v.replace(Some(value))),
            std::marker::PhantomData,
        )
    }
}
#[cfg(test)]
impl Drop for IqScope {
    fn drop(&mut self) {
        IQ_OVERRIDE.with(|v| v.set(self.0));
    }
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
    fn small_matrices_preserve_modes_bounds_and_independent_rollback() {
        use crate::glm5_next_metal::{memory, tensor_down, tensor_gate_up};
        let _rows = Scope::new(true);
        let _small = SmallScope::new(true);
        let _dense = tensor_prefill::Scope::new(true);
        let _up = memory::TensorGateUpScope::new(true);
        let _down = tensor_down::Scope::new(true);
        let half = Arith {
            lineage: PackedLineage::Fast,
            precision: FastPrecision::Half,
        };
        let active = super::super::short_prefill::enabled();
        let up_active = tensor_gate_up::short_selected(
            half,
            StageMode::Fast,
            GgmlType::IQ2_S,
            GgmlType::IQ2_S,
            (4096, 2048, 288, 8),
            512,
        );
        for rows in 0..=8192 {
            assert_eq!(small_expanded(rows), active && (128..512).contains(&rows));
        }
        for rows in [128, 129, 399, 400, 511] {
            for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
                for precision in FastPrecision::ALL {
                    let arith = Arith { lineage, precision };
                    for mode in [StageMode::Fast, StageMode::Exact, StageMode::FastF32] {
                        let expected = active && arith == half && mode == StageMode::Fast;
                        for dtype in [GgmlType::Q6_K, GgmlType::Q8_0] {
                            assert_eq!(
                                tensor_prefill::eligible(arith, mode, dtype, 4096, 8192, rows)
                                    .unwrap(),
                                expected
                            );
                        }
                        assert_eq!(
                            tensor_down::selected(arith, mode, GgmlType::IQ3_S, rows),
                            expected
                        );
                        for dtype in [GgmlType::IQ2_S, GgmlType::IQ3_S] {
                            assert_eq!(
                                tensor_gate_up::short_selected(
                                    arith,
                                    mode,
                                    dtype,
                                    dtype,
                                    (4096, 2048, 288, 8),
                                    rows
                                ),
                                expected && up_active
                            );
                        }
                    }
                }
            }
        }
        {
            let _off = SmallScope::new(false);
            assert!(!small_expanded(400));
            assert!(expanded(400));
            assert!(
                !tensor_prefill::eligible(half, StageMode::Fast, GgmlType::Q6_K, 4096, 8192, 400)
                    .unwrap()
            );
        }
        {
            let _off = Scope::new(false);
            assert!(!small_expanded(400));
        }
        assert_eq!(small_expanded(400), active);
    }

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
    fn iq_rows_require_both_rollbacks_and_preserve_legacy_widths() {
        use crate::glm5_next_metal::{memory, tensor_down, tensor_gate_up};
        let _rows = Scope::new(true);
        let _iq = IqScope::new(true);
        let _up = memory::TensorGateUpScope::new(true);
        let _down = tensor_down::Scope::new(true);
        let half = Arith {
            lineage: PackedLineage::Fast,
            precision: FastPrecision::Half,
        };
        let shape = (4096, 2048, 288, 8);
        let up_on = tensor_gate_up::selected(
            half,
            StageMode::Fast,
            GgmlType::IQ2_S,
            GgmlType::IQ2_S,
            shape,
            1024,
        );
        for rows in [0, 1, 512, 513, 1023, 4097, usize::MAX] {
            assert!(!iq_expanded(rows));
        }
        for rows in [1025, 1084, 1596, 1852, 2047, 2049, 4095, 4096] {
            for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
                for precision in FastPrecision::ALL {
                    let a = Arith { lineage, precision };
                    for mode in [StageMode::Fast, StageMode::Exact, StageMode::FastF32] {
                        assert_eq!(
                            tensor_down::selected(a, mode, GgmlType::IQ3_S, rows),
                            a == half && mode == StageMode::Fast
                        );
                        for dtype in [GgmlType::IQ2_S, GgmlType::IQ3_S] {
                            assert_eq!(
                                tensor_gate_up::selected(a, mode, dtype, dtype, shape, rows),
                                a == half && mode == StageMode::Fast && up_on
                            );
                        }
                    }
                }
            }
            for dtype in [GgmlType::Q4_K, GgmlType::Q5_K] {
                assert!(!tensor_gate_up::selected(
                    half,
                    StageMode::Fast,
                    dtype,
                    dtype,
                    shape,
                    rows
                ));
            }
        }
        {
            let _off = IqScope::new(false);
            assert!(!iq_expanded(1596));
            assert!(expanded(1596));
            assert!(!tensor_down::selected(
                half,
                StageMode::Fast,
                GgmlType::IQ3_S,
                1596
            ));
            for rows in [1024, 2048] {
                assert!(tensor_down::selected(
                    half,
                    StageMode::Fast,
                    GgmlType::IQ3_S,
                    rows
                ));
            }
        }
        assert!(iq_expanded(1596));
        {
            let _off = Scope::new(false);
            assert!(!iq_expanded(1596));
            assert!(!tensor_down::selected(
                half,
                StageMode::Fast,
                GgmlType::IQ3_S,
                1596
            ));
        }
        assert!(iq_expanded(1596));
        assert!(!iq_expanded(usize::MAX));
    }

    #[test]
    fn irregular_rows_keep_precision_stage_and_short_prefill_contracts() {
        let _small = SmallScope::new(false);
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
