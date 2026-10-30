//! Default-on bounded GLM Q5_K down for M5 Fast/Half. The shared adapter owns M5 tier,
//! global rollback, binding and capacity checks before bucketing writes.
//! F32 activation storage, Half operands and F32 sums; only reduction order
//! changes versus the generic Half tile. Gate/up and post-weighting stay put.

use super::packed::{Arith, FastPrecision, PackedLineage, StageMode};
use crate::tensor::GgmlType;

crate::env_flag!(
    /// Default-on bounded M5 Fast/Half GLM Q5_K expert down tensor policy; 0 restores the previous down route. Arithmetic, geometry, committed-row, tier, capacity and global rollback guards remain mandatory.
    default_on requested,
    "QWEN_GLM53_TENSOR_Q5_DOWN"
);

#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) struct Scope {
    previous: Option<bool>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(test)]
impl Scope {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            previous: OVERRIDE.with(|flag| flag.replace(Some(enabled))),
            _thread: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
impl Drop for Scope {
    fn drop(&mut self) {
        OVERRIDE.with(|flag| flag.set(self.previous));
    }
}

fn enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = OVERRIDE.with(std::cell::Cell::get) {
        return value;
    }
    requested()
}

pub(super) fn selected(
    arith: Arith,
    mode: StageMode,
    dtype: GgmlType,
    geometry: (usize, usize, usize, usize),
    rows: usize,
) -> bool {
    enabled()
        && arith.lineage == PackedLineage::Fast
        && arith.precision == FastPrecision::Half
        && mode == StageMode::Fast
        && dtype == GgmlType::Q5_K
        && geometry == (2048, 4096, 288, 8)
        && matches!(rows, 1024 | 2048)
}

#[cfg(test)]
#[path = "tests/tensor_q5_down.rs"]
mod tests;
