//! Default-off, down-only transfer. GLM weights the slots after down, never
//! the gate/up activations. The shared adapter owns binding/capability checks.

use super::packed::{Arith, FastPrecision, PackedLineage, StageMode};
use crate::tensor::GgmlType;

const DEFAULT_ENABLED: bool = false;

#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Same-thread, nestable and panic-safe qualification control. This changes
/// only the request, not arithmetic, geometry, tier or global rollback checks.
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
    DEFAULT_ENABLED
}

pub(super) fn selected(arith: Arith, mode: StageMode, dtype: GgmlType, rows: usize) -> bool {
    enabled()
        && arith.lineage == PackedLineage::Fast
        && arith.precision == FastPrecision::Half
        && mode == StageMode::Fast
        && dtype == GgmlType::IQ3_S
        && matches!(rows, 1024 | 2048)
}

#[cfg(test)]
#[path = "tests/tensor_down.rs"]
mod tests;
