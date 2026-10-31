//! Default-off GLM gate/up transfer; the down policy remains independent.

use super::{
    memory,
    packed::{Arith, FastPrecision, PackedLineage, StageMode},
};
use crate::tensor::GgmlType;

pub(super) fn selected(
    arith: Arith,
    mode: StageMode,
    gate: GgmlType,
    up: GgmlType,
    shape: (usize, usize, usize, usize),
    rows: usize,
) -> bool {
    memory::tensor_gate_up_enabled()
        && arith.lineage == PackedLineage::Fast
        && arith.precision == FastPrecision::Half
        && mode == StageMode::Fast
        && shape == (4096, 2048, 288, 8)
        && gate == up
        && matches!(gate, GgmlType::Q4_K | GgmlType::Q5_K)
        && matches!(rows, 1024 | 2048)
}

#[cfg(test)]
#[path = "tests/tensor_gate_up.rs"]
mod tests;
