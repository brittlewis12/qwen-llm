//! Default-on bounded GLM gate/up transfer; the down policy remains independent.

use super::{
    memory,
    packed::{Arith, FastPrecision, PackedLineage, StageMode},
};
use crate::tensor::GgmlType;

crate::env_flag!(
    /// Opt-in bounded M5 Fast/Half IQ2_S/IQ3_S routed tensor gate/up; preserves the independent down policy and all enclosing guards.
    default_off iq_tensor_gate_up_enabled,
    "QWEN_GLM53_IQ_TENSOR_GATE_UP"
);

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
        && (matches!(gate, GgmlType::Q4_K | GgmlType::Q5_K)
            || (iq_tensor_gate_up_enabled() && matches!(gate, GgmlType::IQ2_S | GgmlType::IQ3_S)))
        && matches!(rows, 1024 | 2048)
}

#[cfg(test)]
#[path = "tests/tensor_gate_up.rs"]
mod tests;
