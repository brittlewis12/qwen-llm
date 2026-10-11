//! Executable coverage by tensor role and storage type.
//!
//! "Kept native" (`metal_forward::weight_dtype_kept_native`) is a storage
//! property. These queries instead record which existing kernel family can
//! execute a role at a dtype, for single-token decode and for packed prefill.
//! A role/dtype pair absent from the dispatch capabilities has no path; binding
//! refuses it before residency rather than falling back to an F32 expansion. `Pending`
//! names a known adaptation inside an existing path, and
//! [`Glm5NextModel::validate_execution`] refuses a mode while any of its cells
//! is pending. Weight coverage is not graph capability: routing, KDA, latent
//! attention and selection are separate implementation gates.
//!
//! Entries are recipes for existing kernel families at this release's shapes;
//! shape-specific GPU fixtures land with the encoders that use them.

use super::Glm5NextModel;
use crate::tensor::GgmlType;
use std::collections::BTreeMap;

/// How the graph consumes a tensor.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TensorRole {
    /// Token embedding row gather.
    Embedding,
    /// Output head (last position only).
    Head,
    /// F32 norm weights, biases, decay parameters and pool position terms.
    Vector,
    /// Per-channel causal depthwise convolution taps (F32).
    Conv,
    /// Hyper-connection mix projection `[hidden * streams, mixes]`.
    HyperMix,
    /// Quantized dense projection.
    Projection,
    /// Router logits and indexer head weights. Historical storage-named role;
    /// native dense dispatch determines supported types, including non-F32.
    F32Projection,
    /// Per-head MLA absorption/expansion (`attn_k_b`, `attn_v_b`).
    LatentAbsorb,
    /// Routed expert gate/up banks, consumed by clamped SwiGLU.
    ExpertGateUp,
    /// Routed expert down banks, weighted and summed.
    ExpertDown,
    /// NextN (MTP) tensors: recognized, never executed.
    NextN,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Support {
    /// An existing kernel family implements the role's contract.
    Kernel(&'static str),
    /// A known adaptation of an existing path is still required.
    Pending(&'static str),
}

impl Support {
    pub fn is_pending(self) -> bool {
        matches!(self, Self::Pending(_))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoleCoverage {
    pub decode: Support,
    pub prefill: Support,
}

/// Execution phase that a residency/session must be able to run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionMode {
    /// One token per forward (prefill also serial).
    SerialDecode,
    /// Serial decode plus packed multi-row prefill.
    PackedPrefill,
}

impl RoleCoverage {
    /// First pending requirement for `mode`, if any.
    pub fn pending_for(self, mode: ExecutionMode) -> Option<&'static str> {
        let pending = |s: Support| match s {
            Support::Pending(reason) => Some(reason),
            Support::Kernel(_) => None,
        };
        match mode {
            ExecutionMode::SerialDecode => pending(self.decode),
            ExecutionMode::PackedPrefill => pending(self.decode).or(pending(self.prefill)),
        }
    }
}

const fn both(kernel: &'static str) -> RoleCoverage {
    RoleCoverage {
        decode: Support::Kernel(kernel),
        prefill: Support::Kernel(kernel),
    }
}

const fn split(decode: Support, prefill: Support) -> RoleCoverage {
    RoleCoverage { decode, prefill }
}

const DENSE: &str = "dense mat_vec / optional mat_mat with per-row mat_vec fallback";

/// Coverage of `role` stored as `dtype`, or `None` when no path exists.
pub fn coverage(role: TensorRole, dtype: GgmlType) -> Option<RoleCoverage> {
    use crate::metal::{
        all_slots_down_supported, all_slots_gate_up_supported, get_rows_pipeline_name,
        grouped_projection_supported,
    };
    use crate::metal_forward::mat_vec_dispatch_supported;
    use GgmlType as T;
    use TensorRole as R;
    Some(match role {
        R::Embedding => both(get_rows_pipeline_name(dtype)?),
        // The head consumes the last position only, even during packed prefill.
        R::Head if mat_vec_dispatch_supported(dtype) => both("dense mat_vec (last position)"),
        R::Vector | R::Conv if dtype == T::F32 => both("elementwise / kernel_ssm_conv_silu_f32"),
        // Packed execution can use per-row GEMV when GEMM is unavailable.
        // An optional optimization must not narrow admission.
        R::Projection | R::F32Projection if mat_vec_dispatch_supported(dtype) => both(DENSE),
        // HC batches F32 projection where supported and otherwise uses row
        // GEMV. Grouped projection is optional, not an admission requirement.
        R::HyperMix if mat_vec_dispatch_supported(dtype) => {
            both("encode_mhc4_pre_generic / fused Q8_0 pre")
        }
        R::LatentAbsorb if grouped_projection_supported(dtype) => {
            both("encode_mat_vec_grouped_f32 / encode_mat_mat_grouped_f32")
        }
        // Packed execution falls back to all-slot row decode when the grouped
        // clamped gate/up pair or grouped down kernel is unavailable. Binding
        // additionally checks the actual independently typed gate/up pair.
        R::ExpertGateUp if all_slots_gate_up_supported(dtype, dtype) => split(
            Support::Kernel("encode_all_slots_gate_up_swiglu"),
            Support::Kernel("optional grouped clamped SwiGLU / all-slot row decode"),
        ),
        R::ExpertDown if all_slots_down_supported(dtype) => split(
            Support::Kernel("encode_all_slots_down"),
            Support::Kernel("optional grouped down / all-slot row decode"),
        ),
        _ => return None,
    })
}

/// One (role, dtype) cell of a bound model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoverageRow {
    pub role: TensorRole,
    pub dtype: GgmlType,
    pub tensors: usize,
    pub bytes: u64,
    pub coverage: RoleCoverage,
}

impl Glm5NextModel<'_> {
    /// Role x dtype matrix over executed tensors (NextN excluded).
    pub fn coverage(&self) -> Vec<CoverageRow> {
        let mut cells = BTreeMap::<(TensorRole, i32), CoverageRow>::new();
        for bound in self.trunk.iter().filter(|b| b.role != TensorRole::NextN) {
            let dtype = bound.tensor.dtype;
            let row = cells
                .entry((bound.role, dtype as i32))
                .or_insert_with(|| CoverageRow {
                    role: bound.role,
                    dtype,
                    tensors: 0,
                    bytes: 0,
                    coverage: coverage(bound.role, dtype)
                        .expect("binding admits only covered role/dtype pairs"),
                });
            row.tensors += 1;
            row.bytes += bound.tensor.n_bytes;
        }
        cells.into_values().collect()
    }

    /// Cells that still need an adaptation in some mode.
    pub fn pending_coverage(&self) -> Vec<CoverageRow> {
        self.coverage()
            .into_iter()
            .filter(|row| row.coverage.decode.is_pending() || row.coverage.prefill.is_pending())
            .collect()
    }

    /// Residency/session admission gate: refuses `mode` while any executed
    /// weight cell is pending for it. Callers must run this before allocating.
    pub fn validate_execution(&self, mode: ExecutionMode) -> super::Result<()> {
        let blocked = self
            .coverage()
            .into_iter()
            .filter_map(|row| {
                row.coverage
                    .pending_for(mode)
                    .map(|reason| format!("{:?} {:?}: {reason}", row.role, row.dtype))
            })
            .collect::<Vec<_>>();
        if blocked.is_empty() {
            Ok(())
        } else {
            Err(super::Glm5NextError::Unsupported {
                key: format!("execution mode {mode:?}"),
                detail: blocked.join("; "),
            })
        }
    }
}
