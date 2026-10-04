//! Executable coverage by tensor role and storage type.
//!
//! "Kept native" (`metal_forward::weight_dtype_kept_native`) is a storage
//! property. This table instead records which existing kernel family can
//! execute a role at a dtype, for single-token decode and for packed prefill.
//! A role/dtype pair absent from the table has no path, and binding refuses it
//! before residency rather than falling back to an F32 expansion. `Pending`
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
    /// F32 dense projection (router logits, indexer head weights).
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

const DENSE: &str = "dense mat_vec / mat_mat";

/// Coverage of `role` stored as `dtype`, or `None` when no path exists.
pub fn coverage(role: TensorRole, dtype: GgmlType) -> Option<RoleCoverage> {
    use GgmlType as T;
    use TensorRole as R;
    Some(match (role, dtype) {
        (R::Embedding, T::Q6_K | T::Q8_0 | T::Q4_K | T::F16 | T::BF16 | T::F32) => {
            both("kernel_get_rows_*")
        }
        (R::Head, T::Q6_K | T::Q8_0 | T::Q4_K | T::Q5_K | T::F16 | T::BF16) => both(DENSE),
        (R::Vector | R::Conv, T::F32) => both("elementwise / kernel_ssm_conv_silu_f32"),
        // Unweighted RMSNorm over [hidden * streams], the Q8_0 projection to the
        // 24 mixes, then controls; controls alone consume projected F32 mixes.
        (R::HyperMix, T::Q8_0) => split(
            Support::Kernel("rms_norm + mat_vec_q8_0 + kernel_deepseek_v4_hc_controls"),
            Support::Kernel("rms_norm + mat_mat_q8_0 + kernel_deepseek_v4_hc_controls_batch"),
        ),
        (
            R::Projection,
            T::Q8_0 | T::Q6_K | T::Q5_K | T::Q4_K | T::IQ4_XS | T::F16 | T::BF16 | T::F32,
        ) => both(DENSE),
        // Logits only: sigmoid top-8 routing is a separate graph capability.
        (R::F32Projection, T::F32) => split(
            Support::Kernel("kernel_mat_vec_f32_f32"),
            Support::Kernel("kernel_mat_mat_f32_f32_router_e8p32"),
        ),
        (R::LatentAbsorb, T::Q8_0) => split(
            Support::Kernel("kernel_mat_vec_q8_0_f32_lcpp_grouped"),
            Support::Kernel("encode_mat_mat_q8_0_grouped_f32 (128-row blocks) + grouped GEMV tail"),
        ),
        // Expert entries list only dtypes verified on both paths (decode via
        // metal::expert's all-slot encoders, GPU-tested at GLM widths); other
        // artifacts add theirs with evidence.
        (R::ExpertGateUp, T::IQ2_S | T::IQ3_S) => split(
            Support::Kernel("encode_all_slots_gate_up_swiglu"),
            Support::Kernel("encode_grouped_routed_experts (clamped grouped SwiGLU)"),
        ),
        (R::ExpertDown, T::IQ3_S) => split(
            Support::Kernel("encode_all_slots_down (all_slots_down_iq3_s)"),
            Support::Kernel("encode_grouped_routed_experts (grouped generic down)"),
        ),
        (R::ExpertDown, T::IQ4_XS) => split(
            Support::Kernel("encode_all_slots_down (moe_down_iq4_xs_fast)"),
            Support::Kernel("encode_grouped_routed_experts (moe_down_iq4_xs grouped)"),
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
