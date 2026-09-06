//! Native Metal residency and correctness-first execution for the strict
//! DeepSeek V4 Flash-0731 binding.
//!
//! GGUF weights remain in their exact storage without dtype conversion. The
//! execution bodies deliberately have no dependency on the Qwen Metal model.

#[cfg(feature = "dsv4-diagnostics")]
mod diagnostics;
mod prefill;

#[cfg(feature = "dsv4-diagnostics")]
pub use prefill::{
    DeepSeekV4MhcBufferRole, DeepSeekV4MhcCommandInterval, DeepSeekV4MhcCommandKind,
    DeepSeekV4MhcDeleteArm, DeepSeekV4MhcDeleteProfile, DeepSeekV4MhcEndpointEvidence,
    DeepSeekV4MhcExecutionKind, DeepSeekV4MhcOracle, DeepSeekV4MhcOracleIdentity,
    DeepSeekV4MhcSiteKind, DeepSeekV4MhcSiteRecord, DeepSeekV4MhcTimedEndpoint,
    DeepSeekV4MhcVerifiedCapture, PackedChunkProfile, PackedPostRouteLayerMetadata,
    PackedPostRouteSampledLayerProfile, PackedPostRouteStageKind, PackedPostRouteStageProfile,
    PackedPostRouteStageTiming, PackedPrefillSampledLayerProfile, PackedPrefillStageKind,
    PackedPrefillStageProfile, PackedPrefillStageTiming, PackedPrefillStageTransition,
    seal_mhc_delete_oracle_pair,
};
mod snapshot;

#[cfg(feature = "dsv4-diagnostics")]
pub use diagnostics::{
    DeepSeekV4CsaDecision, DeepSeekV4DecisionLayer, DeepSeekV4DecisionTranscript,
    DeepSeekV4DiagnosticsError, DeepSeekV4Fp4CounterfactualStateDigest,
    DeepSeekV4Fp4ScoreDispatchLedger, DeepSeekV4Fp4ScorePlanKind, DeepSeekV4Fp4SelectionSource,
    DeepSeekV4Fp4ShadowEligibility, DeepSeekV4Fp4ShadowExecution, DeepSeekV4Fp4ShadowLayer,
    DeepSeekV4Fp4ShadowReport, DeepSeekV4RankedCsaRow, DeepSeekV4RouteDecision,
};

pub use snapshot::{
    DeepSeekV4CausalSnapshot, DeepSeekV4CompatibilityDigest, DeepSeekV4EncodedSnapshot,
    DeepSeekV4ModelContentId, DeepSeekV4SnapshotCaptureErrorKind,
    DeepSeekV4SnapshotCodecConstraints, DeepSeekV4SnapshotCodecError, DeepSeekV4SnapshotFileError,
    DeepSeekV4SnapshotFileOutcome, DeepSeekV4SnapshotFileReport, DeepSeekV4SnapshotObservation,
    DeepSeekV4SnapshotRestoreErrorKind, causal_snapshot_capture_error_kind,
    causal_snapshot_record_bytes, causal_snapshot_restore_error_kind, decode_causal_snapshot,
    encode_causal_snapshot, load_causal_snapshot_file, publish_causal_snapshot_file,
};

use crate::deepseek_v4::{
    AttentionKind, DeepSeekV4Config, DeepSeekV4Error, DeepSeekV4Model,
    flash_0731_expert_count_supported,
};
use crate::gguf::{GgufError, GgufFile};
use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalGgufBacking, MetalMemoryAdmission,
    MetalMemorySignals, MetalTensor, MetalTensorProvenance, MetalTimestampSampleBuffer,
    RetainedStorageDisposition, RetainedStorageFallback, RetainedStoragePlan, encode_get_rows_f32,
    encode_rms_norm_batched_f32, encode_rms_norm_mul_f32, encode_rms_norm_mul_rows_f32,
    encode_scatter_offset_f32_to_f16, evaluate_metal_memory_admission, host_page_size_bytes,
    plan_retained_storage,
};
use crate::tensor::{GgmlType, ggml_type_layout};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
#[cfg(feature = "dsv4-diagnostics")]
use objc2_metal::MTLCommandBufferStatus;
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState,
    MTLDevice, MTLResidencySet, MTLResidencySetDescriptor, MTLSize,
};
use std::cell::Cell;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;

pub const DEEPSEEK_V4_FLASH_0731_TENSOR_COUNT: usize = 1_328;
const GGUF_BINDING_ALIGNMENT: usize = 32;
const DEEPSEEK_V4_FRESH_SOURCE_BYTES: u64 = 104_202_502_492;
const DEEPSEEK_V4_REAP_K160_SOURCE_BYTES: u64 = 89_920_886_108;
const DEEPSEEK_V4_REAP_K216_SOURCE_BYTES: u64 = 89_060_075_612;
pub const DEEPSEEK_V4_CONNECTION_COUNT: usize = 4;
pub const DEEPSEEK_V4_HC_PARAMETER_COUNT: usize = 24;
pub const DEEPSEEK_V4_SINKHORN_ITERATIONS: usize = 20;
pub const DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
const DEEPSEEK_V4_ROUTE_STATUS_READY: i32 = 1;
const DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_LOGIT: i32 = -1;
const DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_BIAS: i32 = -2;
const DEEPSEEK_V4_ROUTE_STATUS_INVALID_TOKEN: i32 = -3;
const DEEPSEEK_V4_ROUTE_STATUS_INVALID_EXPERT: i32 = -4;
const DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_WEIGHT: i32 = -6;
const DEEPSEEK_V4_ROUTE_MAX_EXPERTS: usize = 256;
const DEEPSEEK_V4_ROUTE_MAX_TOP_K: usize = 6;
#[cfg(feature = "dsv4-diagnostics")]
const DEEPSEEK_V4_FP4_STATUS_UNAVAILABLE: i32 = i32::MIN;
pub use prefill::{DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS, DEEPSEEK_V4_PREFILL_MAX_TOKENS};
/// Engine-owned evidence ceiling through the model's exact context length.
/// A request may allocate less, but allocation never authorizes execution past
/// this independently promoted boundary.
pub const DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY: usize = 1_048_576;

/// SHA-256 of the exact metallib embedded in this diagnostics-enabled binary.
#[cfg(feature = "dsv4-diagnostics")]
pub fn deepseek_v4_diagnostics_metallib_sha256() -> [u8; 32] {
    use sha2::{Digest, Sha256};

    Sha256::digest(crate::KERNELS_METALLIB).into()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4SessionCapacity {
    forward_limit: u32,
    csa_physical_rows: usize,
    hca_physical_rows: usize,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MultigroupSelectorGeometry {
    forward_limit: usize,
    physical_capacity_rows: usize,
    max_visible_rows: usize,
}

impl DeepSeekV4MultigroupSelectorGeometry {
    pub fn forward_limit(self) -> usize {
        self.forward_limit
    }

    pub fn physical_capacity_rows(self) -> usize {
        self.physical_capacity_rows
    }

    pub fn max_visible_rows(self) -> usize {
        self.max_visible_rows
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MultigroupSelectorTelemetry {
    sealed: bool,
    multigroup_invocations: u64,
    ineligible_radix4_invocations: u64,
}

impl DeepSeekV4MultigroupSelectorTelemetry {
    pub fn sealed(self) -> bool {
        self.sealed
    }

    pub fn multigroup_invocations(self) -> u64 {
        self.multigroup_invocations
    }

    pub fn ineligible_radix4_invocations(self) -> u64 {
        self.ineligible_radix4_invocations
    }
}

impl DeepSeekV4SessionCapacity {
    pub fn for_forward_limit(
        forward_limit: usize,
        model_context_length: u32,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if forward_limit == 0 {
            return invalid("DeepSeek V4 session capacity requires at least one forward");
        }
        if forward_limit > DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY {
            return invalid(format!(
                "DeepSeek V4 request requires {forward_limit} forwards, beyond promoted evidence capacity {DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY}"
            ));
        }
        if forward_limit > model_context_length as usize {
            return invalid(format!(
                "DeepSeek V4 request requires {forward_limit} forwards, beyond model context length {model_context_length}"
            ));
        }
        let forward_limit = u32::try_from(forward_limit).map_err(|_| {
            DeepSeekV4MetalError::Invalid("DeepSeek V4 forward capacity exceeds u32".into())
        })?;
        let csa_physical_rows = rounded_history_capacity(
            forward_limit as usize / 4,
            DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
            "CSA",
        )?;
        let hca_physical_rows = rounded_history_capacity(
            forward_limit as usize / 128,
            DEEPSEEK_V4_HCA_HISTORY_CAPACITY_ROWS,
            "HCA",
        )?;
        Ok(Self {
            forward_limit,
            csa_physical_rows,
            hca_physical_rows,
        })
    }

    pub fn forward_limit(self) -> usize {
        self.forward_limit as usize
    }

    pub fn csa_physical_rows(self) -> usize {
        self.csa_physical_rows
    }

    pub fn hca_physical_rows(self) -> usize {
        self.hca_physical_rows
    }

    #[doc(hidden)]
    pub fn qualify_multigroup_selector_experiment(
        self,
    ) -> Result<DeepSeekV4MultigroupSelectorGeometry, DeepSeekV4MetalError> {
        let max_visible_rows = self.forward_limit() / 4;
        if !deepseek_v4_multigroup_selector_eligible(self.csa_physical_rows(), max_visible_rows) {
            return invalid(format!(
                "DeepSeek V4 session cannot reach the qualified multi-group selector band: forwards={} physical_capacity_rows={} max_visible_rows={} requires capacity={}..={} visible>={} and visible>=capacity-capacity/4",
                self.forward_limit(),
                self.csa_physical_rows(),
                max_visible_rows,
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS,
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_MAX_CAPACITY_ROWS,
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS,
            ));
        }
        Ok(DeepSeekV4MultigroupSelectorGeometry {
            forward_limit: self.forward_limit(),
            physical_capacity_rows: self.csa_physical_rows(),
            max_visible_rows,
        })
    }

    fn validate_position(self, position: u32) -> Result<(), DeepSeekV4MetalError> {
        if position >= self.forward_limit {
            return invalid(format!(
                "DeepSeek V4 session capacity is {} forwards; next position is {position}",
                self.forward_limit
            ));
        }
        Ok(())
    }

    fn validate_next_position(self, next_position: u32) -> Result<(), DeepSeekV4MetalError> {
        if next_position > self.forward_limit {
            return invalid(format!(
                "DeepSeek V4 state position {next_position} exceeds session capacity {}",
                self.forward_limit
            ));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DeepSeekV4MetalError {
    #[error("invalid DeepSeek V4 Metal residency: {0}")]
    Invalid(String),
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Schema(#[from] DeepSeekV4Error),
    #[cfg(feature = "dsv4-diagnostics")]
    #[error(transparent)]
    Diagnostics(#[from] DeepSeekV4DiagnosticsError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4ResidencyReport {
    pub tensor_count: usize,
    pub source_bytes: u64,
    pub window_count: usize,
    pub window_bytes: u64,
    pub view_count: usize,
    pub unique_view_bytes: u64,
    pub logical_view_bytes: u64,
    pub alias_count: usize,
    pub alias_bytes: u64,
    pub fallback_count: usize,
    pub fallback_bytes: u64,
    pub resident_bytes: u64,
    pub page_size: usize,
    pub max_buffer_length: usize,
    pub required_alignment: usize,
}

impl fmt::Display for DeepSeekV4ResidencyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tensors={} source_bytes={} windows={}/{} views={}/{} logical_view_bytes={} aliases={}/{} fallbacks={}/{} resident_bytes={} page={} max_buffer={} alignment={}",
            self.tensor_count,
            self.source_bytes,
            self.window_count,
            self.window_bytes,
            self.view_count,
            self.unique_view_bytes,
            self.logical_view_bytes,
            self.alias_count,
            self.alias_bytes,
            self.fallback_count,
            self.fallback_bytes,
            self.resident_bytes,
            self.page_size,
            self.max_buffer_length,
            self.required_alignment,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4SessionAllocation {
    pub name: String,
    pub logical_bytes: u64,
    pub priced_bytes: u64,
    pub alignment: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MemoryPlan {
    residency_buffer_count: usize,
    residency_logical_bytes: u64,
    residency_priced_upper_bytes: u64,
    session_logical_bytes: u64,
    session_priced_upper_bytes: u64,
    total_priced_upper_bytes: u64,
    session_allocations: Vec<DeepSeekV4SessionAllocation>,
}

impl DeepSeekV4MemoryPlan {
    pub fn residency_buffer_count(&self) -> usize {
        self.residency_buffer_count
    }

    pub fn residency_logical_bytes(&self) -> u64 {
        self.residency_logical_bytes
    }

    pub fn residency_priced_upper_bytes(&self) -> u64 {
        self.residency_priced_upper_bytes
    }

    pub fn session_logical_bytes(&self) -> u64 {
        self.session_logical_bytes
    }

    pub fn session_priced_upper_bytes(&self) -> u64 {
        self.session_priced_upper_bytes
    }

    pub fn total_priced_upper_bytes(&self) -> u64 {
        self.total_priced_upper_bytes
    }

    pub fn session_allocations(&self) -> &[DeepSeekV4SessionAllocation] {
        &self.session_allocations
    }

    pub fn admission(&self, signals: MetalMemorySignals) -> MetalMemoryAdmission {
        self.admission_for_sessions(signals, 1)
            .expect("the validated one-session memory plan must not overflow")
    }

    pub fn priced_upper_bytes_for_sessions(
        &self,
        session_count: usize,
    ) -> Result<u64, DeepSeekV4MetalError> {
        if session_count == 0 {
            return invalid("DeepSeek V4 memory admission requires at least one session");
        }
        let session_count = u64::try_from(session_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid("DeepSeek V4 session count exceeds u64".into())
        })?;
        let sessions = self
            .session_priced_upper_bytes
            .checked_mul(session_count)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "DeepSeek V4 multi-session priced byte total overflow".into(),
                )
            })?;
        self.residency_priced_upper_bytes
            .checked_add(sessions)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "DeepSeek V4 residency plus multi-session byte total overflow".into(),
                )
            })
    }

    pub fn admission_for_sessions(
        &self,
        signals: MetalMemorySignals,
        session_count: usize,
    ) -> Result<MetalMemoryAdmission, DeepSeekV4MetalError> {
        Ok(evaluate_metal_memory_admission(
            self.priced_upper_bytes_for_sessions(session_count)?,
            DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES,
            signals,
            true,
        ))
    }

    pub fn required_with_reserve_bytes(&self) -> Result<u64, DeepSeekV4MetalError> {
        self.required_with_reserve_bytes_for_sessions(1)
    }

    pub fn required_with_reserve_bytes_for_sessions(
        &self,
        session_count: usize,
    ) -> Result<u64, DeepSeekV4MetalError> {
        self.priced_upper_bytes_for_sessions(session_count)?
            .checked_add(DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "DeepSeek V4 priced memory plus dynamic reserve overflow".into(),
                )
            })
    }

    fn observed_delta(before_residency_bytes: u64, observed_bytes: u64) -> u64 {
        observed_bytes.saturating_sub(before_residency_bytes)
    }

    pub fn reconcile_residency(
        &self,
        before_residency_bytes: u64,
        after_residency_bytes: u64,
    ) -> Result<u64, DeepSeekV4MetalError> {
        let observed = Self::observed_delta(before_residency_bytes, after_residency_bytes);
        let limit = self.residency_priced_upper_bytes;
        if observed > limit {
            return invalid(format!(
                "observed DeepSeek V4 residency delta {observed} exceeds priced residency {limit}"
            ));
        }
        Ok(observed)
    }

    pub fn reconcile_session(
        &self,
        before_residency_bytes: u64,
        after_residency_bytes: u64,
        after_session_bytes: u64,
    ) -> Result<u64, DeepSeekV4MetalError> {
        let observed_total = Self::observed_delta(before_residency_bytes, after_session_bytes);
        let observed_session = after_session_bytes.saturating_sub(after_residency_bytes);
        if observed_total > self.total_priced_upper_bytes {
            return invalid(format!(
                "observed DeepSeek V4 session total {observed_total} exceeds priced model plus session {}",
                self.total_priced_upper_bytes
            ));
        }
        if observed_session > self.session_priced_upper_bytes {
            return invalid(format!(
                "observed DeepSeek V4 session increment {observed_session} exceeds priced session inventory {}",
                self.session_priced_upper_bytes
            ));
        }
        Ok(observed_total)
    }

    pub fn reconcile_first_forward(
        &self,
        before_residency_bytes: u64,
        after_first_forward_bytes: u64,
    ) -> Result<u64, DeepSeekV4MetalError> {
        self.reconcile_total_phase(
            before_residency_bytes,
            after_first_forward_bytes,
            "first forward",
        )
    }

    fn reconcile_total_phase(
        &self,
        before_residency_bytes: u64,
        observed_bytes: u64,
        phase: &str,
    ) -> Result<u64, DeepSeekV4MetalError> {
        let observed = Self::observed_delta(before_residency_bytes, observed_bytes);
        let limit = self.required_with_reserve_bytes()?;
        if observed > limit {
            return invalid(format!(
                "observed DeepSeek V4 {phase} delta {observed} exceeds planned total plus reserve {limit}"
            ));
        }
        Ok(observed)
    }

    pub fn reconcile(
        &self,
        samples: DeepSeekV4MemorySamples,
    ) -> Result<DeepSeekV4MemoryReconciliation, DeepSeekV4MetalError> {
        let observed_residency_delta_bytes = self.reconcile_residency(
            samples.before_residency_bytes,
            samples.after_residency_bytes,
        )?;
        let observed_session_delta_bytes = self.reconcile_session(
            samples.before_residency_bytes,
            samples.after_residency_bytes,
            samples.after_session_bytes,
        )?;
        let observed_first_forward_delta_bytes = self.reconcile_first_forward(
            samples.before_residency_bytes,
            samples.after_first_forward_bytes,
        )?;
        let total_limit = self.required_with_reserve_bytes()?;
        let sampled_peak_bytes = samples
            .after_residency_bytes
            .max(samples.after_session_bytes)
            .max(samples.after_first_forward_bytes);
        let sampled_peak_delta_bytes =
            sampled_peak_bytes.saturating_sub(samples.before_residency_bytes);
        Ok(DeepSeekV4MemoryReconciliation {
            samples,
            observed_residency_delta_bytes,
            observed_session_delta_bytes,
            observed_first_forward_delta_bytes,
            sampled_peak_delta_bytes,
            planned_limit_bytes: total_limit,
        })
    }
}

impl fmt::Display for DeepSeekV4MemoryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "residency_buffers={} residency_logical={} residency_priced={} session_buffers={} session_logical={} session_priced={} total_priced={} reserve={} required={}",
            self.residency_buffer_count,
            self.residency_logical_bytes,
            self.residency_priced_upper_bytes,
            self.session_allocations.len(),
            self.session_logical_bytes,
            self.session_priced_upper_bytes,
            self.total_priced_upper_bytes,
            DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES,
            self.required_with_reserve_bytes().unwrap_or(u64::MAX),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MemorySamples {
    pub before_residency_bytes: u64,
    pub after_residency_bytes: u64,
    pub after_session_bytes: u64,
    pub after_first_forward_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MemoryReconciliation {
    pub samples: DeepSeekV4MemorySamples,
    pub observed_residency_delta_bytes: u64,
    pub observed_session_delta_bytes: u64,
    pub observed_first_forward_delta_bytes: u64,
    pub sampled_peak_delta_bytes: u64,
    pub planned_limit_bytes: u64,
}

impl fmt::Display for DeepSeekV4MemoryReconciliation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "before={} after_residency={} after_session={} after_first_forward={} residency_delta={} session_delta={} first_forward_delta={} sampled_peak_delta={} planned_limit={}",
            self.samples.before_residency_bytes,
            self.samples.after_residency_bytes,
            self.samples.after_session_bytes,
            self.samples.after_first_forward_bytes,
            self.observed_residency_delta_bytes,
            self.observed_session_delta_bytes,
            self.observed_first_forward_delta_bytes,
            self.sampled_peak_delta_bytes,
            self.planned_limit_bytes,
        )
    }
}

pub struct DeepSeekV4MetalLoadPlan {
    config: DeepSeekV4Config,
    session_capacity: DeepSeekV4SessionCapacity,
    retained: RetainedStoragePlan,
    descriptors: Vec<DeepSeekV4DescriptorFingerprint>,
    report: DeepSeekV4ResidencyReport,
    memory: DeepSeekV4MemoryPlan,
    device_registry_id: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DeepSeekV4DescriptorFingerprint {
    name: String,
    shape: Vec<u64>,
    dtype: GgmlType,
    shard_idx: usize,
    data_offset: u64,
    n_bytes: u64,
}

impl From<&crate::tensor::TensorDesc> for DeepSeekV4DescriptorFingerprint {
    fn from(desc: &crate::tensor::TensorDesc) -> Self {
        Self {
            name: desc.name.clone(),
            shape: desc.shape.clone(),
            dtype: desc.dtype,
            shard_idx: desc.shard_idx,
            data_offset: desc.data_offset,
            n_bytes: desc.n_bytes,
        }
    }
}

impl DeepSeekV4MetalLoadPlan {
    pub fn config(&self) -> &DeepSeekV4Config {
        &self.config
    }

    pub fn residency_report(&self) -> &DeepSeekV4ResidencyReport {
        &self.report
    }

    pub fn memory_plan(&self) -> &DeepSeekV4MemoryPlan {
        &self.memory
    }

    pub fn session_capacity(&self) -> DeepSeekV4SessionCapacity {
        self.session_capacity
    }

    pub fn admit(
        self,
        signals: MetalMemorySignals,
    ) -> Result<DeepSeekV4AdmittedLoadPlan, DeepSeekV4MetalError> {
        self.admit_for_sessions(signals, 1)
    }

    pub fn admit_for_sessions(
        self,
        signals: MetalMemorySignals,
        session_count: usize,
    ) -> Result<DeepSeekV4AdmittedLoadPlan, DeepSeekV4MetalError> {
        let admission = self.memory.admission_for_sessions(signals, session_count)?;
        if !admission.admitted {
            return invalid(format!(
                "DeepSeek V4 {session_count}-session memory admission denied before Metal residency: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                admission.reason.as_str(),
                admission.required_bytes,
                admission.working_set_headroom_bytes,
                admission.signals.process_limit_remaining_bytes,
            ));
        }
        Ok(DeepSeekV4AdmittedLoadPlan {
            plan: self,
            admission,
            session_count,
        })
    }
}

pub struct DeepSeekV4AdmittedLoadPlan {
    plan: DeepSeekV4MetalLoadPlan,
    admission: MetalMemoryAdmission,
    session_count: usize,
}

impl DeepSeekV4AdmittedLoadPlan {
    pub fn admission(&self) -> MetalMemoryAdmission {
        self.admission
    }

    pub fn memory_plan(&self) -> &DeepSeekV4MemoryPlan {
        &self.plan.memory
    }

    pub fn session_count(&self) -> usize {
        self.session_count
    }

    pub fn residency_report(&self) -> &DeepSeekV4ResidencyReport {
        &self.plan.report
    }
}

pub struct DeepSeekV4RealizedLoad {
    residency: DeepSeekV4MetalResidency,
    admission: MetalMemoryAdmission,
    after_residency_bytes: u64,
}

impl DeepSeekV4RealizedLoad {
    pub fn admission(&self) -> MetalMemoryAdmission {
        self.admission
    }

    pub fn after_residency_bytes(&self) -> u64 {
        self.after_residency_bytes
    }

    pub fn into_residency(self) -> DeepSeekV4MetalResidency {
        self.residency
    }

    pub fn into_parts(self) -> (DeepSeekV4MetalResidency, MetalMemoryAdmission, u64) {
        (self.residency, self.admission, self.after_residency_bytes)
    }
}

/// Exact, read-only Metal realization of every tensor in a strict DeepSeek V4
/// Flash-0731 GGUF binding.
pub struct DeepSeekV4MetalResidency {
    config: DeepSeekV4Config,
    session_capacity: DeepSeekV4SessionCapacity,
    _residency_set: Option<DeepSeekV4ResidencySetGuard>,
    tensors: BTreeMap<String, MetalTensor>,
    report: DeepSeekV4ResidencyReport,
    device_registry_id: u64,
}

// Metal resources are device-wide and explicitly safe to encode from multiple
// host threads. This type exposes only immutable tensor access after
// realization; the optional residency set is mutated only during construction
// and final Drop, after the last Arc owner is gone. objc2-metal does not encode
// those framework guarantees in its protocol-object auto traits.
unsafe impl Send for DeepSeekV4MetalResidency {}
unsafe impl Sync for DeepSeekV4MetalResidency {}

struct DeepSeekV4ResidencySetGuard {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

impl Drop for DeepSeekV4ResidencySetGuard {
    fn drop(&mut self) {
        let started = std::time::Instant::now();
        let allocations = self.set.allocationCount();
        let committed_allocation_bytes = self.set.allocatedSize();
        self.queue.removeResidencySet(&self.set);
        self.set.endResidency();
        let _ = std::io::Write::write_fmt(
            &mut std::io::stderr().lock(),
            format_args!(
                "deepseek_v4: model residency set API teardown returned allocations={allocations} committed_allocation_bytes={committed_allocation_bytes} elapsed_ms={:.3}\n",
                started.elapsed().as_secs_f64() * 1e3,
            ),
        );
    }
}

// A hard-killed process can leave very large residency sets wired in the
// Metal driver until reboot. Keep whole-model pinning explicit until every
// supported process supervisor provides a teardown window long enough for
// `endResidency` to complete.
crate::env_flag!(
    default_off deepseek_v4_residency_set_enabled,
    "QWEN_DSV4_RESIDENCY_SET"
);

fn buffer_as_allocation(
    buffer: &ProtocolObject<dyn MTLBuffer>,
) -> &ProtocolObject<dyn MTLAllocation> {
    ProtocolObject::from_ref(buffer)
}

fn deepseek_v4_residency_set_scope_qualified(
    enabled: bool,
    device_name: &str,
    layer_count: u32,
    expert_count: u32,
    report: &DeepSeekV4ResidencyReport,
) -> bool {
    enabled
        && device_name == "Apple M4 Max"
        && layer_count as usize == DEEPSEEK_V4_LAYER_COUNT
        && report.tensor_count == DEEPSEEK_V4_FLASH_0731_TENSOR_COUNT
        && ((expert_count == 160 && report.source_bytes == DEEPSEEK_V4_REAP_K160_SOURCE_BYTES)
            || (expert_count == 216 && report.source_bytes == DEEPSEEK_V4_REAP_K216_SOURCE_BYTES)
            || (expert_count == 256 && report.source_bytes == DEEPSEEK_V4_FRESH_SOURCE_BYTES))
}

fn create_deepseek_v4_residency_set(
    ctx: &MetalContext,
    tensors: &BTreeMap<String, MetalTensor>,
    config: &DeepSeekV4Config,
    report: &DeepSeekV4ResidencyReport,
) -> Option<DeepSeekV4ResidencySetGuard> {
    if !deepseek_v4_residency_set_scope_qualified(
        deepseek_v4_residency_set_enabled(),
        &ctx.device.name().to_string(),
        config.layer_count,
        config.expert_count,
        report,
    ) {
        return None;
    }
    let mut seen = HashSet::new();
    let mut buffers = Vec::new();
    for tensor in tensors.values() {
        let ptr = Retained::as_ptr(&tensor.buffer) as *const _ as usize;
        if seen.insert(ptr) {
            buffers.push(&*tensor.buffer);
        }
    }
    if buffers.is_empty() {
        eprintln!("deepseek_v4: model residency set skipped because no buffers were realized");
        return None;
    }

    let descriptor = MTLResidencySetDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str("qwen-dsv4-model-weights")));
    // SAFETY: initialCapacity is advisory and equals the unique allocation count.
    unsafe { descriptor.setInitialCapacity(buffers.len()) };
    let set = match ctx.device.newResidencySetWithDescriptor_error(&descriptor) {
        Ok(set) => set,
        Err(error) => {
            let error: Retained<NSError> = error;
            eprintln!(
                "deepseek_v4: model residency set unavailable; continuing without it: {}",
                error.localizedDescription()
            );
            return None;
        }
    };
    for buffer in buffers {
        set.addAllocation(buffer_as_allocation(buffer));
    }
    set.commit();
    set.requestResidency();
    ctx.queue.addResidencySet(&set);
    eprintln!(
        "deepseek_v4: model residency set active allocations={} committed_allocation_bytes={} device_registry_id={} opt_in=QWEN_DSV4_RESIDENCY_SET=1 warning=forced_termination_may_strand_wired_memory",
        set.allocationCount(),
        set.allocatedSize(),
        ctx.device.registryID(),
    );
    Some(DeepSeekV4ResidencySetGuard {
        queue: ctx.queue.clone(),
        set,
    })
}

impl DeepSeekV4MetalResidency {
    fn validate_context(&self, ctx: &MetalContext) -> Result<(), DeepSeekV4MetalError> {
        if self.device_registry_id != ctx.device.registryID() {
            return invalid(format!(
                "DeepSeek V4 residency belongs to Metal device registry {}, context is {}",
                self.device_registry_id,
                ctx.device.registryID()
            ));
        }
        if let Some(guard) = &self._residency_set {
            let required_queue = Retained::as_ptr(&guard.queue).cast::<()>() as usize;
            let context_queue = Retained::as_ptr(&ctx.queue).cast::<()>() as usize;
            if required_queue != context_queue {
                return invalid(
                    "DeepSeek V4 residency set is attached to a different Metal command queue",
                );
            }
        }
        Ok(())
    }

    pub fn plan(
        ctx: &MetalContext,
        gguf: &GgufFile,
    ) -> Result<DeepSeekV4MetalLoadPlan, DeepSeekV4MetalError> {
        Self::plan_for_forward_limit(ctx, gguf, DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY)
    }

    pub fn plan_for_forward_limit(
        ctx: &MetalContext,
        gguf: &GgufFile,
        forward_limit: usize,
    ) -> Result<DeepSeekV4MetalLoadPlan, DeepSeekV4MetalError> {
        let model = DeepSeekV4Model::from_gguf_flash_0731(gguf)?;
        validate_strict_binding(gguf, &model)?;
        validate_session_config(&model.config)?;
        validate_session_lookup_storage(gguf)?;
        let session_capacity = DeepSeekV4SessionCapacity::for_forward_limit(
            forward_limit,
            model.config.context_length,
        )?;

        let requests = gguf.tensors.iter().collect::<Vec<_>>();
        let retained = plan_retained_storage(
            &gguf.shard_mapped_lengths(),
            &requests,
            host_page_size_bytes()?,
            ctx.max_buffer_length(),
            GGUF_BINDING_ALIGNMENT,
        )?;
        validate_fallback_policy(&retained)?;
        let report = report_for_plan(&retained)?;
        let memory = build_memory_plan(ctx, &retained, &report, &model.config, session_capacity)?;
        let descriptors = gguf
            .tensors
            .iter()
            .map(DeepSeekV4DescriptorFingerprint::from)
            .collect();
        Ok(DeepSeekV4MetalLoadPlan {
            config: model.config,
            session_capacity,
            retained,
            descriptors,
            report,
            memory,
            device_registry_id: ctx.device.registryID(),
        })
    }

    pub fn load_from_plan(
        ctx: &MetalContext,
        gguf: &GgufFile,
        admitted: DeepSeekV4AdmittedLoadPlan,
    ) -> Result<DeepSeekV4RealizedLoad, DeepSeekV4MetalError> {
        let admitted_session_count = admitted.session_count;
        let plan = admitted.plan;
        if plan.device_registry_id != ctx.device.registryID() {
            return invalid(format!(
                "DeepSeek V4 load plan belongs to Metal device registry {}, load context is {}",
                plan.device_registry_id,
                ctx.device.registryID()
            ));
        }
        let model = DeepSeekV4Model::from_gguf_flash_0731(gguf)?;
        validate_strict_binding(gguf, &model)?;
        if model.config != plan.config {
            return invalid("DeepSeek V4 load-plan configuration differs from the GGUF binding");
        }
        validate_descriptor_fingerprints(&gguf.tensors, &plan.descriptors)?;
        validate_fallback_policy(&plan.retained)?;
        validate_retained_plan_against_gguf(ctx, gguf, &plan.retained)?;
        if report_for_plan(&plan.retained)? != plan.report {
            return invalid("DeepSeek V4 retained-plan report changed before realization");
        }
        if build_memory_plan(
            ctx,
            &plan.retained,
            &plan.report,
            &plan.config,
            plan.session_capacity,
        )? != plan.memory
        {
            return invalid("DeepSeek V4 memory plan changed before realization");
        }
        let _allocation_transaction = ctx.begin_allocation_transaction();
        let refreshed_admission = plan
            .memory
            .admission_for_sessions(ctx.memory_signals(), admitted_session_count)?;
        if !refreshed_admission.admitted {
            return invalid(format!(
                "DeepSeek V4 {admitted_session_count}-session memory admission changed before realization: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                refreshed_admission.reason.as_str(),
                refreshed_admission.required_bytes,
                refreshed_admission.working_set_headroom_bytes,
                refreshed_admission.signals.process_limit_remaining_bytes,
            ));
        }
        let windows = realize_windows(ctx, gguf, &plan.retained)?;
        let tensors = realize_tensors(ctx, gguf, &plan.retained, &windows)?;
        validate_realization(gguf, &tensors, &plan.report)?;
        let residency_set =
            create_deepseek_v4_residency_set(ctx, &tensors, &plan.config, &plan.report);
        let after_residency_bytes = ctx.current_allocated_size();
        plan.memory.reconcile_residency(
            refreshed_admission.signals.current_allocated_bytes,
            after_residency_bytes,
        )?;

        Ok(DeepSeekV4RealizedLoad {
            residency: Self {
                config: plan.config,
                session_capacity: plan.session_capacity,
                _residency_set: residency_set,
                tensors,
                report: plan.report,
                device_registry_id: ctx.device.registryID(),
            },
            admission: refreshed_admission,
            after_residency_bytes,
        })
    }

    pub fn config(&self) -> &DeepSeekV4Config {
        &self.config
    }

    pub fn session_capacity(&self) -> DeepSeekV4SessionCapacity {
        self.session_capacity
    }

    pub fn tensor(&self, name: &str) -> Option<&MetalTensor> {
        self.tensors.get(name)
    }

    /// Look up an exact GGUF tensor name without silently accepting aliases.
    pub fn require_tensor(&self, name: &str) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.tensor(name).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "strict DeepSeek V4 residency is missing required tensor {name:?}"
            ))
        })
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = (&str, &MetalTensor)> {
        self.tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn report(&self) -> &DeepSeekV4ResidencyReport {
        &self.report
    }

    pub fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }
}

const DEEPSEEK_V4_HIDDEN_SIZE: usize = 4_096;
const DEEPSEEK_V4_VOCAB_SIZE: usize = 129_280;
const DEEPSEEK_V4_LAYER_COUNT: usize = 43;
const DEEPSEEK_V4_LOCAL_WINDOW: usize = 128;
const DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS: usize = 256;
const DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS: usize = DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS * 3;
const DEEPSEEK_V4_HCA_HISTORY_CAPACITY_ROWS: usize = DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS * 2;
const DEEPSEEK_V4_CSA_TOP_K: usize = 512;
const DEEPSEEK_V4_HCA_TILE_ROWS: usize = 512;

fn rounded_history_capacity(
    required_rows: usize,
    floor_rows: usize,
    name: &str,
) -> Result<usize, DeepSeekV4MetalError> {
    if floor_rows == 0 || !floor_rows.is_multiple_of(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS) {
        return invalid(format!(
            "DeepSeek V4 {name} history floor {floor_rows} is not a nonzero slab multiple"
        ));
    }
    let slabs = required_rows
        .checked_add(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS - 1)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "DeepSeek V4 {name} history row rounding overflow"
            ))
        })?
        / DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS;
    let rounded = slabs
        .checked_mul(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("DeepSeek V4 {name} history capacity overflow"))
        })?;
    Ok(rounded.max(floor_rows))
}

/// Numerical cache contract used by a native DeepSeek V4 execution path.
///
/// The maintained b10222 oracle leaves llama.cpp's K-cache type at its default
/// F16. The mixed contract remains available to the isolated position-zero
/// attention differential; it must not be silently substituted for an F16
/// continuing session because the second token observes the stored first row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4AttentionCacheContract {
    LlamaCppB10222F16,
    MixedFp8NopeBf16RopeOracle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4IndexerContract {
    LlamaCppB10222F16HadamardV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeepSeekV4SessionPhase {
    ReadyWithoutObservation { next_position: u32 },
    ReadyWithObservation { next_position: u32 },
    Poisoned { next_position: u32 },
}

impl DeepSeekV4SessionPhase {
    fn fresh() -> Self {
        Self::ReadyWithoutObservation { next_position: 0 }
    }

    fn next_position(self) -> u32 {
        match self {
            Self::ReadyWithoutObservation { next_position }
            | Self::ReadyWithObservation { next_position }
            | Self::Poisoned { next_position } => next_position,
        }
    }

    fn ready_position(self) -> Result<u32, DeepSeekV4MetalError> {
        match self {
            Self::ReadyWithoutObservation { next_position }
            | Self::ReadyWithObservation { next_position } => Ok(next_position),
            Self::Poisoned { .. } => {
                invalid("DeepSeek V4 session is poisoned by an incomplete token")
            }
        }
    }

    fn observation_valid(self) -> bool {
        matches!(self, Self::ReadyWithObservation { .. })
    }

    fn begin_mutation(&mut self) -> Result<u32, DeepSeekV4MetalError> {
        let next_position = self.ready_position()?;
        *self = Self::Poisoned { next_position };
        Ok(next_position)
    }

    fn complete_mutation(
        &mut self,
        start_position: u32,
        next_position: u32,
        publish_observation: bool,
    ) -> Result<(), DeepSeekV4MetalError> {
        if next_position <= start_position {
            return invalid("DeepSeek V4 mutation did not advance the session position");
        }
        match *self {
            Self::Poisoned {
                next_position: poisoned_position,
            } if poisoned_position == start_position => {
                *self = if publish_observation {
                    Self::ReadyWithObservation { next_position }
                } else {
                    Self::ReadyWithoutObservation { next_position }
                };
                Ok(())
            }
            _ => invalid("DeepSeek V4 mutation completed from an invalid session phase"),
        }
    }

    fn begin_restore(&mut self) -> Result<u32, DeepSeekV4MetalError> {
        let next_position = self.ready_position()?;
        *self = Self::Poisoned { next_position };
        Ok(next_position)
    }

    fn complete_restore(
        &mut self,
        replaced_position: u32,
        restored_position: u32,
    ) -> Result<(), DeepSeekV4MetalError> {
        match *self {
            Self::Poisoned { next_position } if next_position == replaced_position => {
                *self = Self::ReadyWithoutObservation {
                    next_position: restored_position,
                };
                Ok(())
            }
            _ => invalid("DeepSeek V4 restore completed from an invalid session phase"),
        }
    }
}

/// Native qwen-owned DeepSeek V4 decode session.
///
/// Physical retained state is derived from the admitted request. The separate
/// evidence ceiling prevents successful allocation from authorizing positions
/// beyond the promoted 1,048,576-forward model-context contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub enum DeepSeekV4RoutingKind {
    Hash,
    Learned,
}

#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct DeepSeekV4LayerCommandProfile {
    pub layer: usize,
    pub attention_kind: AttentionKind,
    pub routing_kind: DeepSeekV4RoutingKind,
    pub encode_cpu_ms: f64,
    pub command_gpu_ms: f64,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct DeepSeekV4CommandProfile {
    pub position: u32,
    pub forward_wall_ms: f64,
    pub layers: Vec<DeepSeekV4LayerCommandProfile>,
}

#[derive(Clone, Copy, Debug, Default)]
#[doc(hidden)]
pub struct DeepSeekV4WholeTokenProfile {
    pub position: u32,
    pub forward_wall_ms: f64,
    pub guards_phase_cpu_ms: f64,
    pub record_reset_cpu_ms: f64,
    pub command_encoder_create_cpu_ms: f64,
    pub encode_cpu_ms: f64,
    pub commit_cpu_ms: f64,
    pub commit_wait_wall_ms: f64,
    pub command_gpu_start_seconds: f64,
    pub command_gpu_end_seconds: f64,
    pub command_gpu_ms: f64,
    pub command_status_cpu_ms: f64,
    pub record_read_cpu_ms: f64,
    pub record_validate_callback_cpu_ms: f64,
    pub causal_commit_cpu_ms: f64,
}

impl DeepSeekV4WholeTokenProfile {
    pub fn wait_residual_ms(self) -> f64 {
        self.commit_wait_wall_ms - self.command_gpu_ms
    }

    pub fn outside_gpu_ms(self) -> f64 {
        self.forward_wall_ms - self.command_gpu_ms
    }

    pub fn accounted_outside_gpu_ms(self) -> f64 {
        self.guards_phase_cpu_ms
            + self.record_reset_cpu_ms
            + self.command_encoder_create_cpu_ms
            + self.encode_cpu_ms
            + self.wait_residual_ms()
            + self.command_status_cpu_ms
            + self.record_read_cpu_ms
            + self.record_validate_callback_cpu_ms
            + self.causal_commit_cpu_ms
    }

    pub fn reconstruction_residual_ms(self) -> f64 {
        self.outside_gpu_ms() - self.accounted_outside_gpu_ms()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub enum DeepSeekV4StageKind {
    AttentionHyperConnection,
    AttentionPrepare,
    AttentionCore,
    AttentionOutput,
    HyperConnectionBridge,
    MoeRouter,
    MoeRoutedExperts,
    MoeSharedExpert,
    MoeCombine,
    LayerTail,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct DeepSeekV4StageTiming {
    pub kind: DeepSeekV4StageKind,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub duration_ticks: u64,
    pub duration_ms_scaled: f64,
    pub fraction_of_layer_gpu: f64,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct DeepSeekV4SampledLayerProfile {
    pub layer: usize,
    pub command_gpu_ms: f64,
    pub sampled_span_ticks: u64,
    pub raw_span_ms_assuming_ns: f64,
    pub raw_coverage_assuming_ns: f64,
    pub encoder_boundary_ms_scaled: f64,
    pub stages: Vec<DeepSeekV4StageTiming>,
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct DeepSeekV4StageProfile {
    pub position: u32,
    pub forward_wall_ms: f64,
    pub layers: Vec<DeepSeekV4LayerCommandProfile>,
    pub sampled_layers: Vec<DeepSeekV4SampledLayerProfile>,
}

#[derive(Clone, Copy, Debug)]
struct DeepSeekV4PendingStageSample {
    layer: usize,
    kind: DeepSeekV4StageKind,
    start_sample: usize,
    end_sample: usize,
}

const DEEPSEEK_V4_STAGE_KINDS: [DeepSeekV4StageKind; 10] = [
    DeepSeekV4StageKind::AttentionHyperConnection,
    DeepSeekV4StageKind::AttentionPrepare,
    DeepSeekV4StageKind::AttentionCore,
    DeepSeekV4StageKind::AttentionOutput,
    DeepSeekV4StageKind::HyperConnectionBridge,
    DeepSeekV4StageKind::MoeRouter,
    DeepSeekV4StageKind::MoeRoutedExperts,
    DeepSeekV4StageKind::MoeSharedExpert,
    DeepSeekV4StageKind::MoeCombine,
    DeepSeekV4StageKind::LayerTail,
];

fn resolve_deepseek_v4_layer_stage_samples(
    layer: usize,
    records: &[DeepSeekV4PendingStageSample],
    timestamps: &[u64],
    command_gpu_ms: f64,
) -> Result<DeepSeekV4SampledLayerProfile, DeepSeekV4MetalError> {
    if records.len() != DEEPSEEK_V4_STAGE_KINDS.len() {
        return invalid(format!(
            "DeepSeek V4 sampled layer {layer} produced {} stages, expected {}",
            records.len(),
            DEEPSEEK_V4_STAGE_KINDS.len()
        ));
    }
    if !command_gpu_ms.is_finite() || command_gpu_ms <= 0.0 {
        return invalid(format!(
            "DeepSeek V4 sampled layer {layer} has invalid command GPU duration {command_gpu_ms}"
        ));
    }
    for (record, expected) in records.iter().zip(DEEPSEEK_V4_STAGE_KINDS) {
        if record.layer != layer {
            return invalid(format!(
                "DeepSeek V4 sampled layer {layer} contains a record for layer {}",
                record.layer
            ));
        }
        if record.kind != expected {
            return invalid(format!(
                "DeepSeek V4 sampled layer {layer} recorded {:?}, expected {expected:?}",
                record.kind
            ));
        }
        if record.start_sample >= timestamps.len() || record.end_sample >= timestamps.len() {
            return invalid(format!(
                "DeepSeek V4 sampled layer {layer} stage {:?} indexes samples {}..{} from {} timestamps",
                record.kind,
                record.start_sample,
                record.end_sample,
                timestamps.len()
            ));
        }
    }

    let first_timestamp = timestamps[records[0].start_sample];
    let last_timestamp = timestamps[records[records.len() - 1].end_sample];
    let sampled_span_ticks = last_timestamp.checked_sub(first_timestamp).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid(format!(
            "DeepSeek V4 sampled layer {layer} returned non-monotonic span timestamps"
        ))
    })?;
    if sampled_span_ticks == 0 {
        return invalid(format!(
            "DeepSeek V4 sampled layer {layer} returned a zero timestamp span"
        ));
    }

    let scale_ms_per_tick = command_gpu_ms / sampled_span_ticks as f64;
    let mut stage_ticks = 0u64;
    let mut boundary_ticks = 0u64;
    let mut stages = Vec::with_capacity(records.len());
    let mut previous_end = None;
    for record in records {
        let start_timestamp = timestamps[record.start_sample];
        let end_timestamp = timestamps[record.end_sample];
        let duration_ticks = end_timestamp.checked_sub(start_timestamp).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "DeepSeek V4 sampled layer {layer} stage {:?} returned inverted timestamps",
                record.kind
            ))
        })?;
        if let Some(previous_end) = previous_end {
            let gap = start_timestamp.checked_sub(previous_end).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "DeepSeek V4 sampled layer {layer} stage {:?} overlaps its predecessor",
                    record.kind
                ))
            })?;
            boundary_ticks = boundary_ticks.checked_add(gap).ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "DeepSeek V4 sampled layer {layer} encoder-gap tick total overflow"
                ))
            })?;
        }
        previous_end = Some(end_timestamp);
        stage_ticks = stage_ticks.checked_add(duration_ticks).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "DeepSeek V4 sampled layer {layer} stage tick total overflow"
            ))
        })?;
        let duration_ms_scaled = duration_ticks as f64 * scale_ms_per_tick;
        stages.push(DeepSeekV4StageTiming {
            kind: record.kind,
            start_timestamp,
            end_timestamp,
            duration_ticks,
            duration_ms_scaled,
            fraction_of_layer_gpu: duration_ms_scaled / command_gpu_ms,
        });
    }
    let accounted_ticks = stage_ticks.checked_add(boundary_ticks).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid(format!(
            "DeepSeek V4 sampled layer {layer} accounted tick total overflow"
        ))
    })?;
    if accounted_ticks != sampled_span_ticks {
        return invalid(format!(
            "DeepSeek V4 sampled layer {layer} accounts for {accounted_ticks} ticks across a {sampled_span_ticks}-tick span"
        ));
    }
    let raw_span_ms_assuming_ns = sampled_span_ticks as f64 * 1e-6;
    Ok(DeepSeekV4SampledLayerProfile {
        layer,
        command_gpu_ms,
        sampled_span_ticks,
        raw_span_ms_assuming_ns,
        raw_coverage_assuming_ns: raw_span_ms_assuming_ns / command_gpu_ms,
        encoder_boundary_ms_scaled: boundary_ticks as f64 * scale_ms_per_tick,
        stages,
    })
}

struct DeepSeekV4StageRecorder {
    samples: MetalTimestampSampleBuffer,
    sampled_layer_mask: [bool; DEEPSEEK_V4_LAYER_COUNT],
    next_sample: usize,
    records: Vec<DeepSeekV4PendingStageSample>,
    command_gpu_ms: [Option<f64>; DEEPSEEK_V4_LAYER_COUNT],
    resolved: Option<Vec<DeepSeekV4SampledLayerProfile>>,
}

impl DeepSeekV4StageRecorder {
    const STAGES_PER_LAYER: usize = DEEPSEEK_V4_STAGE_KINDS.len();

    fn new(ctx: &MetalContext, sampled_layers: &[usize]) -> Result<Self, DeepSeekV4MetalError> {
        if sampled_layers.is_empty() {
            return invalid("DeepSeek V4 stage profile requires at least one sampled layer");
        }
        let mut sampled_layer_mask = [false; DEEPSEEK_V4_LAYER_COUNT];
        for &layer in sampled_layers {
            if layer >= DEEPSEEK_V4_LAYER_COUNT {
                return invalid(format!(
                    "DeepSeek V4 stage-profile layer {layer} is outside 0..{DEEPSEEK_V4_LAYER_COUNT}"
                ));
            }
            if std::mem::replace(&mut sampled_layer_mask[layer], true) {
                return invalid(format!(
                    "DeepSeek V4 stage-profile layer {layer} was requested twice"
                ));
            }
        }
        let sample_count = sampled_layers
            .len()
            .checked_mul(Self::STAGES_PER_LAYER)
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("DeepSeek V4 stage sample count overflow".into())
            })?;
        Ok(Self {
            samples: ctx.timestamp_sample_buffer(sample_count)?,
            sampled_layer_mask,
            next_sample: 0,
            records: Vec::with_capacity(sample_count / 2),
            command_gpu_ms: [None; DEEPSEEK_V4_LAYER_COUNT],
            resolved: None,
        })
    }

    fn samples_layer(&self, layer: usize) -> bool {
        self.sampled_layer_mask[layer]
    }

    fn begin(
        &mut self,
        command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        layer: usize,
        kind: DeepSeekV4StageKind,
    ) -> Result<KernelEncoder, DeepSeekV4MetalError> {
        if !self.samples_layer(layer) {
            return invalid(format!(
                "DeepSeek V4 stage recorder was asked to sample unselected layer {layer}"
            ));
        }
        let start_sample = self.next_sample;
        let end_sample = start_sample.checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("DeepSeek V4 stage sample index overflow".into())
        })?;
        if end_sample >= self.samples.sample_count() {
            return invalid(format!(
                "DeepSeek V4 stage timestamp buffer exhausted at sample {end_sample}"
            ));
        }
        self.next_sample = end_sample + 1;
        self.records.push(DeepSeekV4PendingStageSample {
            layer,
            kind,
            start_sample,
            end_sample,
        });
        Ok(KernelEncoder::begin_sampled(
            command,
            &self.samples,
            start_sample,
            end_sample,
            false,
        ))
    }

    fn record_command_gpu_ms(&mut self, layer: usize, command_gpu_ms: f64) {
        if self.samples_layer(layer) {
            self.command_gpu_ms[layer] = Some(command_gpu_ms);
        }
    }

    fn resolve(&mut self, ctx: &MetalContext) -> Result<(), DeepSeekV4MetalError> {
        let timestamps = ctx.resolve_timestamp_samples(&self.samples, self.next_sample)?;
        let mut resolved = Vec::with_capacity(
            self.sampled_layer_mask
                .iter()
                .filter(|sampled| **sampled)
                .count(),
        );
        for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
            if !self.samples_layer(layer) {
                continue;
            }
            let records = self
                .records
                .iter()
                .filter(|record| record.layer == layer)
                .copied()
                .collect::<Vec<_>>();
            let command_gpu_ms = self.command_gpu_ms[layer].ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "DeepSeek V4 sampled layer {layer} has no command GPU duration"
                ))
            })?;
            resolved.push(resolve_deepseek_v4_layer_stage_samples(
                layer,
                &records,
                &timestamps,
                command_gpu_ms,
            )?);
        }
        self.resolved = Some(resolved);
        Ok(())
    }

    fn take_resolved(
        &mut self,
    ) -> Result<Vec<DeepSeekV4SampledLayerProfile>, DeepSeekV4MetalError> {
        self.resolved.take().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 stage samples were not resolved before forward completion".into(),
            )
        })
    }
}

struct DeepSeekV4LayerEncoder<'command, 'recorder> {
    command: &'command Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    layer: usize,
    sampled: bool,
    recorder: Option<&'recorder mut DeepSeekV4StageRecorder>,
    encoder: Option<KernelEncoder>,
}

impl<'command, 'recorder> DeepSeekV4LayerEncoder<'command, 'recorder> {
    fn begin(
        command: &'command Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        layer: usize,
        mut recorder: Option<&'recorder mut DeepSeekV4StageRecorder>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let sampled = recorder
            .as_deref()
            .is_some_and(|recorder| recorder.samples_layer(layer));
        let encoder = if sampled {
            recorder
                .as_deref_mut()
                .expect("sampled layer requires a stage recorder")
                .begin(
                    command,
                    layer,
                    DeepSeekV4StageKind::AttentionHyperConnection,
                )?
        } else {
            KernelEncoder::begin(command)
        };
        Ok(Self {
            command,
            layer,
            sampled,
            recorder,
            encoder: Some(encoder),
        })
    }

    fn current(&self) -> &KernelEncoder {
        self.encoder
            .as_ref()
            .expect("DeepSeek V4 layer encoder ended before layer completion")
    }

    fn boundary(&mut self, next: DeepSeekV4StageKind) -> Result<(), DeepSeekV4MetalError> {
        if !self.sampled {
            return Ok(());
        }
        self.encoder
            .take()
            .expect("sampled DeepSeek V4 layer encoder missing at stage boundary")
            .end();
        self.encoder = Some(
            self.recorder
                .as_deref_mut()
                .expect("sampled layer requires a stage recorder")
                .begin(self.command, self.layer, next)?,
        );
        Ok(())
    }

    fn end(mut self) {
        self.encoder
            .take()
            .expect("DeepSeek V4 layer encoder missing at completion")
            .end();
    }
}

struct DeepSeekV4EncodedLayer {
    sparse_visible_count: Option<usize>,
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeepSeekV4Fp4ScorePlan {
    F16Only,
    Fp4Only,
    Paired {
        consume: DeepSeekV4Fp4SelectionSource,
    },
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4Fp4ScorePlan {
    fn runs_f16(self) -> bool {
        matches!(self, Self::F16Only | Self::Paired { .. })
    }

    fn runs_fp4(self) -> bool {
        matches!(self, Self::Fp4Only | Self::Paired { .. })
    }

    fn consumes_fp4(self) -> bool {
        matches!(
            self,
            Self::Fp4Only
                | Self::Paired {
                    consume: DeepSeekV4Fp4SelectionSource::Fp4,
                }
        )
    }

    fn kind(self) -> DeepSeekV4Fp4ScorePlanKind {
        match self {
            Self::F16Only => DeepSeekV4Fp4ScorePlanKind::F16Only,
            Self::Fp4Only => DeepSeekV4Fp4ScorePlanKind::Fp4Only,
            Self::Paired { .. } => DeepSeekV4Fp4ScorePlanKind::Paired,
        }
    }

    fn consumed_source(self) -> DeepSeekV4Fp4SelectionSource {
        match self {
            Self::F16Only
            | Self::Paired {
                consume: DeepSeekV4Fp4SelectionSource::F16,
            } => DeepSeekV4Fp4SelectionSource::F16,
            Self::Fp4Only
            | Self::Paired {
                consume: DeepSeekV4Fp4SelectionSource::Fp4,
            } => DeepSeekV4Fp4SelectionSource::Fp4,
        }
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum DeepSeekV4Fp4SessionMode {
    #[default]
    F16Authoritative,
    PairedCounterfactual,
    Fp4OnlyExperimental,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4Fp4SessionMode {
    fn score_plan(self, audit_active: bool) -> DeepSeekV4Fp4ScorePlan {
        match (self, audit_active) {
            (Self::F16Authoritative, false) => DeepSeekV4Fp4ScorePlan::F16Only,
            (Self::F16Authoritative, true) => DeepSeekV4Fp4ScorePlan::Paired {
                consume: DeepSeekV4Fp4SelectionSource::F16,
            },
            (Self::PairedCounterfactual, _) => DeepSeekV4Fp4ScorePlan::Paired {
                consume: DeepSeekV4Fp4SelectionSource::Fp4,
            },
            (Self::Fp4OnlyExperimental, false) => DeepSeekV4Fp4ScorePlan::Fp4Only,
            (Self::Fp4OnlyExperimental, true) => DeepSeekV4Fp4ScorePlan::Paired {
                consume: DeepSeekV4Fp4SelectionSource::Fp4,
            },
        }
    }

    fn is_counterfactual(self) -> bool {
        self != Self::F16Authoritative
    }
}

impl DeepSeekV4CommandProfile {
    pub fn encode_cpu_ms(&self) -> f64 {
        self.layers.iter().map(|layer| layer.encode_cpu_ms).sum()
    }

    pub fn command_gpu_ms(&self) -> f64 {
        self.layers.iter().map(|layer| layer.command_gpu_ms).sum()
    }
}

pub struct DeepSeekV4Session {
    residency: Arc<DeepSeekV4MetalResidency>,
    capacity: DeepSeekV4SessionCapacity,
    token_id: MetalTensor,
    embedding: MetalTensor,
    residual_primary: MetalTensor,
    residual_secondary: MetalTensor,
    hyper_connection: DeepSeekV4HyperConnectionScratch,
    attention: DeepSeekV4PositionZeroAttentionScratch,
    sparse_csa: DeepSeekV4SparseCsaScratch,
    layer_selections: DeepSeekV4LayerSelectionRecords,
    raw_cache: MetalTensor,
    compressor_frontiers: DeepSeekV4CompressorFrontiers,
    moe: DeepSeekV4MoeScratch,
    layer_routes: DeepSeekV4LayerRouteRecords,
    final_hidden: MetalTensor,
    final_normalized_hidden: MetalTensor,
    logits: MetalTensor,
    prefill: prefill::DeepSeekV4PrefillScratch,
    phase: DeepSeekV4SessionPhase,
    committed_tokens: Vec<u32>,
    snapshot_model_content_id: Option<DeepSeekV4ModelContentId>,
    #[cfg(feature = "dsv4-diagnostics")]
    decision_diagnostics: diagnostics::DeepSeekV4DecisionCapture,
    #[cfg(feature = "dsv4-diagnostics")]
    fp4_shadow: DeepSeekV4Fp4ShadowScratch,
    #[cfg(feature = "dsv4-diagnostics")]
    fp4_collapsed_selections: DeepSeekV4LayerFp4SelectionRecords,
    #[cfg(feature = "dsv4-diagnostics")]
    fp4_shadow_diagnostics: diagnostics::DeepSeekV4Fp4ShadowCapture,
    #[cfg(feature = "dsv4-diagnostics")]
    fp4_selection_mode: DeepSeekV4Fp4SessionMode,
    #[cfg(feature = "dsv4-diagnostics")]
    fp4_counterfactual_trace: diagnostics::DeepSeekV4Fp4CounterfactualTrace,
    #[cfg(feature = "dsv4-diagnostics")]
    fp4_score_dispatch_ledger: Option<DeepSeekV4Fp4ScoreDispatchLedger>,
}

/// Compatibility name retained for the position-zero live differential.
pub type DeepSeekV4PositionZeroForward = DeepSeekV4Session;

pub struct DeepSeekV4SessionConstructionFailure {
    residency: Arc<DeepSeekV4MetalResidency>,
    error: DeepSeekV4MetalError,
}

impl DeepSeekV4SessionConstructionFailure {
    pub fn into_parts(self) -> (DeepSeekV4MetalResidency, DeepSeekV4MetalError) {
        let Self { residency, error } = self;
        let residency = Arc::try_unwrap(residency).unwrap_or_else(|_| {
            unreachable!("failed DeepSeek V4 session construction retained shared residency")
        });
        (residency, error)
    }

    fn into_error(self) -> DeepSeekV4MetalError {
        self.error
    }
}

impl DeepSeekV4Session {
    pub fn new(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Self::new_inner(ctx, Arc::new(residency), None)
    }

    /// Construct one sequence-private session over shared immutable weights.
    /// Mutable cache, routing, scratch, logits, and transcript state remain
    /// owned by the returned session.
    #[doc(hidden)]
    pub fn new_shared(
        ctx: &MetalContext,
        residency: Arc<DeepSeekV4MetalResidency>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Self::new_shared_inner(ctx, residency, None)
    }

    /// Construct a sequence-private shared session whose snapshots are scoped
    /// by a caller-provided identity. Transient in-process users may bind an
    /// ephemeral identity; durable users must bind the full model-content ID.
    #[doc(hidden)]
    pub fn new_shared_with_model_content_id(
        ctx: &MetalContext,
        residency: Arc<DeepSeekV4MetalResidency>,
        model_content_id: DeepSeekV4ModelContentId,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Self::new_shared_inner(ctx, residency, Some(model_content_id))
    }

    fn new_shared_inner(
        ctx: &MetalContext,
        residency: Arc<DeepSeekV4MetalResidency>,
        model_content_id: Option<DeepSeekV4ModelContentId>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if residency._residency_set.is_some() {
            return invalid(
                "shared DeepSeek V4 sessions require QWEN_DSV4_RESIDENCY_SET=0 because residency sets are command-queue scoped",
            );
        }
        Self::new_inner(ctx, residency, model_content_id)
    }

    pub fn new_with_model_content_id(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        model_content_id: DeepSeekV4ModelContentId,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Self::new_with_model_content_id_recoverable(ctx, residency, model_content_id)
            .map_err(DeepSeekV4SessionConstructionFailure::into_error)
    }

    /// Construct an exclusively-owned session without losing residency when
    /// session scratch allocation or validation fails.
    pub fn new_with_model_content_id_recoverable(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        model_content_id: DeepSeekV4ModelContentId,
    ) -> Result<Self, DeepSeekV4SessionConstructionFailure> {
        let residency = Arc::new(residency);
        match Self::new_inner(ctx, Arc::clone(&residency), Some(model_content_id)) {
            Ok(session) => {
                drop(residency);
                Ok(session)
            }
            Err(error) => Err(DeepSeekV4SessionConstructionFailure { residency, error }),
        }
    }

    fn new_inner(
        ctx: &MetalContext,
        residency: Arc<DeepSeekV4MetalResidency>,
        snapshot_model_content_id: Option<DeepSeekV4ModelContentId>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        residency.validate_context(ctx)?;
        validate_session_config(residency.config())?;
        for name in session_required_tensor_names(residency.config()) {
            residency.require_tensor(&name)?;
        }
        validate_session_lookup_dtypes(
            residency.require_tensor("token_embd.weight")?.dtype,
            residency.require_tensor("output.weight")?.dtype,
        )?;

        let token_id =
            MetalTensor::from_bytes(ctx, bytemuck::bytes_of(&0_i32), vec![1], GgmlType::I32)?;
        let residual_shape = vec![
            DEEPSEEK_V4_HIDDEN_SIZE as u64,
            DEEPSEEK_V4_CONNECTION_COUNT as u64,
        ];
        let attention_config = deepseek_v4_session_attention_config();
        let moe_config = deepseek_v4_session_moe_config(residency.config());
        let capacity = residency.session_capacity();
        let compressor_frontiers =
            DeepSeekV4CompressorFrontiers::new(ctx, residency.config(), capacity)?;
        let mut committed_tokens = Vec::new();
        committed_tokens
            .try_reserve_exact(capacity.forward_limit())
            .map_err(|error| {
                DeepSeekV4MetalError::Invalid(format!(
                    "reserve DeepSeek V4 committed-token transcript: {error}"
                ))
            })?;

        Ok(Self {
            residency,
            capacity,
            token_id,
            embedding: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64])?,
            residual_primary: MetalTensor::zeros_f32(ctx, residual_shape.clone())?,
            residual_secondary: MetalTensor::zeros_f32(ctx, residual_shape)?,
            hyper_connection: DeepSeekV4HyperConnectionScratch::new(ctx, DEEPSEEK_V4_HIDDEN_SIZE)?,
            attention: DeepSeekV4PositionZeroAttentionScratch::new(ctx, attention_config)?,
            sparse_csa: DeepSeekV4SparseCsaScratch::new(ctx, capacity.csa_physical_rows())?,
            layer_selections: DeepSeekV4LayerSelectionRecords::new(ctx)?,
            raw_cache: MetalTensor::zeros_f16(
                ctx,
                vec![
                    attention_config.head_dim as u64,
                    DEEPSEEK_V4_LOCAL_WINDOW as u64,
                    DEEPSEEK_V4_LAYER_COUNT as u64,
                ],
            )?,
            compressor_frontiers,
            moe: DeepSeekV4MoeScratch::new(ctx, moe_config)?,
            layer_routes: DeepSeekV4LayerRouteRecords::new(ctx, moe_config)?,
            final_hidden: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64])?,
            final_normalized_hidden: MetalTensor::zeros_f32(
                ctx,
                vec![DEEPSEEK_V4_HIDDEN_SIZE as u64],
            )?,
            logits: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_VOCAB_SIZE as u64])?,
            prefill: prefill::DeepSeekV4PrefillScratch::new(
                ctx,
                capacity.csa_physical_rows(),
                moe_config.expert_count,
            )?,
            phase: DeepSeekV4SessionPhase::fresh(),
            committed_tokens,
            snapshot_model_content_id,
            #[cfg(feature = "dsv4-diagnostics")]
            decision_diagnostics: diagnostics::DeepSeekV4DecisionCapture::default(),
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_shadow: DeepSeekV4Fp4ShadowScratch::new(ctx, capacity.csa_physical_rows())?,
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_collapsed_selections: DeepSeekV4LayerFp4SelectionRecords::new(ctx)?,
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_shadow_diagnostics: diagnostics::DeepSeekV4Fp4ShadowCapture::default(),
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_selection_mode: DeepSeekV4Fp4SessionMode::default(),
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_counterfactual_trace: diagnostics::DeepSeekV4Fp4CounterfactualTrace::default(),
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_score_dispatch_ledger: None,
        })
    }

    pub fn residency(&self) -> &DeepSeekV4MetalResidency {
        self.residency.as_ref()
    }

    /// Consumes the session and returns its retained weight residency.
    ///
    /// This is the multi-request lifecycle primitive: sessions are rebuilt
    /// per request from one long-lived residency rather than reset in place.
    /// It is deliberately valid from **any** phase, including poisoned:
    /// residency tensors are immutable weight views that no session mutation
    /// path can touch, so recovering the residency from a failed session and
    /// constructing a fresh session is the sanctioned poison-recovery story.
    /// All session-owned scratch, cache, and transcript state is dropped.
    pub fn into_residency(self) -> Result<DeepSeekV4MetalResidency, DeepSeekV4MetalError> {
        match Arc::try_unwrap(self.residency) {
            Ok(residency) => Ok(residency),
            Err(_) => invalid(
                "cannot recover exclusive DeepSeek V4 residency while another shared owner exists",
            ),
        }
    }

    /// Consume a session while retaining shared immutable model ownership.
    #[doc(hidden)]
    pub fn into_shared_residency(self) -> Arc<DeepSeekV4MetalResidency> {
        self.residency
    }

    pub fn logits(&self) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        if !self.phase.observation_valid() {
            return invalid("DeepSeek V4 logits have not completed");
        }
        Ok(&self.logits)
    }

    pub fn final_normalized_hidden(&self) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        if !self.phase.observation_valid() {
            return invalid("DeepSeek V4 final normalized hidden state has not completed");
        }
        Ok(&self.final_normalized_hidden)
    }

    pub fn next_position(&self) -> u32 {
        self.phase.next_position()
    }

    pub fn committed_tokens(&self) -> &[u32] {
        &self.committed_tokens
    }

    pub fn capacity(&self) -> DeepSeekV4SessionCapacity {
        self.capacity
    }

    pub(crate) fn bound_model_content_id(&self) -> Option<DeepSeekV4ModelContentId> {
        self.snapshot_model_content_id
    }

    #[doc(hidden)]
    pub fn multigroup_selector_telemetry(&self) -> DeepSeekV4MultigroupSelectorTelemetry {
        self.sparse_csa.multigroup_selector_telemetry()
    }

    /// Explicitly seals the exact multi-group sparse selector for its measured
    /// far-context crossover band. Production already selects that band on the
    /// qualified device; this retains the diagnostics policy surface.
    #[doc(hidden)]
    pub fn enable_multigroup_selector_experiment(&mut self) -> Result<(), DeepSeekV4MetalError> {
        let position = self.phase.ready_position()?;
        if position != 0 {
            return invalid(format!(
                "DeepSeek V4 multi-group selector must be enabled at position zero, got {position}"
            ));
        }
        self.sparse_csa.enable_multigroup_selector_experiment()
    }

    #[doc(hidden)]
    pub fn disable_multigroup_selector(&mut self) -> Result<(), DeepSeekV4MetalError> {
        let position = self.phase.ready_position()?;
        if position != 0 {
            return invalid(format!(
                "DeepSeek V4 multi-group selector must be disabled at position zero, got {position}"
            ));
        }
        self.sparse_csa.disable_multigroup_selector()
    }

    /// Enables exact post-Hadamard FP4 lineage capture for a diagnostics-only
    /// observer. It must be armed before any token mutates the session so every
    /// subsequently visible row has one unambiguous pre-F16 source.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn enable_fp4_shadow_lineage(&mut self) -> Result<(), DeepSeekV4MetalError> {
        let position = self.phase.ready_position()?;
        if position != 0 {
            return invalid(format!(
                "DeepSeek V4 FP4 shadow lineage must be enabled at position zero, got {position}"
            ));
        }
        self.compressor_frontiers.enable_fp4_shadow_lineage()?;
        self.fp4_shadow_diagnostics.enable_lineage();
        Ok(())
    }

    /// Enables a diagnostics-only counterfactual that consumes FP4 selector
    /// IDs while retaining the authoritative F16 attention cache. Such a
    /// session is not compatible with snapshot-v1 export or restore.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn enable_fp4_shadow_selection_counterfactual(
        &mut self,
    ) -> Result<(), DeepSeekV4MetalError> {
        if self.fp4_selection_mode != DeepSeekV4Fp4SessionMode::F16Authoritative {
            return invalid("DeepSeek V4 FP4 score plan is already sealed");
        }
        self.enable_fp4_shadow_lineage()?;
        self.fp4_selection_mode = DeepSeekV4Fp4SessionMode::PairedCounterfactual;
        Ok(())
    }

    /// Enables lineage and seals a diagnostics-only FP4-only score plan at
    /// position zero. Use [`Self::seal_fp4_no_double_score_experiment`] after
    /// an ordinary lineage-preserving prefix instead.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn enable_fp4_no_double_score_experiment(&mut self) -> Result<(), DeepSeekV4MetalError> {
        self.enable_fp4_shadow_lineage()?;
        self.seal_fp4_no_double_score_experiment()
    }

    /// Seals a diagnostics-only FP4-only score plan for all later sparse
    /// positions. Lineage must already have been enabled at position zero.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn seal_fp4_no_double_score_experiment(&mut self) -> Result<(), DeepSeekV4MetalError> {
        self.phase.ready_position()?;
        if self.fp4_selection_mode != DeepSeekV4Fp4SessionMode::F16Authoritative {
            return invalid("DeepSeek V4 FP4 score plan is already sealed");
        }
        if !self.fp4_shadow_diagnostics.lineage_enabled() {
            return invalid(
                "DeepSeek V4 FP4 lineage must be enabled before sealing the no-double-score experiment",
            );
        }
        self.decision_diagnostics
            .ensure_no_active_capture("seal the FP4 score plan")?;
        self.fp4_shadow_diagnostics
            .ensure_no_active_capture("seal the FP4 score plan")?;
        self.fp4_selection_mode = DeepSeekV4Fp4SessionMode::Fp4OnlyExperimental;
        Ok(())
    }

    /// Arms one diagnostics-only packed or singleton FP4 comparison. Packed
    /// capture intentionally admits only a final single sparse query.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn arm_fp4_shadow_report(&mut self, position: u32) -> Result<(), DeepSeekV4MetalError> {
        self.capacity.validate_position(position)?;
        let current = self.phase.ready_position()?;
        self.fp4_shadow_diagnostics.arm(current, position)?;
        Ok(())
    }

    /// Atomically arms a singleton paired-score audit while an FP4-only
    /// experiment remains the consumed selection authority.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn arm_fp4_paired_singleton_audit(
        &mut self,
        position: u32,
    ) -> Result<(), DeepSeekV4MetalError> {
        if self.fp4_selection_mode != DeepSeekV4Fp4SessionMode::Fp4OnlyExperimental {
            return invalid("paired FP4 audit requires a sealed FP4-only experiment");
        }
        self.capacity.validate_position(position)?;
        let current = self.phase.ready_position()?;
        if position != current {
            return Err(DeepSeekV4DiagnosticsError::WrongPosition {
                expected: current,
                actual: position,
            }
            .into());
        }
        self.decision_diagnostics.validate_arm(position)?;
        self.fp4_shadow_diagnostics
            .validate_arm(current, position)?;
        self.decision_diagnostics.arm(position)?;
        self.fp4_shadow_diagnostics.arm(current, position)?;
        Ok(())
    }

    /// Takes a completed owned FP4 observer report and returns the capture to
    /// idle so a subsequent position can be armed.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn take_fp4_shadow_report(
        &mut self,
    ) -> Result<DeepSeekV4Fp4ShadowReport, DeepSeekV4MetalError> {
        Ok(self.fp4_shadow_diagnostics.take()?)
    }

    /// Takes the last successfully encoded singleton or packed CSA score
    /// schedule. The ledger records operation invocations, not GPU duration.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn take_fp4_score_dispatch_ledger(
        &mut self,
    ) -> Result<DeepSeekV4Fp4ScoreDispatchLedger, DeepSeekV4MetalError> {
        self.fp4_score_dispatch_ledger.take().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 FP4 score dispatch ledger is unavailable".into(),
            )
        })
    }

    /// Arms the feature-gated, single-use decision capture for the next sparse
    /// CSA token.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn arm_decision_transcript(&mut self, position: u32) -> Result<(), DeepSeekV4MetalError> {
        if self.fp4_selection_mode == DeepSeekV4Fp4SessionMode::Fp4OnlyExperimental {
            return invalid(
                "FP4-only sessions must arm decisions through arm_fp4_paired_singleton_audit",
            );
        }
        self.capacity.validate_position(position)?;
        if position != self.phase.ready_position()? {
            return Err(DeepSeekV4DiagnosticsError::WrongPosition {
                expected: position,
                actual: self.phase.next_position(),
            }
            .into());
        }
        self.decision_diagnostics.arm(position)?;
        Ok(())
    }

    /// Takes a complete owned transcript. Partial or duplicate takes fail closed.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn take_decision_transcript(
        &mut self,
    ) -> Result<DeepSeekV4DecisionTranscript, DeepSeekV4MetalError> {
        Ok(self.decision_diagnostics.take()?)
    }

    /// Takes a complete transcript and returns the diagnostics capture to idle
    /// for the next consecutive-token observation window.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn take_decision_transcript_and_reset(
        &mut self,
    ) -> Result<DeepSeekV4DecisionTranscript, DeepSeekV4MetalError> {
        Ok(self.decision_diagnostics.take_and_reset()?)
    }

    pub fn cache_contract(&self) -> DeepSeekV4AttentionCacheContract {
        DeepSeekV4AttentionCacheContract::LlamaCppB10222F16
    }

    pub fn indexer_contract(&self) -> DeepSeekV4IndexerContract {
        DeepSeekV4IndexerContract::LlamaCppB10222F16HadamardV1
    }

    /// Copy completed logits out of shared Metal storage.
    pub fn copy_logits_f32(&self) -> Result<Vec<f32>, DeepSeekV4MetalError> {
        host_read_f32(self.logits()?, "completed DeepSeek V4 logits")
    }

    /// Copy the completed normalized hidden state out of shared Metal storage.
    #[cfg(feature = "dsv4-diagnostics")]
    #[doc(hidden)]
    pub fn copy_final_normalized_hidden_f32(&self) -> Result<Vec<f32>, DeepSeekV4MetalError> {
        host_read_f32(
            self.final_normalized_hidden()?,
            "completed DeepSeek V4 final normalized hidden",
        )
    }

    /// Compatibility entry point for the original position-zero differential.
    pub fn forward_token_zero(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.forward_token_zero_with_progress(ctx, token_id, |_| {})
    }

    pub fn forward_token_zero_with_progress(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        layer_completed: impl FnMut(usize),
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        let next_position = self.phase.next_position();
        if next_position != 0 {
            return invalid(format!(
                "position-zero entry point requires a fresh session, next position is {}",
                next_position
            ));
        }
        self.forward_token_with_progress(ctx, token_id, layer_completed)
    }

    pub fn forward_token(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.forward_token_with_progress(ctx, token_id, |_| {})
    }

    /// Execute one complete token and retain the raw cache and every compressor
    /// frontier needed by the next position. Ordinary decode encodes every
    /// layer into one ordered serial Metal pass, then validates immutable
    /// per-layer route and sparse-selection records before publishing progress.
    pub fn forward_token_with_progress(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        layer_completed: impl FnMut(usize),
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.forward_token_with_progress_and_profile(
            ctx,
            token_id,
            layer_completed,
            None,
            None,
            None,
        )
    }

    /// Execute one singleton token while timing the retained, separately
    /// encoded one-command-per-layer comparator. The ordinary whole-token path
    /// does not allocate or sample clocks.
    #[doc(hidden)]
    pub fn forward_token_profiled(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
    ) -> Result<DeepSeekV4CommandProfile, DeepSeekV4MetalError> {
        let position = self.phase.next_position();
        let started = std::time::Instant::now();
        let mut layers = Vec::with_capacity(DEEPSEEK_V4_LAYER_COUNT);
        self.forward_token_with_progress_and_profile(
            ctx,
            token_id,
            |_| {},
            Some(&mut layers),
            None,
            None,
        )?;
        debug_assert_eq!(layers.len(), DEEPSEEK_V4_LAYER_COUNT);
        Ok(DeepSeekV4CommandProfile {
            position,
            forward_wall_ms: started.elapsed().as_secs_f64() * 1e3,
            layers,
        })
    }

    /// Time the ordinary one-command, one-encoder path without adding GPU
    /// samples, command buffers, encoders, or dispatches.
    #[doc(hidden)]
    pub fn forward_token_whole_profiled(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
    ) -> Result<DeepSeekV4WholeTokenProfile, DeepSeekV4MetalError> {
        #[cfg(feature = "dsv4-diagnostics")]
        {
            self.decision_diagnostics
                .ensure_no_active_capture("profile a whole token")?;
            self.fp4_shadow_diagnostics
                .ensure_no_active_capture("profile a whole token")?;
        }
        let mut profile = DeepSeekV4WholeTokenProfile::default();
        self.forward_token_with_progress_and_profile(
            ctx,
            token_id,
            |_| {},
            None,
            None,
            Some(&mut profile),
        )?;
        Ok(profile)
    }

    /// Execute one singleton token while sampling ten encoder-delimited stages
    /// in only the requested layers. This intentionally retains the historical
    /// per-layer command schedule as an attribution comparator.
    #[doc(hidden)]
    pub fn forward_token_stage_profiled(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        sampled_layers: &[usize],
    ) -> Result<DeepSeekV4StageProfile, DeepSeekV4MetalError> {
        let mut recorder = DeepSeekV4StageRecorder::new(ctx, sampled_layers)?;
        let position = self.phase.next_position();
        let started = std::time::Instant::now();
        let mut layers = Vec::with_capacity(DEEPSEEK_V4_LAYER_COUNT);
        self.forward_token_with_progress_and_profile(
            ctx,
            token_id,
            |_| {},
            Some(&mut layers),
            Some(&mut recorder),
            None,
        )?;
        debug_assert_eq!(layers.len(), DEEPSEEK_V4_LAYER_COUNT);
        Ok(DeepSeekV4StageProfile {
            position,
            forward_wall_ms: started.elapsed().as_secs_f64() * 1e3,
            layers,
            sampled_layers: recorder.take_resolved()?,
        })
    }

    fn forward_token_with_progress_and_profile(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        mut layer_completed: impl FnMut(usize),
        routing_profile: Option<&mut Vec<DeepSeekV4LayerCommandProfile>>,
        stage_recorder: Option<&mut DeepSeekV4StageRecorder>,
        mut whole_profile: Option<&mut DeepSeekV4WholeTokenProfile>,
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        let forward_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        let guards_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        self.residency.validate_context(ctx)?;
        let position = self.phase.ready_position()?;
        if token_id as usize >= DEEPSEEK_V4_VOCAB_SIZE {
            return invalid(format!(
                "token id {token_id} is outside vocabulary {DEEPSEEK_V4_VOCAB_SIZE}"
            ));
        }
        self.capacity.validate_position(position)?;
        let next_position = position
            .checked_add(1)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("position overflow".into()))?;
        self.validate_committed_token_append(position, 1)?;

        let begun_position = self.phase.begin_mutation()?;
        debug_assert_eq!(begun_position, position);
        host_write_i32(&self.token_id, &[token_id as i32], "DeepSeek V4 token ID")?;
        #[cfg(feature = "dsv4-diagnostics")]
        self.decision_diagnostics.begin_forward(position)?;
        #[cfg(feature = "dsv4-diagnostics")]
        self.fp4_shadow_diagnostics.begin_singleton(position)?;
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.position = position;
            profile.guards_phase_cpu_ms = guards_started
                .expect("whole-token profile requires a guards timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }
        let result = self.forward_token_inner(
            ctx,
            token_id,
            position,
            &mut layer_completed,
            routing_profile,
            stage_recorder,
            whole_profile.as_deref_mut(),
        );
        match result {
            Ok(()) => {
                let causal_commit_started =
                    whole_profile.as_ref().map(|_| std::time::Instant::now());
                self.commit_tokens(&[token_id]);
                self.phase
                    .complete_mutation(position, next_position, true)?;
                if let Some(profile) = whole_profile {
                    profile.causal_commit_cpu_ms = causal_commit_started
                        .expect("whole-token profile requires a causal-commit timer")
                        .elapsed()
                        .as_secs_f64()
                        * 1e3;
                    profile.forward_wall_ms = forward_started
                        .expect("whole-token profile requires a forward timer")
                        .elapsed()
                        .as_secs_f64()
                        * 1e3;
                }
                Ok(&self.logits)
            }
            Err(error) => Err(error),
        }
    }

    fn validate_committed_token_append(
        &self,
        start_position: u32,
        token_count: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        if self.committed_tokens.len() != start_position as usize {
            return invalid(format!(
                "DeepSeek V4 committed-token transcript has length {}, expected {start_position}",
                self.committed_tokens.len()
            ));
        }
        let end = self
            .committed_tokens
            .len()
            .checked_add(token_count)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "DeepSeek V4 committed-token transcript length overflow".into(),
                )
            })?;
        if end > self.capacity.forward_limit() {
            return invalid(format!(
                "DeepSeek V4 committed-token transcript would reach {end}, beyond capacity {}",
                self.capacity.forward_limit()
            ));
        }
        if self.committed_tokens.capacity() < end {
            return invalid(format!(
                "DeepSeek V4 committed-token transcript capacity {} cannot record {end} tokens",
                self.committed_tokens.capacity()
            ));
        }
        Ok(())
    }

    fn commit_tokens(&mut self, tokens: &[u32]) {
        debug_assert!(
            self.committed_tokens.len() + tokens.len() <= self.committed_tokens.capacity()
        );
        self.committed_tokens.extend_from_slice(tokens);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_token_layer(
        &mut self,
        ctx: &MetalContext,
        layer_encoder: &mut DeepSeekV4LayerEncoder<'_, '_>,
        token_id: u32,
        position: u32,
        layer: usize,
        route_record: &DeepSeekV4RouteRecord,
        selection_record: &DeepSeekV4SelectionRecord,
    ) -> Result<DeepSeekV4EncodedLayer, DeepSeekV4MetalError> {
        let rms_eps = self.residency.config().attention_rms_epsilon;
        let hc_eps = self.residency.config().hyper_connection_epsilon;
        let raw_cache = self.raw_cache_layer(layer)?;
        let rope = deepseek_v4_layer_rope(self.residency.config(), layer)?;

        if layer == 0 {
            encode_get_rows_f32(
                ctx,
                layer_encoder.current(),
                self.residency.require_tensor("token_embd.weight")?,
                &self.token_id,
                &self.embedding,
                1,
                DEEPSEEK_V4_HIDDEN_SIZE,
            )?;
            self.hyper_connection.encode_initial_repeat(
                ctx,
                layer_encoder.current(),
                &self.embedding,
                &self.residual_primary,
            )?;
        }

        self.hyper_connection.encode_pre(
            ctx,
            layer_encoder.current(),
            &self.residual_primary,
            self.layer_tensor(layer, "hc_attn_fn.weight")?,
            self.layer_tensor(layer, "hc_attn_scale.weight")?,
            self.layer_tensor(layer, "hc_attn_base.weight")?,
            rms_eps,
            hc_eps,
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::AttentionPrepare)?;

        self.attention.encode_prepare_local_f16(
            ctx,
            layer_encoder.current(),
            self.hyper_connection.collapsed_input(),
            self.layer_tensor(layer, "attn_norm.weight")?,
            self.layer_tensor(layer, "attn_q_a.weight")?,
            self.layer_tensor(layer, "attn_q_a_norm.weight")?,
            self.layer_tensor(layer, "attn_q_b.weight")?,
            self.layer_tensor(layer, "attn_kv.weight")?,
            self.layer_tensor(layer, "attn_kv_a_norm.weight")?,
            &raw_cache,
            position,
            rope,
            rms_eps,
        )?;
        self.compressor_frontiers.encode_layer(
            ctx,
            layer_encoder.current(),
            &self.residency,
            layer,
            position,
            self.attention.normalized_input(),
            rope,
            rms_eps,
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::AttentionCore)?;

        let csa_rows = self.compressor_frontiers.csa_rows(layer, position)?;
        let sparse_visible_count =
            if let Some(rows) = csa_rows.filter(|rows| rows.count > DEEPSEEK_V4_CSA_TOP_K) {
                #[cfg(not(feature = "dsv4-diagnostics"))]
                self.sparse_csa.encode(
                    ctx,
                    layer_encoder.current(),
                    self.attention.q_lora(),
                    self.attention.normalized_input(),
                    self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                    self.layer_tensor(layer, "indexer.proj.weight")?,
                    rows,
                    position,
                    rope,
                    selection_record,
                )?;
                #[cfg(feature = "dsv4-diagnostics")]
                let collapsed_fp4 =
                    self.fp4_selection_mode == DeepSeekV4Fp4SessionMode::Fp4OnlyExperimental;
                #[cfg(feature = "dsv4-diagnostics")]
                if collapsed_fp4 {
                    self.sparse_csa.encode_prepare(
                        ctx,
                        layer_encoder.current(),
                        self.attention.q_lora(),
                        self.attention.normalized_input(),
                        self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                        self.layer_tensor(layer, "indexer.proj.weight")?,
                        rows,
                        position,
                        rope,
                        selection_record,
                    )?;
                    let fp4_record = self.fp4_collapsed_selections.layer(layer)?;
                    self.fp4_shadow.encode_into(
                        ctx,
                        layer_encoder.current(),
                        &self.sparse_csa.index_queries,
                        &self.sparse_csa.head_weights,
                        rows,
                        &selection_record.visible_count,
                        fp4_record.output(selection_record),
                    )?;
                    self.attention.encode_selected_attention_f16(
                        ctx,
                        layer_encoder.current(),
                        &raw_cache,
                        rows,
                        fp4_record.selection_view(selection_record),
                        self.layer_tensor(layer, "attn_sinks.weight")?,
                        position,
                        rope,
                    )?;
                } else {
                    self.sparse_csa.encode(
                        ctx,
                        layer_encoder.current(),
                        self.attention.q_lora(),
                        self.attention.normalized_input(),
                        self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                        self.layer_tensor(layer, "indexer.proj.weight")?,
                        rows,
                        position,
                        rope,
                        selection_record,
                    )?;
                    self.attention.encode_selected_attention_f16(
                        ctx,
                        layer_encoder.current(),
                        &raw_cache,
                        rows,
                        self.sparse_csa.selection_view(selection_record),
                        self.layer_tensor(layer, "attn_sinks.weight")?,
                        position,
                        rope,
                    )?;
                }
                #[cfg(not(feature = "dsv4-diagnostics"))]
                self.attention.encode_selected_attention_f16(
                    ctx,
                    layer_encoder.current(),
                    &raw_cache,
                    rows,
                    self.sparse_csa.selection_view(selection_record),
                    self.layer_tensor(layer, "attn_sinks.weight")?,
                    position,
                    rope,
                )?;
                Some(rows.count)
            } else {
                let compressed = self.compressor_frontiers.attention_rows(layer, position)?;
                self.attention.encode_dense_attention_f16(
                    ctx,
                    layer_encoder.current(),
                    &raw_cache,
                    compressed,
                    self.residency.config().attention_kinds[layer],
                    self.layer_tensor(layer, "attn_sinks.weight")?,
                    position,
                    rope,
                )?;
                None
            };
        layer_encoder.boundary(DeepSeekV4StageKind::AttentionOutput)?;

        let attention_output = self.attention.encode_attention_output(
            ctx,
            layer_encoder.current(),
            self.layer_tensor(layer, "attn_output_a.weight")?,
            self.layer_tensor(layer, "attn_output_b.weight")?,
            position,
            rope,
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::HyperConnectionBridge)?;

        self.hyper_connection.encode_post(
            ctx,
            layer_encoder.current(),
            attention_output,
            &self.residual_primary,
            &self.residual_secondary,
        )?;
        self.hyper_connection.encode_pre(
            ctx,
            layer_encoder.current(),
            &self.residual_secondary,
            self.layer_tensor(layer, "hc_ffn_fn.weight")?,
            self.layer_tensor(layer, "hc_ffn_scale.weight")?,
            self.layer_tensor(layer, "hc_ffn_base.weight")?,
            rms_eps,
            hc_eps,
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::MoeRouter)?;

        self.moe.encode_router(
            ctx,
            layer_encoder.current(),
            self.hyper_connection.collapsed_input(),
            self.layer_tensor(layer, "ffn_norm.weight")?,
            self.layer_tensor(layer, "ffn_gate_inp.weight")?,
            rms_eps,
        )?;
        if layer < self.residency.config().hash_layer_count as usize {
            self.moe.encode_route_hash_gpu_into(
                ctx,
                layer_encoder.current(),
                token_id as usize,
                self.layer_tensor(layer, "ffn_gate_tid2eid.weight")?,
                route_record,
            )?;
        } else {
            self.moe.encode_route_learned_gpu_into(
                ctx,
                layer_encoder.current(),
                self.layer_tensor(layer, "exp_probs_b.bias")?,
                route_record,
            )?;
        }
        self.moe.validate_indexed_experts(
            self.layer_tensor(layer, "ffn_gate_exps.weight")?,
            self.layer_tensor(layer, "ffn_up_exps.weight")?,
            self.layer_tensor(layer, "ffn_down_exps.weight")?,
            self.layer_tensor(layer, "ffn_gate_shexp.weight")?,
            self.layer_tensor(layer, "ffn_up_shexp.weight")?,
            self.layer_tensor(layer, "ffn_down_shexp.weight")?,
            self.residency.config().swiglu_clamp_experts[layer],
            self.residency.config().swiglu_clamp_shared[layer],
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::MoeRoutedExperts)?;

        self.moe.encode_routed_experts_all_slots_from_record(
            ctx,
            layer_encoder.current(),
            self.layer_tensor(layer, "ffn_gate_exps.weight")?,
            self.layer_tensor(layer, "ffn_up_exps.weight")?,
            self.layer_tensor(layer, "ffn_down_exps.weight")?,
            self.residency.config().swiglu_clamp_experts[layer],
            route_record,
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::MoeSharedExpert)?;

        self.moe.encode_shared_expert(
            ctx,
            layer_encoder.current(),
            self.layer_tensor(layer, "ffn_gate_shexp.weight")?,
            self.layer_tensor(layer, "ffn_up_shexp.weight")?,
            self.layer_tensor(layer, "ffn_down_shexp.weight")?,
            self.residency.config().swiglu_clamp_shared[layer],
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::MoeCombine)?;

        let moe_output = self.moe.encode_expert_combine_from_record(
            ctx,
            layer_encoder.current(),
            route_record,
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::LayerTail)?;

        self.hyper_connection.encode_post(
            ctx,
            layer_encoder.current(),
            moe_output,
            &self.residual_secondary,
            &self.residual_primary,
        )?;
        if layer + 1 == DEEPSEEK_V4_LAYER_COUNT {
            self.hyper_connection.encode_head(
                ctx,
                layer_encoder.current(),
                &self.residual_primary,
                self.residency.require_tensor("output_hc_fn.weight")?,
                self.residency.require_tensor("output_hc_scale.weight")?,
                self.residency.require_tensor("output_hc_base.weight")?,
                &self.final_hidden,
                rms_eps,
                hc_eps,
            )?;
            encode_rms_norm_mul_f32(
                ctx,
                layer_encoder.current(),
                &self.final_hidden,
                self.residency.require_tensor("output_norm.weight")?,
                &self.final_normalized_hidden,
                rms_eps,
            )?;
            encode_projection(
                ctx,
                layer_encoder.current(),
                self.residency.require_tensor("output.weight")?,
                &self.final_normalized_hidden,
                &self.logits,
                DEEPSEEK_V4_HIDDEN_SIZE,
                DEEPSEEK_V4_VOCAB_SIZE,
                "output logits",
            )?;
        }
        Ok(DeepSeekV4EncodedLayer {
            sparse_visible_count,
        })
    }

    fn forward_token_inner(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        position: u32,
        layer_completed: &mut impl FnMut(usize),
        mut routing_profile: Option<&mut Vec<DeepSeekV4LayerCommandProfile>>,
        mut stage_recorder: Option<&mut DeepSeekV4StageRecorder>,
        whole_profile: Option<&mut DeepSeekV4WholeTokenProfile>,
    ) -> Result<(), DeepSeekV4MetalError> {
        #[cfg(feature = "dsv4-diagnostics")]
        let decision_capture_active = self.decision_diagnostics.is_capturing();
        #[cfg(not(feature = "dsv4-diagnostics"))]
        let decision_capture_active = false;
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_shadow_capture_active = self.fp4_shadow_diagnostics.is_capturing();
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_score_plan = self
            .fp4_selection_mode
            .score_plan(fp4_shadow_capture_active);
        #[cfg(feature = "dsv4-diagnostics")]
        if decision_capture_active && !fp4_score_plan.runs_f16() {
            return invalid("decision capture requires an F16 or paired CSA score plan");
        }
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_requires_instrumented_schedule =
            matches!(fp4_score_plan, DeepSeekV4Fp4ScorePlan::Paired { .. });
        #[cfg(feature = "dsv4-diagnostics")]
        let mut fp4_score_dispatch_ledger = DeepSeekV4Fp4ScoreDispatchLedger::new(
            DeepSeekV4Fp4ShadowExecution::Singleton,
            position,
            fp4_score_plan.kind(),
            fp4_score_plan.consumed_source(),
        );
        #[cfg(feature = "dsv4-diagnostics")]
        {
            self.fp4_score_dispatch_ledger = None;
        }
        #[cfg(not(feature = "dsv4-diagnostics"))]
        let fp4_requires_instrumented_schedule = false;
        if routing_profile.is_none()
            && stage_recorder.is_none()
            && !decision_capture_active
            && !fp4_requires_instrumented_schedule
        {
            return self.forward_token_inner_collapsed(
                ctx,
                token_id,
                position,
                layer_completed,
                whole_profile,
            );
        }
        if whole_profile.is_some() {
            return invalid(
                "whole-token profiling is incompatible with layer, stage, or decision profiling",
            );
        }

        let rms_eps = self.residency.config().attention_rms_epsilon;
        let hc_eps = self.residency.config().hyper_connection_epsilon;

        for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
            let selection_record = self.sparse_csa.default_record();
            let route_record = self.moe.default_route_record();
            let raw_cache = self.raw_cache_layer(layer)?;
            let rope = deepseek_v4_layer_rope(self.residency.config(), layer)?;
            let encode_started = routing_profile.as_ref().map(|_| std::time::Instant::now());
            let command = ctx.queue.commandBuffer().ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "failed to allocate layer {layer} command buffer"
                ))
            })?;
            let stage_sampled = stage_recorder
                .as_ref()
                .is_some_and(|recorder| recorder.samples_layer(layer));
            let mut encoder = if stage_sampled {
                stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(
                        &command,
                        layer,
                        DeepSeekV4StageKind::AttentionHyperConnection,
                    )?
            } else {
                KernelEncoder::begin(&command)
            };
            if layer == 0 {
                encode_get_rows_f32(
                    ctx,
                    &encoder,
                    self.residency.require_tensor("token_embd.weight")?,
                    &self.token_id,
                    &self.embedding,
                    1,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                )?;
                self.hyper_connection.encode_initial_repeat(
                    ctx,
                    &encoder,
                    &self.embedding,
                    &self.residual_primary,
                )?;
            }

            self.hyper_connection.encode_pre(
                ctx,
                &encoder,
                &self.residual_primary,
                self.layer_tensor(layer, "hc_attn_fn.weight")?,
                self.layer_tensor(layer, "hc_attn_scale.weight")?,
                self.layer_tensor(layer, "hc_attn_base.weight")?,
                rms_eps,
                hc_eps,
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::AttentionPrepare)?;
            }
            self.attention.encode_prepare_local_f16(
                ctx,
                &encoder,
                self.hyper_connection.collapsed_input(),
                self.layer_tensor(layer, "attn_norm.weight")?,
                self.layer_tensor(layer, "attn_q_a.weight")?,
                self.layer_tensor(layer, "attn_q_a_norm.weight")?,
                self.layer_tensor(layer, "attn_q_b.weight")?,
                self.layer_tensor(layer, "attn_kv.weight")?,
                self.layer_tensor(layer, "attn_kv_a_norm.weight")?,
                &raw_cache,
                position,
                rope,
                rms_eps,
            )?;
            self.compressor_frontiers.encode_layer(
                ctx,
                &encoder,
                &self.residency,
                layer,
                position,
                self.attention.normalized_input(),
                rope,
                rms_eps,
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::AttentionCore)?;
            }
            let csa_rows = self.compressor_frontiers.csa_rows(layer, position)?;
            if let Some(rows) = csa_rows.filter(|rows| rows.count > DEEPSEEK_V4_CSA_TOP_K) {
                #[cfg(not(feature = "dsv4-diagnostics"))]
                self.sparse_csa.encode(
                    ctx,
                    &encoder,
                    self.attention.q_lora(),
                    self.attention.normalized_input(),
                    self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                    self.layer_tensor(layer, "indexer.proj.weight")?,
                    rows,
                    position,
                    rope,
                    &selection_record,
                )?;
                #[cfg(feature = "dsv4-diagnostics")]
                {
                    self.sparse_csa.encode_prepare(
                        ctx,
                        &encoder,
                        self.attention.q_lora(),
                        self.attention.normalized_input(),
                        self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                        self.layer_tensor(layer, "indexer.proj.weight")?,
                        rows,
                        position,
                        rope,
                        &selection_record,
                    )?;
                    fp4_score_dispatch_ledger.record_common_prepare()?;
                    if fp4_score_plan.runs_f16() {
                        self.sparse_csa.encode_f16_score_and_select(
                            ctx,
                            &encoder,
                            rows,
                            &selection_record,
                        )?;
                        fp4_score_dispatch_ledger.record_f16_score_and_selector()?;
                    }
                }
                #[cfg(feature = "dsv4-diagnostics")]
                if fp4_score_plan.runs_fp4() {
                    self.fp4_shadow.encode(
                        ctx,
                        &encoder,
                        &self.sparse_csa.index_queries,
                        &self.sparse_csa.head_weights,
                        rows,
                        &selection_record.visible_count,
                    )?;
                    fp4_score_dispatch_ledger.record_fp4_pipeline()?;
                }
                #[cfg(feature = "dsv4-diagnostics")]
                let selected = if fp4_score_plan.consumes_fp4() {
                    self.fp4_shadow.selection_view()
                } else {
                    self.sparse_csa.selection_view(&selection_record)
                };
                #[cfg(not(feature = "dsv4-diagnostics"))]
                let selected = self.sparse_csa.selection_view(&selection_record);
                self.attention.encode_selected_attention_f16(
                    ctx,
                    &encoder,
                    &raw_cache,
                    rows,
                    selected,
                    self.layer_tensor(layer, "attn_sinks.weight")?,
                    position,
                    rope,
                )?;
            } else {
                let compressed = self.compressor_frontiers.attention_rows(layer, position)?;
                self.attention.encode_dense_attention_f16(
                    ctx,
                    &encoder,
                    &raw_cache,
                    compressed,
                    self.residency.config().attention_kinds[layer],
                    self.layer_tensor(layer, "attn_sinks.weight")?,
                    position,
                    rope,
                )?;
            }
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::AttentionOutput)?;
            }
            let attention_output = self.attention.encode_attention_output(
                ctx,
                &encoder,
                self.layer_tensor(layer, "attn_output_a.weight")?,
                self.layer_tensor(layer, "attn_output_b.weight")?,
                position,
                rope,
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::HyperConnectionBridge)?;
            }
            self.hyper_connection.encode_post(
                ctx,
                &encoder,
                attention_output,
                &self.residual_primary,
                &self.residual_secondary,
            )?;
            self.hyper_connection.encode_pre(
                ctx,
                &encoder,
                &self.residual_secondary,
                self.layer_tensor(layer, "hc_ffn_fn.weight")?,
                self.layer_tensor(layer, "hc_ffn_scale.weight")?,
                self.layer_tensor(layer, "hc_ffn_base.weight")?,
                rms_eps,
                hc_eps,
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::MoeRouter)?;
            }
            self.moe.encode_router(
                ctx,
                &encoder,
                self.hyper_connection.collapsed_input(),
                self.layer_tensor(layer, "ffn_norm.weight")?,
                self.layer_tensor(layer, "ffn_gate_inp.weight")?,
                rms_eps,
            )?;
            if layer < self.residency.config().hash_layer_count as usize {
                self.moe.encode_route_hash_gpu(
                    ctx,
                    &encoder,
                    token_id as usize,
                    self.layer_tensor(layer, "ffn_gate_tid2eid.weight")?,
                )?;
            } else {
                self.moe.encode_route_learned_gpu(
                    ctx,
                    &encoder,
                    self.layer_tensor(layer, "exp_probs_b.bias")?,
                )?;
            }
            self.moe.validate_indexed_experts(
                self.layer_tensor(layer, "ffn_gate_exps.weight")?,
                self.layer_tensor(layer, "ffn_up_exps.weight")?,
                self.layer_tensor(layer, "ffn_down_exps.weight")?,
                self.layer_tensor(layer, "ffn_gate_shexp.weight")?,
                self.layer_tensor(layer, "ffn_up_shexp.weight")?,
                self.layer_tensor(layer, "ffn_down_shexp.weight")?,
                self.residency.config().swiglu_clamp_experts[layer],
                self.residency.config().swiglu_clamp_shared[layer],
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::MoeRoutedExperts)?;
            }
            self.moe.encode_routed_experts_all_slots(
                ctx,
                &encoder,
                self.layer_tensor(layer, "ffn_gate_exps.weight")?,
                self.layer_tensor(layer, "ffn_up_exps.weight")?,
                self.layer_tensor(layer, "ffn_down_exps.weight")?,
                self.residency.config().swiglu_clamp_experts[layer],
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::MoeSharedExpert)?;
            }
            self.moe.encode_shared_expert(
                ctx,
                &encoder,
                self.layer_tensor(layer, "ffn_gate_shexp.weight")?,
                self.layer_tensor(layer, "ffn_up_shexp.weight")?,
                self.layer_tensor(layer, "ffn_down_shexp.weight")?,
                self.residency.config().swiglu_clamp_shared[layer],
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::MoeCombine)?;
            }
            let moe_output = self.moe.encode_expert_combine(ctx, &encoder)?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::LayerTail)?;
            }
            self.hyper_connection.encode_post(
                ctx,
                &encoder,
                moe_output,
                &self.residual_secondary,
                &self.residual_primary,
            )?;

            if layer + 1 == DEEPSEEK_V4_LAYER_COUNT {
                self.hyper_connection.encode_head(
                    ctx,
                    &encoder,
                    &self.residual_primary,
                    self.residency.require_tensor("output_hc_fn.weight")?,
                    self.residency.require_tensor("output_hc_scale.weight")?,
                    self.residency.require_tensor("output_hc_base.weight")?,
                    &self.final_hidden,
                    rms_eps,
                    hc_eps,
                )?;
                encode_rms_norm_mul_f32(
                    ctx,
                    &encoder,
                    &self.final_hidden,
                    self.residency.require_tensor("output_norm.weight")?,
                    &self.final_normalized_hidden,
                    rms_eps,
                )?;
                encode_projection(
                    ctx,
                    &encoder,
                    self.residency.require_tensor("output.weight")?,
                    &self.final_normalized_hidden,
                    &self.logits,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    DEEPSEEK_V4_VOCAB_SIZE,
                    "output logits",
                )?;
            }
            encoder.end();
            let encode_cpu_ms = encode_started
                .map(|started| started.elapsed().as_secs_f64() * 1e3)
                .unwrap_or_default();
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return invalid(format!("layer {layer} command failed: {error:?}"));
            }
            self.moe
                .validate_gpu_route_record_completed(&route_record)?;
            if self.residency.config().attention_kinds[layer] == AttentionKind::CompressedSparse
                && (position as usize + 1) / 4 > DEEPSEEK_V4_CSA_TOP_K
                && {
                    #[cfg(feature = "dsv4-diagnostics")]
                    {
                        fp4_score_plan.runs_f16()
                    }
                    #[cfg(not(feature = "dsv4-diagnostics"))]
                    {
                        true
                    }
                }
            {
                self.sparse_csa.validate_completed(&selection_record)?;
            }

            #[cfg(feature = "dsv4-diagnostics")]
            let (csa_decision, indexer_head_weights, indexer_query_norms, indexer_key_norms) =
                if self.decision_diagnostics.is_capturing()
                    && self.residency.config().attention_kinds[layer]
                        == AttentionKind::CompressedSparse
                {
                    let rows = self.compressor_frontiers.csa_rows(layer, position)?;
                    match rows.filter(|rows| rows.count > DEEPSEEK_V4_CSA_TOP_K) {
                        Some(rows) => {
                            let query_norms =
                                if self.sparse_csa.use_f16_matrix_score(ctx, rows.count) {
                                    host_l2_norms_f16_rows(
                                        &self.sparse_csa.matrix_queries_f16,
                                        64,
                                        128,
                                        "diagnostic sparse CSA F16 matrix queries",
                                    )?
                                } else {
                                    host_l2_norms_f32_rows(
                                        &self.sparse_csa.index_queries,
                                        64,
                                        128,
                                        "diagnostic sparse CSA index queries",
                                    )?
                                };
                            (
                                Some(
                                    self.sparse_csa
                                        .capture_decision(rows.count, &selection_record)?,
                                ),
                                host_read_f32(
                                    &self.sparse_csa.head_weights,
                                    "diagnostic sparse CSA head weights",
                                )?,
                                query_norms,
                                host_l2_norms_f16_rows(
                                    rows.indexer_cache,
                                    rows.count,
                                    128,
                                    "diagnostic sparse CSA index keys",
                                )?,
                            )
                        }
                        None => (None, Vec::new(), Vec::new(), Vec::new()),
                    }
                } else {
                    (None, Vec::new(), Vec::new(), Vec::new())
                };

            #[cfg(feature = "dsv4-diagnostics")]
            if self.decision_diagnostics.is_capturing() {
                let route = self.moe.capture_route_decision(&route_record)?;
                self.decision_diagnostics.capture_layer(
                    layer,
                    csa_decision,
                    indexer_head_weights,
                    indexer_query_norms,
                    indexer_key_norms,
                    route,
                )?;
            }

            #[cfg(feature = "dsv4-diagnostics")]
            if self.fp4_shadow_diagnostics.is_capturing()
                && self.residency.config().attention_kinds[layer] == AttentionKind::CompressedSparse
            {
                let rows = self
                    .compressor_frontiers
                    .csa_rows(layer, position)?
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "FP4 shadow CSA layer {layer} has no published rows"
                        ))
                    })?;
                let report = self.fp4_shadow.capture_layer(
                    layer,
                    position,
                    rows,
                    &self.sparse_csa.scores,
                    &self.sparse_csa.selected_mask,
                    &self.sparse_csa.cache_order_ids,
                    &selection_record.selected_count,
                    &selection_record.status,
                )?;
                self.fp4_shadow_diagnostics.capture_layer(report)?;
            }
            #[cfg(feature = "dsv4-diagnostics")]
            if fp4_score_plan.consumes_fp4()
                && self.residency.config().attention_kinds[layer] == AttentionKind::CompressedSparse
                && (position as usize + 1) / 4 > DEEPSEEK_V4_CSA_TOP_K
            {
                self.fp4_shadow.validate_completed()?;
                self.fp4_shadow.record_counterfactual_selection(
                    &mut self.fp4_counterfactual_trace,
                    DeepSeekV4Fp4ShadowExecution::Singleton,
                    DeepSeekV4Fp4SelectionSource::Fp4,
                    position,
                    layer,
                )?;
            }

            let command_gpu_ms = if routing_profile.is_some() || stage_sampled {
                let gpu_seconds = command.GPUEndTime() - command.GPUStartTime();
                if !gpu_seconds.is_finite() || gpu_seconds <= 0.0 {
                    return invalid(format!(
                        "layer {layer} returned invalid Metal command timestamps: ({}, {})",
                        command.GPUStartTime(),
                        command.GPUEndTime()
                    ));
                }
                Some(gpu_seconds * 1e3)
            } else {
                None
            };
            if let Some(profile) = routing_profile.as_deref_mut() {
                profile.push(DeepSeekV4LayerCommandProfile {
                    layer,
                    attention_kind: self.residency.config().attention_kinds[layer],
                    routing_kind: if layer < self.residency.config().hash_layer_count as usize {
                        DeepSeekV4RoutingKind::Hash
                    } else {
                        DeepSeekV4RoutingKind::Learned
                    },
                    encode_cpu_ms,
                    command_gpu_ms: command_gpu_ms
                        .expect("profiled command requires a GPU duration"),
                });
            }
            if stage_sampled {
                stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .record_command_gpu_ms(
                        layer,
                        command_gpu_ms.expect("sampled command requires a GPU duration"),
                    );
            }
            layer_completed(layer);
        }

        if let Some(recorder) = stage_recorder {
            recorder.resolve(ctx)?;
        }

        #[cfg(feature = "dsv4-diagnostics")]
        self.decision_diagnostics.finish()?;
        #[cfg(feature = "dsv4-diagnostics")]
        self.fp4_shadow_diagnostics.finish()?;
        #[cfg(feature = "dsv4-diagnostics")]
        {
            fp4_score_dispatch_ledger.validate_completed()?;
            self.fp4_score_dispatch_ledger = Some(fp4_score_dispatch_ledger);
        }

        Ok(())
    }

    fn forward_token_inner_collapsed(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        position: u32,
        layer_completed: &mut impl FnMut(usize),
        mut whole_profile: Option<&mut DeepSeekV4WholeTokenProfile>,
    ) -> Result<(), DeepSeekV4MetalError> {
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_score_plan = self.fp4_selection_mode.score_plan(false);
        #[cfg(feature = "dsv4-diagnostics")]
        if matches!(fp4_score_plan, DeepSeekV4Fp4ScorePlan::Paired { .. }) {
            return invalid("paired FP4 scoring requires the instrumented singleton schedule");
        }
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_collapsed_active = fp4_score_plan == DeepSeekV4Fp4ScorePlan::Fp4Only;
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_execution = DeepSeekV4Fp4ShadowExecution::SingletonCollapsed;
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_source = fp4_score_plan.consumed_source();
        #[cfg(feature = "dsv4-diagnostics")]
        let mut fp4_score_dispatch_ledger = fp4_collapsed_active.then(|| {
            DeepSeekV4Fp4ScoreDispatchLedger::new(
                fp4_execution,
                position,
                fp4_score_plan.kind(),
                fp4_source,
            )
        });
        #[cfg(feature = "dsv4-diagnostics")]
        {
            self.fp4_score_dispatch_ledger = None;
        }
        let reset_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        self.layer_routes.reset_for_token()?;
        self.layer_selections.reset_for_token()?;
        #[cfg(feature = "dsv4-diagnostics")]
        if fp4_collapsed_active {
            self.fp4_collapsed_selections
                .reset_for_token(fp4_execution, fp4_source)?;
        }
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.record_reset_cpu_ms = reset_started
                .expect("whole-token profile requires a reset timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        let create_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        let command = ctx.queue.commandBuffer().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "failed to allocate DeepSeek V4 whole-token command buffer".into(),
            )
        })?;
        let mut sparse_visible_counts = [None; DEEPSEEK_V4_LAYER_COUNT];
        let mut token_encoder = DeepSeekV4LayerEncoder::begin(&command, 0, None)?;
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.command_encoder_create_cpu_ms = create_started
                .expect("whole-token profile requires a command-creation timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        let encode_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        for (layer, sparse_visible_count) in sparse_visible_counts.iter_mut().enumerate() {
            let route_record = self.layer_routes.layer(layer)?;
            let selection_record = self.layer_selections.layer(layer)?;
            let encoded = self.encode_token_layer(
                ctx,
                &mut token_encoder,
                token_id,
                position,
                layer,
                &route_record,
                &selection_record,
            )?;
            *sparse_visible_count = encoded.sparse_visible_count;
            #[cfg(feature = "dsv4-diagnostics")]
            if encoded.sparse_visible_count.is_some() && fp4_collapsed_active {
                let ledger = fp4_score_dispatch_ledger
                    .as_mut()
                    .expect("collapsed FP4 score plan requires a dispatch ledger");
                ledger.record_common_prepare()?;
                ledger.record_fp4_pipeline()?;
            }
        }
        token_encoder.end();
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.encode_cpu_ms = encode_started
                .expect("whole-token profile requires an encode timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        let commit_wait_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        let commit_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        command.commit();
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.commit_cpu_ms = commit_started
                .expect("whole-token profile requires a commit timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }
        command.waitUntilCompleted();
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.commit_wait_wall_ms = commit_wait_started
                .expect("whole-token profile requires a commit/wait timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        let status_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        if let Some(error) = command.error() {
            return invalid(format!("whole-token Metal command failed: {error:?}"));
        }
        if let Some(profile) = whole_profile.as_deref_mut() {
            let gpu_seconds = command.GPUEndTime() - command.GPUStartTime();
            if !gpu_seconds.is_finite() || gpu_seconds <= 0.0 {
                return invalid(format!(
                    "whole-token command returned invalid Metal timestamps: ({}, {})",
                    command.GPUStartTime(),
                    command.GPUEndTime()
                ));
            }
            profile.command_gpu_start_seconds = command.GPUStartTime();
            profile.command_gpu_end_seconds = command.GPUEndTime();
            profile.command_gpu_ms = gpu_seconds * 1e3;
            if profile.wait_residual_ms() < -0.25 {
                return invalid(format!(
                    "whole-token commit/wait envelope {:.6} ms is shorter than GPU duration {:.6} ms",
                    profile.commit_wait_wall_ms, profile.command_gpu_ms
                ));
            }
            profile.command_status_cpu_ms = status_started
                .expect("whole-token profile requires a command-status timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        let record_read_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        let completed_routes = self.layer_routes.read_completed()?;
        let completed_selections = self.layer_selections.read_completed()?;
        #[cfg(feature = "dsv4-diagnostics")]
        let completed_fp4_selections = fp4_collapsed_active
            .then(|| self.fp4_collapsed_selections.read_completed())
            .transpose()?;
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.record_read_cpu_ms = record_read_started
                .expect("whole-token profile requires a record-read timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        let validate_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        #[cfg(feature = "dsv4-diagnostics")]
        let mut fp4_ids = [None; DEEPSEEK_V4_LAYER_COUNT];
        for (layer, sparse_visible_count) in sparse_visible_counts.iter().copied().enumerate() {
            completed_routes.validate_layer(layer)?;
            if let Some(visible_count) = sparse_visible_count {
                completed_selections.validate_layer(layer, visible_count)?;
                #[cfg(feature = "dsv4-diagnostics")]
                if let Some(completed_fp4) = completed_fp4_selections.as_ref() {
                    fp4_ids[layer] = Some(completed_fp4.validate_layer(
                        layer,
                        visible_count,
                        fp4_execution,
                        fp4_source,
                    )?);
                }
            } else {
                #[cfg(feature = "dsv4-diagnostics")]
                if let Some(completed_fp4) = completed_fp4_selections.as_ref() {
                    completed_fp4.validate_inactive_layer(layer, fp4_execution, fp4_source)?;
                }
            }
        }
        #[cfg(feature = "dsv4-diagnostics")]
        if let Some(ledger) = fp4_score_dispatch_ledger.as_ref() {
            ledger.validate_completed()?;
        }
        #[cfg(feature = "dsv4-diagnostics")]
        if fp4_collapsed_active {
            for (layer, sparse_visible_count) in sparse_visible_counts.iter().copied().enumerate() {
                if let Some(visible_count) = sparse_visible_count {
                    self.fp4_counterfactual_trace.record(
                        fp4_execution,
                        fp4_source,
                        position,
                        layer,
                        visible_count as i32,
                        fp4_ids[layer]
                            .expect("validated collapsed FP4 layer requires retained IDs"),
                    )?;
                }
            }
        }
        for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
            layer_completed(layer);
        }

        #[cfg(feature = "dsv4-diagnostics")]
        self.decision_diagnostics.finish()?;
        #[cfg(feature = "dsv4-diagnostics")]
        {
            self.fp4_score_dispatch_ledger = fp4_score_dispatch_ledger;
        }
        if let Some(profile) = whole_profile {
            profile.record_validate_callback_cpu_ms = validate_started
                .expect("whole-token profile requires a validation timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        Ok(())
    }

    fn raw_cache_layer(&self, layer: usize) -> Result<MetalTensor, DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!("raw-cache layer {layer} is out of range"));
        }
        let layer_elements = checked_mul(
            self.attention.config().head_dim,
            DEEPSEEK_V4_LOCAL_WINDOW,
            "raw-cache layer elements",
        )?;
        Ok(self.raw_cache.view_subrange(
            checked_mul(layer, layer_elements, "raw-cache layer offset")? as u64,
            vec![
                self.attention.config().head_dim as u64,
                DEEPSEEK_V4_LOCAL_WINDOW as u64,
            ],
        ))
    }

    fn layer_tensor(
        &self,
        layer: usize,
        suffix: &str,
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.residency
            .require_tensor(&format!("blk.{layer}.{suffix}"))
    }
}

fn deepseek_v4_session_attention_config() -> DeepSeekV4PositionZeroAttentionConfig {
    DeepSeekV4PositionZeroAttentionConfig {
        hidden_size: DEEPSEEK_V4_HIDDEN_SIZE,
        q_lora_rank: 1_024,
        head_count: 64,
        head_dim: 512,
        rotary_dim: 64,
        group_count: 8,
        output_rank: 1_024,
    }
}

fn deepseek_v4_session_moe_config(config: &DeepSeekV4Config) -> DeepSeekV4MoeConfig {
    DeepSeekV4MoeConfig {
        hidden_size: DEEPSEEK_V4_HIDDEN_SIZE,
        ffn_size: 2_048,
        expert_count: config.expert_count as usize,
        top_k: 6,
        routed_scale: config.expert_weights_scale,
    }
}

fn validate_session_config(config: &DeepSeekV4Config) -> Result<(), DeepSeekV4MetalError> {
    config.validate_flash_0731_profile()?;
    if config.hidden_size != 4_096
        || config.vocab_size != 129_280
        || config.layer_count != 43
        || config.hyper_connection_count != 4
        || config.attention_head_count != 64
        || config.key_length != 512
        || config.value_length != 512
        || config.q_lora_rank != 1_024
        || config.output_group_count != 8
        || config.output_lora_rank != 1_024
        || config.expert_used_count != 6
        || config.expert_feed_forward_length != 2_048
        || config.shared_expert_count != 1
        || config.sinkhorn_iterations != DEEPSEEK_V4_SINKHORN_ITERATIONS as u32
    {
        return invalid("native session requires the exact Flash-0731 dimensions");
    }
    if !flash_0731_expert_count_supported(config.expert_count) {
        return invalid(format!(
            "native session supports Flash-0731 expert counts 160, 216, and 256, got {}",
            config.expert_count
        ));
    }
    if !config.expert_weights_norm || config.expert_gating_func != 4 {
        return invalid(
            "native route helper requires normalized expert weights and sqrt-softplus gating function 4",
        );
    }
    Ok(())
}

fn session_required_tensor_names(config: &DeepSeekV4Config) -> Vec<String> {
    let mut names = [
        "token_embd.weight",
        "output_hc_fn.weight",
        "output_hc_scale.weight",
        "output_hc_base.weight",
        "output_norm.weight",
        "output.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    let common = [
        "hc_attn_fn.weight",
        "hc_attn_scale.weight",
        "hc_attn_base.weight",
        "attn_norm.weight",
        "attn_sinks.weight",
        "attn_q_a.weight",
        "attn_q_a_norm.weight",
        "attn_q_b.weight",
        "attn_kv.weight",
        "attn_kv_a_norm.weight",
        "attn_output_a.weight",
        "attn_output_b.weight",
        "hc_ffn_fn.weight",
        "hc_ffn_scale.weight",
        "hc_ffn_base.weight",
        "ffn_norm.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
    ];
    for layer in 0..config.layer_count as usize {
        names.extend(common.iter().map(|suffix| format!("blk.{layer}.{suffix}")));
        let route = if layer < config.hash_layer_count as usize {
            "ffn_gate_tid2eid.weight"
        } else {
            "exp_probs_b.bias"
        };
        names.push(format!("blk.{layer}.{route}"));
        match config.attention_kinds[layer] {
            AttentionKind::SlidingWindow => {}
            AttentionKind::CompressedSparse => {
                names.extend(
                    [
                        "attn_compressor_kv.weight",
                        "attn_compressor_gate.weight",
                        "attn_compressor_ape.weight",
                        "attn_compressor_norm.weight",
                        "indexer.attn_q_b.weight",
                        "indexer.proj.weight",
                        "indexer_compressor_kv.weight",
                        "indexer_compressor_gate.weight",
                        "indexer_compressor_ape.weight",
                        "indexer_compressor_norm.weight",
                    ]
                    .into_iter()
                    .map(|suffix| format!("blk.{layer}.{suffix}")),
                );
            }
            AttentionKind::HeavilyCompressed => {
                names.extend(
                    [
                        "attn_compressor_kv.weight",
                        "attn_compressor_gate.weight",
                        "attn_compressor_ape.weight",
                        "attn_compressor_norm.weight",
                    ]
                    .into_iter()
                    .map(|suffix| format!("blk.{layer}.{suffix}")),
                );
            }
        }
    }
    names
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekV4RopeParameters {
    pub rotary_dim: usize,
    pub theta: f32,
    pub scaling_factor: f32,
    pub original_context_length: u32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

fn deepseek_v4_layer_rope(
    config: &DeepSeekV4Config,
    layer: usize,
) -> Result<DeepSeekV4RopeParameters, DeepSeekV4MetalError> {
    let kind = config.attention_kinds.get(layer).copied().ok_or_else(|| {
        DeepSeekV4MetalError::Invalid(format!("RoPE layer {layer} is out of range"))
    })?;
    let compressed = kind != AttentionKind::SlidingWindow;
    Ok(DeepSeekV4RopeParameters {
        rotary_dim: config.rope_dimension_count as usize,
        theta: if compressed {
            config.compress_rope_freq_base
        } else {
            config.rope_freq_base
        },
        scaling_factor: if compressed {
            config.rope_scaling_factor
        } else {
            1.0
        },
        original_context_length: if compressed {
            config.rope_original_context_length
        } else {
            0
        },
        beta_fast: config.rope_yarn_beta_fast,
        beta_slow: config.rope_yarn_beta_slow,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeepSeekV4CompressorPublication {
    Attention,
    IndexerHadamard,
}

#[cfg(feature = "dsv4-diagnostics")]
struct DeepSeekV4IndexerFp4Sidecar {
    enabled: bool,
    capacity_rows: usize,
    values: MetalTensor,
    scales: MetalTensor,
    status: MetalTensor,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4IndexerFp4Sidecar {
    fn new(ctx: &MetalContext, capacity_rows: usize) -> Result<Self, DeepSeekV4MetalError> {
        if capacity_rows == 0 || u32::try_from(capacity_rows).is_err() {
            return invalid("indexer FP4 sidecar capacity must be nonzero and fit u32");
        }
        let unavailable = vec![DEEPSEEK_V4_FP4_STATUS_UNAVAILABLE; capacity_rows];
        Ok(Self {
            enabled: false,
            capacity_rows,
            values: MetalTensor::zeros_dtype(
                ctx,
                vec![
                    crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES as u64,
                    capacity_rows as u64,
                ],
                GgmlType::I8,
            )?,
            scales: MetalTensor::zeros_dtype(
                ctx,
                vec![
                    crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES as u64,
                    capacity_rows as u64,
                ],
                GgmlType::I8,
            )?,
            status: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&unavailable),
                vec![capacity_rows as u64],
                GgmlType::I32,
            )?,
        })
    }

    fn enable(&mut self) {
        self.enabled = true;
    }

    fn disable(&mut self) {
        self.enabled = false;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn invalidate(&self) -> Result<(), DeepSeekV4MetalError> {
        host_write_i32(
            &self.status,
            &vec![DEEPSEEK_V4_FP4_STATUS_UNAVAILABLE; self.capacity_rows],
            "indexer FP4 sidecar status",
        )
    }

    fn encode_row(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        source: &MetalTensor,
        row: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        self.encode_rows(ctx, enc, source, row, 1)
    }

    fn encode_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        source: &MetalTensor,
        first_row: usize,
        row_count: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        if !self.enabled {
            return Ok(());
        }
        if row_count == 0
            || first_row
                .checked_add(row_count)
                .is_none_or(|end| end > self.capacity_rows)
        {
            return invalid(format!(
                "indexer FP4 sidecar rows {first_row}..{} exceed capacity {}",
                first_row.saturating_add(row_count),
                self.capacity_rows,
            ));
        }
        let mut value_shape = source.shape.clone();
        let mut scale_shape = source.shape.clone();
        if value_shape.is_empty() {
            return invalid("indexer FP4 sidecar source has no row width");
        }
        value_shape[0] = crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES as u64;
        scale_shape[0] = crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES as u64;
        let values = raw_i8_subview(
            &self.values,
            checked_mul(
                first_row,
                crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES,
                "indexer FP4 sidecar value offset",
            )?,
            value_shape,
            "indexer FP4 sidecar value rows",
        )?;
        let scales = raw_i8_subview(
            &self.scales,
            checked_mul(
                first_row,
                crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES,
                "indexer FP4 sidecar scale offset",
            )?,
            scale_shape,
            "indexer FP4 sidecar scale rows",
        )?;
        let status = self
            .status
            .view_subrange(first_row as u64, vec![row_count as u64]);
        encode_pack_indexer_fp4_rows_shadow(ctx, enc, source, &values, &scales, &status, row_count)
    }
}

struct DeepSeekV4CompressorFrontier {
    ratio: usize,
    head_dim: usize,
    width: usize,
    rows: usize,
    capacity_rows: usize,
    publication: DeepSeekV4CompressorPublication,
    kv_state: MetalTensor,
    score_state: MetalTensor,
    projected_kv: MetalTensor,
    projected_score: MetalTensor,
    pooled: MetalTensor,
    normalized: MetalTensor,
    published: MetalTensor,
    #[cfg(feature = "dsv4-diagnostics")]
    fp4_sidecar: Option<DeepSeekV4IndexerFp4Sidecar>,
}

impl DeepSeekV4CompressorFrontier {
    fn new(
        ctx: &MetalContext,
        ratio: usize,
        head_dim: usize,
        publication: DeepSeekV4CompressorPublication,
        capacity_rows: usize,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if !matches!(ratio, 4 | 128) || head_dim == 0 {
            return invalid(format!(
                "compressor frontier requires ratio 4 or 128 and a nonzero head dimension, got ratio={ratio} head_dim={head_dim}"
            ));
        }
        if publication == DeepSeekV4CompressorPublication::IndexerHadamard && head_dim != 128 {
            return invalid("indexer publication requires exactly 128 dimensions");
        }
        if capacity_rows == 0
            || !capacity_rows.is_multiple_of(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS)
        {
            return invalid(format!(
                "compressor publication capacity {capacity_rows} is not a nonzero slab multiple"
            ));
        }
        let (width, rows, state_elements) = compressor_frontier_geometry(ratio, head_dim)?;
        let zeros = vec![0.0f32; state_elements];
        let negative_infinity = vec![f32::NEG_INFINITY; state_elements];
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_sidecar = (publication == DeepSeekV4CompressorPublication::IndexerHadamard)
            .then(|| DeepSeekV4IndexerFp4Sidecar::new(ctx, capacity_rows))
            .transpose()?;
        Ok(Self {
            ratio,
            head_dim,
            width,
            rows,
            capacity_rows,
            publication,
            kv_state: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&zeros),
                vec![width as u64, rows as u64],
                GgmlType::F32,
            )?,
            score_state: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&negative_infinity),
                vec![width as u64, rows as u64],
                GgmlType::F32,
            )?,
            projected_kv: MetalTensor::zeros_f32(ctx, vec![width as u64])?,
            projected_score: MetalTensor::zeros_f32(ctx, vec![width as u64])?,
            pooled: MetalTensor::zeros_f32(ctx, vec![head_dim as u64])?,
            normalized: MetalTensor::zeros_f32(ctx, vec![head_dim as u64])?,
            published: MetalTensor::zeros_f16(ctx, vec![head_dim as u64, capacity_rows as u64])?,
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_sidecar,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        input: &MetalTensor,
        kv_weight: &MetalTensor,
        score_weight: &MetalTensor,
        ape: &MetalTensor,
        norm_weight: &MetalTensor,
        position: u32,
        hidden_size: usize,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_compressor_frontier")?;
        validate_f32(input, &[hidden_size as u64], false, "compressor input")?;
        validate_matvec_weight(kv_weight, hidden_size, self.width, "compressor KV weight")?;
        validate_matvec_weight(
            score_weight,
            hidden_size,
            self.width,
            "compressor score weight",
        )?;
        validate_f32(
            ape,
            &[self.width as u64, self.ratio as u64],
            false,
            "compressor APE",
        )?;
        validate_f32(
            norm_weight,
            &[self.head_dim as u64],
            false,
            "compressor norm weight",
        )?;
        validate_f32(
            &self.kv_state,
            &[self.width as u64, self.rows as u64],
            true,
            "compressor KV state",
        )?;
        validate_f32(
            &self.score_state,
            &[self.width as u64, self.rows as u64],
            true,
            "compressor score state",
        )?;
        for (tensor, name) in [
            (&self.projected_kv, "projected compressor KV"),
            (&self.projected_score, "projected compressor score"),
        ] {
            validate_f32(tensor, &[self.width as u64], true, name)?;
        }
        validate_f32(
            &self.pooled,
            &[self.head_dim as u64],
            true,
            "pooled compressor row",
        )?;
        validate_f32(
            &self.normalized,
            &[self.head_dim as u64],
            true,
            "normalized compressor row",
        )?;
        validate_f16(
            &self.published,
            &[self.head_dim as u64, self.capacity_rows as u64],
            true,
            "published compressor rows",
        )?;

        let fused = deepseek_v4_decode_compressor_fused_enabled()
            && kv_weight.dtype == GgmlType::Q8_0
            && score_weight.dtype == GgmlType::Q8_0
            && crate::metal::mat_vec_q8_0_lcpp_enabled();
        if fused {
            validate_ds4_rope(rope, self.head_dim, rope.rotary_dim)?;
            validate_eps(rms_eps, "compressor RMSNorm epsilon")?;
            let (following_position, state_row, published_row) = self.step(position)?;
            let ape_row = ape.view_subrange(
                ((position as usize % self.ratio) * self.width) as u64,
                vec![self.width as u64],
            );
            let state_offset = checked_mul(
                state_row,
                self.width,
                "fused compressor frontier row offset",
            )?;
            let kv_state_row = self
                .kv_state
                .view_subrange(state_offset as u64, vec![self.width as u64]);
            let score_state_row = self
                .score_state
                .view_subrange(state_offset as u64, vec![self.width as u64]);
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                eprintln!(
                    "deepseek_v4: decode Q8 compressor projections and frontier write run one fused dispatch; rollback=QWEN_DSV4_DECODE_COMPRESSOR_FUSED=0"
                );
            });
            crate::metal::encode_ds4_compressor_pair_q8_0_f32(
                ctx,
                enc,
                kv_weight,
                score_weight,
                input,
                &self.projected_score,
                &ape_row,
                &kv_state_row,
                &score_state_row,
                hidden_size,
                self.width,
            )?;
            return self.encode_after_frontier_write(
                ctx,
                enc,
                norm_weight,
                following_position,
                published_row,
                rope,
                rms_eps,
            );
        }

        encode_projection(
            ctx,
            enc,
            kv_weight,
            input,
            &self.projected_kv,
            hidden_size,
            self.width,
            "compressor KV",
        )?;
        encode_projection(
            ctx,
            enc,
            score_weight,
            input,
            &self.projected_score,
            hidden_size,
            self.width,
            "compressor score",
        )?;
        self.encode_projected(
            ctx,
            enc,
            &self.projected_kv,
            &self.projected_score,
            ape,
            norm_weight,
            position,
            rope,
            rms_eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_projected(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        projected_kv: &MetalTensor,
        projected_score: &MetalTensor,
        ape: &MetalTensor,
        norm_weight: &MetalTensor,
        position: u32,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_compressor_frontier_projected")?;
        validate_f32(
            projected_kv,
            &[self.width as u64],
            false,
            "projected compressor KV",
        )?;
        validate_f32(
            projected_score,
            &[self.width as u64],
            false,
            "projected compressor score",
        )?;
        validate_f32(
            ape,
            &[self.width as u64, self.ratio as u64],
            false,
            "compressor APE",
        )?;
        validate_f32(
            norm_weight,
            &[self.head_dim as u64],
            false,
            "compressor norm weight",
        )?;
        validate_f32(
            &self.kv_state,
            &[self.width as u64, self.rows as u64],
            true,
            "compressor KV state",
        )?;
        validate_f32(
            &self.score_state,
            &[self.width as u64, self.rows as u64],
            true,
            "compressor score state",
        )?;
        validate_f32(
            &self.pooled,
            &[self.head_dim as u64],
            true,
            "pooled compressor row",
        )?;
        validate_f32(
            &self.normalized,
            &[self.head_dim as u64],
            true,
            "normalized compressor row",
        )?;
        validate_f16(
            &self.published,
            &[self.head_dim as u64, self.capacity_rows as u64],
            true,
            "published compressor rows",
        )?;
        validate_ds4_rope(rope, self.head_dim, rope.rotary_dim)?;
        validate_eps(rms_eps, "compressor RMSNorm epsilon")?;

        let (following_position, state_row, published_row) = self.step(position)?;
        let ape_row = ape.view_subrange(
            ((position as usize % self.ratio) * self.width) as u64,
            vec![self.width as u64],
        );
        encode_compressor_frontier_write(
            ctx,
            enc,
            projected_kv,
            projected_score,
            &ape_row,
            &self.kv_state,
            &self.score_state,
            self.width,
            state_row,
        )?;
        self.encode_after_frontier_write(
            ctx,
            enc,
            norm_weight,
            following_position,
            published_row,
            rope,
            rms_eps,
        )
    }

    fn step(&self, position: u32) -> Result<(u32, usize, Option<usize>), DeepSeekV4MetalError> {
        let following_position = position
            .checked_add(1)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("compressor position overflow".into()))?;
        let boundary = (following_position as usize).is_multiple_of(self.ratio);
        let published_row = if boundary {
            let row = following_position as usize / self.ratio - 1;
            if row >= self.capacity_rows {
                return invalid(format!(
                    "compressor published row {row} exceeds the allocated {}-row history",
                    self.capacity_rows
                ));
            }
            Some(row)
        } else {
            None
        };
        let state_row = if self.ratio == 4 {
            self.ratio + position as usize % self.ratio
        } else {
            position as usize % self.ratio
        };
        Ok((following_position, state_row, published_row))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_after_frontier_write(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        norm_weight: &MetalTensor,
        following_position: u32,
        published_row: Option<usize>,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let Some(published_row) = published_row else {
            return Ok(());
        };
        encode_compressor_pool(
            ctx,
            enc,
            &self.kv_state,
            &self.score_state,
            &self.pooled,
            self.ratio,
            self.head_dim,
            self.width,
            self.rows,
        )?;
        encode_rms_norm_mul_f32(
            ctx,
            enc,
            &self.pooled,
            norm_weight,
            &self.normalized,
            rms_eps,
        )?;
        let start_position = following_position - self.ratio as u32;
        encode_ds4_rope_tail_adjacent_in_place(
            ctx,
            enc,
            &self.normalized,
            start_position,
            rope,
            false,
        )?;
        if self.publication == DeepSeekV4CompressorPublication::IndexerHadamard {
            encode_hadamard_128_in_place(ctx, enc, &self.normalized)?;
            #[cfg(feature = "dsv4-diagnostics")]
            self.fp4_sidecar
                .as_ref()
                .expect("indexer publication requires an FP4 diagnostics sidecar")
                .encode_row(ctx, enc, &self.normalized, published_row)?;
        }
        encode_scatter_offset_f32_to_f16(
            ctx,
            enc,
            &self.normalized,
            &self.published,
            published_row * self.head_dim,
            self.head_dim,
        )?;
        if self.ratio == 4 {
            encode_compressor_roll_ratio4(ctx, enc, &self.kv_state, &self.score_state, self.width)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_projected_chunk(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        projected_kv: &MetalTensor,
        projected_score: &MetalTensor,
        ape: &MetalTensor,
        norm_weight: &MetalTensor,
        pooled_scratch: &MetalTensor,
        normalized_scratch: &MetalTensor,
        start_position: u32,
        row_count: usize,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_compressor_frontier_projected_chunk")?;
        if row_count == 0 || row_count > DEEPSEEK_V4_PREFILL_MAX_TOKENS {
            return invalid(format!(
                "compressor chunk requires 1..={DEEPSEEK_V4_PREFILL_MAX_TOKENS} rows, got {row_count}"
            ));
        }
        validate_f32(
            projected_kv,
            &[self.width as u64, row_count as u64],
            false,
            "projected compressor KV chunk",
        )?;
        validate_f32(
            projected_score,
            &[self.width as u64, row_count as u64],
            false,
            "projected compressor score chunk",
        )?;
        validate_f32(
            ape,
            &[self.width as u64, self.ratio as u64],
            false,
            "compressor chunk APE",
        )?;
        validate_f32(
            norm_weight,
            &[self.head_dim as u64],
            false,
            "compressor chunk norm weight",
        )?;
        validate_f16(
            &self.published,
            &[self.head_dim as u64, self.capacity_rows as u64],
            true,
            "published compressor rows",
        )?;
        validate_ds4_rope(rope, self.head_dim, rope.rotary_dim)?;
        validate_eps(rms_eps, "compressor RMSNorm epsilon")?;

        let end_position = start_position
            .checked_add(u32::try_from(row_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("compressor chunk rows exceed u32".into())
            })?)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("compressor chunk position overflow".into())
            })?;
        let first_published_row = start_position as usize / self.ratio;
        let end_published_row = end_position as usize / self.ratio;
        let published_rows = end_published_row - first_published_row;
        if end_published_row > self.capacity_rows {
            return invalid(format!(
                "compressor chunk publication end {end_published_row} exceeds the allocated {}-row history",
                self.capacity_rows
            ));
        }
        for (scratch, name) in [
            (pooled_scratch, "pooled compressor chunk scratch"),
            (normalized_scratch, "normalized compressor chunk scratch"),
        ] {
            if scratch.dtype != GgmlType::F32
                || !scratch.is_writable()
                || scratch.n_elements() < self.head_dim as u64 * published_rows as u64
            {
                return invalid(format!(
                    "{name} cannot hold {published_rows} rows of {} values",
                    self.head_dim
                ));
            }
        }
        let pooled_capacity = pooled_scratch.n_elements() as usize / self.head_dim;
        let normalized_capacity = normalized_scratch.n_elements() as usize / self.head_dim;
        let pooled_backing =
            pooled_scratch.view_subrange(0, vec![self.head_dim as u64, pooled_capacity as u64]);
        let normalized_backing = normalized_scratch
            .view_subrange(0, vec![self.head_dim as u64, normalized_capacity as u64]);

        encode_compressor_frontier_chunk(
            ctx,
            enc,
            projected_kv,
            projected_score,
            ape,
            &self.kv_state,
            &self.score_state,
            &pooled_backing,
            self.ratio,
            self.head_dim,
            self.width,
            row_count,
            start_position,
            published_rows,
        )?;
        if published_rows == 0 {
            return Ok(());
        }

        let pooled =
            pooled_backing.view_subrange(0, vec![self.head_dim as u64, published_rows as u64]);
        let normalized =
            normalized_backing.view_subrange(0, vec![self.head_dim as u64, published_rows as u64]);
        validate_f32(
            &pooled,
            &[self.head_dim as u64, published_rows as u64],
            true,
            "pooled compressor chunk rows",
        )?;
        validate_f32(
            &normalized,
            &[self.head_dim as u64, published_rows as u64],
            true,
            "normalized compressor chunk rows",
        )?;
        encode_rms_norm_mul_rows_f32(
            ctx,
            enc,
            &pooled,
            norm_weight,
            &normalized,
            published_rows,
            self.head_dim,
            rms_eps,
        )?;
        let rope_start = u32::try_from(checked_mul(
            first_published_row,
            self.ratio,
            "compressor chunk RoPE start",
        )?)
        .map_err(|_| {
            DeepSeekV4MetalError::Invalid("compressor chunk RoPE start exceeds u32".into())
        })?;
        encode_ds4_rope_tail_adjacent_batch_in_place(
            ctx,
            enc,
            &normalized,
            rope_start,
            published_rows,
            self.ratio as u32,
            rope,
            false,
        )?;
        if self.publication == DeepSeekV4CompressorPublication::IndexerHadamard {
            encode_hadamard_128_rows_in_place(ctx, enc, &normalized, published_rows)?;
            #[cfg(feature = "dsv4-diagnostics")]
            self.fp4_sidecar
                .as_ref()
                .expect("indexer publication requires an FP4 diagnostics sidecar")
                .encode_rows(ctx, enc, &normalized, first_published_row, published_rows)?;
        }
        encode_scatter_offset_f32_to_f16(
            ctx,
            enc,
            &normalized,
            &self.published,
            checked_mul(
                first_published_row,
                self.head_dim,
                "compressor chunk publication offset",
            )?,
            checked_mul(
                published_rows,
                self.head_dim,
                "compressor chunk publication elements",
            )?,
        )?;
        Ok(())
    }

    fn published_count(&self, position: u32) -> usize {
        (position as usize + 1) / self.ratio
    }
}

fn compressor_frontier_geometry(
    ratio: usize,
    head_dim: usize,
) -> Result<(usize, usize, usize), DeepSeekV4MetalError> {
    if !matches!(ratio, 4 | 128) || head_dim == 0 {
        return invalid(format!(
            "compressor frontier requires ratio 4 or 128 and a nonzero head dimension, got ratio={ratio} head_dim={head_dim}"
        ));
    }
    let coefficient = if ratio == 4 { 2 } else { 1 };
    let width = checked_mul(coefficient, head_dim, "compressor frontier width")?;
    let rows = checked_mul(coefficient, ratio, "compressor frontier rows")?;
    let state_elements = checked_mul(width, rows, "compressor frontier elements")?;
    Ok((width, rows, state_elements))
}

enum DeepSeekV4LayerCompressorFrontiers {
    SlidingWindow,
    CompressedSparse {
        attention: Box<DeepSeekV4CompressorFrontier>,
        indexer: Box<DeepSeekV4CompressorFrontier>,
    },
    HeavilyCompressed {
        attention: Box<DeepSeekV4CompressorFrontier>,
    },
}

struct DeepSeekV4CompressorFrontiers {
    hidden_size: usize,
    layers: Vec<DeepSeekV4LayerCompressorFrontiers>,
}

impl DeepSeekV4CompressorFrontiers {
    fn new(
        ctx: &MetalContext,
        config: &DeepSeekV4Config,
        capacity: DeepSeekV4SessionCapacity,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let hidden_size = config.hidden_size as usize;
        let attention_dim = config.key_length as usize;
        let indexer_dim = config.indexer_key_length as usize;
        let mut layers = Vec::with_capacity(config.attention_kinds.len());
        for kind in config.attention_kinds.iter().copied() {
            layers.push(match kind {
                AttentionKind::SlidingWindow => DeepSeekV4LayerCompressorFrontiers::SlidingWindow,
                AttentionKind::CompressedSparse => {
                    DeepSeekV4LayerCompressorFrontiers::CompressedSparse {
                        attention: Box::new(DeepSeekV4CompressorFrontier::new(
                            ctx,
                            4,
                            attention_dim,
                            DeepSeekV4CompressorPublication::Attention,
                            capacity.csa_physical_rows(),
                        )?),
                        indexer: Box::new(DeepSeekV4CompressorFrontier::new(
                            ctx,
                            4,
                            indexer_dim,
                            DeepSeekV4CompressorPublication::IndexerHadamard,
                            capacity.csa_physical_rows(),
                        )?),
                    }
                }
                AttentionKind::HeavilyCompressed => {
                    DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed {
                        attention: Box::new(DeepSeekV4CompressorFrontier::new(
                            ctx,
                            128,
                            attention_dim,
                            DeepSeekV4CompressorPublication::Attention,
                            capacity.hca_physical_rows(),
                        )?),
                    }
                }
            });
        }
        Ok(Self {
            hidden_size,
            layers,
        })
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn enable_fp4_shadow_lineage(&mut self) -> Result<(), DeepSeekV4MetalError> {
        let mut enabled = 0usize;
        for layer in &mut self.layers {
            if let DeepSeekV4LayerCompressorFrontiers::CompressedSparse { indexer, .. } = layer {
                indexer
                    .fp4_sidecar
                    .as_mut()
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(
                            "CSA indexer frontier is missing its FP4 diagnostics sidecar".into(),
                        )
                    })?
                    .enable();
                enabled += 1;
            }
        }
        if enabled != diagnostics::CSA_LAYER_COUNT {
            return invalid(format!(
                "enabled {enabled} FP4 indexer sidecars, expected {}",
                diagnostics::CSA_LAYER_COUNT
            ));
        }
        Ok(())
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn invalidate_fp4_shadow_lineage(&self) -> Result<(), DeepSeekV4MetalError> {
        for layer in &self.layers {
            if let DeepSeekV4LayerCompressorFrontiers::CompressedSparse { indexer, .. } = layer {
                indexer
                    .fp4_sidecar
                    .as_ref()
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(
                            "CSA indexer frontier is missing its FP4 diagnostics sidecar".into(),
                        )
                    })?
                    .invalidate()?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn disable_fp4_shadow_lineage(&mut self) -> Result<(), DeepSeekV4MetalError> {
        for layer in &mut self.layers {
            if let DeepSeekV4LayerCompressorFrontiers::CompressedSparse { indexer, .. } = layer {
                indexer
                    .fp4_sidecar
                    .as_mut()
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(
                            "CSA indexer frontier is missing its FP4 diagnostics sidecar".into(),
                        )
                    })?
                    .disable();
            }
        }
        Ok(())
    }

    fn encode_layer(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residency: &DeepSeekV4MetalResidency,
        layer: usize,
        position: u32,
        normalized_input: &MetalTensor,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let frontiers = self.layers.get(layer).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("compressor layer {layer} is out of range"))
        })?;
        let tensor = |suffix: &str| residency.require_tensor(&format!("blk.{layer}.{suffix}"));
        match frontiers {
            DeepSeekV4LayerCompressorFrontiers::SlidingWindow => Ok(()),
            DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer } => {
                attention.encode(
                    ctx,
                    enc,
                    normalized_input,
                    tensor("attn_compressor_kv.weight")?,
                    tensor("attn_compressor_gate.weight")?,
                    tensor("attn_compressor_ape.weight")?,
                    tensor("attn_compressor_norm.weight")?,
                    position,
                    self.hidden_size,
                    rope,
                    rms_eps,
                )?;
                indexer.encode(
                    ctx,
                    enc,
                    normalized_input,
                    tensor("indexer_compressor_kv.weight")?,
                    tensor("indexer_compressor_gate.weight")?,
                    tensor("indexer_compressor_ape.weight")?,
                    tensor("indexer_compressor_norm.weight")?,
                    position,
                    self.hidden_size,
                    rope,
                    rms_eps,
                )
            }
            DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention } => attention
                .encode(
                    ctx,
                    enc,
                    normalized_input,
                    tensor("attn_compressor_kv.weight")?,
                    tensor("attn_compressor_gate.weight")?,
                    tensor("attn_compressor_ape.weight")?,
                    tensor("attn_compressor_norm.weight")?,
                    position,
                    self.hidden_size,
                    rope,
                    rms_eps,
                ),
        }
    }

    fn attention_rows(
        &self,
        layer: usize,
        position: u32,
    ) -> Result<Option<DeepSeekV4PublishedRows<'_>>, DeepSeekV4MetalError> {
        let frontier = match self.layers.get(layer).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("compressor layer {layer} is out of range"))
        })? {
            DeepSeekV4LayerCompressorFrontiers::SlidingWindow => return Ok(None),
            DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, .. }
            | DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention } => attention,
        };
        let count = frontier.published_count(position);
        if count == 0 {
            return Ok(None);
        }
        if count > frontier.capacity_rows {
            return invalid(format!(
                "visible compressed rows {count} exceed the allocated {}-row history",
                frontier.capacity_rows
            ));
        }
        Ok(Some(DeepSeekV4PublishedRows {
            cache: &frontier.published,
            count,
            capacity_rows: frontier.capacity_rows,
        }))
    }

    fn csa_rows(
        &self,
        layer: usize,
        position: u32,
    ) -> Result<Option<DeepSeekV4CsaRows<'_>>, DeepSeekV4MetalError> {
        let Some(DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer }) =
            self.layers.get(layer)
        else {
            return Ok(None);
        };
        let attention_count = attention.published_count(position);
        let indexer_count = indexer.published_count(position);
        if attention_count != indexer_count || attention.capacity_rows != indexer.capacity_rows {
            return invalid(format!(
                "CSA layer {layer} histories are misaligned: attention={attention_count}/{} indexer={indexer_count}/{}",
                attention.capacity_rows, indexer.capacity_rows
            ));
        }
        if attention_count == 0 {
            return Ok(None);
        }
        Ok(Some(DeepSeekV4CsaRows {
            attention_cache: &attention.published,
            indexer_cache: &indexer.published,
            #[cfg(feature = "dsv4-diagnostics")]
            indexer_fp4_sidecar: indexer.fp4_sidecar.as_ref(),
            count: attention_count,
            capacity_rows: attention.capacity_rows,
        }))
    }
}

#[derive(Clone, Copy)]
struct DeepSeekV4PublishedRows<'a> {
    cache: &'a MetalTensor,
    count: usize,
    capacity_rows: usize,
}

#[derive(Clone, Copy)]
struct DeepSeekV4CsaRows<'a> {
    attention_cache: &'a MetalTensor,
    indexer_cache: &'a MetalTensor,
    #[cfg(feature = "dsv4-diagnostics")]
    indexer_fp4_sidecar: Option<&'a DeepSeekV4IndexerFp4Sidecar>,
    count: usize,
    capacity_rows: usize,
}

const DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH: usize = 3;
#[cfg(feature = "dsv4-diagnostics")]
const DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH: usize = 6;

#[derive(Clone)]
struct DeepSeekV4SelectionRecord {
    visible_count: MetalTensor,
    selected_count: MetalTensor,
    status: MetalTensor,
}

#[derive(Clone, Copy)]
struct DeepSeekV4CsaSelectionView<'a> {
    cache_order_ids: &'a MetalTensor,
    selected_count: &'a MetalTensor,
    visible_count: &'a MetalTensor,
}

impl DeepSeekV4CsaSelectionView<'_> {
    fn validate(&self) -> Result<(), DeepSeekV4MetalError> {
        validate_i32(
            self.cache_order_ids,
            &[DEEPSEEK_V4_CSA_TOP_K as u64, 1],
            false,
            "CSA selection-view IDs",
        )?;
        validate_i32(self.selected_count, &[1], false, "CSA selection-view count")?;
        validate_i32(
            self.visible_count,
            &[1],
            false,
            "CSA selection-view visibility",
        )
    }
}

impl DeepSeekV4SelectionRecord {
    fn validate(&self) -> Result<(), DeepSeekV4MetalError> {
        validate_i32(&self.visible_count, &[1], true, "CSA visible-count record")?;
        validate_i32(
            &self.selected_count,
            &[1],
            true,
            "CSA selected-count record",
        )?;
        validate_i32(&self.status, &[1], true, "CSA status record")
    }
}

struct DeepSeekV4LayerSelectionRecords {
    integers: MetalTensor,
}

struct DeepSeekV4CompletedLayerSelectionRecords {
    integers: Vec<i32>,
}

impl DeepSeekV4LayerSelectionRecords {
    fn new(ctx: &MetalContext) -> Result<Self, DeepSeekV4MetalError> {
        Ok(Self {
            integers: MetalTensor::zeros_i32(
                ctx,
                vec![
                    DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH as u64,
                    DEEPSEEK_V4_LAYER_COUNT as u64,
                ],
            )?,
        })
    }

    fn layer(&self, layer: usize) -> Result<DeepSeekV4SelectionRecord, DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!(
                "CSA selection-record layer {layer} is out of range"
            ));
        }
        validate_i32(
            &self.integers,
            &[
                DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH as u64,
                DEEPSEEK_V4_LAYER_COUNT as u64,
            ],
            true,
            "CSA layer-selection records",
        )?;
        let base = checked_mul(
            layer,
            DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH,
            "CSA selection-record layer offset",
        )? as u64;
        let record = DeepSeekV4SelectionRecord {
            visible_count: self.integers.view_subrange(base, vec![1]),
            selected_count: self.integers.view_subrange(base + 1, vec![1]),
            status: self.integers.view_subrange(base + 2, vec![1]),
        };
        record.validate()?;
        Ok(record)
    }

    fn reset_for_token(&self) -> Result<(), DeepSeekV4MetalError> {
        host_write_i32(
            &self.integers,
            &[-1; DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH * DEEPSEEK_V4_LAYER_COUNT],
            "reset CSA layer-selection records",
        )
    }

    fn read_completed(
        &self,
    ) -> Result<DeepSeekV4CompletedLayerSelectionRecords, DeepSeekV4MetalError> {
        validate_i32(
            &self.integers,
            &[
                DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH as u64,
                DEEPSEEK_V4_LAYER_COUNT as u64,
            ],
            false,
            "completed CSA layer-selection records",
        )?;
        Ok(DeepSeekV4CompletedLayerSelectionRecords {
            integers: host_read_i32(&self.integers, "completed CSA layer-selection records")?,
        })
    }
}

impl DeepSeekV4CompletedLayerSelectionRecords {
    fn validate_layer(
        &self,
        layer: usize,
        expected_visible_count: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!(
                "completed CSA selection layer {layer} is out of range"
            ));
        }
        let base = checked_mul(
            layer,
            DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH,
            "completed CSA selection-record layer offset",
        )?;
        let visible_count = self.integers[base];
        let selected_count = self.integers[base + 1];
        let status = self.integers[base + 2];
        if visible_count != expected_visible_count as i32
            || selected_count != DEEPSEEK_V4_CSA_TOP_K as i32
            || status != 0
        {
            return invalid(format!(
                "layer {layer} sparse CSA selection failed with visible={visible_count} expected={expected_visible_count} selected={selected_count} status={status}"
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone)]
struct DeepSeekV4LayerFp4SelectionRecord {
    cache_order_ids: MetalTensor,
    eligible_visible: MetalTensor,
    eligibility_record: MetalTensor,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4LayerFp4SelectionRecord {
    fn output<'a>(
        &'a self,
        selection_record: &'a DeepSeekV4SelectionRecord,
    ) -> DeepSeekV4Fp4SelectionOutput<'a> {
        DeepSeekV4Fp4SelectionOutput {
            eligible_visible: &self.eligible_visible,
            eligibility_record: &self.eligibility_record,
            cache_order_ids: &self.cache_order_ids,
            selected_count: &selection_record.selected_count,
            status: &selection_record.status,
        }
    }

    fn selection_view<'a>(
        &'a self,
        selection_record: &'a DeepSeekV4SelectionRecord,
    ) -> DeepSeekV4CsaSelectionView<'a> {
        DeepSeekV4CsaSelectionView {
            cache_order_ids: &self.cache_order_ids,
            selected_count: &selection_record.selected_count,
            visible_count: &self.eligible_visible,
        }
    }
}

#[cfg(feature = "dsv4-diagnostics")]
#[derive(Clone, Copy)]
struct DeepSeekV4Fp4SelectionOutput<'a> {
    eligible_visible: &'a MetalTensor,
    eligibility_record: &'a MetalTensor,
    cache_order_ids: &'a MetalTensor,
    selected_count: &'a MetalTensor,
    status: &'a MetalTensor,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4Fp4SelectionOutput<'_> {
    fn validate(&self) -> Result<(), DeepSeekV4MetalError> {
        validate_i32(
            self.eligible_visible,
            &[1],
            true,
            "FP4 selection eligible visibility",
        )?;
        validate_i32(
            self.eligibility_record,
            &[3],
            true,
            "FP4 selection eligibility record",
        )?;
        validate_i32(
            self.cache_order_ids,
            &[DEEPSEEK_V4_CSA_TOP_K as u64, 1],
            true,
            "FP4 selection cache-order IDs",
        )?;
        validate_i32(self.selected_count, &[1], true, "FP4 selection count")?;
        validate_i32(self.status, &[1], true, "FP4 selection status")
    }
}

#[cfg(feature = "dsv4-diagnostics")]
struct DeepSeekV4LayerFp4SelectionRecords {
    cache_order_ids: MetalTensor,
    integers: MetalTensor,
}

#[cfg(feature = "dsv4-diagnostics")]
struct DeepSeekV4CompletedLayerFp4SelectionRecords {
    cache_order_ids: Vec<i32>,
    integers: Vec<i32>,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4LayerFp4SelectionRecords {
    fn new(ctx: &MetalContext) -> Result<Self, DeepSeekV4MetalError> {
        Ok(Self {
            cache_order_ids: MetalTensor::zeros_i32(
                ctx,
                vec![DEEPSEEK_V4_CSA_TOP_K as u64, DEEPSEEK_V4_LAYER_COUNT as u64],
            )?,
            integers: MetalTensor::zeros_i32(
                ctx,
                vec![
                    DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH as u64,
                    DEEPSEEK_V4_LAYER_COUNT as u64,
                ],
            )?,
        })
    }

    fn layer(
        &self,
        layer: usize,
    ) -> Result<DeepSeekV4LayerFp4SelectionRecord, DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!(
                "collapsed FP4 selection-record layer {layer} is out of range"
            ));
        }
        validate_i32(
            &self.cache_order_ids,
            &[DEEPSEEK_V4_CSA_TOP_K as u64, DEEPSEEK_V4_LAYER_COUNT as u64],
            true,
            "collapsed FP4 layer-selection IDs",
        )?;
        validate_i32(
            &self.integers,
            &[
                DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH as u64,
                DEEPSEEK_V4_LAYER_COUNT as u64,
            ],
            true,
            "collapsed FP4 layer-selection records",
        )?;
        let ids_base = checked_mul(
            layer,
            DEEPSEEK_V4_CSA_TOP_K,
            "collapsed FP4 selection-ID layer offset",
        )? as u64;
        let record_base = checked_mul(
            layer,
            DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH,
            "collapsed FP4 completion-record layer offset",
        )? as u64;
        Ok(DeepSeekV4LayerFp4SelectionRecord {
            cache_order_ids: self
                .cache_order_ids
                .view_subrange(ids_base, vec![DEEPSEEK_V4_CSA_TOP_K as u64, 1]),
            eligible_visible: self.integers.view_subrange(record_base, vec![1]),
            eligibility_record: self.integers.view_subrange(record_base + 1, vec![3]),
        })
    }

    fn reset_for_token(
        &self,
        execution: DeepSeekV4Fp4ShadowExecution,
        source: DeepSeekV4Fp4SelectionSource,
    ) -> Result<(), DeepSeekV4MetalError> {
        let mut integers =
            vec![-1; DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH * DEEPSEEK_V4_LAYER_COUNT];
        for record in integers.chunks_exact_mut(DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH) {
            record[4] = i32::from(source.domain_code());
            record[5] = i32::from(execution.domain_code());
        }
        host_write_i32(
            &self.cache_order_ids,
            &vec![-1; DEEPSEEK_V4_CSA_TOP_K * DEEPSEEK_V4_LAYER_COUNT],
            "reset collapsed FP4 layer-selection IDs",
        )?;
        host_write_i32(
            &self.integers,
            &integers,
            "reset collapsed FP4 layer-selection records",
        )
    }

    fn read_completed(
        &self,
    ) -> Result<DeepSeekV4CompletedLayerFp4SelectionRecords, DeepSeekV4MetalError> {
        validate_i32(
            &self.cache_order_ids,
            &[DEEPSEEK_V4_CSA_TOP_K as u64, DEEPSEEK_V4_LAYER_COUNT as u64],
            false,
            "completed collapsed FP4 layer-selection IDs",
        )?;
        validate_i32(
            &self.integers,
            &[
                DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH as u64,
                DEEPSEEK_V4_LAYER_COUNT as u64,
            ],
            false,
            "completed collapsed FP4 layer-selection records",
        )?;
        Ok(DeepSeekV4CompletedLayerFp4SelectionRecords {
            cache_order_ids: host_read_i32(
                &self.cache_order_ids,
                "completed collapsed FP4 layer-selection IDs",
            )?,
            integers: host_read_i32(
                &self.integers,
                "completed collapsed FP4 layer-selection records",
            )?,
        })
    }
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4CompletedLayerFp4SelectionRecords {
    fn record_and_ids(&self, layer: usize) -> Result<(&[i32], &[i32]), DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!(
                "completed collapsed FP4 selection layer {layer} is out of range"
            ));
        }
        let record_base = checked_mul(
            layer,
            DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH,
            "completed collapsed FP4 record layer offset",
        )?;
        let record_end = record_base + DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH;
        let record = self.integers.get(record_base..record_end).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "completed collapsed FP4 record layer {layer} is truncated"
            ))
        })?;
        let ids_base = checked_mul(
            layer,
            DEEPSEEK_V4_CSA_TOP_K,
            "completed collapsed FP4 ID layer offset",
        )?;
        let ids_end = ids_base + DEEPSEEK_V4_CSA_TOP_K;
        let ids = self.cache_order_ids.get(ids_base..ids_end).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "completed collapsed FP4 IDs for layer {layer} are truncated"
            ))
        })?;
        Ok((record, ids))
    }

    fn validate_layer(
        &self,
        layer: usize,
        expected_visible_count: usize,
        expected_execution: DeepSeekV4Fp4ShadowExecution,
        expected_source: DeepSeekV4Fp4SelectionSource,
    ) -> Result<&[i32], DeepSeekV4MetalError> {
        let (record, ids) = self.record_and_ids(layer)?;
        let expected_record = [
            expected_visible_count as i32,
            0,
            -1,
            0,
            i32::from(expected_source.domain_code()),
            i32::from(expected_execution.domain_code()),
        ];
        if record != expected_record {
            return invalid(format!(
                "layer {layer} collapsed FP4 completion record {record:?} differs from {expected_record:?}"
            ));
        }
        if ids.windows(2).any(|pair| pair[0] >= pair[1])
            || ids
                .iter()
                .any(|&id| id < 0 || id >= expected_visible_count as i32)
        {
            return invalid(format!(
                "layer {layer} collapsed FP4 IDs are not sorted, unique, and in range"
            ));
        }
        Ok(ids)
    }

    fn validate_inactive_layer(
        &self,
        layer: usize,
        expected_execution: DeepSeekV4Fp4ShadowExecution,
        expected_source: DeepSeekV4Fp4SelectionSource,
    ) -> Result<(), DeepSeekV4MetalError> {
        let (record, ids) = self.record_and_ids(layer)?;
        let expected_record = [
            -1,
            -1,
            -1,
            -1,
            i32::from(expected_source.domain_code()),
            i32::from(expected_execution.domain_code()),
        ];
        if record != expected_record || ids.iter().any(|&id| id != -1) {
            return invalid(format!(
                "inactive layer {layer} collapsed FP4 slice was modified: record={record:?}"
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "dsv4-diagnostics")]
struct DeepSeekV4Fp4ShadowScratch {
    capacity_rows: usize,
    query_values: MetalTensor,
    query_scales: MetalTensor,
    query_status: MetalTensor,
    query_units: MetalTensor,
    eligible_visible: MetalTensor,
    eligibility_record: MetalTensor,
    scores: MetalTensor,
    selected_mask: MetalTensor,
    cache_order_ids: MetalTensor,
    selected_count: MetalTensor,
    status: MetalTensor,
}

#[cfg(feature = "dsv4-diagnostics")]
impl DeepSeekV4Fp4ShadowScratch {
    fn new(ctx: &MetalContext, capacity_rows: usize) -> Result<Self, DeepSeekV4MetalError> {
        if capacity_rows < DEEPSEEK_V4_CSA_TOP_K
            || !capacity_rows.is_multiple_of(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS)
        {
            return invalid(format!(
                "FP4 shadow scratch capacity {capacity_rows} is not an aligned top-k superset"
            ));
        }
        Ok(Self {
            capacity_rows,
            query_values: MetalTensor::zeros_dtype(
                ctx,
                vec![
                    crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES as u64,
                    64,
                    1,
                ],
                GgmlType::I8,
            )?,
            query_scales: MetalTensor::zeros_dtype(
                ctx,
                vec![
                    crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES as u64,
                    64,
                    1,
                ],
                GgmlType::I8,
            )?,
            query_status: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&[DEEPSEEK_V4_FP4_STATUS_UNAVAILABLE; 64]),
                vec![64],
                GgmlType::I32,
            )?,
            query_units: MetalTensor::zeros_f16(ctx, vec![128, 64, 1])?,
            eligible_visible: MetalTensor::zeros_i32(ctx, vec![1])?,
            eligibility_record: MetalTensor::zeros_i32(ctx, vec![3])?,
            scores: MetalTensor::zeros_f32(ctx, vec![capacity_rows as u64, 1])?,
            selected_mask: MetalTensor::zeros_i32(ctx, vec![capacity_rows as u64, 1])?,
            cache_order_ids: MetalTensor::zeros_i32(ctx, vec![DEEPSEEK_V4_CSA_TOP_K as u64, 1])?,
            selected_count: MetalTensor::zeros_i32(ctx, vec![1])?,
            status: MetalTensor::zeros_i32(ctx, vec![1])?,
        })
    }

    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        index_queries: &MetalTensor,
        head_weights: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        requested_visible: &MetalTensor,
    ) -> Result<(), DeepSeekV4MetalError> {
        let output = DeepSeekV4Fp4SelectionOutput {
            eligible_visible: &self.eligible_visible,
            eligibility_record: &self.eligibility_record,
            cache_order_ids: &self.cache_order_ids,
            selected_count: &self.selected_count,
            status: &self.status,
        };
        self.encode_into(
            ctx,
            enc,
            index_queries,
            head_weights,
            rows,
            requested_visible,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_into(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        index_queries: &MetalTensor,
        head_weights: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        requested_visible: &MetalTensor,
        output: DeepSeekV4Fp4SelectionOutput<'_>,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_fp4_shadow")?;
        if rows.count <= DEEPSEEK_V4_CSA_TOP_K
            || rows.count > rows.capacity_rows
            || rows.capacity_rows != self.capacity_rows
        {
            return invalid(format!(
                "FP4 shadow requires 513..={} rows, got {}/{}",
                self.capacity_rows, rows.count, rows.capacity_rows
            ));
        }
        let sidecar = rows.indexer_fp4_sidecar.ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("FP4 shadow query has no indexer sidecar".into())
        })?;
        if !sidecar.is_enabled() {
            return invalid("FP4 shadow lineage is not enabled");
        }
        validate_f32(
            index_queries,
            &[128, 64, 1],
            false,
            "FP4 shadow index queries",
        )?;
        validate_f32(head_weights, &[64, 1], false, "FP4 shadow head weights")?;
        validate_i32(
            requested_visible,
            &[1],
            false,
            "FP4 shadow requested visibility",
        )?;
        output.validate()?;
        encode_pack_indexer_fp4_rows_shadow(
            ctx,
            enc,
            index_queries,
            &self.query_values,
            &self.query_scales,
            &self.query_status,
            64,
        )?;
        encode_unpack_indexer_fp4_units_shadow(
            ctx,
            enc,
            &self.query_values,
            &self.query_status,
            &self.query_units,
            64,
        )?;
        encode_indexer_fp4_shadow_preflight(
            ctx,
            enc,
            &self.query_status,
            &sidecar.status,
            requested_visible,
            output.eligible_visible,
            output.eligibility_record,
            rows.capacity_rows,
            rows.count,
        )?;
        encode_lightning_indexer_scores_fp4_matrix_shadow(
            ctx,
            enc,
            &self.query_units,
            &self.query_scales,
            head_weights,
            &sidecar.values,
            &sidecar.scales,
            output.eligible_visible,
            &self.scores,
            rows.capacity_rows,
            1,
        )?;
        encode_select_top_k_f32_with_policy(
            ctx,
            enc,
            &self.scores,
            output.eligible_visible,
            &self.selected_mask,
            None,
            output.cache_order_ids,
            output.selected_count,
            output.status,
            rows.capacity_rows,
            rows.count,
            DEEPSEEK_V4_CSA_TOP_K,
            1,
            DeepSeekV4SelectorDispatchPolicy::Production,
            true,
        )
    }

    fn selection_view(&self) -> DeepSeekV4CsaSelectionView<'_> {
        DeepSeekV4CsaSelectionView {
            cache_order_ids: &self.cache_order_ids,
            selected_count: &self.selected_count,
            visible_count: &self.eligible_visible,
        }
    }

    fn validate_completed(&self) -> Result<(), DeepSeekV4MetalError> {
        let eligibility = host_read_i32(
            &self.eligibility_record,
            "completed FP4 shadow eligibility record",
        )?;
        let selected_count =
            host_read_i32(&self.selected_count, "completed FP4 shadow selected count")?;
        let status = host_read_i32(&self.status, "completed FP4 shadow selection status")?;
        if eligibility.as_slice() != [0, -1, 0]
            || selected_count.as_slice() != [DEEPSEEK_V4_CSA_TOP_K as i32]
            || status.as_slice() != [0]
        {
            return invalid(format!(
                "FP4 shadow selection failed with eligibility={eligibility:?} count={selected_count:?} status={status:?}"
            ));
        }
        Ok(())
    }

    fn record_counterfactual_selection(
        &self,
        trace: &mut diagnostics::DeepSeekV4Fp4CounterfactualTrace,
        execution: DeepSeekV4Fp4ShadowExecution,
        source: DeepSeekV4Fp4SelectionSource,
        position: u32,
        layer: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        let visible = host_read_i32(
            &self.eligible_visible,
            "consumed FP4 counterfactual visibility",
        )?;
        let ids = host_read_i32(&self.cache_order_ids, "consumed FP4 counterfactual IDs")?;
        trace.record(execution, source, position, layer, visible[0], &ids)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_layer(
        &self,
        layer: usize,
        position: u32,
        rows: DeepSeekV4CsaRows<'_>,
        authoritative_scores: &MetalTensor,
        authoritative_mask: &MetalTensor,
        authoritative_ids: &MetalTensor,
        authoritative_count: &MetalTensor,
        authoritative_status: &MetalTensor,
    ) -> Result<DeepSeekV4Fp4ShadowLayer, DeepSeekV4MetalError> {
        let sidecar = rows.indexer_fp4_sidecar.ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("FP4 shadow report has no indexer sidecar".into())
        })?;
        let visible_key_status = sidecar.status.view_subrange(0, vec![rows.count as u64]);
        Ok(diagnostics::build_fp4_shadow_layer(
            diagnostics::DeepSeekV4Fp4ShadowLayerInputs {
                layer,
                position,
                visible_count: rows.count,
                capacity_rows: rows.capacity_rows,
                query_statuses: host_read_i32(&self.query_status, "FP4 shadow query statuses")?,
                visible_key_statuses: host_read_i32(
                    &visible_key_status,
                    "FP4 shadow visible key statuses",
                )?,
                eligibility_record: host_read_i32(
                    &self.eligibility_record,
                    "FP4 shadow eligibility record",
                )?,
                authoritative_scores: host_read_f32(
                    authoritative_scores,
                    "FP4 authoritative scores",
                )?,
                authoritative_mask: host_read_i32(authoritative_mask, "FP4 authoritative mask")?,
                authoritative_ids: host_read_i32(authoritative_ids, "FP4 authoritative IDs")?,
                authoritative_count: host_read_i32(
                    authoritative_count,
                    "FP4 authoritative selected count",
                )?,
                authoritative_status: host_read_i32(
                    authoritative_status,
                    "FP4 authoritative selection status",
                )?,
                shadow_scores: host_read_f32(&self.scores, "FP4 shadow scores")?,
                shadow_mask: host_read_i32(&self.selected_mask, "FP4 shadow mask")?,
                shadow_ids: host_read_i32(&self.cache_order_ids, "FP4 shadow IDs")?,
                shadow_count: host_read_i32(&self.selected_count, "FP4 shadow selected count")?,
                shadow_status: host_read_i32(&self.status, "FP4 shadow selection status")?,
            },
        )?)
    }
}

const DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS: usize = 20;
const DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS: usize = 10;
const DEEPSEEK_V4_MULTIGROUP_SELECTOR_DIGITS: usize = 8;
const DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS: usize = 5;
const DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS: usize = 32;
#[cfg(not(test))]
const DEEPSEEK_V4_F16_MATRIX_SCORER_MIN_VISIBLE_ROWS: usize = 16_384;
#[cfg(not(test))]
const DEEPSEEK_V4_MULTIGROUP_SELECTOR_QUALIFIED_DEVICE: &str = "Apple M4 Max";
#[cfg(not(test))]
const DEEPSEEK_V4_F16_MATRIX_SCORER_QUALIFIED_DEVICE: &str = "Apple M4 Max";
#[doc(hidden)]
pub const DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS: usize = 196_608;
#[doc(hidden)]
pub const DEEPSEEK_V4_MULTIGROUP_SELECTOR_MAX_CAPACITY_ROWS: usize = 262_144;

fn deepseek_v4_multigroup_selector_capacity_supported(capacity_rows: usize) -> bool {
    (DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS
        ..=DEEPSEEK_V4_MULTIGROUP_SELECTOR_MAX_CAPACITY_ROWS)
        .contains(&capacity_rows)
}

fn deepseek_v4_multigroup_selector_eligible(capacity_rows: usize, visible_rows: usize) -> bool {
    deepseek_v4_multigroup_selector_capacity_supported(capacity_rows)
        && visible_rows <= capacity_rows
        && visible_rows >= DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS
        && visible_rows >= capacity_rows - capacity_rows / 4
}

#[cfg(not(test))]
crate::env_flag!(
    default_on deepseek_v4_multigroup_selector_enabled,
    "QWEN_DSV4_MULTIGROUP_SELECTOR"
);

#[cfg(not(test))]
crate::env_flag!(
    default_off deepseek_v4_f16_matrix_scorer_enabled,
    "QWEN_DSV4_LIGHTNING_F16_MATRIX"
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeepSeekV4SparseSelectorMode {
    Radix4,
    MultigroupProduction,
    MultigroupExperimental,
}

#[derive(Debug, Default)]
struct DeepSeekV4MultigroupSelectorInvocationCounters {
    multigroup: Cell<u64>,
    ineligible_radix4: Cell<u64>,
}

impl DeepSeekV4MultigroupSelectorInvocationCounters {
    fn next_multigroup(&self) -> Result<u64, DeepSeekV4MetalError> {
        self.multigroup.get().checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 multi-group selector invocation count overflowed".into(),
            )
        })
    }

    fn commit_multigroup(&self, next: u64) {
        self.multigroup.set(next);
    }

    fn next_ineligible_radix4(&self) -> Result<u64, DeepSeekV4MetalError> {
        self.ineligible_radix4.get().checked_add(1).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 ineligible radix4 selector invocation count overflowed".into(),
            )
        })
    }

    fn commit_ineligible_radix4(&self, next: u64) {
        self.ineligible_radix4.set(next);
    }

    fn telemetry(&self, sealed: bool) -> DeepSeekV4MultigroupSelectorTelemetry {
        DeepSeekV4MultigroupSelectorTelemetry {
            sealed,
            multigroup_invocations: self.multigroup.get(),
            ineligible_radix4_invocations: self.ineligible_radix4.get(),
        }
    }
}

#[derive(Debug)]
struct DeepSeekV4MultigroupSelectorGeneration {
    next: Cell<Option<NonZeroU32>>,
}

impl DeepSeekV4MultigroupSelectorGeneration {
    fn new() -> Self {
        Self::from_next(NonZeroU32::MIN)
    }

    fn from_next(next: NonZeroU32) -> Self {
        Self {
            next: Cell::new(Some(next)),
        }
    }

    fn take(&self) -> Result<u32, DeepSeekV4MetalError> {
        let generation = self.next.get().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 multi-group selector generation space is exhausted".into(),
            )
        })?;
        self.next
            .set(generation.get().checked_add(1).and_then(NonZeroU32::new));
        Ok(generation.get())
    }
}

struct DeepSeekV4MultigroupSelectorScratch {
    records: MetalTensor,
    partition_plan: MetalTensor,
    state: MetalTensor,
    private_mask: MetalTensor,
    private_ids: MetalTensor,
    generation: DeepSeekV4MultigroupSelectorGeneration,
}

impl DeepSeekV4MultigroupSelectorScratch {
    fn new(ctx: &MetalContext, capacity_rows: usize) -> Result<Self, DeepSeekV4MetalError> {
        if !deepseek_v4_multigroup_selector_capacity_supported(capacity_rows) {
            return invalid(format!(
                "multi-group selector scratch does not support capacity {capacity_rows}"
            ));
        }
        Ok(Self {
            records: MetalTensor::zeros_i32(
                ctx,
                vec![
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS as u64,
                ],
            )?,
            partition_plan: MetalTensor::zeros_i32(
                ctx,
                vec![
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS as u64,
                ],
            )?,
            state: MetalTensor::zeros_i32(
                ctx,
                vec![DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
            )?,
            private_mask: MetalTensor::zeros_dtype(
                ctx,
                vec![capacity_rows as u64, 1],
                GgmlType::I8,
            )?,
            private_ids: MetalTensor::zeros_i32(ctx, vec![DEEPSEEK_V4_CSA_TOP_K as u64, 1])?,
            generation: DeepSeekV4MultigroupSelectorGeneration::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        scores: &MetalTensor,
        visible_counts: &MetalTensor,
        selected_mask: &MetalTensor,
        cache_order_ids: &MetalTensor,
        selected_counts: &MetalTensor,
        status: &MetalTensor,
        capacity_rows: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        encode_select_top_k_multigroup_full_f32(
            ctx,
            enc,
            scores,
            visible_counts,
            &self.records,
            &self.partition_plan,
            &self.state,
            &self.private_mask,
            &self.private_ids,
            selected_mask,
            cache_order_ids,
            selected_counts,
            status,
            capacity_rows,
            DEEPSEEK_V4_CSA_TOP_K,
            self.generation.take()?,
            None,
            false,
        )
    }
}

struct DeepSeekV4SparseCsaScratch {
    capacity_rows: usize,
    index_queries: MetalTensor,
    matrix_queries_f16: MetalTensor,
    head_weights: MetalTensor,
    visible_counts: MetalTensor,
    scores: MetalTensor,
    selected_mask: MetalTensor,
    cache_order_ids: MetalTensor,
    selected_counts: MetalTensor,
    status: MetalTensor,
    selector_mode: DeepSeekV4SparseSelectorMode,
    multigroup: Option<DeepSeekV4MultigroupSelectorScratch>,
    multigroup_invocations: DeepSeekV4MultigroupSelectorInvocationCounters,
    #[cfg(test)]
    score_test_policy: DeepSeekV4IndexerScoreTestPolicy,
    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    selector_test_policy: DeepSeekV4SelectorTestPolicy,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeepSeekV4IndexerScoreTestPolicy {
    Production,
    ScalarOracle,
    MatrixF16,
}

#[cfg(all(test, feature = "dsv4-diagnostics"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeepSeekV4SelectorTestPolicy {
    Production,
    BitwiseOracle,
}

impl DeepSeekV4SparseCsaScratch {
    fn new(ctx: &MetalContext, capacity_rows: usize) -> Result<Self, DeepSeekV4MetalError> {
        if capacity_rows < DEEPSEEK_V4_CSA_TOP_K
            || !capacity_rows.is_multiple_of(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS)
        {
            return invalid(format!(
                "sparse CSA scratch capacity {capacity_rows} is not an aligned top-k superset"
            ));
        }
        let multigroup = deepseek_v4_multigroup_selector_capacity_supported(capacity_rows)
            .then(|| DeepSeekV4MultigroupSelectorScratch::new(ctx, capacity_rows))
            .transpose()?;
        #[cfg(test)]
        let selector_mode = DeepSeekV4SparseSelectorMode::Radix4;
        #[cfg(not(test))]
        let selector_mode = if multigroup.is_some()
            && ctx.device.name().to_string() == DEEPSEEK_V4_MULTIGROUP_SELECTOR_QUALIFIED_DEVICE
            && deepseek_v4_multigroup_selector_enabled()
        {
            DeepSeekV4SparseSelectorMode::MultigroupProduction
        } else {
            DeepSeekV4SparseSelectorMode::Radix4
        };
        Ok(Self {
            capacity_rows,
            index_queries: MetalTensor::zeros_f32(ctx, vec![128, 64, 1])?,
            matrix_queries_f16: MetalTensor::zeros_f16(ctx, vec![128, 64, 1])?,
            head_weights: MetalTensor::zeros_f32(ctx, vec![64, 1])?,
            visible_counts: MetalTensor::zeros_i32(ctx, vec![1])?,
            scores: MetalTensor::zeros_f32(ctx, vec![capacity_rows as u64, 1])?,
            selected_mask: MetalTensor::zeros_i32(ctx, vec![capacity_rows as u64, 1])?,
            cache_order_ids: MetalTensor::zeros_i32(ctx, vec![DEEPSEEK_V4_CSA_TOP_K as u64, 1])?,
            selected_counts: MetalTensor::zeros_i32(ctx, vec![1])?,
            status: MetalTensor::zeros_i32(ctx, vec![1])?,
            selector_mode,
            multigroup,
            multigroup_invocations: DeepSeekV4MultigroupSelectorInvocationCounters::default(),
            #[cfg(test)]
            score_test_policy: DeepSeekV4IndexerScoreTestPolicy::Production,
            #[cfg(all(test, feature = "dsv4-diagnostics"))]
            selector_test_policy: DeepSeekV4SelectorTestPolicy::Production,
        })
    }

    fn enable_multigroup_selector_experiment(&mut self) -> Result<(), DeepSeekV4MetalError> {
        if self.selector_mode == DeepSeekV4SparseSelectorMode::MultigroupExperimental {
            return invalid("DeepSeek V4 sparse selector experiment is already sealed");
        }
        if self.multigroup.is_none() {
            return invalid(format!(
                "DeepSeek V4 session CSA capacity {} cannot enter the measured multi-group selector band",
                self.capacity_rows
            ));
        }
        self.selector_mode = DeepSeekV4SparseSelectorMode::MultigroupExperimental;
        Ok(())
    }

    fn disable_multigroup_selector(&mut self) -> Result<(), DeepSeekV4MetalError> {
        if self.selector_mode == DeepSeekV4SparseSelectorMode::MultigroupExperimental {
            return invalid("DeepSeek V4 sparse selector experiment is already sealed");
        }
        self.selector_mode = DeepSeekV4SparseSelectorMode::Radix4;
        Ok(())
    }

    fn multigroup_selector_telemetry(&self) -> DeepSeekV4MultigroupSelectorTelemetry {
        self.multigroup_invocations
            .telemetry(self.selector_mode == DeepSeekV4SparseSelectorMode::MultigroupExperimental)
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    fn set_score_test_policy(&mut self, policy: DeepSeekV4IndexerScoreTestPolicy) {
        self.score_test_policy = policy;
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    fn set_selector_test_policy(&mut self, policy: DeepSeekV4SelectorTestPolicy) {
        self.selector_test_policy = policy;
    }

    fn force_scalar_score_kernel(&self) -> bool {
        #[cfg(test)]
        {
            self.score_test_policy == DeepSeekV4IndexerScoreTestPolicy::ScalarOracle
        }
        #[cfg(not(test))]
        {
            false
        }
    }

    fn use_f16_matrix_score(&self, ctx: &MetalContext, visible_rows: usize) -> bool {
        #[cfg(test)]
        {
            let _ = (ctx, visible_rows);
            self.score_test_policy == DeepSeekV4IndexerScoreTestPolicy::MatrixF16
        }
        #[cfg(not(test))]
        {
            visible_rows >= DEEPSEEK_V4_F16_MATRIX_SCORER_MIN_VISIBLE_ROWS
                && ctx.device.name().to_string() == DEEPSEEK_V4_F16_MATRIX_SCORER_QUALIFIED_DEVICE
                && deepseek_v4_f16_matrix_scorer_enabled()
        }
    }

    fn use_radix4_selector(&self) -> bool {
        #[cfg(all(test, feature = "dsv4-diagnostics"))]
        {
            self.selector_test_policy == DeepSeekV4SelectorTestPolicy::Production
        }
        #[cfg(any(not(test), all(test, not(feature = "dsv4-diagnostics"))))]
        {
            true
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        q_lora: &MetalTensor,
        normalized_input: &MetalTensor,
        indexer_q_weight: &MetalTensor,
        indexer_projection: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        position: u32,
        rope: DeepSeekV4RopeParameters,
        record: &DeepSeekV4SelectionRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        self.encode_prepare(
            ctx,
            enc,
            q_lora,
            normalized_input,
            indexer_q_weight,
            indexer_projection,
            rows,
            position,
            rope,
            record,
        )?;
        self.encode_f16_score_and_select(ctx, enc, rows, record)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_prepare(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        q_lora: &MetalTensor,
        normalized_input: &MetalTensor,
        indexer_q_weight: &MetalTensor,
        indexer_projection: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        position: u32,
        rope: DeepSeekV4RopeParameters,
        record: &DeepSeekV4SelectionRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_sparse_csa_indexer")?;
        if rows.count <= DEEPSEEK_V4_CSA_TOP_K
            || rows.count > rows.capacity_rows
            || rows.capacity_rows != self.capacity_rows
        {
            return invalid(format!(
                "sparse CSA requires 513..={} aligned rows, got count={} capacity={}",
                self.capacity_rows, rows.count, rows.capacity_rows
            ));
        }
        validate_f32(q_lora, &[1_024], false, "sparse CSA Q-LoRA input")?;
        validate_f32(
            normalized_input,
            &[DEEPSEEK_V4_HIDDEN_SIZE as u64],
            false,
            "sparse CSA normalized input",
        )?;
        validate_matvec_weight(indexer_q_weight, 1_024, 64 * 128, "indexer Q weight")?;
        validate_matvec_weight(
            indexer_projection,
            DEEPSEEK_V4_HIDDEN_SIZE,
            64,
            "indexer projection weight",
        )?;
        validate_f16(
            rows.indexer_cache,
            &[128, rows.capacity_rows as u64],
            false,
            "sparse CSA indexer cache",
        )?;
        record.validate()?;
        host_write_i32(
            &record.visible_count,
            &[rows.count as i32],
            "sparse CSA visible count",
        )?;
        encode_projection(
            ctx,
            enc,
            indexer_q_weight,
            q_lora,
            &self.index_queries,
            1_024,
            64 * 128,
            "indexer Q",
        )?;
        encode_ds4_rope_tail_adjacent_in_place(
            ctx,
            enc,
            &self.index_queries,
            position,
            rope,
            false,
        )?;
        encode_hadamard_128_rows_in_place(ctx, enc, &self.index_queries, 64)?;
        encode_projection(
            ctx,
            enc,
            indexer_projection,
            normalized_input,
            &self.head_weights,
            DEEPSEEK_V4_HIDDEN_SIZE,
            64,
            "indexer head weights",
        )?;
        encode_scale_f32_in_place(
            ctx,
            enc,
            &self.head_weights,
            1.0 / (64.0f32 * 128.0).sqrt(),
            "indexer head weights",
        )
    }

    fn encode_f16_score_and_select(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        rows: DeepSeekV4CsaRows<'_>,
        record: &DeepSeekV4SelectionRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        if self.use_f16_matrix_score(ctx, rows.count) {
            #[cfg(not(test))]
            {
                static REPORTED: std::sync::Once = std::sync::Once::new();
                REPORTED.call_once(|| {
                    eprintln!(
                        "deepseek_v4: forced F16-staged Lightning matrix scorer active for far singleton rows; disable=QWEN_DSV4_LIGHTNING_F16_MATRIX=0"
                    );
                });
            }
            encode_scatter_offset_f32_to_f16(
                ctx,
                enc,
                &self.index_queries,
                &self.matrix_queries_f16,
                0,
                64 * 128,
            )?;
            encode_lightning_indexer_scores_f16_matrix(
                ctx,
                enc,
                &self.matrix_queries_f16,
                &self.head_weights,
                rows.indexer_cache,
                &record.visible_count,
                &self.scores,
                64,
                128,
                rows.capacity_rows,
                rows.count,
                1,
            )?;
        } else {
            encode_lightning_indexer_scores_f16_with_policy(
                ctx,
                enc,
                &self.index_queries,
                &self.head_weights,
                rows.indexer_cache,
                &record.visible_count,
                &self.scores,
                64,
                128,
                rows.capacity_rows,
                1,
                self.force_scalar_score_kernel(),
            )?;
        }
        self.encode_scored_rows(ctx, enc, rows.capacity_rows, rows.count, record)
    }

    fn encode_scored_rows(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        capacity_rows: usize,
        visible_rows: usize,
        record: &DeepSeekV4SelectionRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        if self.selector_mode != DeepSeekV4SparseSelectorMode::Radix4
            && deepseek_v4_multigroup_selector_eligible(capacity_rows, visible_rows)
        {
            if self.selector_mode == DeepSeekV4SparseSelectorMode::MultigroupProduction {
                static REPORTED: std::sync::Once = std::sync::Once::new();
                REPORTED.call_once(|| {
                    eprintln!(
                        "deepseek_v4: exact multi-group selector owns the qualified far-context band; rollback=QWEN_DSV4_MULTIGROUP_SELECTOR=0"
                    );
                });
            }
            let next_invocations = self.multigroup_invocations.next_multigroup()?;
            self.multigroup
                .as_ref()
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(
                        "eligible multi-group selector has no session-owned scratch".into(),
                    )
                })?
                .encode(
                    ctx,
                    enc,
                    &self.scores,
                    &record.visible_count,
                    &self.selected_mask,
                    &self.cache_order_ids,
                    &record.selected_count,
                    &record.status,
                    capacity_rows,
                )?;
            self.multigroup_invocations
                .commit_multigroup(next_invocations);
            return Ok(());
        }
        let next_ineligible = (self.selector_mode != DeepSeekV4SparseSelectorMode::Radix4)
            .then(|| self.multigroup_invocations.next_ineligible_radix4())
            .transpose()?;
        encode_select_top_k_f32_with_policy(
            ctx,
            enc,
            &self.scores,
            &record.visible_count,
            &self.selected_mask,
            None,
            &self.cache_order_ids,
            &record.selected_count,
            &record.status,
            capacity_rows,
            visible_rows,
            DEEPSEEK_V4_CSA_TOP_K,
            1,
            DeepSeekV4SelectorDispatchPolicy::Production,
            self.use_radix4_selector(),
        )?;
        if let Some(next) = next_ineligible {
            self.multigroup_invocations.commit_ineligible_radix4(next);
        }
        Ok(())
    }

    fn default_record(&self) -> DeepSeekV4SelectionRecord {
        DeepSeekV4SelectionRecord {
            visible_count: self.visible_counts.clone(),
            selected_count: self.selected_counts.clone(),
            status: self.status.clone(),
        }
    }

    fn selection_view<'a>(
        &'a self,
        record: &'a DeepSeekV4SelectionRecord,
    ) -> DeepSeekV4CsaSelectionView<'a> {
        DeepSeekV4CsaSelectionView {
            cache_order_ids: &self.cache_order_ids,
            selected_count: &record.selected_count,
            visible_count: &record.visible_count,
        }
    }

    fn validate_completed(
        &self,
        record: &DeepSeekV4SelectionRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        record.validate()?;
        let status = host_read_i32(&record.status, "sparse CSA selection status")?;
        let count = host_read_i32(&record.selected_count, "sparse CSA selected count")?;
        if status.as_slice() != [0] || count.as_slice() != [DEEPSEEK_V4_CSA_TOP_K as i32] {
            return invalid(format!(
                "sparse CSA selection failed with status={status:?} count={count:?}"
            ));
        }
        Ok(())
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn capture_decision(
        &self,
        visible_count: usize,
        record: &DeepSeekV4SelectionRecord,
    ) -> Result<DeepSeekV4CsaDecision, DeepSeekV4MetalError> {
        record.validate()?;
        let scores = host_read_f32(&self.scores, "diagnostic sparse CSA scores")?;
        let selected_ids = host_read_i32(
            &self.cache_order_ids,
            "diagnostic sparse CSA cache-order IDs",
        )?;
        let selected_count = host_read_i32(
            &record.selected_count,
            "diagnostic sparse CSA selected count",
        )?;
        let status = host_read_i32(&record.status, "diagnostic sparse CSA status")?;
        Ok(diagnostics::build_csa_decision(
            scores,
            visible_count,
            selected_ids,
            selected_count,
            status,
        )?)
    }
}

/// Dimensions for the native, position-zero shared-KV attention body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4PositionZeroAttentionConfig {
    pub hidden_size: usize,
    pub q_lora_rank: usize,
    pub head_count: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub group_count: usize,
    pub output_rank: usize,
}

impl DeepSeekV4PositionZeroAttentionConfig {
    fn checked(self) -> Result<CheckedAttentionDims, DeepSeekV4MetalError> {
        let values = [
            ("hidden size", self.hidden_size),
            ("Q LoRA rank", self.q_lora_rank),
            ("head count", self.head_count),
            ("head dimension", self.head_dim),
            ("rotary dimension", self.rotary_dim),
            ("group count", self.group_count),
            ("output rank", self.output_rank),
        ];
        for (name, value) in values {
            if value == 0 {
                return invalid(format!("position-zero attention {name} must be nonzero"));
            }
            u32::try_from(value).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("position-zero attention {name} exceeds u32"))
            })?;
        }
        if !self.head_count.is_multiple_of(self.group_count) {
            return invalid("position-zero attention group count must divide head count");
        }
        if self.rotary_dim > self.head_dim {
            return invalid("position-zero attention rotary dimension exceeds head dimension");
        }
        let nope_dim = self.head_dim - self.rotary_dim;
        if !nope_dim.is_multiple_of(64) {
            return invalid("position-zero attention NoPE dimension must be divisible by 64");
        }
        let query_width = checked_mul(self.head_count, self.head_dim, "query width")?;
        let group_width = query_width / self.group_count;
        let low_rank_width = checked_mul(self.group_count, self.output_rank, "low-rank width")?;
        u32::try_from(query_width)
            .map_err(|_| DeepSeekV4MetalError::Invalid("query width exceeds u32".into()))?;
        u32::try_from(group_width)
            .map_err(|_| DeepSeekV4MetalError::Invalid("group width exceeds u32".into()))?;
        u32::try_from(low_rank_width)
            .map_err(|_| DeepSeekV4MetalError::Invalid("low-rank width exceeds u32".into()))?;
        Ok(CheckedAttentionDims {
            query_width,
            group_width,
            low_rank_width,
        })
    }
}

#[derive(Clone, Copy)]
struct CheckedAttentionDims {
    query_width: usize,
    group_width: usize,
    low_rank_width: usize,
}

/// Reusable F32 activation storage for one native DS4 position-zero attention
/// body. The intermediate accessors are intended for differential inspection.
pub struct DeepSeekV4PositionZeroAttentionScratch {
    config: DeepSeekV4PositionZeroAttentionConfig,
    normalized_input: MetalTensor,
    q_lora_raw: MetalTensor,
    q_lora: MetalTensor,
    queries_raw: MetalTensor,
    queries: MetalTensor,
    kv_raw: MetalTensor,
    kv: MetalTensor,
    cached_kv: MetalTensor,
    attention: MetalTensor,
    hca_partial_output: MetalTensor,
    hca_partial_ml: MetalTensor,
    low_rank: MetalTensor,
    output: MetalTensor,
    head_norm_ones: MetalTensor,
    paired_prepare_capabilities: DeepSeekV4PairedPrepareCapabilities,
    #[cfg(test)]
    hca_test_policy: DeepSeekV4HcaTestPolicy,
    #[cfg(test)]
    prepare_test_policy: DeepSeekV4PrepareTestPolicy,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeepSeekV4HcaTestPolicy {
    Production,
    #[cfg(feature = "dsv4-diagnostics")]
    GroupedOnline,
    LegacyTiled,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeepSeekV4PrepareTestPolicy {
    Production,
    Paired,
    Composed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DeepSeekV4PairedPrepareCapabilities {
    q6_projection: bool,
    q8_projection: bool,
    norm_and_rope: bool,
}

impl DeepSeekV4PairedPrepareCapabilities {
    fn probe(ctx: &MetalContext) -> Self {
        let supports = |kernel: &str, threads: usize| {
            ctx.pipeline(kernel)
                .is_ok_and(|pipeline| pipeline.maxTotalThreadsPerThreadgroup() >= threads)
        };
        let norm_threads = ctx
            .pipeline("kernel_rms_norm_mul_f32")
            .ok()
            .map(|pipeline| pipeline.maxTotalThreadsPerThreadgroup().min(1024))
            .filter(|&threads| threads > 0);
        Self {
            q6_projection: supports("kernel_ds4_prepare_projection_pair_q6_q8_f32", 128),
            q8_projection: supports("kernel_ds4_prepare_projection_pair_q8_q8_f32", 128),
            norm_and_rope: norm_threads.is_some_and(|threads| {
                supports("kernel_ds4_prepare_norm_pair_f32", threads)
                    && supports("kernel_deepseek_v4_rope_pair_in_place", 256)
            }),
        }
    }

    fn supports(self, q_dtype: GgmlType) -> bool {
        self.norm_and_rope
            && match q_dtype {
                GgmlType::Q6_K => self.q6_projection,
                GgmlType::Q8_0 => self.q8_projection,
                _ => false,
            }
    }
}

impl DeepSeekV4PositionZeroAttentionScratch {
    pub fn new(
        ctx: &MetalContext,
        config: DeepSeekV4PositionZeroAttentionConfig,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let dims = config.checked()?;
        let ones = vec![1.0f32; config.head_dim];
        #[cfg(test)]
        let probe_paired_prepare = true;
        #[cfg(not(test))]
        let probe_paired_prepare = deepseek_v4_decode_prepare_paired_enabled()
            && crate::metal::mat_vec_q8_0_lcpp_enabled();
        let paired_prepare_capabilities = if probe_paired_prepare {
            DeepSeekV4PairedPrepareCapabilities::probe(ctx)
        } else {
            DeepSeekV4PairedPrepareCapabilities::default()
        };
        Ok(Self {
            config,
            normalized_input: MetalTensor::zeros_f32(ctx, vec![config.hidden_size as u64])?,
            q_lora_raw: MetalTensor::zeros_f32(ctx, vec![config.q_lora_rank as u64])?,
            q_lora: MetalTensor::zeros_f32(ctx, vec![config.q_lora_rank as u64])?,
            queries_raw: MetalTensor::zeros_f32(ctx, vec![dims.query_width as u64])?,
            queries: MetalTensor::zeros_f32(
                ctx,
                vec![config.head_dim as u64, config.head_count as u64],
            )?,
            kv_raw: MetalTensor::zeros_f32(ctx, vec![config.head_dim as u64])?,
            kv: MetalTensor::zeros_f32(ctx, vec![config.head_dim as u64])?,
            cached_kv: MetalTensor::zeros_f32(ctx, vec![config.head_dim as u64])?,
            attention: MetalTensor::zeros_f32(
                ctx,
                vec![config.head_dim as u64, config.head_count as u64],
            )?,
            hca_partial_output: MetalTensor::zeros_f32(
                ctx,
                vec![
                    config.head_dim as u64,
                    config.head_count as u64,
                    DEEPSEEK_V4_SPLITK_HCA_PARTITIONS as u64,
                ],
            )?,
            hca_partial_ml: MetalTensor::zeros_f32(
                ctx,
                vec![
                    2,
                    config.head_count as u64,
                    DEEPSEEK_V4_SPLITK_HCA_PARTITIONS as u64,
                ],
            )?,
            low_rank: MetalTensor::zeros_f32(ctx, vec![dims.low_rank_width as u64])?,
            output: MetalTensor::zeros_f32(ctx, vec![config.hidden_size as u64])?,
            head_norm_ones: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&ones),
                vec![config.head_dim as u64],
                GgmlType::F32,
            )?,
            paired_prepare_capabilities,
            #[cfg(test)]
            hca_test_policy: DeepSeekV4HcaTestPolicy::Production,
            #[cfg(test)]
            prepare_test_policy: DeepSeekV4PrepareTestPolicy::Production,
        })
    }

    pub fn config(&self) -> DeepSeekV4PositionZeroAttentionConfig {
        self.config
    }

    pub fn normalized_input(&self) -> &MetalTensor {
        &self.normalized_input
    }

    pub fn q_lora_raw(&self) -> &MetalTensor {
        &self.q_lora_raw
    }

    pub fn q_lora(&self) -> &MetalTensor {
        &self.q_lora
    }

    pub fn queries(&self) -> &MetalTensor {
        &self.queries
    }

    pub fn kv_raw(&self) -> &MetalTensor {
        &self.kv_raw
    }

    pub fn kv(&self) -> &MetalTensor {
        &self.kv
    }

    pub fn cached_kv(&self) -> &MetalTensor {
        &self.cached_kv
    }

    pub fn attention_heads(&self) -> &MetalTensor {
        &self.attention
    }

    pub fn low_rank(&self) -> &MetalTensor {
        &self.low_rank
    }

    pub fn output(&self) -> &MetalTensor {
        &self.output
    }

    #[cfg(all(test, feature = "dsv4-diagnostics"))]
    fn set_hca_test_policy(&mut self, policy: DeepSeekV4HcaTestPolicy) {
        self.hca_test_policy = policy;
    }

    #[cfg(test)]
    fn set_prepare_test_policy(&mut self, policy: DeepSeekV4PrepareTestPolicy) {
        self.prepare_test_policy = policy;
    }

    fn use_paired_prepare(&self, q_a: &MetalTensor, kv_weight: &MetalTensor) -> bool {
        #[cfg(test)]
        if self.prepare_test_policy == DeepSeekV4PrepareTestPolicy::Composed {
            return false;
        }
        #[cfg(not(test))]
        if !deepseek_v4_decode_prepare_paired_enabled() {
            return false;
        }
        #[cfg(test)]
        if self.prepare_test_policy == DeepSeekV4PrepareTestPolicy::Production
            && !deepseek_v4_decode_prepare_paired_enabled()
        {
            return false;
        }
        kv_weight.dtype == GgmlType::Q8_0
            && matches!(q_a.dtype, GgmlType::Q6_K | GgmlType::Q8_0)
            && crate::metal::mat_vec_q8_0_lcpp_enabled()
            && self.paired_prepare_capabilities.supports(q_a.dtype)
    }

    fn use_online_hca(&self) -> bool {
        #[cfg(test)]
        {
            self.hca_test_policy != DeepSeekV4HcaTestPolicy::LegacyTiled
        }
        #[cfg(not(test))]
        {
            true
        }
    }

    fn use_splitk_hca(&self, ctx: &MetalContext) -> bool {
        #[cfg(test)]
        {
            let _ = ctx;
            self.hca_test_policy == DeepSeekV4HcaTestPolicy::Production
        }
        #[cfg(not(test))]
        {
            ctx.device.name().to_string() == DEEPSEEK_V4_LONG_HCA_QUALIFIED_DEVICE
        }
    }

    fn use_grouped_long_hca(&self, ctx: &MetalContext) -> bool {
        #[cfg(test)]
        {
            let _ = ctx;
            true
        }
        #[cfg(not(test))]
        {
            ctx.device.name().to_string() == DEEPSEEK_V4_LONG_HCA_QUALIFIED_DEVICE
        }
    }

    /// Encode exactly the position-zero DS4 attention body. Forward and inverse
    /// RoPE are omitted because both are identity at position zero.
    #[allow(clippy::too_many_arguments)]
    pub fn encode<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        input: &MetalTensor,
        attention_norm: &MetalTensor,
        q_a: &MetalTensor,
        q_a_norm: &MetalTensor,
        q_b: &MetalTensor,
        kv_weight: &MetalTensor,
        kv_norm: &MetalTensor,
        sinks: &MetalTensor,
        output_a: &MetalTensor,
        output_b: &MetalTensor,
        rms_eps: f32,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_position_zero_attention")?;
        validate_eps(rms_eps, "RMSNorm epsilon")?;
        let c = self.config;
        let dims = c.checked()?;
        self.validate_scratch(dims)?;

        validate_f32(input, &[c.hidden_size as u64], false, "attention input")?;
        validate_f32(
            attention_norm,
            &[c.hidden_size as u64],
            false,
            "attention norm weight",
        )?;
        validate_matvec_weight(q_a, c.hidden_size, c.q_lora_rank, "Q A weight")?;
        validate_f32(q_a_norm, &[c.q_lora_rank as u64], false, "Q A norm weight")?;
        validate_matvec_weight(q_b, c.q_lora_rank, dims.query_width, "Q B weight")?;
        validate_matvec_weight(kv_weight, c.hidden_size, c.head_dim, "KV weight")?;
        validate_f32(kv_norm, &[c.head_dim as u64], false, "KV norm weight")?;
        validate_f32(sinks, &[c.head_count as u64], false, "attention sinks")?;

        encode_rms_norm_mul_f32(
            ctx,
            enc,
            input,
            attention_norm,
            &self.normalized_input,
            rms_eps,
        )?;
        encode_projection(
            ctx,
            enc,
            q_a,
            &self.normalized_input,
            &self.q_lora_raw,
            c.hidden_size,
            c.q_lora_rank,
            "Q A",
        )?;
        encode_rms_norm_mul_f32(ctx, enc, &self.q_lora_raw, q_a_norm, &self.q_lora, rms_eps)?;
        encode_projection(
            ctx,
            enc,
            q_b,
            &self.q_lora,
            &self.queries_raw,
            c.q_lora_rank,
            dims.query_width,
            "Q B",
        )?;
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &self.queries_raw,
            &self.head_norm_ones,
            &self.queries,
            c.head_count,
            c.head_dim,
            rms_eps,
        )?;
        encode_projection(
            ctx,
            enc,
            kv_weight,
            &self.normalized_input,
            &self.kv_raw,
            c.hidden_size,
            c.head_dim,
            "KV",
        )?;
        encode_rms_norm_mul_f32(ctx, enc, &self.kv_raw, kv_norm, &self.kv, rms_eps)?;
        encode_attention_cache_roundtrip(ctx, enc, &self.kv, &self.cached_kv, c)?;
        encode_position_zero_sink_attention(
            ctx,
            enc,
            &self.queries,
            &self.cached_kv,
            sinks,
            &self.attention,
            c,
        )?;

        for group in 0..c.group_count {
            let input_view = self.attention.view_subrange(
                (group * dims.group_width) as u64,
                vec![dims.group_width as u64],
            );
            let output_view = self
                .low_rank
                .view_subrange((group * c.output_rank) as u64, vec![c.output_rank as u64]);
            let weight_view = group_weight_view(output_a, dims.group_width, c.output_rank, group)?;
            encode_projection(
                ctx,
                enc,
                &weight_view,
                &input_view,
                &output_view,
                dims.group_width,
                c.output_rank,
                "grouped output A",
            )?;
        }
        encode_projection(
            ctx,
            enc,
            output_b,
            &self.low_rank,
            &self.output,
            dims.low_rank_width,
            c.hidden_size,
            "output B",
        )?;
        Ok(&self.output)
    }

    /// Project and rotate Q/shared-KV, then publish the current raw F16 row.
    /// Compressor publication is ordered between this preparation and
    /// `encode_finish_dense_f16` so a boundary token can see its own row.
    #[allow(clippy::too_many_arguments)]
    fn encode_prepare_local_f16(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        input: &MetalTensor,
        attention_norm: &MetalTensor,
        q_a: &MetalTensor,
        q_a_norm: &MetalTensor,
        q_b: &MetalTensor,
        kv_weight: &MetalTensor,
        kv_norm: &MetalTensor,
        raw_cache: &MetalTensor,
        position: u32,
        rope: DeepSeekV4RopeParameters,
        rms_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_attention_prepare")?;
        validate_eps(rms_eps, "RMSNorm epsilon")?;
        let c = self.config;
        let dims = c.checked()?;
        self.validate_scratch(dims)?;
        validate_ds4_rope(rope, c.head_dim, c.rotary_dim)?;

        validate_f32(input, &[c.hidden_size as u64], false, "attention input")?;
        validate_f32(
            attention_norm,
            &[c.hidden_size as u64],
            false,
            "attention norm weight",
        )?;
        validate_matvec_weight(q_a, c.hidden_size, c.q_lora_rank, "Q A weight")?;
        validate_f32(q_a_norm, &[c.q_lora_rank as u64], false, "Q A norm weight")?;
        validate_matvec_weight(q_b, c.q_lora_rank, dims.query_width, "Q B weight")?;
        validate_matvec_weight(kv_weight, c.hidden_size, c.head_dim, "KV weight")?;
        validate_f32(kv_norm, &[c.head_dim as u64], false, "KV norm weight")?;
        validate_f16(
            raw_cache,
            &[c.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            true,
            "local raw cache",
        )?;

        encode_rms_norm_mul_f32(
            ctx,
            enc,
            input,
            attention_norm,
            &self.normalized_input,
            rms_eps,
        )?;
        let paired = self.use_paired_prepare(q_a, kv_weight);
        if paired {
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                eprintln!(
                    "deepseek_v4: decode Q/KV projections, norms, and RoPE run as paired dispatches; rollback=QWEN_DSV4_DECODE_PREPARE_PAIRED=0"
                );
            });
            encode_ds4_prepare_projection_pair(
                ctx,
                enc,
                q_a,
                kv_weight,
                &self.normalized_input,
                &self.q_lora_raw,
                &self.kv_raw,
                c.hidden_size,
                c.q_lora_rank,
                c.head_dim,
            )?;
            encode_ds4_prepare_norm_pair(
                ctx,
                enc,
                &self.q_lora_raw,
                q_a_norm,
                &self.q_lora,
                &self.kv_raw,
                kv_norm,
                &self.kv,
                rms_eps,
            )?;
        } else {
            encode_projection(
                ctx,
                enc,
                q_a,
                &self.normalized_input,
                &self.q_lora_raw,
                c.hidden_size,
                c.q_lora_rank,
                "Q A",
            )?;
            encode_rms_norm_mul_f32(ctx, enc, &self.q_lora_raw, q_a_norm, &self.q_lora, rms_eps)?;
        }
        encode_projection(
            ctx,
            enc,
            q_b,
            &self.q_lora,
            &self.queries_raw,
            c.q_lora_rank,
            dims.query_width,
            "Q B",
        )?;
        encode_rms_norm_batched_f32(
            ctx,
            enc,
            &self.queries_raw,
            &self.head_norm_ones,
            &self.queries,
            c.head_count,
            c.head_dim,
            rms_eps,
        )?;
        if paired {
            encode_ds4_rope_pair_in_place(ctx, enc, &self.queries, &self.kv, position, rope)?;
        } else {
            encode_projection(
                ctx,
                enc,
                kv_weight,
                &self.normalized_input,
                &self.kv_raw,
                c.hidden_size,
                c.head_dim,
                "KV",
            )?;
            encode_rms_norm_mul_f32(ctx, enc, &self.kv_raw, kv_norm, &self.kv, rms_eps)?;
            encode_ds4_rope_tail_adjacent_in_place(ctx, enc, &self.queries, position, rope, false)?;
            encode_ds4_rope_tail_adjacent_in_place(ctx, enc, &self.kv, position, rope, false)?;
        }
        let cache_slot = position as usize % DEEPSEEK_V4_LOCAL_WINDOW;
        encode_scatter_offset_f32_to_f16(
            ctx,
            enc,
            &self.kv,
            raw_cache,
            cache_slot * c.head_dim,
            c.head_dim,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_dense_attention_f16(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        raw_cache: &MetalTensor,
        compressed: Option<DeepSeekV4PublishedRows<'_>>,
        kind: AttentionKind,
        sinks: &MetalTensor,
        position: u32,
        rope: DeepSeekV4RopeParameters,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_attention_finish")?;
        let c = self.config;
        let dims = c.checked()?;
        self.validate_scratch(dims)?;
        validate_ds4_rope(rope, c.head_dim, c.rotary_dim)?;
        validate_f16(
            raw_cache,
            &[c.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            false,
            "local raw cache",
        )?;
        validate_f32(sinks, &[c.head_count as u64], false, "attention sinks")?;
        if let Some(rows) = compressed {
            validate_f16(
                rows.cache,
                &[c.head_dim as u64, rows.capacity_rows as u64],
                false,
                "compressed attention cache",
            )?;
            if rows.count == 0 || rows.count > rows.capacity_rows {
                return invalid(format!(
                    "compressed attention row count {} is out of range",
                    rows.count
                ));
            }
        }

        if let Some(rows) = compressed.filter(|rows| rows.count > DEEPSEEK_V4_HCA_TILE_ROWS) {
            if kind != AttentionKind::HeavilyCompressed {
                return invalid("tiled dense attention is only valid for HCA layers");
            }
            let queries = self
                .queries
                .view_subrange(0, vec![dims.query_width as u64, 1]);
            let output = self
                .attention
                .view_subrange(0, vec![dims.query_width as u64, 1]);
            if self.use_online_hca() {
                if self.use_splitk_hca(ctx) && deepseek_v4_splitk_hca_enabled() {
                    static REPORTED: std::sync::Once = std::sync::Once::new();
                    REPORTED.call_once(|| {
                        eprintln!(
                            "deepseek_v4: grouped split-K HCA runs eight independent history partitions; rollback=QWEN_DSV4_SPLITK_HCA=0"
                        );
                    });
                    encode_grouped_splitk_hca_f16(
                        ctx,
                        enc,
                        &queries,
                        raw_cache,
                        raw_cache,
                        DeepSeekV4RawCacheLayout::Ring,
                        rows,
                        sinks,
                        &self.hca_partial_output,
                        &self.hca_partial_ml,
                        &output,
                        position,
                        DEEPSEEK_V4_SPLITK_HCA_PARTITIONS,
                        c,
                    )?;
                } else if self.use_grouped_long_hca(ctx) && deepseek_v4_grouped_long_hca_enabled() {
                    static REPORTED: std::sync::Once = std::sync::Once::new();
                    REPORTED.call_once(|| {
                        eprintln!(
                            "deepseek_v4: grouped online HCA shares long-history rows across eight heads; rollback=QWEN_DSV4_GROUPED_LONG_HCA=0"
                        );
                    });
                    encode_grouped_online_dense_sink_attention_f16(
                        ctx,
                        enc,
                        &queries,
                        raw_cache,
                        raw_cache,
                        DeepSeekV4RawCacheLayout::Ring,
                        Some(rows),
                        sinks,
                        &output,
                        kind,
                        position,
                        1,
                        c,
                    )?;
                } else {
                    let direct_load = deepseek_v4_online_direct_load_enabled();
                    if direct_load {
                        static REPORTED: std::sync::Once = std::sync::Once::new();
                        REPORTED.call_once(|| {
                            eprintln!(
                                "deepseek_v4: online HCA loads rows directly; rollback=QWEN_DSV4_ONLINE_DIRECT_LOAD=0"
                            );
                        });
                    }
                    encode_online_dense_sink_attention_f16(
                        ctx,
                        enc,
                        &queries,
                        raw_cache,
                        raw_cache,
                        DeepSeekV4RawCacheLayout::Ring,
                        rows,
                        sinks,
                        &output,
                        position,
                        0,
                        1,
                        128,
                        direct_load,
                        c,
                    )?;
                }
            } else {
                encode_tiled_dense_sink_attention_f16(
                    ctx,
                    enc,
                    &queries,
                    raw_cache,
                    raw_cache,
                    DeepSeekV4RawCacheLayout::Ring,
                    rows,
                    sinks,
                    &output,
                    position,
                    0,
                    1,
                    128,
                    c,
                )?;
            }
        } else if position == 0 {
            if compressed.is_some() {
                return invalid("position-zero dense attention cannot have compressed rows");
            }
            encode_dense_sink_attention_f16(
                ctx,
                enc,
                &self.queries,
                raw_cache,
                None,
                sinks,
                &self.attention,
                position,
                c,
            )?;
        } else {
            let queries = self
                .queries
                .view_subrange(0, vec![dims.query_width as u64, 1]);
            let output = self
                .attention
                .view_subrange(0, vec![dims.query_width as u64, 1]);
            encode_cooperative_dense_sink_attention_f16(
                ctx,
                enc,
                &queries,
                raw_cache,
                raw_cache,
                DeepSeekV4RawCacheLayout::Ring,
                compressed,
                sinks,
                &output,
                kind,
                position,
                1,
                c,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_selected_attention_f16(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        raw_cache: &MetalTensor,
        rows: DeepSeekV4CsaRows<'_>,
        selection: DeepSeekV4CsaSelectionView<'_>,
        sinks: &MetalTensor,
        position: u32,
        rope: DeepSeekV4RopeParameters,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_selected_attention_finish")?;
        let c = self.config;
        let dims = c.checked()?;
        self.validate_scratch(dims)?;
        validate_ds4_rope(rope, c.head_dim, c.rotary_dim)?;
        selection.validate()?;
        let queries = self
            .queries
            .view_subrange(0, vec![dims.query_width as u64, 1]);
        let output = self
            .attention
            .view_subrange(0, vec![dims.query_width as u64, 1]);
        encode_cooperative_selected_sink_attention_f16(
            ctx,
            enc,
            &queries,
            raw_cache,
            raw_cache,
            DeepSeekV4RawCacheLayout::Ring,
            rows.attention_cache,
            rows.capacity_rows,
            selection.cache_order_ids,
            selection.selected_count,
            selection.visible_count,
            sinks,
            &output,
            position,
            0,
            1,
            1,
            DEEPSEEK_V4_CSA_TOP_K,
            false,
            false,
            c,
        )?;
        Ok(())
    }

    fn encode_attention_output<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        output_a: &MetalTensor,
        output_b: &MetalTensor,
        position: u32,
        rope: DeepSeekV4RopeParameters,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
        let c = self.config;
        let dims = c.checked()?;
        validate_matvec_weight(
            output_a,
            dims.group_width,
            dims.low_rank_width,
            "output A weight",
        )?;
        validate_matvec_weight(
            output_b,
            dims.low_rank_width,
            c.hidden_size,
            "output B weight",
        )?;
        encode_ds4_rope_tail_adjacent_in_place(ctx, enc, &self.attention, position, rope, true)?;

        let grouped_output_a = deepseek_v4_decode_output_grouped_enabled()
            && output_a.dtype == GgmlType::Q8_0
            && crate::metal::mat_vec_q8_0_lcpp_enabled();
        if grouped_output_a {
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                eprintln!(
                    "deepseek_v4: decode output A runs one grouped GEMV; rollback=QWEN_DSV4_DECODE_OUTPUT_GROUPED=0"
                );
            });
            crate::metal::encode_mat_vec_q8_0_grouped_f32(
                ctx,
                enc,
                output_a,
                &self.attention,
                &self.low_rank,
                dims.group_width,
                c.output_rank,
                c.group_count,
            )
            .map_err(DeepSeekV4MetalError::Metal)?;
        } else {
            for group in 0..c.group_count {
                let input_view = self.attention.view_subrange(
                    (group * dims.group_width) as u64,
                    vec![dims.group_width as u64],
                );
                let output_view = self
                    .low_rank
                    .view_subrange((group * c.output_rank) as u64, vec![c.output_rank as u64]);
                let weight_view =
                    group_weight_view(output_a, dims.group_width, c.output_rank, group)?;
                encode_projection(
                    ctx,
                    enc,
                    &weight_view,
                    &input_view,
                    &output_view,
                    dims.group_width,
                    c.output_rank,
                    "grouped output A",
                )?;
            }
        }
        encode_projection(
            ctx,
            enc,
            output_b,
            &self.low_rank,
            &self.output,
            dims.low_rank_width,
            c.hidden_size,
            "output B",
        )?;
        Ok(&self.output)
    }

    fn validate_scratch(&self, dims: CheckedAttentionDims) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        validate_f32(
            &self.normalized_input,
            &[c.hidden_size as u64],
            true,
            "normalized input scratch",
        )?;
        validate_f32(
            &self.q_lora_raw,
            &[c.q_lora_rank as u64],
            true,
            "raw Q LoRA scratch",
        )?;
        validate_f32(
            &self.q_lora,
            &[c.q_lora_rank as u64],
            true,
            "Q LoRA scratch",
        )?;
        validate_f32(
            &self.queries_raw,
            &[dims.query_width as u64],
            true,
            "raw query scratch",
        )?;
        validate_f32(
            &self.queries,
            &[c.head_dim as u64, c.head_count as u64],
            true,
            "query scratch",
        )?;
        validate_f32(&self.kv_raw, &[c.head_dim as u64], true, "raw KV scratch")?;
        validate_f32(&self.kv, &[c.head_dim as u64], true, "KV scratch")?;
        validate_f32(
            &self.cached_kv,
            &[c.head_dim as u64],
            true,
            "cached KV scratch",
        )?;
        validate_f32(
            &self.attention,
            &[c.head_dim as u64, c.head_count as u64],
            true,
            "attention scratch",
        )?;
        validate_f32(
            &self.low_rank,
            &[dims.low_rank_width as u64],
            true,
            "low-rank scratch",
        )?;
        validate_f32(
            &self.output,
            &[c.hidden_size as u64],
            true,
            "attention output scratch",
        )?;
        validate_f32(
            &self.head_norm_ones,
            &[c.head_dim as u64],
            false,
            "head norm ones scratch",
        )
    }
}

/// Session-owned storage for DeepSeek V4's four-stream manifold-constrained
/// hyper-connections. The all-ones tensor makes the existing weighted RMSNorm
/// kernel implement the required unweighted norm over the flattened `4H` row.
pub struct DeepSeekV4HyperConnectionScratch {
    hidden_size: usize,
    ones: MetalTensor,
    normalized: MetalTensor,
    mixes: MetalTensor,
    pre: MetalTensor,
    post: MetalTensor,
    combination: MetalTensor,
    collapsed: MetalTensor,
    head_mixes: MetalTensor,
    head_gates: MetalTensor,
}

impl DeepSeekV4HyperConnectionScratch {
    pub fn new(ctx: &MetalContext, hidden_size: usize) -> Result<Self, DeepSeekV4MetalError> {
        let residual_len = residual_len(hidden_size)?;
        let ones_values = vec![1.0f32; residual_len];
        Ok(Self {
            hidden_size,
            ones: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&ones_values),
                vec![residual_len as u64],
                GgmlType::F32,
            )?,
            normalized: MetalTensor::zeros_f32(ctx, vec![residual_len as u64])?,
            mixes: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_HC_PARAMETER_COUNT as u64])?,
            pre: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_CONNECTION_COUNT as u64])?,
            post: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_CONNECTION_COUNT as u64])?,
            combination: MetalTensor::zeros_f32(
                ctx,
                vec![
                    DEEPSEEK_V4_CONNECTION_COUNT as u64,
                    DEEPSEEK_V4_CONNECTION_COUNT as u64,
                ],
            )?,
            collapsed: MetalTensor::zeros_f32(ctx, vec![hidden_size as u64])?,
            head_mixes: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_CONNECTION_COUNT as u64])?,
            head_gates: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_CONNECTION_COUNT as u64])?,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn normalized(&self) -> &MetalTensor {
        &self.normalized
    }

    pub fn mixes(&self) -> &MetalTensor {
        &self.mixes
    }

    pub fn pre_gates(&self) -> &MetalTensor {
        &self.pre
    }

    pub fn post_gates(&self) -> &MetalTensor {
        &self.post
    }

    /// Source-major `[source, destination]` matrix.
    pub fn combination(&self) -> &MetalTensor {
        &self.combination
    }

    pub fn collapsed_input(&self) -> &MetalTensor {
        &self.collapsed
    }

    pub fn head_mixes(&self) -> &MetalTensor {
        &self.head_mixes
    }

    pub fn head_gates(&self) -> &MetalTensor {
        &self.head_gates
    }

    /// Repeat one embedding into four stream-major residual rows.
    pub fn encode_initial_repeat(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        embedding: &MetalTensor,
        residual: &MetalTensor,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_repeat")?;
        validate_f32(embedding, &[self.hidden_size as u64], false, "embedding")?;
        validate_f32(
            residual,
            &[self.hidden_size as u64, DEEPSEEK_V4_CONNECTION_COUNT as u64],
            true,
            "residual",
        )?;
        let pso = ctx.pipeline("kernel_deepseek_v4_hc_repeat")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &u32_hidden(self.hidden_size)?);
        enc.set_tensor(1, embedding);
        enc.set_tensor(2, residual);
        enc.dispatch(
            MTLSize {
                width: residual_len(self.hidden_size)?.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Flattened `4H` RMSNorm, `[4H,24]` projection, exact split-Sinkhorn
    /// controls, and weighted stream collapse.
    pub fn encode_pre(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residual: &MetalTensor,
        function: &MetalTensor,
        scale: &MetalTensor,
        base: &MetalTensor,
        rms_eps: f32,
        hc_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_pre")?;
        validate_eps(rms_eps, "RMSNorm epsilon")?;
        validate_eps(hc_eps, "hyper-connection epsilon")?;
        let residual_shape = [self.hidden_size as u64, DEEPSEEK_V4_CONNECTION_COUNT as u64];
        validate_f32(residual, &residual_shape, false, "residual")?;
        validate_matvec_weight(
            function,
            residual_len(self.hidden_size)?,
            DEEPSEEK_V4_HC_PARAMETER_COUNT,
            "function",
        )?;
        validate_f32(scale, &[3], false, "scale")?;
        validate_f32(
            base,
            &[DEEPSEEK_V4_HC_PARAMETER_COUNT as u64],
            false,
            "base",
        )?;
        self.validate_scratch()?;

        encode_rms_norm_mul_f32(ctx, enc, residual, &self.ones, &self.normalized, rms_eps)?;
        if function.dtype == GgmlType::F32 {
            encode_f32_projection(
                ctx,
                enc,
                function,
                &self.normalized,
                &self.mixes,
                residual_len(self.hidden_size)?,
                DEEPSEEK_V4_HC_PARAMETER_COUNT,
            )?;
        } else {
            encode_projection(
                ctx,
                enc,
                function,
                &self.normalized,
                &self.mixes,
                residual_len(self.hidden_size)?,
                DEEPSEEK_V4_HC_PARAMETER_COUNT,
                "hyper-connection function",
            )?;
        }

        let pso = ctx.pipeline("kernel_deepseek_v4_hc_controls")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &hc_eps);
        enc.set_tensor(1, &self.mixes);
        enc.set_tensor(2, scale);
        enc.set_tensor(3, base);
        enc.set_tensor(4, &self.pre);
        enc.set_tensor(5, &self.post);
        enc.set_tensor(6, &self.combination);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );

        let pso = ctx.pipeline("kernel_deepseek_v4_hc_collapse")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &u32_hidden(self.hidden_size)?);
        enc.set_tensor(1, residual);
        enc.set_tensor(2, &self.pre);
        enc.set_tensor(3, &self.collapsed);
        enc.dispatch(
            MTLSize {
                width: self.hidden_size.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Apply the most recently encoded pre controls to a block output and its
    /// source residual. The matrix is consumed as `source*4 + destination`.
    pub fn encode_post(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        block_output: &MetalTensor,
        residual: &MetalTensor,
        output: &MetalTensor,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_post")?;
        validate_f32(
            block_output,
            &[self.hidden_size as u64],
            false,
            "block output",
        )?;
        let shape = [self.hidden_size as u64, DEEPSEEK_V4_CONNECTION_COUNT as u64];
        validate_f32(residual, &shape, false, "residual")?;
        validate_f32(output, &shape, true, "post residual")?;
        self.validate_scratch()?;
        let pso = ctx.pipeline("kernel_deepseek_v4_hc_post")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &u32_hidden(self.hidden_size)?);
        enc.set_tensor(1, block_output);
        enc.set_tensor(2, residual);
        enc.set_tensor(3, &self.post);
        enc.set_tensor(4, &self.combination);
        enc.set_tensor(5, output);
        enc.dispatch(
            MTLSize {
                width: residual_len(self.hidden_size)?.div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Flattened norm and `[4H,4]` projection followed by sigmoid+epsilon
    /// gates and four-stream collapse for the final output head.
    pub fn encode_head(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residual: &MetalTensor,
        function: &MetalTensor,
        scale: &MetalTensor,
        base: &MetalTensor,
        output: &MetalTensor,
        rms_eps: f32,
        hc_eps: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_hc_head")?;
        validate_eps(rms_eps, "RMSNorm epsilon")?;
        validate_eps(hc_eps, "hyper-connection epsilon")?;
        let shape = [self.hidden_size as u64, DEEPSEEK_V4_CONNECTION_COUNT as u64];
        validate_f32(residual, &shape, false, "residual")?;
        validate_matvec_weight(
            function,
            residual_len(self.hidden_size)?,
            DEEPSEEK_V4_CONNECTION_COUNT,
            "head function",
        )?;
        validate_f32(scale, &[1], false, "head scale")?;
        validate_f32(
            base,
            &[DEEPSEEK_V4_CONNECTION_COUNT as u64],
            false,
            "head base",
        )?;
        validate_f32(output, &[self.hidden_size as u64], true, "head output")?;
        self.validate_scratch()?;

        encode_rms_norm_mul_f32(ctx, enc, residual, &self.ones, &self.normalized, rms_eps)?;
        if function.dtype == GgmlType::F32 {
            encode_f32_projection(
                ctx,
                enc,
                function,
                &self.normalized,
                &self.head_mixes,
                residual_len(self.hidden_size)?,
                DEEPSEEK_V4_CONNECTION_COUNT,
            )?;
        } else {
            encode_projection(
                ctx,
                enc,
                function,
                &self.normalized,
                &self.head_mixes,
                residual_len(self.hidden_size)?,
                DEEPSEEK_V4_CONNECTION_COUNT,
                "hyper-connection head function",
            )?;
        }
        let pso = ctx.pipeline("kernel_deepseek_v4_hc_head")?;
        enc.set_pipeline(&pso);
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Args {
            hidden_size: u32,
            eps: f32,
        }
        enc.set_bytes(
            0,
            &Args {
                hidden_size: u32_hidden(self.hidden_size)?,
                eps: hc_eps,
            },
        );
        enc.set_tensor(1, residual);
        enc.set_tensor(2, &self.head_mixes);
        enc.set_tensor(3, scale);
        enc.set_tensor(4, base);
        enc.set_tensor(5, &self.head_gates);
        enc.set_tensor(6, output);
        enc.dispatch(
            MTLSize {
                width: self
                    .hidden_size
                    .max(DEEPSEEK_V4_CONNECTION_COUNT)
                    .div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    fn validate_scratch(&self) -> Result<(), DeepSeekV4MetalError> {
        let residual_len = residual_len(self.hidden_size)? as u64;
        validate_f32(&self.ones, &[residual_len], false, "ones scratch")?;
        validate_f32(
            &self.normalized,
            &[residual_len],
            true,
            "normalized scratch",
        )?;
        validate_f32(&self.mixes, &[24], true, "mix scratch")?;
        validate_f32(&self.pre, &[4], true, "pre scratch")?;
        validate_f32(&self.post, &[4], true, "post scratch")?;
        validate_f32(&self.combination, &[4, 4], true, "combination scratch")?;
        validate_f32(
            &self.collapsed,
            &[self.hidden_size as u64],
            true,
            "collapse scratch",
        )?;
        validate_f32(&self.head_mixes, &[4], true, "head mix scratch")?;
        validate_f32(&self.head_gates, &[4], true, "head gate scratch")
    }
}

/// Dimensions and routing scale for the correctness-first, single-token DS4
/// MoE body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekV4MoeConfig {
    pub hidden_size: usize,
    pub ffn_size: usize,
    pub expert_count: usize,
    pub top_k: usize,
    pub routed_scale: f32,
}

impl DeepSeekV4MoeConfig {
    fn checked(self) -> Result<(), DeepSeekV4MetalError> {
        for (name, value) in [
            ("hidden size", self.hidden_size),
            ("FFN size", self.ffn_size),
            ("expert count", self.expert_count),
            ("router top-k", self.top_k),
        ] {
            if value == 0 {
                return invalid(format!("MoE {name} must be nonzero"));
            }
            u32::try_from(value)
                .map_err(|_| DeepSeekV4MetalError::Invalid(format!("MoE {name} exceeds u32")))?;
        }
        if self.top_k > self.expert_count {
            return invalid("MoE top-k must not exceed expert count");
        }
        if self.expert_count > DEEPSEEK_V4_ROUTE_MAX_EXPERTS {
            return invalid(format!(
                "MoE expert count {} exceeds GPU route capacity {DEEPSEEK_V4_ROUTE_MAX_EXPERTS}",
                self.expert_count
            ));
        }
        if self.top_k > DEEPSEEK_V4_ROUTE_MAX_TOP_K {
            return invalid(format!(
                "MoE top-k {} exceeds GPU route capacity {DEEPSEEK_V4_ROUTE_MAX_TOP_K}",
                self.top_k
            ));
        }
        if !self.routed_scale.is_finite() || self.routed_scale <= 0.0 {
            return invalid("MoE routed scale must be finite and positive");
        }
        checked_mul(self.hidden_size, self.top_k, "MoE routed output scratch")?;
        checked_mul(self.ffn_size, self.top_k, "MoE all-slot inner scratch")?;
        checked_mul(self.ffn_size, 3, "MoE gate and fused Q6 scratch")?;
        Ok(())
    }
}

/// Reusable session-owned storage for one native DS4 single-token MoE body.
///
/// Production routing publishes a transient GPU record consumed by indexed
/// expert projections in the same serial command. The host `route_*` and static
/// `encode_experts` methods remain independent differential oracles.
pub struct DeepSeekV4MoeScratch {
    config: DeepSeekV4MoeConfig,
    normalized_input: MetalTensor,
    logits: MetalTensor,
    expert_ids: MetalTensor,
    weights: MetalTensor,
    route_status: MetalTensor,
    gate: MetalTensor,
    up: MetalTensor,
    inner: MetalTensor,
    routed_inner: MetalTensor,
    expert_outputs: MetalTensor,
    routed_output: MetalTensor,
    shared_output: MetalTensor,
    final_output: MetalTensor,
}

const DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH: usize = DEEPSEEK_V4_ROUTE_MAX_TOP_K + 1;

#[derive(Clone)]
struct DeepSeekV4RouteRecord {
    expert_ids: MetalTensor,
    weights: MetalTensor,
    status: MetalTensor,
}

impl DeepSeekV4RouteRecord {
    fn validate(&self, config: DeepSeekV4MoeConfig) -> Result<(), DeepSeekV4MetalError> {
        validate_i32(
            &self.expert_ids,
            &[config.top_k as u64],
            true,
            "MoE route-record expert IDs",
        )?;
        validate_f32(
            &self.weights,
            &[config.top_k as u64],
            true,
            "MoE route-record weights",
        )?;
        validate_i32(&self.status, &[1], true, "MoE route-record status")
    }
}

struct DeepSeekV4LayerRouteRecords {
    integers: MetalTensor,
    weights: MetalTensor,
    config: DeepSeekV4MoeConfig,
}

struct DeepSeekV4CompletedLayerRouteRecords {
    integers: Vec<i32>,
    weights: Vec<f32>,
    config: DeepSeekV4MoeConfig,
}

impl DeepSeekV4LayerRouteRecords {
    fn new(ctx: &MetalContext, config: DeepSeekV4MoeConfig) -> Result<Self, DeepSeekV4MetalError> {
        config.checked()?;
        if config.top_k != DEEPSEEK_V4_ROUTE_MAX_TOP_K {
            return invalid(format!(
                "layer route records require top-k {DEEPSEEK_V4_ROUTE_MAX_TOP_K}, got {}",
                config.top_k
            ));
        }
        Ok(Self {
            integers: MetalTensor::zeros_i32(
                ctx,
                vec![
                    DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH as u64,
                    DEEPSEEK_V4_LAYER_COUNT as u64,
                ],
            )?,
            weights: MetalTensor::zeros_f32(
                ctx,
                vec![config.top_k as u64, DEEPSEEK_V4_LAYER_COUNT as u64],
            )?,
            config,
        })
    }

    fn layer(&self, layer: usize) -> Result<DeepSeekV4RouteRecord, DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!("MoE route-record layer {layer} is out of range"));
        }
        validate_i32(
            &self.integers,
            &[
                DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH as u64,
                DEEPSEEK_V4_LAYER_COUNT as u64,
            ],
            true,
            "MoE layer-route integer records",
        )?;
        validate_f32(
            &self.weights,
            &[self.config.top_k as u64, DEEPSEEK_V4_LAYER_COUNT as u64],
            true,
            "MoE layer-route weight records",
        )?;
        let integer_base = checked_mul(
            layer,
            DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH,
            "MoE route-record integer layer offset",
        )? as u64;
        let weight_base = checked_mul(
            layer,
            self.config.top_k,
            "MoE route-record weight layer offset",
        )? as u64;
        let record = DeepSeekV4RouteRecord {
            expert_ids: self
                .integers
                .view_subrange(integer_base, vec![self.config.top_k as u64]),
            weights: self
                .weights
                .view_subrange(weight_base, vec![self.config.top_k as u64]),
            status: self
                .integers
                .view_subrange(integer_base + self.config.top_k as u64, vec![1]),
        };
        record.validate(self.config)?;
        Ok(record)
    }

    fn reset_for_token(&self) -> Result<(), DeepSeekV4MetalError> {
        host_write_i32(
            &self.integers,
            &[0; DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH * DEEPSEEK_V4_LAYER_COUNT],
            "reset MoE layer-route records",
        )
    }

    fn read_completed(&self) -> Result<DeepSeekV4CompletedLayerRouteRecords, DeepSeekV4MetalError> {
        validate_i32(
            &self.integers,
            &[
                DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH as u64,
                DEEPSEEK_V4_LAYER_COUNT as u64,
            ],
            false,
            "completed MoE layer-route integer records",
        )?;
        validate_f32(
            &self.weights,
            &[self.config.top_k as u64, DEEPSEEK_V4_LAYER_COUNT as u64],
            false,
            "completed MoE layer-route weight records",
        )?;
        Ok(DeepSeekV4CompletedLayerRouteRecords {
            integers: host_read_i32(&self.integers, "completed MoE layer-route integers")?,
            weights: host_read_f32(&self.weights, "completed MoE layer-route weights")?,
            config: self.config,
        })
    }
}

impl DeepSeekV4CompletedLayerRouteRecords {
    fn validate_layer(&self, layer: usize) -> Result<(), DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!("completed MoE route layer {layer} is out of range"));
        }
        let integer_base = checked_mul(
            layer,
            DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH,
            "completed MoE route integer layer offset",
        )?;
        let weight_base = checked_mul(
            layer,
            self.config.top_k,
            "completed MoE route weight layer offset",
        )?;
        let status = self.integers[integer_base + self.config.top_k];
        if status != DEEPSEEK_V4_ROUTE_STATUS_READY {
            return invalid(format!(
                "layer {layer} GPU route failed with status {status} ({})",
                deepseek_v4_route_status_name(status)
            ));
        }
        for slot in 0..self.config.top_k {
            let expert = self.integers[integer_base + slot];
            let Ok(expert_index) = usize::try_from(expert) else {
                return invalid(format!(
                    "layer {layer} GPU route slot {slot} returned negative expert {expert}"
                ));
            };
            if expert_index >= self.config.expert_count {
                return invalid(format!(
                    "layer {layer} GPU route slot {slot} returned expert {expert_index} outside {}",
                    self.config.expert_count
                ));
            }
            let weight = self.weights[weight_base + slot];
            if !weight.is_finite() || weight < 0.0 {
                return invalid(format!(
                    "layer {layer} GPU route slot {slot} returned invalid weight {weight}"
                ));
            }
        }
        Ok(())
    }
}

impl DeepSeekV4MoeScratch {
    pub fn new(
        ctx: &MetalContext,
        config: DeepSeekV4MoeConfig,
    ) -> Result<Self, DeepSeekV4MetalError> {
        config.checked()?;
        let c = config;
        let fused_q6_scratch = checked_mul(c.ffn_size, 3, "MoE gate and fused Q6 scratch")?;
        let ids = vec![0i32; c.top_k];
        Ok(Self {
            config,
            normalized_input: MetalTensor::zeros_f32(ctx, vec![c.hidden_size as u64])?,
            logits: MetalTensor::zeros_f32(ctx, vec![c.expert_count as u64])?,
            expert_ids: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&ids),
                vec![c.top_k as u64],
                GgmlType::I32,
            )?,
            weights: MetalTensor::zeros_f32(ctx, vec![c.top_k as u64])?,
            route_status: MetalTensor::zeros_i32(ctx, vec![1])?,
            gate: MetalTensor::zeros_f32(ctx, vec![fused_q6_scratch as u64])?,
            up: MetalTensor::zeros_f32(ctx, vec![c.ffn_size as u64])?,
            inner: MetalTensor::zeros_f32(ctx, vec![c.ffn_size as u64])?,
            routed_inner: MetalTensor::zeros_f32(ctx, vec![c.ffn_size as u64, c.top_k as u64])?,
            expert_outputs: MetalTensor::zeros_f32(
                ctx,
                vec![c.hidden_size as u64, c.top_k as u64],
            )?,
            routed_output: MetalTensor::zeros_f32(ctx, vec![c.hidden_size as u64])?,
            shared_output: MetalTensor::zeros_f32(ctx, vec![c.hidden_size as u64])?,
            final_output: MetalTensor::zeros_f32(ctx, vec![c.hidden_size as u64])?,
        })
    }

    pub fn config(&self) -> DeepSeekV4MoeConfig {
        self.config
    }

    pub fn normalized_input(&self) -> &MetalTensor {
        &self.normalized_input
    }

    pub fn logits(&self) -> &MetalTensor {
        &self.logits
    }

    pub fn expert_ids(&self) -> &MetalTensor {
        &self.expert_ids
    }

    pub fn weights(&self) -> &MetalTensor {
        &self.weights
    }

    pub fn expert_outputs(&self) -> &MetalTensor {
        &self.expert_outputs
    }

    #[cfg(test)]
    fn routed_inner(&self) -> &MetalTensor {
        &self.routed_inner
    }

    pub fn routed_output(&self) -> &MetalTensor {
        &self.routed_output
    }

    pub fn shared_output(&self) -> &MetalTensor {
        &self.shared_output
    }

    pub fn final_output(&self) -> &MetalTensor {
        &self.final_output
    }

    /// Encode input RMSNorm and the `[H,E]` router projection. The returned
    /// buffers are not host-readable until the caller completes the command.
    pub fn encode_router<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        input: &MetalTensor,
        ffn_norm: &MetalTensor,
        gate_inp: &MetalTensor,
        rms_eps: f32,
    ) -> Result<(&'a MetalTensor, &'a MetalTensor), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_router")?;
        validate_eps(rms_eps, "MoE RMSNorm epsilon")?;
        let c = self.config;
        self.validate_scratch()?;
        validate_f32(input, &[c.hidden_size as u64], false, "MoE input")?;
        validate_f32(ffn_norm, &[c.hidden_size as u64], false, "MoE norm weight")?;
        validate_matvec_weight(gate_inp, c.hidden_size, c.expert_count, "MoE router weight")?;
        encode_rms_norm_mul_f32(ctx, enc, input, ffn_norm, &self.normalized_input, rms_eps)?;
        encode_projection(
            ctx,
            enc,
            gate_inp,
            &self.normalized_input,
            &self.logits,
            c.hidden_size,
            c.expert_count,
            "MoE router",
        )?;
        Ok((&self.normalized_input, &self.logits))
    }

    fn encode_route_learned_gpu(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        correction_bias: &MetalTensor,
    ) -> Result<(), DeepSeekV4MetalError> {
        let record = self.default_route_record();
        self.encode_route_learned_gpu_into(ctx, enc, correction_bias, &record)
    }

    fn encode_route_learned_gpu_into(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        correction_bias: &MetalTensor,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        validate_f32(
            correction_bias,
            &[c.expert_count as u64],
            false,
            "GPU router correction bias",
        )?;
        self.encode_route_gpu(
            ctx,
            enc,
            correction_bias,
            0,
            c.expert_count,
            "kernel_deepseek_v4_route_learned",
            DEEPSEEK_V4_ROUTE_MAX_EXPERTS,
            record,
        )
    }

    fn encode_route_hash_gpu(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        token_id: usize,
        token_to_expert: &MetalTensor,
    ) -> Result<(), DeepSeekV4MetalError> {
        let record = self.default_route_record();
        self.encode_route_hash_gpu_into(ctx, enc, token_id, token_to_expert, &record)
    }

    fn encode_route_hash_gpu_into(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        token_id: usize,
        token_to_expert: &MetalTensor,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        validate_i32_bank(token_to_expert, c.top_k, "GPU token-to-expert map")?;
        let vocab_size = usize::try_from(token_to_expert.shape[1]).map_err(|_| {
            DeepSeekV4MetalError::Invalid("GPU hash vocabulary exceeds usize".into())
        })?;
        self.encode_route_gpu(
            ctx,
            enc,
            token_to_expert,
            token_id,
            vocab_size,
            "kernel_deepseek_v4_route_hash",
            1,
            record,
        )
    }

    fn encode_route_gpu(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        auxiliary: &MetalTensor,
        token_id: usize,
        vocab_size: usize,
        kernel: &'static str,
        threads_per_group: usize,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, kernel)?;
        self.validate_scratch()?;
        let c = self.config;
        record.validate(c)?;
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Args {
            expert_count: u32,
            top_k: u32,
            token_id: u32,
            vocab_size: u32,
            routed_scale: f32,
        }
        let args = Args {
            expert_count: u32::try_from(c.expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("GPU route expert count exceeds u32".into())
            })?,
            top_k: u32::try_from(c.top_k)
                .map_err(|_| DeepSeekV4MetalError::Invalid("GPU route top-k exceeds u32".into()))?,
            token_id: u32::try_from(token_id).map_err(|_| {
                DeepSeekV4MetalError::Invalid("GPU route token ID exceeds u32".into())
            })?,
            vocab_size: u32::try_from(vocab_size).map_err(|_| {
                DeepSeekV4MetalError::Invalid("GPU route vocabulary exceeds u32".into())
            })?,
            routed_scale: c.routed_scale,
        };
        let pso = ctx.pipeline(kernel)?;
        validate_deepseek_v4_route_pipeline_geometry(
            kernel,
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup(),
            threads_per_group,
        )?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, &self.logits);
        enc.set_tensor(2, auxiliary);
        enc.set_tensor(3, &record.expert_ids);
        enc.set_tensor(4, &record.weights);
        enc.set_tensor(5, &record.status);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[cfg(test)]
    fn validate_gpu_route_completed(&self) -> Result<(), DeepSeekV4MetalError> {
        let record = self.default_route_record();
        self.validate_gpu_route_record_completed(&record)
    }

    fn validate_gpu_route_record_completed(
        &self,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        record.validate(self.config)?;
        let status = host_read_i32(&record.status, "GPU route status")?;
        if status.as_slice() != [DEEPSEEK_V4_ROUTE_STATUS_READY] {
            return invalid(format!(
                "GPU route failed with status {} ({})",
                status[0],
                deepseek_v4_route_status_name(status[0])
            ));
        }
        Ok(())
    }

    fn default_route_record(&self) -> DeepSeekV4RouteRecord {
        DeepSeekV4RouteRecord {
            expert_ids: self.expert_ids.clone(),
            weights: self.weights.clone(),
            status: self.route_status.clone(),
        }
    }

    #[cfg(test)]
    fn capture_gpu_route_record(&self) -> Result<DeepSeekV4GpuRouteRecord, DeepSeekV4MetalError> {
        let status = host_read_i32(&self.route_status, "GPU route status")?;
        Ok(DeepSeekV4GpuRouteRecord {
            status: status[0],
            expert_ids: host_read_i32(&self.expert_ids, "GPU selected expert IDs")?,
            weights: host_read_f32(&self.weights, "GPU selected expert weights")?,
        })
    }

    /// Host differential for token-major contiguous `[K,V]` I32 storage.
    pub fn route_hash(
        &self,
        token_id: usize,
        token_to_expert: &MetalTensor,
    ) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        validate_i32_bank(token_to_expert, c.top_k, "token-to-expert map")?;
        let vocab = usize::try_from(token_to_expert.shape[1])
            .map_err(|_| DeepSeekV4MetalError::Invalid("hash vocabulary exceeds usize".into()))?;
        if token_id >= vocab {
            return invalid(format!(
                "hash token id {token_id} is outside vocabulary {vocab}"
            ));
        }
        let logits = host_read_f32(&self.logits, "MoE logits")?;
        let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(&logits)
            .map_err(|error| DeepSeekV4MetalError::Invalid(format!("router scores: {error}")))?;
        let map = host_read_i32(token_to_expert, "token-to-expert map")?;
        let start = checked_mul(token_id, c.top_k, "hash route row offset")?;
        let mut selected = Vec::with_capacity(c.top_k);
        for &expert in &map[start..start + c.top_k] {
            let expert = usize::try_from(expert).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("hash route contains negative ID {expert}"))
            })?;
            if expert >= c.expert_count {
                return invalid(format!(
                    "hash route expert {expert} exceeds expert count {}",
                    c.expert_count
                ));
            }
            selected.push(expert);
        }
        let decision = crate::deepseek_v4_oracle::hash_route(&scores, &selected, c.routed_scale)
            .map_err(|error| DeepSeekV4MetalError::Invalid(format!("hash route: {error}")))?;
        self.store_route(&decision.expert_ids, &decision.weights)
    }

    /// Host differential for tie-stable learned routing. Selected weights use
    /// the unbiased `sqrt(softplus(logit))` score.
    pub fn route_learned(&self, correction_bias: &MetalTensor) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        validate_f32(
            correction_bias,
            &[c.expert_count as u64],
            false,
            "router correction bias",
        )?;
        let logits = host_read_f32(&self.logits, "MoE logits")?;
        let bias = host_read_f32(correction_bias, "router correction bias")?;
        let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(&logits)
            .map_err(|error| DeepSeekV4MetalError::Invalid(format!("router scores: {error}")))?;
        let decision =
            crate::deepseek_v4_oracle::learned_route(&scores, &bias, c.top_k, c.routed_scale)
                .map_err(|error| {
                    DeepSeekV4MetalError::Invalid(format!("learned route: {error}"))
                })?;
        self.store_route(&decision.expert_ids, &decision.weights)
    }

    /// Static-view differential for selected and shared expert execution.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_experts<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        shared_gate: &MetalTensor,
        shared_up: &MetalTensor,
        shared_down: &MetalTensor,
        expert_clamp: f32,
        shared_clamp: f32,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_experts")?;
        if !expert_clamp.is_finite() || expert_clamp <= 0.0 {
            return invalid("MoE expert clamp must be finite and positive");
        }
        if !shared_clamp.is_finite() || shared_clamp <= 0.0 {
            return invalid("MoE shared clamp must be finite and positive");
        }
        let c = self.config;
        self.validate_scratch()?;
        validate_expert_bank(
            gate_bank,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            "routed gate bank",
        )?;
        validate_expert_bank(
            up_bank,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            "routed up bank",
        )?;
        validate_expert_bank(
            down_bank,
            c.ffn_size,
            c.hidden_size,
            c.expert_count,
            "routed down bank",
        )?;
        validate_matvec_weight(shared_gate, c.hidden_size, c.ffn_size, "shared gate")?;
        validate_matvec_weight(shared_up, c.hidden_size, c.ffn_size, "shared up")?;
        validate_matvec_weight(shared_down, c.ffn_size, c.hidden_size, "shared down")?;

        let ids = host_read_i32(&self.expert_ids, "selected expert IDs")?;
        for (slot, &id) in ids.iter().enumerate() {
            let expert = usize::try_from(id).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("selected expert ID {id} is negative"))
            })?;
            if expert >= c.expert_count {
                return invalid(format!(
                    "selected expert ID {expert} exceeds expert count {}",
                    c.expert_count
                ));
            }
            let gate = expert_weight_view(
                gate_bank,
                c.hidden_size,
                c.ffn_size,
                expert,
                "routed gate slice",
            )?;
            let up = expert_weight_view(
                up_bank,
                c.hidden_size,
                c.ffn_size,
                expert,
                "routed up slice",
            )?;
            let down = expert_weight_view(
                down_bank,
                c.ffn_size,
                c.hidden_size,
                expert,
                "routed down slice",
            )?;
            let gate_scratch = self.gate.view_subrange(0, vec![c.ffn_size as u64]);
            encode_projection(
                ctx,
                enc,
                &gate,
                &self.normalized_input,
                &gate_scratch,
                c.hidden_size,
                c.ffn_size,
                "routed gate",
            )?;
            encode_projection(
                ctx,
                enc,
                &up,
                &self.normalized_input,
                &self.up,
                c.hidden_size,
                c.ffn_size,
                "routed up",
            )?;
            encode_ds4_clamped_swiglu(
                ctx,
                enc,
                &gate_scratch,
                &self.up,
                &self.inner,
                expert_clamp,
            )?;
            let output = self
                .expert_outputs
                .view_subrange((slot * c.hidden_size) as u64, vec![c.hidden_size as u64]);
            encode_projection(
                ctx,
                enc,
                &down,
                &self.inner,
                &output,
                c.ffn_size,
                c.hidden_size,
                "routed down",
            )?;
        }

        let gate_scratch = self.gate.view_subrange(0, vec![c.ffn_size as u64]);
        encode_projection(
            ctx,
            enc,
            shared_gate,
            &self.normalized_input,
            &gate_scratch,
            c.hidden_size,
            c.ffn_size,
            "shared gate",
        )?;
        encode_projection(
            ctx,
            enc,
            shared_up,
            &self.normalized_input,
            &self.up,
            c.hidden_size,
            c.ffn_size,
            "shared up",
        )?;
        encode_ds4_clamped_swiglu(ctx, enc, &gate_scratch, &self.up, &self.inner, shared_clamp)?;
        encode_projection(
            ctx,
            enc,
            shared_down,
            &self.inner,
            &self.shared_output,
            c.ffn_size,
            c.hidden_size,
            "shared down",
        )?;
        crate::metal::encode_moe_weighted_sum_f32(
            ctx,
            enc,
            &self.expert_outputs,
            &self.weights,
            &self.routed_output,
            c.hidden_size,
            c.top_k,
        )?;
        crate::metal::encode_add_f32(
            ctx,
            enc,
            &self.routed_output,
            &self.shared_output,
            &self.final_output,
        )?;
        Ok(&self.final_output)
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_indexed_experts(
        &self,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        shared_gate: &MetalTensor,
        shared_up: &MetalTensor,
        shared_down: &MetalTensor,
        expert_clamp: f32,
        shared_clamp: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        if !expert_clamp.is_finite() || expert_clamp <= 0.0 {
            return invalid("indexed MoE expert clamp must be finite and positive");
        }
        if !shared_clamp.is_finite() || shared_clamp <= 0.0 {
            return invalid("indexed MoE shared clamp must be finite and positive");
        }
        let c = self.config;
        self.validate_scratch()?;
        validate_expert_bank(
            gate_bank,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            "indexed routed gate bank",
        )?;
        validate_expert_bank(
            up_bank,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            "indexed routed up bank",
        )?;
        validate_expert_bank(
            down_bank,
            c.ffn_size,
            c.hidden_size,
            c.expert_count,
            "indexed routed down bank",
        )?;
        validate_matvec_weight(shared_gate, c.hidden_size, c.ffn_size, "shared gate")?;
        validate_matvec_weight(shared_up, c.hidden_size, c.ffn_size, "shared up")?;
        validate_matvec_weight(shared_down, c.ffn_size, c.hidden_size, "shared down")?;
        Ok(())
    }

    #[cfg(test)]
    fn encode_routed_experts_indexed(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        expert_clamp: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let record = self.default_route_record();
        self.encode_routed_experts_indexed_from_record(
            ctx,
            enc,
            gate_bank,
            up_bank,
            down_bank,
            expert_clamp,
            &record,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_routed_experts_indexed_from_record(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        expert_clamp: f32,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_routed_experts_indexed")?;
        let c = self.config;
        record.validate(c)?;
        let gate_scratch = self.gate.view_subrange(0, vec![c.ffn_size as u64]);
        for slot in 0..c.top_k {
            encode_ds4_indexed_expert_projection(
                ctx,
                enc,
                gate_bank,
                &self.normalized_input,
                &record.expert_ids,
                &record.status,
                &gate_scratch,
                c.hidden_size,
                c.ffn_size,
                c.expert_count,
                slot,
                "indexed routed gate",
            )?;
            encode_ds4_indexed_expert_projection(
                ctx,
                enc,
                up_bank,
                &self.normalized_input,
                &record.expert_ids,
                &record.status,
                &self.up,
                c.hidden_size,
                c.ffn_size,
                c.expert_count,
                slot,
                "indexed routed up",
            )?;
            encode_ds4_clamped_swiglu(
                ctx,
                enc,
                &gate_scratch,
                &self.up,
                &self.inner,
                expert_clamp,
            )?;
            let output = self
                .expert_outputs
                .view_subrange((slot * c.hidden_size) as u64, vec![c.hidden_size as u64]);
            encode_ds4_indexed_expert_projection(
                ctx,
                enc,
                down_bank,
                &self.inner,
                &record.expert_ids,
                &record.status,
                &output,
                c.ffn_size,
                c.hidden_size,
                c.expert_count,
                slot,
                "indexed routed down",
            )?;
        }
        Ok(())
    }

    fn encode_routed_experts_all_slots(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        expert_clamp: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let record = self.default_route_record();
        self.encode_routed_experts_all_slots_from_record(
            ctx,
            enc,
            gate_bank,
            up_bank,
            down_bank,
            expert_clamp,
            &record,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_routed_experts_all_slots_from_record(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        expert_clamp: f32,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_routed_experts_all_slots")?;
        let c = self.config;
        if c.top_k != DEEPSEEK_V4_ROUTE_MAX_TOP_K {
            return invalid(format!(
                "all-slot routed experts require top-k {DEEPSEEK_V4_ROUTE_MAX_TOP_K}, got {}",
                c.top_k
            ));
        }
        if !expert_clamp.is_finite() || expert_clamp <= 0.0 {
            return invalid("all-slot routed expert clamp must be finite and positive");
        }
        self.validate_scratch()?;
        record.validate(c)?;
        validate_expert_bank(
            gate_bank,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            "all-slot routed gate bank",
        )?;
        validate_expert_bank(
            up_bank,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            "all-slot routed up bank",
        )?;
        validate_expert_bank(
            down_bank,
            c.ffn_size,
            c.hidden_size,
            c.expert_count,
            "all-slot routed down bank",
        )?;
        if deepseek_v4_all_slots_q3q4_scope_qualified(
            &ctx.device.name().to_string(),
            c,
            gate_bank.dtype,
            up_bank.dtype,
            down_bank.dtype,
        ) {
            if !deepseek_v4_all_slots_q3q4_enabled() {
                return self.encode_routed_experts_indexed_from_record(
                    ctx,
                    enc,
                    gate_bank,
                    up_bank,
                    down_bank,
                    expert_clamp,
                    record,
                );
            }
            if deepseek_v4_all_slots_q3q4_fast_enabled() {
                static REPORTED_FAST: std::sync::Once = std::sync::Once::new();
                REPORTED_FAST.call_once(|| {
                    eprintln!(
                        "deepseek_v4: fast all-slot Q3_K/Q4_K routed experts active for K160 REAP; arithmetic rollback=QWEN_DSV4_ALL_SLOTS_Q3Q4_FAST=0 serial rollback=QWEN_DSV4_ALL_SLOTS_Q3Q4=0"
                    );
                });
                return self.encode_routed_experts_all_slots_q3q4_fast_from_record(
                    ctx,
                    enc,
                    gate_bank,
                    up_bank,
                    down_bank,
                    expert_clamp,
                    record,
                );
            }
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                eprintln!(
                    "deepseek_v4: all-slot Q3_K/Q4_K routed experts active for K160 REAP; rollback=QWEN_DSV4_ALL_SLOTS_Q3Q4=0"
                );
            });
        }
        if gate_bank.dtype != up_bank.dtype
            || !matches!(
                gate_bank.dtype,
                GgmlType::IQ2_XS
                    | GgmlType::IQ2_S
                    | GgmlType::IQ3_XXS
                    | GgmlType::IQ3_S
                    | GgmlType::Q3_K
            )
        {
            return invalid(format!(
                "all-slot routed gate/up require matching IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, or Q3_K storage, got {:?}/{:?}",
                gate_bank.dtype, up_bank.dtype
            ));
        }
        if !matches!(
            down_bank.dtype,
            GgmlType::IQ3_XXS | GgmlType::MXFP4 | GgmlType::Q4_K
        ) {
            return invalid(format!(
                "all-slot routed down requires IQ3_XXS, MXFP4, or Q4_K storage, got {:?}",
                down_bank.dtype
            ));
        }
        let gate_kernel = match gate_bank.dtype {
            GgmlType::IQ2_XS => "kernel_deepseek_v4_all_slots_swiglu_iq2_xs_f32_fast",
            GgmlType::IQ2_S => "kernel_deepseek_v4_all_slots_swiglu_iq2_s_f32_fast",
            GgmlType::IQ3_XXS => "kernel_deepseek_v4_all_slots_swiglu_iq3_xxs_f32_fast",
            GgmlType::IQ3_S => "kernel_deepseek_v4_all_slots_swiglu_iq3_s_f32_fast",
            GgmlType::Q3_K => "kernel_deepseek_v4_all_slots_swiglu_q3_K_f32",
            _ => unreachable!("gate/up dtype was validated above"),
        };
        let (down_kernel, down_threads) = match down_bank.dtype {
            GgmlType::IQ3_XXS => ("kernel_deepseek_v4_all_slots_down_iq3_xxs_f32_fast", 64),
            GgmlType::MXFP4 => ("kernel_deepseek_v4_all_slots_down_mxfp4_f32", 128),
            GgmlType::Q4_K => ("kernel_deepseek_v4_all_slots_down_q4_K_f32", 64),
            _ => unreachable!("down dtype was validated above"),
        };
        validate_deepseek_v4_all_slot_pipeline(ctx, gate_kernel, 64, 64)?;
        validate_deepseek_v4_all_slot_pipeline(ctx, down_kernel, down_threads, 0)?;
        encode_ds4_all_slots_gate_up_swiglu(
            ctx,
            enc,
            gate_bank,
            up_bank,
            &self.normalized_input,
            &record.expert_ids,
            &record.status,
            &self.routed_inner,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            expert_clamp,
        )?;
        encode_ds4_all_slots_down(
            ctx,
            enc,
            down_bank,
            &self.routed_inner,
            &record.expert_ids,
            &record.status,
            &self.expert_outputs,
            c.ffn_size,
            c.hidden_size,
            c.expert_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_routed_experts_all_slots_q3q4_fast_from_record(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        expert_clamp: f32,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "DeepSeek V4 fast all-slot Q3_K/Q4_K routed experts")?;
        let c = self.config;
        record.validate(c)?;
        if c.ffn_size > c.hidden_size {
            return invalid(format!(
                "fast all-slot Q3_K/Q4_K requires FFN width {} <= hidden width {} for the up-projection scratch alias",
                c.ffn_size, c.hidden_size
            ));
        }
        let up_scratch = self.expert_outputs.view_subrange(
            0,
            vec![c.ffn_size as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        );
        encode_ds4_all_slots_q3_k_fast(
            ctx,
            enc,
            gate_bank,
            &self.normalized_input,
            &record.expert_ids,
            &record.status,
            &self.routed_inner,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
        )?;
        encode_ds4_all_slots_q3_k_fast(
            ctx,
            enc,
            up_bank,
            &self.normalized_input,
            &record.expert_ids,
            &record.status,
            &up_scratch,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
        )?;
        let routed_elements = c.ffn_size.checked_mul(c.top_k).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("fast all-slot routed width overflow".into())
        })?;
        let gate = self
            .routed_inner
            .view_subrange(0, vec![routed_elements as u64]);
        let up = up_scratch.view_subrange(0, vec![routed_elements as u64]);
        encode_ds4_clamped_swiglu(ctx, enc, &gate, &up, &gate, expert_clamp)?;
        encode_ds4_all_slots_q4_k_fast(
            ctx,
            enc,
            down_bank,
            &self.routed_inner,
            &record.expert_ids,
            &record.status,
            &self.expert_outputs,
            c.ffn_size,
            c.hidden_size,
            c.expert_count,
        )
    }

    fn encode_shared_expert(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        shared_gate: &MetalTensor,
        shared_up: &MetalTensor,
        shared_down: &MetalTensor,
        shared_clamp: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_shared_expert")?;
        let c = self.config;
        let fused_q6_scratch = checked_mul(c.ffn_size, 3, "MoE gate and fused Q6 scratch")?;
        let fused = deepseek_v4_decode_shared_swiglu_enabled()
            && shared_gate.dtype == shared_up.dtype
            && matches!(shared_gate.dtype, GgmlType::Q6_K | GgmlType::Q8_0);
        let shared_inner = if fused && shared_gate.dtype == GgmlType::Q6_K {
            self.gate.view_subrange(0, vec![c.ffn_size as u64])
        } else {
            self.inner.clone()
        };
        if fused {
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                eprintln!(
                    "deepseek_v4: decode shared gate/up/clamped-SwiGLU runs one fused dispatch; rollback=QWEN_DSV4_DECODE_SHARED_SWIGLU=0"
                );
            });
            match shared_gate.dtype {
                GgmlType::Q6_K => {
                    let scratch = self.gate.view_subrange(0, vec![fused_q6_scratch as u64]);
                    crate::metal::encode_ds4_shared_swiglu_q6_k_f32(
                        ctx,
                        enc,
                        shared_gate,
                        shared_up,
                        &self.normalized_input,
                        &scratch,
                        c.hidden_size,
                        c.ffn_size,
                        shared_clamp,
                    )
                }
                GgmlType::Q8_0 => crate::metal::encode_ds4_shared_swiglu_q8_0_f32(
                    ctx,
                    enc,
                    shared_gate,
                    shared_up,
                    &self.normalized_input,
                    &shared_inner,
                    c.hidden_size,
                    c.ffn_size,
                    shared_clamp,
                ),
                _ => unreachable!("fused shared-expert dtype was qualified"),
            }
            .map_err(DeepSeekV4MetalError::Metal)?;
        } else {
            let gate_scratch = self.gate.view_subrange(0, vec![c.ffn_size as u64]);
            encode_projection(
                ctx,
                enc,
                shared_gate,
                &self.normalized_input,
                &gate_scratch,
                c.hidden_size,
                c.ffn_size,
                "shared gate",
            )?;
            encode_projection(
                ctx,
                enc,
                shared_up,
                &self.normalized_input,
                &self.up,
                c.hidden_size,
                c.ffn_size,
                "shared up",
            )?;
            encode_ds4_clamped_swiglu(
                ctx,
                enc,
                &gate_scratch,
                &self.up,
                &self.inner,
                shared_clamp,
            )?;
        }
        encode_projection(
            ctx,
            enc,
            shared_down,
            &shared_inner,
            &self.shared_output,
            c.ffn_size,
            c.hidden_size,
            "shared down",
        )?;
        Ok(())
    }

    fn encode_expert_combine<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
        let record = self.default_route_record();
        self.encode_expert_combine_from_record(ctx, enc, &record)
    }

    fn encode_expert_combine_from_record<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_expert_combine")?;
        let c = self.config;
        record.validate(c)?;
        crate::metal::encode_moe_weighted_sum_f32(
            ctx,
            enc,
            &self.expert_outputs,
            &record.weights,
            &self.routed_output,
            c.hidden_size,
            c.top_k,
        )?;
        crate::metal::encode_add_f32(
            ctx,
            enc,
            &self.routed_output,
            &self.shared_output,
            &self.final_output,
        )?;
        Ok(&self.final_output)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn encode_experts_indexed<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        shared_gate: &MetalTensor,
        shared_up: &MetalTensor,
        shared_down: &MetalTensor,
        expert_clamp: f32,
        shared_clamp: f32,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
        self.validate_indexed_experts(
            gate_bank,
            up_bank,
            down_bank,
            shared_gate,
            shared_up,
            shared_down,
            expert_clamp,
            shared_clamp,
        )?;
        self.encode_routed_experts_indexed(ctx, enc, gate_bank, up_bank, down_bank, expert_clamp)?;
        self.encode_shared_expert(ctx, enc, shared_gate, shared_up, shared_down, shared_clamp)?;
        self.encode_expert_combine(ctx, enc)
    }

    fn store_route(
        &self,
        expert_ids: &[usize],
        weights: &[f32],
    ) -> Result<(), DeepSeekV4MetalError> {
        if expert_ids.len() != self.config.top_k || weights.len() != self.config.top_k {
            return invalid("router returned an unexpected top-k length");
        }
        let ids = expert_ids
            .iter()
            .map(|&id| {
                i32::try_from(id).map_err(|_| {
                    DeepSeekV4MetalError::Invalid(format!("expert ID {id} exceeds i32"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        host_write_i32(&self.expert_ids, &ids, "selected expert IDs")?;
        host_write_f32(&self.weights, weights, "selected expert weights")
    }

    #[cfg(feature = "dsv4-diagnostics")]
    fn capture_route_decision(
        &self,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<DeepSeekV4RouteDecision, DeepSeekV4MetalError> {
        record.validate(self.config)?;
        Ok(diagnostics::build_route_decision(
            host_read_i32(&record.expert_ids, "diagnostic routed expert IDs")?,
            host_read_f32(&record.weights, "diagnostic routed expert weights")?,
            self.config.expert_count,
            self.config.routed_scale,
        )?)
    }

    fn validate_scratch(&self) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        c.checked()?;
        let fused_q6_scratch = checked_mul(c.ffn_size, 3, "MoE gate and fused Q6 scratch")?;
        validate_f32(
            &self.normalized_input,
            &[c.hidden_size as u64],
            true,
            "MoE normalized input scratch",
        )?;
        validate_f32(
            &self.logits,
            &[c.expert_count as u64],
            true,
            "MoE logits scratch",
        )?;
        validate_i32(&self.expert_ids, &[c.top_k as u64], true, "MoE ID scratch")?;
        validate_f32(&self.weights, &[c.top_k as u64], true, "MoE weight scratch")?;
        validate_i32(&self.route_status, &[1], true, "MoE route status")?;
        validate_f32(
            &self.gate,
            &[fused_q6_scratch as u64],
            true,
            "MoE gate and fused Q6 scratch",
        )?;
        for (tensor, name) in [
            (&self.up, "MoE up scratch"),
            (&self.inner, "MoE inner scratch"),
        ] {
            validate_f32(tensor, &[c.ffn_size as u64], true, name)?;
        }
        validate_f32(
            &self.routed_inner,
            &[c.ffn_size as u64, c.top_k as u64],
            true,
            "MoE all-slot inner scratch",
        )?;
        validate_f32(
            &self.expert_outputs,
            &[c.hidden_size as u64, c.top_k as u64],
            true,
            "MoE expert output scratch",
        )?;
        for (tensor, name) in [
            (&self.routed_output, "MoE routed output scratch"),
            (&self.shared_output, "MoE shared output scratch"),
            (&self.final_output, "MoE final output scratch"),
        ] {
            validate_f32(tensor, &[c.hidden_size as u64], true, name)?;
        }
        Ok(())
    }
}

crate::env_flag!(
    default_on deepseek_v4_all_slots_q3q4_enabled,
    "QWEN_DSV4_ALL_SLOTS_Q3Q4"
);

crate::env_flag!(
    default_on deepseek_v4_all_slots_q3q4_fast_enabled,
    "QWEN_DSV4_ALL_SLOTS_Q3Q4_FAST"
);

const DEEPSEEK_V4_ALL_SLOTS_Q3Q4_QUALIFIED_DEVICE: &str = "Apple M4 Max";
const DEEPSEEK_V4_ALL_SLOTS_Q3Q4_EXPERT_COUNT: usize = 160;
const DEEPSEEK_V4_ALL_SLOTS_Q3Q4_FFN_SIZE: usize = 2_048;

fn deepseek_v4_all_slots_q3q4_scope_qualified(
    device_name: &str,
    config: DeepSeekV4MoeConfig,
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    down_dtype: GgmlType,
) -> bool {
    device_name == DEEPSEEK_V4_ALL_SLOTS_Q3Q4_QUALIFIED_DEVICE
        && config.hidden_size == DEEPSEEK_V4_HIDDEN_SIZE
        && config.ffn_size == DEEPSEEK_V4_ALL_SLOTS_Q3Q4_FFN_SIZE
        && config.expert_count == DEEPSEEK_V4_ALL_SLOTS_Q3Q4_EXPERT_COUNT
        && config.top_k == DEEPSEEK_V4_ROUTE_MAX_TOP_K
        && gate_dtype == GgmlType::Q3_K
        && up_dtype == GgmlType::Q3_K
        && down_dtype == GgmlType::Q4_K
}

fn validate_deepseek_v4_route_pipeline_geometry(
    kernel: &str,
    thread_execution_width: usize,
    max_threads_per_group: usize,
    requested_threads: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if max_threads_per_group < requested_threads {
        return invalid(format!(
            "{kernel} supports only {max_threads_per_group} threads, need {requested_threads}"
        ));
    }
    if requested_threads == DEEPSEEK_V4_ROUTE_MAX_EXPERTS && thread_execution_width != 32 {
        return invalid(format!(
            "{kernel} requires 32-lane simdgroups for eight-group reduction, got {thread_execution_width}"
        ));
    }
    Ok(())
}

fn validate_deepseek_v4_all_slot_pipeline(
    ctx: &MetalContext,
    kernel: &str,
    requested_threads: usize,
    threadgroup_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < requested_threads {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and {requested_threads} threads, got width {} max {}",
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    if threadgroup_bytes > ctx.device.maxThreadgroupMemoryLength() {
        return invalid(format!(
            "{kernel} requires {threadgroup_bytes} threadgroup bytes, device allows {}",
            ctx.device.maxThreadgroupMemoryLength()
        ));
    }
    Ok(())
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct DeepSeekV4GpuRouteRecord {
    status: i32,
    expert_ids: Vec<i32>,
    weights: Vec<f32>,
}

fn deepseek_v4_route_status_name(status: i32) -> &'static str {
    match status {
        0 => "pending",
        DEEPSEEK_V4_ROUTE_STATUS_READY => "ready",
        DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_LOGIT => "non-finite logit",
        DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_BIAS => "non-finite bias or corrected score",
        DEEPSEEK_V4_ROUTE_STATUS_INVALID_TOKEN => "invalid hash token",
        DEEPSEEK_V4_ROUTE_STATUS_INVALID_EXPERT => "invalid hash expert",
        DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_WEIGHT => "non-finite normalized weight",
        _ => "unknown",
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_ds4_indexed_expert_projection(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    expert_ids: &MetalTensor,
    route_status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    slot: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, name)?;
    if expert_ids.shape.len() != 1 || expert_ids.shape[0] == 0 {
        return invalid(format!(
            "{name} expert IDs must be a nonempty I32 vector, got {:?}",
            expert_ids.shape
        ));
    }
    validate_i32(
        expert_ids,
        &expert_ids.shape,
        false,
        &format!("{name} expert IDs"),
    )?;
    validate_i32(route_status, &[1], false, &format!("{name} route status"))?;
    let top_k = usize::try_from(expert_ids.shape[0])
        .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} top-k exceeds usize")))?;
    if slot >= top_k {
        return invalid(format!("{name} slot {slot} exceeds top-k {top_k}"));
    }
    validate_expert_bank(bank, n_in, n_out, expert_count, name)?;
    validate_f32(input, &[n_in as u64], false, &format!("{name} input"))?;
    validate_f32(output, &[n_out as u64], true, &format!("{name} output"))?;

    let (kernel, rows_per_group, threads_per_group, block_size) = match bank.dtype {
        GgmlType::IQ2_XS => (
            "kernel_deepseek_v4_indexed_mat_vec_iq2_xs_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::IQ2_S => (
            "kernel_deepseek_v4_indexed_mat_vec_iq2_s_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::IQ3_XXS => (
            "kernel_deepseek_v4_indexed_mat_vec_iq3_xxs_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::IQ3_S => (
            "kernel_deepseek_v4_indexed_mat_vec_iq3_s_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::Q3_K => ("kernel_deepseek_v4_indexed_mat_vec_q3_K_f32", 8, 64, 256),
        GgmlType::Q4_K => ("kernel_deepseek_v4_indexed_mat_vec_q4_K_f32", 4, 64, 256),
        GgmlType::MXFP4 => ("kernel_deepseek_v4_indexed_mat_vec_mxfp4_f32", 4, 128, 32),
        dtype => {
            return invalid(format!(
                "{name} has unsupported indexed expert dtype {dtype:?}"
            ));
        }
    };
    if !n_in.is_multiple_of(block_size) {
        return invalid(format!(
            "{name} input width {n_in} is not divisible by block size {block_size}"
        ));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        slot: u32,
    }
    let args = Args {
        n_in: u32::try_from(n_in)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} n_in exceeds u32")))?,
        n_out: u32::try_from(n_out)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} n_out exceeds u32")))?,
        n_expert: u32::try_from(expert_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid(format!("{name} expert count exceeds u32"))
        })?,
        slot: u32::try_from(slot)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} slot exceeds u32")))?,
    };
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < threads_per_group {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and {threads_per_group} threads, got width {} max {}",
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, expert_ids);
    enc.set_tensor(4, route_status);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(rows_per_group),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_ds4_all_slots_gate_up_swiglu(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    input: &MetalTensor,
    expert_ids: &MetalTensor,
    route_status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    clamp: f32,
) -> Result<(), DeepSeekV4MetalError> {
    const NAME: &str = "DeepSeek V4 all-slot gate/up SwiGLU";
    require_serial(enc, NAME)?;
    if !clamp.is_finite() || clamp <= 0.0 {
        return invalid(format!("{NAME} clamp must be finite and positive"));
    }
    validate_i32(
        expert_ids,
        &[DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        false,
        &format!("{NAME} expert IDs"),
    )?;
    validate_i32(route_status, &[1], false, &format!("{NAME} route status"))?;
    validate_expert_bank(gate_bank, n_in, n_out, expert_count, "all-slot gate bank")?;
    validate_expert_bank(up_bank, n_in, n_out, expert_count, "all-slot up bank")?;
    validate_f32(input, &[n_in as u64], false, &format!("{NAME} input"))?;
    validate_f32(
        output,
        &[n_out as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        true,
        &format!("{NAME} output"),
    )?;
    if gate_bank.dtype != up_bank.dtype {
        return invalid(format!(
            "{NAME} requires matching gate/up dtypes, got {:?} and {:?}",
            gate_bank.dtype, up_bank.dtype
        ));
    }
    let kernel = match gate_bank.dtype {
        GgmlType::IQ2_XS => "kernel_deepseek_v4_all_slots_swiglu_iq2_xs_f32_fast",
        GgmlType::IQ2_S => "kernel_deepseek_v4_all_slots_swiglu_iq2_s_f32_fast",
        GgmlType::IQ3_XXS => "kernel_deepseek_v4_all_slots_swiglu_iq3_xxs_f32_fast",
        GgmlType::IQ3_S => "kernel_deepseek_v4_all_slots_swiglu_iq3_s_f32_fast",
        GgmlType::Q3_K => "kernel_deepseek_v4_all_slots_swiglu_q3_K_f32",
        dtype => return invalid(format!("{NAME} does not support {dtype:?}")),
    };
    if !n_in.is_multiple_of(256) {
        return invalid(format!("{NAME} input width {n_in} is not divisible by 256"));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        top_k: u32,
        clamp: f32,
    }
    let args = Args {
        n_in: u32::try_from(n_in)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{NAME} n_in exceeds u32")))?,
        n_out: u32::try_from(n_out)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{NAME} n_out exceeds u32")))?,
        n_expert: u32::try_from(expert_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid(format!("{NAME} expert count exceeds u32"))
        })?,
        top_k: DEEPSEEK_V4_ROUTE_MAX_TOP_K as u32,
        clamp,
    };
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 64 {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and 64 threads, got width {} max {}",
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, gate_bank);
    enc.set_tensor(2, up_bank);
    enc.set_tensor(3, input);
    enc.set_tensor(4, expert_ids);
    enc.set_tensor(5, route_status);
    enc.set_tensor(6, output);
    enc.set_threadgroup_memory(0, 16 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(8),
            height: DEEPSEEK_V4_ROUTE_MAX_TOP_K,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_ds4_all_slots_down(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    expert_ids: &MetalTensor,
    route_status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const NAME: &str = "DeepSeek V4 all-slot down";
    require_serial(enc, NAME)?;
    validate_i32(
        expert_ids,
        &[DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        false,
        &format!("{NAME} expert IDs"),
    )?;
    validate_i32(route_status, &[1], false, &format!("{NAME} route status"))?;
    validate_expert_bank(bank, n_in, n_out, expert_count, "all-slot down bank")?;
    validate_f32(
        input,
        &[n_in as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        false,
        &format!("{NAME} input"),
    )?;
    validate_f32(
        output,
        &[n_out as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        true,
        &format!("{NAME} output"),
    )?;
    let (kernel, rows_per_group, threads_per_group, block_size) = match bank.dtype {
        GgmlType::IQ3_XXS => (
            "kernel_deepseek_v4_all_slots_down_iq3_xxs_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::MXFP4 => ("kernel_deepseek_v4_all_slots_down_mxfp4_f32", 4, 128, 32),
        GgmlType::Q4_K => ("kernel_deepseek_v4_all_slots_down_q4_K_f32", 4, 64, 256),
        dtype => return invalid(format!("{NAME} does not support {dtype:?}")),
    };
    if !n_in.is_multiple_of(block_size) {
        return invalid(format!(
            "{NAME} input width {n_in} is not divisible by {block_size}"
        ));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        top_k: u32,
        clamp: f32,
    }
    let args = Args {
        n_in: u32::try_from(n_in)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{NAME} n_in exceeds u32")))?,
        n_out: u32::try_from(n_out)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{NAME} n_out exceeds u32")))?,
        n_expert: u32::try_from(expert_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid(format!("{NAME} expert count exceeds u32"))
        })?,
        top_k: DEEPSEEK_V4_ROUTE_MAX_TOP_K as u32,
        clamp: 0.0,
    };
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < threads_per_group {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and {threads_per_group} threads, got width {} max {}",
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, expert_ids);
    enc.set_tensor(4, route_status);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(rows_per_group),
            height: DEEPSEEK_V4_ROUTE_MAX_TOP_K,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_ds4_all_slots_q3_k_fast(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    expert_ids: &MetalTensor,
    route_status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const NAME: &str = "DeepSeek V4 fast all-slot Q3_K projection";
    require_serial(enc, NAME)?;
    if bank.dtype != GgmlType::Q3_K {
        return invalid(format!("{NAME} requires Q3_K, got {:?}", bank.dtype));
    }
    validate_expert_bank(bank, n_in, n_out, expert_count, NAME)?;
    validate_f32(input, &[n_in as u64], false, &format!("{NAME} input"))?;
    validate_i32(
        expert_ids,
        &[DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        false,
        &format!("{NAME} expert IDs"),
    )?;
    validate_i32(route_status, &[1], false, &format!("{NAME} route status"))?;
    validate_f32(
        output,
        &[n_out as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        true,
        &format!("{NAME} output"),
    )?;
    if !n_in.is_multiple_of(256) {
        return invalid(format!("{NAME} input width {n_in} is not divisible by 256"));
    }
    let args = deepseek_v4_all_slots_args(n_in, n_out, expert_count, 0.0, NAME)?;
    let kernel = "kernel_deepseek_v4_all_slots_mat_vec_q3_K_f32_fast";
    let pso = ctx.pipeline(kernel)?;
    validate_deepseek_v4_all_slot_pipeline(ctx, kernel, 64, 0)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, expert_ids);
    enc.set_tensor(4, route_status);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(4),
            height: DEEPSEEK_V4_ROUTE_MAX_TOP_K,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_ds4_all_slots_q4_k_fast(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    expert_ids: &MetalTensor,
    route_status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const NAME: &str = "DeepSeek V4 fast all-slot Q4_K projection";
    require_serial(enc, NAME)?;
    if bank.dtype != GgmlType::Q4_K {
        return invalid(format!("{NAME} requires Q4_K, got {:?}", bank.dtype));
    }
    validate_expert_bank(bank, n_in, n_out, expert_count, NAME)?;
    validate_f32(
        input,
        &[n_in as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        false,
        &format!("{NAME} input"),
    )?;
    validate_i32(
        expert_ids,
        &[DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        false,
        &format!("{NAME} expert IDs"),
    )?;
    validate_i32(route_status, &[1], false, &format!("{NAME} route status"))?;
    validate_f32(
        output,
        &[n_out as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        true,
        &format!("{NAME} output"),
    )?;
    if !n_in.is_multiple_of(256) {
        return invalid(format!("{NAME} input width {n_in} is not divisible by 256"));
    }
    let args = deepseek_v4_all_slots_args(n_in, n_out, expert_count, 0.0, NAME)?;
    let kernel = "kernel_deepseek_v4_all_slots_mat_vec_q4_K_f32_fast";
    let pso = ctx.pipeline(kernel)?;
    validate_deepseek_v4_all_slot_pipeline(ctx, kernel, 64, 0)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, expert_ids);
    enc.set_tensor(4, route_status);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(4),
            height: DEEPSEEK_V4_ROUTE_MAX_TOP_K,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DeepSeekV4AllSlotsArgs {
    n_in: u32,
    n_out: u32,
    n_expert: u32,
    top_k: u32,
    clamp: f32,
}

fn deepseek_v4_all_slots_args(
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    clamp: f32,
    name: &str,
) -> Result<DeepSeekV4AllSlotsArgs, DeepSeekV4MetalError> {
    Ok(DeepSeekV4AllSlotsArgs {
        n_in: u32::try_from(n_in)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} n_in exceeds u32")))?,
        n_out: u32::try_from(n_out)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} n_out exceeds u32")))?,
        n_expert: u32::try_from(expert_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid(format!("{name} expert count exceeds u32"))
        })?,
        top_k: DEEPSEEK_V4_ROUTE_MAX_TOP_K as u32,
        clamp,
    })
}

fn encode_ds4_clamped_swiglu(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    output: &MetalTensor,
    clamp: f32,
) -> Result<(), DeepSeekV4MetalError> {
    let n = gate.n_elements() as usize;
    validate_f32(gate, &[n as u64], false, "DS4 SwiGLU gate")?;
    validate_f32(up, &[n as u64], false, "DS4 SwiGLU up")?;
    validate_f32(output, &[n as u64], true, "DS4 SwiGLU output")?;
    let n = u32::try_from(n)
        .map_err(|_| DeepSeekV4MetalError::Invalid("DS4 SwiGLU width exceeds u32".into()))?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        clamp: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_clamped_swiglu")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &Args { n, clamp });
    enc.set_tensor(1, gate);
    enc.set_tensor(2, up);
    enc.set_tensor(3, output);
    enc.dispatch(
        MTLSize {
            width: (n as usize).div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn validate_expert_bank(
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    let shape = [n_in as u64, n_out as u64, expert_count as u64];
    if tensor.shape != shape {
        return invalid(format!(
            "{name} must have exact GGUF shape {shape:?}, got {:?}",
            tensor.shape
        ));
    }
    let (_, expert_bytes) = matvec_weight_bytes(tensor.dtype, n_in, n_out, name)?;
    let bank_bytes = checked_mul(expert_bytes, expert_count, &format!("{name} bank bytes"))?;
    if tensor.n_bytes() != bank_bytes as u64 {
        return invalid(format!(
            "{name} byte layout is not {expert_count} contiguous {expert_bytes}-byte slices"
        ));
    }
    let end = tensor
        .offset
        .checked_add(bank_bytes as u64)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range [{}, {end}) exceeds buffer length {}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    let alignment = weight_offset_alignment(tensor.dtype);
    for expert in 0..expert_count {
        let offset = tensor.offset + (expert * expert_bytes) as u64;
        if !offset.is_multiple_of(alignment) {
            return invalid(format!(
                "{name} expert {expert} offset {offset} is not aligned to {alignment} bytes"
            ));
        }
    }
    Ok(())
}

fn expert_weight_view(
    bank: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert: usize,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let expert_count = bank
        .shape
        .get(2)
        .copied()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} has no expert axis")))?;
    validate_expert_bank(bank, n_in, n_out, expert_count, name)?;
    if expert >= expert_count {
        return invalid(format!("{name} expert {expert} exceeds {expert_count}"));
    }
    let (_, expert_bytes) = matvec_weight_bytes(bank.dtype, n_in, n_out, name)?;
    let byte_offset = checked_mul(expert, expert_bytes, &format!("{name} byte offset"))?;
    let offset = bank
        .offset
        .checked_add(byte_offset as u64)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} offset overflow")))?;
    let view = MetalTensor {
        buffer: bank.buffer.clone(),
        offset,
        shape: vec![n_in as u64, n_out as u64],
        dtype: bank.dtype,
        provenance: bank.provenance(),
    };
    validate_matvec_weight(&view, n_in, n_out, name)?;
    Ok(view)
}

fn validate_i32_bank(
    tensor: &MetalTensor,
    row_width: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    if tensor.shape.len() != 2 || tensor.shape[0] != row_width as u64 || tensor.shape[1] == 0 {
        return invalid(format!(
            "{name} must be I32 [K,V] with K={row_width}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    validate_i32(tensor, &tensor.shape, false, name)
}

fn validate_i32(
    tensor: &MetalTensor,
    shape: &[u64],
    writable: bool,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    if tensor.dtype != GgmlType::I32 || tensor.shape != shape {
        return invalid(format!(
            "{name} must be I32 with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    if writable && !tensor.is_writable() {
        return invalid(format!("{name} must be writable"));
    }
    if !tensor
        .offset
        .is_multiple_of(std::mem::align_of::<i32>() as u64)
    {
        return invalid(format!(
            "{name} offset {} is not I32-aligned",
            tensor.offset
        ));
    }
    let bytes = tensor
        .n_elements()
        .checked_mul(std::mem::size_of::<i32>() as u64)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} byte size overflow")))?;
    let end = tensor
        .offset
        .checked_add(bytes)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range [{}, {end}) exceeds buffer length {}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

/// Host access is valid only after every command that writes `tensor` has
/// completed. Validation keeps the unsafe shared-buffer read to this copy.
fn host_read_f32(tensor: &MetalTensor, name: &str) -> Result<Vec<f32>, DeepSeekV4MetalError> {
    validate_f32(tensor, &tensor.shape, false, name)?;
    let len = tensor.n_elements() as usize;
    let mut values = vec![0.0f32; len];
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<f32>();
        std::ptr::copy_nonoverlapping(source, values.as_mut_ptr(), len);
    }
    Ok(values)
}

fn host_read_i32(tensor: &MetalTensor, name: &str) -> Result<Vec<i32>, DeepSeekV4MetalError> {
    validate_i32(tensor, &tensor.shape, false, name)?;
    let len = tensor.n_elements() as usize;
    let mut values = vec![0i32; len];
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<i32>();
        std::ptr::copy_nonoverlapping(source, values.as_mut_ptr(), len);
    }
    Ok(values)
}

#[cfg(feature = "dsv4-diagnostics")]
fn next_up_nonnegative(value: f64) -> Result<f64, DeepSeekV4MetalError> {
    if !value.is_finite() || value < 0.0 {
        return invalid(format!("cannot outward-round invalid norm {value}"));
    }
    Ok(f64::from_bits(value.to_bits() + 1))
}

#[cfg(feature = "dsv4-diagnostics")]
fn host_l2_norms_f32_rows(
    tensor: &MetalTensor,
    rows: usize,
    width: usize,
    name: &str,
) -> Result<Vec<f64>, DeepSeekV4MetalError> {
    let values = host_read_f32(tensor, name)?;
    let required = checked_mul(rows, width, name)?;
    if required > values.len() {
        return invalid(format!(
            "{name} requires {required} values for {rows}x{width} rows, tensor has {}",
            values.len()
        ));
    }
    values[..required]
        .chunks_exact(width)
        .map(|row| {
            let mut squared_norm = 0.0f64;
            for &value in row {
                if !value.is_finite() {
                    return invalid(format!("{name} contains a non-finite value"));
                }
                let value = f64::from(value).abs();
                let square = next_up_nonnegative(value * value)?;
                squared_norm = next_up_nonnegative(squared_norm + square)?;
            }
            next_up_nonnegative(squared_norm.sqrt())
        })
        .collect()
}

#[cfg(feature = "dsv4-diagnostics")]
fn host_l2_norms_f16_rows(
    tensor: &MetalTensor,
    rows: usize,
    width: usize,
    name: &str,
) -> Result<Vec<f64>, DeepSeekV4MetalError> {
    validate_f16(tensor, &tensor.shape, false, name)?;
    let required = checked_mul(rows, width, name)?;
    if required > tensor.n_elements() as usize {
        return invalid(format!(
            "{name} requires {required} values for {rows}x{width} rows, tensor has {}",
            tensor.n_elements()
        ));
    }
    let source = unsafe {
        tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<u16>()
    };
    (0..rows)
        .map(|row| {
            let mut squared_norm = 0.0f64;
            for column in 0..width {
                let value =
                    half::f16::from_bits(unsafe { *source.add(row * width + column) }).to_f32();
                if !value.is_finite() {
                    return invalid(format!("{name} contains a non-finite value"));
                }
                let value = f64::from(value).abs();
                let square = next_up_nonnegative(value * value)?;
                squared_norm = next_up_nonnegative(squared_norm + square)?;
            }
            next_up_nonnegative(squared_norm.sqrt())
        })
        .collect()
}

/// Host writes must occur between completed and not-yet-committed commands.
fn host_write_f32(
    tensor: &MetalTensor,
    values: &[f32],
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    validate_f32(tensor, &tensor.shape, true, name)?;
    if tensor.n_elements() != values.len() as u64 {
        return invalid(format!(
            "{name} has {} elements, host write has {}",
            tensor.n_elements(),
            values.len()
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return invalid(format!("{name} contains a non-finite value"));
    }
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<f32>();
        std::ptr::copy_nonoverlapping(values.as_ptr(), destination, values.len());
    }
    Ok(())
}

fn host_write_i32(
    tensor: &MetalTensor,
    values: &[i32],
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    validate_i32(tensor, &tensor.shape, true, name)?;
    if tensor.n_elements() != values.len() as u64 {
        return invalid(format!(
            "{name} has {} elements, host write has {}",
            tensor.n_elements(),
            values.len()
        ));
    }
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<i32>();
        std::ptr::copy_nonoverlapping(values.as_ptr(), destination, values.len());
    }
    Ok(())
}

fn checked_mul(a: usize, b: usize, name: &str) -> Result<usize, DeepSeekV4MetalError> {
    a.checked_mul(b)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} overflows usize")))
}

fn supported_matvec_dtype(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::IQ2_XS
            | GgmlType::IQ2_S
            | GgmlType::IQ3_XXS
            | GgmlType::IQ3_S
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::MXFP4
            | GgmlType::Q8_0
            | GgmlType::IQ4_NL
            | GgmlType::IQ4_XS
    )
}

fn matvec_weight_bytes(
    dtype: GgmlType,
    n_in: usize,
    n_out: usize,
    name: &str,
) -> Result<(usize, usize), DeepSeekV4MetalError> {
    if !supported_matvec_dtype(dtype) {
        return invalid(format!("{name} dtype {dtype:?} is not supported by matvec"));
    }
    let (block_elements, block_bytes) = ggml_type_layout(dtype)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} has unknown dtype layout")))?;
    let block_elements = usize::try_from(block_elements)
        .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} block size exceeds usize")))?;
    let block_bytes = usize::try_from(block_bytes)
        .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} block bytes exceed usize")))?;
    if block_elements == 0 || !n_in.is_multiple_of(block_elements) {
        return invalid(format!(
            "{name} input width {n_in} is not aligned to {block_elements}-element {:?} blocks",
            dtype
        ));
    }
    let blocks_per_row = n_in / block_elements;
    let row_bytes = checked_mul(blocks_per_row, block_bytes, &format!("{name} row bytes"))?;
    let total_bytes = checked_mul(row_bytes, n_out, &format!("{name} bytes"))?;
    Ok((row_bytes, total_bytes))
}

fn weight_offset_alignment(dtype: GgmlType) -> u64 {
    match dtype {
        GgmlType::F32 => 4,
        GgmlType::MXFP4 => 1,
        _ => 2,
    }
}

fn validate_matvec_weight(
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    let expected_shape = [n_in as u64, n_out as u64];
    if tensor.shape != expected_shape {
        return invalid(format!(
            "{name} must have shape {expected_shape:?}, got {:?}",
            tensor.shape
        ));
    }
    let alignment = weight_offset_alignment(tensor.dtype);
    if !tensor.offset.is_multiple_of(alignment) {
        return invalid(format!(
            "{name} offset {} is not aligned to {alignment} bytes for {:?}",
            tensor.offset, tensor.dtype
        ));
    }
    let (_, bytes) = matvec_weight_bytes(tensor.dtype, n_in, n_out, name)?;
    let end = tensor
        .offset
        .checked_add(bytes as u64)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range [{}, {end}) exceeds buffer length {}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

fn group_weight_view(
    weight: &MetalTensor,
    group_width: usize,
    rank: usize,
    group: usize,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let (row_bytes, group_bytes) =
        matvec_weight_bytes(weight.dtype, group_width, rank, "grouped output A slice")?;
    let byte_offset = checked_mul(group, group_bytes, "group output A byte offset")?;
    if !byte_offset.is_multiple_of(row_bytes) {
        return invalid("group output A byte slice does not begin on a complete weight row");
    }
    let end = byte_offset
        .checked_add(group_bytes)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("group output A slice overflow".into()))?;
    if end > weight.n_bytes() as usize {
        return invalid(format!(
            "group output A slice [{byte_offset}, {end}) exceeds {} bytes",
            weight.n_bytes()
        ));
    }
    let offset = weight
        .offset
        .checked_add(byte_offset as u64)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("group output A buffer offset overflow".into())
        })?;
    let alignment = weight_offset_alignment(weight.dtype);
    if !offset.is_multiple_of(alignment) {
        return invalid(format!(
            "group output A slice offset {offset} is not aligned to {alignment} bytes for {:?}",
            weight.dtype
        ));
    }
    Ok(MetalTensor {
        buffer: weight.buffer.clone(),
        offset,
        shape: vec![group_width as u64, rank as u64],
        dtype: weight.dtype,
        provenance: weight.provenance(),
    })
}

fn encode_projection(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    crate::metal_forward::encode_mat_vec_dispatch(ctx, enc, weight, input, output, n_in, n_out)
        .map_err(|error| match error {
            crate::metal_forward::MfError::Metal(error) => DeepSeekV4MetalError::Metal(error),
            other => DeepSeekV4MetalError::Invalid(format!("{name} projection failed: {other}")),
        })
}

#[allow(clippy::too_many_arguments)]
fn encode_ds4_prepare_projection_pair(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_weight: &MetalTensor,
    kv_weight: &MetalTensor,
    input: &MetalTensor,
    q_output: &MetalTensor,
    kv_output: &MetalTensor,
    n_in: usize,
    q_out: usize,
    kv_out: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "paired prepare projection")?;
    validate_matvec_weight(q_weight, n_in, q_out, "paired Q A weight")?;
    validate_matvec_weight(kv_weight, n_in, kv_out, "paired KV weight")?;
    validate_f32(input, &[n_in as u64], false, "paired projection input")?;
    validate_f32(q_output, &[q_out as u64], true, "paired Q A output")?;
    validate_f32(kv_output, &[kv_out as u64], true, "paired KV output")?;
    if n_in == 0
        || q_out == 0
        || kv_out == 0
        || kv_weight.dtype != GgmlType::Q8_0
        || !matches!(q_weight.dtype, GgmlType::Q6_K | GgmlType::Q8_0)
        || metal_tensor_ranges_overlap(q_output, kv_output)
        || [q_output, kv_output].iter().any(|output| {
            metal_tensor_ranges_overlap(output, q_weight)
                || metal_tensor_ranges_overlap(output, kv_weight)
                || metal_tensor_ranges_overlap(output, input)
        })
        || u32::try_from(n_in).is_err()
        || u32::try_from(q_out).is_err()
        || u32::try_from(kv_out).is_err()
    {
        return invalid(format!(
            "paired prepare projection requires Q6_K/Q8_0 Q A and Q8_0 KV weights over nonzero, distinct F32 outputs; got {:?}/{:?} {n_in} -> {q_out}/{kv_out}",
            q_weight.dtype, kv_weight.dtype,
        ));
    }
    let (kernel, q_rows_per_group) = match q_weight.dtype {
        GgmlType::Q6_K => ("kernel_ds4_prepare_projection_pair_q6_q8_f32", 4usize),
        GgmlType::Q8_0 => ("kernel_ds4_prepare_projection_pair_q8_q8_f32", 2usize),
        _ => unreachable!("paired Q A dtype was qualified"),
    };
    let pso = ctx.pipeline(kernel)?;
    if pso.maxTotalThreadsPerThreadgroup() < 128 {
        return invalid(format!(
            "paired prepare projection pipeline supports {} threads, requires 128",
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.note_read(q_weight);
    enc.note_read(kv_weight);
    enc.note_read(input);
    enc.note_write(q_output);
    enc.note_write(kv_output);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        q_out: u32,
        kv_out: u32,
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            q_out: q_out as u32,
            kv_out: kv_out as u32,
        },
    );
    enc.set_tensor(1, q_weight);
    enc.set_tensor(2, kv_weight);
    enc.set_tensor(3, input);
    enc.set_tensor(4, q_output);
    enc.set_tensor(5, kv_output);
    enc.set_threadgroup_memory(0, 32 * 2 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: q_out.div_ceil(q_rows_per_group).max(kv_out.div_ceil(2)),
            height: 1,
            depth: 2,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_ds4_prepare_norm_pair(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_input: &MetalTensor,
    q_weight: &MetalTensor,
    q_output: &MetalTensor,
    kv_input: &MetalTensor,
    kv_weight: &MetalTensor,
    kv_output: &MetalTensor,
    eps: f32,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "paired prepare RMSNorm")?;
    let q_dim = usize::try_from(q_input.n_elements())
        .map_err(|_| DeepSeekV4MetalError::Invalid("paired Q norm width exceeds usize".into()))?;
    let kv_dim = usize::try_from(kv_input.n_elements())
        .map_err(|_| DeepSeekV4MetalError::Invalid("paired KV norm width exceeds usize".into()))?;
    validate_eps(eps, "paired prepare RMSNorm epsilon")?;
    validate_f32(q_input, &[q_dim as u64], false, "paired Q norm input")?;
    validate_f32(q_weight, &[q_dim as u64], false, "paired Q norm weight")?;
    validate_f32(q_output, &[q_dim as u64], true, "paired Q norm output")?;
    validate_f32(kv_input, &[kv_dim as u64], false, "paired KV norm input")?;
    validate_f32(kv_weight, &[kv_dim as u64], false, "paired KV norm weight")?;
    validate_f32(kv_output, &[kv_dim as u64], true, "paired KV norm output")?;
    if q_dim == 0
        || kv_dim == 0
        || metal_tensor_ranges_overlap(q_output, kv_output)
        || [q_output, kv_output].iter().any(|output| {
            [q_input, q_weight, kv_input, kv_weight]
                .iter()
                .any(|input| metal_tensor_ranges_overlap(output, input))
        })
        || u32::try_from(q_dim).is_err()
        || u32::try_from(kv_dim).is_err()
    {
        return invalid("paired prepare RMSNorm requires nonzero, distinct F32 input/output rows");
    }
    let reference = ctx.pipeline("kernel_rms_norm_mul_f32")?;
    let pso = ctx.pipeline("kernel_ds4_prepare_norm_pair_f32")?;
    let tg_threads = reference.maxTotalThreadsPerThreadgroup().min(1024);
    if tg_threads == 0 || pso.maxTotalThreadsPerThreadgroup() < tg_threads {
        return invalid(format!(
            "paired prepare RMSNorm pipeline supports {} threads, exact reference requires {tg_threads}",
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.note_read(q_input);
    enc.note_read(q_weight);
    enc.note_read(kv_input);
    enc.note_read(kv_weight);
    enc.note_write(q_output);
    enc.note_write(kv_output);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        q_dim: u32,
        kv_dim: u32,
        eps: f32,
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            q_dim: q_dim as u32,
            kv_dim: kv_dim as u32,
            eps,
        },
    );
    enc.set_tensor(1, q_input);
    enc.set_tensor(2, q_weight);
    enc.set_tensor(3, q_output);
    enc.set_tensor(4, kv_input);
    enc.set_tensor(5, kv_weight);
    enc.set_tensor(6, kv_output);
    let simdgroups = tg_threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (simdgroups * std::mem::size_of::<f32>()).max(32));
    enc.dispatch(
        MTLSize {
            width: 2,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_attention_cache_roundtrip(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    output: &MetalTensor,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        width: u32,
        rotary_dim: u32,
    }

    validate_f32(input, &[config.head_dim as u64], false, "pre-cache KV")?;
    validate_f32(output, &[config.head_dim as u64], true, "cached KV")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_attention_cache_roundtrip")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            width: config.head_dim as u32,
            rotary_dim: config.rotary_dim as u32,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, output);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_position_zero_sink_attention(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    kv: &MetalTensor,
    sinks: &MetalTensor,
    output: &MetalTensor,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        scale: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_position_zero_sink_attention")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: config.head_count as u32,
            head_dim: config.head_dim as u32,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, kv);
    enc.set_tensor(3, sinks);
    enc.set_tensor(4, output);
    let width = checked_mul(
        config.head_count,
        config.head_dim,
        "attention dispatch width",
    )?;
    enc.dispatch(
        MTLSize {
            width: width.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn ds4_rope_correction_bounds(rope: DeepSeekV4RopeParameters) -> (f32, f32) {
    let (correction_low, correction_high) = if rope.scaling_factor > 1.0 {
        let correction = |rotations: f32| {
            rope.rotary_dim as f32
                * (rope.original_context_length as f32 / (rotations * 2.0 * std::f32::consts::PI))
                    .ln()
                / (2.0 * rope.theta.ln())
        };
        (
            correction(rope.beta_fast).floor().max(0.0),
            correction(rope.beta_slow)
                .ceil()
                .min((rope.rotary_dim - 1) as f32),
        )
    } else {
        (0.0, 0.0)
    };
    (correction_low, correction_high)
}

fn encode_ds4_rope_tail_adjacent_in_place(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    tensor: &MetalTensor,
    position: u32,
    rope: DeepSeekV4RopeParameters,
    inverse: bool,
) -> Result<(), DeepSeekV4MetalError> {
    validate_ds4_rope(
        rope,
        tensor.shape.first().copied().unwrap_or(0) as usize,
        rope.rotary_dim,
    )?;
    validate_f32(tensor, &tensor.shape, true, "DS4 RoPE tensor")?;
    let head_dim = usize::try_from(*tensor.shape.first().ok_or_else(|| {
        DeepSeekV4MetalError::Invalid("DS4 RoPE tensor has no head dimension".into())
    })?)
    .map_err(|_| DeepSeekV4MetalError::Invalid("DS4 RoPE head dimension exceeds usize".into()))?;
    if head_dim == 0 || !tensor.n_elements().is_multiple_of(head_dim as u64) {
        return invalid("DS4 RoPE tensor is not a complete set of heads");
    }
    let head_count = usize::try_from(tensor.n_elements() / head_dim as u64)
        .map_err(|_| DeepSeekV4MetalError::Invalid("DS4 RoPE head count exceeds usize".into()))?;
    if position == 0 {
        return Ok(());
    }

    let (correction_low, correction_high) = ds4_rope_correction_bounds(rope);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        rotary_dim: u32,
        position: u32,
        inverse: u32,
        yarn: u32,
        theta: f32,
        frequency_scale: f32,
        correction_low: f32,
        correction_high: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_rope_tail_adjacent_in_place")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: head_count as u32,
            head_dim: head_dim as u32,
            rotary_dim: rope.rotary_dim as u32,
            position,
            inverse: u32::from(inverse),
            yarn: u32::from(rope.scaling_factor > 1.0),
            theta: rope.theta,
            frequency_scale: 1.0 / rope.scaling_factor,
            correction_low,
            correction_high,
        },
    );
    enc.set_tensor(1, tensor);
    let pair_count = checked_mul(head_count, rope.rotary_dim / 2, "DS4 RoPE pair count")?;
    enc.dispatch(
        MTLSize {
            width: pair_count.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_ds4_rope_pair_in_place(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q: &MetalTensor,
    kv: &MetalTensor,
    position: u32,
    rope: DeepSeekV4RopeParameters,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "paired prepare RoPE")?;
    validate_ds4_rope(
        rope,
        q.shape.first().copied().unwrap_or(0) as usize,
        rope.rotary_dim,
    )?;
    validate_f32(q, &q.shape, true, "paired Q RoPE tensor")?;
    validate_f32(kv, &kv.shape, true, "paired KV RoPE tensor")?;
    let head_dim = usize::try_from(*q.shape.first().ok_or_else(|| {
        DeepSeekV4MetalError::Invalid("paired Q RoPE tensor has no head dimension".into())
    })?)
    .map_err(|_| {
        DeepSeekV4MetalError::Invalid("paired RoPE head dimension exceeds usize".into())
    })?;
    if head_dim == 0
        || kv.shape.first().copied() != Some(head_dim as u64)
        || !q.n_elements().is_multiple_of(head_dim as u64)
        || !kv.n_elements().is_multiple_of(head_dim as u64)
        || metal_tensor_ranges_overlap(q, kv)
    {
        return invalid(
            "paired RoPE requires distinct complete Q/KV head sets with one head width",
        );
    }
    if position == 0 {
        return Ok(());
    }
    let q_head_count = usize::try_from(q.n_elements() / head_dim as u64)
        .map_err(|_| DeepSeekV4MetalError::Invalid("paired Q head count exceeds usize".into()))?;
    let kv_head_count = usize::try_from(kv.n_elements() / head_dim as u64)
        .map_err(|_| DeepSeekV4MetalError::Invalid("paired KV head count exceeds usize".into()))?;
    u32::try_from(q.n_elements())
        .map_err(|_| DeepSeekV4MetalError::Invalid("paired Q elements exceed u32".into()))?;
    u32::try_from(kv.n_elements())
        .map_err(|_| DeepSeekV4MetalError::Invalid("paired KV elements exceed u32".into()))?;
    let pairs_per_head = rope.rotary_dim / 2;
    let q_pair_count = checked_mul(q_head_count, pairs_per_head, "paired Q RoPE pair count")?;
    let kv_pair_count = checked_mul(kv_head_count, pairs_per_head, "paired KV RoPE pair count")?;
    let pair_count = q_pair_count
        .checked_add(kv_pair_count)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("paired RoPE pair count overflow".into()))?;
    let (correction_low, correction_high) = ds4_rope_correction_bounds(rope);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        q_pair_count: u32,
        pair_count: u32,
        head_dim: u32,
        rotary_dim: u32,
        position: u32,
        inverse: u32,
        yarn: u32,
        theta: f32,
        frequency_scale: f32,
        correction_low: f32,
        correction_high: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_rope_pair_in_place")?;
    enc.note_write(q);
    enc.note_write(kv);
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            q_pair_count: u32::try_from(q_pair_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("paired Q RoPE pairs exceed u32".into())
            })?,
            pair_count: u32::try_from(pair_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("paired RoPE pairs exceed u32".into())
            })?,
            head_dim: u32::try_from(head_dim)
                .map_err(|_| DeepSeekV4MetalError::Invalid("paired head dim exceeds u32".into()))?,
            rotary_dim: rope.rotary_dim as u32,
            position,
            inverse: 0,
            yarn: u32::from(rope.scaling_factor > 1.0),
            theta: rope.theta,
            frequency_scale: 1.0 / rope.scaling_factor,
            correction_low,
            correction_high,
        },
    );
    enc.set_tensor(1, q);
    enc.set_tensor(2, kv);
    enc.dispatch(
        MTLSize {
            width: pair_count.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_ds4_rope_tail_adjacent_batch_in_place(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    tensor: &MetalTensor,
    start_position: u32,
    row_count: usize,
    position_stride: u32,
    rope: DeepSeekV4RopeParameters,
    inverse: bool,
) -> Result<(), DeepSeekV4MetalError> {
    if row_count == 0 {
        return invalid("DS4 batched RoPE requires at least one row");
    }
    if position_stride == 0 {
        return invalid("DS4 batched RoPE requires a nonzero position stride");
    }
    let head_dim = usize::try_from(*tensor.shape.first().ok_or_else(|| {
        DeepSeekV4MetalError::Invalid("DS4 batched RoPE tensor has no head dimension".into())
    })?)
    .map_err(|_| {
        DeepSeekV4MetalError::Invalid("DS4 batched RoPE head dimension exceeds usize".into())
    })?;
    validate_ds4_rope(rope, head_dim, rope.rotary_dim)?;
    validate_f32(tensor, &tensor.shape, true, "DS4 batched RoPE tensor")?;
    let row_width = tensor
        .n_elements()
        .checked_div(row_count as u64)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("DS4 batched RoPE row overflow".into()))?;
    if head_dim == 0
        || row_width == 0
        || row_width * row_count as u64 != tensor.n_elements()
        || !row_width.is_multiple_of(head_dim as u64)
    {
        return invalid("DS4 batched RoPE tensor is not a complete row-major head set");
    }
    start_position
        .checked_add(
            u32::try_from(row_count - 1)
                .map_err(|_| {
                    DeepSeekV4MetalError::Invalid("DS4 batched RoPE row count exceeds u32".into())
                })?
                .checked_mul(position_stride)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("DS4 batched RoPE position span overflow".into())
                })?,
        )
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("DS4 batched RoPE position overflow".into())
        })?;
    let head_count = usize::try_from(row_width / head_dim as u64).map_err(|_| {
        DeepSeekV4MetalError::Invalid("DS4 batched RoPE head count exceeds usize".into())
    })?;
    let (correction_low, correction_high) = ds4_rope_correction_bounds(rope);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        rotary_dim: u32,
        start_position: u32,
        row_count: u32,
        position_stride: u32,
        inverse: u32,
        yarn: u32,
        theta: f32,
        frequency_scale: f32,
        correction_low: f32,
        correction_high: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_rope_tail_adjacent_batch_in_place")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: u32::try_from(head_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("DS4 batched RoPE heads exceed u32".into())
            })?,
            head_dim: u32::try_from(head_dim).map_err(|_| {
                DeepSeekV4MetalError::Invalid("DS4 batched RoPE head dimension exceeds u32".into())
            })?,
            rotary_dim: u32::try_from(rope.rotary_dim).map_err(|_| {
                DeepSeekV4MetalError::Invalid(
                    "DS4 batched RoPE rotary dimension exceeds u32".into(),
                )
            })?,
            start_position,
            row_count: u32::try_from(row_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("DS4 batched RoPE rows exceed u32".into())
            })?,
            position_stride,
            inverse: u32::from(inverse),
            yarn: u32::from(rope.scaling_factor > 1.0),
            theta: rope.theta,
            frequency_scale: 1.0 / rope.scaling_factor,
            correction_low,
            correction_high,
        },
    );
    enc.set_tensor(1, tensor);
    let pair_count = checked_mul(
        checked_mul(row_count, head_count, "DS4 batched RoPE row heads")?,
        rope.rotary_dim / 2,
        "DS4 batched RoPE pair count",
    )?;
    enc.dispatch(
        MTLSize {
            width: pair_count.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
fn encode_local_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    sinks: &MetalTensor,
    output: &MetalTensor,
    position: u32,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    encode_dense_sink_attention_f16(
        ctx, enc, queries, raw_cache, None, sinks, output, position, config,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    compressed: Option<DeepSeekV4PublishedRows<'_>>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    position: u32,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    let query_width = checked_mul(config.head_count, config.head_dim, "local query width")?;
    validate_f32(
        queries,
        &[config.head_dim as u64, config.head_count as u64],
        false,
        "local attention queries",
    )?;
    validate_f16(
        raw_cache,
        &[config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        false,
        "local attention cache",
    )?;
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "local attention sinks",
    )?;
    validate_f32(
        output,
        &[config.head_dim as u64, config.head_count as u64],
        true,
        "local attention output",
    )?;
    let (compressed_cache, compressed_count) = if let Some(rows) = compressed {
        validate_f16(
            rows.cache,
            &[config.head_dim as u64, rows.capacity_rows as u64],
            false,
            "dense compressed attention cache",
        )?;
        if rows.count == 0 || rows.count > DEEPSEEK_V4_CSA_TOP_K || rows.count > rows.capacity_rows
        {
            return invalid(format!(
                "dense compressed attention count {} is out of range",
                rows.count
            ));
        }
        (rows.cache, rows.count)
    } else {
        (raw_cache, 0)
    };

    let visible_end = u64::from(position) + 1;
    let raw_count = visible_end.min(DEEPSEEK_V4_LOCAL_WINDOW as u64) as u32;
    let raw_start = visible_end - u64::from(raw_count);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        window: u32,
        raw_count: u32,
        raw_start: u32,
        compressed_count: u32,
        scale: f32,
    }
    let raw_start = u32::try_from(raw_start)
        .map_err(|_| DeepSeekV4MetalError::Invalid("local attention start exceeds u32".into()))?;
    let pso = ctx.pipeline("kernel_deepseek_v4_dense_sink_attention_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: config.head_count as u32,
            head_dim: config.head_dim as u32,
            window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
            raw_count,
            raw_start,
            compressed_count: compressed_count as u32,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, compressed_cache);
    enc.set_tensor(4, sinks);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: query_width.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeepSeekV4RawCacheLayout {
    Ring,
    Chunk,
}

impl DeepSeekV4RawCacheLayout {
    fn rows(self, token_count: usize) -> usize {
        match self {
            Self::Ring => DEEPSEEK_V4_LOCAL_WINDOW,
            Self::Chunk => token_count,
        }
    }

    fn is_chunk(self) -> u32 {
        u32::from(self == Self::Chunk)
    }
}

fn validate_raw_attention_caches(
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    layout: DeepSeekV4RawCacheLayout,
    head_dim: usize,
    token_count: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    validate_f16(
        raw_cache,
        &[head_dim as u64, layout.rows(token_count) as u64],
        false,
        &format!("{name} current raw cache"),
    )?;
    validate_f16(
        raw_cache_before_chunk,
        &[head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        false,
        &format!("{name} preserved raw cache"),
    )?;
    if layout == DeepSeekV4RawCacheLayout::Chunk
        && Retained::as_ptr(&raw_cache.buffer) == Retained::as_ptr(&raw_cache_before_chunk.buffer)
    {
        let raw_end = raw_cache
            .offset
            .checked_add(raw_cache.n_bytes())
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("raw cache range overflow".into()))?;
        let preserved_end = raw_cache_before_chunk
            .offset
            .checked_add(raw_cache_before_chunk.n_bytes())
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("preserved raw cache range overflow".into())
            })?;
        if raw_cache.offset < preserved_end && raw_cache_before_chunk.offset < raw_end {
            return invalid(format!(
                "{name} requires disjoint current and preserved raw caches"
            ));
        }
    }
    Ok(())
}

const DEEPSEEK_V4_GROUPED_DENSE_HEADS: usize = 8;
const DEEPSEEK_V4_GROUPED_DENSE_THREADS: usize = 256;
const DEEPSEEK_V4_GROUPED_DENSE_STAGED_ROWS: usize = 16;
const DEEPSEEK_V4_SPLITK_HCA_PARTITIONS: usize = 8;
#[cfg(not(test))]
const DEEPSEEK_V4_LONG_HCA_QUALIFIED_DEVICE: &str = "Apple M4 Max";
const DEEPSEEK_V4_GROUPED_DENSE_THREADGROUP_BYTES: usize = DEEPSEEK_V4_GROUPED_DENSE_STAGED_ROWS
    * DEEPSEEK_V4_HCA_TILE_ROWS
    * std::mem::size_of::<half::f16>();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeepSeekV4DenseAttentionKernel {
    Cooperative,
    GroupedOnline,
}

#[allow(clippy::too_many_arguments)]
fn encode_cooperative_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: Option<DeepSeekV4PublishedRows<'_>>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    kind: AttentionKind,
    start_position: u32,
    token_count: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    encode_dense_sink_attention_f16_with_kernel(
        ctx,
        enc,
        queries,
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        compressed,
        sinks,
        output,
        kind,
        start_position,
        token_count,
        config,
        DeepSeekV4DenseAttentionKernel::Cooperative,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_grouped_online_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: Option<DeepSeekV4PublishedRows<'_>>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    kind: AttentionKind,
    start_position: u32,
    token_count: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    encode_dense_sink_attention_f16_with_kernel(
        ctx,
        enc,
        queries,
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        compressed,
        sinks,
        output,
        kind,
        start_position,
        token_count,
        config,
        DeepSeekV4DenseAttentionKernel::GroupedOnline,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_grouped_splitk_hca_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: DeepSeekV4PublishedRows<'_>,
    sinks: &MetalTensor,
    partial_output: &MetalTensor,
    partial_ml: &MetalTensor,
    output: &MetalTensor,
    position: u32,
    partitions: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    const MAX_PARTITIONS: usize = 16;
    require_serial(enc, "deepseek_v4_grouped_splitk_hca_f16")?;
    validate_deepseek_v4_online_hca_request_geometry(config, 128, 1, 0, 1)?;
    if raw_cache_layout != DeepSeekV4RawCacheLayout::Ring {
        return invalid("grouped split-K HCA requires the singleton ring raw cache");
    }
    if !matches!(partitions, 4 | 8 | 16) {
        return invalid("grouped split-K HCA requires 4, 8, or 16 partitions");
    }
    let query_width = checked_mul(
        config.head_count,
        config.head_dim,
        "split-K HCA query width",
    )?;
    validate_f32(
        queries,
        &[query_width as u64, 1],
        false,
        "split-K HCA queries",
    )?;
    validate_raw_attention_caches(
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        config.head_dim,
        1,
        "split-K HCA",
    )?;
    validate_f16(
        compressed.cache,
        &[config.head_dim as u64, compressed.capacity_rows as u64],
        false,
        "split-K HCA compressed cache",
    )?;
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "split-K HCA sinks",
    )?;
    validate_f32(
        partial_output,
        &[
            config.head_dim as u64,
            config.head_count as u64,
            partitions as u64,
        ],
        true,
        "split-K HCA partial output",
    )?;
    validate_f32(
        partial_ml,
        &[2, config.head_count as u64, partitions as u64],
        true,
        "split-K HCA partial max/mass",
    )?;
    validate_f32(output, &[query_width as u64, 1], true, "split-K HCA output")?;
    let expected_rows = (position as usize + 1) / 128;
    if compressed.count != expected_rows || compressed.count > compressed.capacity_rows {
        return invalid(format!(
            "split-K HCA expected {expected_rows} compressed rows, got {}/{}",
            compressed.count, compressed.capacity_rows
        ));
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        compression_ratio: u32,
        start_position: u32,
        window: u32,
        raw_cache_is_chunk: u32,
        partitions: u32,
        scale: f32,
    }
    let args = Args {
        head_count: config.head_count as u32,
        head_dim: config.head_dim as u32,
        compression_ratio: 128,
        start_position: position,
        window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
        raw_cache_is_chunk: raw_cache_layout.is_chunk(),
        partitions: partitions as u32,
        scale: 1.0 / (config.head_dim as f32).sqrt(),
    };

    let main = ctx.pipeline("kernel_deepseek_v4_grouped_splitk_hca_main_f16")?;
    if main.threadExecutionWidth() != 32
        || main.maxTotalThreadsPerThreadgroup() < DEEPSEEK_V4_GROUPED_DENSE_THREADS
        || ctx.device.maxThreadgroupMemoryLength() < DEEPSEEK_V4_GROUPED_DENSE_THREADGROUP_BYTES
    {
        return invalid("grouped split-K HCA main pipeline does not support its launch geometry");
    }
    enc.set_pipeline(&main);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, raw_cache_before_chunk);
    enc.set_tensor(4, compressed.cache);
    enc.set_tensor(5, partial_output);
    enc.set_tensor(6, partial_ml);
    enc.set_threadgroup_memory(0, DEEPSEEK_V4_GROUPED_DENSE_THREADGROUP_BYTES);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: config.head_count / DEEPSEEK_V4_GROUPED_DENSE_HEADS,
            depth: partitions,
        },
        MTLSize {
            width: 32,
            height: DEEPSEEK_V4_GROUPED_DENSE_HEADS,
            depth: 1,
        },
    );

    let reduce = ctx.pipeline("kernel_deepseek_v4_grouped_splitk_hca_reduce_f32")?;
    if reduce.threadExecutionWidth() != 32 || reduce.maxTotalThreadsPerThreadgroup() < 32 {
        return invalid("grouped split-K HCA reducer does not support one SIMDgroup");
    }
    enc.set_pipeline(&reduce);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, partial_output);
    enc.set_tensor(2, partial_ml);
    enc.set_tensor(3, sinks);
    enc.set_tensor(4, output);
    enc.set_threadgroup_memory(0, MAX_PARTITIONS * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: config.head_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_dense_sink_attention_f16_with_kernel(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: Option<DeepSeekV4PublishedRows<'_>>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    kind: AttentionKind,
    start_position: u32,
    token_count: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
    kernel: DeepSeekV4DenseAttentionKernel,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "deepseek_v4_cooperative_dense_attention")?;
    let dims = config.checked()?;
    if token_count == 0 {
        return invalid("cooperative dense attention requires at least one token");
    }
    if token_count > DEEPSEEK_V4_PREFILL_MAX_TOKENS {
        return invalid(format!(
            "cooperative dense attention token count {token_count} exceeds retained chunk limit {DEEPSEEK_V4_PREFILL_MAX_TOKENS}"
        ));
    }
    let token_count_u32 = u32::try_from(token_count).map_err(|_| {
        DeepSeekV4MetalError::Invalid("cooperative dense token count exceeds u32".into())
    })?;
    validate_f32(
        queries,
        &[dims.query_width as u64, token_count as u64],
        false,
        "cooperative dense attention queries",
    )?;
    validate_raw_attention_caches(
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        config.head_dim,
        token_count,
        "cooperative dense attention",
    )?;
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "cooperative dense attention sinks",
    )?;
    validate_f32(
        output,
        &[dims.query_width as u64, token_count as u64],
        true,
        "cooperative dense attention output",
    )?;

    let ratio = match kind {
        AttentionKind::SlidingWindow => 0,
        AttentionKind::CompressedSparse => 4,
        AttentionKind::HeavilyCompressed => 128,
    };
    let end_position = start_position.checked_add(token_count_u32).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid("cooperative dense position overflow".into())
    })?;
    let expected_rows = (end_position as usize).checked_div(ratio).unwrap_or(0);
    let grouped_long_hca = kernel == DeepSeekV4DenseAttentionKernel::GroupedOnline
        && kind == AttentionKind::HeavilyCompressed
        && raw_cache_layout == DeepSeekV4RawCacheLayout::Ring
        && token_count == 1;
    let compressed_cache = match compressed {
        None if expected_rows == 0 => raw_cache,
        Some(rows) if rows.count == expected_rows => {
            validate_f16(
                rows.cache,
                &[config.head_dim as u64, rows.capacity_rows as u64],
                false,
                "cooperative dense compressed cache",
            )?;
            if expected_rows > rows.capacity_rows
                || (!grouped_long_hca && expected_rows > DEEPSEEK_V4_CSA_TOP_K)
            {
                return invalid(format!(
                    "cooperative dense attention cannot consume {expected_rows} rows from capacity {}",
                    rows.capacity_rows
                ));
            }
            rows.cache
        }
        rows => {
            return invalid(format!(
                "cooperative dense attention expected {expected_rows} compressed rows, got {}",
                rows.map_or(0, |rows| rows.count)
            ));
        }
    };
    let final_raw_rows = (end_position as usize).min(DEEPSEEK_V4_LOCAL_WINDOW);
    let maximum_rows = final_raw_rows
        .checked_add(expected_rows)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("cooperative dense row overflow".into()))?;
    if maximum_rows == 0
        || (!grouped_long_hca && maximum_rows > DEEPSEEK_V4_LOCAL_WINDOW + DEEPSEEK_V4_CSA_TOP_K)
    {
        return invalid(format!(
            "cooperative dense attention row count {maximum_rows} is out of range"
        ));
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        token_count: u32,
        compression_ratio: u32,
        start_position: u32,
        window: u32,
        raw_cache_is_chunk: u32,
        scale: f32,
    }
    let threadgroup_width = config.head_dim.max(maximum_rows);
    let (kernel_name, threadgroup_bytes, grid, threads) = match kernel {
        DeepSeekV4DenseAttentionKernel::Cooperative => (
            "kernel_deepseek_v4_packed_dense_sink_attention_f16",
            (maximum_rows + 1) * std::mem::size_of::<f32>(),
            MTLSize {
                width: token_count,
                height: config.head_count,
                depth: 1,
            },
            MTLSize {
                width: threadgroup_width,
                height: 1,
                depth: 1,
            },
        ),
        DeepSeekV4DenseAttentionKernel::GroupedOnline => {
            if config.head_count != 64 || config.head_dim != DEEPSEEK_V4_HCA_TILE_ROWS {
                return invalid("grouped online dense attention requires 64 heads of width 512");
            }
            (
                "kernel_deepseek_v4_grouped_online_dense_sink_attention_f16",
                DEEPSEEK_V4_GROUPED_DENSE_THREADGROUP_BYTES,
                MTLSize {
                    width: token_count,
                    height: config.head_count / DEEPSEEK_V4_GROUPED_DENSE_HEADS,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: DEEPSEEK_V4_GROUPED_DENSE_HEADS,
                    depth: 1,
                },
            )
        }
    };
    let pso = ctx.pipeline(kernel_name)?;
    match kernel {
        DeepSeekV4DenseAttentionKernel::Cooperative => {
            if pso.maxTotalThreadsPerThreadgroup() < threadgroup_width {
                return invalid(format!(
                    "cooperative dense attention pipeline supports {} threads, requires {threadgroup_width}",
                    pso.maxTotalThreadsPerThreadgroup()
                ));
            }
        }
        DeepSeekV4DenseAttentionKernel::GroupedOnline => {
            if pso.threadExecutionWidth() != 32
                || pso.maxTotalThreadsPerThreadgroup() < DEEPSEEK_V4_GROUPED_DENSE_THREADS
                || ctx.device.maxThreadgroupMemoryLength()
                    < DEEPSEEK_V4_GROUPED_DENSE_THREADGROUP_BYTES
            {
                return invalid(format!(
                    "grouped online dense attention requires SIMD width 32, {} threads, and {} threadgroup bytes; pipeline width={} max_threads={} device_bytes={}",
                    DEEPSEEK_V4_GROUPED_DENSE_THREADS,
                    DEEPSEEK_V4_GROUPED_DENSE_THREADGROUP_BYTES,
                    pso.threadExecutionWidth(),
                    pso.maxTotalThreadsPerThreadgroup(),
                    ctx.device.maxThreadgroupMemoryLength(),
                ));
            }
        }
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: config.head_count as u32,
            head_dim: config.head_dim as u32,
            token_count: token_count_u32,
            compression_ratio: ratio as u32,
            start_position,
            window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
            raw_cache_is_chunk: raw_cache_layout.is_chunk(),
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, raw_cache_before_chunk);
    enc.set_tensor(4, compressed_cache);
    enc.set_tensor(5, sinks);
    enc.set_tensor(6, output);
    enc.set_threadgroup_memory(0, threadgroup_bytes);
    enc.dispatch(grid, threads);
    Ok(())
}

const DEEPSEEK_V4_ONLINE_HCA_THREADS: usize = 32;
const DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES: usize =
    DEEPSEEK_V4_HCA_TILE_ROWS * std::mem::size_of::<half::f16>();

crate::env_flag!(
    default_on deepseek_v4_online_direct_load_enabled,
    "QWEN_DSV4_ONLINE_DIRECT_LOAD"
);

crate::env_flag!(
    default_on deepseek_v4_decode_output_grouped_enabled,
    "QWEN_DSV4_DECODE_OUTPUT_GROUPED"
);

crate::env_flag!(
    default_on deepseek_v4_decode_shared_swiglu_enabled,
    "QWEN_DSV4_DECODE_SHARED_SWIGLU"
);

crate::env_flag!(
    default_on deepseek_v4_decode_compressor_fused_enabled,
    "QWEN_DSV4_DECODE_COMPRESSOR_FUSED"
);

crate::env_flag!(
    default_on deepseek_v4_decode_prepare_paired_enabled,
    "QWEN_DSV4_DECODE_PREPARE_PAIRED"
);

crate::env_flag!(
    default_on deepseek_v4_grouped_long_hca_enabled,
    "QWEN_DSV4_GROUPED_LONG_HCA"
);

crate::env_flag!(
    default_on deepseek_v4_splitk_hca_enabled,
    "QWEN_DSV4_SPLITK_HCA"
);

fn validate_deepseek_v4_online_hca_request_geometry(
    config: DeepSeekV4PositionZeroAttentionConfig,
    compression_ratio: usize,
    token_count: usize,
    query_token_offset: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if config.head_count != 64
        || config.head_dim != DEEPSEEK_V4_HCA_TILE_ROWS
        || compression_ratio != 128
        || token_count != 1
        || query_token_offset != 0
        || query_count != 1
    {
        return invalid(
            "online tiled HCA requires one complete 64-head x 512-dimension ratio-128 singleton query",
        );
    }
    Ok(())
}

fn validate_deepseek_v4_online_hca_launch_geometry(
    thread_execution_width: usize,
    max_threads_per_group: usize,
    max_threadgroup_bytes: usize,
    required_threadgroup_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if thread_execution_width != DEEPSEEK_V4_ONLINE_HCA_THREADS
        || max_threads_per_group < DEEPSEEK_V4_ONLINE_HCA_THREADS
    {
        return invalid(format!(
            "online tiled HCA requires SIMD width {} and {} threads, got width {thread_execution_width} max {max_threads_per_group}",
            DEEPSEEK_V4_ONLINE_HCA_THREADS, DEEPSEEK_V4_ONLINE_HCA_THREADS,
        ));
    }
    if max_threadgroup_bytes < required_threadgroup_bytes {
        return invalid(format!(
            "online tiled HCA requires {} threadgroup bytes, device allows {max_threadgroup_bytes}",
            required_threadgroup_bytes,
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_tiled_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: DeepSeekV4PublishedRows<'_>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    chunk_start_position: u32,
    query_token_offset: usize,
    query_count: usize,
    compression_ratio: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    encode_tiled_dense_sink_attention_f16_with_mode(
        ctx,
        enc,
        queries,
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        compressed,
        sinks,
        output,
        chunk_start_position,
        query_token_offset,
        query_count,
        compression_ratio,
        config,
        false,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_online_dense_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: DeepSeekV4PublishedRows<'_>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    chunk_start_position: u32,
    query_token_offset: usize,
    query_count: usize,
    compression_ratio: usize,
    direct_load: bool,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    encode_tiled_dense_sink_attention_f16_with_mode(
        ctx,
        enc,
        queries,
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        compressed,
        sinks,
        output,
        chunk_start_position,
        query_token_offset,
        query_count,
        compression_ratio,
        config,
        true,
        direct_load,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_tiled_dense_sink_attention_f16_with_mode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed: DeepSeekV4PublishedRows<'_>,
    sinks: &MetalTensor,
    output: &MetalTensor,
    chunk_start_position: u32,
    query_token_offset: usize,
    query_count: usize,
    compression_ratio: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
    online: bool,
    direct_load: bool,
) -> Result<(), DeepSeekV4MetalError> {
    let query_width = checked_mul(config.head_count, config.head_dim, "tiled query width")?;
    if config.head_dim != DEEPSEEK_V4_HCA_TILE_ROWS || compression_ratio != 128 || query_count == 0
    {
        return invalid("tiled dense attention requires 512-wide ratio-128 HCA queries");
    }
    if queries.shape.len() != 2 || output.shape.len() != 2 {
        return invalid("tiled dense attention requires token-major rank-2 queries and output");
    }
    let token_count = usize::try_from(queries.shape[1])
        .map_err(|_| DeepSeekV4MetalError::Invalid("tiled query count exceeds usize".into()))?;
    if direct_load && !online {
        return invalid("direct row loading requires online dense attention");
    }
    if online {
        validate_deepseek_v4_online_hca_request_geometry(
            config,
            compression_ratio,
            token_count,
            query_token_offset,
            query_count,
        )?;
        require_serial(enc, "deepseek_v4_online_dense_sink_attention_f16")?;
    }
    let query_end = query_token_offset
        .checked_add(query_count)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("tiled query range overflow".into()))?;
    if query_end > token_count || output.shape != queries.shape {
        return invalid(format!(
            "tiled dense attention query range {query_token_offset}..{query_end} exceeds {token_count} tokens or output shape differs"
        ));
    }
    validate_f32(
        queries,
        &[query_width as u64, token_count as u64],
        false,
        "tiled attention queries",
    )?;
    validate_f32(
        output,
        &[query_width as u64, token_count as u64],
        true,
        "tiled attention output",
    )?;
    validate_raw_attention_caches(
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        config.head_dim,
        token_count,
        "tiled attention",
    )?;
    validate_f16(
        compressed.cache,
        &[config.head_dim as u64, compressed.capacity_rows as u64],
        false,
        "tiled compressed attention cache",
    )?;
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "tiled attention sinks",
    )?;

    let first_position =
        chunk_start_position
            .checked_add(u32::try_from(query_token_offset).map_err(|_| {
                DeepSeekV4MetalError::Invalid("tiled query offset exceeds u32".into())
            })?)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("tiled first position overflow".into()))?;
    let end_position = chunk_start_position
        .checked_add(u32::try_from(query_end).map_err(|_| {
            DeepSeekV4MetalError::Invalid("tiled query endpoint exceeds u32".into())
        })?)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("tiled end position overflow".into()))?;
    let first_visible = (u64::from(first_position) + 1) as usize / compression_ratio;
    let final_visible = end_position as usize / compression_ratio;
    if first_visible == 0
        || compressed.count != final_visible
        || compressed.count > compressed.capacity_rows
    {
        return invalid(format!(
            "tiled HCA requires 1..={} visible rows, first={first_visible} final={final_visible} stored={}/{}",
            compressed.capacity_rows, compressed.count, compressed.capacity_rows
        ));
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        query_count: u32,
        query_token_offset: u32,
        chunk_start_position: u32,
        window: u32,
        compression_ratio: u32,
        raw_cache_is_chunk: u32,
        scale: f32,
    }
    let args = Args {
        head_count: u32::try_from(config.head_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid("tiled HCA head count exceeds u32".into())
        })?,
        head_dim: DEEPSEEK_V4_HCA_TILE_ROWS as u32,
        query_count: u32::try_from(query_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid("tiled HCA query count exceeds u32".into())
        })?,
        query_token_offset: u32::try_from(query_token_offset).map_err(|_| {
            DeepSeekV4MetalError::Invalid("tiled HCA query offset exceeds u32".into())
        })?,
        chunk_start_position,
        window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
        compression_ratio: compression_ratio as u32,
        raw_cache_is_chunk: raw_cache_layout.is_chunk(),
        scale: 1.0 / (config.head_dim as f32).sqrt(),
    };
    let (kernel, threadgroup_width, threadgroup_bytes) = if online {
        (
            if direct_load {
                "kernel_deepseek_v4_online_dense_sink_attention_f16_direct"
            } else {
                "kernel_deepseek_v4_online_dense_sink_attention_f16"
            },
            DEEPSEEK_V4_ONLINE_HCA_THREADS,
            if direct_load {
                0
            } else {
                DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES
            },
        )
    } else {
        (
            "kernel_deepseek_v4_tiled_dense_sink_attention_f16",
            DEEPSEEK_V4_HCA_TILE_ROWS,
            (2 * DEEPSEEK_V4_HCA_TILE_ROWS + 1) * std::mem::size_of::<f32>(),
        )
    };
    let pso = ctx.pipeline(kernel)?;
    if online {
        validate_deepseek_v4_online_hca_launch_geometry(
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup(),
            ctx.device.maxThreadgroupMemoryLength(),
            threadgroup_bytes,
        )?;
    } else if pso.maxTotalThreadsPerThreadgroup() < threadgroup_width {
        return invalid(format!(
            "tiled HCA pipeline supports {} threads, requires {threadgroup_width}",
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, raw_cache_before_chunk);
    enc.set_tensor(4, compressed.cache);
    enc.set_tensor(5, sinks);
    enc.set_tensor(6, output);
    if threadgroup_bytes != 0 {
        enc.set_threadgroup_memory(0, threadgroup_bytes);
    }
    enc.dispatch(
        MTLSize {
            width: query_count,
            height: config.head_count,
            depth: 1,
        },
        MTLSize {
            width: threadgroup_width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_compressor_frontier_write(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    projected_kv: &MetalTensor,
    projected_score: &MetalTensor,
    ape: &MetalTensor,
    kv_state: &MetalTensor,
    score_state: &MetalTensor,
    width: usize,
    row: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let row_offset = checked_mul(row, width, "compressor frontier row offset")?;
    for (tensor, name) in [
        (projected_kv, "projected compressor KV"),
        (projected_score, "projected compressor score"),
        (ape, "compressor APE row"),
    ] {
        validate_f32(tensor, &[width as u64], false, name)?;
    }
    validate_f32(kv_state, &kv_state.shape, true, "compressor KV state")?;
    validate_f32(
        score_state,
        &score_state.shape,
        true,
        "compressor score state",
    )?;
    if kv_state.shape != score_state.shape
        || row_offset
            .checked_add(width)
            .is_none_or(|end| end as u64 > kv_state.n_elements())
    {
        return invalid("compressor frontier row exceeds aligned KV/score state");
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        width: u32,
        row_offset: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_compressor_frontier_write")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            width: width as u32,
            row_offset: row_offset as u32,
        },
    );
    enc.set_tensor(1, projected_kv);
    enc.set_tensor(2, projected_score);
    enc.set_tensor(3, ape);
    enc.set_tensor(4, kv_state);
    enc.set_tensor(5, score_state);
    enc.dispatch(
        MTLSize {
            width: width.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_compressor_frontier_chunk(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    projected_kv: &MetalTensor,
    projected_score: &MetalTensor,
    ape: &MetalTensor,
    kv_state: &MetalTensor,
    score_state: &MetalTensor,
    pooled_rows: &MetalTensor,
    ratio: usize,
    head_dim: usize,
    width: usize,
    row_count: usize,
    start_position: u32,
    output_rows: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if !matches!(ratio, 4 | 128) || row_count == 0 {
        return invalid("compressor chunk requires ratio 4 or 128 and at least one row");
    }
    let coefficient = if ratio == 4 { 2 } else { 1 };
    if width != coefficient * head_dim {
        return invalid("compressor chunk width differs from its ratio geometry");
    }
    let state_rows = checked_mul(coefficient, ratio, "compressor chunk state rows")?;
    validate_f32(
        projected_kv,
        &[width as u64, row_count as u64],
        false,
        "projected compressor KV chunk",
    )?;
    validate_f32(
        projected_score,
        &[width as u64, row_count as u64],
        false,
        "projected compressor score chunk",
    )?;
    validate_f32(
        ape,
        &[width as u64, ratio as u64],
        false,
        "compressor chunk APE",
    )?;
    for (state, name) in [
        (kv_state, "compressor chunk KV state"),
        (score_state, "compressor chunk score state"),
    ] {
        validate_f32(state, &[width as u64, state_rows as u64], true, name)?;
    }
    if pooled_rows.dtype != GgmlType::F32
        || !pooled_rows.is_writable()
        || pooled_rows.shape.len() != 2
        || pooled_rows.shape[0] != head_dim as u64
        || pooled_rows.shape[1] < output_rows as u64
    {
        return invalid(format!(
            "compressor pooled scratch must hold [{head_dim}, >={output_rows}], got {:?} {:?}",
            pooled_rows.dtype, pooled_rows.shape
        ));
    }
    let end_position = start_position
        .checked_add(u32::try_from(row_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid("compressor chunk row count exceeds u32".into())
        })?)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("compressor chunk position overflow".into())
        })?;
    let expected_rows = end_position as usize / ratio - start_position as usize / ratio;
    if output_rows != expected_rows {
        return invalid(format!(
            "compressor chunk expected {expected_rows} pooled rows, got {output_rows}"
        ));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ratio: u32,
        head_dim: u32,
        width: u32,
        row_count: u32,
        start_position: u32,
        output_rows: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_compressor_frontier_chunk")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            ratio: u32::try_from(ratio).map_err(|_| {
                DeepSeekV4MetalError::Invalid("compressor chunk ratio exceeds u32".into())
            })?,
            head_dim: u32::try_from(head_dim).map_err(|_| {
                DeepSeekV4MetalError::Invalid("compressor chunk head dimension exceeds u32".into())
            })?,
            width: u32::try_from(width).map_err(|_| {
                DeepSeekV4MetalError::Invalid("compressor chunk width exceeds u32".into())
            })?,
            row_count: u32::try_from(row_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("compressor chunk rows exceed u32".into())
            })?,
            start_position,
            output_rows: u32::try_from(output_rows).map_err(|_| {
                DeepSeekV4MetalError::Invalid("compressor output rows exceed u32".into())
            })?,
        },
    );
    enc.set_tensor(1, projected_kv);
    enc.set_tensor(2, projected_score);
    enc.set_tensor(3, ape);
    enc.set_tensor(4, kv_state);
    enc.set_tensor(5, score_state);
    enc.set_tensor(6, pooled_rows);
    enc.dispatch(
        MTLSize {
            width: head_dim.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_compressor_pool(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kv_state: &MetalTensor,
    score_state: &MetalTensor,
    output: &MetalTensor,
    ratio: usize,
    head_dim: usize,
    width: usize,
    rows: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if !matches!(ratio, 4 | 128) {
        return invalid(format!("unsupported compressor pooling ratio {ratio}"));
    }
    let coefficient = if ratio == 4 { 2 } else { 1 };
    if width != coefficient * head_dim || rows != coefficient * ratio {
        return invalid("compressor pooling geometry is inconsistent");
    }
    validate_f32(
        kv_state,
        &[width as u64, rows as u64],
        false,
        "compressor pooling KV state",
    )?;
    validate_f32(
        score_state,
        &[width as u64, rows as u64],
        false,
        "compressor pooling score state",
    )?;
    validate_f32(output, &[head_dim as u64], true, "compressor pooled output")?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ratio: u32,
        head_dim: u32,
        width: u32,
        rows: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_compressor_pool")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            ratio: ratio as u32,
            head_dim: head_dim as u32,
            width: width as u32,
            rows: rows as u32,
        },
    );
    enc.set_tensor(1, kv_state);
    enc.set_tensor(2, score_state);
    enc.set_tensor(3, output);
    enc.dispatch(
        MTLSize {
            width: head_dim.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_compressor_roll_ratio4(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    kv_state: &MetalTensor,
    score_state: &MetalTensor,
    width: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let shape = [width as u64, 8];
    validate_f32(kv_state, &shape, true, "ratio-4 KV frontier")?;
    validate_f32(score_state, &shape, true, "ratio-4 score frontier")?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        width: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_compressor_roll_ratio4")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            width: width as u32,
        },
    );
    enc.set_tensor(1, kv_state);
    enc.set_tensor(2, score_state);
    let elements = checked_mul(4, width, "ratio-4 roll elements")?;
    enc.dispatch(
        MTLSize {
            width: elements.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_hadamard_128_in_place(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    values: &MetalTensor,
) -> Result<(), DeepSeekV4MetalError> {
    validate_f32(values, &[128], true, "Hadamard-128 row")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_hadamard_128_in_place")?;
    enc.set_pipeline(&pso);
    enc.set_tensor(0, values);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_hadamard_128_rows_in_place(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    values: &MetalTensor,
    row_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if row_count == 0 {
        return invalid("Hadamard-128 row count must be nonzero");
    }
    let expected_elements = checked_mul(row_count, 128, "Hadamard-128 row elements")?;
    let row_count_u32 = u32::try_from(row_count)
        .map_err(|_| DeepSeekV4MetalError::Invalid("Hadamard-128 row count exceeds u32".into()))?;
    validate_f32(values, &values.shape, true, "Hadamard-128 rows")?;
    if values.shape.first().copied() != Some(128) || values.n_elements() != expected_elements as u64
    {
        return invalid(format!(
            "Hadamard-128 rows require leading width 128 and {row_count} rows, got {:?}",
            values.shape
        ));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_count: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_hadamard_128_rows_in_place")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_count: row_count_u32,
        },
    );
    enc.set_tensor(1, values);
    enc.dispatch(
        MTLSize {
            width: row_count.div_ceil(64),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_scale_f32_in_place(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    values: &MetalTensor,
    scale: f32,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    if !scale.is_finite() {
        return invalid(format!("{name} scale must be finite"));
    }
    validate_f32(values, &values.shape, true, name)?;
    let count = usize::try_from(values.n_elements())
        .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} length exceeds usize")))?;
    if count == 0 {
        return invalid(format!("{name} must be nonempty"));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        count: u32,
        scale: f32,
    }
    let count = u32::try_from(count)
        .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} length exceeds u32")))?;
    let pso = ctx.pipeline("kernel_deepseek_v4_scale_f32_in_place")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &Args { count, scale });
    enc.set_tensor(1, values);
    enc.dispatch(
        MTLSize {
            width: (count as usize).div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_lightning_indexer_scores_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    head_weights: &MetalTensor,
    keys: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    encode_lightning_indexer_scores_f16_with_policy(
        ctx,
        enc,
        queries,
        head_weights,
        keys,
        visible_counts,
        scores,
        head_count,
        head_dim,
        row_capacity,
        query_count,
        false,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeepSeekV4LightningScoreKernel {
    Scalar,
    Cooperative,
    TiledF32,
}

#[allow(clippy::too_many_arguments)]
fn encode_lightning_indexer_scores_f16_with_limit(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    head_weights: &MetalTensor,
    keys: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    max_dispatched_rows: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    encode_lightning_indexer_scores_f16_with_limit_and_kernel(
        ctx,
        enc,
        queries,
        head_weights,
        keys,
        visible_counts,
        scores,
        head_count,
        head_dim,
        row_capacity,
        max_dispatched_rows,
        query_count,
        if head_count == 64 && head_dim == 128 {
            DeepSeekV4LightningScoreKernel::Cooperative
        } else {
            DeepSeekV4LightningScoreKernel::Scalar
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_lightning_indexer_scores_f16_tiled_f32_with_limit(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    head_weights: &MetalTensor,
    keys: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    max_dispatched_rows: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    encode_lightning_indexer_scores_f16_with_limit_and_kernel(
        ctx,
        enc,
        queries,
        head_weights,
        keys,
        visible_counts,
        scores,
        head_count,
        head_dim,
        row_capacity,
        max_dispatched_rows,
        query_count,
        DeepSeekV4LightningScoreKernel::TiledF32,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_lightning_indexer_scores_f16_with_policy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    head_weights: &MetalTensor,
    keys: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    query_count: usize,
    force_scalar: bool,
) -> Result<(), DeepSeekV4MetalError> {
    encode_lightning_indexer_scores_f16_with_limit_and_kernel(
        ctx,
        enc,
        queries,
        head_weights,
        keys,
        visible_counts,
        scores,
        head_count,
        head_dim,
        row_capacity,
        row_capacity,
        query_count,
        if !force_scalar && head_count == 64 && head_dim == 128 {
            DeepSeekV4LightningScoreKernel::Cooperative
        } else {
            DeepSeekV4LightningScoreKernel::Scalar
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_lightning_indexer_scores_f16_with_limit_and_kernel(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    head_weights: &MetalTensor,
    keys: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    max_dispatched_rows: usize,
    query_count: usize,
    kernel: DeepSeekV4LightningScoreKernel,
) -> Result<(), DeepSeekV4MetalError> {
    for (name, value) in [
        ("indexer head count", head_count),
        ("indexer head dimension", head_dim),
        ("indexer row capacity", row_capacity),
        ("indexer maximum dispatched rows", max_dispatched_rows),
        ("indexer query count", query_count),
    ] {
        if value == 0 || u32::try_from(value).is_err() {
            return invalid(format!("{name} must be nonzero and fit u32"));
        }
    }
    if max_dispatched_rows > row_capacity {
        return invalid(format!(
            "indexer maximum dispatched rows {max_dispatched_rows} exceed row capacity {row_capacity}"
        ));
    }
    if kernel == DeepSeekV4LightningScoreKernel::TiledF32 && (head_count != 64 || head_dim != 128) {
        return invalid(format!(
            "tiled F32 indexer scoring requires 64 heads of width 128, got {head_count}x{head_dim}"
        ));
    }
    validate_lightning_indexer_score_offsets(head_count, head_dim, row_capacity, query_count)?;
    validate_f32(
        queries,
        &[head_dim as u64, head_count as u64, query_count as u64],
        false,
        "indexer queries",
    )?;
    validate_f32(
        head_weights,
        &[head_count as u64, query_count as u64],
        false,
        "indexer head weights",
    )?;
    validate_f16(
        keys,
        &[head_dim as u64, row_capacity as u64],
        false,
        "indexer keys",
    )?;
    validate_i32(
        visible_counts,
        &[query_count as u64],
        false,
        "indexer visible counts",
    )?;
    validate_f32(
        scores,
        &[row_capacity as u64, query_count as u64],
        true,
        "indexer scores",
    )?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        row_capacity: u32,
        query_count: u32,
    }
    let kernel_name = match kernel {
        DeepSeekV4LightningScoreKernel::Scalar => "kernel_deepseek_v4_lightning_indexer_scores_f16",
        DeepSeekV4LightningScoreKernel::Cooperative => {
            "kernel_deepseek_v4_lightning_indexer_scores_f16_cooperative"
        }
        DeepSeekV4LightningScoreKernel::TiledF32 => {
            "kernel_deepseek_v4_lightning_indexer_scores_f16_tiled_f32"
        }
    };
    let pso = ctx.pipeline(kernel_name)?;
    match kernel {
        DeepSeekV4LightningScoreKernel::Scalar => {}
        DeepSeekV4LightningScoreKernel::Cooperative => {
            validate_cooperative_lightning_score_geometry(
                kernel_name,
                pso.threadExecutionWidth(),
                pso.maxTotalThreadsPerThreadgroup(),
                ctx.device.maxThreadgroupMemoryLength(),
            )?;
        }
        DeepSeekV4LightningScoreKernel::TiledF32 => {
            validate_tiled_f32_lightning_score_geometry(
                kernel_name,
                pso.threadExecutionWidth(),
                pso.maxTotalThreadsPerThreadgroup(),
                ctx.device.maxThreadgroupMemoryLength(),
            )?;
        }
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: head_count as u32,
            head_dim: head_dim as u32,
            row_capacity: row_capacity as u32,
            query_count: query_count as u32,
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, head_weights);
    enc.set_tensor(3, keys);
    enc.set_tensor(4, visible_counts);
    enc.set_tensor(5, scores);
    match kernel {
        DeepSeekV4LightningScoreKernel::Scalar => {
            enc.dispatch(
                MTLSize {
                    width: max_dispatched_rows.div_ceil(256),
                    height: query_count,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        DeepSeekV4LightningScoreKernel::Cooperative => {
            enc.set_threadgroup_memory(0, 8 * 128 * std::mem::size_of::<u16>());
            enc.set_threadgroup_memory(1, 8 * 64 * std::mem::size_of::<f32>());
            enc.dispatch(
                MTLSize {
                    width: max_dispatched_rows.div_ceil(8),
                    height: query_count,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        DeepSeekV4LightningScoreKernel::TiledF32 => {
            enc.set_threadgroup_memory(0, 5_376 * std::mem::size_of::<f32>());
            enc.dispatch(
                MTLSize {
                    width: max_dispatched_rows.div_ceil(32),
                    height: query_count.div_ceil(8),
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_lightning_indexer_scores_f16_matrix(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    head_weights: &MetalTensor,
    keys: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    max_dispatched_rows: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const KERNEL: &str = "kernel_deepseek_v4_lightning_indexer_scores_f16_matrix_ceiling";
    if head_count != 64 || head_dim != 128 {
        return invalid(format!(
            "{KERNEL} requires 64 heads of width 128, got {head_count}x{head_dim}"
        ));
    }
    for (name, value) in [
        ("indexer row capacity", row_capacity),
        ("indexer maximum dispatched rows", max_dispatched_rows),
        ("indexer query count", query_count),
    ] {
        if value == 0 || u32::try_from(value).is_err() {
            return invalid(format!("{name} must be nonzero and fit u32"));
        }
    }
    if max_dispatched_rows > row_capacity {
        return invalid(format!(
            "matrix-ceiling indexer maximum dispatched rows {max_dispatched_rows} exceed row capacity {row_capacity}"
        ));
    }
    validate_lightning_indexer_score_offsets(head_count, head_dim, row_capacity, query_count)?;
    validate_f16(
        queries,
        &[head_dim as u64, head_count as u64, query_count as u64],
        false,
        "matrix-ceiling indexer queries",
    )?;
    validate_f32(
        head_weights,
        &[head_count as u64, query_count as u64],
        false,
        "matrix-ceiling indexer head weights",
    )?;
    validate_f16(
        keys,
        &[head_dim as u64, row_capacity as u64],
        false,
        "matrix-ceiling indexer keys",
    )?;
    validate_i32(
        visible_counts,
        &[query_count as u64],
        false,
        "matrix-ceiling indexer visible counts",
    )?;
    validate_f32(
        scores,
        &[row_capacity as u64, query_count as u64],
        true,
        "matrix-ceiling indexer scores",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        row_capacity: u32,
        query_count: u32,
    }

    let pso = ctx.pipeline(KERNEL)?;
    validate_cooperative_lightning_score_geometry(
        KERNEL,
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        ctx.device.maxThreadgroupMemoryLength(),
    )?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: head_count as u32,
            head_dim: head_dim as u32,
            row_capacity: row_capacity as u32,
            query_count: query_count as u32,
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, head_weights);
    enc.set_tensor(3, keys);
    enc.set_tensor(4, visible_counts);
    enc.set_tensor(5, scores);
    enc.set_threadgroup_memory(0, 8 * 128 * std::mem::size_of::<u16>());
    enc.set_threadgroup_memory(1, 8 * 64 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: max_dispatched_rows.div_ceil(8),
            height: query_count,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
fn encode_indexer_fp4_contract_primitives(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    e2m1_values: &MetalTensor,
    scale_maxima: &MetalTensor,
    e2m1_codes: &MetalTensor,
    scale_codes: &MetalTensor,
    scale_status: &MetalTensor,
    e2m1_count: usize,
    scale_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if e2m1_count == 0
        || scale_count == 0
        || u32::try_from(e2m1_count).is_err()
        || u32::try_from(scale_count).is_err()
    {
        return invalid("indexer FP4 primitive counts must be nonzero and fit u32");
    }
    validate_f32(
        e2m1_values,
        &[e2m1_count as u64],
        false,
        "indexer FP4 E2M1 primitive inputs",
    )?;
    validate_f32(
        scale_maxima,
        &[scale_count as u64],
        false,
        "indexer FP4 scale primitive inputs",
    )?;
    validate_i8(
        e2m1_codes,
        &[e2m1_count as u64],
        true,
        "indexer FP4 E2M1 primitive codes",
    )?;
    validate_i8(
        scale_codes,
        &[scale_count as u64],
        true,
        "indexer FP4 scale primitive codes",
    )?;
    validate_i32(
        scale_status,
        &[scale_count as u64],
        true,
        "indexer FP4 scale primitive status",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        e2m1_count: u32,
        scale_count: u32,
    }

    let pso = ctx.pipeline("kernel_deepseek_v4_indexer_fp4_contract_primitives")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            e2m1_count: e2m1_count as u32,
            scale_count: scale_count as u32,
        },
    );
    enc.set_tensor(1, e2m1_values);
    enc.set_tensor(2, scale_maxima);
    enc.set_tensor(3, e2m1_codes);
    enc.set_tensor(4, scale_codes);
    enc.set_tensor(5, scale_status);
    enc.dispatch(
        MTLSize {
            width: e2m1_count.max(scale_count).div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn encode_pack_indexer_fp4_rows_shadow(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    packed_values: &MetalTensor,
    packed_scales: &MetalTensor,
    status: &MetalTensor,
    row_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const VALUES_PER_ROW: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUES_PER_ROW;
    const VALUE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES;
    const SCALE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES;
    if row_count == 0 || u32::try_from(row_count).is_err() {
        return invalid("indexer FP4 row count must be nonzero and fit u32");
    }
    for (name, elements) in [
        (
            "indexer FP4 input elements",
            checked_mul(row_count, VALUES_PER_ROW, "indexer FP4 input elements")?,
        ),
        (
            "indexer FP4 value bytes",
            checked_mul(row_count, VALUE_BYTES, "indexer FP4 value bytes")?,
        ),
        (
            "indexer FP4 scale bytes",
            checked_mul(row_count, SCALE_BYTES, "indexer FP4 scale bytes")?,
        ),
    ] {
        if u32::try_from(elements).is_err() {
            return invalid(format!("{name} exceed u32 shader offsets"));
        }
    }
    let trailing_rows = input
        .shape
        .get(1..)
        .and_then(crate::tensor::checked_shape_elements)
        .and_then(|rows| usize::try_from(rows).ok());
    if input.dtype != GgmlType::F32
        || input.shape.first().copied() != Some(VALUES_PER_ROW as u64)
        || trailing_rows != Some(row_count)
    {
        return invalid(format!(
            "indexer FP4 pack input must be F32 with width {VALUES_PER_ROW} and {row_count} rows, got {:?} {:?}",
            input.dtype, input.shape
        ));
    }
    let mut value_shape = input.shape.clone();
    value_shape[0] = VALUE_BYTES as u64;
    let mut scale_shape = input.shape.clone();
    scale_shape[0] = SCALE_BYTES as u64;
    validate_f32(input, &input.shape, false, "indexer FP4 pack input")?;
    validate_i8(
        packed_values,
        &value_shape,
        true,
        "indexer FP4 packed values",
    )?;
    validate_i8(
        packed_scales,
        &scale_shape,
        true,
        "indexer FP4 packed scales",
    )?;
    validate_i32(status, &[row_count as u64], true, "indexer FP4 pack status")?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_count: u32,
    }

    let pso = ctx.pipeline("kernel_deepseek_v4_pack_indexer_fp4_rows_shadow")?;
    validate_fp4_pack_geometry(
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        ctx.device.maxThreadgroupMemoryLength(),
    )?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_count: row_count as u32,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, packed_values);
    enc.set_tensor(3, packed_scales);
    enc.set_tensor(4, status);
    enc.dispatch(
        MTLSize {
            width: row_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn encode_unpack_indexer_fp4_units_shadow(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    packed_values: &MetalTensor,
    status: &MetalTensor,
    units: &MetalTensor,
    row_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const VALUE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES;
    const VALUES_PER_ROW: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUES_PER_ROW;
    if row_count == 0 || u32::try_from(row_count).is_err() {
        return invalid("indexer FP4 unpack row count must be nonzero and fit u32");
    }
    let trailing_rows = packed_values
        .shape
        .get(1..)
        .and_then(crate::tensor::checked_shape_elements)
        .and_then(|rows| usize::try_from(rows).ok());
    if packed_values.dtype != GgmlType::I8
        || packed_values.shape.first().copied() != Some(VALUE_BYTES as u64)
        || trailing_rows != Some(row_count)
    {
        return invalid(format!(
            "indexer FP4 unpack values must be raw I8 width {VALUE_BYTES} with {row_count} rows, got {:?} {:?}",
            packed_values.dtype, packed_values.shape
        ));
    }
    let mut unit_shape = packed_values.shape.clone();
    unit_shape[0] = VALUES_PER_ROW as u64;
    validate_i8(
        packed_values,
        &packed_values.shape,
        false,
        "indexer FP4 unpack values",
    )?;
    validate_i32(
        status,
        &[row_count as u64],
        false,
        "indexer FP4 unpack status",
    )?;
    validate_f16(units, &unit_shape, true, "indexer FP4 unpacked units")?;
    let element_count = checked_mul(row_count, VALUES_PER_ROW, "indexer FP4 unpack elements")?;
    if u32::try_from(element_count).is_err() {
        return invalid("indexer FP4 unpack elements exceed u32 shader offsets");
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_count: u32,
    }

    let pso = ctx.pipeline("kernel_deepseek_v4_unpack_indexer_fp4_units_shadow")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_count: row_count as u32,
        },
    );
    enc.set_tensor(1, packed_values);
    enc.set_tensor(2, status);
    enc.set_tensor(3, units);
    enc.dispatch(
        MTLSize {
            width: element_count.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(feature = "dsv4-diagnostics")]
fn encode_indexer_fp4_shadow_preflight(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query_status: &MetalTensor,
    key_status: &MetalTensor,
    requested_visible: &MetalTensor,
    eligible_visible: &MetalTensor,
    eligibility_record: &MetalTensor,
    row_capacity: usize,
    expected_visible: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const QUERY_ROWS: usize = 64;
    if row_capacity == 0
        || expected_visible == 0
        || expected_visible > row_capacity
        || u32::try_from(row_capacity).is_err()
        || u32::try_from(expected_visible).is_err()
    {
        return invalid(
            "indexer FP4 preflight capacity/visibility must be nonzero, ordered, and fit u32",
        );
    }
    validate_i32(
        query_status,
        &[QUERY_ROWS as u64],
        false,
        "indexer FP4 query status",
    )?;
    validate_i32(
        key_status,
        &[row_capacity as u64],
        false,
        "indexer FP4 key status",
    )?;
    validate_i32(
        requested_visible,
        &[1],
        false,
        "indexer FP4 requested visibility",
    )?;
    validate_i32(
        eligible_visible,
        &[1],
        true,
        "indexer FP4 eligible visibility",
    )?;
    validate_i32(
        eligibility_record,
        &[3],
        true,
        "indexer FP4 eligibility record",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        query_rows: u32,
        row_capacity: u32,
        expected_visible: u32,
    }

    let pso = ctx.pipeline("kernel_deepseek_v4_indexer_fp4_shadow_preflight")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            query_rows: QUERY_ROWS as u32,
            row_capacity: row_capacity as u32,
            expected_visible: expected_visible as u32,
        },
    );
    enc.set_tensor(1, query_status);
    enc.set_tensor(2, key_status);
    enc.set_tensor(3, requested_visible);
    enc.set_tensor(4, eligible_visible);
    enc.set_tensor(5, eligibility_record);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
fn encode_validate_indexer_fp4_rows_shadow(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    packed_values: &MetalTensor,
    packed_scales: &MetalTensor,
    status: &MetalTensor,
    row_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const VALUE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES;
    const SCALE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES;
    if row_count == 0 || u32::try_from(row_count).is_err() {
        return invalid("indexer FP4 validation row count must be nonzero and fit u32");
    }
    validate_i8(
        packed_values,
        &[VALUE_BYTES as u64, row_count as u64],
        false,
        "indexer FP4 validation values",
    )?;
    validate_i8(
        packed_scales,
        &[SCALE_BYTES as u64, row_count as u64],
        false,
        "indexer FP4 validation scales",
    )?;
    validate_i32(
        status,
        &[row_count as u64],
        true,
        "indexer FP4 validation status",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_count: u32,
    }

    let pso = ctx.pipeline("kernel_deepseek_v4_validate_indexer_fp4_rows_shadow")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_count: row_count as u32,
        },
    );
    enc.set_tensor(1, packed_values);
    enc.set_tensor(2, packed_scales);
    enc.set_tensor(3, status);
    enc.dispatch(
        MTLSize {
            width: row_count.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
#[allow(clippy::too_many_arguments)]
fn encode_lightning_indexer_scores_fp4_matrix_shadow(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query_units: &MetalTensor,
    query_scales: &MetalTensor,
    head_weights: &MetalTensor,
    key_values: &MetalTensor,
    key_scales: &MetalTensor,
    visible_counts: &MetalTensor,
    scores: &MetalTensor,
    row_capacity: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const KERNEL: &str = "kernel_deepseek_v4_lightning_indexer_scores_fp4_matrix_shadow";
    const HEAD_COUNT: usize = 64;
    const HEAD_DIM: usize = 128;
    const VALUE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES;
    const SCALE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES;
    if row_capacity == 0
        || query_count == 0
        || u32::try_from(row_capacity).is_err()
        || u32::try_from(query_count).is_err()
    {
        return invalid("FP4 matrix score row and query counts must be nonzero and fit u32");
    }
    validate_fp4_lightning_score_offsets(row_capacity, query_count)?;
    validate_f16(
        query_units,
        &[HEAD_DIM as u64, HEAD_COUNT as u64, query_count as u64],
        false,
        "FP4 matrix indexer query units",
    )?;
    validate_i8(
        query_scales,
        &[SCALE_BYTES as u64, HEAD_COUNT as u64, query_count as u64],
        false,
        "FP4 matrix indexer query scales",
    )?;
    validate_f32(
        head_weights,
        &[HEAD_COUNT as u64, query_count as u64],
        false,
        "FP4 matrix indexer head weights",
    )?;
    validate_i8(
        key_values,
        &[VALUE_BYTES as u64, row_capacity as u64],
        false,
        "FP4 matrix indexer key values",
    )?;
    validate_i8(
        key_scales,
        &[SCALE_BYTES as u64, row_capacity as u64],
        false,
        "FP4 matrix indexer key scales",
    )?;
    validate_i32(
        visible_counts,
        &[query_count as u64],
        false,
        "FP4 matrix indexer visible counts",
    )?;
    validate_f32(
        scores,
        &[row_capacity as u64, query_count as u64],
        true,
        "FP4 matrix indexer scores",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        row_capacity: u32,
        query_count: u32,
    }

    let pso = ctx.pipeline(KERNEL)?;
    validate_fp4_matrix_score_geometry(
        KERNEL,
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        ctx.device.maxThreadgroupMemoryLength(),
    )?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: HEAD_COUNT as u32,
            head_dim: HEAD_DIM as u32,
            row_capacity: row_capacity as u32,
            query_count: query_count as u32,
        },
    );
    enc.set_tensor(1, query_units);
    enc.set_tensor(2, query_scales);
    enc.set_tensor(3, head_weights);
    enc.set_tensor(4, key_values);
    enc.set_tensor(5, key_scales);
    enc.set_tensor(6, visible_counts);
    enc.set_tensor(7, scores);
    enc.set_threadgroup_memory(0, 8 * 32 * std::mem::size_of::<u16>());
    enc.set_threadgroup_memory(1, 64 * 8 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: row_capacity.div_ceil(8),
            height: query_count,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn validate_fp4_lightning_score_offsets(
    row_capacity: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const HEAD_COUNT: usize = 64;
    const VALUE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES;
    const SCALE_BYTES: usize = crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES;
    for (name, elements) in [
        (
            "FP4 indexer score elements",
            checked_mul(row_capacity, query_count, "FP4 indexer score elements")?,
        ),
        (
            "FP4 indexer weight elements",
            checked_mul(HEAD_COUNT, query_count, "FP4 indexer weight elements")?,
        ),
        (
            "FP4 indexer query unit elements",
            checked_mul(
                checked_mul(HEAD_COUNT, query_count, "FP4 indexer query rows")?,
                crate::deepseek_v4_oracle::INDEXER_FP4_VALUES_PER_ROW,
                "FP4 indexer query unit elements",
            )?,
        ),
        (
            "FP4 indexer query scale bytes",
            checked_mul(
                checked_mul(HEAD_COUNT, query_count, "FP4 indexer query rows")?,
                SCALE_BYTES,
                "FP4 indexer query scale bytes",
            )?,
        ),
        (
            "FP4 indexer key value bytes",
            checked_mul(row_capacity, VALUE_BYTES, "FP4 indexer key value bytes")?,
        ),
        (
            "FP4 indexer key scale bytes",
            checked_mul(row_capacity, SCALE_BYTES, "FP4 indexer key scale bytes")?,
        ),
    ] {
        if u32::try_from(elements).is_err() {
            return invalid(format!("{name} exceed u32 shader offsets"));
        }
    }
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn validate_fp4_matrix_score_geometry(
    kernel: &str,
    thread_execution_width: usize,
    max_threads_per_group: usize,
    max_threadgroup_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADS: usize = 256;
    const KEY_BYTES: usize = 8 * 32 * std::mem::size_of::<u16>();
    const DOT_BYTES: usize = 64 * 8 * std::mem::size_of::<f32>();
    const THREADGROUP_BYTES: usize = KEY_BYTES + DOT_BYTES;
    if thread_execution_width != 32 || max_threads_per_group < THREADS {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and {THREADS} threads, got width {thread_execution_width} max {max_threads_per_group}"
        ));
    }
    if max_threadgroup_bytes < THREADGROUP_BYTES {
        return invalid(format!(
            "{kernel} requires {THREADGROUP_BYTES} threadgroup bytes, device allows {max_threadgroup_bytes}"
        ));
    }
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
fn validate_fp4_pack_geometry(
    thread_execution_width: usize,
    max_threads_per_group: usize,
    max_threadgroup_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADS: usize = 128;
    const THREADGROUP_BYTES: usize = 128 * std::mem::size_of::<f32>()
        + 64 * std::mem::size_of::<u8>()
        + 4 * std::mem::size_of::<u8>()
        + 4 * std::mem::size_of::<u32>();
    if thread_execution_width != 32 || max_threads_per_group < THREADS {
        return invalid(format!(
            "FP4 row pack requires SIMD width 32 and {THREADS} threads, got width {thread_execution_width} max {max_threads_per_group}"
        ));
    }
    if max_threadgroup_bytes < THREADGROUP_BYTES {
        return invalid(format!(
            "FP4 row pack requires {THREADGROUP_BYTES} threadgroup bytes, device allows {max_threadgroup_bytes}"
        ));
    }
    Ok(())
}

fn validate_lightning_indexer_score_offsets(
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let query_width = checked_mul(head_count, head_dim, "indexer score query width")?;
    let products = [
        (
            "indexer score elements",
            checked_mul(row_capacity, query_count, "indexer score elements")?,
        ),
        (
            "indexer score weight elements",
            checked_mul(head_count, query_count, "indexer score weight elements")?,
        ),
        (
            "indexer score query elements",
            checked_mul(query_width, query_count, "indexer score query elements")?,
        ),
        (
            "indexer score key elements",
            checked_mul(row_capacity, head_dim, "indexer score key elements")?,
        ),
    ];
    for (name, elements) in products {
        if u32::try_from(elements).is_err() {
            return invalid(format!("{name} exceed u32 shader offsets"));
        }
    }
    Ok(())
}

fn validate_cooperative_lightning_score_geometry(
    kernel: &str,
    thread_execution_width: usize,
    max_threads_per_group: usize,
    max_threadgroup_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADS: usize = 256;
    const KEY_BYTES: usize = 8 * 128 * std::mem::size_of::<u16>();
    const DOT_BYTES: usize = 8 * 64 * std::mem::size_of::<f32>();
    const THREADGROUP_BYTES: usize = KEY_BYTES + DOT_BYTES;

    if thread_execution_width != 32 || max_threads_per_group < THREADS {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and {THREADS} threads, got width {thread_execution_width} max {max_threads_per_group}"
        ));
    }
    if max_threadgroup_bytes < THREADGROUP_BYTES {
        return invalid(format!(
            "{kernel} requires {THREADGROUP_BYTES} threadgroup bytes, device allows {max_threadgroup_bytes}"
        ));
    }
    Ok(())
}

fn validate_tiled_f32_lightning_score_geometry(
    kernel: &str,
    thread_execution_width: usize,
    max_threads_per_group: usize,
    max_threadgroup_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADS: usize = 128;
    const THREADGROUP_BYTES: usize = 5_376 * std::mem::size_of::<f32>();
    if thread_execution_width != 32 || max_threads_per_group < THREADS {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and {THREADS} threads, got width {thread_execution_width} max {max_threads_per_group}"
        ));
    }
    if max_threadgroup_bytes < THREADGROUP_BYTES {
        return invalid(format!(
            "{kernel} requires {THREADGROUP_BYTES} threadgroup bytes, device allows {max_threadgroup_bytes}"
        ));
    }
    Ok(())
}

#[cfg(any(test, feature = "dsv4-diagnostics"))]
#[allow(clippy::too_many_arguments)]
fn encode_select_top_k_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    visible_counts: &MetalTensor,
    selected_mask: &MetalTensor,
    ranked_ids: Option<&MetalTensor>,
    cache_order_ids: &MetalTensor,
    selected_counts: &MetalTensor,
    status: &MetalTensor,
    row_capacity: usize,
    max_visible_rows: usize,
    top_k: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    encode_select_top_k_f32_with_policy(
        ctx,
        enc,
        scores,
        visible_counts,
        selected_mask,
        ranked_ids,
        cache_order_ids,
        selected_counts,
        status,
        row_capacity,
        max_visible_rows,
        top_k,
        query_count,
        DeepSeekV4SelectorDispatchPolicy::Production,
        true,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeepSeekV4SelectorDispatchPolicy {
    Production,
    #[cfg(test)]
    ScalarOracle,
    #[cfg(test)]
    Parallel,
}

fn use_parallel_selector(
    dispatch_policy: DeepSeekV4SelectorDispatchPolicy,
    max_visible_rows: usize,
    top_k: usize,
) -> bool {
    match dispatch_policy {
        // Once one row must be pruned, the scalar oracle's repeated worst-row
        // scans grow as (visible - top_k) * visible through the shallow band.
        DeepSeekV4SelectorDispatchPolicy::Production => max_visible_rows > top_k,
        #[cfg(test)]
        DeepSeekV4SelectorDispatchPolicy::ScalarOracle => false,
        #[cfg(test)]
        DeepSeekV4SelectorDispatchPolicy::Parallel => true,
    }
}

#[cfg(test)]
const DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_GREATER: usize = 0;
#[cfg(test)]
const DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_EQUAL: usize = 1;
#[cfg(test)]
const DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_TIE_QUOTA: usize = 2;
#[cfg(test)]
const DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_SELECTED: usize = 3;
#[cfg(test)]
const DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_ID_OFFSET: usize = 4;
#[cfg(test)]
const DEEPSEEK_V4_MULTIGROUP_SELECTOR_COMPACT_PHASE: i32 = 8;

#[cfg(test)]
fn deepseek_v4_multigroup_selector_record_completion(
    generation: u32,
    digit: usize,
    group: usize,
) -> u32 {
    0xd541_0000 ^ generation ^ ((digit as u32) << 12) ^ group as u32
}

#[cfg(test)]
fn deepseek_v4_multigroup_selector_state_completion(generation: u32, digit: usize) -> u32 {
    0xd542_0000 ^ generation ^ ((digit as u32) << 12)
}

#[cfg(test)]
fn deepseek_v4_multigroup_selector_compact_completion(generation: u32, group: usize) -> u32 {
    0xd543_0000 ^ generation ^ group as u32
}

#[allow(clippy::too_many_arguments)]
fn encode_select_top_k_multigroup_threshold_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    visible_counts: &MetalTensor,
    records: &MetalTensor,
    partition_plan: &MetalTensor,
    state: &MetalTensor,
    row_capacity: usize,
    top_k: usize,
    group_count: usize,
    generation: u32,
    fault_digit: Option<usize>,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADGROUP_WIDTH: usize = 256;
    const SIMD_GROUPS: usize = THREADGROUP_WIDTH / 32;
    const HISTOGRAM_WORDS: usize = SIMD_GROUPS * 16;
    const ERROR_WORDS: usize = SIMD_GROUPS;
    require_serial(enc, "deepseek_v4_multigroup_selector_threshold")?;
    if row_capacity == 0
        || top_k == 0
        || top_k > row_capacity
        || !(2..=THREADGROUP_WIDTH).contains(&group_count)
        || generation == 0
        || fault_digit.is_some_and(|digit| digit >= DEEPSEEK_V4_MULTIGROUP_SELECTOR_DIGITS)
        || [row_capacity, top_k, group_count]
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
    {
        return invalid("multi-group selector threshold geometry is invalid");
    }
    let score_elements = checked_mul(row_capacity, 1, "multi-group selector score elements")?;
    let record_elements = checked_mul(
        DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS,
        group_count,
        "multi-group selector record elements",
    )?;
    let partition_elements = checked_mul(
        DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS,
        group_count,
        "multi-group selector partition plan",
    )?;
    if [score_elements, record_elements, partition_elements]
        .into_iter()
        .any(|value| u32::try_from(value).is_err())
    {
        return invalid("multi-group selector threshold offsets exceed u32");
    }
    validate_f32(
        scores,
        &[row_capacity as u64, 1],
        false,
        "multi-group selector scores",
    )?;
    validate_i32(
        visible_counts,
        &[1],
        false,
        "multi-group selector visibility",
    )?;
    validate_i32(
        records,
        &[
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
            group_count as u64,
        ],
        true,
        "multi-group selector records",
    )?;
    validate_i32(
        partition_plan,
        &[
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
            group_count as u64,
        ],
        true,
        "multi-group selector partition plan",
    )?;
    validate_i32(
        state,
        &[DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
        true,
        "multi-group selector state",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        group_count: u32,
        generation: u32,
        digit: u32,
        shift: u32,
    }

    let producer = ctx.pipeline("kernel_deepseek_v4_select_top_k_multigroup_histogram_f32")?;
    validate_parallel_selector_pipeline(
        producer.threadExecutionWidth(),
        producer.maxTotalThreadsPerThreadgroup(),
    )?;
    let reducer = ctx.pipeline("kernel_deepseek_v4_select_top_k_multigroup_reduce_f32")?;
    for digit in 0..DEEPSEEK_V4_MULTIGROUP_SELECTOR_DIGITS {
        let args = Args {
            row_capacity: row_capacity as u32,
            top_k: top_k as u32,
            group_count: group_count as u32,
            generation,
            digit: digit as u32,
            shift: 28 - digit as u32 * 4,
        };
        enc.set_pipeline(&producer);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, scores);
        enc.set_tensor(2, visible_counts);
        enc.set_tensor(3, state);
        enc.set_tensor(4, records);
        enc.set_threadgroup_memory(
            0,
            (HISTOGRAM_WORDS + ERROR_WORDS) * std::mem::size_of::<u32>(),
        );
        let producer_groups = if fault_digit == Some(digit) {
            group_count - 1
        } else {
            group_count
        };
        enc.dispatch(
            MTLSize {
                width: producer_groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADGROUP_WIDTH,
                height: 1,
                depth: 1,
            },
        );

        enc.set_pipeline(&reducer);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, visible_counts);
        enc.set_tensor(2, records);
        enc.set_tensor(3, partition_plan);
        enc.set_tensor(4, state);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_select_top_k_multigroup_full_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    visible_counts: &MetalTensor,
    records: &MetalTensor,
    partition_plan: &MetalTensor,
    state: &MetalTensor,
    private_mask: &MetalTensor,
    private_ids: &MetalTensor,
    selected_mask: &MetalTensor,
    cache_order_ids: &MetalTensor,
    selected_counts: &MetalTensor,
    status: &MetalTensor,
    row_capacity: usize,
    top_k: usize,
    generation: u32,
    fault_digit: Option<usize>,
    omit_last_compactor: bool,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADGROUP_WIDTH: usize = 256;
    const COMPACT_SCRATCH_WORDS: usize = 3 * THREADGROUP_WIDTH + 20;
    const PUBLISH_SCRATCH_WORDS: usize = 20;
    let group_count = DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS;
    require_serial(enc, "deepseek_v4_multigroup_selector_full")?;
    validate_i8(
        private_mask,
        &[row_capacity as u64, 1],
        true,
        "multi-group selector private mask",
    )?;
    for (tensor, shape, name) in [
        (
            private_ids,
            vec![top_k as u64, 1],
            "multi-group selector private IDs",
        ),
        (
            selected_mask,
            vec![row_capacity as u64, 1],
            "multi-group selector published mask",
        ),
        (
            cache_order_ids,
            vec![top_k as u64, 1],
            "multi-group selector published IDs",
        ),
        (
            selected_counts,
            vec![1],
            "multi-group selector published count",
        ),
        (status, vec![1], "multi-group selector published status"),
    ] {
        validate_i32(tensor, &shape, true, name)?;
    }
    let compactor = ctx.pipeline("kernel_deepseek_v4_select_top_k_multigroup_compact_f32")?;
    validate_parallel_selector_pipeline(
        compactor.threadExecutionWidth(),
        compactor.maxTotalThreadsPerThreadgroup(),
    )?;
    let publisher = ctx.pipeline("kernel_deepseek_v4_select_top_k_multigroup_publish_f32")?;
    validate_parallel_selector_pipeline(
        publisher.threadExecutionWidth(),
        publisher.maxTotalThreadsPerThreadgroup(),
    )?;

    encode_select_top_k_multigroup_threshold_f32(
        ctx,
        enc,
        scores,
        visible_counts,
        records,
        partition_plan,
        state,
        row_capacity,
        top_k,
        group_count,
        generation,
        fault_digit,
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        group_count: u32,
        generation: u32,
        digit: u32,
        shift: u32,
    }
    let args = Args {
        row_capacity: row_capacity as u32,
        top_k: top_k as u32,
        group_count: group_count as u32,
        generation,
        digit: DEEPSEEK_V4_MULTIGROUP_SELECTOR_DIGITS as u32,
        shift: 0,
    };
    enc.set_pipeline(&compactor);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, scores);
    enc.set_tensor(2, visible_counts);
    enc.set_tensor(3, state);
    enc.set_tensor(4, partition_plan);
    enc.set_tensor(5, private_mask);
    enc.set_tensor(6, private_ids);
    enc.set_tensor(7, records);
    enc.set_threadgroup_memory(0, COMPACT_SCRATCH_WORDS * std::mem::size_of::<u32>());
    enc.dispatch(
        MTLSize {
            width: group_count - usize::from(omit_last_compactor),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );

    enc.set_pipeline(&publisher);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, visible_counts);
    enc.set_tensor(2, state);
    enc.set_tensor(3, partition_plan);
    enc.set_tensor(4, records);
    enc.set_tensor(5, private_mask);
    enc.set_tensor(6, private_ids);
    enc.set_tensor(7, selected_mask);
    enc.set_tensor(8, cache_order_ids);
    enc.set_tensor(9, selected_counts);
    enc.set_tensor(10, status);
    enc.set_threadgroup_memory(0, PUBLISH_SCRATCH_WORDS * std::mem::size_of::<u32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_select_top_k_multigroup_publish_only_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    visible_counts: &MetalTensor,
    records: &MetalTensor,
    partition_plan: &MetalTensor,
    state: &MetalTensor,
    private_mask: &MetalTensor,
    private_ids: &MetalTensor,
    selected_mask: &MetalTensor,
    cache_order_ids: &MetalTensor,
    selected_counts: &MetalTensor,
    status: &MetalTensor,
    row_capacity: usize,
    top_k: usize,
    generation: u32,
) -> Result<(), DeepSeekV4MetalError> {
    const THREADGROUP_WIDTH: usize = 256;
    const PUBLISH_SCRATCH_WORDS: usize = 20;
    let group_count = DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS;
    require_serial(enc, "deepseek_v4_multigroup_selector_publish_only")?;
    if row_capacity == 0
        || top_k == 0
        || top_k > row_capacity
        || generation == 0
        || [row_capacity, top_k]
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
    {
        return invalid("multi-group selector publisher geometry is invalid");
    }
    validate_i32(
        visible_counts,
        &[1],
        false,
        "multi-group selector publisher visibility",
    )?;
    validate_i32(
        records,
        &[
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS as u64,
            group_count as u64,
        ],
        false,
        "multi-group selector publisher records",
    )?;
    validate_i32(
        partition_plan,
        &[
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS as u64,
            group_count as u64,
        ],
        false,
        "multi-group selector publisher plan",
    )?;
    validate_i32(
        state,
        &[DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS as u64],
        false,
        "multi-group selector publisher state",
    )?;
    validate_i8(
        private_mask,
        &[row_capacity as u64, 1],
        false,
        "multi-group selector publisher private mask",
    )?;
    for (tensor, shape, writable, name) in [
        (
            private_ids,
            vec![top_k as u64, 1],
            false,
            "multi-group selector publisher private IDs",
        ),
        (
            selected_mask,
            vec![row_capacity as u64, 1],
            true,
            "multi-group selector publisher mask",
        ),
        (
            cache_order_ids,
            vec![top_k as u64, 1],
            true,
            "multi-group selector publisher IDs",
        ),
        (
            selected_counts,
            vec![1],
            true,
            "multi-group selector publisher count",
        ),
        (
            status,
            vec![1],
            true,
            "multi-group selector publisher status",
        ),
    ] {
        validate_i32(tensor, &shape, writable, name)?;
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        group_count: u32,
        generation: u32,
        digit: u32,
        shift: u32,
    }
    let publisher = ctx.pipeline("kernel_deepseek_v4_select_top_k_multigroup_publish_f32")?;
    validate_parallel_selector_pipeline(
        publisher.threadExecutionWidth(),
        publisher.maxTotalThreadsPerThreadgroup(),
    )?;
    enc.set_pipeline(&publisher);
    enc.set_bytes(
        0,
        &Args {
            row_capacity: row_capacity as u32,
            top_k: top_k as u32,
            group_count: group_count as u32,
            generation,
            digit: DEEPSEEK_V4_MULTIGROUP_SELECTOR_DIGITS as u32,
            shift: 0,
        },
    );
    enc.set_tensor(1, visible_counts);
    enc.set_tensor(2, state);
    enc.set_tensor(3, partition_plan);
    enc.set_tensor(4, records);
    enc.set_tensor(5, private_mask);
    enc.set_tensor(6, private_ids);
    enc.set_tensor(7, selected_mask);
    enc.set_tensor(8, cache_order_ids);
    enc.set_tensor(9, selected_counts);
    enc.set_tensor(10, status);
    enc.set_threadgroup_memory(0, PUBLISH_SCRATCH_WORDS * std::mem::size_of::<u32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_select_top_k_f32_with_policy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    visible_counts: &MetalTensor,
    selected_mask: &MetalTensor,
    ranked_ids: Option<&MetalTensor>,
    cache_order_ids: &MetalTensor,
    selected_counts: &MetalTensor,
    status: &MetalTensor,
    row_capacity: usize,
    max_visible_rows: usize,
    top_k: usize,
    query_count: usize,
    dispatch_policy: DeepSeekV4SelectorDispatchPolicy,
    radix4: bool,
) -> Result<(), DeepSeekV4MetalError> {
    if row_capacity == 0
        || max_visible_rows == 0
        || max_visible_rows > row_capacity
        || top_k == 0
        || top_k > row_capacity
        || query_count == 0
        || [row_capacity, top_k, query_count]
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
    {
        return invalid("indexer selection geometry is invalid");
    }
    let score_elements = checked_mul(
        row_capacity,
        query_count,
        "indexer selection score elements",
    )?;
    let id_elements = checked_mul(top_k, query_count, "indexer selection ID elements")?;
    if u32::try_from(score_elements).is_err() || u32::try_from(id_elements).is_err() {
        return invalid("indexer selection buffer offsets exceed u32");
    }
    validate_f32(
        scores,
        &[row_capacity as u64, query_count as u64],
        false,
        "indexer selection scores",
    )?;
    validate_i32(
        visible_counts,
        &[query_count as u64],
        false,
        "indexer selection visible counts",
    )?;
    validate_i32(
        selected_mask,
        &[row_capacity as u64, query_count as u64],
        true,
        "indexer selection mask",
    )?;
    if let Some(ranked_ids) = ranked_ids {
        validate_i32(
            ranked_ids,
            &[top_k as u64, query_count as u64],
            true,
            "ranked indexer IDs",
        )?;
    }
    validate_i32(
        cache_order_ids,
        &[top_k as u64, query_count as u64],
        true,
        "cache-order indexer IDs",
    )?;
    validate_i32(
        selected_counts,
        &[query_count as u64],
        true,
        "indexer selected counts",
    )?;
    validate_i32(
        status,
        &[query_count as u64],
        true,
        "indexer selection status",
    )?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        query_count: u32,
        emit_ranked: u32,
    }
    let emit_ranked = ranked_ids.is_some();
    let ranked_ids = ranked_ids.unwrap_or(cache_order_ids);
    let parallel = use_parallel_selector(dispatch_policy, max_visible_rows, top_k);
    let radix4 = radix4 && parallel;
    let pso = ctx.pipeline(if radix4 {
        "kernel_deepseek_v4_select_top_k_radix4_f32"
    } else if parallel {
        "kernel_deepseek_v4_select_top_k_parallel_f32"
    } else {
        "kernel_deepseek_v4_select_top_k_f32"
    })?;
    if parallel {
        validate_parallel_selector_pipeline(
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup(),
        )?;
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_capacity: row_capacity as u32,
            top_k: top_k as u32,
            query_count: query_count as u32,
            emit_ranked: u32::from(emit_ranked),
        },
    );
    enc.set_tensor(1, scores);
    enc.set_tensor(2, visible_counts);
    enc.set_tensor(3, selected_mask);
    enc.set_tensor(4, ranked_ids);
    enc.set_tensor(5, cache_order_ids);
    enc.set_tensor(6, selected_counts);
    enc.set_tensor(7, status);
    if parallel {
        const THREADGROUP_WIDTH: usize = 256;
        enc.set_threadgroup_memory(0, THREADGROUP_WIDTH * std::mem::size_of::<u32>());
        enc.set_threadgroup_memory(1, THREADGROUP_WIDTH * std::mem::size_of::<u32>());
        enc.dispatch(
            MTLSize {
                width: query_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADGROUP_WIDTH,
                height: 1,
                depth: 1,
            },
        );
    } else {
        enc.dispatch(
            MTLSize {
                width: query_count.div_ceil(64),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(feature = "dsv4-diagnostics", allow(dead_code))]
fn encode_select_top_k_radix4_ids_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    visible_counts: &MetalTensor,
    cache_order_ids: &MetalTensor,
    selected_counts: &MetalTensor,
    status: &MetalTensor,
    row_capacity: usize,
    max_visible_rows: usize,
    top_k: usize,
    query_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if row_capacity == 0
        || max_visible_rows <= top_k
        || max_visible_rows > row_capacity
        || top_k == 0
        || top_k > row_capacity
        || query_count == 0
        || [row_capacity, top_k, query_count]
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
    {
        return invalid("maskless radix4 selector requires parallel selection geometry");
    }
    let score_elements = checked_mul(
        row_capacity,
        query_count,
        "maskless radix4 selector score elements",
    )?;
    let id_elements = checked_mul(top_k, query_count, "maskless radix4 selector ID elements")?;
    if u32::try_from(score_elements).is_err() || u32::try_from(id_elements).is_err() {
        return invalid("maskless radix4 selector buffer offsets exceed u32");
    }
    validate_f32(
        scores,
        &[row_capacity as u64, query_count as u64],
        false,
        "maskless radix4 selector scores",
    )?;
    validate_i32(
        visible_counts,
        &[query_count as u64],
        false,
        "maskless radix4 selector visible counts",
    )?;
    validate_i32(
        cache_order_ids,
        &[top_k as u64, query_count as u64],
        true,
        "maskless radix4 selector cache-order IDs",
    )?;
    validate_i32(
        selected_counts,
        &[query_count as u64],
        true,
        "maskless radix4 selector counts",
    )?;
    validate_i32(
        status,
        &[query_count as u64],
        true,
        "maskless radix4 selector status",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        query_count: u32,
        emit_ranked: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_select_top_k_radix4_ids_f32")?;
    validate_parallel_selector_pipeline(
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
    )?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_capacity: row_capacity as u32,
            top_k: top_k as u32,
            query_count: query_count as u32,
            emit_ranked: 0,
        },
    );
    enc.set_tensor(1, scores);
    enc.set_tensor(2, visible_counts);
    enc.set_tensor(3, cache_order_ids);
    enc.set_tensor(4, selected_counts);
    enc.set_tensor(5, status);
    const THREADGROUP_WIDTH: usize = 256;
    enc.set_threadgroup_memory(0, THREADGROUP_WIDTH * std::mem::size_of::<u32>());
    enc.set_threadgroup_memory(1, THREADGROUP_WIDTH * std::mem::size_of::<u32>());
    enc.dispatch(
        MTLSize {
            width: query_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn validate_parallel_selector_pipeline(
    thread_execution_width: usize,
    max_threads_per_threadgroup: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const REQUIRED_SIMD_WIDTH: usize = 32;
    const THREADGROUP_WIDTH: usize = 256;
    if thread_execution_width != REQUIRED_SIMD_WIDTH {
        return invalid(format!(
            "parallel indexer selector requires {REQUIRED_SIMD_WIDTH}-lane SIMD groups, pipeline reports {thread_execution_width}"
        ));
    }
    if max_threads_per_threadgroup < THREADGROUP_WIDTH {
        return invalid(format!(
            "parallel indexer selector supports {max_threads_per_threadgroup} threads, requires {THREADGROUP_WIDTH}"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_selected_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    compressed_cache: &MetalTensor,
    selected_ids: &MetalTensor,
    sinks: &MetalTensor,
    output: &MetalTensor,
    position: u32,
    compressed_count: usize,
    selected_slots: usize,
    compressed_capacity: usize,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    let query_width = checked_mul(config.head_count, config.head_dim, "selected query width")?;
    validate_f32(
        queries,
        &[config.head_dim as u64, config.head_count as u64],
        false,
        "selected attention queries",
    )?;
    validate_f16(
        raw_cache,
        &[config.head_dim as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
        false,
        "selected attention raw cache",
    )?;
    validate_f16(
        compressed_cache,
        &[config.head_dim as u64, compressed_capacity as u64],
        false,
        "selected attention compressed cache",
    )?;
    validate_i32(
        selected_ids,
        &[selected_slots as u64],
        false,
        "selected attention row IDs",
    )?;
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "selected attention sinks",
    )?;
    validate_f32(
        output,
        &[config.head_dim as u64, config.head_count as u64],
        true,
        "selected attention output",
    )?;
    if compressed_count == 0
        || compressed_count > compressed_capacity
        || selected_slots == 0
        || selected_slots > compressed_count
        || [compressed_count, selected_slots, compressed_capacity]
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
    {
        return invalid("selected attention compressed geometry is invalid");
    }
    let visible_end = u64::from(position) + 1;
    let raw_count = visible_end.min(DEEPSEEK_V4_LOCAL_WINDOW as u64) as u32;
    let raw_start = u32::try_from(visible_end - u64::from(raw_count)).map_err(|_| {
        DeepSeekV4MetalError::Invalid("selected attention raw start exceeds u32".into())
    })?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        window: u32,
        raw_count: u32,
        raw_start: u32,
        compressed_count: u32,
        selected_slots: u32,
        scale: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_selected_sink_attention_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: config.head_count as u32,
            head_dim: config.head_dim as u32,
            window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
            raw_count,
            raw_start,
            compressed_count: compressed_count as u32,
            selected_slots: selected_slots as u32,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, compressed_cache);
    enc.set_tensor(4, selected_ids);
    enc.set_tensor(5, sinks);
    enc.set_tensor(6, output);
    enc.dispatch(
        MTLSize {
            width: query_width.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_cooperative_selected_sink_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    raw_cache: &MetalTensor,
    raw_cache_before_chunk: &MetalTensor,
    raw_cache_layout: DeepSeekV4RawCacheLayout,
    compressed_cache: &MetalTensor,
    compressed_capacity: usize,
    selected_ids: &MetalTensor,
    selected_counts: &MetalTensor,
    visible_counts: &MetalTensor,
    sinks: &MetalTensor,
    output: &MetalTensor,
    chunk_start_position: u32,
    query_token_offset: usize,
    query_count: usize,
    token_count: usize,
    selected_slots: usize,
    online: bool,
    direct_load: bool,
    config: DeepSeekV4PositionZeroAttentionConfig,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "deepseek_v4_cooperative_selected_attention")?;
    let query_width = checked_mul(
        config.head_count,
        config.head_dim,
        "cooperative selected query width",
    )?;
    let query_end = query_token_offset.checked_add(query_count).ok_or_else(|| {
        DeepSeekV4MetalError::Invalid("cooperative selected query range overflow".into())
    })?;
    if compressed_capacity == 0
        || selected_slots == 0
        || selected_slots > compressed_capacity
        || query_count == 0
        || token_count == 0
        || query_token_offset >= token_count
        || query_end > token_count
        || [
            config.head_count,
            config.head_dim,
            query_width,
            compressed_capacity,
            selected_slots,
            query_token_offset,
            query_count,
            token_count,
        ]
        .into_iter()
        .any(|value| u32::try_from(value).is_err())
    {
        return invalid("cooperative selected attention geometry is invalid");
    }
    let final_token = query_end - 1;
    chunk_start_position
        .checked_add(u32::try_from(final_token).map_err(|_| {
            DeepSeekV4MetalError::Invalid("cooperative selected final token exceeds u32".into())
        })?)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("cooperative selected absolute position overflow".into())
        })?;
    validate_f32(
        queries,
        &[query_width as u64, token_count as u64],
        false,
        "cooperative selected attention queries",
    )?;
    validate_raw_attention_caches(
        raw_cache,
        raw_cache_before_chunk,
        raw_cache_layout,
        config.head_dim,
        token_count,
        "cooperative selected attention",
    )?;
    validate_f16(
        compressed_cache,
        &[config.head_dim as u64, compressed_capacity as u64],
        false,
        "cooperative selected compressed cache",
    )?;
    validate_i32(
        selected_ids,
        &[selected_slots as u64, query_count as u64],
        false,
        "cooperative selected row IDs",
    )?;
    for (tensor, name) in [
        (selected_counts, "cooperative selected row counts"),
        (visible_counts, "cooperative selected visible counts"),
    ] {
        validate_i32(tensor, &[query_count as u64], false, name)?;
    }
    validate_f32(
        sinks,
        &[config.head_count as u64],
        false,
        "cooperative selected attention sinks",
    )?;
    validate_f32(
        output,
        &[query_width as u64, token_count as u64],
        true,
        "cooperative selected attention output",
    )?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        query_count: u32,
        query_token_offset: u32,
        chunk_start_position: u32,
        window: u32,
        selected_slots: u32,
        compressed_capacity: u32,
        raw_cache_is_chunk: u32,
        scale: f32,
    }
    let maximum_rows = DEEPSEEK_V4_LOCAL_WINDOW
        .checked_add(selected_slots)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("cooperative selected maximum row count overflow".into())
        })?;
    if direct_load && !online {
        return invalid("direct row loading requires online selected attention");
    }
    let (kernel, threadgroup_width, threadgroup_bytes) = if online {
        if config.head_count != 64
            || config.head_dim != DEEPSEEK_V4_HCA_TILE_ROWS
            || selected_slots != DEEPSEEK_V4_CSA_TOP_K
        {
            return invalid(
                "online selected attention requires 64 heads x 512 dimensions and top-512 rows",
            );
        }
        (
            if direct_load {
                "kernel_deepseek_v4_online_packed_selected_sink_attention_f16_direct"
            } else {
                "kernel_deepseek_v4_online_packed_selected_sink_attention_f16"
            },
            DEEPSEEK_V4_ONLINE_HCA_THREADS,
            if direct_load {
                0
            } else {
                DEEPSEEK_V4_ONLINE_HCA_THREADGROUP_BYTES
            },
        )
    } else {
        (
            "kernel_deepseek_v4_packed_selected_sink_attention_f16",
            config.head_dim.max(maximum_rows),
            (maximum_rows + 1) * std::mem::size_of::<f32>(),
        )
    };
    let pso = ctx.pipeline(kernel)?;
    if online {
        validate_deepseek_v4_online_hca_launch_geometry(
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup(),
            ctx.device.maxThreadgroupMemoryLength(),
            threadgroup_bytes,
        )?;
    } else if pso.maxTotalThreadsPerThreadgroup() < threadgroup_width {
        return invalid(format!(
            "cooperative selected attention pipeline supports {} threads, requires {threadgroup_width}",
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: config.head_count as u32,
            head_dim: config.head_dim as u32,
            query_count: query_count as u32,
            query_token_offset: query_token_offset as u32,
            chunk_start_position,
            window: DEEPSEEK_V4_LOCAL_WINDOW as u32,
            selected_slots: selected_slots as u32,
            compressed_capacity: compressed_capacity as u32,
            raw_cache_is_chunk: raw_cache_layout.is_chunk(),
            scale: 1.0 / (config.head_dim as f32).sqrt(),
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, raw_cache);
    enc.set_tensor(3, raw_cache_before_chunk);
    enc.set_tensor(4, compressed_cache);
    enc.set_tensor(5, selected_ids);
    enc.set_tensor(6, selected_counts);
    enc.set_tensor(7, visible_counts);
    enc.set_tensor(8, sinks);
    enc.set_tensor(9, output);
    if threadgroup_bytes != 0 {
        enc.set_threadgroup_memory(0, threadgroup_bytes);
    }
    enc.dispatch(
        MTLSize {
            width: query_count,
            height: config.head_count,
            depth: 1,
        },
        MTLSize {
            width: threadgroup_width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_f32_projection(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    function: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), DeepSeekV4MetalError> {
    crate::metal_forward::encode_mat_vec_dispatch(ctx, enc, function, input, output, n_in, n_out)
        .map_err(|error| match error {
            crate::metal_forward::MfError::Metal(error) => DeepSeekV4MetalError::Metal(error),
            other => DeepSeekV4MetalError::Invalid(format!("F32 HC projection failed: {other}")),
        })
}

fn residual_len(hidden_size: usize) -> Result<usize, DeepSeekV4MetalError> {
    if hidden_size == 0 {
        return invalid("hidden size must be nonzero");
    }
    let len = hidden_size
        .checked_mul(DEEPSEEK_V4_CONNECTION_COUNT)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("4 * hidden size overflow".into()))?;
    u32::try_from(len)
        .map_err(|_| DeepSeekV4MetalError::Invalid("4 * hidden size exceeds u32".into()))?;
    Ok(len)
}

fn u32_hidden(hidden_size: usize) -> Result<u32, DeepSeekV4MetalError> {
    residual_len(hidden_size)?;
    u32::try_from(hidden_size)
        .map_err(|_| DeepSeekV4MetalError::Invalid("hidden size exceeds u32".into()))
}

fn validate_eps(value: f32, name: &str) -> Result<(), DeepSeekV4MetalError> {
    if !value.is_finite() || value <= 0.0 {
        return invalid(format!("{name} must be finite and positive, got {value}"));
    }
    Ok(())
}

fn validate_ds4_rope(
    rope: DeepSeekV4RopeParameters,
    head_dim: usize,
    expected_rotary_dim: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if head_dim == 0
        || rope.rotary_dim != expected_rotary_dim
        || rope.rotary_dim == 0
        || rope.rotary_dim > head_dim
        || !rope.rotary_dim.is_multiple_of(2)
    {
        return invalid(format!(
            "invalid DS4 RoPE dimensions: head={head_dim} rotary={} expected={expected_rotary_dim}",
            rope.rotary_dim
        ));
    }
    if !rope.theta.is_finite() || rope.theta <= 1.0 {
        return invalid(format!("invalid DS4 RoPE theta {}", rope.theta));
    }
    if !rope.scaling_factor.is_finite() || rope.scaling_factor < 1.0 {
        return invalid(format!(
            "invalid DS4 RoPE scaling factor {}",
            rope.scaling_factor
        ));
    }
    if rope.scaling_factor > 1.0
        && (rope.original_context_length == 0
            || !rope.beta_fast.is_finite()
            || rope.beta_fast <= 0.0
            || !rope.beta_slow.is_finite()
            || rope.beta_slow <= 0.0)
    {
        return invalid("scaled DS4 RoPE requires an original context and positive YaRN betas");
    }
    Ok(())
}

fn require_serial(enc: &KernelEncoder, kernel: &str) -> Result<(), DeepSeekV4MetalError> {
    if enc.is_concurrent() {
        return invalid(format!("{kernel} requires ordered serial dispatches"));
    }
    Ok(())
}

#[cfg(feature = "dsv4-diagnostics")]
fn raw_i8_subview(
    tensor: &MetalTensor,
    element_offset: usize,
    shape: Vec<u64>,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    if tensor.dtype != GgmlType::I8 {
        return invalid(format!(
            "{name} requires raw I8 storage, got {:?}",
            tensor.dtype
        ));
    }
    let elements = crate::tensor::checked_shape_elements(&shape)
        .and_then(|elements| usize::try_from(elements).ok())
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} shape overflows usize")))?;
    let end = element_offset
        .checked_add(elements)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} range overflows usize")))?;
    if end > tensor.n_elements() as usize {
        return invalid(format!(
            "{name} range [{element_offset}, {end}) exceeds {} I8 elements",
            tensor.n_elements()
        ));
    }
    let offset = tensor
        .offset
        .checked_add(element_offset as u64)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("{name} byte offset overflows u64"))
        })?;
    let view = MetalTensor {
        buffer: tensor.buffer.clone(),
        offset,
        shape,
        dtype: GgmlType::I8,
        provenance: tensor.provenance,
    };
    validate_i8(&view, &view.shape, tensor.is_writable(), name)?;
    Ok(view)
}

fn validate_i8(
    tensor: &MetalTensor,
    shape: &[u64],
    writable: bool,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    if tensor.dtype != GgmlType::I8 || tensor.shape != shape {
        return invalid(format!(
            "{name} must be raw I8 bytes with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    if writable && !tensor.is_writable() {
        return invalid(format!("{name} must be writable"));
    }
    let end = tensor
        .offset
        .checked_add(tensor.n_bytes())
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range [{}, {end}) exceeds buffer length {}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

fn validate_f32(
    tensor: &MetalTensor,
    shape: &[u64],
    writable: bool,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    if tensor.dtype != GgmlType::F32 || tensor.shape != shape {
        return invalid(format!(
            "{name} must be F32 with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    if writable && !tensor.is_writable() {
        return invalid(format!("{name} must be writable"));
    }
    if !tensor
        .offset
        .is_multiple_of(std::mem::align_of::<f32>() as u64)
    {
        return invalid(format!(
            "{name} offset {} is not F32-aligned",
            tensor.offset
        ));
    }
    let end = tensor
        .offset
        .checked_add(tensor.n_bytes())
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range [{}, {end}) exceeds buffer length {}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

fn metal_tensor_ranges_overlap(left: &MetalTensor, right: &MetalTensor) -> bool {
    if Retained::as_ptr(&left.buffer) != Retained::as_ptr(&right.buffer) {
        return false;
    }
    let Some(left_end) = left.offset.checked_add(left.n_bytes()) else {
        return true;
    };
    let Some(right_end) = right.offset.checked_add(right.n_bytes()) else {
        return true;
    };
    left.offset < right_end && right.offset < left_end
}

fn validate_f16(
    tensor: &MetalTensor,
    shape: &[u64],
    writable: bool,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    if tensor.dtype != GgmlType::F16 || tensor.shape != shape {
        return invalid(format!(
            "{name} must be F16 with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    if writable && !tensor.is_writable() {
        return invalid(format!("{name} must be writable"));
    }
    if !tensor
        .offset
        .is_multiple_of(std::mem::align_of::<u16>() as u64)
    {
        return invalid(format!(
            "{name} offset {} is not F16-aligned",
            tensor.offset
        ));
    }
    let end = tensor
        .offset
        .checked_add(tensor.n_bytes())
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range [{}, {end}) exceeds buffer length {}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

fn validate_strict_binding(
    gguf: &GgufFile,
    model: &DeepSeekV4Model<'_>,
) -> Result<(), DeepSeekV4MetalError> {
    if gguf.tensors.len() != DEEPSEEK_V4_FLASH_0731_TENSOR_COUNT
        || model.source_tensor_count != DEEPSEEK_V4_FLASH_0731_TENSOR_COUNT
    {
        return invalid(format!(
            "expected {DEEPSEEK_V4_FLASH_0731_TENSOR_COUNT} tensors, GGUF has {} and binding declares {}",
            gguf.tensors.len(),
            model.source_tensor_count
        ));
    }

    Ok(())
}

fn validate_session_lookup_storage(gguf: &GgufFile) -> Result<(), DeepSeekV4MetalError> {
    let dtype = |name: &str| {
        gguf.tensors
            .iter()
            .find(|desc| desc.name == name)
            .map(|desc| desc.dtype)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "DeepSeek V4 session requires tensor {name:?}"
                ))
            })
    };
    validate_session_lookup_dtypes(dtype("token_embd.weight")?, dtype("output.weight")?)
}

fn validate_session_lookup_dtypes(
    embedding_dtype: GgmlType,
    output_dtype: GgmlType,
) -> Result<(), DeepSeekV4MetalError> {
    if !matches!(embedding_dtype, GgmlType::Q6_K | GgmlType::Q8_0) {
        return invalid(format!(
            "token_embd.weight must be Q6_K or Q8_0 for the native session get_rows path, got {embedding_dtype:?}"
        ));
    }
    if !matches!(output_dtype, GgmlType::Q6_K | GgmlType::Q8_0) {
        return invalid(format!(
            "output.weight must be Q6_K or Q8_0 for the native session logits path, got {output_dtype:?}"
        ));
    }
    Ok(())
}

fn validate_fallback_policy(plan: &RetainedStoragePlan) -> Result<(), DeepSeekV4MetalError> {
    for entry in &plan.entries {
        if let RetainedStorageDisposition::CopyFallback { reason } = entry.disposition
            && reason != RetainedStorageFallback::FinalPartialPage
        {
            return invalid(format!(
                "tensor {:?} requires disallowed {reason:?} copy fallback",
                entry.name
            ));
        }
    }
    Ok(())
}

fn validate_descriptor_fingerprints(
    tensors: &[crate::tensor::TensorDesc],
    fingerprints: &[DeepSeekV4DescriptorFingerprint],
) -> Result<(), DeepSeekV4MetalError> {
    if tensors.len() != fingerprints.len() {
        return invalid("DeepSeek V4 descriptor fingerprint count changed before realization");
    }
    for (index, (desc, fingerprint)) in tensors.iter().zip(fingerprints).enumerate() {
        if fingerprint.name != desc.name
            || fingerprint.shape != desc.shape
            || fingerprint.dtype != desc.dtype
            || fingerprint.shard_idx != desc.shard_idx
            || fingerprint.data_offset != desc.data_offset
            || fingerprint.n_bytes != desc.n_bytes
        {
            return invalid(format!(
                "DeepSeek V4 descriptor fingerprint changed at index {index}"
            ));
        }
    }
    Ok(())
}

fn validate_retained_plan_against_gguf(
    ctx: &MetalContext,
    gguf: &GgufFile,
    plan: &RetainedStoragePlan,
) -> Result<(), DeepSeekV4MetalError> {
    validate_retained_plan_geometry(
        host_page_size_bytes()?,
        ctx.max_buffer_length(),
        &gguf.shard_mapped_lengths(),
        &gguf.tensors,
        plan,
    )
}

fn validate_retained_plan_geometry(
    page_size: usize,
    max_buffer_length: usize,
    shard_lengths: &[usize],
    tensors: &[crate::tensor::TensorDesc],
    plan: &RetainedStoragePlan,
) -> Result<(), DeepSeekV4MetalError> {
    if page_size == 0 {
        return invalid("DeepSeek V4 retained-plan page size is zero");
    }
    let usable_window_length = max_buffer_length / page_size * page_size;
    if usable_window_length == 0
        || plan.page_size != page_size
        || plan.max_buffer_length != max_buffer_length
        || plan.usable_window_length != usable_window_length
        || plan.required_alignment != GGUF_BINDING_ALIGNMENT
        || plan.entries.len() != tensors.len()
    {
        return invalid("DeepSeek V4 retained-plan geometry changed before realization");
    }
    for (index, window) in plan.windows.iter().enumerate() {
        let shard_length = shard_lengths
            .get(window.shard_idx)
            .copied()
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "retained window {index} references missing shard {}",
                    window.shard_idx
                ))
            })?;
        let mmap_offset = usize::try_from(window.mmap_offset).map_err(|_| {
            DeepSeekV4MetalError::Invalid(format!("retained window {index} offset exceeds usize"))
        })?;
        let end = mmap_offset.checked_add(window.length).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("retained window {index} endpoint overflow"))
        })?;
        if window.length == 0
            || window.length > plan.usable_window_length
            || !mmap_offset.is_multiple_of(page_size)
            || !window.length.is_multiple_of(page_size)
            || end > shard_length
        {
            return invalid(format!(
                "retained window {index} is outside the planned shard/page/buffer geometry"
            ));
        }
    }
    for (index, (entry, desc)) in plan.entries.iter().zip(tensors).enumerate() {
        if entry.request_index != index
            || entry.name != desc.name
            || entry.shard_idx != desc.shard_idx
            || entry.data_offset != desc.data_offset
            || entry.n_bytes != desc.n_bytes
        {
            return invalid(format!("planner descriptor drift at index {index}"));
        }
        match entry.disposition {
            RetainedStorageDisposition::View {
                window_index,
                buffer_offset,
            } => {
                let window = plan.windows.get(window_index).ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "retained entry {index} references missing window {window_index}"
                    ))
                })?;
                let expected_data_offset = window
                    .mmap_offset
                    .checked_add(buffer_offset)
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "retained entry {index} source offset overflow"
                        ))
                    })?;
                let buffer_end = buffer_offset.checked_add(entry.n_bytes).ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "retained entry {index} buffer endpoint overflow"
                    ))
                })?;
                if entry.shard_idx != window.shard_idx
                    || entry.data_offset != expected_data_offset
                    || buffer_end > window.length as u64
                    || !buffer_offset.is_multiple_of(plan.required_alignment as u64)
                {
                    return invalid(format!(
                        "retained entry {index} differs from its planned window binding"
                    ));
                }
            }
            RetainedStorageDisposition::Alias {
                source_request_index,
            } => {
                if source_request_index >= index {
                    return invalid(format!(
                        "retained alias {index} has non-prior source {source_request_index}"
                    ));
                }
                let source_entry = &plan.entries[source_request_index];
                let source_desc = &tensors[source_request_index];
                if source_entry.shard_idx != entry.shard_idx
                    || source_entry.data_offset != entry.data_offset
                    || source_entry.n_bytes != entry.n_bytes
                    || source_desc.shape != desc.shape
                    || source_desc.dtype != desc.dtype
                    || matches!(
                        source_entry.disposition,
                        RetainedStorageDisposition::Alias { .. }
                    )
                {
                    return invalid(format!(
                        "retained alias {index} differs from source {source_request_index}"
                    ));
                }
            }
            RetainedStorageDisposition::CopyFallback { reason } => {
                let shard_length =
                    shard_lengths.get(entry.shard_idx).copied().ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "retained fallback {index} references missing shard {}",
                            entry.shard_idx
                        ))
                    })?;
                let start = usize::try_from(entry.data_offset).map_err(|_| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "retained fallback {index} offset exceeds usize"
                    ))
                })?;
                let length = usize::try_from(entry.n_bytes).map_err(|_| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "retained fallback {index} length exceeds usize"
                    ))
                })?;
                let end = start.checked_add(length).ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "retained fallback {index} endpoint overflow"
                    ))
                })?;
                let rounded_end = end
                    .checked_add(page_size - 1)
                    .map(|value| value / page_size * page_size)
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "retained fallback {index} page endpoint overflow"
                        ))
                    })?;
                let window_start = start / page_size * page_size;
                let candidate_window_length =
                    rounded_end.checked_sub(window_start).ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "retained fallback {index} window range underflow"
                        ))
                    })?;
                let full_page_end = shard_length / page_size * page_size;
                if reason != RetainedStorageFallback::FinalPartialPage
                    || length == 0
                    || start % plan.required_alignment != 0
                    || end > shard_length
                    || end <= full_page_end
                    || candidate_window_length > plan.usable_window_length
                {
                    return invalid(format!(
                        "retained entry {index} differs from a final-partial-page fallback"
                    ));
                }
            }
        }
    }
    let requests = tensors.iter().collect::<Vec<_>>();
    let rebuilt = plan_retained_storage(
        shard_lengths,
        &requests,
        page_size,
        max_buffer_length,
        GGUF_BINDING_ALIGNMENT,
    )?;
    if rebuilt != *plan {
        return invalid("DeepSeek V4 retained plan differs from deterministic planner output");
    }
    Ok(())
}

fn realize_windows(
    ctx: &MetalContext,
    gguf: &GgufFile,
    plan: &RetainedStoragePlan,
) -> Result<Vec<MetalGgufBacking>, DeepSeekV4MetalError> {
    let mut windows = Vec::with_capacity(plan.windows.len());
    for window in &plan.windows {
        let mmap = gguf.retained_shard_mmap(window.shard_idx).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "planned window references missing shard {}",
                window.shard_idx
            ))
        })?;
        let mmap_offset = usize::try_from(window.mmap_offset)
            .map_err(|_| DeepSeekV4MetalError::Invalid("window offset exceeds usize".into()))?;
        let backing = ctx.gguf_no_copy_window(
            mmap,
            window.shard_idx,
            mmap_offset,
            window.length,
            GGUF_BINDING_ALIGNMENT,
        )?;
        if backing.mmap_offset() != mmap_offset
            || backing.exposed_len() != window.length
            || backing.required_alignment() != GGUF_BINDING_ALIGNMENT
        {
            return invalid(format!(
                "window realization drift at shard {} offset {}",
                window.shard_idx, window.mmap_offset
            ));
        }
        windows.push(backing);
    }
    Ok(windows)
}

fn realize_tensors(
    ctx: &MetalContext,
    gguf: &GgufFile,
    plan: &RetainedStoragePlan,
    windows: &[MetalGgufBacking],
) -> Result<BTreeMap<String, MetalTensor>, DeepSeekV4MetalError> {
    if plan.entries.len() != gguf.tensors.len() {
        return invalid("planner entry count differs from GGUF tensor count");
    }
    let mut realized: Vec<Option<MetalTensor>> = vec![None; plan.entries.len()];
    let mut tensors = BTreeMap::new();
    for (index, (entry, desc)) in plan.entries.iter().zip(&gguf.tensors).enumerate() {
        if entry.request_index != index
            || entry.name != desc.name
            || entry.shard_idx != desc.shard_idx
            || entry.data_offset != desc.data_offset
            || entry.n_bytes != desc.n_bytes
        {
            return invalid(format!("planner descriptor drift at index {index}"));
        }
        let tensor = match entry.disposition {
            RetainedStorageDisposition::View {
                window_index,
                buffer_offset,
            } => {
                let backing = windows.get(window_index).ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "tensor {:?} references missing window {window_index}",
                        desc.name
                    ))
                })?;
                let (eligibility, tensor) = backing.tensor(desc)?;
                let tensor = tensor.ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "tensor {:?} failed retained realization: {eligibility:?}",
                        desc.name
                    ))
                })?;
                if tensor.offset != buffer_offset
                    || tensor.provenance() != MetalTensorProvenance::RetainedGgufReadOnly
                {
                    return invalid(format!("retained realization drift for {:?}", desc.name));
                }
                tensor
            }
            RetainedStorageDisposition::Alias {
                source_request_index,
            } => realized
                .get(source_request_index)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "alias {:?} has unrealized source {source_request_index}",
                        desc.name
                    ))
                })?
                .clone(),
            RetainedStorageDisposition::CopyFallback { reason } => {
                if reason != RetainedStorageFallback::FinalPartialPage {
                    return invalid(format!(
                        "disallowed fallback reached realization: {reason:?}"
                    ));
                }
                MetalTensor::copied_gguf_weight(ctx, desc, gguf.try_slice(desc)?)?
            }
        };
        validate_tensor(desc, &tensor)?;
        if tensors.insert(desc.name.clone(), tensor.clone()).is_some() {
            return invalid(format!("duplicate realization for {:?}", desc.name));
        }
        realized[index] = Some(tensor);
    }
    if realized.iter().any(Option::is_none) {
        return invalid("one or more planned tensors were not realized");
    }
    Ok(tensors)
}

fn validate_tensor(
    desc: &crate::tensor::TensorDesc,
    tensor: &MetalTensor,
) -> Result<(), DeepSeekV4MetalError> {
    if tensor.dtype != desc.dtype
        || tensor.shape != desc.shape
        || tensor.n_bytes() != desc.n_bytes
        || tensor.is_writable()
    {
        return invalid(format!(
            "realized tensor {:?} does not exactly preserve dtype, shape, bytes, and read-only storage",
            desc.name
        ));
    }
    let end = tensor
        .offset
        .checked_add(tensor.n_bytes())
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("tensor buffer endpoint overflow".into()))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "realized tensor {:?} exceeds its Metal buffer",
            desc.name
        ));
    }
    Ok(())
}

fn validate_realization(
    gguf: &GgufFile,
    tensors: &BTreeMap<String, MetalTensor>,
    report: &DeepSeekV4ResidencyReport,
) -> Result<(), DeepSeekV4MetalError> {
    if tensors.len() != DEEPSEEK_V4_FLASH_0731_TENSOR_COUNT
        || tensors.len() != gguf.tensors.len()
        || report.tensor_count != tensors.len()
    {
        return invalid("final tensor count mismatch");
    }
    for desc in &gguf.tensors {
        let tensor = tensors.get(&desc.name).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("missing realization for {:?}", desc.name))
        })?;
        validate_tensor(desc, tensor)?;
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DeepSeekV4SessionAllocationRequest {
    name: String,
    logical_bytes: u64,
}

fn push_session_allocation(
    requests: &mut Vec<DeepSeekV4SessionAllocationRequest>,
    name: impl Into<String>,
    elements: usize,
    element_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let logical_bytes = checked_mul(elements, element_bytes, "session allocation bytes")?;
    requests.push(DeepSeekV4SessionAllocationRequest {
        name: name.into(),
        logical_bytes: u64::try_from(logical_bytes).map_err(|_| {
            DeepSeekV4MetalError::Invalid("session allocation bytes exceed u64".into())
        })?,
    });
    Ok(())
}

fn append_compressor_frontier_allocations(
    requests: &mut Vec<DeepSeekV4SessionAllocationRequest>,
    prefix: &str,
    ratio: usize,
    head_dim: usize,
    publication: DeepSeekV4CompressorPublication,
    capacity_rows: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if publication == DeepSeekV4CompressorPublication::IndexerHadamard && head_dim != 128 {
        return invalid("indexer publication requires exactly 128 dimensions");
    }
    let (width, _, state_elements) = compressor_frontier_geometry(ratio, head_dim)?;
    for suffix in ["kv_state", "score_state"] {
        push_session_allocation(
            requests,
            format!("{prefix}.{suffix}"),
            state_elements,
            std::mem::size_of::<f32>(),
        )?;
    }
    for suffix in ["projected_kv", "projected_score"] {
        push_session_allocation(
            requests,
            format!("{prefix}.{suffix}"),
            width,
            std::mem::size_of::<f32>(),
        )?;
    }
    for suffix in ["pooled", "normalized"] {
        push_session_allocation(
            requests,
            format!("{prefix}.{suffix}"),
            head_dim,
            std::mem::size_of::<f32>(),
        )?;
    }
    push_session_allocation(
        requests,
        format!("{prefix}.published"),
        checked_mul(
            head_dim,
            capacity_rows,
            "published compressor history elements",
        )?,
        std::mem::size_of::<u16>(),
    )?;
    #[cfg(feature = "dsv4-diagnostics")]
    if publication == DeepSeekV4CompressorPublication::IndexerHadamard {
        push_session_allocation(
            requests,
            format!("{prefix}.fp4_values"),
            checked_mul(
                crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES,
                capacity_rows,
                "indexer FP4 sidecar value bytes",
            )?,
            std::mem::size_of::<u8>(),
        )?;
        push_session_allocation(
            requests,
            format!("{prefix}.fp4_scales"),
            checked_mul(
                crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES,
                capacity_rows,
                "indexer FP4 sidecar scale bytes",
            )?,
            std::mem::size_of::<u8>(),
        )?;
        push_session_allocation(
            requests,
            format!("{prefix}.fp4_status"),
            capacity_rows,
            std::mem::size_of::<i32>(),
        )?;
    }
    Ok(())
}

fn deepseek_v4_session_allocation_requests(
    config: &DeepSeekV4Config,
    capacity: DeepSeekV4SessionCapacity,
) -> Result<Vec<DeepSeekV4SessionAllocationRequest>, DeepSeekV4MetalError> {
    validate_session_config(config)?;
    deepseek_v4_session_allocation_requests_for_kinds(
        &config.attention_kinds,
        config.expert_count as usize,
        capacity,
    )
}

fn deepseek_v4_session_allocation_requests_for_kinds(
    attention_kinds: &[AttentionKind],
    expert_count: usize,
    capacity: DeepSeekV4SessionCapacity,
) -> Result<Vec<DeepSeekV4SessionAllocationRequest>, DeepSeekV4MetalError> {
    let sliding = attention_kinds
        .iter()
        .filter(|&&kind| kind == AttentionKind::SlidingWindow)
        .count();
    let csa = attention_kinds
        .iter()
        .filter(|&&kind| kind == AttentionKind::CompressedSparse)
        .count();
    let hca = attention_kinds
        .iter()
        .filter(|&&kind| kind == AttentionKind::HeavilyCompressed)
        .count();
    if attention_kinds.len() != DEEPSEEK_V4_LAYER_COUNT || (sliding, csa, hca) != (2, 21, 20) {
        return invalid(format!(
            "session allocation inventory requires 43 layers split 2/21/20, got {}/{sliding}/{csa}/{hca}",
            attention_kinds.len()
        ));
    }
    let attention = deepseek_v4_session_attention_config();
    let attention_dims = attention.checked()?;
    let moe = DeepSeekV4MoeConfig {
        hidden_size: DEEPSEEK_V4_HIDDEN_SIZE,
        ffn_size: 2_048,
        expert_count,
        top_k: 6,
        routed_scale: 1.0,
    };
    moe.checked()?;
    let mut requests = Vec::with_capacity(521);
    let f32_bytes = std::mem::size_of::<f32>();
    let i32_bytes = std::mem::size_of::<i32>();
    let f16_bytes = std::mem::size_of::<u16>();
    let residual_elements = residual_len(DEEPSEEK_V4_HIDDEN_SIZE)?;

    push_session_allocation(&mut requests, "token_id", 1, i32_bytes)?;
    push_session_allocation(
        &mut requests,
        "embedding",
        DEEPSEEK_V4_HIDDEN_SIZE,
        f32_bytes,
    )?;
    for name in ["residual_primary", "residual_secondary"] {
        push_session_allocation(&mut requests, name, residual_elements, f32_bytes)?;
    }

    for name in ["hyper.ones", "hyper.normalized"] {
        push_session_allocation(&mut requests, name, residual_elements, f32_bytes)?;
    }
    push_session_allocation(
        &mut requests,
        "hyper.mixes",
        DEEPSEEK_V4_HC_PARAMETER_COUNT,
        f32_bytes,
    )?;
    for name in ["hyper.pre", "hyper.post"] {
        push_session_allocation(&mut requests, name, DEEPSEEK_V4_CONNECTION_COUNT, f32_bytes)?;
    }
    push_session_allocation(
        &mut requests,
        "hyper.combination",
        checked_mul(
            DEEPSEEK_V4_CONNECTION_COUNT,
            DEEPSEEK_V4_CONNECTION_COUNT,
            "hyper combination elements",
        )?,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "hyper.collapsed",
        DEEPSEEK_V4_HIDDEN_SIZE,
        f32_bytes,
    )?;
    for name in ["hyper.head_mixes", "hyper.head_gates"] {
        push_session_allocation(&mut requests, name, DEEPSEEK_V4_CONNECTION_COUNT, f32_bytes)?;
    }

    push_session_allocation(
        &mut requests,
        "attention.normalized_input",
        attention.hidden_size,
        f32_bytes,
    )?;
    for name in ["attention.q_lora_raw", "attention.q_lora"] {
        push_session_allocation(&mut requests, name, attention.q_lora_rank, f32_bytes)?;
    }
    for name in ["attention.queries_raw", "attention.queries"] {
        push_session_allocation(&mut requests, name, attention_dims.query_width, f32_bytes)?;
    }
    for name in ["attention.kv_raw", "attention.kv", "attention.cached_kv"] {
        push_session_allocation(&mut requests, name, attention.head_dim, f32_bytes)?;
    }
    push_session_allocation(
        &mut requests,
        "attention.heads",
        attention_dims.query_width,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "attention.hca_partial_output",
        checked_mul(
            attention_dims.query_width,
            DEEPSEEK_V4_SPLITK_HCA_PARTITIONS,
            "split-K HCA partial output elements",
        )?,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "attention.hca_partial_ml",
        checked_mul(
            checked_mul(
                2,
                attention.head_count,
                "split-K HCA max/mass per partition",
            )?,
            DEEPSEEK_V4_SPLITK_HCA_PARTITIONS,
            "split-K HCA partial max/mass elements",
        )?,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "attention.low_rank",
        attention_dims.low_rank_width,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "attention.output",
        attention.hidden_size,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "attention.head_norm_ones",
        attention.head_dim,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "sparse_csa.index_queries",
        64 * 128,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "sparse_csa.matrix_queries_f16",
        64 * 128,
        f16_bytes,
    )?;
    push_session_allocation(&mut requests, "sparse_csa.head_weights", 64, f32_bytes)?;
    push_session_allocation(&mut requests, "sparse_csa.visible_counts", 1, i32_bytes)?;
    push_session_allocation(
        &mut requests,
        "sparse_csa.scores",
        capacity.csa_physical_rows(),
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "sparse_csa.selected_mask",
        capacity.csa_physical_rows(),
        i32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "sparse_csa.cache_order_ids",
        DEEPSEEK_V4_CSA_TOP_K,
        i32_bytes,
    )?;
    for name in ["sparse_csa.selected_counts", "sparse_csa.status"] {
        push_session_allocation(&mut requests, name, 1, i32_bytes)?;
    }
    if deepseek_v4_multigroup_selector_capacity_supported(capacity.csa_physical_rows()) {
        push_session_allocation(
            &mut requests,
            "sparse_csa.multigroup.records",
            checked_mul(
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS,
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS,
                "multi-group selector record elements",
            )?,
            i32_bytes,
        )?;
        push_session_allocation(
            &mut requests,
            "sparse_csa.multigroup.partition_plan",
            checked_mul(
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS,
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS,
                "multi-group selector plan elements",
            )?,
            i32_bytes,
        )?;
        push_session_allocation(
            &mut requests,
            "sparse_csa.multigroup.state",
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS,
            i32_bytes,
        )?;
        push_session_allocation(
            &mut requests,
            "sparse_csa.multigroup.private_mask",
            capacity.csa_physical_rows(),
            std::mem::size_of::<u8>(),
        )?;
        push_session_allocation(
            &mut requests,
            "sparse_csa.multigroup.private_ids",
            DEEPSEEK_V4_CSA_TOP_K,
            i32_bytes,
        )?;
    }
    #[cfg(feature = "dsv4-diagnostics")]
    {
        push_session_allocation(
            &mut requests,
            "fp4_shadow.query_values",
            crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES * 64,
            std::mem::size_of::<u8>(),
        )?;
        push_session_allocation(
            &mut requests,
            "fp4_shadow.query_scales",
            crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES * 64,
            std::mem::size_of::<u8>(),
        )?;
        push_session_allocation(&mut requests, "fp4_shadow.query_status", 64, i32_bytes)?;
        push_session_allocation(&mut requests, "fp4_shadow.query_units", 128 * 64, f16_bytes)?;
        push_session_allocation(&mut requests, "fp4_shadow.eligible_visible", 1, i32_bytes)?;
        push_session_allocation(&mut requests, "fp4_shadow.eligibility_record", 3, i32_bytes)?;
        push_session_allocation(
            &mut requests,
            "fp4_shadow.scores",
            capacity.csa_physical_rows(),
            f32_bytes,
        )?;
        push_session_allocation(
            &mut requests,
            "fp4_shadow.selected_mask",
            capacity.csa_physical_rows(),
            i32_bytes,
        )?;
        push_session_allocation(
            &mut requests,
            "fp4_shadow.cache_order_ids",
            DEEPSEEK_V4_CSA_TOP_K,
            i32_bytes,
        )?;
        for name in ["fp4_shadow.selected_count", "fp4_shadow.status"] {
            push_session_allocation(&mut requests, name, 1, i32_bytes)?;
        }
        push_session_allocation(
            &mut requests,
            "fp4_collapsed_selections.cache_order_ids",
            checked_mul(
                DEEPSEEK_V4_CSA_TOP_K,
                DEEPSEEK_V4_LAYER_COUNT,
                "collapsed FP4 layer-selection ID elements",
            )?,
            i32_bytes,
        )?;
        push_session_allocation(
            &mut requests,
            "fp4_collapsed_selections.integers",
            checked_mul(
                DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH,
                DEEPSEEK_V4_LAYER_COUNT,
                "collapsed FP4 completion-record elements",
            )?,
            i32_bytes,
        )?;
    }
    push_session_allocation(
        &mut requests,
        "layer_selections.integers",
        checked_mul(
            DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH,
            DEEPSEEK_V4_LAYER_COUNT,
            "layer-selection record elements",
        )?,
        i32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "raw_cache",
        checked_mul(
            checked_mul(
                attention.head_dim,
                DEEPSEEK_V4_LOCAL_WINDOW,
                "raw-cache layer elements",
            )?,
            DEEPSEEK_V4_LAYER_COUNT,
            "raw-cache session elements",
        )?,
        f16_bytes,
    )?;

    let attention_dim = 512;
    let indexer_dim = 128;
    for (layer, kind) in attention_kinds.iter().copied().enumerate() {
        match kind {
            AttentionKind::SlidingWindow => {}
            AttentionKind::CompressedSparse => {
                append_compressor_frontier_allocations(
                    &mut requests,
                    &format!("compressor.{layer}.attention"),
                    4,
                    attention_dim,
                    DeepSeekV4CompressorPublication::Attention,
                    capacity.csa_physical_rows(),
                )?;
                append_compressor_frontier_allocations(
                    &mut requests,
                    &format!("compressor.{layer}.indexer"),
                    4,
                    indexer_dim,
                    DeepSeekV4CompressorPublication::IndexerHadamard,
                    capacity.csa_physical_rows(),
                )?;
            }
            AttentionKind::HeavilyCompressed => append_compressor_frontier_allocations(
                &mut requests,
                &format!("compressor.{layer}.attention"),
                128,
                attention_dim,
                DeepSeekV4CompressorPublication::Attention,
                capacity.hca_physical_rows(),
            )?,
        }
    }

    push_session_allocation(
        &mut requests,
        "moe.normalized_input",
        moe.hidden_size,
        f32_bytes,
    )?;
    push_session_allocation(&mut requests, "moe.logits", moe.expert_count, f32_bytes)?;
    push_session_allocation(&mut requests, "moe.expert_ids", moe.top_k, i32_bytes)?;
    push_session_allocation(&mut requests, "moe.weights", moe.top_k, f32_bytes)?;
    push_session_allocation(&mut requests, "moe.route_status", 1, i32_bytes)?;
    push_session_allocation(
        &mut requests,
        "layer_routes.integers",
        checked_mul(
            DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH,
            DEEPSEEK_V4_LAYER_COUNT,
            "layer-route integer record elements",
        )?,
        i32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "layer_routes.weights",
        checked_mul(
            moe.top_k,
            DEEPSEEK_V4_LAYER_COUNT,
            "layer-route weight record elements",
        )?,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "moe.gate",
        checked_mul(moe.ffn_size, 3, "MoE gate and fused Q6 scratch elements")?,
        f32_bytes,
    )?;
    for name in ["moe.up", "moe.inner"] {
        push_session_allocation(&mut requests, name, moe.ffn_size, f32_bytes)?;
    }
    push_session_allocation(
        &mut requests,
        "moe.routed_inner",
        checked_mul(moe.ffn_size, moe.top_k, "MoE all-slot inner elements")?,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "moe.expert_outputs",
        checked_mul(moe.hidden_size, moe.top_k, "MoE expert-output elements")?,
        f32_bytes,
    )?;
    for name in ["moe.routed_output", "moe.shared_output", "moe.final_output"] {
        push_session_allocation(&mut requests, name, moe.hidden_size, f32_bytes)?;
    }

    for name in ["final_hidden", "final_normalized_hidden"] {
        push_session_allocation(&mut requests, name, DEEPSEEK_V4_HIDDEN_SIZE, f32_bytes)?;
    }
    push_session_allocation(&mut requests, "logits", DEEPSEEK_V4_VOCAB_SIZE, f32_bytes)?;
    prefill::append_session_allocation_requests(&mut requests, capacity.csa_physical_rows())?;
    Ok(requests)
}

fn price_shared_buffer(
    ctx: &MetalContext,
    logical_bytes: u64,
    name: &str,
) -> Result<(u64, u64), DeepSeekV4MetalError> {
    if logical_bytes == 0 {
        return invalid(format!("planned Metal buffer {name:?} has zero bytes"));
    }
    let max_buffer_length = u64::try_from(ctx.max_buffer_length()).map_err(|_| {
        DeepSeekV4MetalError::Invalid("Metal maximum buffer length exceeds u64".into())
    })?;
    if logical_bytes > max_buffer_length {
        return invalid(format!(
            "planned Metal buffer {name:?} requires {logical_bytes} bytes, beyond device maximum {max_buffer_length}"
        ));
    }
    let priced = ctx.shared_buffer_size_and_align(logical_bytes)?;
    if priced.size < logical_bytes || priced.alignment == 0 || !priced.alignment.is_power_of_two() {
        return invalid(format!(
            "invalid Metal pricing for {name:?}: logical={logical_bytes} priced={} alignment={}",
            priced.size, priced.alignment
        ));
    }
    let allocation_alignment = priced.alignment.max(host_page_size_bytes()? as u64);
    let priced_bytes = priced
        .size
        .checked_add(allocation_alignment - 1)
        .map(|bytes| bytes / allocation_alignment * allocation_alignment)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "aligned Metal pricing for {name:?} overflows u64"
            ))
        })?;
    Ok((priced_bytes, allocation_alignment))
}

fn build_memory_plan(
    ctx: &MetalContext,
    retained: &RetainedStoragePlan,
    report: &DeepSeekV4ResidencyReport,
    config: &DeepSeekV4Config,
    capacity: DeepSeekV4SessionCapacity,
) -> Result<DeepSeekV4MemoryPlan, DeepSeekV4MetalError> {
    let mut residency_priced_upper_bytes = 0_u64;
    let mut residency_buffer_count = 0_usize;
    let mut residency_logical_bytes = 0_u64;
    for (index, window) in retained.windows.iter().enumerate() {
        let logical = u64::try_from(window.length).map_err(|_| {
            DeepSeekV4MetalError::Invalid("retained window length exceeds u64".into())
        })?;
        let (priced, _) = price_shared_buffer(ctx, logical, &format!("weight_window[{index}]"))?;
        residency_priced_upper_bytes = residency_priced_upper_bytes
            .checked_add(priced)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("priced residency byte count overflow".into())
            })?;
        residency_logical_bytes =
            residency_logical_bytes
                .checked_add(logical)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("logical residency byte count overflow".into())
                })?;
        residency_buffer_count += 1;
    }
    for entry in &retained.entries {
        if matches!(
            entry.disposition,
            RetainedStorageDisposition::CopyFallback { .. }
        ) {
            let (priced, _) = price_shared_buffer(ctx, entry.n_bytes, &entry.name)?;
            residency_priced_upper_bytes = residency_priced_upper_bytes
                .checked_add(priced)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("priced fallback byte count overflow".into())
                })?;
            residency_logical_bytes = residency_logical_bytes
                .checked_add(entry.n_bytes)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("logical fallback byte count overflow".into())
                })?;
            residency_buffer_count += 1;
        }
    }
    if residency_logical_bytes != report.resident_bytes {
        return invalid(format!(
            "memory-plan residency bytes {residency_logical_bytes} differ from report {}",
            report.resident_bytes
        ));
    }

    let requests = deepseek_v4_session_allocation_requests(config, capacity)?;
    let mut session_allocations = Vec::with_capacity(requests.len());
    let mut session_logical_bytes = 0_u64;
    let mut session_priced_upper_bytes = 0_u64;
    for request in requests {
        let (priced_bytes, alignment) =
            price_shared_buffer(ctx, request.logical_bytes, &request.name)?;
        session_logical_bytes = session_logical_bytes
            .checked_add(request.logical_bytes)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("session logical byte count overflow".into())
            })?;
        session_priced_upper_bytes = session_priced_upper_bytes
            .checked_add(priced_bytes)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("session priced byte count overflow".into())
            })?;
        session_allocations.push(DeepSeekV4SessionAllocation {
            name: request.name,
            logical_bytes: request.logical_bytes,
            priced_bytes,
            alignment,
        });
    }
    let total_priced_upper_bytes = residency_priced_upper_bytes
        .checked_add(session_priced_upper_bytes)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("total priced Metal byte count overflow".into())
        })?;
    total_priced_upper_bytes
        .checked_add(DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "total priced Metal bytes plus dynamic reserve overflow".into(),
            )
        })?;
    Ok(DeepSeekV4MemoryPlan {
        residency_buffer_count,
        residency_logical_bytes,
        residency_priced_upper_bytes,
        session_logical_bytes,
        session_priced_upper_bytes,
        total_priced_upper_bytes,
        session_allocations,
    })
}

fn report_for_plan(
    plan: &RetainedStoragePlan,
) -> Result<DeepSeekV4ResidencyReport, DeepSeekV4MetalError> {
    let source_bytes = plan.entries.iter().try_fold(0_u64, |total, entry| {
        total
            .checked_add(entry.n_bytes)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("source byte count overflow".into()))
    })?;
    let window_bytes = plan.windows.iter().try_fold(0_u64, |total, window| {
        total
            .checked_add(window.length as u64)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("window byte count overflow".into()))
    })?;
    let resident_bytes = window_bytes
        .checked_add(plan.unique_fallback_bytes)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("resident byte count overflow".into()))?;
    let view_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
        .count();
    let alias_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::Alias { .. }))
        .count();
    let fallback_count = plan
        .entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.disposition,
                RetainedStorageDisposition::CopyFallback { .. }
            )
        })
        .count();
    Ok(DeepSeekV4ResidencyReport {
        tensor_count: plan.entries.len(),
        source_bytes,
        window_count: plan.windows.len(),
        window_bytes,
        view_count,
        unique_view_bytes: plan.unique_view_bytes,
        logical_view_bytes: plan.logical_view_bytes,
        alias_count,
        alias_bytes: plan.alias_bytes,
        fallback_count,
        fallback_bytes: plan.unique_fallback_bytes,
        resident_bytes,
        page_size: plan.page_size,
        max_buffer_length: plan.max_buffer_length,
        required_alignment: plan.required_alignment,
    })
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, DeepSeekV4MetalError> {
    Err(DeepSeekV4MetalError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests;
