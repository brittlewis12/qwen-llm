//! Native Metal residency and correctness-first execution for the strict
//! DeepSeek V4 Flash-0731 binding.
//!
//! GGUF weights remain in their exact storage without dtype conversion. The
//! execution bodies deliberately have no dependency on the Qwen Metal model.

use crate::deepseek_v4::{AttentionKind, DeepSeekV4Config, DeepSeekV4Error, DeepSeekV4Model};
use crate::gguf::{GgufError, GgufFile};
use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalGgufBacking, MetalMemoryAdmission,
    MetalMemorySignals, MetalTensor, MetalTensorProvenance, RetainedStorageDisposition,
    RetainedStorageFallback, RetainedStoragePlan, encode_get_rows_f32, encode_rms_norm_batched_f32,
    encode_rms_norm_mul_f32, encode_scatter_offset_f32_to_f16, evaluate_metal_memory_admission,
    host_page_size_bytes, plan_retained_storage,
};
use crate::tensor::{GgmlType, ggml_type_layout};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLSize};
use std::collections::BTreeMap;
use std::fmt;

pub const DEEPSEEK_V4_FLASH_0731_TENSOR_COUNT: usize = 1_328;
const GGUF_BINDING_ALIGNMENT: usize = 32;
pub const DEEPSEEK_V4_CONNECTION_COUNT: usize = 4;
pub const DEEPSEEK_V4_HC_PARAMETER_COUNT: usize = 24;
pub const DEEPSEEK_V4_SINKHORN_ITERATIONS: usize = 20;
pub const DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
/// Number of token forwards traversed by the longest retained-session
/// differential, through the position-512 fourth-HCA continuation. Callers use
/// this to reject requests before streaming beyond the current evidence
/// boundary.
pub const DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY: usize = 513;

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
        evaluate_metal_memory_admission(
            self.total_priced_upper_bytes,
            DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES,
            signals,
            true,
        )
    }

    pub fn required_with_reserve_bytes(&self) -> Result<u64, DeepSeekV4MetalError> {
        self.total_priced_upper_bytes
            .checked_add(DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "DeepSeek V4 priced memory plus dynamic reserve overflow".into(),
                )
            })
    }

    fn observed_delta(
        before_residency_bytes: u64,
        observed_bytes: u64,
        phase: &str,
    ) -> Result<u64, DeepSeekV4MetalError> {
        observed_bytes
            .checked_sub(before_residency_bytes)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "Metal allocation counter regressed during DeepSeek V4 {phase}"
                ))
            })
    }

    pub fn reconcile_residency(
        &self,
        before_residency_bytes: u64,
        after_residency_bytes: u64,
    ) -> Result<u64, DeepSeekV4MetalError> {
        let observed =
            Self::observed_delta(before_residency_bytes, after_residency_bytes, "residency")?;
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
        let observed_total =
            Self::observed_delta(before_residency_bytes, after_session_bytes, "session")?;
        let observed_session = after_session_bytes
            .checked_sub(after_residency_bytes)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "Metal allocation counter regressed during DeepSeek V4 session construction"
                        .into(),
                )
            })?;
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
        let observed = Self::observed_delta(before_residency_bytes, observed_bytes, phase)?;
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
        let sampled_peak_delta_bytes = sampled_peak_bytes
            .checked_sub(samples.before_residency_bytes)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "DeepSeek V4 sampled Metal peak precedes the allocation baseline".into(),
                )
            })?;
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

    pub fn admit(
        self,
        signals: MetalMemorySignals,
    ) -> Result<DeepSeekV4AdmittedLoadPlan, DeepSeekV4MetalError> {
        let admission = self.memory.admission(signals);
        if !admission.admitted {
            return invalid(format!(
                "DeepSeek V4 memory admission denied before Metal residency: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                admission.reason.as_str(),
                admission.required_bytes,
                admission.working_set_headroom_bytes,
                admission.signals.process_limit_remaining_bytes,
            ));
        }
        Ok(DeepSeekV4AdmittedLoadPlan {
            plan: self,
            admission,
        })
    }
}

pub struct DeepSeekV4AdmittedLoadPlan {
    plan: DeepSeekV4MetalLoadPlan,
    admission: MetalMemoryAdmission,
}

impl DeepSeekV4AdmittedLoadPlan {
    pub fn admission(&self) -> MetalMemoryAdmission {
        self.admission
    }

    pub fn memory_plan(&self) -> &DeepSeekV4MemoryPlan {
        &self.plan.memory
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
    tensors: BTreeMap<String, MetalTensor>,
    report: DeepSeekV4ResidencyReport,
    device_registry_id: u64,
}

impl DeepSeekV4MetalResidency {
    pub fn plan(
        ctx: &MetalContext,
        gguf: &GgufFile,
    ) -> Result<DeepSeekV4MetalLoadPlan, DeepSeekV4MetalError> {
        let model = DeepSeekV4Model::from_gguf_flash_0731(gguf)?;
        validate_strict_binding(gguf, &model)?;
        validate_session_config(&model.config)?;
        validate_session_lookup_storage(gguf)?;

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
        let memory = build_memory_plan(ctx, &retained, &report, &model.config)?;
        let descriptors = gguf
            .tensors
            .iter()
            .map(DeepSeekV4DescriptorFingerprint::from)
            .collect();
        Ok(DeepSeekV4MetalLoadPlan {
            config: model.config,
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
        if build_memory_plan(ctx, &plan.retained, &plan.report, &plan.config)? != plan.memory {
            return invalid("DeepSeek V4 memory plan changed before realization");
        }
        let refreshed_admission = plan.memory.admission(ctx.memory_signals());
        if !refreshed_admission.admitted {
            return invalid(format!(
                "DeepSeek V4 memory admission changed before realization: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                refreshed_admission.reason.as_str(),
                refreshed_admission.required_bytes,
                refreshed_admission.working_set_headroom_bytes,
                refreshed_admission.signals.process_limit_remaining_bytes,
            ));
        }
        let windows = realize_windows(ctx, gguf, &plan.retained)?;
        let tensors = realize_tensors(ctx, gguf, &plan.retained, &windows)?;
        validate_realization(gguf, &tensors, &plan.report)?;
        let after_residency_bytes = ctx.current_allocated_size();
        plan.memory.reconcile_residency(
            refreshed_admission.signals.current_allocated_bytes,
            after_residency_bytes,
        )?;

        Ok(DeepSeekV4RealizedLoad {
            residency: Self {
                config: plan.config,
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
const DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS: usize = 256;
const DEEPSEEK_V4_NEXT_UNVALIDATED_CONTINUATION_POSITION: u32 =
    DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY as u32;
const DEEPSEEK_V4_NEXT_UNVALIDATED_HCA_BOUNDARY: u32 = 639;
const DEEPSEEK_V4_FIRST_UNALLOCATED_CSA_BOUNDARY: u32 =
    ((DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS + 1) * 4 - 1) as u32;

fn validate_promoted_session_position(position: u32) -> Result<(), DeepSeekV4MetalError> {
    if position >= DEEPSEEK_V4_FIRST_UNALLOCATED_CSA_BOUNDARY {
        return invalid(format!(
            "native session stops before unallocated CSA row 256 at position {DEEPSEEK_V4_FIRST_UNALLOCATED_CSA_BOUNDARY}; next position is {position}"
        ));
    }
    if position >= DEEPSEEK_V4_NEXT_UNVALIDATED_HCA_BOUNDARY {
        return invalid(format!(
            "native session stops before unvalidated HCA publication at position {DEEPSEEK_V4_NEXT_UNVALIDATED_HCA_BOUNDARY}; next position is {position}"
        ));
    }
    if position >= DEEPSEEK_V4_NEXT_UNVALIDATED_CONTINUATION_POSITION {
        return invalid(format!(
            "native session stops after the promoted fourth-HCA continuation at position 512; next position is {position}"
        ));
    }
    Ok(())
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

/// Native qwen-owned DeepSeek V4 decode session.
///
/// Dense-all CSA and four HCA rows are promoted through the position-512
/// continuation. The session fails closed before position 513 until the next
/// retained interval earns its own differential.
pub struct DeepSeekV4Session {
    residency: DeepSeekV4MetalResidency,
    device_registry_id: u64,
    token_id: MetalTensor,
    embedding: MetalTensor,
    residual_primary: MetalTensor,
    residual_secondary: MetalTensor,
    hyper_connection: DeepSeekV4HyperConnectionScratch,
    attention: DeepSeekV4PositionZeroAttentionScratch,
    raw_cache: MetalTensor,
    compressor_frontiers: DeepSeekV4CompressorFrontiers,
    moe: DeepSeekV4MoeScratch,
    final_hidden: MetalTensor,
    final_normalized_hidden: MetalTensor,
    logits: MetalTensor,
    next_position: u32,
    completed: bool,
    poisoned: bool,
}

/// Compatibility name retained for the position-zero live differential.
pub type DeepSeekV4PositionZeroForward = DeepSeekV4Session;

impl DeepSeekV4Session {
    pub fn new(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if residency.device_registry_id() != ctx.device.registryID() {
            return invalid(format!(
                "DeepSeek V4 residency belongs to Metal device registry {}, session context is {}",
                residency.device_registry_id(),
                ctx.device.registryID()
            ));
        }
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
        let compressor_frontiers = DeepSeekV4CompressorFrontiers::new(ctx, residency.config())?;

        Ok(Self {
            device_registry_id: residency.device_registry_id(),
            residency,
            token_id,
            embedding: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64])?,
            residual_primary: MetalTensor::zeros_f32(ctx, residual_shape.clone())?,
            residual_secondary: MetalTensor::zeros_f32(ctx, residual_shape)?,
            hyper_connection: DeepSeekV4HyperConnectionScratch::new(ctx, DEEPSEEK_V4_HIDDEN_SIZE)?,
            attention: DeepSeekV4PositionZeroAttentionScratch::new(ctx, attention_config)?,
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
            final_hidden: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64])?,
            final_normalized_hidden: MetalTensor::zeros_f32(
                ctx,
                vec![DEEPSEEK_V4_HIDDEN_SIZE as u64],
            )?,
            logits: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_VOCAB_SIZE as u64])?,
            next_position: 0,
            completed: false,
            poisoned: false,
        })
    }

    pub fn residency(&self) -> &DeepSeekV4MetalResidency {
        &self.residency
    }

    pub fn logits(&self) -> &MetalTensor {
        &self.logits
    }

    pub fn final_normalized_hidden(&self) -> &MetalTensor {
        &self.final_normalized_hidden
    }

    pub fn next_position(&self) -> u32 {
        self.next_position
    }

    pub fn cache_contract(&self) -> DeepSeekV4AttentionCacheContract {
        DeepSeekV4AttentionCacheContract::LlamaCppB10222F16
    }

    /// Copy completed logits out of shared Metal storage.
    pub fn copy_logits_f32(&self) -> Result<Vec<f32>, DeepSeekV4MetalError> {
        if !self.completed {
            return invalid("DeepSeek V4 logits have not completed");
        }
        host_read_f32(&self.logits, "completed DeepSeek V4 logits")
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
        if self.next_position != 0 {
            return invalid(format!(
                "position-zero entry point requires a fresh session, next position is {}",
                self.next_position
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
    /// frontier needed by the next position. Each layer completes its router
    /// command before host selection and then runs a fresh expert command.
    pub fn forward_token_with_progress(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        mut layer_completed: impl FnMut(usize),
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        if ctx.device.registryID() != self.device_registry_id {
            return invalid(format!(
                "DeepSeek V4 session belongs to Metal device registry {}, got {}",
                self.device_registry_id,
                ctx.device.registryID()
            ));
        }
        if self.poisoned {
            return invalid("DeepSeek V4 session is poisoned by an incomplete token");
        }
        if token_id as usize >= DEEPSEEK_V4_VOCAB_SIZE {
            return invalid(format!(
                "token id {token_id} is outside vocabulary {DEEPSEEK_V4_VOCAB_SIZE}"
            ));
        }
        validate_promoted_session_position(self.next_position)?;

        self.completed = false;
        host_write_i32(&self.token_id, &[token_id as i32], "DeepSeek V4 token ID")?;
        let position = self.next_position;
        // Set before the first cache mutation. If a user progress callback
        // unwinds, a caught panic cannot make a partially published token
        // reusable at the same logical position.
        self.poisoned = true;
        let result = self.forward_token_inner(ctx, token_id, position, &mut layer_completed);
        match result {
            Ok(()) => {
                self.next_position = self
                    .next_position
                    .checked_add(1)
                    .ok_or_else(|| DeepSeekV4MetalError::Invalid("position overflow".into()))?;
                self.completed = true;
                self.poisoned = false;
                Ok(&self.logits)
            }
            Err(error) => Err(error),
        }
    }

    fn forward_token_inner(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        position: u32,
        layer_completed: &mut impl FnMut(usize),
    ) -> Result<(), DeepSeekV4MetalError> {
        let rms_eps = self.residency.config().attention_rms_epsilon;
        let hc_eps = self.residency.config().hyper_connection_epsilon;

        for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
            let raw_cache = self.raw_cache_layer(layer)?;
            let rope = deepseek_v4_layer_rope(self.residency.config(), layer)?;
            let command = ctx.queue.commandBuffer().ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "failed to allocate layer {layer} router command buffer"
                ))
            })?;
            let encoder = KernelEncoder::begin(&command);
            let encode_result = (|| {
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
                let compressed = self.compressor_frontiers.attention_rows(layer, position)?;
                let attention_output = self.attention.encode_finish_dense_f16(
                    ctx,
                    &encoder,
                    &raw_cache,
                    compressed,
                    self.layer_tensor(layer, "attn_sinks.weight")?,
                    self.layer_tensor(layer, "attn_output_a.weight")?,
                    self.layer_tensor(layer, "attn_output_b.weight")?,
                    position,
                    rope,
                )?;
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
                self.moe.encode_router(
                    ctx,
                    &encoder,
                    self.hyper_connection.collapsed_input(),
                    self.layer_tensor(layer, "ffn_norm.weight")?,
                    self.layer_tensor(layer, "ffn_gate_inp.weight")?,
                    rms_eps,
                )?;
                Ok::<(), DeepSeekV4MetalError>(())
            })();
            encoder.end();
            encode_result?;
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return invalid(format!("layer {layer} router command failed: {error:?}"));
            }

            if layer < self.residency.config().hash_layer_count as usize {
                self.moe.route_hash(
                    token_id as usize,
                    self.layer_tensor(layer, "ffn_gate_tid2eid.weight")?,
                )?;
            } else {
                self.moe
                    .route_learned(self.layer_tensor(layer, "exp_probs_b.bias")?)?;
            }

            let command = ctx.queue.commandBuffer().ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "failed to allocate layer {layer} expert command buffer"
                ))
            })?;
            let encoder = KernelEncoder::begin(&command);
            let encode_result = (|| {
                let moe_output = self.moe.encode_experts(
                    ctx,
                    &encoder,
                    self.layer_tensor(layer, "ffn_gate_exps.weight")?,
                    self.layer_tensor(layer, "ffn_up_exps.weight")?,
                    self.layer_tensor(layer, "ffn_down_exps.weight")?,
                    self.layer_tensor(layer, "ffn_gate_shexp.weight")?,
                    self.layer_tensor(layer, "ffn_up_shexp.weight")?,
                    self.layer_tensor(layer, "ffn_down_shexp.weight")?,
                    self.residency.config().swiglu_clamp_experts[layer],
                    self.residency.config().swiglu_clamp_shared[layer],
                )?;
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
                Ok::<(), DeepSeekV4MetalError>(())
            })();
            encoder.end();
            encode_result?;
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return invalid(format!("layer {layer} expert command failed: {error:?}"));
            }
            layer_completed(layer);
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
        expert_count: 256,
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
        || config.expert_count != 256
        || config.expert_used_count != 6
        || config.expert_feed_forward_length != 2_048
        || config.shared_expert_count != 1
        || config.sinkhorn_iterations != DEEPSEEK_V4_SINKHORN_ITERATIONS as u32
    {
        return invalid("native session requires the exact Flash-0731 dimensions");
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

struct DeepSeekV4CompressorFrontier {
    ratio: usize,
    head_dim: usize,
    width: usize,
    rows: usize,
    publication: DeepSeekV4CompressorPublication,
    kv_state: MetalTensor,
    score_state: MetalTensor,
    projected_kv: MetalTensor,
    projected_score: MetalTensor,
    pooled: MetalTensor,
    normalized: MetalTensor,
    published: MetalTensor,
}

impl DeepSeekV4CompressorFrontier {
    fn new(
        ctx: &MetalContext,
        ratio: usize,
        head_dim: usize,
        publication: DeepSeekV4CompressorPublication,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if !matches!(ratio, 4 | 128) || head_dim == 0 {
            return invalid(format!(
                "compressor frontier requires ratio 4 or 128 and a nonzero head dimension, got ratio={ratio} head_dim={head_dim}"
            ));
        }
        if publication == DeepSeekV4CompressorPublication::IndexerHadamard && head_dim != 128 {
            return invalid("indexer publication requires exactly 128 dimensions");
        }
        let (width, rows, state_elements) = compressor_frontier_geometry(ratio, head_dim)?;
        let zeros = vec![0.0f32; state_elements];
        let negative_infinity = vec![f32::NEG_INFINITY; state_elements];
        Ok(Self {
            ratio,
            head_dim,
            width,
            rows,
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
            published: MetalTensor::zeros_f16(
                ctx,
                vec![head_dim as u64, DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS as u64],
            )?,
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
            &[
                self.head_dim as u64,
                DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS as u64,
            ],
            true,
            "published compressor rows",
        )?;

        let following_position = position
            .checked_add(1)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("compressor position overflow".into()))?;
        let boundary = (following_position as usize).is_multiple_of(self.ratio);
        let published_row = if boundary {
            let row = following_position as usize / self.ratio - 1;
            if row >= DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS {
                return invalid(format!(
                    "compressor published row {row} exceeds the first {}-row slab",
                    DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS
                ));
            }
            Some(row)
        } else {
            None
        };

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
        let ape_row = ape.view_subrange(
            ((position as usize % self.ratio) * self.width) as u64,
            vec![self.width as u64],
        );
        let state_row = if self.ratio == 4 {
            self.ratio + position as usize % self.ratio
        } else {
            position as usize % self.ratio
        };
        encode_compressor_frontier_write(
            ctx,
            enc,
            &self.projected_kv,
            &self.projected_score,
            &ape_row,
            &self.kv_state,
            &self.score_state,
            self.width,
            state_row,
        )?;

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
        attention: DeepSeekV4CompressorFrontier,
        indexer: DeepSeekV4CompressorFrontier,
    },
    HeavilyCompressed {
        attention: DeepSeekV4CompressorFrontier,
    },
}

struct DeepSeekV4CompressorFrontiers {
    hidden_size: usize,
    layers: Vec<DeepSeekV4LayerCompressorFrontiers>,
}

impl DeepSeekV4CompressorFrontiers {
    fn new(ctx: &MetalContext, config: &DeepSeekV4Config) -> Result<Self, DeepSeekV4MetalError> {
        let hidden_size = config.hidden_size as usize;
        let attention_dim = config.key_length as usize;
        let indexer_dim = config.indexer_key_length as usize;
        let mut layers = Vec::with_capacity(config.attention_kinds.len());
        for kind in config.attention_kinds.iter().copied() {
            layers.push(match kind {
                AttentionKind::SlidingWindow => DeepSeekV4LayerCompressorFrontiers::SlidingWindow,
                AttentionKind::CompressedSparse => {
                    DeepSeekV4LayerCompressorFrontiers::CompressedSparse {
                        attention: DeepSeekV4CompressorFrontier::new(
                            ctx,
                            4,
                            attention_dim,
                            DeepSeekV4CompressorPublication::Attention,
                        )?,
                        indexer: DeepSeekV4CompressorFrontier::new(
                            ctx,
                            4,
                            indexer_dim,
                            DeepSeekV4CompressorPublication::IndexerHadamard,
                        )?,
                    }
                }
                AttentionKind::HeavilyCompressed => {
                    DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed {
                        attention: DeepSeekV4CompressorFrontier::new(
                            ctx,
                            128,
                            attention_dim,
                            DeepSeekV4CompressorPublication::Attention,
                        )?,
                    }
                }
            });
        }
        Ok(Self {
            hidden_size,
            layers,
        })
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
        if count > DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS {
            return invalid(format!(
                "visible compressed rows {count} exceed the first {}-row slab",
                DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS
            ));
        }
        Ok(Some(DeepSeekV4PublishedRows {
            cache: &frontier.published,
            count,
        }))
    }
}

#[derive(Clone, Copy)]
struct DeepSeekV4PublishedRows<'a> {
    cache: &'a MetalTensor,
    count: usize,
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
    low_rank: MetalTensor,
    output: MetalTensor,
    head_norm_ones: MetalTensor,
}

impl DeepSeekV4PositionZeroAttentionScratch {
    pub fn new(
        ctx: &MetalContext,
        config: DeepSeekV4PositionZeroAttentionConfig,
    ) -> Result<Self, DeepSeekV4MetalError> {
        let dims = config.checked()?;
        let ones = vec![1.0f32; config.head_dim];
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
            low_rank: MetalTensor::zeros_f32(ctx, vec![dims.low_rank_width as u64])?,
            output: MetalTensor::zeros_f32(ctx, vec![config.hidden_size as u64])?,
            head_norm_ones: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&ones),
                vec![config.head_dim as u64],
                GgmlType::F32,
            )?,
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

        encode_ds4_rope_tail_adjacent_in_place(ctx, enc, &self.queries, position, rope, false)?;
        encode_ds4_rope_tail_adjacent_in_place(ctx, enc, &self.kv, position, rope, false)?;
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
    fn encode_finish_dense_f16<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        raw_cache: &MetalTensor,
        compressed: Option<DeepSeekV4PublishedRows<'_>>,
        sinks: &MetalTensor,
        output_a: &MetalTensor,
        output_b: &MetalTensor,
        position: u32,
        rope: DeepSeekV4RopeParameters,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
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
        if let Some(rows) = compressed {
            validate_f16(
                rows.cache,
                &[
                    c.head_dim as u64,
                    DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS as u64,
                ],
                false,
                "compressed attention cache",
            )?;
            if rows.count == 0 || rows.count > DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS {
                return invalid(format!(
                    "compressed attention row count {} is out of range",
                    rows.count
                ));
            }
        }

        encode_dense_sink_attention_f16(
            ctx,
            enc,
            &self.queries,
            raw_cache,
            compressed,
            sinks,
            &self.attention,
            position,
            c,
        )?;
        encode_ds4_rope_tail_adjacent_in_place(ctx, enc, &self.attention, position, rope, true)?;

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
        validate_f32(
            function,
            &[
                residual_len(self.hidden_size)? as u64,
                DEEPSEEK_V4_HC_PARAMETER_COUNT as u64,
            ],
            false,
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
        encode_f32_projection(
            ctx,
            enc,
            function,
            &self.normalized,
            &self.mixes,
            residual_len(self.hidden_size)?,
            DEEPSEEK_V4_HC_PARAMETER_COUNT,
        )?;

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
        validate_f32(
            function,
            &[
                residual_len(self.hidden_size)? as u64,
                DEEPSEEK_V4_CONNECTION_COUNT as u64,
            ],
            false,
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
        encode_f32_projection(
            ctx,
            enc,
            function,
            &self.normalized,
            &self.head_mixes,
            residual_len(self.hidden_size)?,
            DEEPSEEK_V4_CONNECTION_COUNT,
        )?;
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
        if !self.routed_scale.is_finite() || self.routed_scale <= 0.0 {
            return invalid("MoE routed scale must be finite and positive");
        }
        checked_mul(self.hidden_size, self.top_k, "MoE routed output scratch")?;
        Ok(())
    }
}

/// Reusable session-owned storage for one native DS4 single-token MoE body.
///
/// CPU routing is an intentional correctness seam. After `encode_router`, the
/// caller must end encoding, commit, and wait for that command buffer before
/// calling either `route_*` method. The route methods perform narrowly scoped
/// host access to shared Metal buffers. A subsequent `encode_experts` must be
/// placed in a new serial command encoder so those host writes are visible.
pub struct DeepSeekV4MoeScratch {
    config: DeepSeekV4MoeConfig,
    normalized_input: MetalTensor,
    logits: MetalTensor,
    expert_ids: MetalTensor,
    weights: MetalTensor,
    gate: MetalTensor,
    up: MetalTensor,
    inner: MetalTensor,
    expert_outputs: MetalTensor,
    routed_output: MetalTensor,
    shared_output: MetalTensor,
    final_output: MetalTensor,
}

impl DeepSeekV4MoeScratch {
    pub fn new(
        ctx: &MetalContext,
        config: DeepSeekV4MoeConfig,
    ) -> Result<Self, DeepSeekV4MetalError> {
        config.checked()?;
        let c = config;
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
            gate: MetalTensor::zeros_f32(ctx, vec![c.ffn_size as u64])?,
            up: MetalTensor::zeros_f32(ctx, vec![c.ffn_size as u64])?,
            inner: MetalTensor::zeros_f32(ctx, vec![c.ffn_size as u64])?,
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

    /// Route from token-major contiguous `[K,V]` I32 physical storage.
    /// Requires the completed command boundary documented on this type.
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

    /// Tie-stable learned routing by `sqrt(softplus(logit)) + bias`; selected
    /// weights remain the unbiased scores. Requires a completed router command.
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

    /// Execute selected expert slices and the shared expert with generic native
    /// Metal matvecs, then form `weighted_routed + shared`.
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
            encode_projection(
                ctx,
                enc,
                &gate,
                &self.normalized_input,
                &self.gate,
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
            encode_ds4_clamped_swiglu(ctx, enc, &self.gate, &self.up, &self.inner, expert_clamp)?;
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

        encode_projection(
            ctx,
            enc,
            shared_gate,
            &self.normalized_input,
            &self.gate,
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
        encode_ds4_clamped_swiglu(ctx, enc, &self.gate, &self.up, &self.inner, shared_clamp)?;
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

    fn validate_scratch(&self) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        c.checked()?;
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
        for (tensor, name) in [
            (&self.gate, "MoE gate scratch"),
            (&self.up, "MoE up scratch"),
            (&self.inner, "MoE inner scratch"),
        ] {
            validate_f32(tensor, &[c.ffn_size as u64], true, name)?;
        }
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

/// Host writes must occur between completed and not-yet-committed commands.
fn host_write_f32(
    tensor: &MetalTensor,
    values: &[f32],
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    validate_f32(tensor, &[values.len() as u64], true, name)?;
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
    validate_i32(tensor, &[values.len() as u64], true, name)?;
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
            &[
                config.head_dim as u64,
                DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS as u64,
            ],
            false,
            "dense compressed attention cache",
        )?;
        if rows.count == 0 || rows.count > DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS {
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
    if enc.concurrent {
        return invalid(format!("{kernel} requires ordered serial dispatches"));
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
    if tensor.offset % std::mem::align_of::<f32>() as u64 != 0 {
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
    if embedding_dtype != GgmlType::Q6_K {
        return invalid(format!(
            "token_embd.weight must be Q6_K for the native session get_rows path, got {embedding_dtype:?}"
        ));
    }
    if output_dtype != GgmlType::Q6_K {
        return invalid(format!(
            "output.weight must be Q6_K for the native session logits path, got {output_dtype:?}"
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
            DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS,
            "published compressor history elements",
        )?,
        std::mem::size_of::<u16>(),
    )
}

fn deepseek_v4_session_allocation_requests(
    config: &DeepSeekV4Config,
) -> Result<Vec<DeepSeekV4SessionAllocationRequest>, DeepSeekV4MetalError> {
    validate_session_config(config)?;
    deepseek_v4_session_allocation_requests_for_kinds(&config.attention_kinds)
}

fn deepseek_v4_session_allocation_requests_for_kinds(
    attention_kinds: &[AttentionKind],
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
        expert_count: 256,
        top_k: 6,
        routed_scale: 1.0,
    };
    moe.checked()?;
    let mut requests = Vec::with_capacity(474);
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
                )?;
                append_compressor_frontier_allocations(
                    &mut requests,
                    &format!("compressor.{layer}.indexer"),
                    4,
                    indexer_dim,
                    DeepSeekV4CompressorPublication::IndexerHadamard,
                )?;
            }
            AttentionKind::HeavilyCompressed => append_compressor_frontier_allocations(
                &mut requests,
                &format!("compressor.{layer}.attention"),
                128,
                attention_dim,
                DeepSeekV4CompressorPublication::Attention,
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
    for name in ["moe.gate", "moe.up", "moe.inner"] {
        push_session_allocation(&mut requests, name, moe.ffn_size, f32_bytes)?;
    }
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
    let priced = ctx.shared_buffer_size_and_align(logical_bytes)?;
    if priced.size < logical_bytes || priced.alignment == 0 || !priced.alignment.is_power_of_two() {
        return invalid(format!(
            "invalid Metal pricing for {name:?}: logical={logical_bytes} priced={} alignment={}",
            priced.size, priced.alignment
        ));
    }
    Ok((priced.size, priced.alignment))
}

fn build_memory_plan(
    ctx: &MetalContext,
    retained: &RetainedStoragePlan,
    report: &DeepSeekV4ResidencyReport,
    config: &DeepSeekV4Config,
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

    let requests = deepseek_v4_session_allocation_requests(config)?;
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
mod tests {
    use super::*;
    use crate::deepseek_v4_oracle::{
        CompressorState, RopeDirection, RopeParameters,
        attention_fp8_nope_bf16_rope_roundtrip_in_place, grouped_low_rank_projection,
        hadamard_128_in_place, hyper_connection_head, hyper_connection_post, hyper_connection_pre,
        mat_vec, rms_norm, rope_tail_in_place, shared_kv_attention, shared_kv_projection,
    };
    use crate::tensor::{GgmlType, TensorDesc};
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(ctx) => Some(ctx),
            Err(MetalError::NoDevice) => None,
            Err(error) => panic!("Metal context: {error}"),
        }
    }

    fn offset_f32(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        let prefix = 16usize;
        let mut bytes = vec![0xA5u8; prefix];
        bytes.extend_from_slice(bytemuck::cast_slice(values));
        bytes.extend_from_slice(&[0x5Au8; 20]);
        MetalTensor {
            buffer: ctx.buffer_from(&bytes).expect("offset F32 buffer"),
            offset: prefix as u64,
            shape,
            dtype: GgmlType::F32,
            provenance: MetalTensorProvenance::OwnedWritable,
        }
    }

    fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
        unsafe {
            let pointer = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>();
            std::slice::from_raw_parts(pointer, tensor.n_elements() as usize).to_vec()
        }
    }

    fn read_f16(tensor: &MetalTensor) -> Vec<f32> {
        assert_eq!(tensor.dtype, GgmlType::F16);
        unsafe {
            let pointer = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<u16>();
            std::slice::from_raw_parts(pointer, tensor.n_elements() as usize)
                .iter()
                .map(|bits| half::f16::from_bits(*bits).to_f32())
                .collect()
        }
    }

    fn offset_i32(ctx: &MetalContext, values: &[i32], shape: Vec<u64>) -> MetalTensor {
        let prefix = 20usize;
        let mut bytes = vec![0xA5u8; prefix];
        bytes.extend_from_slice(bytemuck::cast_slice(values));
        bytes.extend_from_slice(&[0x5Au8; 20]);
        MetalTensor {
            buffer: ctx.buffer_from(&bytes).expect("offset I32 buffer"),
            offset: prefix as u64,
            shape,
            dtype: GgmlType::I32,
            provenance: MetalTensorProvenance::OwnedWritable,
        }
    }

    fn read_i32(tensor: &MetalTensor) -> Vec<i32> {
        host_read_i32(tensor, "test I32 tensor").expect("read I32")
    }

    fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let allowed = tolerance * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= allowed,
                "{label}[{index}] = {actual}, expected {expected}, tolerance {allowed}"
            );
        }
    }

    fn f32_desc(name: &str, offset: u64) -> TensorDesc {
        TensorDesc {
            name: name.into(),
            shape: vec![8],
            dtype: GgmlType::F32,
            shard_idx: 0,
            data_offset: offset,
            n_bytes: 32,
        }
    }

    #[test]
    fn session_position_guard_stops_after_fourth_hca_and_at_later_boundaries() {
        assert_eq!(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY, 513);
        validate_promoted_session_position((DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY - 1) as u32)
            .unwrap();
        let continuation =
            validate_promoted_session_position(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY as u32)
                .unwrap_err();
        assert!(continuation.to_string().contains("next position is 513"));
        let hca = validate_promoted_session_position(639).unwrap_err();
        assert!(hca.to_string().contains("HCA publication at position 639"));
        let csa = validate_promoted_session_position(1027).unwrap_err();
        assert!(csa.to_string().contains("CSA row 256 at position 1027"));
    }

    #[test]
    fn session_memory_inventory_is_complete_and_unique() {
        let mut kinds = vec![AttentionKind::SlidingWindow; 2];
        kinds.extend(std::iter::repeat_n(AttentionKind::CompressedSparse, 21));
        kinds.extend(std::iter::repeat_n(AttentionKind::HeavilyCompressed, 20));
        let requests = deepseek_v4_session_allocation_requests_for_kinds(&kinds).unwrap();
        assert_eq!(requests.len(), 474);
        assert_eq!(
            requests
                .iter()
                .map(|request| request.logical_bytes)
                .sum::<u64>(),
            31_962_388
        );
        let names = requests
            .iter()
            .map(|request| request.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(names.len(), requests.len());
        assert_eq!(
            requests
                .iter()
                .filter(|request| request.name.ends_with(".published"))
                .map(|request| request.logical_bytes)
                .sum::<u64>(),
            12_124_160
        );
        assert_eq!(
            requests
                .iter()
                .find(|request| request.name == "raw_cache")
                .unwrap()
                .logical_bytes,
            5_636_096
        );
    }

    #[test]
    fn memory_plan_admission_and_reconciliation_fail_closed() {
        let plan = DeepSeekV4MemoryPlan {
            residency_buffer_count: 4,
            residency_logical_bytes: 900,
            residency_priced_upper_bytes: 1_000,
            session_logical_bytes: 400,
            session_priced_upper_bytes: 500,
            total_priced_upper_bytes: 1_500,
            session_allocations: Vec::new(),
        };
        let required = plan.required_with_reserve_bytes().unwrap();
        let baseline = 100_u64;
        let exact = plan.admission(MetalMemorySignals {
            recommended_max_bytes: baseline + required,
            current_allocated_bytes: baseline,
            process_limit_remaining_bytes: Some(0),
        });
        assert!(exact.admitted);
        assert_eq!(
            exact.reason,
            crate::metal::MetalMemoryAdmissionReason::AdmittedProcessBudgetOmitted
        );
        let short = plan.admission(MetalMemorySignals {
            recommended_max_bytes: baseline + required - 1,
            current_allocated_bytes: baseline,
            process_limit_remaining_bytes: Some(0),
        });
        assert!(!short.admitted);
        assert_eq!(
            short.reason,
            crate::metal::MetalMemoryAdmissionReason::WorkingSetInsufficient
        );

        let reconciliation = plan
            .reconcile(DeepSeekV4MemorySamples {
                before_residency_bytes: baseline,
                after_residency_bytes: baseline + 1_000,
                after_session_bytes: baseline + 1_500,
                after_first_forward_bytes: baseline + required,
            })
            .unwrap();
        assert_eq!(reconciliation.observed_residency_delta_bytes, 1_000);
        assert_eq!(reconciliation.observed_session_delta_bytes, 1_500);
        assert_eq!(reconciliation.sampled_peak_delta_bytes, required);
        let error = plan
            .reconcile_session(baseline, baseline + 999, baseline + 1_500)
            .unwrap_err();
        assert!(error.to_string().contains("session increment"));
        let error = plan
            .reconcile(DeepSeekV4MemorySamples {
                before_residency_bytes: baseline,
                after_residency_bytes: baseline + 1_000,
                after_session_bytes: baseline + 1_500,
                after_first_forward_bytes: baseline + required + 1,
            })
            .unwrap_err();
        assert!(error.to_string().contains("first forward delta"));
    }

    #[test]
    fn retained_plan_preflight_rejects_descriptor_and_window_drift() {
        let tensors = vec![f32_desc("a", 0), f32_desc("b", 32)];
        let requests = tensors.iter().collect::<Vec<_>>();
        let plan = plan_retained_storage(&[131_072], &requests, 4_096, 65_536, 32).unwrap();
        validate_retained_plan_geometry(4_096, 65_536, &[131_072], &tensors, &plan).unwrap();

        let mut descriptor_drift = plan.clone();
        descriptor_drift.entries[1].data_offset += 32;
        let error =
            validate_retained_plan_geometry(4_096, 65_536, &[131_072], &tensors, &descriptor_drift)
                .unwrap_err();
        assert!(error.to_string().contains("descriptor drift"));

        let mut window_drift = plan;
        window_drift.windows[0].length = 135_168;
        let error =
            validate_retained_plan_geometry(4_096, 65_536, &[131_072], &tensors, &window_drift)
                .unwrap_err();
        assert!(error.to_string().contains("window 0"));

        let plan = plan_retained_storage(&[131_072], &requests, 4_096, 65_536, 32).unwrap();
        let mut usable_window_drift = plan.clone();
        usable_window_drift.usable_window_length -= 4_096;
        let error = validate_retained_plan_geometry(
            4_096,
            65_536,
            &[131_072],
            &tensors,
            &usable_window_drift,
        )
        .unwrap_err();
        assert!(error.to_string().contains("geometry changed"));

        let mut view_drift = plan;
        let RetainedStorageDisposition::View { buffer_offset, .. } =
            &mut view_drift.entries[0].disposition
        else {
            panic!("expected retained view");
        };
        *buffer_offset += 1;
        let error =
            validate_retained_plan_geometry(4_096, 65_536, &[131_072], &tensors, &view_drift)
                .unwrap_err();
        assert!(error.to_string().contains("window binding"));

        let alias_tensors = vec![f32_desc("source", 0), f32_desc("alias", 0)];
        let alias_requests = alias_tensors.iter().collect::<Vec<_>>();
        let mut alias_plan =
            plan_retained_storage(&[131_072], &alias_requests, 4_096, 65_536, 32).unwrap();
        alias_plan.entries[1].disposition = RetainedStorageDisposition::Alias {
            source_request_index: 1,
        };
        let error =
            validate_retained_plan_geometry(4_096, 65_536, &[131_072], &alias_tensors, &alias_plan)
                .unwrap_err();
        assert!(error.to_string().contains("non-prior source"));

        let tail_tensors = vec![f32_desc("tail", 131_072)];
        let tail_requests = tail_tensors.iter().collect::<Vec<_>>();
        let tail_plan =
            plan_retained_storage(&[131_120], &tail_requests, 4_096, 65_536, 32).unwrap();
        assert!(matches!(
            tail_plan.entries[0].disposition,
            RetainedStorageDisposition::CopyFallback {
                reason: RetainedStorageFallback::FinalPartialPage
            }
        ));
        validate_retained_plan_geometry(4_096, 65_536, &[131_120], &tail_tensors, &tail_plan)
            .unwrap();
        let error =
            validate_retained_plan_geometry(4_096, 65_536, &[135_168], &tail_tensors, &tail_plan)
                .unwrap_err();
        assert!(error.to_string().contains("final-partial-page fallback"));
    }

    #[test]
    fn descriptor_fingerprints_cover_representation_not_just_storage_range() {
        let tensors = vec![f32_desc("a", 0), f32_desc("b", 32)];
        let fingerprints = tensors
            .iter()
            .map(DeepSeekV4DescriptorFingerprint::from)
            .collect::<Vec<_>>();
        validate_descriptor_fingerprints(&tensors, &fingerprints).unwrap();

        let mut dtype_drift = tensors.clone();
        dtype_drift[0].dtype = GgmlType::I32;
        let error = validate_descriptor_fingerprints(&dtype_drift, &fingerprints).unwrap_err();
        assert!(error.to_string().contains("fingerprint changed"));

        let mut shape_drift = tensors;
        shape_drift[1].shape = vec![4, 2];
        let error = validate_descriptor_fingerprints(&shape_drift, &fingerprints).unwrap_err();
        assert!(error.to_string().contains("fingerprint changed"));
    }

    #[test]
    fn session_lookup_storage_rejects_unsupported_dtypes_before_residency() {
        validate_session_lookup_dtypes(GgmlType::Q6_K, GgmlType::Q6_K).unwrap();
        let embedding = validate_session_lookup_dtypes(GgmlType::F32, GgmlType::Q6_K).unwrap_err();
        assert!(embedding.to_string().contains("token_embd.weight"));
        let output = validate_session_lookup_dtypes(GgmlType::Q6_K, GgmlType::F32).unwrap_err();
        assert!(output.to_string().contains("output.weight"));
    }

    #[test]
    fn position_zero_attention_matches_operation_oracles_with_offsets_and_groups() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let config = DeepSeekV4PositionZeroAttentionConfig {
            hidden_size: 7,
            q_lora_rank: 3,
            head_count: 3,
            head_dim: 192,
            rotary_dim: 64,
            group_count: 3,
            output_rank: 2,
        };
        let c = config;
        let query_width = c.head_count * c.head_dim;
        let group_width = query_width / c.group_count;
        let low_rank_width = c.group_count * c.output_rank;
        let rms_eps = 2.0e-5;
        let input_values = (0..c.hidden_size)
            .map(|i| (i as f32 - 2.6) * 0.31 + if i % 2 == 0 { 0.17 } else { -0.23 })
            .collect::<Vec<_>>();
        let attention_norm_values = (0..c.hidden_size)
            .map(|i| 0.61 + i as f32 * 0.083)
            .collect::<Vec<_>>();
        let q_a_values = (0..c.hidden_size * c.q_lora_rank)
            .map(|i| ((i * 11 + 3) % 23) as f32 * 0.037 - 0.39)
            .collect::<Vec<_>>();
        let q_a_norm_values = vec![0.73, 1.19, 0.52];
        let q_b_values = (0..c.q_lora_rank * query_width)
            .map(|i| ((i * 7 + i / 5 + 1) % 29) as f32 * 0.029 - 0.36)
            .collect::<Vec<_>>();
        let kv_weight_values = (0..c.hidden_size * c.head_dim)
            .map(|i| ((i * 13 + 5) % 31) as f32 * 0.021 - 0.28)
            .collect::<Vec<_>>();
        let kv_norm_values = (0..c.head_dim)
            .map(|i| 0.51 + ((i * 19 + i / 7) % 47) as f32 * 0.029)
            .collect::<Vec<_>>();
        let sink_values = vec![-0.83, 0.41, 1.27];
        let output_a_values = (0..group_width * low_rank_width)
            .map(|i| {
                let row = i / group_width;
                let column = i % group_width;
                (row as f32 - 2.1) * 0.17
                    + (column as f32 - 1.7) * 0.09
                    + if (row + column) % 2 == 0 { 0.14 } else { -0.08 }
            })
            .collect::<Vec<_>>();
        let output_b_values = (0..low_rank_width * c.hidden_size)
            .map(|i| ((i * 17 + i / 4 + 2) % 37) as f32 * 0.018 - 0.31)
            .collect::<Vec<_>>();

        let expected_normalized =
            rms_norm(&input_values, Some(&attention_norm_values), rms_eps).unwrap();
        let expected_projection = shared_kv_projection(
            &expected_normalized,
            &q_a_values,
            &q_a_norm_values,
            &q_b_values,
            &kv_weight_values,
            &kv_norm_values,
            c.q_lora_rank,
            c.head_count,
            c.head_dim,
            rms_eps,
        )
        .unwrap();
        let mut expected_cached_kv = expected_projection.kv.clone();
        attention_fp8_nope_bf16_rope_roundtrip_in_place(&mut expected_cached_kv, c.rotary_dim)
            .unwrap();
        let expected_attention = shared_kv_attention(
            &expected_projection.queries,
            c.head_count,
            c.head_dim,
            &expected_cached_kv,
            &[],
            None,
            &sink_values,
        )
        .unwrap();
        let direct_f32_attention = shared_kv_attention(
            &expected_projection.queries,
            c.head_count,
            c.head_dim,
            &expected_projection.kv,
            &[],
            None,
            &sink_values,
        )
        .unwrap();
        assert!(
            direct_f32_attention
                .iter()
                .zip(&expected_attention)
                .any(|(old, expected)| (old - expected).abs() > 1.0e-4),
            "fixture must reject direct-F32 same-token KV"
        );
        let expected_low_rank = grouped_low_rank_projection(
            &expected_attention,
            c.head_count,
            c.head_dim,
            c.group_count,
            c.output_rank,
            &output_a_values,
        )
        .unwrap();
        let expected_output = mat_vec(
            &output_b_values,
            low_rank_width,
            c.hidden_size,
            &expected_low_rank,
        )
        .unwrap();

        let input = offset_f32(&ctx, &input_values, vec![c.hidden_size as u64]);
        let attention_norm = offset_f32(&ctx, &attention_norm_values, vec![c.hidden_size as u64]);
        let q_a = offset_f32(
            &ctx,
            &q_a_values,
            vec![c.hidden_size as u64, c.q_lora_rank as u64],
        );
        let q_a_norm = offset_f32(&ctx, &q_a_norm_values, vec![c.q_lora_rank as u64]);
        let q_b = offset_f32(
            &ctx,
            &q_b_values,
            vec![c.q_lora_rank as u64, query_width as u64],
        );
        let kv_weight = offset_f32(
            &ctx,
            &kv_weight_values,
            vec![c.hidden_size as u64, c.head_dim as u64],
        );
        let kv_norm = offset_f32(&ctx, &kv_norm_values, vec![c.head_dim as u64]);
        let sinks = offset_f32(&ctx, &sink_values, vec![c.head_count as u64]);
        let output_a = offset_f32(
            &ctx,
            &output_a_values,
            vec![group_width as u64, low_rank_width as u64],
        );
        let output_b = offset_f32(
            &ctx,
            &output_b_values,
            vec![low_rank_width as u64, c.hidden_size as u64],
        );
        let scratch =
            DeepSeekV4PositionZeroAttentionScratch::new(&ctx, config).expect("attention scratch");
        assert_eq!(read_f32(&scratch.head_norm_ones), vec![1.0; c.head_dim]);

        let mut cache_fixture = (0..c.head_dim)
            .map(|i| (i as f32 - 91.0) * 0.000_061_035_156_25)
            .collect::<Vec<_>>();
        cache_fixture[0] = 1.00390625;
        cache_fixture[1] = 0.004150390625;
        cache_fixture[2] = 0.004638671875;
        cache_fixture[3] = -0.004150390625;
        cache_fixture[63] = 1.5;
        cache_fixture[64] = 1.0625;
        cache_fixture[65] = 1.1875;
        cache_fixture[66] = -1.0625;
        cache_fixture[127] = 448.0;
        cache_fixture[128] = 1.00390625;
        cache_fixture[129] = 1.01171875;
        cache_fixture[130] = -1.00390625;
        let mut expected_cache_fixture = cache_fixture.clone();
        attention_fp8_nope_bf16_rope_roundtrip_in_place(&mut expected_cache_fixture, c.rotary_dim)
            .unwrap();
        let cache_fixture_input = offset_f32(&ctx, &cache_fixture, vec![c.head_dim as u64]);
        let cache_fixture_output =
            offset_f32(&ctx, &vec![0.0; c.head_dim], vec![c.head_dim as u64]);
        let cache_command = ctx.queue.commandBuffer().expect("cache command buffer");
        let cache_encoder = KernelEncoder::begin(&cache_command);
        encode_attention_cache_roundtrip(
            &ctx,
            &cache_encoder,
            &cache_fixture_input,
            &cache_fixture_output,
            c,
        )
        .expect("encode attention cache roundtrip");
        cache_encoder.end();
        cache_command.commit();
        cache_command.waitUntilCompleted();
        assert!(
            cache_command.error().is_none(),
            "cache command failed: {:?}",
            cache_command.error()
        );
        let actual_cache_fixture = read_f32(&cache_fixture_output);
        assert!(
            actual_cache_fixture
                .iter()
                .zip(&expected_cache_fixture)
                .all(|(&actual, &expected)| actual.to_bits() == expected.to_bits()),
            "cache fixture must match the oracle bit-for-bit"
        );

        let command = ctx.queue.commandBuffer().expect("attention command buffer");
        let encoder = KernelEncoder::begin(&command);
        let encoded_output = scratch
            .encode(
                &ctx,
                &encoder,
                &input,
                &attention_norm,
                &q_a,
                &q_a_norm,
                &q_b,
                &kv_weight,
                &kv_norm,
                &sinks,
                &output_a,
                &output_b,
                rms_eps,
            )
            .expect("encode position-zero attention");
        assert!(std::ptr::eq(encoded_output, scratch.output()));
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "attention command failed: {:?}",
            command.error()
        );

        assert_close(
            "attention input norm",
            &read_f32(scratch.normalized_input()),
            &expected_normalized,
            3e-5,
        );
        assert_close(
            "Q LoRA raw",
            &read_f32(scratch.q_lora_raw()),
            &expected_projection.q_lora_raw,
            4e-5,
        );
        assert_close(
            "Q LoRA",
            &read_f32(scratch.q_lora()),
            &expected_projection.q_lora,
            4e-5,
        );
        assert_close(
            "queries",
            &read_f32(scratch.queries()),
            &expected_projection.queries,
            5e-5,
        );
        assert_close(
            "KV raw",
            &read_f32(scratch.kv_raw()),
            &expected_projection.kv_raw,
            4e-5,
        );
        assert_close("KV", &read_f32(scratch.kv()), &expected_projection.kv, 4e-5);
        assert_close(
            "cached KV",
            &read_f32(scratch.cached_kv()),
            &expected_cached_kv,
            0.0,
        );
        assert_close(
            "sink attention",
            &read_f32(scratch.attention_heads()),
            &expected_attention,
            6e-5,
        );
        assert_close(
            "grouped low rank",
            &read_f32(scratch.low_rank()),
            &expected_low_rank,
            7e-5,
        );
        assert_close(
            "block output",
            &read_f32(scratch.output()),
            &expected_output,
            8e-5,
        );

        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let mut sink_as_value = vec![0.0; query_width];
        for head in 0..c.head_count {
            let query = &expected_projection.queries[head * c.head_dim..(head + 1) * c.head_dim];
            let score = query
                .iter()
                .zip(&expected_cached_kv)
                .map(|(q, k)| q * k)
                .sum::<f32>()
                * scale;
            let maximum = score.max(sink_values[head]);
            let kv_mass = (score - maximum).exp();
            let sink_mass = (sink_values[head] - maximum).exp();
            for dimension in 0..c.head_dim {
                sink_as_value[head * c.head_dim + dimension] =
                    (expected_cached_kv[dimension] * kv_mass + sink_values[head] * sink_mass)
                        / (kv_mass + sink_mass);
            }
        }
        assert!(
            sink_as_value
                .iter()
                .zip(&expected_attention)
                .any(|(wrong, right)| (wrong - right).abs() > 1e-2),
            "fixture must reject treating sink denominator mass as a value"
        );

        let mut reused_first_group_rows = Vec::with_capacity(low_rank_width);
        for group in 0..c.group_count {
            reused_first_group_rows.extend(
                mat_vec(
                    &output_a_values[..group_width * c.output_rank],
                    group_width,
                    c.output_rank,
                    &expected_attention[group * group_width..(group + 1) * group_width],
                )
                .unwrap(),
            );
        }
        assert!(
            reused_first_group_rows
                .iter()
                .zip(&expected_low_rank)
                .any(|(wrong, right)| (wrong - right).abs() > 1e-2),
            "fixture must reject cross-group output A row indexing"
        );
    }

    #[test]
    fn continuing_rope_f16_cache_and_local_attention_match_cpu() {
        let Some(ctx) = metal_context() else {
            return;
        };
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 128;
        const ROTARY: usize = 64;
        let config = DeepSeekV4PositionZeroAttentionConfig {
            hidden_size: 1,
            q_lora_rank: 1,
            head_count: HEADS,
            head_dim: HEAD_DIM,
            rotary_dim: ROTARY,
            group_count: 1,
            output_rank: 1,
        };
        let rope = DeepSeekV4RopeParameters {
            rotary_dim: ROTARY,
            theta: 160_000.0,
            scaling_factor: 16.0,
            original_context_length: 65_536,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let oracle_rope = RopeParameters::yarn(
            ROTARY,
            rope.theta,
            rope.scaling_factor,
            rope.original_context_length,
            rope.beta_fast,
            rope.beta_slow,
        );
        let query_values = (0..HEADS * HEAD_DIM)
            .map(|index| {
                (index as f32 - 91.0) * 0.0037 + if index.is_multiple_of(5) { 0.19 } else { -0.07 }
            })
            .collect::<Vec<_>>();
        let kv0 = (0..HEAD_DIM)
            .map(|index| (index as f32 - 43.0) * 0.0051 + (index % 7) as f32 * 0.013)
            .collect::<Vec<_>>();
        let kv1 = (0..HEAD_DIM)
            .map(|index| (67.0 - index as f32) * 0.0043 - (index % 11) as f32 * 0.009)
            .collect::<Vec<_>>();
        let sinks = vec![-0.37, 0.82];

        let mut expected_queries = query_values.clone();
        rope_tail_in_place(
            &mut expected_queries,
            HEADS,
            HEAD_DIM,
            1,
            oracle_rope,
            RopeDirection::Forward,
        )
        .unwrap();
        let mut expected_kv1 = kv1.clone();
        rope_tail_in_place(
            &mut expected_kv1,
            1,
            HEAD_DIM,
            1,
            oracle_rope,
            RopeDirection::Forward,
        )
        .unwrap();
        let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
        let cached_kv0 = kv0.iter().copied().map(round_f16).collect::<Vec<_>>();
        let cached_kv1 = expected_kv1
            .iter()
            .copied()
            .map(round_f16)
            .collect::<Vec<_>>();
        let mut raw_rows = cached_kv0.clone();
        raw_rows.extend_from_slice(&cached_kv1);
        let mut expected_output = shared_kv_attention(
            &expected_queries,
            HEADS,
            HEAD_DIM,
            &raw_rows,
            &[],
            None,
            &sinks,
        )
        .unwrap();
        rope_tail_in_place(
            &mut expected_output,
            HEADS,
            HEAD_DIM,
            1,
            oracle_rope,
            RopeDirection::Inverse,
        )
        .unwrap();

        let queries = offset_f32(&ctx, &query_values, vec![HEAD_DIM as u64, HEADS as u64]);
        let kv0_tensor = offset_f32(&ctx, &kv0, vec![HEAD_DIM as u64]);
        let kv1_tensor = offset_f32(&ctx, &kv1, vec![HEAD_DIM as u64]);
        let sink_tensor = offset_f32(&ctx, &sinks, vec![HEADS as u64]);
        let raw_cache =
            MetalTensor::zeros_f16(&ctx, vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64])
                .unwrap();
        let output = offset_f32(
            &ctx,
            &vec![0.0; HEADS * HEAD_DIM],
            vec![HEAD_DIM as u64, HEADS as u64],
        );
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_scatter_offset_f32_to_f16(&ctx, &encoder, &kv0_tensor, &raw_cache, 0, HEAD_DIM)
            .unwrap();
        encode_ds4_rope_tail_adjacent_in_place(&ctx, &encoder, &queries, 1, rope, false).unwrap();
        encode_ds4_rope_tail_adjacent_in_place(&ctx, &encoder, &kv1_tensor, 1, rope, false)
            .unwrap();
        encode_scatter_offset_f32_to_f16(
            &ctx,
            &encoder,
            &kv1_tensor,
            &raw_cache,
            HEAD_DIM,
            HEAD_DIM,
        )
        .unwrap();
        encode_local_sink_attention_f16(
            &ctx,
            &encoder,
            &queries,
            &raw_cache,
            &sink_tensor,
            &output,
            1,
            config,
        )
        .unwrap();
        encode_ds4_rope_tail_adjacent_in_place(&ctx, &encoder, &output, 1, rope, true).unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "command failed: {:?}",
            command.error()
        );

        assert_close(
            "forward adjacent RoPE",
            &read_f32(&queries),
            &expected_queries,
            2e-6,
        );
        assert_close(
            "forward KV RoPE",
            &read_f32(&kv1_tensor),
            &expected_kv1,
            2e-6,
        );
        assert_close(
            "continuing local attention",
            &read_f32(&output),
            &expected_output,
            2e-5,
        );
    }

    #[test]
    fn compressor_frontier_projects_ape_into_the_position_one_lane() {
        let Some(ctx) = metal_context() else {
            return;
        };
        const HIDDEN: usize = 3;
        const HEAD_DIM: usize = 2;
        const WIDTH: usize = HEAD_DIM * 2;
        let frontier = DeepSeekV4CompressorFrontier::new(
            &ctx,
            4,
            HEAD_DIM,
            DeepSeekV4CompressorPublication::Attention,
        )
        .unwrap();
        let input_values = [0.7, -0.4, 1.1];
        let kv_weights = (0..HIDDEN * WIDTH)
            .map(|index| (index as f32 - 4.0) * 0.07)
            .collect::<Vec<_>>();
        let score_weights = (0..HIDDEN * WIDTH)
            .map(|index| (5.0 - index as f32) * 0.043)
            .collect::<Vec<_>>();
        let ape_values = (0..4 * WIDTH)
            .map(|index| (index as f32 - 6.0) * 0.019)
            .collect::<Vec<_>>();
        let expected_kv = mat_vec(&kv_weights, HIDDEN, WIDTH, &input_values).unwrap();
        let expected_score = mat_vec(&score_weights, HIDDEN, WIDTH, &input_values)
            .unwrap()
            .into_iter()
            .zip(&ape_values[WIDTH..2 * WIDTH])
            .map(|(score, ape)| score + ape)
            .collect::<Vec<_>>();

        let input = offset_f32(&ctx, &input_values, vec![HIDDEN as u64]);
        let kv_weight = offset_f32(&ctx, &kv_weights, vec![HIDDEN as u64, WIDTH as u64]);
        let score_weight = offset_f32(&ctx, &score_weights, vec![HIDDEN as u64, WIDTH as u64]);
        let ape = offset_f32(&ctx, &ape_values, vec![WIDTH as u64, 4]);
        let norm = offset_f32(&ctx, &[1.0; HEAD_DIM], vec![HEAD_DIM as u64]);
        let rope = DeepSeekV4RopeParameters {
            rotary_dim: HEAD_DIM,
            theta: 10_000.0,
            scaling_factor: 1.0,
            original_context_length: 0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        frontier
            .encode(
                &ctx,
                &encoder,
                &input,
                &kv_weight,
                &score_weight,
                &ape,
                &norm,
                1,
                HIDDEN,
                rope,
                1e-5,
            )
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "command failed: {:?}",
            command.error()
        );

        let kv_state = read_f32(&frontier.kv_state);
        let score_state = read_f32(&frontier.score_state);
        let row_start = 5 * WIDTH;
        assert_close(
            "position-one compressor KV",
            &kv_state[row_start..row_start + WIDTH],
            &expected_kv,
            2e-5,
        );
        assert_close(
            "position-one compressor score plus APE",
            &score_state[row_start..row_start + WIDTH],
            &expected_score,
            2e-5,
        );
        assert!(
            score_state[..row_start]
                .iter()
                .chain(&score_state[row_start + WIDTH..])
                .all(|value| *value == f32::NEG_INFINITY)
        );
    }

    #[test]
    fn ratio4_frontier_publishes_rolls_and_continues_at_second_boundary() {
        let Some(ctx) = metal_context() else {
            return;
        };
        const HIDDEN: usize = 3;
        const HEAD_DIM: usize = 4;
        const WIDTH: usize = HEAD_DIM * 2;
        let rms_eps = 1.0e-5;
        let frontier = DeepSeekV4CompressorFrontier::new(
            &ctx,
            4,
            HEAD_DIM,
            DeepSeekV4CompressorPublication::Attention,
        )
        .unwrap();
        let kv_weights = (0..HIDDEN * WIDTH)
            .map(|index| ((index * 7 + 3) % 19) as f32 * 0.041 - 0.37)
            .collect::<Vec<_>>();
        let score_weights = (0..HIDDEN * WIDTH)
            .map(|index| ((index * 11 + 5) % 23) as f32 * 0.033 - 0.31)
            .collect::<Vec<_>>();
        let ape_values = (0..4 * WIDTH)
            .map(|index| ((index * 5 + 2) % 17) as f32 * 0.027 - 0.19)
            .collect::<Vec<_>>();
        let norm_values = [0.71, 1.13, 0.58, 0.92];
        let input_values = (0..8)
            .map(|position| {
                (0..HIDDEN)
                    .map(|dimension| {
                        (position as f32 - 2.4) * 0.17
                            + (dimension as f32 - 0.8) * 0.23
                            + if (position + dimension).is_multiple_of(2) {
                                0.11
                            } else {
                                -0.07
                            }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let kv_weight = offset_f32(&ctx, &kv_weights, vec![HIDDEN as u64, WIDTH as u64]);
        let score_weight = offset_f32(&ctx, &score_weights, vec![HIDDEN as u64, WIDTH as u64]);
        let ape = offset_f32(&ctx, &ape_values, vec![WIDTH as u64, 4]);
        let norm = offset_f32(&ctx, &norm_values, vec![HEAD_DIM as u64]);
        let inputs = input_values
            .iter()
            .map(|values| offset_f32(&ctx, values, vec![HIDDEN as u64]))
            .collect::<Vec<_>>();
        let rope = DeepSeekV4RopeParameters {
            rotary_dim: 2,
            theta: 10_000.0,
            scaling_factor: 1.0,
            original_context_length: 0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let oracle_rope = RopeParameters::local(2, rope.theta);
        let mut oracle = CompressorState::new(4, HEAD_DIM).unwrap();
        let mut expected_rows = Vec::new();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for (position, (input, values)) in inputs.iter().zip(&input_values).enumerate() {
            let projected_kv = mat_vec(&kv_weights, HIDDEN, WIDTH, values).unwrap();
            let projected_scores = mat_vec(&score_weights, HIDDEN, WIDTH, values).unwrap();
            if let Some(row) = oracle
                .push_projected(
                    position as u32,
                    &projected_kv,
                    &projected_scores,
                    &ape_values,
                    &norm_values,
                    rms_eps,
                    oracle_rope,
                )
                .unwrap()
            {
                expected_rows.extend(row.value);
            }
            frontier
                .encode(
                    &ctx,
                    &encoder,
                    input,
                    &kv_weight,
                    &score_weight,
                    &ape,
                    &norm,
                    position as u32,
                    HIDDEN,
                    rope,
                    rms_eps,
                )
                .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "command failed: {:?}",
            command.error()
        );

        assert_eq!(frontier.published_count(7), 2);
        assert_close(
            "ratio-4 rolled KV state",
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            "ratio-4 rolled score state",
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );
        let expected_rows = expected_rows
            .into_iter()
            .map(|value| half::f16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        let published = read_f16(&frontier.published);
        assert_close(
            "ratio-4 published rows",
            &published[..2 * HEAD_DIM],
            &expected_rows,
            1e-3,
        );
        assert!(
            published[2 * HEAD_DIM..].iter().all(|value| *value == 0.0),
            "unpublished rows must remain zero"
        );
    }

    #[test]
    fn ratio4_indexer_publication_matches_the_integrated_oracle() {
        let Some(ctx) = metal_context() else {
            return;
        };
        const HIDDEN: usize = 3;
        const HEAD_DIM: usize = 128;
        const WIDTH: usize = HEAD_DIM * 2;
        let rms_eps = 1.0e-5;
        let frontier = DeepSeekV4CompressorFrontier::new(
            &ctx,
            4,
            HEAD_DIM,
            DeepSeekV4CompressorPublication::IndexerHadamard,
        )
        .unwrap();
        let kv_weights = (0..HIDDEN * WIDTH)
            .map(|index| ((index * 13 + index / 5 + 3) % 47) as f32 * 0.011 - 0.24)
            .collect::<Vec<_>>();
        let score_weights = (0..HIDDEN * WIDTH)
            .map(|index| ((index * 17 + index / 7 + 1) % 53) as f32 * 0.009 - 0.21)
            .collect::<Vec<_>>();
        let ape_values = (0..4 * WIDTH)
            .map(|index| ((index * 19 + index / 11 + 4) % 59) as f32 * 0.007 - 0.18)
            .collect::<Vec<_>>();
        let norm_values = (0..HEAD_DIM)
            .map(|index| 0.53 + (index % 23) as f32 * 0.027)
            .collect::<Vec<_>>();
        let input_values = (0..8)
            .map(|position| {
                (0..HIDDEN)
                    .map(|dimension| {
                        (position as f32 - 3.1) * 0.13
                            + (dimension as f32 - 0.9) * 0.19
                            + if (position + dimension).is_multiple_of(3) {
                                0.08
                            } else {
                                -0.04
                            }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let kv_weight = offset_f32(&ctx, &kv_weights, vec![HIDDEN as u64, WIDTH as u64]);
        let score_weight = offset_f32(&ctx, &score_weights, vec![HIDDEN as u64, WIDTH as u64]);
        let ape = offset_f32(&ctx, &ape_values, vec![WIDTH as u64, 4]);
        let norm = offset_f32(&ctx, &norm_values, vec![HEAD_DIM as u64]);
        let inputs = input_values
            .iter()
            .map(|values| offset_f32(&ctx, values, vec![HIDDEN as u64]))
            .collect::<Vec<_>>();
        let rope = DeepSeekV4RopeParameters {
            rotary_dim: 64,
            theta: 160_000.0,
            scaling_factor: 1.0,
            original_context_length: 0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let oracle_rope = RopeParameters::local(rope.rotary_dim, rope.theta);
        let mut oracle = CompressorState::new(4, HEAD_DIM).unwrap();
        let mut expected_rows = Vec::new();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for (position, (input, values)) in inputs.iter().zip(&input_values).enumerate() {
            let projected_kv = mat_vec(&kv_weights, HIDDEN, WIDTH, values).unwrap();
            let projected_scores = mat_vec(&score_weights, HIDDEN, WIDTH, values).unwrap();
            if let Some(mut row) = oracle
                .push_projected(
                    position as u32,
                    &projected_kv,
                    &projected_scores,
                    &ape_values,
                    &norm_values,
                    rms_eps,
                    oracle_rope,
                )
                .unwrap()
            {
                hadamard_128_in_place(&mut row.value).unwrap();
                expected_rows.extend(row.value);
            }
            frontier
                .encode(
                    &ctx,
                    &encoder,
                    input,
                    &kv_weight,
                    &score_weight,
                    &ape,
                    &norm,
                    position as u32,
                    HIDDEN,
                    rope,
                    rms_eps,
                )
                .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "command failed: {:?}",
            command.error()
        );

        let expected_rows = expected_rows
            .into_iter()
            .map(|value| half::f16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        let published = read_f16(&frontier.published);
        assert_close(
            "integrated ratio-4 indexer publication",
            &published[..2 * HEAD_DIM],
            &expected_rows,
            2e-3,
        );
        assert!(
            published[2 * HEAD_DIM..].iter().all(|value| *value == 0.0),
            "unpublished index rows must remain zero"
        );
    }

    #[test]
    fn ratio128_attention_publications_match_the_integrated_oracle() {
        let Some(ctx) = metal_context() else {
            return;
        };
        const HIDDEN: usize = 3;
        const HEAD_DIM: usize = 512;
        const RATIO: usize = 128;
        let rms_eps = 1.0e-5;
        let frontier = DeepSeekV4CompressorFrontier::new(
            &ctx,
            RATIO,
            HEAD_DIM,
            DeepSeekV4CompressorPublication::Attention,
        )
        .unwrap();
        let kv_weights = (0..HIDDEN * HEAD_DIM)
            .map(|index| ((index * 13 + index / 7 + 5) % 61) as f32 * 0.008 - 0.23)
            .collect::<Vec<_>>();
        let score_weights = (0..HIDDEN * HEAD_DIM)
            .map(|index| ((index * 17 + index / 11 + 3) % 67) as f32 * 0.007 - 0.21)
            .collect::<Vec<_>>();
        let ape_values = (0..RATIO * HEAD_DIM)
            .map(|index| ((index * 19 + index / 13 + 1) % 71) as f32 * 0.006 - 0.19)
            .collect::<Vec<_>>();
        let norm_values = (0..HEAD_DIM)
            .map(|index| 0.49 + (index % 29) as f32 * 0.021)
            .collect::<Vec<_>>();
        let input_values = (0..4 * RATIO)
            .map(|position| {
                (0..HIDDEN)
                    .map(|dimension| {
                        (position as f32 - 61.0) * 0.004
                            + (dimension as f32 - 0.7) * 0.17
                            + if (position + dimension).is_multiple_of(5) {
                                0.06
                            } else {
                                -0.03
                            }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let kv_weight = offset_f32(&ctx, &kv_weights, vec![HIDDEN as u64, HEAD_DIM as u64]);
        let score_weight = offset_f32(&ctx, &score_weights, vec![HIDDEN as u64, HEAD_DIM as u64]);
        let ape = offset_f32(&ctx, &ape_values, vec![HEAD_DIM as u64, RATIO as u64]);
        let norm = offset_f32(&ctx, &norm_values, vec![HEAD_DIM as u64]);
        let inputs = input_values
            .iter()
            .map(|values| offset_f32(&ctx, values, vec![HIDDEN as u64]))
            .collect::<Vec<_>>();
        let rope = DeepSeekV4RopeParameters {
            rotary_dim: 64,
            theta: 160_000.0,
            scaling_factor: 16.0,
            original_context_length: 65_536,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let oracle_rope = RopeParameters::yarn(
            rope.rotary_dim,
            rope.theta,
            rope.scaling_factor,
            rope.original_context_length,
            rope.beta_fast,
            rope.beta_slow,
        );
        let mut oracle = CompressorState::new(RATIO, HEAD_DIM).unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for position in 0..RATIO - 1 {
            let values = &input_values[position];
            let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, values).unwrap();
            let projected_scores = mat_vec(&score_weights, HIDDEN, HEAD_DIM, values).unwrap();
            let emitted = oracle
                .push_projected(
                    position as u32,
                    &projected_kv,
                    &projected_scores,
                    &ape_values,
                    &norm_values,
                    rms_eps,
                    oracle_rope,
                )
                .unwrap();
            assert!(emitted.is_none());
            frontier
                .encode(
                    &ctx,
                    &encoder,
                    &inputs[position],
                    &kv_weight,
                    &score_weight,
                    &ape,
                    &norm,
                    position as u32,
                    HIDDEN,
                    rope,
                    rms_eps,
                )
                .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "pre-boundary command failed: {:?}",
            command.error()
        );
        assert_eq!(frontier.published_count(126), 0);
        assert!(
            read_f16(&frontier.published)
                .iter()
                .all(|value| *value == 0.0)
        );

        let position = RATIO - 1;
        let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
        let projected_scores =
            mat_vec(&score_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
        let expected = oracle
            .push_projected(
                position as u32,
                &projected_kv,
                &projected_scores,
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap()
            .expect("position 127 must publish");
        assert_eq!(expected.start_position, 0);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        frontier
            .encode(
                &ctx,
                &encoder,
                &inputs[position],
                &kv_weight,
                &score_weight,
                &ape,
                &norm,
                position as u32,
                HIDDEN,
                rope,
                rms_eps,
            )
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "boundary command failed: {:?}",
            command.error()
        );

        assert_eq!(frontier.published_count(127), 1);
        assert_close(
            "ratio-128 KV state",
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            "ratio-128 score state",
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );
        let expected_first = expected
            .value
            .into_iter()
            .map(|value| half::f16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        let published = read_f16(&frontier.published);
        assert_close(
            "ratio-128 published row 0",
            &published[..HEAD_DIM],
            &expected_first,
            1e-3,
        );
        assert!(published[HEAD_DIM..].iter().all(|value| *value == 0.0));

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for position in RATIO..2 * RATIO - 1 {
            let values = &input_values[position];
            let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, values).unwrap();
            let projected_scores = mat_vec(&score_weights, HIDDEN, HEAD_DIM, values).unwrap();
            let emitted = oracle
                .push_projected(
                    position as u32,
                    &projected_kv,
                    &projected_scores,
                    &ape_values,
                    &norm_values,
                    rms_eps,
                    oracle_rope,
                )
                .unwrap();
            assert!(emitted.is_none());
            frontier
                .encode(
                    &ctx,
                    &encoder,
                    &inputs[position],
                    &kv_weight,
                    &score_weight,
                    &ape,
                    &norm,
                    position as u32,
                    HIDDEN,
                    rope,
                    rms_eps,
                )
                .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "second pre-boundary command failed: {:?}",
            command.error()
        );
        assert_eq!(frontier.published_count(254), 1);
        let published = read_f16(&frontier.published);
        assert_close(
            "ratio-128 retained row 0",
            &published[..HEAD_DIM],
            &expected_first,
            1e-3,
        );
        assert!(published[HEAD_DIM..].iter().all(|value| *value == 0.0));

        let position = 2 * RATIO - 1;
        let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
        let projected_scores =
            mat_vec(&score_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
        let expected_second = oracle
            .push_projected(
                position as u32,
                &projected_kv,
                &projected_scores,
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap()
            .expect("position 255 must publish");
        assert_eq!(expected_second.start_position, 128);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        frontier
            .encode(
                &ctx,
                &encoder,
                &inputs[position],
                &kv_weight,
                &score_weight,
                &ape,
                &norm,
                position as u32,
                HIDDEN,
                rope,
                rms_eps,
            )
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "second boundary command failed: {:?}",
            command.error()
        );

        assert_eq!(frontier.published_count(255), 2);
        assert_close(
            "ratio-128 second-boundary KV state",
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            "ratio-128 second-boundary score state",
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );
        let expected_second = expected_second
            .value
            .into_iter()
            .map(|value| half::f16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        let published = read_f16(&frontier.published);
        assert_close(
            "ratio-128 retained row 0 after second publication",
            &published[..HEAD_DIM],
            &expected_first,
            1e-3,
        );
        assert_close(
            "ratio-128 published row 1",
            &published[HEAD_DIM..2 * HEAD_DIM],
            &expected_second,
            1e-3,
        );
        assert!(published[2 * HEAD_DIM..].iter().all(|value| *value == 0.0));

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for position in 2 * RATIO..3 * RATIO - 1 {
            let values = &input_values[position];
            let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, values).unwrap();
            let projected_scores = mat_vec(&score_weights, HIDDEN, HEAD_DIM, values).unwrap();
            let emitted = oracle
                .push_projected(
                    position as u32,
                    &projected_kv,
                    &projected_scores,
                    &ape_values,
                    &norm_values,
                    rms_eps,
                    oracle_rope,
                )
                .unwrap();
            assert!(emitted.is_none());
            frontier
                .encode(
                    &ctx,
                    &encoder,
                    &inputs[position],
                    &kv_weight,
                    &score_weight,
                    &ape,
                    &norm,
                    position as u32,
                    HIDDEN,
                    rope,
                    rms_eps,
                )
                .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "third pre-boundary command failed: {:?}",
            command.error()
        );
        assert_eq!(frontier.published_count(382), 2);
        let published = read_f16(&frontier.published);
        assert_close(
            "ratio-128 row 0 before third publication",
            &published[..HEAD_DIM],
            &expected_first,
            1e-3,
        );
        assert_close(
            "ratio-128 row 1 before third publication",
            &published[HEAD_DIM..2 * HEAD_DIM],
            &expected_second,
            1e-3,
        );
        assert!(published[2 * HEAD_DIM..].iter().all(|value| *value == 0.0));

        let position = 3 * RATIO - 1;
        let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
        let projected_scores =
            mat_vec(&score_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
        let expected_third = oracle
            .push_projected(
                position as u32,
                &projected_kv,
                &projected_scores,
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap()
            .expect("position 383 must publish");
        assert_eq!(expected_third.start_position, 256);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        frontier
            .encode(
                &ctx,
                &encoder,
                &inputs[position],
                &kv_weight,
                &score_weight,
                &ape,
                &norm,
                position as u32,
                HIDDEN,
                rope,
                rms_eps,
            )
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "third boundary command failed: {:?}",
            command.error()
        );

        assert_eq!(frontier.published_count(383), 3);
        assert_close(
            "ratio-128 third-boundary KV state",
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            "ratio-128 third-boundary score state",
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );
        let expected_third = expected_third
            .value
            .into_iter()
            .map(|value| half::f16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        let published = read_f16(&frontier.published);
        assert_close(
            "ratio-128 retained row 0 after third publication",
            &published[..HEAD_DIM],
            &expected_first,
            1e-3,
        );
        assert_close(
            "ratio-128 retained row 1 after third publication",
            &published[HEAD_DIM..2 * HEAD_DIM],
            &expected_second,
            1e-3,
        );
        assert_close(
            "ratio-128 published row 2",
            &published[2 * HEAD_DIM..3 * HEAD_DIM],
            &expected_third,
            1e-3,
        );
        assert!(published[3 * HEAD_DIM..].iter().all(|value| *value == 0.0));

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for position in 3 * RATIO..4 * RATIO - 1 {
            let values = &input_values[position];
            let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, values).unwrap();
            let projected_scores = mat_vec(&score_weights, HIDDEN, HEAD_DIM, values).unwrap();
            let emitted = oracle
                .push_projected(
                    position as u32,
                    &projected_kv,
                    &projected_scores,
                    &ape_values,
                    &norm_values,
                    rms_eps,
                    oracle_rope,
                )
                .unwrap();
            assert!(emitted.is_none());
            frontier
                .encode(
                    &ctx,
                    &encoder,
                    &inputs[position],
                    &kv_weight,
                    &score_weight,
                    &ape,
                    &norm,
                    position as u32,
                    HIDDEN,
                    rope,
                    rms_eps,
                )
                .unwrap();
        }
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "fourth pre-boundary command failed: {:?}",
            command.error()
        );
        assert_eq!(frontier.published_count(510), 3);
        let published = read_f16(&frontier.published);
        assert_close(
            "ratio-128 row 0 before fourth publication",
            &published[..HEAD_DIM],
            &expected_first,
            1e-3,
        );
        assert_close(
            "ratio-128 row 1 before fourth publication",
            &published[HEAD_DIM..2 * HEAD_DIM],
            &expected_second,
            1e-3,
        );
        assert_close(
            "ratio-128 row 2 before fourth publication",
            &published[2 * HEAD_DIM..3 * HEAD_DIM],
            &expected_third,
            1e-3,
        );
        assert!(published[3 * HEAD_DIM..].iter().all(|value| *value == 0.0));

        let position = 4 * RATIO - 1;
        let projected_kv = mat_vec(&kv_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
        let projected_scores =
            mat_vec(&score_weights, HIDDEN, HEAD_DIM, &input_values[position]).unwrap();
        let expected_fourth = oracle
            .push_projected(
                position as u32,
                &projected_kv,
                &projected_scores,
                &ape_values,
                &norm_values,
                rms_eps,
                oracle_rope,
            )
            .unwrap()
            .expect("position 511 must publish");
        assert_eq!(expected_fourth.start_position, 384);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        frontier
            .encode(
                &ctx,
                &encoder,
                &inputs[position],
                &kv_weight,
                &score_weight,
                &ape,
                &norm,
                position as u32,
                HIDDEN,
                rope,
                rms_eps,
            )
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "fourth boundary command failed: {:?}",
            command.error()
        );

        assert_eq!(frontier.published_count(511), 4);
        assert_close(
            "ratio-128 fourth-boundary KV state",
            &read_f32(&frontier.kv_state),
            oracle.kv_state(),
            4e-5,
        );
        assert_close(
            "ratio-128 fourth-boundary score state",
            &read_f32(&frontier.score_state),
            oracle.score_state(),
            4e-5,
        );
        let expected_fourth = expected_fourth
            .value
            .into_iter()
            .map(|value| half::f16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        let published = read_f16(&frontier.published);
        assert_close(
            "ratio-128 retained row 0 after fourth publication",
            &published[..HEAD_DIM],
            &expected_first,
            1e-3,
        );
        assert_close(
            "ratio-128 retained row 1 after fourth publication",
            &published[HEAD_DIM..2 * HEAD_DIM],
            &expected_second,
            1e-3,
        );
        assert_close(
            "ratio-128 retained row 2 after fourth publication",
            &published[2 * HEAD_DIM..3 * HEAD_DIM],
            &expected_third,
            1e-3,
        );
        assert_close(
            "ratio-128 published row 3",
            &published[3 * HEAD_DIM..4 * HEAD_DIM],
            &expected_fourth,
            1e-3,
        );
        assert!(published[4 * HEAD_DIM..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn hadamard_128_matches_the_indexer_oracle() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let values = (0usize..128)
            .map(|index| {
                ((index * 17 + index / 3 + 5) % 61) as f32 * 0.031 - 0.83
                    + if index.is_multiple_of(7) { 0.19 } else { 0.0 }
            })
            .collect::<Vec<_>>();
        let mut expected = values.clone();
        hadamard_128_in_place(&mut expected).unwrap();
        let tensor = offset_f32(&ctx, &values, vec![128]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_hadamard_128_in_place(&ctx, &encoder, &tensor).unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "command failed: {:?}",
            command.error()
        );
        assert_close("Hadamard-128", &read_f32(&tensor), &expected, 2e-6);
    }

    #[test]
    fn dense_compressed_attention_includes_the_newest_published_row() {
        let Some(ctx) = metal_context() else {
            return;
        };
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 128;
        let config = DeepSeekV4PositionZeroAttentionConfig {
            hidden_size: 1,
            q_lora_rank: 1,
            head_count: HEADS,
            head_dim: HEAD_DIM,
            rotary_dim: 64,
            group_count: 1,
            output_rank: 1,
        };
        let queries = (0..HEADS * HEAD_DIM)
            .map(|index| {
                let head = index / HEAD_DIM;
                let dimension = index % HEAD_DIM;
                if head == 0 {
                    0.09 + (dimension % 13) as f32 * 0.003
                } else {
                    -0.07 + (dimension % 11) as f32 * 0.002
                }
            })
            .collect::<Vec<_>>();
        let raw_rows = (0..4)
            .flat_map(|row| {
                (0..HEAD_DIM).map(move |dimension| {
                    -0.24 + row as f32 * 0.071 + (dimension % 9) as f32 * 0.006
                })
            })
            .collect::<Vec<_>>();
        let compressed_rows = (0..4)
            .flat_map(|row| {
                (0..HEAD_DIM).map(move |dimension| match row {
                    0 => 0.52 - (dimension % 17) as f32 * 0.004,
                    1 => -0.41 + (dimension % 19) as f32 * 0.005,
                    2 => 0.33 - (dimension % 23) as f32 * 0.006,
                    _ => -0.28 + (dimension % 29) as f32 * 0.004,
                })
            })
            .collect::<Vec<_>>();
        let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();
        let raw_rows = raw_rows.into_iter().map(round_f16).collect::<Vec<_>>();
        let compressed_rows = compressed_rows
            .into_iter()
            .map(round_f16)
            .collect::<Vec<_>>();
        let sinks = [-0.43, 0.18];
        let expected = shared_kv_attention(
            &queries,
            HEADS,
            HEAD_DIM,
            &raw_rows,
            &compressed_rows,
            None,
            &sinks,
        )
        .unwrap();
        let prior_rows = shared_kv_attention(
            &queries,
            HEADS,
            HEAD_DIM,
            &raw_rows,
            &compressed_rows[..3 * HEAD_DIM],
            None,
            &sinks,
        )
        .unwrap();
        let local_only =
            shared_kv_attention(&queries, HEADS, HEAD_DIM, &raw_rows, &[], None, &sinks).unwrap();
        assert!(
            expected
                .iter()
                .zip(&local_only)
                .any(|(dense, local)| (dense - local).abs() > 1e-2),
            "fixture must distinguish dense CSA from local-only attention"
        );
        assert!(
            expected
                .iter()
                .zip(&prior_rows)
                .any(|(four_rows, prior)| (four_rows - prior).abs() > 1e-2),
            "fixture must make the newest compressed row materially visible"
        );

        let query_tensor = offset_f32(&ctx, &queries, vec![HEAD_DIM as u64, HEADS as u64]);
        let sink_tensor = offset_f32(&ctx, &sinks, vec![HEADS as u64]);
        let raw_cache =
            MetalTensor::zeros_f16(&ctx, vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64])
                .unwrap();
        let compressed_cache = MetalTensor::zeros_f16(
            &ctx,
            vec![HEAD_DIM as u64, DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS as u64],
        )
        .unwrap();
        let raw_sources = raw_rows
            .chunks_exact(HEAD_DIM)
            .map(|row| offset_f32(&ctx, row, vec![HEAD_DIM as u64]))
            .collect::<Vec<_>>();
        let compressed_source = offset_f32(&ctx, &compressed_rows, vec![HEAD_DIM as u64, 4]);
        let output = offset_f32(
            &ctx,
            &vec![0.0; HEADS * HEAD_DIM],
            vec![HEAD_DIM as u64, HEADS as u64],
        );
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        for (row, source) in raw_sources.iter().enumerate() {
            encode_scatter_offset_f32_to_f16(
                &ctx,
                &encoder,
                source,
                &raw_cache,
                row * HEAD_DIM,
                HEAD_DIM,
            )
            .unwrap();
        }
        encode_scatter_offset_f32_to_f16(
            &ctx,
            &encoder,
            &compressed_source,
            &compressed_cache,
            0,
            4 * HEAD_DIM,
        )
        .unwrap();
        encode_dense_sink_attention_f16(
            &ctx,
            &encoder,
            &query_tensor,
            &raw_cache,
            Some(DeepSeekV4PublishedRows {
                cache: &compressed_cache,
                count: 4,
            }),
            &sink_tensor,
            &output,
            3,
            config,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "command failed: {:?}",
            command.error()
        );
        assert_close(
            "four-row dense compressed attention",
            &read_f32(&output),
            &expected,
            4e-5,
        );
    }

    #[test]
    fn dense_attention_matches_wrapped_promoted_hca_geometries() {
        let Some(ctx) = metal_context() else {
            return;
        };
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 512;
        const COMPRESSED_ROWS: usize = 128;
        let config = DeepSeekV4PositionZeroAttentionConfig {
            hidden_size: 1,
            q_lora_rank: 1,
            head_count: HEADS,
            head_dim: HEAD_DIM,
            rotary_dim: 64,
            group_count: 1,
            output_rank: 1,
        };
        let sinks = [-0.31, 0.22];
        let round_f16 = |value: f32| half::f16::from_f32(value).to_f32();

        for (position, expected_hca_count, expected_csa_count) in [
            (382usize, 2usize, 95usize),
            (383, 3, 96),
            (384, 3, 96),
            (510, 3, 127),
            (511, 4, 128),
            (512, 4, 128),
        ] {
            let raw_start = position + 1 - DEEPSEEK_V4_LOCAL_WINDOW;
            let mut raw_rows = Vec::with_capacity(DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM);
            let mut raw_ring = vec![0.0; DEEPSEEK_V4_LOCAL_WINDOW * HEAD_DIM];
            for logical_position in raw_start..=position {
                let row = (0..HEAD_DIM)
                    .map(|dimension| {
                        let tag =
                            (logical_position * 29 + dimension * 11 + logical_position / 5) % 127;
                        round_f16(
                            (tag as f32 - 63.0) * 0.0025
                                + if (logical_position + dimension).is_multiple_of(31) {
                                    0.037
                                } else {
                                    -0.009
                                },
                        )
                    })
                    .collect::<Vec<_>>();
                raw_rows.extend_from_slice(&row);
                let slot = logical_position % DEEPSEEK_V4_LOCAL_WINDOW;
                raw_ring[slot * HEAD_DIM..(slot + 1) * HEAD_DIM].copy_from_slice(&row);
            }
            let compressed_rows = (0..COMPRESSED_ROWS)
                .flat_map(|row| {
                    (0..HEAD_DIM).map(move |dimension| {
                        let tag = (row * 37 + dimension * 7 + row / 3) % 113;
                        round_f16(
                            (tag as f32 - 56.0) * 0.0031
                                + if (row + dimension).is_multiple_of(23) {
                                    0.041
                                } else {
                                    -0.013
                                },
                        )
                    })
                })
                .collect::<Vec<_>>();
            let queries = (0..HEADS * HEAD_DIM)
                .map(|index| {
                    let head = index / HEAD_DIM;
                    let dimension = index % HEAD_DIM;
                    let tag = (position * 13 + head * 17 + dimension * 5) % 97;
                    (tag as f32 - 48.0) * 0.0027
                })
                .collect::<Vec<_>>();
            let hca_count = (position + 1) / 128;
            let csa_count = (position + 1) / 4;
            assert_eq!(hca_count, expected_hca_count);
            assert_eq!(csa_count, expected_csa_count);
            let expected_hca = shared_kv_attention(
                &queries,
                HEADS,
                HEAD_DIM,
                &raw_rows,
                &compressed_rows[..hca_count * HEAD_DIM],
                None,
                &sinks,
            )
            .unwrap();
            let expected_csa = shared_kv_attention(
                &queries,
                HEADS,
                HEAD_DIM,
                &raw_rows,
                &compressed_rows[..csa_count * HEAD_DIM],
                None,
                &sinks,
            )
            .unwrap();
            assert!(
                expected_hca
                    .iter()
                    .zip(&expected_csa)
                    .any(|(hca, csa)| (hca - csa).abs() > 1e-3),
                "position {position} fixture must distinguish HCA and CSA row counts"
            );

            let query_tensor = offset_f32(&ctx, &queries, vec![HEAD_DIM as u64, HEADS as u64]);
            let sink_tensor = offset_f32(&ctx, &sinks, vec![HEADS as u64]);
            let raw_source = offset_f32(
                &ctx,
                &raw_ring,
                vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            );
            let compressed_source = offset_f32(
                &ctx,
                &compressed_rows,
                vec![HEAD_DIM as u64, COMPRESSED_ROWS as u64],
            );
            let raw_cache = MetalTensor::zeros_f16(
                &ctx,
                vec![HEAD_DIM as u64, DEEPSEEK_V4_LOCAL_WINDOW as u64],
            )
            .unwrap();
            let compressed_cache = MetalTensor::zeros_f16(
                &ctx,
                vec![HEAD_DIM as u64, DEEPSEEK_V4_COMPRESSED_HISTORY_ROWS as u64],
            )
            .unwrap();
            let hca_output = offset_f32(
                &ctx,
                &vec![0.0; HEADS * HEAD_DIM],
                vec![HEAD_DIM as u64, HEADS as u64],
            );
            let csa_output = offset_f32(
                &ctx,
                &vec![0.0; HEADS * HEAD_DIM],
                vec![HEAD_DIM as u64, HEADS as u64],
            );
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_scatter_offset_f32_to_f16(
                &ctx,
                &encoder,
                &raw_source,
                &raw_cache,
                0,
                raw_ring.len(),
            )
            .unwrap();
            encode_scatter_offset_f32_to_f16(
                &ctx,
                &encoder,
                &compressed_source,
                &compressed_cache,
                0,
                compressed_rows.len(),
            )
            .unwrap();
            encode_dense_sink_attention_f16(
                &ctx,
                &encoder,
                &query_tensor,
                &raw_cache,
                Some(DeepSeekV4PublishedRows {
                    cache: &compressed_cache,
                    count: hca_count,
                }),
                &sink_tensor,
                &hca_output,
                position as u32,
                config,
            )
            .unwrap();
            encode_dense_sink_attention_f16(
                &ctx,
                &encoder,
                &query_tensor,
                &raw_cache,
                Some(DeepSeekV4PublishedRows {
                    cache: &compressed_cache,
                    count: csa_count,
                }),
                &sink_tensor,
                &csa_output,
                position as u32,
                config,
            )
            .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(
                command.error().is_none(),
                "position {position} command failed: {:?}",
                command.error()
            );
            assert_close(
                &format!("position {position} wrapped HCA attention"),
                &read_f32(&hca_output),
                &expected_hca,
                7e-5,
            );
            assert_close(
                &format!("position {position} wrapped CSA attention"),
                &read_f32(&csa_output),
                &expected_csa,
                7e-5,
            );
        }
    }

    fn oracle_expert(
        normalized: &[f32],
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        hidden: usize,
        ffn: usize,
        clamp: f32,
    ) -> Vec<f32> {
        let gate = mat_vec(gate, hidden, ffn, normalized).expect("gate oracle");
        let up = mat_vec(up, hidden, ffn, normalized).expect("up oracle");
        let inner =
            crate::deepseek_v4_oracle::clamped_swiglu(&gate, &up, clamp).expect("SwiGLU oracle");
        mat_vec(down, ffn, hidden, &inner).expect("down oracle")
    }

    #[test]
    fn single_token_moe_learned_and_hash_match_oracles_with_exact_bank_slices() {
        let Some(ctx) = metal_context() else {
            return;
        };
        const H: usize = 3;
        const F: usize = 3;
        const E: usize = 3;
        const K: usize = 2;
        let config = DeepSeekV4MoeConfig {
            hidden_size: H,
            ffn_size: F,
            expert_count: E,
            top_k: K,
            routed_scale: 1.7,
        };
        let input_values = [1.3, -0.8, 0.45];
        let norm_values = [1.1, 0.65, 1.35];
        let router_values = [0.31, -0.27, 0.18, -0.22, 0.43, 0.09, 0.16, 0.08, -0.51];
        let gate_bank_values = (0..E * F * H)
            .map(|index| {
                let expert = index / (F * H);
                let row = (index / H) % F;
                let column = index % H;
                let lead = [2.7, -1.4, 1.1][row] * (expert as f32 + 1.0);
                if column == 0 {
                    lead
                } else {
                    (column as f32 - 1.5) * 0.23 + expert as f32 * 0.17
                }
            })
            .collect::<Vec<_>>();
        let up_bank_values = (0..E * F * H)
            .map(|index| {
                let expert = index / (F * H);
                let row = (index / H) % F;
                let column = index % H;
                let lead = [2.4, 2.1, -2.8][row] * (expert as f32 + 0.7);
                if column == 1 {
                    lead
                } else {
                    (row as f32 - column as f32) * 0.19 - expert as f32 * 0.11
                }
            })
            .collect::<Vec<_>>();
        let down_bank_values = (0..E * H * F)
            .map(|index| {
                let expert = index / (H * F);
                let row = (index / F) % H;
                let column = index % F;
                (expert as f32 + 1.0) * 0.37 + row as f32 * 0.21 - column as f32 * 0.16
            })
            .collect::<Vec<_>>();
        let shared_gate_values = [1.9, -0.2, 0.1, -1.1, 0.4, 0.2, 0.8, -0.7, 0.3];
        let shared_up_values = [-0.3, -2.2, 0.4, 0.2, 2.5, -0.1, 0.6, -1.8, 0.7];
        let shared_down_values = [0.7, -0.2, 0.4, -0.3, 0.8, 0.1, 0.2, -0.5, 0.9];
        let clamp = 0.55;
        let rms_eps = 1.0e-5;

        let input = offset_f32(&ctx, &input_values, vec![H as u64]);
        let norm = offset_f32(&ctx, &norm_values, vec![H as u64]);
        let router = offset_f32(&ctx, &router_values, vec![H as u64, E as u64]);
        let gate_bank = offset_f32(&ctx, &gate_bank_values, vec![H as u64, F as u64, E as u64]);
        let up_bank = offset_f32(&ctx, &up_bank_values, vec![H as u64, F as u64, E as u64]);
        let down_bank = offset_f32(&ctx, &down_bank_values, vec![F as u64, H as u64, E as u64]);
        let shared_gate = offset_f32(&ctx, &shared_gate_values, vec![H as u64, F as u64]);
        let shared_up = offset_f32(&ctx, &shared_up_values, vec![H as u64, F as u64]);
        let shared_down = offset_f32(&ctx, &shared_down_values, vec![F as u64, H as u64]);
        let normalized = rms_norm(&input_values, Some(&norm_values), rms_eps).unwrap();
        let logits = mat_vec(&router_values, H, E, &normalized).unwrap();
        let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(&logits).unwrap();
        let shared_gate_projection =
            mat_vec(&shared_gate_values, H, F, &normalized).expect("shared gate projection");
        let shared_up_projection =
            mat_vec(&shared_up_values, H, F, &normalized).expect("shared up projection");
        assert!(shared_gate_projection.iter().any(|&value| value > clamp));
        assert!(shared_up_projection.iter().any(|&value| value > clamp));
        assert!(shared_up_projection.iter().any(|&value| value < -clamp));
        let shared_expected = oracle_expert(
            &normalized,
            &shared_gate_values,
            &shared_up_values,
            &shared_down_values,
            H,
            F,
            clamp,
        );

        let routes = [
            (
                "learned",
                vec![0.14, -0.31, 0.47],
                vec![0i32, 1, 2, 0, 1, 2],
                0usize,
            ),
            ("hash", vec![0.0; E], vec![0i32, 1, 2, 0, 1, 2], 1usize),
        ];
        for (label, bias_values, map_values, token_id) in routes {
            let scratch = DeepSeekV4MoeScratch::new(&ctx, config).expect("MoE scratch");
            let command = ctx.queue.commandBuffer().expect("router command");
            let encoder = KernelEncoder::begin(&command);
            scratch
                .encode_router(&ctx, &encoder, &input, &norm, &router, rms_eps)
                .expect("encode router");
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{label} router command failed");
            assert_close(
                label,
                &read_f32(scratch.normalized_input()),
                &normalized,
                3e-5,
            );
            assert_close(label, &read_f32(scratch.logits()), &logits, 3e-5);

            let decision = if label == "learned" {
                let bias = offset_f32(&ctx, &bias_values, vec![E as u64]);
                scratch.route_learned(&bias).expect("learned route");
                crate::deepseek_v4_oracle::learned_route(
                    &scores,
                    &bias_values,
                    K,
                    config.routed_scale,
                )
                .unwrap()
            } else {
                let map = offset_i32(&ctx, &map_values, vec![K as u64, 3]);
                scratch.route_hash(token_id, &map).expect("hash route");
                let selected = map_values[token_id * K..token_id * K + K]
                    .iter()
                    .map(|&id| id as usize)
                    .collect::<Vec<_>>();
                crate::deepseek_v4_oracle::hash_route(&scores, &selected, config.routed_scale)
                    .unwrap()
            };
            assert_eq!(
                read_i32(scratch.expert_ids()),
                decision
                    .expert_ids
                    .iter()
                    .map(|&id| id as i32)
                    .collect::<Vec<_>>()
            );
            assert_close(label, &read_f32(scratch.weights()), &decision.weights, 1e-6);

            let mut expected_slots = Vec::new();
            for &expert in &decision.expert_ids {
                let gate = &gate_bank_values[expert * H * F..(expert + 1) * H * F];
                let up = &up_bank_values[expert * H * F..(expert + 1) * H * F];
                let down = &down_bank_values[expert * F * H..(expert + 1) * F * H];
                expected_slots.extend(oracle_expert(&normalized, gate, up, down, H, F, clamp));
            }
            let mut routed_expected = vec![0.0f32; H];
            for slot in 0..K {
                for dimension in 0..H {
                    routed_expected[dimension] +=
                        expected_slots[slot * H + dimension] * decision.weights[slot];
                }
            }
            let final_expected = routed_expected
                .iter()
                .zip(&shared_expected)
                .map(|(routed, shared)| routed + shared)
                .collect::<Vec<_>>();

            let command = ctx.queue.commandBuffer().expect("expert command");
            let encoder = KernelEncoder::begin(&command);
            scratch
                .encode_experts(
                    &ctx,
                    &encoder,
                    &gate_bank,
                    &up_bank,
                    &down_bank,
                    &shared_gate,
                    &shared_up,
                    &shared_down,
                    clamp,
                    clamp,
                )
                .expect("encode experts");
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{label} expert command failed");
            assert_close(
                label,
                &read_f32(scratch.expert_outputs()),
                &expected_slots,
                8e-5,
            );
            assert_close(
                label,
                &read_f32(scratch.routed_output()),
                &routed_expected,
                1e-4,
            );
            assert_close(
                label,
                &read_f32(scratch.shared_output()),
                &shared_expected,
                8e-5,
            );
            assert_close(
                label,
                &read_f32(scratch.final_output()),
                &final_expected,
                1e-4,
            );
        }
    }

    #[test]
    fn learned_moe_route_breaks_exact_score_bias_ties_by_expert_id() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let config = DeepSeekV4MoeConfig {
            hidden_size: 2,
            ffn_size: 2,
            expert_count: 4,
            top_k: 3,
            routed_scale: 1.0,
        };
        let scratch = DeepSeekV4MoeScratch::new(&ctx, config).unwrap();
        let input = offset_f32(&ctx, &[0.7, -0.2], vec![2]);
        let norm = offset_f32(&ctx, &[1.0, 1.0], vec![2]);
        let zero_router = offset_f32(&ctx, &[0.0; 8], vec![2, 4]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        scratch
            .encode_router(&ctx, &encoder, &input, &norm, &zero_router, 1e-5)
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        let tied_bias = offset_f32(&ctx, &[0.5, 0.5, 0.5, -0.2], vec![4]);
        scratch.route_learned(&tied_bias).unwrap();
        assert_eq!(read_i32(scratch.expert_ids()), vec![0, 1, 2]);
        let expected_scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(&[0.0; 4]).unwrap();
        let expected = crate::deepseek_v4_oracle::learned_route(
            &expected_scores,
            &[0.5, 0.5, 0.5, -0.2],
            3,
            1.0,
        )
        .unwrap();
        assert_close(
            "tie weights",
            &read_f32(scratch.weights()),
            &expected.weights,
            1e-6,
        );
    }

    #[test]
    fn report_is_exact_for_views_aliases_and_final_page_fallback() {
        let view = f32_desc("view", 64);
        let alias = f32_desc("alias", 64);
        let tail = f32_desc("tail", 128);
        let plan = plan_retained_storage(&[160], &[&view, &alias, &tail], 64, 128, 32)
            .expect("plan retained synthetic storage");
        validate_fallback_policy(&plan).expect("only final-page fallback");
        let report = report_for_plan(&plan).expect("report");

        assert_eq!(report.tensor_count, 3);
        assert_eq!(report.source_bytes, 96);
        assert_eq!(report.window_count, 1);
        assert_eq!(report.window_bytes, 64);
        assert_eq!(report.view_count, 1);
        assert_eq!(report.unique_view_bytes, 32);
        assert_eq!(report.logical_view_bytes, 64);
        assert_eq!(report.alias_count, 1);
        assert_eq!(report.alias_bytes, 32);
        assert_eq!(report.fallback_count, 1);
        assert_eq!(report.fallback_bytes, 32);
        assert_eq!(report.resident_bytes, 96);
        assert_eq!(report.required_alignment, 32);
    }

    #[test]
    fn non_final_page_fallback_is_rejected() {
        let misaligned = f32_desc("misaligned", 4);
        let plan = plan_retained_storage(&[128], &[&misaligned], 64, 128, 32)
            .expect("planner classifies misalignment");
        let error = validate_fallback_policy(&plan).expect_err("misalignment must fail closed");
        assert!(error.to_string().contains("BindingMisalignment"));
    }

    #[test]
    fn native_hyper_connections_match_oracle_with_offsets_and_asymmetric_streams() {
        let Some(ctx) = metal_context() else {
            return;
        };
        const H: usize = 7;
        const N: usize = H * DEEPSEEK_V4_CONNECTION_COUNT;
        let residual_values = (0..N)
            .map(|index| {
                let stream = index / H;
                let dimension = index % H;
                (stream as f32 - 1.25) * 0.73
                    + (dimension as f32 - 2.4) * (0.11 + stream as f32 * 0.07)
                    + if (index + stream).is_multiple_of(3) {
                        -0.19
                    } else {
                        0.13
                    }
            })
            .collect::<Vec<_>>();
        let function = (0..N * DEEPSEEK_V4_HC_PARAMETER_COUNT)
            .map(|index| ((index * 17 + index / 11) % 31) as f32 * 0.006 - 0.087)
            .collect::<Vec<_>>();
        let scale_values = [0.7, -0.4, 1.2];
        let base_values = (0..DEEPSEEK_V4_HC_PARAMETER_COUNT)
            .map(|index| ((index * 7 + 3) % 19) as f32 * 0.035 - 0.29)
            .collect::<Vec<_>>();
        let block_values = (0..H)
            .map(|index| (index as f32 - 2.2) * 0.31)
            .collect::<Vec<_>>();
        let head_function = (0..N * DEEPSEEK_V4_CONNECTION_COUNT)
            .map(|index| ((index * 13 + 5) % 23) as f32 * 0.009 - 0.091)
            .collect::<Vec<_>>();
        let head_scale_values = [-0.63];
        let head_base_values = [0.17, -0.31, 0.08, 0.27];
        let rms_eps = 1.0e-5;
        let hc_eps = 1.0e-6;

        let expected_pre = hyper_connection_pre(
            &residual_values,
            H,
            4,
            &function,
            &scale_values,
            &base_values,
            rms_eps,
            DEEPSEEK_V4_SINKHORN_ITERATIONS,
            hc_eps,
        )
        .expect("pre oracle");
        let expected_post = hyper_connection_post(
            &block_values,
            &residual_values,
            &expected_pre.controls,
            H,
            4,
        )
        .expect("post oracle");
        let expected_head = hyper_connection_head(
            &expected_post,
            H,
            4,
            &head_function,
            head_scale_values[0],
            &head_base_values,
            rms_eps,
            hc_eps,
        )
        .expect("head oracle");
        let expected_head_mixes = mat_vec(
            &head_function,
            N,
            4,
            &rms_norm(&expected_post, None, rms_eps).expect("head norm oracle"),
        )
        .expect("head mix oracle");
        let expected_head_gates = expected_head_mixes
            .iter()
            .zip(head_base_values)
            .map(|(&mix, base)| 1.0 / (1.0 + (-(mix * head_scale_values[0] + base)).exp()) + hc_eps)
            .collect::<Vec<_>>();

        let residual = offset_f32(&ctx, &residual_values, vec![H as u64, 4]);
        let function_tensor = offset_f32(&ctx, &function, vec![N as u64, 24]);
        let scale = offset_f32(&ctx, &scale_values, vec![3]);
        let base = offset_f32(&ctx, &base_values, vec![24]);
        let block = offset_f32(&ctx, &block_values, vec![H as u64]);
        let post_output = offset_f32(&ctx, &vec![0.0; N], vec![H as u64, 4]);
        let head_function_tensor = offset_f32(&ctx, &head_function, vec![N as u64, 4]);
        let head_scale = offset_f32(&ctx, &head_scale_values, vec![1]);
        let head_base = offset_f32(&ctx, &head_base_values, vec![4]);
        let head_output = offset_f32(&ctx, &vec![0.0; H], vec![H as u64]);
        let scratch = DeepSeekV4HyperConnectionScratch::new(&ctx, H).expect("HC scratch");
        assert_eq!(read_f32(&residual), residual_values);
        assert_eq!(read_f32(&scratch.ones), vec![1.0; N]);

        let command = ctx.queue.commandBuffer().expect("norm preflight buffer");
        let encoder = KernelEncoder::begin(&command);
        encode_rms_norm_mul_f32(
            &ctx,
            &encoder,
            &residual,
            &scratch.ones,
            &scratch.normalized,
            rms_eps,
        )
        .expect("norm preflight");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_close(
            "offset norm preflight",
            &read_f32(scratch.normalized()),
            &rms_norm(&residual_values, None, rms_eps).unwrap(),
            2e-5,
        );

        let command = ctx.queue.commandBuffer().expect("HC command buffer");
        let encoder = KernelEncoder::begin(&command);
        scratch
            .encode_pre(
                &ctx,
                &encoder,
                &residual,
                &function_tensor,
                &scale,
                &base,
                rms_eps,
                hc_eps,
            )
            .expect("encode pre");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "pre command failed: {:?}",
            command.error()
        );

        assert_close(
            "flattened norm",
            &read_f32(scratch.normalized()),
            &rms_norm(&residual_values, None, rms_eps).unwrap(),
            2e-5,
        );
        assert_close(
            "pre mixes",
            &read_f32(scratch.mixes()),
            &expected_pre.mixes,
            3e-5,
        );
        assert_close(
            "pre gates",
            &read_f32(scratch.pre_gates()),
            &expected_pre.controls.pre,
            2e-5,
        );
        assert_close(
            "post gates",
            &read_f32(scratch.post_gates()),
            &expected_pre.controls.post,
            2e-5,
        );
        assert_close(
            "combination source-major",
            &read_f32(scratch.combination()),
            &expected_pre.controls.combination,
            3e-5,
        );
        assert_close(
            "collapsed input",
            &read_f32(scratch.collapsed_input()),
            &expected_pre.input,
            3e-5,
        );

        let command = ctx.queue.commandBuffer().expect("post/head command buffer");
        let encoder = KernelEncoder::begin(&command);
        scratch
            .encode_post(&ctx, &encoder, &block, &residual, &post_output)
            .expect("encode post");
        scratch
            .encode_head(
                &ctx,
                &encoder,
                &post_output,
                &head_function_tensor,
                &head_scale,
                &head_base,
                &head_output,
                rms_eps,
                hc_eps,
            )
            .expect("encode head");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "post/head command failed: {:?}",
            command.error()
        );

        assert_close(
            "post residual",
            &read_f32(&post_output),
            &expected_post,
            4e-5,
        );
        assert_close(
            "head mixes",
            &read_f32(scratch.head_mixes()),
            &expected_head_mixes,
            3e-5,
        );
        assert_close(
            "head gates",
            &read_f32(scratch.head_gates()),
            &expected_head_gates,
            3e-5,
        );
        assert_close("final head", &read_f32(&head_output), &expected_head, 4e-5);

        let mut transposed_post = vec![0.0; N];
        for destination in 0..4 {
            for dimension in 0..H {
                let mut value = block_values[dimension] * expected_pre.controls.post[destination];
                for source in 0..4 {
                    value += expected_pre.controls.combination[destination * 4 + source]
                        * residual_values[source * H + dimension];
                }
                transposed_post[destination * H + dimension] = value;
            }
        }
        assert!(
            transposed_post
                .iter()
                .zip(&expected_post)
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "fixture must detect transposed combination axes"
        );
        let per_stream_norm = residual_values
            .chunks_exact(H)
            .flat_map(|stream| rms_norm(stream, None, rms_eps).unwrap())
            .collect::<Vec<_>>();
        let wrong_mixes = mat_vec(
            &function,
            N,
            DEEPSEEK_V4_HC_PARAMETER_COUNT,
            &per_stream_norm,
        )
        .unwrap();
        assert!(
            wrong_mixes
                .iter()
                .zip(&expected_pre.mixes)
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "fixture must distinguish flattened and per-stream RMSNorm"
        );
    }

    #[test]
    fn initial_repeat_and_first_pre_match_equal_stream_oracle() {
        let Some(ctx) = metal_context() else {
            return;
        };
        const H: usize = 9;
        const N: usize = H * 4;
        let embedding_values = (0..H)
            .map(|index| (index as f32 - 3.7) * 0.23 + if index % 2 == 0 { 0.14 } else { -0.09 })
            .collect::<Vec<_>>();
        let repeated = (0..4)
            .flat_map(|_| embedding_values.iter().copied())
            .collect::<Vec<_>>();
        let function = (0..N * 24)
            .map(|index| ((index * 5 + 1) % 29) as f32 * 0.004 - 0.052)
            .collect::<Vec<_>>();
        let scale_values = [0.41, 0.78, -0.57];
        let base_values = (0..24)
            .map(|index| index as f32 * 0.013 - 0.11)
            .collect::<Vec<_>>();
        let expected = hyper_connection_pre(
            &repeated,
            H,
            4,
            &function,
            &scale_values,
            &base_values,
            1e-5,
            20,
            1e-6,
        )
        .expect("equal stream oracle");

        let embedding = offset_f32(&ctx, &embedding_values, vec![H as u64]);
        let residual = offset_f32(&ctx, &vec![0.0; N], vec![H as u64, 4]);
        let function = offset_f32(&ctx, &function, vec![N as u64, 24]);
        let scale = offset_f32(&ctx, &scale_values, vec![3]);
        let base = offset_f32(&ctx, &base_values, vec![24]);
        let scratch = DeepSeekV4HyperConnectionScratch::new(&ctx, H).expect("HC scratch");
        let command = ctx.queue.commandBuffer().expect("repeat command buffer");
        let encoder = KernelEncoder::begin(&command);
        scratch
            .encode_initial_repeat(&ctx, &encoder, &embedding, &residual)
            .expect("repeat embedding");
        scratch
            .encode_pre(
                &ctx, &encoder, &residual, &function, &scale, &base, 1e-5, 1e-6,
            )
            .expect("first pre");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();

        assert_eq!(read_f32(&residual), repeated);
        assert_close(
            "equal-stream pre",
            &read_f32(scratch.pre_gates()),
            &expected.controls.pre,
            2e-5,
        );
        assert_close(
            "equal-stream combination",
            &read_f32(scratch.combination()),
            &expected.controls.combination,
            3e-5,
        );
        assert_close(
            "equal-stream collapse",
            &read_f32(scratch.collapsed_input()),
            &expected.input,
            3e-5,
        );
    }
}
