//! Zero-allocation Metal weight planning for Qwen3.8-Flash-Next.
//!
//! The released PLE table remains CPU row-addressed. Other weights use
//! retained, read-only GGUF views where page geometry permits; copied
//! final-partial-page fallbacks are explicit in the report. No Metal buffers
//! are created until a later admitted realization step.
//! Backing GGUF files must remain immutable while a plan, PLE binding, or
//! realized mmap view is in use.

use crate::gguf::{GgufError, GgufFile, GgufShardStamp};
use crate::metal::{
    MetalContext, MetalError, MetalGgufBacking, MetalMemoryAdmission, MetalMemorySignals,
    MetalTensor, MetalTensorProvenance, RetainedStorageDisposition, RetainedStorageFallback,
    RetainedStoragePlan, RetainedStorageWindow, evaluate_metal_memory_admission,
    host_page_size_bytes, plan_retained_storage,
};
use crate::qwen4exp::Qwen4ExpConfig;
use crate::qwen4exp_loader::{Qwen4ExpLoadError, Qwen4ExpModel};
use crate::qwen4exp_ple::{PleGatherError, PleIq4NlTable};
use crate::tensor::{GgmlType, TensorDesc};
use objc2_metal::{MTLBuffer, MTLDevice};
use std::collections::BTreeMap;

pub const QWEN4EXP_RELEASE_TENSOR_COUNT: usize = 1_224;
pub const QWEN4EXP_METAL_TENSOR_COUNT: usize = QWEN4EXP_RELEASE_TENSOR_COUNT - 1;
pub const QWEN4EXP_GGUF_BINDING_ALIGNMENT: usize = 32;
pub const QWEN4EXP_RETAINED_WINDOW_CEILING_BYTES: usize = 8 * 1024 * 1024 * 1024;
pub const QWEN4EXP_PLE_PHYSICAL_ROW_COUNT: u64 = 320_001_536;
pub const QWEN4EXP_PLE_SOURCE_BYTES: u64 = 28_800_138_240;

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpResidencyError {
    #[error(transparent)]
    Load(#[from] Qwen4ExpLoadError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Ple(#[from] PleGatherError),
    #[error("invalid Qwen3.8-Flash-Next residency contract: {0}")]
    Invalid(String),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen4ExpDtypeCensus {
    pub f32: usize,
    pub q8_0: usize,
    pub iq3_xxs: usize,
    pub iq4_nl: usize,
    pub bf16: usize,
    pub iq4_xs: usize,
    pub q6_k: usize,
}

impl Qwen4ExpDtypeCensus {
    pub const UD_Q3_K_XL: Self = Self {
        f32: 557,
        q8_0: 502,
        iq3_xxs: 94,
        iq4_nl: 44,
        bf16: 24,
        iq4_xs: 2,
        q6_k: 1,
    };

    pub fn total(self) -> usize {
        self.f32 + self.q8_0 + self.iq3_xxs + self.iq4_nl + self.bf16 + self.iq4_xs + self.q6_k
    }

    fn record(&mut self, dtype: GgmlType) -> Result<(), Qwen4ExpResidencyError> {
        let count = match dtype {
            GgmlType::F32 => &mut self.f32,
            GgmlType::Q8_0 => &mut self.q8_0,
            GgmlType::IQ3_XXS => &mut self.iq3_xxs,
            GgmlType::IQ4_NL => &mut self.iq4_nl,
            GgmlType::BF16 => &mut self.bf16,
            GgmlType::IQ4_XS => &mut self.iq4_xs,
            GgmlType::Q6_K => &mut self.q6_k,
            _ => {
                return invalid(format!(
                    "UD-Q3_K_XL census cannot record unsupported dtype {dtype:?}"
                ));
            }
        };
        *count = count
            .checked_add(1)
            .ok_or_else(|| Qwen4ExpResidencyError::Invalid("dtype census overflow".into()))?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen4ExpMetalWeightPlanReport {
    pub source_tensor_count: usize,
    pub metal_tensor_count: usize,
    pub cpu_ple_tensor_count: usize,
    pub source_bytes: u64,
    pub metal_source_bytes: u64,
    pub cpu_ple_source_bytes: u64,
    pub planned_window_count: usize,
    pub planned_window_bytes: u64,
    pub view_count: usize,
    pub fallback_count: usize,
    pub fallback_bytes: u64,
    pub ple_boundary_overlap_bytes: u64,
    pub page_size: usize,
    pub device_max_buffer_length: usize,
    pub planning_max_buffer_length: usize,
    pub dtype_census: Qwen4ExpDtypeCensus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen4ExpMetalWeightMemoryPlan {
    buffer_count: usize,
    logical_bytes: u64,
    priced_upper_bytes: u64,
}

impl Qwen4ExpMetalWeightMemoryPlan {
    pub fn buffer_count(&self) -> usize {
        self.buffer_count
    }

    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    pub fn priced_upper_bytes(&self) -> u64 {
        self.priced_upper_bytes
    }

    pub fn admission(&self, signals: MetalMemorySignals) -> MetalMemoryAdmission {
        evaluate_metal_memory_admission(self.priced_upper_bytes, 0, signals, true)
    }

    pub fn reconcile(
        &self,
        allocated_before: u64,
        allocated_after: u64,
    ) -> Result<u64, Qwen4ExpResidencyError> {
        let observed = allocated_after.saturating_sub(allocated_before);
        if observed > self.priced_upper_bytes {
            return invalid(format!(
                "observed Metal weight allocation {observed} exceeds priced upper bound {}",
                self.priced_upper_bytes
            ));
        }
        Ok(observed)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DescriptorFingerprint {
    name: String,
    shape: Vec<u64>,
    dtype: GgmlType,
    shard_idx: usize,
    data_offset: u64,
    n_bytes: u64,
}

impl From<&TensorDesc> for DescriptorFingerprint {
    fn from(desc: &TensorDesc) -> Self {
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

impl DescriptorFingerprint {
    fn matches(&self, desc: &TensorDesc) -> bool {
        self.name == desc.name
            && self.shape == desc.shape
            && self.dtype == desc.dtype
            && self.shard_idx == desc.shard_idx
            && self.data_offset == desc.data_offset
            && self.n_bytes == desc.n_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen4ExpPleSourcePlan {
    descriptor: DescriptorFingerprint,
    logical_row_count: u64,
    shard_stamp: GgufShardStamp,
}

impl Qwen4ExpPleSourcePlan {
    pub fn descriptor_name(&self) -> &str {
        &self.descriptor.name
    }

    pub fn shape(&self) -> &[u64] {
        &self.descriptor.shape
    }

    pub fn dtype(&self) -> GgmlType {
        self.descriptor.dtype
    }

    pub fn shard_idx(&self) -> usize {
        self.descriptor.shard_idx
    }

    pub fn data_offset(&self) -> u64 {
        self.descriptor.data_offset
    }

    pub fn n_bytes(&self) -> u64 {
        self.descriptor.n_bytes
    }

    pub fn logical_row_count(&self) -> u64 {
        self.logical_row_count
    }

    pub fn bind<'a>(
        &self,
        gguf: &'a GgufFile,
    ) -> Result<PleIq4NlTable<'a>, Qwen4ExpResidencyError> {
        let stamps = gguf.revalidate_retained_shard_stamps()?;
        if stamps.get(self.descriptor.shard_idx) != Some(&self.shard_stamp) {
            return invalid("PLE source shard identity changed after weight planning");
        }
        let desc = gguf.find(&self.descriptor.name).ok_or_else(|| {
            Qwen4ExpResidencyError::Invalid(format!(
                "planned PLE tensor {:?} is missing",
                self.descriptor.name
            ))
        })?;
        if !self.descriptor.matches(desc) {
            return invalid("PLE descriptor changed after weight planning");
        }
        Ok(PleIq4NlTable::new(
            desc,
            gguf.try_slice(desc)?,
            self.logical_row_count,
        )?)
    }
}

pub struct Qwen4ExpMetalWeightPlan {
    config: Qwen4ExpConfig,
    retained: RetainedStoragePlan,
    descriptors: Vec<DescriptorFingerprint>,
    ple_source: Qwen4ExpPleSourcePlan,
    report: Qwen4ExpMetalWeightPlanReport,
    memory: Qwen4ExpMetalWeightMemoryPlan,
    device_registry_id: u64,
    shard_stamps: Vec<GgufShardStamp>,
}

pub struct Qwen4ExpAdmittedMetalWeightPlan {
    plan: Qwen4ExpMetalWeightPlan,
    admission: MetalMemoryAdmission,
}

impl Qwen4ExpAdmittedMetalWeightPlan {
    pub fn admission(&self) -> MetalMemoryAdmission {
        self.admission
    }

    pub fn memory_plan(&self) -> &Qwen4ExpMetalWeightMemoryPlan {
        &self.plan.memory
    }

    pub fn report(&self) -> &Qwen4ExpMetalWeightPlanReport {
        &self.plan.report
    }
}

/// Immutable lookup surface over weights realized with read-only provenance.
/// `MetalTensor::buffer` remains a low-level public handle, so this is not a
/// tamper-proof capability boundary.
pub struct Qwen4ExpMetalWeights {
    config: Qwen4ExpConfig,
    tensors: BTreeMap<String, MetalTensor>,
    ple_source: Qwen4ExpPleSourcePlan,
    report: Qwen4ExpMetalWeightPlanReport,
    memory: Qwen4ExpMetalWeightMemoryPlan,
    device_registry_id: u64,
}

// Metal resources are device-wide and safe to encode from multiple host
// threads. The map and its tensor views are immutable after construction.
unsafe impl Send for Qwen4ExpMetalWeights {}
unsafe impl Sync for Qwen4ExpMetalWeights {}

pub struct Qwen4ExpRealizedMetalWeights {
    weights: Qwen4ExpMetalWeights,
    admission: MetalMemoryAdmission,
    allocated_before: u64,
    allocated_after: u64,
    observed_allocation_delta: u64,
}

impl Qwen4ExpRealizedMetalWeights {
    pub fn admission(&self) -> MetalMemoryAdmission {
        self.admission
    }

    pub fn allocated_before(&self) -> u64 {
        self.allocated_before
    }

    pub fn allocated_after(&self) -> u64 {
        self.allocated_after
    }

    pub fn observed_allocation_delta(&self) -> u64 {
        self.observed_allocation_delta
    }

    pub fn weights(&self) -> &Qwen4ExpMetalWeights {
        &self.weights
    }

    pub fn into_weights(self) -> Qwen4ExpMetalWeights {
        self.weights
    }
}

impl Qwen4ExpMetalWeightPlan {
    pub fn for_ud_q3_k_xl(
        ctx: &MetalContext,
        gguf: &GgufFile,
    ) -> Result<Self, Qwen4ExpResidencyError> {
        let shard_stamps = gguf.revalidate_retained_shard_stamps()?;
        let model = Qwen4ExpModel::from_gguf(gguf)?;
        let reference = Qwen4ExpConfig::flash_next_reference();
        if model.config != reference {
            return invalid("GGUF configuration differs from the released Flash-Next contract");
        }
        if gguf.tensors.len() != QWEN4EXP_RELEASE_TENSOR_COUNT {
            return invalid(format!(
                "released UD-Q3_K_XL requires {QWEN4EXP_RELEASE_TENSOR_COUNT} tensors, got {}",
                gguf.tensors.len()
            ));
        }
        if model.tied_embeddings {
            return invalid("released UD-Q3_K_XL requires a separate output.weight tensor");
        }

        let ple_desc = model.ple_embedding.ok_or_else(|| {
            Qwen4ExpResidencyError::Invalid("released PLE table is missing".into())
        })?;
        validate_release_ple_geometry(ple_desc)?;
        let ple_shard_stamp = shard_stamps
            .get(ple_desc.shard_idx)
            .cloned()
            .ok_or_else(|| {
                Qwen4ExpResidencyError::Invalid(format!(
                    "PLE tensor references missing shard stamp {}",
                    ple_desc.shard_idx
                ))
            })?;
        let logical_row_count = model
            .config
            .ple
            .as_ref()
            .expect("reference configuration has PLE")
            .logical_row_count()
            .map_err(Qwen4ExpLoadError::from)?;
        let dtype_census = validate_ud_q3_k_xl_dtypes(&gguf.tensors)?;

        let page_size = host_page_size_bytes()?;
        let device_max_buffer_length = ctx.max_buffer_length();
        let planning_max_buffer_length =
            device_max_buffer_length.min(QWEN4EXP_RETAINED_WINDOW_CEILING_BYTES);
        let requests = gguf
            .tensors
            .iter()
            .filter(|desc| desc.name != ple_desc.name)
            .collect::<Vec<_>>();
        if requests.len() != QWEN4EXP_METAL_TENSOR_COUNT {
            return invalid(format!(
                "expected {QWEN4EXP_METAL_TENSOR_COUNT} non-PLE tensors, got {}",
                requests.len()
            ));
        }
        let retained = plan_retained_storage(
            &gguf.shard_mapped_lengths(),
            &requests,
            page_size,
            planning_max_buffer_length,
            QWEN4EXP_GGUF_BINDING_ALIGNMENT,
        )?;
        validate_retained_policy(&retained)?;
        let ple_boundary_overlap_bytes =
            validate_ple_window_isolation(&retained.windows, ple_desc, retained.page_size)?;
        let report = report_for_plan(
            gguf,
            &retained,
            ple_desc,
            ple_boundary_overlap_bytes,
            device_max_buffer_length,
            dtype_census,
        )?;
        let memory = build_weight_memory_plan(ctx, &retained, &report)?;
        let descriptors = gguf
            .tensors
            .iter()
            .map(DescriptorFingerprint::from)
            .collect();

        Ok(Self {
            config: model.config,
            retained,
            descriptors,
            ple_source: Qwen4ExpPleSourcePlan {
                descriptor: DescriptorFingerprint::from(ple_desc),
                logical_row_count,
                shard_stamp: ple_shard_stamp,
            },
            report,
            memory,
            device_registry_id: ctx.device.registryID(),
            shard_stamps,
        })
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.config
    }

    pub fn report(&self) -> &Qwen4ExpMetalWeightPlanReport {
        &self.report
    }

    pub fn memory_plan(&self) -> &Qwen4ExpMetalWeightMemoryPlan {
        &self.memory
    }

    pub fn ple_source(&self) -> &Qwen4ExpPleSourcePlan {
        &self.ple_source
    }

    pub fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }

    pub fn admit(
        self,
        signals: MetalMemorySignals,
    ) -> Result<Qwen4ExpAdmittedMetalWeightPlan, Qwen4ExpResidencyError> {
        let admission = self.memory.admission(signals);
        if !admission.admitted {
            return invalid(format!(
                "Metal weight admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                admission.reason.as_str(),
                admission.required_bytes,
                admission.working_set_headroom_bytes,
                admission.signals.process_limit_remaining_bytes
            ));
        }
        Ok(Qwen4ExpAdmittedMetalWeightPlan {
            plan: self,
            admission,
        })
    }

    pub fn revalidate(
        &self,
        ctx: &MetalContext,
        gguf: &GgufFile,
    ) -> Result<(), Qwen4ExpResidencyError> {
        let rebuilt = Self::for_ud_q3_k_xl(ctx, gguf)?;
        if self.device_registry_id != rebuilt.device_registry_id
            || self.config != rebuilt.config
            || self.retained != rebuilt.retained
            || self.descriptors != rebuilt.descriptors
            || self.ple_source != rebuilt.ple_source
            || self.report != rebuilt.report
            || self.memory != rebuilt.memory
            || self.shard_stamps != rebuilt.shard_stamps
        {
            return invalid("weight plan changed before realization");
        }
        Ok(())
    }
}

impl Qwen4ExpMetalWeights {
    pub fn realize(
        ctx: &MetalContext,
        gguf: &GgufFile,
        admitted: Qwen4ExpAdmittedMetalWeightPlan,
    ) -> Result<Qwen4ExpRealizedMetalWeights, Qwen4ExpResidencyError> {
        let plan = admitted.plan;
        if plan.device_registry_id != ctx.device.registryID() {
            return invalid(format!(
                "weight plan belongs to Metal device registry {}, realization context is {}",
                plan.device_registry_id,
                ctx.device.registryID()
            ));
        }
        plan.revalidate(ctx, gguf)?;
        let _allocation_transaction = ctx.begin_allocation_transaction();
        let refreshed_admission = plan.memory.admission(ctx.memory_signals());
        if !refreshed_admission.admitted {
            return invalid(format!(
                "Metal weight admission changed before realization: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                refreshed_admission.reason.as_str(),
                refreshed_admission.required_bytes,
                refreshed_admission.working_set_headroom_bytes,
                refreshed_admission.signals.process_limit_remaining_bytes
            ));
        }

        let allocated_before = refreshed_admission.signals.current_allocated_bytes;
        let windows = realize_windows(ctx, gguf, &plan.retained)?;
        let tensors = realize_tensors(ctx, gguf, &plan.retained, &windows)?;
        let final_stamps = gguf.revalidate_retained_shard_stamps()?;
        if final_stamps != plan.shard_stamps {
            return invalid("GGUF shard identity changed during weight realization");
        }
        validate_realized_weights(gguf, &tensors)?;
        let allocated_after = ctx.current_allocated_size();
        let observed_allocation_delta = plan.memory.reconcile(allocated_before, allocated_after)?;

        Ok(Qwen4ExpRealizedMetalWeights {
            weights: Self {
                config: plan.config,
                tensors,
                ple_source: plan.ple_source,
                report: plan.report,
                memory: plan.memory,
                device_registry_id: plan.device_registry_id,
            },
            admission: refreshed_admission,
            allocated_before,
            allocated_after,
            observed_allocation_delta,
        })
    }

    pub fn validate_context(&self, ctx: &MetalContext) -> Result<(), Qwen4ExpResidencyError> {
        if self.device_registry_id != ctx.device.registryID() {
            return invalid(format!(
                "Metal weights belong to device registry {}, context is {}",
                self.device_registry_id,
                ctx.device.registryID()
            ));
        }
        Ok(())
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.config
    }

    pub fn tensor(&self, name: &str) -> Option<&MetalTensor> {
        self.tensors.get(name)
    }

    pub fn require_tensor(&self, name: &str) -> Result<&MetalTensor, Qwen4ExpResidencyError> {
        self.tensor(name).ok_or_else(|| {
            Qwen4ExpResidencyError::Invalid(format!(
                "realized UD-Q3_K_XL weights are missing tensor {name:?}"
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

    pub fn ple_source(&self) -> &Qwen4ExpPleSourcePlan {
        &self.ple_source
    }

    pub fn report(&self) -> &Qwen4ExpMetalWeightPlanReport {
        &self.report
    }

    pub fn memory_plan(&self) -> &Qwen4ExpMetalWeightMemoryPlan {
        &self.memory
    }

    pub fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }
}

fn validate_release_ple_geometry(desc: &TensorDesc) -> Result<(), Qwen4ExpResidencyError> {
    if desc.shape != [160, QWEN4EXP_PLE_PHYSICAL_ROW_COUNT]
        || desc.n_bytes != QWEN4EXP_PLE_SOURCE_BYTES
    {
        return invalid(format!(
            "released PLE storage must have shape [160, {QWEN4EXP_PLE_PHYSICAL_ROW_COUNT}] and {QWEN4EXP_PLE_SOURCE_BYTES} bytes, got {:?} and {} bytes",
            desc.shape, desc.n_bytes
        ));
    }
    Ok(())
}

fn validate_ud_q3_k_xl_dtypes(
    tensors: &[TensorDesc],
) -> Result<Qwen4ExpDtypeCensus, Qwen4ExpResidencyError> {
    let mut census = Qwen4ExpDtypeCensus::default();
    for desc in tensors {
        let expected = expected_ud_q3_k_xl_dtype(&desc.name).ok_or_else(|| {
            Qwen4ExpResidencyError::Invalid(format!(
                "tensor {:?} has no released UD-Q3_K_XL dtype role",
                desc.name
            ))
        })?;
        if desc.dtype != expected {
            return invalid(format!(
                "tensor {:?} must use {expected:?} in UD-Q3_K_XL, got {:?}",
                desc.name, desc.dtype
            ));
        }
        census.record(desc.dtype)?;
    }
    if census != Qwen4ExpDtypeCensus::UD_Q3_K_XL {
        return invalid(format!(
            "UD-Q3_K_XL dtype census mismatch: expected {:?}, got {census:?}",
            Qwen4ExpDtypeCensus::UD_Q3_K_XL
        ));
    }
    Ok(census)
}

fn expected_ud_q3_k_xl_dtype(name: &str) -> Option<GgmlType> {
    match name {
        "token_embd.weight" => return Some(GgmlType::Q8_0),
        "output.weight" => return Some(GgmlType::Q6_K),
        "output_hc_norm.weight" => return Some(GgmlType::F32),
        "output_hc_down.weight" | "output_hc_up.weight" => return Some(GgmlType::Q8_0),
        "per_layer_token_embd.weight" => return Some(GgmlType::IQ4_NL),
        _ => {}
    }

    let (layer, suffix) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer = layer.parse::<u32>().ok()?;
    if layer >= 48 {
        return None;
    }
    match suffix {
        "hc_attn_norm.weight"
        | "hc_attn_inject.weight"
        | "hc_ffn_norm.weight"
        | "hc_ffn_inject.weight" => Some(GgmlType::F32),
        "hc_attn_down.weight" | "hc_attn_up.weight" | "hc_ffn_down.weight" | "hc_ffn_up.weight" => {
            Some(GgmlType::Q8_0)
        }
        "attn_qkv.weight" | "attn_gate.weight" | "ssm_out.weight" => Some(GgmlType::Q8_0),
        "ssm_beta.weight" | "ssm_alpha.weight" | "ssm_a" | "ssm_dt.bias" | "ssm_conv1d.weight"
        | "ssm_norm.weight" => Some(GgmlType::F32),
        "attn_q.weight" | "attn_k.weight" | "attn_v.weight" | "attn_output.weight" => {
            Some(GgmlType::Q8_0)
        }
        "attn_q_norm.weight"
        | "attn_k_norm.weight"
        | "indexer.q_norm.weight"
        | "indexer.k_norm.weight" => Some(GgmlType::F32),
        "indexer.q_proj.weight" | "indexer.k_proj.weight" => Some(GgmlType::BF16),
        "ffn_gate_inp.weight" | "ffn_gate_inp_shexp.weight" => Some(GgmlType::F32),
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" if layer == 2 => Some(GgmlType::IQ4_XS),
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" => Some(GgmlType::IQ3_XXS),
        "ffn_down_exps.weight" if matches!(layer, 2 | 4 | 30 | 46 | 47) => Some(GgmlType::Q8_0),
        "ffn_down_exps.weight" => Some(GgmlType::IQ4_NL),
        "ffn_gate_shexp.weight"
        | "ffn_up_shexp.weight"
        | "ffn_down_shexp.weight"
        | "ple_key.weight"
        | "ple_value.weight" => Some(GgmlType::Q8_0),
        "ple_norm_key.weight"
        | "ple_norm_query.weight"
        | "ple_norm_conv.weight"
        | "ple_conv1d.weight" => Some(GgmlType::F32),
        _ => None,
    }
}

fn validate_retained_policy(plan: &RetainedStoragePlan) -> Result<(), Qwen4ExpResidencyError> {
    if plan.entries.len() != QWEN4EXP_METAL_TENSOR_COUNT {
        return invalid(format!(
            "retained planner produced {} entries, expected {QWEN4EXP_METAL_TENSOR_COUNT}",
            plan.entries.len()
        ));
    }
    for entry in &plan.entries {
        match entry.disposition {
            RetainedStorageDisposition::View { .. } => {}
            RetainedStorageDisposition::CopyFallback {
                reason: RetainedStorageFallback::FinalPartialPage,
            } => {}
            RetainedStorageDisposition::CopyFallback { reason } => {
                return invalid(format!(
                    "tensor {:?} requires disallowed {reason:?} copy fallback",
                    entry.name
                ));
            }
            RetainedStorageDisposition::Alias {
                source_request_index,
            } => {
                return invalid(format!(
                    "tensor {:?} unexpectedly aliases request {source_request_index}",
                    entry.name
                ));
            }
        }
    }
    Ok(())
}

fn price_shared_buffer(
    ctx: &MetalContext,
    logical_bytes: u64,
    name: &str,
) -> Result<u64, Qwen4ExpResidencyError> {
    ctx.price_shared_buffer_upper(logical_bytes)
        .map(|priced| priced.priced_upper_bytes)
        .map_err(|error| {
            Qwen4ExpResidencyError::Invalid(format!("planned Metal buffer {name:?} {error}"))
        })
}

fn build_weight_memory_plan(
    ctx: &MetalContext,
    retained: &RetainedStoragePlan,
    report: &Qwen4ExpMetalWeightPlanReport,
) -> Result<Qwen4ExpMetalWeightMemoryPlan, Qwen4ExpResidencyError> {
    let mut buffer_count = 0_usize;
    let mut logical_bytes = 0_u64;
    let mut priced_upper_bytes = 0_u64;
    for (index, window) in retained.windows.iter().enumerate() {
        let logical = u64::try_from(window.length).map_err(|_| {
            Qwen4ExpResidencyError::Invalid("retained window length exceeds u64".into())
        })?;
        let priced = price_shared_buffer(ctx, logical, &format!("weight_window[{index}]"))?;
        buffer_count = buffer_count.checked_add(1).ok_or_else(|| {
            Qwen4ExpResidencyError::Invalid("weight buffer count overflow".into())
        })?;
        logical_bytes = logical_bytes.checked_add(logical).ok_or_else(|| {
            Qwen4ExpResidencyError::Invalid("logical weight byte count overflow".into())
        })?;
        priced_upper_bytes = priced_upper_bytes.checked_add(priced).ok_or_else(|| {
            Qwen4ExpResidencyError::Invalid("priced weight byte count overflow".into())
        })?;
    }
    for entry in &retained.entries {
        if matches!(
            entry.disposition,
            RetainedStorageDisposition::CopyFallback { .. }
        ) {
            let priced = price_shared_buffer(ctx, entry.n_bytes, &entry.name)?;
            buffer_count = buffer_count.checked_add(1).ok_or_else(|| {
                Qwen4ExpResidencyError::Invalid("weight buffer count overflow".into())
            })?;
            logical_bytes = logical_bytes.checked_add(entry.n_bytes).ok_or_else(|| {
                Qwen4ExpResidencyError::Invalid("logical fallback byte count overflow".into())
            })?;
            priced_upper_bytes = priced_upper_bytes.checked_add(priced).ok_or_else(|| {
                Qwen4ExpResidencyError::Invalid("priced fallback byte count overflow".into())
            })?;
        }
    }

    let expected_buffers = report
        .planned_window_count
        .checked_add(report.fallback_count)
        .ok_or_else(|| Qwen4ExpResidencyError::Invalid("report buffer count overflow".into()))?;
    let expected_logical = report
        .planned_window_bytes
        .checked_add(report.fallback_bytes)
        .ok_or_else(|| Qwen4ExpResidencyError::Invalid("report weight bytes overflow".into()))?;
    if buffer_count != expected_buffers || logical_bytes != expected_logical {
        return invalid(format!(
            "weight memory plan differs from report: buffers={buffer_count}/{expected_buffers} logical={logical_bytes}/{expected_logical}"
        ));
    }
    if priced_upper_bytes < logical_bytes {
        return invalid("priced Metal weight bytes are below logical bytes");
    }
    Ok(Qwen4ExpMetalWeightMemoryPlan {
        buffer_count,
        logical_bytes,
        priced_upper_bytes,
    })
}

fn realize_windows(
    ctx: &MetalContext,
    gguf: &GgufFile,
    plan: &RetainedStoragePlan,
) -> Result<Vec<MetalGgufBacking>, Qwen4ExpResidencyError> {
    let mut windows = Vec::with_capacity(plan.windows.len());
    for (index, window) in plan.windows.iter().enumerate() {
        let mmap = gguf.retained_shard_mmap(window.shard_idx).ok_or_else(|| {
            Qwen4ExpResidencyError::Invalid(format!(
                "planned window {index} references missing shard {}",
                window.shard_idx
            ))
        })?;
        let mmap_offset = usize::try_from(window.mmap_offset).map_err(|_| {
            Qwen4ExpResidencyError::Invalid(format!("planned window {index} offset exceeds usize"))
        })?;
        let backing = ctx.gguf_no_copy_window(
            mmap,
            window.shard_idx,
            mmap_offset,
            window.length,
            QWEN4EXP_GGUF_BINDING_ALIGNMENT,
        )?;
        if backing.mmap_offset() != mmap_offset
            || backing.exposed_len() != window.length
            || backing.required_alignment() != QWEN4EXP_GGUF_BINDING_ALIGNMENT
        {
            return invalid(format!("retained window {index} realization drift"));
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
) -> Result<BTreeMap<String, MetalTensor>, Qwen4ExpResidencyError> {
    let descriptors = gguf
        .tensors
        .iter()
        .filter(|desc| desc.name != "per_layer_token_embd.weight")
        .collect::<Vec<_>>();
    if descriptors.len() != plan.entries.len() {
        return invalid(format!(
            "realization descriptor count {} differs from planned {}",
            descriptors.len(),
            plan.entries.len()
        ));
    }

    let mut tensors = BTreeMap::new();
    for (index, (entry, desc)) in plan.entries.iter().zip(descriptors).enumerate() {
        if entry.request_index != index
            || entry.name != desc.name
            || entry.shard_idx != desc.shard_idx
            || entry.data_offset != desc.data_offset
            || entry.n_bytes != desc.n_bytes
        {
            return invalid(format!("planner descriptor drift at request {index}"));
        }
        let (tensor, expected_provenance) = match entry.disposition {
            RetainedStorageDisposition::View {
                window_index,
                buffer_offset,
            } => {
                let backing = windows.get(window_index).ok_or_else(|| {
                    Qwen4ExpResidencyError::Invalid(format!(
                        "tensor {:?} references missing window {window_index}",
                        desc.name
                    ))
                })?;
                let (eligibility, tensor) = backing.tensor(desc)?;
                let tensor = tensor.ok_or_else(|| {
                    Qwen4ExpResidencyError::Invalid(format!(
                        "tensor {:?} failed retained realization: {eligibility:?}",
                        desc.name
                    ))
                })?;
                if tensor.offset != buffer_offset {
                    return invalid(format!("retained offset drift for tensor {:?}", desc.name));
                }
                (tensor, MetalTensorProvenance::RetainedGgufReadOnly)
            }
            RetainedStorageDisposition::CopyFallback {
                reason: RetainedStorageFallback::FinalPartialPage,
            } => (
                MetalTensor::copied_gguf_weight(ctx, desc, gguf.try_slice(desc)?)?,
                MetalTensorProvenance::OwnedWeightReadOnly,
            ),
            RetainedStorageDisposition::CopyFallback { reason } => {
                return invalid(format!(
                    "disallowed {reason:?} fallback reached realization for {:?}",
                    desc.name
                ));
            }
            RetainedStorageDisposition::Alias {
                source_request_index,
            } => {
                return invalid(format!(
                    "unexpected alias for {:?} to request {source_request_index}",
                    desc.name
                ));
            }
        };
        validate_realized_tensor(desc, &tensor, expected_provenance)?;
        if tensors.insert(desc.name.clone(), tensor).is_some() {
            return invalid(format!("duplicate realized tensor {:?}", desc.name));
        }
    }
    Ok(tensors)
}

fn validate_realized_tensor(
    desc: &TensorDesc,
    tensor: &MetalTensor,
    expected_provenance: MetalTensorProvenance,
) -> Result<(), Qwen4ExpResidencyError> {
    if tensor.shape != desc.shape
        || tensor.dtype != desc.dtype
        || tensor.n_bytes() != desc.n_bytes
        || tensor.provenance() != expected_provenance
        || tensor.is_writable()
    {
        return invalid(format!(
            "realized tensor metadata drift for {:?}",
            desc.name
        ));
    }
    let end = tensor.offset.checked_add(desc.n_bytes).ok_or_else(|| {
        Qwen4ExpResidencyError::Invalid(format!(
            "realized tensor {:?} buffer range overflow",
            desc.name
        ))
    })?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "realized tensor {:?} ends at {end}, beyond buffer length {}",
            desc.name,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

fn validate_realized_weights(
    gguf: &GgufFile,
    tensors: &BTreeMap<String, MetalTensor>,
) -> Result<(), Qwen4ExpResidencyError> {
    if tensors.len() != QWEN4EXP_METAL_TENSOR_COUNT
        || tensors.contains_key("per_layer_token_embd.weight")
    {
        return invalid(format!(
            "realized map has {} tensors or includes the CPU-only PLE table",
            tensors.len()
        ));
    }
    for desc in gguf
        .tensors
        .iter()
        .filter(|desc| desc.name != "per_layer_token_embd.weight")
    {
        let tensor = tensors.get(&desc.name).ok_or_else(|| {
            Qwen4ExpResidencyError::Invalid(format!(
                "realized map is missing tensor {:?}",
                desc.name
            ))
        })?;
        if !matches!(
            tensor.provenance(),
            MetalTensorProvenance::RetainedGgufReadOnly
                | MetalTensorProvenance::OwnedWeightReadOnly
        ) {
            return invalid(format!(
                "realized tensor {:?} is not read-only by provenance",
                desc.name
            ));
        }
        if tensor.shape != desc.shape
            || tensor.dtype != desc.dtype
            || tensor.n_bytes() != desc.n_bytes
        {
            return invalid(format!(
                "realized tensor metadata drift for {:?}",
                desc.name
            ));
        }
    }
    Ok(())
}

fn validate_ple_window_isolation(
    windows: &[RetainedStorageWindow],
    ple: &TensorDesc,
    page_size: usize,
) -> Result<u64, Qwen4ExpResidencyError> {
    let ple_end = ple
        .data_offset
        .checked_add(ple.n_bytes)
        .ok_or_else(|| Qwen4ExpResidencyError::Invalid("PLE source range overflow".into()))?;
    let mut total = 0_u64;
    for (index, window) in windows.iter().enumerate() {
        if window.shard_idx != ple.shard_idx {
            continue;
        }
        let window_len = u64::try_from(window.length).map_err(|_| {
            Qwen4ExpResidencyError::Invalid(format!(
                "retained window {index} length does not fit u64"
            ))
        })?;
        let window_end = window.mmap_offset.checked_add(window_len).ok_or_else(|| {
            Qwen4ExpResidencyError::Invalid(format!(
                "retained window {index} source range overflow"
            ))
        })?;
        let overlap = window_end
            .min(ple_end)
            .saturating_sub(window.mmap_offset.max(ple.data_offset));
        if overlap >= page_size as u64 {
            return invalid(format!(
                "retained window {index} crosses {overlap} PLE bytes instead of stopping at its page boundary"
            ));
        }
        total = total.checked_add(overlap).ok_or_else(|| {
            Qwen4ExpResidencyError::Invalid("PLE boundary overlap accounting overflow".into())
        })?;
    }
    Ok(total)
}

fn report_for_plan(
    gguf: &GgufFile,
    plan: &RetainedStoragePlan,
    ple: &TensorDesc,
    ple_boundary_overlap_bytes: u64,
    device_max_buffer_length: usize,
    dtype_census: Qwen4ExpDtypeCensus,
) -> Result<Qwen4ExpMetalWeightPlanReport, Qwen4ExpResidencyError> {
    let source_bytes = checked_sum(
        gguf.tensors.iter().map(|desc| desc.n_bytes),
        "source byte count",
    )?;
    let metal_source_bytes = checked_sum(
        plan.entries.iter().map(|entry| entry.n_bytes),
        "Metal source byte count",
    )?;
    if source_bytes.checked_sub(ple.n_bytes) != Some(metal_source_bytes) {
        return invalid("non-PLE source byte accounting differs from retained plan");
    }
    let planned_window_bytes = checked_sum(
        plan.windows.iter().map(|window| window.length as u64),
        "planned window byte count",
    )?;
    let view_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
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

    Ok(Qwen4ExpMetalWeightPlanReport {
        source_tensor_count: gguf.tensors.len(),
        metal_tensor_count: plan.entries.len(),
        cpu_ple_tensor_count: 1,
        source_bytes,
        metal_source_bytes,
        cpu_ple_source_bytes: ple.n_bytes,
        planned_window_count: plan.windows.len(),
        planned_window_bytes,
        view_count,
        fallback_count,
        fallback_bytes: plan.unique_fallback_bytes,
        ple_boundary_overlap_bytes,
        page_size: plan.page_size,
        device_max_buffer_length,
        planning_max_buffer_length: plan.max_buffer_length,
        dtype_census,
    })
}

fn checked_sum(
    values: impl IntoIterator<Item = u64>,
    label: &str,
) -> Result<u64, Qwen4ExpResidencyError> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| Qwen4ExpResidencyError::Invalid(format!("{label} overflow")))
    })
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpResidencyError> {
    Err(Qwen4ExpResidencyError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn release_tensor_names() -> Vec<String> {
        let mut names = vec![
            "token_embd.weight".into(),
            "output.weight".into(),
            "output_hc_norm.weight".into(),
            "output_hc_down.weight".into(),
            "output_hc_up.weight".into(),
            "per_layer_token_embd.weight".into(),
        ];
        for layer in 0..48 {
            for role in ["attn", "ffn"] {
                for parameter in ["norm", "down", "up", "inject"] {
                    names.push(format!("blk.{layer}.hc_{role}_{parameter}.weight"));
                }
            }
            if layer % 4 == 3 {
                for suffix in [
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                    "attn_q_norm.weight",
                    "attn_k_norm.weight",
                    "indexer.q_proj.weight",
                    "indexer.k_proj.weight",
                    "indexer.q_norm.weight",
                    "indexer.k_norm.weight",
                ] {
                    names.push(format!("blk.{layer}.{suffix}"));
                }
            } else {
                for suffix in [
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_beta.weight",
                    "ssm_alpha.weight",
                    "ssm_a",
                    "ssm_dt.bias",
                    "ssm_conv1d.weight",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ] {
                    names.push(format!("blk.{layer}.{suffix}"));
                }
            }
            for suffix in [
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
                "ffn_gate_inp_shexp.weight",
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
                "ffn_down_shexp.weight",
            ] {
                names.push(format!("blk.{layer}.{suffix}"));
            }
            if layer == 1 {
                for suffix in [
                    "ple_key.weight",
                    "ple_value.weight",
                    "ple_norm_key.weight",
                    "ple_norm_query.weight",
                    "ple_norm_conv.weight",
                    "ple_conv1d.weight",
                ] {
                    names.push(format!("blk.{layer}.{suffix}"));
                }
            }
        }
        names
    }

    #[test]
    fn ud_q3_k_xl_policy_covers_exact_release_schema_and_census() {
        let names = release_tensor_names();
        assert_eq!(names.len(), QWEN4EXP_RELEASE_TENSOR_COUNT);
        assert_eq!(names.iter().collect::<HashSet<_>>().len(), names.len());
        let mut census = Qwen4ExpDtypeCensus::default();
        for name in names {
            census
                .record(
                    expected_ud_q3_k_xl_dtype(&name)
                        .unwrap_or_else(|| panic!("missing released dtype policy for {name}")),
                )
                .unwrap();
        }
        assert_eq!(census, Qwen4ExpDtypeCensus::UD_Q3_K_XL);
        assert_eq!(census.total(), QWEN4EXP_RELEASE_TENSOR_COUNT);
    }

    #[test]
    fn ud_q3_k_xl_exceptions_are_layer_exact() {
        assert_eq!(
            expected_ud_q3_k_xl_dtype("blk.2.ffn_gate_exps.weight"),
            Some(GgmlType::IQ4_XS)
        );
        assert_eq!(
            expected_ud_q3_k_xl_dtype("blk.1.ffn_gate_exps.weight"),
            Some(GgmlType::IQ3_XXS)
        );
        for layer in [2, 4, 30, 46, 47] {
            assert_eq!(
                expected_ud_q3_k_xl_dtype(&format!("blk.{layer}.ffn_down_exps.weight")),
                Some(GgmlType::Q8_0)
            );
        }
        assert_eq!(
            expected_ud_q3_k_xl_dtype("blk.45.ffn_down_exps.weight"),
            Some(GgmlType::IQ4_NL)
        );
        assert_eq!(expected_ud_q3_k_xl_dtype("blk.48.ssm_a"), None);
        assert_eq!(expected_ud_q3_k_xl_dtype("blk.0.unknown.weight"), None);
    }

    #[test]
    fn ud_q3_k_xl_policy_rejects_role_dtype_substitution() {
        let tensors = [TensorDesc {
            name: "output.weight".into(),
            shape: vec![2_560, 248_320],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: 0,
        }];
        let error = validate_ud_q3_k_xl_dtypes(&tensors)
            .unwrap_err()
            .to_string();
        assert!(error.contains("must use Q6_K"));
    }

    #[test]
    fn release_profile_rejects_unpadded_ple_storage() {
        let desc = TensorDesc {
            name: "per_layer_token_embd.weight".into(),
            shape: vec![160, 320_001_446],
            dtype: GgmlType::IQ4_NL,
            shard_idx: 1,
            data_offset: 0,
            n_bytes: 28_800_130_140,
        };
        let error = validate_release_ple_geometry(&desc)
            .unwrap_err()
            .to_string();
        assert!(error.contains("320001536"));
        assert!(error.contains("28800138240"));
    }

    #[test]
    fn weight_memory_admission_and_reconciliation_fail_closed() {
        let memory = Qwen4ExpMetalWeightMemoryPlan {
            buffer_count: 3,
            logical_bytes: 900,
            priced_upper_bytes: 1_000,
        };
        let exact = MetalMemorySignals {
            recommended_max_bytes: 1_100,
            current_allocated_bytes: 100,
            process_limit_remaining_bytes: Some(1_000),
        };
        assert!(memory.admission(exact).admitted);

        let working_set_short = MetalMemorySignals {
            recommended_max_bytes: 1_099,
            ..exact
        };
        assert!(!memory.admission(working_set_short).admitted);
        let process_short = MetalMemorySignals {
            process_limit_remaining_bytes: Some(999),
            ..exact
        };
        assert!(!memory.admission(process_short).admitted);

        assert_eq!(memory.reconcile(100, 1_100).unwrap(), 1_000);
        assert_eq!(memory.reconcile(101, 100).unwrap(), 0);
        assert!(memory.reconcile(100, 1_101).is_err());
    }

    #[test]
    fn retained_windows_cannot_bridge_the_ple_table() {
        let ple = TensorDesc {
            name: "per_layer_token_embd.weight".into(),
            shape: vec![160, 320_001_536],
            dtype: GgmlType::IQ4_NL,
            shard_idx: 1,
            data_offset: 40_960,
            n_bytes: 28_800_138_240,
        };
        let page_size = 16_384;
        let boundary_windows = [
            RetainedStorageWindow {
                shard_idx: 1,
                mmap_offset: 16_384,
                length: 32_768,
            },
            RetainedStorageWindow {
                shard_idx: 1,
                mmap_offset: ple.data_offset + ple.n_bytes - 8_192,
                length: 16_384,
            },
        ];
        assert_eq!(
            validate_ple_window_isolation(&boundary_windows, &ple, page_size).unwrap(),
            16_384
        );

        let bridging = [RetainedStorageWindow {
            shard_idx: 1,
            mmap_offset: 16_384,
            length: usize::try_from(ple.n_bytes + 32_768).unwrap(),
        }];
        assert!(validate_ple_window_isolation(&bridging, &ple, page_size).is_err());
    }
}
