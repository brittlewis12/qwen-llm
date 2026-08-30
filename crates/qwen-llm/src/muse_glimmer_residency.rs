//! Retained GGUF weight residency for the released Muse Glimmer 30B text lane.
//!
//! Q8_0 and BF16 matrices stay in their native on-disk representation. Metal
//! buffers retain page-aligned GGUF mmap windows where possible; only tensors
//! touching a shard's final partial page are copied.

use crate::gguf::{GgufError, GgufFile, GgufShardStamp};
use crate::metal::{
    MetalContext, MetalError, MetalGgufBacking, MetalMemoryAdmission, MetalMemorySignals,
    MetalTensor, MetalTensorProvenance, RetainedStorageDisposition, RetainedStorageFallback,
    RetainedStoragePlan, evaluate_metal_memory_admission, host_page_size_bytes,
    plan_retained_storage,
};
use crate::muse_glimmer::{
    MuseGlimmerArtifactProfile, MuseGlimmerConfig, MuseGlimmerError, MuseGlimmerModel,
    RELEASE_MATRIX_TENSOR_COUNT, RELEASE_NORM_TENSOR_COUNT, RELEASE_TENSOR_COUNT,
};
use crate::tensor::{GgmlType, TensorDesc};
use objc2_metal::{MTLBuffer, MTLDevice};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const MUSE_GLIMMER_GGUF_BINDING_ALIGNMENT: usize = 32;
pub const MUSE_GLIMMER_RETAINED_WINDOW_CEILING_BYTES: usize = 8 * 1024 * 1024 * 1024;
pub const MUSE_GLIMMER_UNSLOTH_REVISION: &str = "faa5b025c584459c13febfa5c59883516710ae39";

const UNSLOTH_Q8_SHARDS: [ReleaseShardIdentity; 1] = [ReleaseShardIdentity {
    size: 29_612_957_984,
    sha256: [
        0xf2, 0xc0, 0x87, 0xd6, 0x94, 0xca, 0x82, 0x42, 0xa4, 0xa4, 0x36, 0x07, 0x6d, 0xf7, 0xc0,
        0x41, 0x70, 0x3a, 0xb0, 0x51, 0xac, 0x4b, 0x72, 0xbb, 0x1b, 0xfe, 0x26, 0x98, 0x29, 0x9b,
        0x0e, 0x86,
    ],
}];

const UNSLOTH_BF16_SHARDS: [ReleaseShardIdentity; 2] = [
    ReleaseShardIdentity {
        size: 29_805_191_936,
        sha256: [
            0x9e, 0xd3, 0xc9, 0x04, 0xca, 0x80, 0xbe, 0x99, 0xe7, 0x87, 0xd9, 0x5c, 0xc4, 0x30,
            0xc5, 0xb1, 0xce, 0x3a, 0x44, 0xcb, 0x7a, 0xcc, 0x01, 0x2f, 0xda, 0x63, 0xc8, 0xe1,
            0x64, 0x9d, 0xf7, 0xe5,
        ],
    },
    ReleaseShardIdentity {
        size: 25_920_319_232,
        sha256: [
            0xbb, 0xec, 0x2a, 0xef, 0x54, 0xb0, 0x1b, 0xff, 0x34, 0x97, 0x7a, 0x7d, 0xa2, 0xa6,
            0xe9, 0xe7, 0x3c, 0x3e, 0xdc, 0x06, 0x3f, 0xab, 0xf0, 0xae, 0x4b, 0xd6, 0x28, 0x17,
            0xb7, 0xef, 0xe1, 0xae,
        ],
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReleaseShardIdentity {
    size: u64,
    sha256: [u8; 32],
}

#[derive(Debug, thiserror::Error)]
pub enum MuseGlimmerResidencyError {
    #[error(transparent)]
    Model(#[from] MuseGlimmerError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("invalid Muse Glimmer residency contract: {0}")]
    Invalid(String),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MuseGlimmerDtypeCensus {
    f32: usize,
    q8_0: usize,
    bf16: usize,
}

impl MuseGlimmerDtypeCensus {
    pub const UNSLOTH_Q8_0: Self = Self {
        f32: RELEASE_NORM_TENSOR_COUNT,
        q8_0: RELEASE_MATRIX_TENSOR_COUNT,
        bf16: 0,
    };

    pub const UNSLOTH_BF16: Self = Self {
        f32: RELEASE_NORM_TENSOR_COUNT,
        q8_0: 0,
        bf16: RELEASE_MATRIX_TENSOR_COUNT,
    };

    pub fn total(self) -> usize {
        self.f32 + self.q8_0 + self.bf16
    }

    pub fn f32(self) -> usize {
        self.f32
    }

    pub fn q8_0(self) -> usize {
        self.q8_0
    }

    pub fn bf16(self) -> usize {
        self.bf16
    }

    fn record(&mut self, dtype: GgmlType) -> Result<(), MuseGlimmerResidencyError> {
        let count = match dtype {
            GgmlType::F32 => &mut self.f32,
            GgmlType::Q8_0 => &mut self.q8_0,
            GgmlType::BF16 => &mut self.bf16,
            other => return invalid(format!("unsupported released weight dtype {other:?}")),
        };
        *count = count
            .checked_add(1)
            .ok_or_else(|| MuseGlimmerResidencyError::Invalid("dtype census overflow".into()))?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MuseGlimmerMetalWeightPlanReport {
    pub source_tensor_count: usize,
    pub source_bytes: u64,
    pub planned_window_count: usize,
    pub planned_window_bytes: u64,
    pub view_count: usize,
    pub fallback_count: usize,
    pub fallback_bytes: u64,
    pub page_size: usize,
    pub device_max_buffer_length: usize,
    pub planning_max_buffer_length: usize,
    pub dtype_census: MuseGlimmerDtypeCensus,
    pub contents_authenticated: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MuseGlimmerMetalWeightMemoryPlan {
    buffer_count: usize,
    logical_bytes: u64,
    priced_upper_bytes: u64,
}

impl MuseGlimmerMetalWeightMemoryPlan {
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
    ) -> Result<u64, MuseGlimmerResidencyError> {
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

pub struct MuseGlimmerMetalWeightPlan {
    config: MuseGlimmerConfig,
    artifact_profile: MuseGlimmerArtifactProfile,
    retained: RetainedStoragePlan,
    descriptors: Vec<DescriptorFingerprint>,
    report: MuseGlimmerMetalWeightPlanReport,
    memory: MuseGlimmerMetalWeightMemoryPlan,
    device_registry_id: u64,
    shard_stamps: Vec<GgufShardStamp>,
    authenticated_shards: Vec<ReleaseShardIdentity>,
    contents_authenticated: bool,
}

pub struct MuseGlimmerAdmittedMetalWeightPlan {
    plan: MuseGlimmerMetalWeightPlan,
    admission: MetalMemoryAdmission,
}

impl MuseGlimmerAdmittedMetalWeightPlan {
    pub fn admission(&self) -> MetalMemoryAdmission {
        self.admission
    }

    pub fn memory_plan(&self) -> &MuseGlimmerMetalWeightMemoryPlan {
        &self.plan.memory
    }

    pub fn report(&self) -> &MuseGlimmerMetalWeightPlanReport {
        &self.plan.report
    }
}

/// Immutable lookup surface over weights realized with read-only provenance.
/// `MetalTensor::buffer` remains public, so this is not a capability boundary.
pub struct MuseGlimmerMetalWeights {
    config: MuseGlimmerConfig,
    artifact_profile: MuseGlimmerArtifactProfile,
    tensors: BTreeMap<String, MetalTensor>,
    report: MuseGlimmerMetalWeightPlanReport,
    memory: MuseGlimmerMetalWeightMemoryPlan,
    device_registry_id: u64,
}

// Metal resources are device-wide and safe to encode from multiple host
// threads. The map and tensor views are immutable after realization.
unsafe impl Send for MuseGlimmerMetalWeights {}
unsafe impl Sync for MuseGlimmerMetalWeights {}

pub struct MuseGlimmerRealizedMetalWeights {
    weights: MuseGlimmerMetalWeights,
    admission: MetalMemoryAdmission,
    allocated_before: u64,
    allocated_after: u64,
    observed_allocation_delta: u64,
}

impl MuseGlimmerRealizedMetalWeights {
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

    pub fn weights(&self) -> &MuseGlimmerMetalWeights {
        &self.weights
    }

    pub fn into_weights(self) -> MuseGlimmerMetalWeights {
        self.weights
    }
}

impl MuseGlimmerMetalWeightPlan {
    pub fn for_release(
        ctx: &MetalContext,
        gguf: &GgufFile,
    ) -> Result<Self, MuseGlimmerResidencyError> {
        Self::build_release(ctx, gguf, false, false)
    }

    pub fn for_authenticated_release(
        ctx: &MetalContext,
        gguf: &GgufFile,
    ) -> Result<Self, MuseGlimmerResidencyError> {
        Self::build_release(ctx, gguf, true, true)
    }

    fn build_release(
        ctx: &MetalContext,
        gguf: &GgufFile,
        authenticate_contents: bool,
        contents_authenticated: bool,
    ) -> Result<Self, MuseGlimmerResidencyError> {
        let shard_stamps = gguf.revalidate_retained_shard_stamps()?;
        let model = MuseGlimmerModel::from_gguf(gguf)?;
        if gguf.tensors.len() != RELEASE_TENSOR_COUNT {
            return invalid(format!(
                "released model requires {RELEASE_TENSOR_COUNT} tensors, got {}",
                gguf.tensors.len()
            ));
        }
        let expected_shards = expected_release_shards(model.artifact_profile);
        validate_release_shard_sizes(gguf, expected_shards)?;
        if authenticate_contents {
            preflight_authentication_admission(ctx, expected_shards)?;
            authenticate_release_shards(gguf, expected_shards)?;
            if gguf.revalidate_retained_shard_stamps()? != shard_stamps {
                return invalid("GGUF shard identity changed during content authentication");
            }
        }
        let dtype_census = validate_release_dtype_census(&gguf.tensors, model.artifact_profile)?;
        let page_size = host_page_size_bytes()?;
        let device_max_buffer_length = ctx.max_buffer_length();
        let planning_max_buffer_length =
            device_max_buffer_length.min(MUSE_GLIMMER_RETAINED_WINDOW_CEILING_BYTES);
        let requests = gguf.tensors.iter().collect::<Vec<_>>();
        let retained = plan_retained_storage(
            &gguf.shard_mapped_lengths(),
            &requests,
            page_size,
            planning_max_buffer_length,
            MUSE_GLIMMER_GGUF_BINDING_ALIGNMENT,
        )?;
        validate_retained_policy(&retained)?;
        let report = report_for_plan(
            gguf,
            &retained,
            device_max_buffer_length,
            dtype_census,
            contents_authenticated,
        )?;
        let memory = build_weight_memory_plan(ctx, &retained, &report)?;
        let descriptors = gguf
            .tensors
            .iter()
            .map(DescriptorFingerprint::from)
            .collect();

        Ok(Self {
            config: model.config,
            artifact_profile: model.artifact_profile,
            retained,
            descriptors,
            report,
            memory,
            device_registry_id: ctx.device.registryID(),
            shard_stamps,
            authenticated_shards: expected_shards.to_vec(),
            contents_authenticated,
        })
    }

    pub fn config(&self) -> &MuseGlimmerConfig {
        &self.config
    }

    pub fn artifact_profile(&self) -> MuseGlimmerArtifactProfile {
        self.artifact_profile
    }

    pub fn report(&self) -> &MuseGlimmerMetalWeightPlanReport {
        &self.report
    }

    pub fn memory_plan(&self) -> &MuseGlimmerMetalWeightMemoryPlan {
        &self.memory
    }

    pub fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }

    pub fn contents_authenticated(&self) -> bool {
        self.contents_authenticated
    }

    pub fn admit(
        self,
        signals: MetalMemorySignals,
    ) -> Result<MuseGlimmerAdmittedMetalWeightPlan, MuseGlimmerResidencyError> {
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
        Ok(MuseGlimmerAdmittedMetalWeightPlan {
            plan: self,
            admission,
        })
    }

    pub fn revalidate(
        &self,
        ctx: &MetalContext,
        gguf: &GgufFile,
    ) -> Result<(), MuseGlimmerResidencyError> {
        if gguf.revalidate_retained_shard_stamps()? != self.shard_stamps {
            return invalid("GGUF shard identity changed after authenticated planning");
        }
        let rebuilt = Self::build_release(ctx, gguf, false, self.contents_authenticated)?;
        if self.device_registry_id != rebuilt.device_registry_id
            || self.config != rebuilt.config
            || self.artifact_profile != rebuilt.artifact_profile
            || self.retained != rebuilt.retained
            || self.descriptors != rebuilt.descriptors
            || self.report != rebuilt.report
            || self.memory != rebuilt.memory
            || self.shard_stamps != rebuilt.shard_stamps
            || self.authenticated_shards != rebuilt.authenticated_shards
            || self.contents_authenticated != rebuilt.contents_authenticated
        {
            return invalid("weight plan changed before realization");
        }
        Ok(())
    }
}

impl MuseGlimmerMetalWeights {
    pub fn realize(
        ctx: &MetalContext,
        gguf: &GgufFile,
        admitted: MuseGlimmerAdmittedMetalWeightPlan,
    ) -> Result<MuseGlimmerRealizedMetalWeights, MuseGlimmerResidencyError> {
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

        Ok(MuseGlimmerRealizedMetalWeights {
            weights: Self {
                config: plan.config,
                artifact_profile: plan.artifact_profile,
                tensors,
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

    pub fn validate_context(&self, ctx: &MetalContext) -> Result<(), MuseGlimmerResidencyError> {
        if self.device_registry_id != ctx.device.registryID() {
            return invalid(format!(
                "weights belong to Metal device registry {}, context is {}",
                self.device_registry_id,
                ctx.device.registryID()
            ));
        }
        Ok(())
    }

    pub fn config(&self) -> &MuseGlimmerConfig {
        &self.config
    }

    pub fn artifact_profile(&self) -> MuseGlimmerArtifactProfile {
        self.artifact_profile
    }

    pub fn tensor(&self, name: &str) -> Option<&MetalTensor> {
        self.tensors.get(name)
    }

    pub fn require_tensor(&self, name: &str) -> Result<&MetalTensor, MuseGlimmerResidencyError> {
        self.tensor(name).ok_or_else(|| {
            MuseGlimmerResidencyError::Invalid(format!(
                "realized Muse Glimmer weights are missing tensor {name:?}"
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

    pub fn report(&self) -> &MuseGlimmerMetalWeightPlanReport {
        &self.report
    }

    pub fn memory_plan(&self) -> &MuseGlimmerMetalWeightMemoryPlan {
        &self.memory
    }

    pub fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }
}

#[derive(Clone, Copy)]
pub struct MuseGlimmerMetalLayerWeights<'a> {
    pub attention_norm: &'a MetalTensor,
    pub attention_query: &'a MetalTensor,
    pub attention_key: &'a MetalTensor,
    pub attention_value: &'a MetalTensor,
    pub attention_gate: &'a MetalTensor,
    pub attention_output: &'a MetalTensor,
    pub query_norm: &'a MetalTensor,
    pub key_norm: &'a MetalTensor,
    pub post_attention_norm: &'a MetalTensor,
    pub feed_forward_norm: &'a MetalTensor,
    pub feed_forward_gate: &'a MetalTensor,
    pub feed_forward_up: &'a MetalTensor,
    pub feed_forward_down: &'a MetalTensor,
    pub post_feed_forward_norm: &'a MetalTensor,
    pub sliding_attention: bool,
}

pub struct MuseGlimmerMetalModelWeights<'a> {
    pub config: &'a MuseGlimmerConfig,
    pub token_embedding: &'a MetalTensor,
    pub output_norm: &'a MetalTensor,
    pub output: &'a MetalTensor,
    pub layers: Vec<MuseGlimmerMetalLayerWeights<'a>>,
}

impl<'a> MuseGlimmerMetalModelWeights<'a> {
    pub fn bind(weights: &'a MuseGlimmerMetalWeights) -> Result<Self, MuseGlimmerResidencyError> {
        let mut layers = Vec::with_capacity(weights.config.layer_count as usize);
        for layer in 0..weights.config.layer_count {
            let prefix = format!("blk.{layer}");
            let tensor = |suffix: &str| weights.require_tensor(&format!("{prefix}.{suffix}"));
            layers.push(MuseGlimmerMetalLayerWeights {
                attention_norm: tensor("attn_norm.weight")?,
                attention_query: tensor("attn_q.weight")?,
                attention_key: tensor("attn_k.weight")?,
                attention_value: tensor("attn_v.weight")?,
                attention_gate: tensor("attn_gate.weight")?,
                attention_output: tensor("attn_output.weight")?,
                query_norm: tensor("attn_q_norm.weight")?,
                key_norm: tensor("attn_k_norm.weight")?,
                post_attention_norm: tensor("post_attention_norm.weight")?,
                feed_forward_norm: tensor("ffn_norm.weight")?,
                feed_forward_gate: tensor("ffn_gate.weight")?,
                feed_forward_up: tensor("ffn_up.weight")?,
                feed_forward_down: tensor("ffn_down.weight")?,
                post_feed_forward_norm: tensor("post_ffw_norm.weight")?,
                sliding_attention: weights.config.sliding_layers[layer as usize],
            });
        }
        Ok(Self {
            config: &weights.config,
            token_embedding: weights.require_tensor("token_embd.weight")?,
            output_norm: weights.require_tensor("output_norm.weight")?,
            output: weights.require_tensor("output.weight")?,
            layers,
        })
    }
}

fn validate_release_dtype_census(
    tensors: &[TensorDesc],
    profile: MuseGlimmerArtifactProfile,
) -> Result<MuseGlimmerDtypeCensus, MuseGlimmerResidencyError> {
    let mut census = MuseGlimmerDtypeCensus::default();
    for desc in tensors {
        census.record(desc.dtype)?;
    }
    let expected = expected_dtype_census(profile);
    if census != expected {
        return invalid(format!(
            "released {:?} dtype census mismatch: expected {expected:?}, got {census:?}",
            profile
        ));
    }
    Ok(census)
}

fn expected_dtype_census(profile: MuseGlimmerArtifactProfile) -> MuseGlimmerDtypeCensus {
    match profile {
        MuseGlimmerArtifactProfile::UnslothQ8_0 => MuseGlimmerDtypeCensus::UNSLOTH_Q8_0,
        MuseGlimmerArtifactProfile::UnslothBf16 => MuseGlimmerDtypeCensus::UNSLOTH_BF16,
    }
}

fn validate_retained_policy(plan: &RetainedStoragePlan) -> Result<(), MuseGlimmerResidencyError> {
    if plan.entries.len() != RELEASE_TENSOR_COUNT {
        return invalid(format!(
            "retained planner produced {} entries, expected {RELEASE_TENSOR_COUNT}",
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
) -> Result<u64, MuseGlimmerResidencyError> {
    if logical_bytes == 0 {
        return invalid(format!("planned Metal buffer {name:?} has zero bytes"));
    }
    let max_buffer_length = u64::try_from(ctx.max_buffer_length()).map_err(|_| {
        MuseGlimmerResidencyError::Invalid("Metal maximum buffer length exceeds u64".into())
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
    let alignment = priced.alignment.max(host_page_size_bytes()? as u64);
    priced
        .size
        .checked_add(alignment - 1)
        .map(|bytes| bytes / alignment * alignment)
        .ok_or_else(|| {
            MuseGlimmerResidencyError::Invalid(format!(
                "aligned Metal pricing for {name:?} overflows u64"
            ))
        })
}

fn build_weight_memory_plan(
    ctx: &MetalContext,
    retained: &RetainedStoragePlan,
    report: &MuseGlimmerMetalWeightPlanReport,
) -> Result<MuseGlimmerMetalWeightMemoryPlan, MuseGlimmerResidencyError> {
    let mut buffer_count = 0_usize;
    let mut logical_bytes = 0_u64;
    let mut priced_upper_bytes = 0_u64;
    for (index, window) in retained.windows.iter().enumerate() {
        let logical = u64::try_from(window.length).map_err(|_| {
            MuseGlimmerResidencyError::Invalid("retained window length exceeds u64".into())
        })?;
        let priced = price_shared_buffer(ctx, logical, &format!("weight_window[{index}]"))?;
        buffer_count = buffer_count.checked_add(1).ok_or_else(|| {
            MuseGlimmerResidencyError::Invalid("weight buffer count overflow".into())
        })?;
        logical_bytes = logical_bytes.checked_add(logical).ok_or_else(|| {
            MuseGlimmerResidencyError::Invalid("logical weight byte count overflow".into())
        })?;
        priced_upper_bytes = priced_upper_bytes.checked_add(priced).ok_or_else(|| {
            MuseGlimmerResidencyError::Invalid("priced weight byte count overflow".into())
        })?;
    }
    for entry in &retained.entries {
        if matches!(
            entry.disposition,
            RetainedStorageDisposition::CopyFallback { .. }
        ) {
            let priced = price_shared_buffer(ctx, entry.n_bytes, &entry.name)?;
            buffer_count = buffer_count.checked_add(1).ok_or_else(|| {
                MuseGlimmerResidencyError::Invalid("weight buffer count overflow".into())
            })?;
            logical_bytes = logical_bytes.checked_add(entry.n_bytes).ok_or_else(|| {
                MuseGlimmerResidencyError::Invalid("logical fallback byte count overflow".into())
            })?;
            priced_upper_bytes = priced_upper_bytes.checked_add(priced).ok_or_else(|| {
                MuseGlimmerResidencyError::Invalid("priced fallback byte count overflow".into())
            })?;
        }
    }

    let expected_buffers = report
        .planned_window_count
        .checked_add(report.fallback_count)
        .ok_or_else(|| MuseGlimmerResidencyError::Invalid("report buffer count overflow".into()))?;
    let expected_logical = report
        .planned_window_bytes
        .checked_add(report.fallback_bytes)
        .ok_or_else(|| MuseGlimmerResidencyError::Invalid("report weight bytes overflow".into()))?;
    if buffer_count != expected_buffers || logical_bytes != expected_logical {
        return invalid(format!(
            "weight memory plan differs from report: buffers={buffer_count}/{expected_buffers} logical={logical_bytes}/{expected_logical}"
        ));
    }
    if priced_upper_bytes < logical_bytes {
        return invalid("priced Metal weight bytes are below logical bytes");
    }
    Ok(MuseGlimmerMetalWeightMemoryPlan {
        buffer_count,
        logical_bytes,
        priced_upper_bytes,
    })
}

fn realize_windows(
    ctx: &MetalContext,
    gguf: &GgufFile,
    plan: &RetainedStoragePlan,
) -> Result<Vec<MetalGgufBacking>, MuseGlimmerResidencyError> {
    let mut windows = Vec::with_capacity(plan.windows.len());
    for (index, window) in plan.windows.iter().enumerate() {
        let mmap = gguf.retained_shard_mmap(window.shard_idx).ok_or_else(|| {
            MuseGlimmerResidencyError::Invalid(format!(
                "planned window {index} references missing shard {}",
                window.shard_idx
            ))
        })?;
        let mmap_offset = usize::try_from(window.mmap_offset).map_err(|_| {
            MuseGlimmerResidencyError::Invalid(format!(
                "planned window {index} offset exceeds usize"
            ))
        })?;
        let backing = ctx.gguf_no_copy_window(
            mmap,
            window.shard_idx,
            mmap_offset,
            window.length,
            MUSE_GLIMMER_GGUF_BINDING_ALIGNMENT,
        )?;
        if backing.mmap_offset() != mmap_offset
            || backing.exposed_len() != window.length
            || backing.required_alignment() != MUSE_GLIMMER_GGUF_BINDING_ALIGNMENT
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
) -> Result<BTreeMap<String, MetalTensor>, MuseGlimmerResidencyError> {
    if gguf.tensors.len() != plan.entries.len() {
        return invalid(format!(
            "realization descriptor count {} differs from planned {}",
            gguf.tensors.len(),
            plan.entries.len()
        ));
    }

    let mut tensors = BTreeMap::new();
    for (index, (entry, desc)) in plan.entries.iter().zip(&gguf.tensors).enumerate() {
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
                    MuseGlimmerResidencyError::Invalid(format!(
                        "tensor {:?} references missing window {window_index}",
                        desc.name
                    ))
                })?;
                let (eligibility, tensor) = backing.tensor(desc)?;
                let tensor = tensor.ok_or_else(|| {
                    MuseGlimmerResidencyError::Invalid(format!(
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
) -> Result<(), MuseGlimmerResidencyError> {
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
        MuseGlimmerResidencyError::Invalid(format!(
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
) -> Result<(), MuseGlimmerResidencyError> {
    if tensors.len() != RELEASE_TENSOR_COUNT {
        return invalid(format!(
            "realized map has {} tensors, expected {RELEASE_TENSOR_COUNT}",
            tensors.len()
        ));
    }
    for desc in &gguf.tensors {
        let tensor = tensors.get(&desc.name).ok_or_else(|| {
            MuseGlimmerResidencyError::Invalid(format!(
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

fn report_for_plan(
    gguf: &GgufFile,
    plan: &RetainedStoragePlan,
    device_max_buffer_length: usize,
    dtype_census: MuseGlimmerDtypeCensus,
    contents_authenticated: bool,
) -> Result<MuseGlimmerMetalWeightPlanReport, MuseGlimmerResidencyError> {
    let source_bytes = checked_sum(
        gguf.tensors.iter().map(|desc| desc.n_bytes),
        "source byte count",
    )?;
    let planned_entry_bytes = checked_sum(
        plan.entries.iter().map(|entry| entry.n_bytes),
        "planned source byte count",
    )?;
    if source_bytes != planned_entry_bytes {
        return invalid("source byte accounting differs from retained plan");
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

    Ok(MuseGlimmerMetalWeightPlanReport {
        source_tensor_count: gguf.tensors.len(),
        source_bytes,
        planned_window_count: plan.windows.len(),
        planned_window_bytes,
        view_count,
        fallback_count,
        fallback_bytes: plan.unique_fallback_bytes,
        page_size: plan.page_size,
        device_max_buffer_length,
        planning_max_buffer_length: plan.max_buffer_length,
        dtype_census,
        contents_authenticated,
    })
}

fn expected_release_shards(profile: MuseGlimmerArtifactProfile) -> &'static [ReleaseShardIdentity] {
    match profile {
        MuseGlimmerArtifactProfile::UnslothQ8_0 => &UNSLOTH_Q8_SHARDS,
        MuseGlimmerArtifactProfile::UnslothBf16 => &UNSLOTH_BF16_SHARDS,
    }
}

fn validate_release_shard_sizes(
    gguf: &GgufFile,
    expected: &[ReleaseShardIdentity],
) -> Result<(), MuseGlimmerResidencyError> {
    let observed = gguf.shard_mapped_lengths();
    if observed.len() != expected.len() {
        return invalid(format!(
            "pinned Unsloth revision {MUSE_GLIMMER_UNSLOTH_REVISION} requires {} shards, got {}",
            expected.len(),
            observed.len()
        ));
    }
    for (index, (&observed, expected)) in observed.iter().zip(expected).enumerate() {
        if observed as u64 != expected.size {
            return invalid(format!(
                "pinned shard {index} requires {} bytes, got {observed}",
                expected.size
            ));
        }
    }
    Ok(())
}

fn authenticate_release_shards(
    gguf: &GgufFile,
    expected: &[ReleaseShardIdentity],
) -> Result<(), MuseGlimmerResidencyError> {
    for (index, expected) in expected.iter().enumerate() {
        let mmap = gguf.retained_shard_mmap(index).ok_or_else(|| {
            MuseGlimmerResidencyError::Invalid(format!(
                "pinned shard {index} is unavailable for authentication"
            ))
        })?;
        let observed: [u8; 32] = Sha256::digest(&mmap[..]).into();
        if observed != expected.sha256 {
            return invalid(format!(
                "pinned shard {index} SHA-256 mismatch: expected {}, got {}",
                hex_digest(expected.sha256),
                hex_digest(observed)
            ));
        }
    }
    Ok(())
}

fn preflight_authentication_admission(
    ctx: &MetalContext,
    expected: &[ReleaseShardIdentity],
) -> Result<(), MuseGlimmerResidencyError> {
    let source_bytes = checked_sum(
        expected.iter().map(|identity| identity.size),
        "authenticated source byte count",
    )?;
    let admission = evaluate_metal_memory_admission(source_bytes, 0, ctx.memory_signals(), true);
    if !admission.admitted {
        return invalid(format!(
            "Metal pre-admission denied before content authentication: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
            admission.reason.as_str(),
            admission.required_bytes,
            admission.working_set_headroom_bytes,
            admission.signals.process_limit_remaining_bytes
        ));
    }
    Ok(())
}

fn hex_digest(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn checked_sum(
    values: impl IntoIterator<Item = u64>,
    label: &str,
) -> Result<u64, MuseGlimmerResidencyError> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| MuseGlimmerResidencyError::Invalid(format!("{label} overflow")))
    })
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, MuseGlimmerResidencyError> {
    Err(MuseGlimmerResidencyError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_dtype_censuses_cover_exact_tensor_inventory() {
        assert_eq!(MuseGlimmerDtypeCensus::UNSLOTH_Q8_0.total(), 731);
        assert_eq!(MuseGlimmerDtypeCensus::UNSLOTH_BF16.total(), 731);
        assert_eq!(
            expected_dtype_census(MuseGlimmerArtifactProfile::UnslothQ8_0),
            MuseGlimmerDtypeCensus::UNSLOTH_Q8_0
        );
        assert_eq!(
            expected_dtype_census(MuseGlimmerArtifactProfile::UnslothBf16),
            MuseGlimmerDtypeCensus::UNSLOTH_BF16
        );
    }

    #[test]
    #[ignore = "requires the pinned local Unsloth Muse Glimmer Q8_0 GGUF and Metal"]
    fn plans_pinned_q8_target_without_realizing_weights() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF").unwrap_or_else(|_| {
            "/Users/tito/models/muse-glimmer/Muse-Glimmer-30B-Q8_0.gguf".into()
        });
        let gguf = GgufFile::open(path).expect("open Muse Q8 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let plan = MuseGlimmerMetalWeightPlan::for_authenticated_release(&ctx, &gguf)
            .expect("authenticate and plan Muse Q8 residency");
        assert_eq!(
            plan.artifact_profile(),
            MuseGlimmerArtifactProfile::UnslothQ8_0
        );
        assert_eq!(plan.report().source_tensor_count, 731);
        assert_eq!(plan.report().dtype_census.q8_0(), 418);
        assert!(plan.contents_authenticated());
    }

    #[test]
    #[ignore = "requires the pinned local two-shard Unsloth Muse Glimmer BF16 GGUF and Metal"]
    fn plans_pinned_bf16_target_without_realizing_weights() {
        let path = std::env::var("MUSE_GLIMMER_BF16_GGUF").unwrap_or_else(|_| {
            "/Users/tito/models/muse-glimmer/BF16/Muse-Glimmer-30B-BF16-00001-of-00002.gguf".into()
        });
        let gguf = GgufFile::open(path).expect("open Muse BF16 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let plan = MuseGlimmerMetalWeightPlan::for_authenticated_release(&ctx, &gguf)
            .expect("authenticate and plan Muse BF16 residency");
        assert_eq!(
            plan.artifact_profile(),
            MuseGlimmerArtifactProfile::UnslothBf16
        );
        assert_eq!(plan.report().source_tensor_count, 731);
        assert_eq!(plan.report().dtype_census.bf16(), 418);
        assert!(plan.contents_authenticated());
        assert_eq!(gguf.shard_count(), 2);
    }
}
