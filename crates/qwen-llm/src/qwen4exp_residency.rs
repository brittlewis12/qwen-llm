//! Zero-allocation Metal weight planning for Qwen3.8-Flash-Next.
//!
//! The released PLE table remains CPU row-addressed. Other weights use
//! retained, read-only GGUF views where page geometry permits; copied
//! final-partial-page fallbacks are explicit in the report. No Metal buffers
//! are created until a later admitted realization step.

use crate::gguf::{GgufError, GgufFile, GgufShardStamp};
use crate::metal::{
    MetalContext, MetalError, RetainedStorageDisposition, RetainedStorageFallback,
    RetainedStoragePlan, RetainedStorageWindow, host_page_size_bytes, plan_retained_storage,
};
use crate::qwen4exp::Qwen4ExpConfig;
use crate::qwen4exp_loader::{Qwen4ExpLoadError, Qwen4ExpModel};
use crate::qwen4exp_ple::{PleGatherError, PleIq4NlTable};
use crate::tensor::{GgmlType, TensorDesc};
use objc2_metal::MTLDevice;

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
    device_registry_id: u64,
    shard_stamps: Vec<GgufShardStamp>,
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

    pub fn ple_source(&self) -> &Qwen4ExpPleSourcePlan {
        &self.ple_source
    }

    pub fn device_registry_id(&self) -> u64 {
        self.device_registry_id
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
            || self.shard_stamps != rebuilt.shard_stamps
        {
            return invalid("weight plan changed before realization");
        }
        Ok(())
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
