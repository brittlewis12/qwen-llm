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
pub const DEEPSEEK_V4_SINKHORN_ITERATIONS: usize = 20;
pub const DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
pub use prefill::{DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS, DEEPSEEK_V4_PREFILL_MAX_TOKENS};

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

crate::env_flag!(
    default_off deepseek_v4_residency_set_enabled,
    "QWEN_DSV4_RESIDENCY_SET"
);

const DEEPSEEK_V4_HIDDEN_SIZE: usize = 4_096;
const DEEPSEEK_V4_VOCAB_SIZE: usize = 129_280;
const DEEPSEEK_V4_LAYER_COUNT: usize = 43;
const DEEPSEEK_V4_LOCAL_WINDOW: usize = 128;
const DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS: usize = 256;

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

#[derive(Clone, Copy)]
struct DeepSeekV4PublishedRows<'a> {
    cache: &'a MetalTensor,
    count: usize,
    capacity_rows: usize,
}

#[cfg(not(test))]
const DEEPSEEK_V4_F16_MATRIX_SCORER_MIN_VISIBLE_ROWS: usize = 16_384;
#[cfg(not(test))]
const DEEPSEEK_V4_F16_MATRIX_SCORER_QUALIFIED_DEVICE: &str = "Apple M4 Max";

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

#[cfg(all(test, feature = "dsv4-diagnostics"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeepSeekV4SelectorTestPolicy {
    Production,
    BitwiseOracle,
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

crate::env_flag!(
    default_on deepseek_v4_all_slots_q3q4_enabled,
    "QWEN_DSV4_ALL_SLOTS_Q3Q4"
);

crate::env_flag!(
    default_on deepseek_v4_all_slots_q3q4_fast_enabled,
    "QWEN_DSV4_ALL_SLOTS_Q3Q4_FAST"
);

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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DeepSeekV4AllSlotsArgs {
    n_in: u32,
    n_out: u32,
    n_expert: u32,
    top_k: u32,
    clamp: f32,
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

const DEEPSEEK_V4_GROUPED_DENSE_HEADS: usize = 8;
const DEEPSEEK_V4_GROUPED_DENSE_THREADS: usize = 256;
const DEEPSEEK_V4_GROUPED_DENSE_THREADGROUP_BYTES: usize = DEEPSEEK_V4_GROUPED_DENSE_STAGED_ROWS
    * DEEPSEEK_V4_HCA_TILE_ROWS
    * std::mem::size_of::<half::f16>();

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

fn invalid<T>(detail: impl Into<String>) -> Result<T, DeepSeekV4MetalError> {
    Err(DeepSeekV4MetalError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests;

mod attention;
mod indexer;
mod moe;
mod records;
mod residency;
mod session;
#[allow(unused_imports)]
pub use attention::*;
#[allow(unused_imports)]
pub use indexer::*;
#[allow(unused_imports)]
pub use moe::*;
#[allow(unused_imports)]
pub use records::*;
#[allow(unused_imports)]
pub use residency::*;
#[allow(unused_imports)]
pub use session::*;
