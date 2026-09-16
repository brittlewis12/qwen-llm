//! Stateful one-token, text-only Metal Qwen Sparse Attention for
//! Qwen3.8-Flash-Next.
//!
//! Plain NEOX RoPE is valid only because all IMRoPE axes coincide for text.
//! F32 pending index keys with an explicit pooled F16 rounding point, F16
//! compressed index keys, and F16 main K/V caches are this Metal checkpoint's
//! numerical contract. It intentionally differs from BF16 serving backends.
//! Packed attention preserves that state contract while qualifying the
//! F16-staged dense and selected kernels against chronological scalar execution.

#[cfg(test)]
use crate::metal::encode_copy_offset_i32;
use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, MetalTensorProvenance,
    encode_attn_matrix_kq_f32, encode_attn_matrix_kqv_direct_v_f32, encode_attn_matrix_softmax_f32,
    encode_copy_offset_f32, encode_mat_mat_bf16_f32, encode_mat_vec_q8_0_batch_f32,
    encode_qk_rms_norm_rope_f32_packed_consecutive, encode_scatter_offset_f32_to_f16_kv,
    encode_sigmoid_mul_gate_strided_f32, mat_vec_q8_0_lcpp_enabled,
};
use crate::metal_forward::{
    MfError, encode_mat_mat_dispatch, encode_mat_vec_dispatch, validate_f32_q8_mat_mat_addressing,
};
use crate::qwen4exp::{MixerKind, Qwen4ExpConfig};
use crate::qwen4exp_profile::{
    Qwen4ExpPackedProfileLabel, Qwen4ExpPackedProfileRecorder, begin_optional, end_optional,
};
use crate::qwen4exp_residency::{Qwen4ExpMetalWeights, Qwen4ExpResidencyError};
use crate::tensor::{GgmlType, ggml_type_layout};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLComputePipelineState, MTLDevice,
    MTLResource, MTLSize,
};

const INDEX_HEAD_DIM: usize = 128;
const INDEX_QUERY_HEADS: usize = 4;
const MAIN_HEAD_DIM: usize = 256;
const ATTENTION_THREADS: usize = 256;
const ATTENTION_SCRATCH_FLOATS: usize = 9;
const ATTENTION_SCRATCH_BYTES: usize =
    (ATTENTION_SCRATCH_FLOATS * size_of::<f32>()).next_multiple_of(16);
const LOGITS_SIMDGROUPS_PER_TG: usize = 8;
const PACKED_ATTENTION_HEADS_PER_TG: usize = 4;
const PACKED_ATTENTION_THREADS: usize = 128;
const SELECTED_COUNT_MISMATCH_STATUS: i32 = 4;
const SELECTED_AUDIT_ORDER_MISMATCH_STATUS: i32 = 5;
const SELECTOR_SCRATCH_BYTES: usize = 2 * ATTENTION_THREADS * size_of::<u32>();
const DENSE_PACKED_QUERY_TILE: usize = 32;

#[cfg(test)]
pub(crate) mod split_decode;
#[cfg(test)]
pub(crate) mod split_decode_probe;

crate::env_flag!(
    default_on configured_qwen4exp_qsa_gqa4_logits_enabled,
    "QWEN4EXP_QSA_GQA4_LOGITS"
);
crate::env_flag!(
    default_on configured_qwen4exp_qsa_gqa4_value_enabled,
    "QWEN4EXP_QSA_GQA4_VALUE"
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QwenSparseAttentionPackedRangePlan {
    end_position: usize,
    dense_tokens: usize,
    // Chunk-local offset of the first query that requires block selection.
    selected_offset: usize,
    selected_tokens: usize,
    selected_bands: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen4ExpQsaDecisionCaptureRecord {
    pub layer: u32,
    pub start_position: usize,
    pub query_count: usize,
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct Qwen4ExpQsaDecisionCaptureBanks {
    pub layers: Vec<u32>,
    pub start_position: usize,
    pub tokens: usize,
    pub hidden_size: usize,
    pub index_query_width: usize,
    pub block_capacity: usize,
    pub block_budget: usize,
    pub inputs: MetalTensor,
    pub index_queries: MetalTensor,
    pub scores: MetalTensor,
    pub visible_blocks: MetalTensor,
    pub selected_blocks: MetalTensor,
    pub selected_count: MetalTensor,
    pub selector_status: MetalTensor,
}

#[cfg(test)]
impl Qwen4ExpQsaDecisionCaptureBanks {
    pub(crate) fn new(
        ctx: &MetalContext,
        config: &Qwen4ExpConfig,
        capacity: usize,
        start_position: usize,
        tokens: usize,
    ) -> Result<Self, Qwen4ExpQsaError> {
        let layers = (0..config.layer_count)
            .filter(|&layer| config.mixer_kind(layer) == Some(MixerKind::QwenSparseAttention))
            .collect::<Vec<_>>();
        let Some(&first_layer) = layers.first() else {
            return invalid("QSA decision capture requires at least one QSA layer");
        };
        if tokens == 0 {
            return invalid("QSA decision capture token count must be nonzero");
        }
        let end_position = start_position.checked_add(tokens).ok_or_else(|| {
            Qwen4ExpQsaError::Invalid("QSA decision capture range overflow".into())
        })?;
        if end_position > capacity {
            return invalid(format!(
                "QSA decision capture end {end_position} exceeds capacity {capacity}"
            ));
        }
        let geometry =
            QwenSparseAttentionMetalGeometry::from_config(config, first_layer, capacity)?;
        if start_position < geometry.output_width() {
            return invalid(format!(
                "QSA decision capture start {start_position} precedes selected range {}",
                geometry.output_width()
            ));
        }
        for &layer in &layers[1..] {
            let candidate = QwenSparseAttentionMetalGeometry::from_config(config, layer, capacity)?;
            if candidate != geometry {
                return invalid(format!(
                    "QSA decision capture layer {layer} geometry differs from layer {first_layer}"
                ));
            }
        }
        let layer_count = layers.len();
        let checked_elements = |name: &str, factors: &[usize]| {
            factors
                .iter()
                .try_fold(1_usize, |product, &factor| product.checked_mul(factor))
                .ok_or_else(|| {
                    Qwen4ExpQsaError::Invalid(format!(
                        "QSA decision capture {name} element count overflow"
                    ))
                })
        };
        for (name, factors) in [
            (
                "inputs",
                [geometry.hidden_size(), tokens, layer_count].as_slice(),
            ),
            (
                "index queries",
                [geometry.index_query_width(), tokens, layer_count].as_slice(),
            ),
            (
                "scores",
                [geometry.block_capacity(), tokens, layer_count].as_slice(),
            ),
            (
                "selected blocks",
                [geometry.block_budget(), tokens, layer_count].as_slice(),
            ),
        ] {
            let elements = checked_elements(name, factors)?;
            if u32::try_from(elements).is_err() {
                return invalid(format!(
                    "QSA decision capture {name} has {elements} elements, exceeding u32"
                ));
            }
        }
        let rows = checked_elements("control rows", &[tokens, layer_count])?;
        if u32::try_from(rows).is_err() {
            return invalid(format!(
                "QSA decision capture has {rows} control rows, exceeding u32"
            ));
        }
        Ok(Self {
            layers,
            start_position,
            tokens,
            hidden_size: geometry.hidden_size(),
            index_query_width: geometry.index_query_width(),
            block_capacity: geometry.block_capacity(),
            block_budget: geometry.block_budget(),
            inputs: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.hidden_size() as u64,
                    tokens as u64,
                    layer_count as u64,
                ],
            )?,
            index_queries: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.index_query_width() as u64,
                    tokens as u64,
                    layer_count as u64,
                ],
            )?,
            scores: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.block_capacity() as u64,
                    tokens as u64,
                    layer_count as u64,
                ],
            )?,
            visible_blocks: MetalTensor::zeros_i32(ctx, vec![tokens as u64, layer_count as u64])?,
            selected_blocks: MetalTensor::zeros_i32(
                ctx,
                vec![
                    geometry.block_budget() as u64,
                    tokens as u64,
                    layer_count as u64,
                ],
            )?,
            selected_count: MetalTensor::zeros_i32(ctx, vec![tokens as u64, layer_count as u64])?,
            selector_status: MetalTensor::zeros_i32(ctx, vec![tokens as u64, layer_count as u64])?,
        })
    }

    fn layer_ordinal(&self, layer: u32) -> Option<usize> {
        self.layers.iter().position(|&candidate| candidate == layer)
    }
}

#[cfg(test)]
#[derive(Clone)]
struct Qwen4ExpQsaDecisionCaptureBinding {
    banks: Qwen4ExpQsaDecisionCaptureBanks,
    records: std::rc::Rc<std::cell::RefCell<Vec<Qwen4ExpQsaDecisionCaptureRecord>>>,
    seen: std::rc::Rc<std::cell::RefCell<Vec<bool>>>,
}

#[cfg(test)]
thread_local! {
    static QWEN4EXP_QSA_CAPTURE_LAYER: std::cell::Cell<Option<u32>> = const {
        std::cell::Cell::new(None)
    };
    static QWEN4EXP_QSA_DECISION_CAPTURE: std::cell::RefCell<Option<Qwen4ExpQsaDecisionCaptureBinding>> = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(test)]
pub(crate) fn with_qwen4exp_qsa_decision_capture<R>(
    banks: &Qwen4ExpQsaDecisionCaptureBanks,
    f: impl FnOnce() -> R,
) -> (R, Vec<Qwen4ExpQsaDecisionCaptureRecord>) {
    struct RestoreCapture(Option<Qwen4ExpQsaDecisionCaptureBinding>);

    impl Drop for RestoreCapture {
        fn drop(&mut self) {
            QWEN4EXP_QSA_DECISION_CAPTURE.with(|slot| {
                *slot.borrow_mut() = self.0.take();
            });
        }
    }

    QWEN4EXP_QSA_CAPTURE_LAYER.with(|slot| {
        assert!(
            slot.get().is_none(),
            "QSA decision capture cannot begin inside a layer"
        );
    });
    QWEN4EXP_QSA_DECISION_CAPTURE.with(|slot| {
        assert!(slot.borrow().is_none(), "QSA decision captures cannot nest");
    });
    let records = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let seen = std::rc::Rc::new(std::cell::RefCell::new(vec![
        false;
        banks.layers.len()
            * banks.tokens
    ]));
    let previous = QWEN4EXP_QSA_DECISION_CAPTURE.with(|slot| {
        slot.borrow_mut()
            .replace(Qwen4ExpQsaDecisionCaptureBinding {
                banks: banks.clone(),
                records: records.clone(),
                seen: seen.clone(),
            })
    });
    debug_assert!(previous.is_none());
    let _restore = RestoreCapture(previous);
    let result = f();
    let records = records.borrow().clone();
    (result, records)
}

#[cfg(test)]
pub(crate) fn qwen4exp_qsa_decision_capture_active() -> bool {
    QWEN4EXP_QSA_DECISION_CAPTURE.with(|slot| slot.borrow().is_some())
}

#[inline(always)]
pub(crate) fn with_qwen4exp_qsa_capture_layer<R>(layer: u32, f: impl FnOnce() -> R) -> R {
    #[cfg(test)]
    {
        struct RestoreLayer(Option<u32>);

        impl Drop for RestoreLayer {
            fn drop(&mut self) {
                QWEN4EXP_QSA_CAPTURE_LAYER.with(|slot| slot.set(self.0));
            }
        }

        let previous = QWEN4EXP_QSA_CAPTURE_LAYER.with(|slot| {
            let previous = slot.get();
            slot.set(Some(layer));
            previous
        });
        let _restore = RestoreLayer(previous);
        return f();
    }
    #[cfg(not(test))]
    {
        let _ = layer;
        f()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpQsaError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Forward(#[from] MfError),
    #[error(transparent)]
    Residency(#[from] Qwen4ExpResidencyError),
    #[error("invalid Qwen3.8-Flash-Next QSA contract: {0}")]
    Invalid(String),
    #[error("Qwen3.8-Flash-Next QSA command buffer failed: {0}")]
    CommandBuffer(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QwenSparseAttentionMetalGeometry {
    hidden_size: usize,
    index_query_heads: usize,
    index_head_dim: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    token_budget: usize,
    ratio: usize,
    capacity: usize,
    theta: f32,
    eps: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenSparseAttentionWorkspaceByteEstimate {
    pub main_key_cache_bytes: usize,
    pub main_value_cache_bytes: usize,
    pub compressed_index_cache_bytes: usize,
    pub pending_index_key_bytes: usize,
    pub logits_bytes: usize,
    pub total_workspace_bytes: usize,
}

impl QwenSparseAttentionMetalGeometry {
    #[cfg(test)]
    pub(crate) fn supports_split_decode(self) -> bool {
        (
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.token_budget,
            self.ratio,
        ) == (24, 2, 256, 2048, 4)
    }
    pub fn from_config(
        config: &Qwen4ExpConfig,
        layer: u32,
        capacity: usize,
    ) -> Result<Self, Qwen4ExpQsaError> {
        if config.mixer_kind(layer) != Some(MixerKind::QwenSparseAttention) {
            return invalid(format!("layer {layer} is not a QSA layer"));
        }
        if config.qsa.key_heads != 1 {
            return invalid("QSA requires exactly one shared index key head");
        }
        if config.attention.key_head_dim != config.attention.value_head_dim {
            return invalid("QSA main key and value head dimensions must match");
        }
        let ratio =
            *config.compress_ratios.get(layer as usize).ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(format!("layer {layer} is out of range"))
            })? as usize;
        let geometry = Self {
            hidden_size: config.hidden_size as usize,
            index_query_heads: config.qsa.query_heads as usize,
            index_head_dim: config.qsa.head_dim as usize,
            query_heads: config.attention.query_heads as usize,
            kv_heads: config.attention.kv_heads as usize,
            head_dim: config.attention.key_head_dim as usize,
            rotary_dim: config.attention.rotary_dim as usize,
            token_budget: config.qsa.token_budget as usize,
            ratio,
            capacity,
            theta: config.attention.rope_theta,
            eps: config.rms_norm_eps,
        };
        geometry.validate(config.context_length as usize)?;
        Ok(geometry)
    }

    fn validate(self, context_length: usize) -> Result<(), Qwen4ExpQsaError> {
        if self.hidden_size == 0
            || self.index_query_heads == 0
            || self.query_heads == 0
            || self.kv_heads == 0
            || self.ratio == 0
            || self.token_budget == 0
            || self.capacity == 0
        {
            return invalid("QSA dimensions, budget, ratio, and capacity must be nonzero");
        }
        if self.index_head_dim != INDEX_HEAD_DIM {
            return invalid(format!(
                "QSA index head dimension must be {INDEX_HEAD_DIM}, got {}",
                self.index_head_dim
            ));
        }
        if self.index_query_heads != INDEX_QUERY_HEADS {
            return invalid(format!(
                "QSA index query head count must be {INDEX_QUERY_HEADS}, got {}",
                self.index_query_heads
            ));
        }
        if self.head_dim != MAIN_HEAD_DIM {
            return invalid(format!(
                "QSA main head dimension must be {MAIN_HEAD_DIM}, got {}",
                self.head_dim
            ));
        }
        if !self.query_heads.is_multiple_of(self.kv_heads) {
            return invalid("QSA query heads must be divisible by KV heads");
        }
        self.index_query_heads
            .checked_mul(self.index_head_dim)
            .and_then(|_| self.query_heads.checked_mul(self.head_dim))
            .and_then(|width| width.checked_mul(2))
            .and_then(|_| self.kv_heads.checked_mul(self.head_dim))
            .ok_or_else(|| Qwen4ExpQsaError::Invalid("QSA head geometry overflow".into()))?;
        if self.rotary_dim == 0
            || self.rotary_dim > self.index_head_dim.min(self.head_dim)
            || !self.rotary_dim.is_multiple_of(2)
        {
            return invalid("QSA rotary dimension must be nonzero, even, and fit both head widths");
        }
        if !self.token_budget.is_multiple_of(self.ratio) {
            return invalid("QSA token budget must be divisible by the compression ratio");
        }
        if !self.capacity.is_multiple_of(self.ratio) || self.capacity > context_length {
            return invalid(format!(
                "QSA capacity {} must be ratio-aligned and no larger than context {context_length}",
                self.capacity
            ));
        }
        if !self.theta.is_finite() || self.theta <= 0.0 || !self.eps.is_finite() || self.eps <= 0.0
        {
            return invalid("QSA theta and RMS epsilon must be finite and positive");
        }
        let values = [
            self.hidden_size,
            self.index_query_heads,
            self.index_head_dim,
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.token_budget,
            self.ratio,
            self.capacity,
            self.block_budget(),
            self.block_capacity(),
            self.output_width(),
            self.index_query_width(),
            self.query_width(),
            self.kv_width(),
        ];
        if values
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
        {
            return invalid("QSA geometry exceeds Metal u32 arguments");
        }
        if self.capacity > i32::MAX as usize + 1 || self.block_capacity() > i32::MAX as usize + 1 {
            return invalid("QSA logical token and block IDs must fit signed i32");
        }
        let shader_products = [
            (
                "index query projection",
                self.hidden_size.checked_mul(self.index_query_width()),
            ),
            (
                "index key projection",
                self.hidden_size.checked_mul(self.index_head_dim),
            ),
            (
                "query/gate projection",
                self.hidden_size.checked_mul(self.query_projection_width()),
            ),
            (
                "key/value projection",
                self.hidden_size.checked_mul(self.kv_width()),
            ),
            (
                "output projection",
                self.query_width().checked_mul(self.hidden_size),
            ),
            (
                "pending index keys",
                self.ratio.checked_mul(self.index_head_dim),
            ),
            (
                "compressed index keys",
                self.block_capacity().checked_mul(self.index_head_dim),
            ),
            ("main cache", self.capacity.checked_mul(self.kv_width())),
            (
                "attention logits",
                self.output_width().checked_mul(self.query_heads),
            ),
        ];
        for (name, product) in shader_products {
            let product = product
                .ok_or_else(|| Qwen4ExpQsaError::Invalid(format!("QSA {name} offset overflow")))?;
            if u32::try_from(product).is_err() {
                return invalid(format!("QSA {name} offsets exceed u32"));
            }
        }
        self.checked_workspace_byte_estimate()?;
        Ok(())
    }

    pub fn checked_workspace_byte_estimate(
        self,
    ) -> Result<QwenSparseAttentionWorkspaceByteEstimate, Qwen4ExpQsaError> {
        fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, Qwen4ExpQsaError> {
            left.checked_mul(right).ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(format!("QSA {name} byte arithmetic overflow"))
            })
        }
        fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, Qwen4ExpQsaError> {
            left.checked_add(right).ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(format!("QSA {name} byte arithmetic overflow"))
            })
        }

        let cache_elements = checked_mul(self.capacity, self.kv_width(), "main cache elements")?;
        let main_key_cache_bytes = checked_mul(cache_elements, 2, "main key cache")?;
        let main_value_cache_bytes = checked_mul(cache_elements, 2, "main value cache")?;
        let compressed_index_cache_bytes = checked_mul(
            checked_mul(
                self.block_capacity(),
                self.index_head_dim,
                "compressed index elements",
            )?,
            2,
            "compressed index cache",
        )?;
        let pending_index_key_bytes = checked_mul(
            checked_mul(self.ratio, self.index_head_dim, "pending index elements")?,
            4,
            "pending index keys",
        )?;
        let logits_bytes = checked_mul(
            checked_mul(
                self.output_width(),
                self.query_heads,
                "attention logits elements",
            )?,
            4,
            "attention logits",
        )?;
        let f32_elements = [
            self.index_query_width(),
            self.index_query_width(),
            self.index_head_dim,
            self.block_capacity(),
            self.query_projection_width(),
            self.query_width(),
            self.query_width(),
            self.kv_width(),
            self.kv_width(),
            self.kv_width(),
            self.query_width(),
            self.hidden_size,
        ]
        .into_iter()
        .try_fold(0_usize, |total, elements| {
            checked_add(total, elements, "F32 workspace elements")
        })?;
        let i32_elements = [self.block_budget(), self.output_width(), 1, 1, 1]
            .into_iter()
            .try_fold(0_usize, |total, elements| {
                checked_add(total, elements, "I32 workspace elements")
            })?;
        let other_bytes = checked_add(
            checked_mul(f32_elements, 4, "F32 workspace")?,
            checked_mul(i32_elements, 4, "I32 workspace")?,
            "non-cache workspace",
        )?;
        let total_workspace_bytes = [
            main_key_cache_bytes,
            main_value_cache_bytes,
            compressed_index_cache_bytes,
            pending_index_key_bytes,
            logits_bytes,
            other_bytes,
        ]
        .into_iter()
        .try_fold(0_usize, |total, bytes| {
            checked_add(total, bytes, "total workspace")
        })?;
        Ok(QwenSparseAttentionWorkspaceByteEstimate {
            main_key_cache_bytes,
            main_value_cache_bytes,
            compressed_index_cache_bytes,
            pending_index_key_bytes,
            logits_bytes,
            total_workspace_bytes,
        })
    }

    pub fn workspace_logical_allocations(self) -> Result<Vec<usize>, Qwen4ExpQsaError> {
        fn bytes(elements: Option<usize>, width: usize) -> Result<usize, Qwen4ExpQsaError> {
            elements
                .and_then(|elements| elements.checked_mul(width))
                .ok_or_else(|| {
                    Qwen4ExpQsaError::Invalid("QSA workspace allocation overflow".into())
                })
        }
        let f32_bytes = |elements| bytes(Some(elements), size_of::<f32>());
        let i32_bytes = |elements| bytes(Some(elements), size_of::<i32>());
        let f16_bytes = |elements| bytes(Some(elements), size_of::<u16>());
        Ok(vec![
            f32_bytes(self.index_query_width())?,
            f32_bytes(self.index_query_width())?,
            f32_bytes(self.index_head_dim)?,
            bytes(
                self.index_head_dim.checked_mul(self.ratio),
                size_of::<f32>(),
            )?,
            bytes(
                self.index_head_dim.checked_mul(self.block_capacity()),
                size_of::<u16>(),
            )?,
            f32_bytes(self.block_capacity())?,
            i32_bytes(1)?,
            i32_bytes(self.block_budget())?,
            i32_bytes(1)?,
            i32_bytes(1)?,
            i32_bytes(self.output_width())?,
            bytes(
                self.output_width().checked_mul(self.query_heads),
                size_of::<f32>(),
            )?,
            f32_bytes(self.query_projection_width())?,
            f32_bytes(self.query_width())?,
            f32_bytes(self.query_width())?,
            f32_bytes(self.kv_width())?,
            f32_bytes(self.kv_width())?,
            f32_bytes(self.kv_width())?,
            f16_bytes(self.capacity.checked_mul(self.kv_width()).ok_or_else(|| {
                Qwen4ExpQsaError::Invalid("QSA key cache allocation overflow".into())
            })?)?,
            f16_bytes(self.capacity.checked_mul(self.kv_width()).ok_or_else(|| {
                Qwen4ExpQsaError::Invalid("QSA value cache allocation overflow".into())
            })?)?,
            f32_bytes(self.query_width())?,
            f32_bytes(self.hidden_size)?,
        ])
    }

    pub(crate) fn packed_scratch_logical_allocations(
        self,
        capacity: usize,
    ) -> Result<Vec<usize>, Qwen4ExpQsaError> {
        if capacity == 0 || capacity > self.token_budget {
            return invalid(format!(
                "dense packed QSA capacity must be in 1..={}, got {capacity}",
                self.token_budget
            ));
        }
        let f32_bytes = |name: &str, factors: &[usize]| {
            factors
                .iter()
                .try_fold(1_usize, |product, &factor| product.checked_mul(factor))
                .and_then(|elements| elements.checked_mul(size_of::<f32>()))
                .ok_or_else(|| {
                    Qwen4ExpQsaError::Invalid(format!(
                        "dense packed QSA {name} allocation overflow"
                    ))
                })
        };
        Ok(vec![
            f32_bytes("index key", &[self.index_head_dim, capacity])?,
            f32_bytes(
                "query/gate projection",
                &[self.query_projection_width(), capacity],
            )?,
            f32_bytes("query", &[self.query_width(), capacity])?,
            f32_bytes("raw key", &[self.kv_width(), capacity])?,
            f32_bytes("key", &[self.kv_width(), capacity])?,
            f32_bytes("value", &[self.kv_width(), capacity])?,
            f32_bytes(
                "attention scores",
                &[self.packed_attention_score_elements(capacity)?],
            )?,
            f32_bytes("attention", &[self.query_width(), capacity])?,
            f32_bytes("output", &[self.hidden_size, capacity])?,
        ])
    }

    pub(crate) fn selected_packed_scratch_logical_allocations(
        self,
        capacity: usize,
    ) -> Result<Vec<usize>, Qwen4ExpQsaError> {
        if capacity == 0 || capacity > self.token_budget {
            return invalid(format!(
                "selected packed QSA capacity must be in 1..={}, got {capacity}",
                self.token_budget
            ));
        }
        let query_tile = capacity.min(DENSE_PACKED_QUERY_TILE);
        let bytes = |name: &str, factors: &[usize]| {
            let elements = factors
                .iter()
                .try_fold(1_usize, |product, &factor| product.checked_mul(factor))
                .ok_or_else(|| {
                    Qwen4ExpQsaError::Invalid(format!(
                        "selected packed QSA {name} element count overflow"
                    ))
                })?;
            if u32::try_from(elements).is_err() {
                return invalid(format!(
                    "selected packed QSA {name} has {elements} elements, exceeding u32"
                ));
            }
            elements.checked_mul(size_of::<u32>()).ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(format!("selected packed QSA {name} byte count overflow"))
            })
        };
        Ok(vec![
            bytes("raw index query", &[self.index_query_width(), capacity])?,
            bytes("index query", &[self.index_query_width(), capacity])?,
            bytes("scores", &[self.block_capacity(), query_tile])?,
            bytes("visible blocks", &[query_tile])?,
            bytes("selected blocks", &[self.block_budget(), query_tile])?,
            bytes("selected count", &[query_tile])?,
            bytes("selector status", &[query_tile])?,
            bytes("token IDs", &[self.output_width(), query_tile])?,
            bytes(
                "attention logits",
                &[self.output_width(), self.query_heads, query_tile],
            )?,
        ])
    }

    pub fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub fn compression_ratio(self) -> usize {
        self.ratio
    }

    pub fn capacity(self) -> usize {
        self.capacity
    }

    pub fn token_budget(self) -> usize {
        self.token_budget
    }

    pub fn block_budget(self) -> usize {
        self.token_budget / self.ratio
    }

    pub fn block_capacity(self) -> usize {
        self.capacity / self.ratio
    }

    pub fn output_width(self) -> usize {
        self.token_budget + self.ratio - 1
    }

    fn plan_packed_range(
        self,
        start_position: usize,
        tokens: usize,
    ) -> Result<QwenSparseAttentionPackedRangePlan, Qwen4ExpQsaError> {
        if tokens == 0 || tokens > self.token_budget {
            return invalid(format!(
                "packed QSA token count must be in 1..={}, got {tokens}",
                self.token_budget
            ));
        }
        let end_position = start_position.checked_add(tokens).ok_or_else(|| {
            Qwen4ExpQsaError::Invalid("packed QSA sequence length overflow".into())
        })?;
        if end_position > self.capacity {
            return invalid(format!(
                "packed QSA end {end_position} exceeds cache capacity {}",
                self.capacity
            ));
        }
        let dense_end = end_position.min(self.output_width());
        let dense_tokens = dense_end.saturating_sub(start_position).min(tokens);
        let selected_offset = dense_tokens;
        let selected_tokens = tokens - dense_tokens;
        let selected_bands = selected_tokens.div_ceil(DENSE_PACKED_QUERY_TILE);
        Ok(QwenSparseAttentionPackedRangePlan {
            end_position,
            dense_tokens,
            selected_offset,
            selected_tokens,
            selected_bands,
        })
    }

    fn packed_attention_score_elements(self, capacity: usize) -> Result<usize, Qwen4ExpQsaError> {
        if capacity == 0 || capacity > self.token_budget {
            return invalid(format!(
                "dense packed QSA capacity must be in 1..={}, got {capacity}",
                self.token_budget
            ));
        }
        let elements = self
            .output_width()
            .checked_mul(self.query_heads)
            .and_then(|elements| elements.checked_mul(capacity.min(DENSE_PACKED_QUERY_TILE)))
            .ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(
                    "dense packed QSA attention-score capacity overflow".into(),
                )
            })?;
        if u32::try_from(elements).is_err() {
            return invalid(format!(
                "dense packed QSA attention-score scratch has {elements} elements, exceeding u32"
            ));
        }
        elements.checked_mul(size_of::<f32>()).ok_or_else(|| {
            Qwen4ExpQsaError::Invalid("dense packed QSA attention-score byte count overflow".into())
        })?;
        Ok(elements)
    }

    fn index_query_width(self) -> usize {
        self.index_query_heads * self.index_head_dim
    }

    fn query_width(self) -> usize {
        self.query_heads * self.head_dim
    }

    fn query_projection_width(self) -> usize {
        self.query_width() * 2
    }

    fn kv_width(self) -> usize {
        self.kv_heads * self.head_dim
    }
}

#[derive(Clone, Copy)]
pub struct QwenSparseAttentionMetalWeights<'a> {
    pub geometry: QwenSparseAttentionMetalGeometry,
    pub query: &'a MetalTensor,
    pub key: &'a MetalTensor,
    pub value: &'a MetalTensor,
    pub output: &'a MetalTensor,
    pub query_norm: &'a MetalTensor,
    pub key_norm: &'a MetalTensor,
    pub index_query: &'a MetalTensor,
    pub index_key: &'a MetalTensor,
    pub index_query_norm: &'a MetalTensor,
    pub index_key_norm: &'a MetalTensor,
}

impl<'a> QwenSparseAttentionMetalWeights<'a> {
    pub fn bind(
        weights: &'a Qwen4ExpMetalWeights,
        layer: u32,
        capacity: usize,
    ) -> Result<Self, Qwen4ExpQsaError> {
        let geometry =
            QwenSparseAttentionMetalGeometry::from_config(weights.config(), layer, capacity)?;
        let prefix = format!("blk.{layer}");
        Ok(Self {
            geometry,
            query: weights.require_tensor(&format!("{prefix}.attn_q.weight"))?,
            key: weights.require_tensor(&format!("{prefix}.attn_k.weight"))?,
            value: weights.require_tensor(&format!("{prefix}.attn_v.weight"))?,
            output: weights.require_tensor(&format!("{prefix}.attn_output.weight"))?,
            query_norm: weights.require_tensor(&format!("{prefix}.attn_q_norm.weight"))?,
            key_norm: weights.require_tensor(&format!("{prefix}.attn_k_norm.weight"))?,
            index_query: weights.require_tensor(&format!("{prefix}.indexer.q_proj.weight"))?,
            index_key: weights.require_tensor(&format!("{prefix}.indexer.k_proj.weight"))?,
            index_query_norm: weights.require_tensor(&format!("{prefix}.indexer.q_norm.weight"))?,
            index_key_norm: weights.require_tensor(&format!("{prefix}.indexer.k_norm.weight"))?,
        })
    }
}

pub struct QwenSparseAttentionMetalWorkspace {
    geometry: QwenSparseAttentionMetalGeometry,
    #[cfg(test)]
    split_decode_scratch: Option<MetalTensor>,
    index_query_raw: MetalTensor,
    index_query: MetalTensor,
    index_key_raw: MetalTensor,
    pending_index_keys: MetalTensor,
    compressed_index_keys: MetalTensor,
    scores: MetalTensor,
    visible_blocks: MetalTensor,
    selected_blocks: MetalTensor,
    selected_count: MetalTensor,
    selector_status: MetalTensor,
    token_ids: MetalTensor,
    attention_logits: MetalTensor,
    query_gate_projection: MetalTensor,
    query: MetalTensor,
    raw_gate: MetalTensor,
    key_raw: MetalTensor,
    key: MetalTensor,
    value: MetalTensor,
    key_cache: MetalTensor,
    value_cache: MetalTensor,
    attention: MetalTensor,
    output: MetalTensor,
    committed_length: usize,
    pending_length: Option<usize>,
    pending_selected_bands: Option<usize>,
    active_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    state_poisoned: bool,
}

pub(crate) struct QwenSparseAttentionPackedScratch {
    geometry: QwenSparseAttentionMetalGeometry,
    capacity: usize,
    query_tile: usize,
    index_key_raw: MetalTensor,
    query_gate_projection: MetalTensor,
    query: MetalTensor,
    key_raw: MetalTensor,
    key: MetalTensor,
    value: MetalTensor,
    attention_scores: MetalTensor,
    attention: MetalTensor,
    output: MetalTensor,
    selected: Option<QwenSparseAttentionSelectedPackedScratch>,
}

struct QwenSparseAttentionSelectedPackedScratch {
    index_query_raw: MetalTensor,
    index_query: MetalTensor,
    scores: MetalTensor,
    visible_blocks: MetalTensor,
    selected_blocks: MetalTensor,
    selected_count: MetalTensor,
    selector_status: MetalTensor,
    token_ids: MetalTensor,
    attention_logits: MetalTensor,
}

#[allow(dead_code)]
struct QwenSparseAttentionSelectedPackedViews {
    index_query_raw: MetalTensor,
    index_query: MetalTensor,
    scores: MetalTensor,
    visible_blocks: MetalTensor,
    selected_blocks: MetalTensor,
    selected_count: MetalTensor,
    selector_status: MetalTensor,
    token_ids: MetalTensor,
    attention_logits: MetalTensor,
}

struct QwenSparseAttentionPackedViews {
    index_key_raw: MetalTensor,
    query_gate_projection: MetalTensor,
    query: MetalTensor,
    key_raw: MetalTensor,
    key: MetalTensor,
    value: MetalTensor,
    attention: MetalTensor,
    output: MetalTensor,
}

impl QwenSparseAttentionPackedScratch {
    pub(crate) fn new(
        ctx: &MetalContext,
        geometry: QwenSparseAttentionMetalGeometry,
        capacity: usize,
    ) -> Result<Self, Qwen4ExpQsaError> {
        Self::new_with_selected_capability(ctx, geometry, capacity, false)
    }

    pub(crate) fn new_with_selected_capability(
        ctx: &MetalContext,
        geometry: QwenSparseAttentionMetalGeometry,
        capacity: usize,
        selected_capable: bool,
    ) -> Result<Self, Qwen4ExpQsaError> {
        geometry.validate(geometry.capacity)?;
        if capacity == 0 || capacity > geometry.token_budget {
            return invalid(format!(
                "dense packed QSA capacity must be in 1..={}, got {capacity}",
                geometry.token_budget
            ));
        }
        let query_tile = capacity.min(DENSE_PACKED_QUERY_TILE);
        for (name, width) in [
            ("index key", geometry.index_head_dim),
            ("query/gate", geometry.query_projection_width()),
            ("query", geometry.query_width()),
            ("raw key", geometry.kv_width()),
            ("key", geometry.kv_width()),
            ("value", geometry.kv_width()),
            ("attention", geometry.query_width()),
            ("output", geometry.hidden_size),
        ] {
            let elements = width.checked_mul(capacity).ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(format!(
                    "dense packed QSA {name} scratch element count overflow"
                ))
            })?;
            if u32::try_from(elements).is_err() {
                return invalid(format!(
                    "dense packed QSA {name} scratch has {elements} elements, exceeding u32"
                ));
            }
            elements.checked_mul(size_of::<f32>()).ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(format!(
                    "dense packed QSA {name} scratch byte count overflow"
                ))
            })?;
        }
        geometry.packed_attention_score_elements(capacity)?;

        let shape = |width: usize| vec![width as u64, capacity as u64];
        let selected = selected_capable
            .then(|| {
                QwenSparseAttentionSelectedPackedScratch::new(ctx, geometry, capacity, query_tile)
            })
            .transpose()?;
        Ok(Self {
            geometry,
            capacity,
            query_tile,
            index_key_raw: MetalTensor::zeros_f32(ctx, shape(geometry.index_head_dim))?,
            query_gate_projection: MetalTensor::zeros_f32(
                ctx,
                shape(geometry.query_projection_width()),
            )?,
            query: MetalTensor::zeros_f32(ctx, shape(geometry.query_width()))?,
            key_raw: MetalTensor::zeros_f32(ctx, shape(geometry.kv_width()))?,
            key: MetalTensor::zeros_f32(ctx, shape(geometry.kv_width()))?,
            value: MetalTensor::zeros_f32(ctx, shape(geometry.kv_width()))?,
            attention_scores: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.output_width() as u64,
                    geometry.query_heads as u64,
                    query_tile as u64,
                ],
            )?,
            attention: MetalTensor::zeros_f32(ctx, shape(geometry.query_width()))?,
            output: MetalTensor::zeros_f32(ctx, shape(geometry.hidden_size))?,
            selected,
        })
    }

    pub(crate) fn selected_capable(&self) -> bool {
        self.selected.is_some()
    }

    fn prefix_view(
        &self,
        name: &str,
        tensor: &MetalTensor,
        width: usize,
        tokens: usize,
    ) -> Result<MetalTensor, Qwen4ExpQsaError> {
        if tokens == 0 || tokens > self.capacity {
            return invalid(format!(
                "{name} token count {tokens} is outside capacity {}",
                self.capacity
            ));
        }
        let elements = width
            .checked_mul(tokens)
            .ok_or_else(|| Qwen4ExpQsaError::Invalid(format!("{name} element count overflow")))?;
        let view = tensor.view_subrange(0, vec![width as u64, tokens as u64]);
        if view.n_elements() as usize != elements {
            return invalid(format!("{name} prefix view has the wrong element count"));
        }
        Ok(view)
    }

    fn views(&self, tokens: usize) -> Result<QwenSparseAttentionPackedViews, Qwen4ExpQsaError> {
        let g = self.geometry;
        Ok(QwenSparseAttentionPackedViews {
            index_key_raw: self.prefix_view(
                "dense packed QSA index key",
                &self.index_key_raw,
                g.index_head_dim,
                tokens,
            )?,
            query_gate_projection: self.prefix_view(
                "dense packed QSA query/gate projection",
                &self.query_gate_projection,
                g.query_projection_width(),
                tokens,
            )?,
            query: self.prefix_view(
                "dense packed QSA query",
                &self.query,
                g.query_width(),
                tokens,
            )?,
            key_raw: self.prefix_view(
                "dense packed QSA raw key",
                &self.key_raw,
                g.kv_width(),
                tokens,
            )?,
            key: self.prefix_view("dense packed QSA key", &self.key, g.kv_width(), tokens)?,
            value: self.prefix_view("dense packed QSA value", &self.value, g.kv_width(), tokens)?,
            attention: self.prefix_view(
                "dense packed QSA attention",
                &self.attention,
                g.query_width(),
                tokens,
            )?,
            output: self.prefix_view(
                "dense packed QSA output",
                &self.output,
                g.hidden_size,
                tokens,
            )?,
        })
    }

    fn score_view(
        &self,
        rows: usize,
        sequence_length: usize,
    ) -> Result<MetalTensor, Qwen4ExpQsaError> {
        if rows == 0 || rows > self.query_tile || sequence_length > self.geometry.output_width() {
            return invalid(format!(
                "dense packed QSA score view rows={rows} length={sequence_length} exceeds tile={} dense limit={}",
                self.query_tile,
                self.geometry.output_width()
            ));
        }
        let elements = rows
            .checked_mul(self.geometry.query_heads)
            .and_then(|elements| elements.checked_mul(sequence_length))
            .ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(
                    "dense packed QSA score-view element count overflow".into(),
                )
            })?;
        Ok(self.attention_scores.view_subrange(
            0,
            vec![
                sequence_length as u64,
                self.geometry.query_heads as u64,
                rows as u64,
            ],
        ))
        .and_then(|view| {
            if view.n_elements() as usize == elements {
                Ok(view)
            } else {
                invalid("dense packed QSA score view has the wrong element count")
            }
        })
    }
}

impl QwenSparseAttentionSelectedPackedScratch {
    fn new(
        ctx: &MetalContext,
        geometry: QwenSparseAttentionMetalGeometry,
        capacity: usize,
        query_tile: usize,
    ) -> Result<Self, Qwen4ExpQsaError> {
        geometry.selected_packed_scratch_logical_allocations(capacity)?;
        let scratch = Self {
            index_query_raw: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.index_head_dim as u64,
                    geometry.index_query_heads as u64,
                    capacity as u64,
                ],
            )?,
            index_query: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.index_head_dim as u64,
                    geometry.index_query_heads as u64,
                    capacity as u64,
                ],
            )?,
            scores: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.block_capacity() as u64, query_tile as u64],
            )?,
            visible_blocks: MetalTensor::zeros_i32(ctx, vec![query_tile as u64])?,
            selected_blocks: MetalTensor::zeros_i32(
                ctx,
                vec![geometry.block_budget() as u64, query_tile as u64],
            )?,
            selected_count: MetalTensor::zeros_i32(ctx, vec![query_tile as u64])?,
            selector_status: MetalTensor::zeros_i32(ctx, vec![query_tile as u64])?,
            token_ids: MetalTensor::zeros_i32(
                ctx,
                vec![geometry.output_width() as u64, query_tile as u64],
            )?,
            attention_logits: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.output_width() as u64,
                    geometry.query_heads as u64,
                    query_tile as u64,
                ],
            )?,
        };
        scratch.validate(geometry, capacity, query_tile)?;
        Ok(scratch)
    }

    fn validate(
        &self,
        geometry: QwenSparseAttentionMetalGeometry,
        capacity: usize,
        query_tile: usize,
    ) -> Result<(), Qwen4ExpQsaError> {
        for (name, tensor, dtype, shape) in [
            (
                "selected packed QSA raw index query",
                &self.index_query_raw,
                GgmlType::F32,
                vec![
                    geometry.index_head_dim as u64,
                    geometry.index_query_heads as u64,
                    capacity as u64,
                ],
            ),
            (
                "selected packed QSA index query",
                &self.index_query,
                GgmlType::F32,
                vec![
                    geometry.index_head_dim as u64,
                    geometry.index_query_heads as u64,
                    capacity as u64,
                ],
            ),
            (
                "selected packed QSA scores",
                &self.scores,
                GgmlType::F32,
                vec![geometry.block_capacity() as u64, query_tile as u64],
            ),
            (
                "selected packed QSA visible blocks",
                &self.visible_blocks,
                GgmlType::I32,
                vec![query_tile as u64],
            ),
            (
                "selected packed QSA selected blocks",
                &self.selected_blocks,
                GgmlType::I32,
                vec![geometry.block_budget() as u64, query_tile as u64],
            ),
            (
                "selected packed QSA selected count",
                &self.selected_count,
                GgmlType::I32,
                vec![query_tile as u64],
            ),
            (
                "selected packed QSA selector status",
                &self.selector_status,
                GgmlType::I32,
                vec![query_tile as u64],
            ),
            (
                "selected packed QSA token IDs",
                &self.token_ids,
                GgmlType::I32,
                vec![geometry.output_width() as u64, query_tile as u64],
            ),
            (
                "selected packed QSA attention logits",
                &self.attention_logits,
                GgmlType::F32,
                vec![
                    geometry.output_width() as u64,
                    geometry.query_heads as u64,
                    query_tile as u64,
                ],
            ),
        ] {
            require_tensor(name, tensor, dtype, &shape, true)?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn views(
        &self,
        geometry: QwenSparseAttentionMetalGeometry,
        capacity: usize,
        query_tile: usize,
        query_count: usize,
    ) -> Result<QwenSparseAttentionSelectedPackedViews, Qwen4ExpQsaError> {
        self.band_views(geometry, capacity, query_tile, 0, query_count)
    }

    fn raw_query_projection_view(
        &self,
        geometry: QwenSparseAttentionMetalGeometry,
        capacity: usize,
        query_tile: usize,
        query_count: usize,
    ) -> Result<MetalTensor, Qwen4ExpQsaError> {
        if query_count == 0 || query_count > capacity {
            return invalid(format!(
                "selected packed QSA raw-query count {query_count} exceeds capacity {capacity}"
            ));
        }
        self.validate(geometry, capacity, query_tile)?;
        let view = self.index_query_raw.view_subrange(
            0,
            vec![geometry.index_query_width() as u64, query_count as u64],
        );
        if view.n_elements() as usize != geometry.index_query_width() * query_count {
            return invalid("selected packed QSA raw-query projection view has the wrong size");
        }
        Ok(view)
    }

    fn band_views(
        &self,
        geometry: QwenSparseAttentionMetalGeometry,
        capacity: usize,
        query_tile: usize,
        raw_query_offset: usize,
        query_count: usize,
    ) -> Result<QwenSparseAttentionSelectedPackedViews, Qwen4ExpQsaError> {
        if query_count == 0 || query_count > capacity.min(query_tile) {
            return invalid(format!(
                "selected packed QSA query count {query_count} exceeds capacity {capacity} or tile {query_tile}"
            ));
        }
        let raw_query_end = raw_query_offset.checked_add(query_count).ok_or_else(|| {
            Qwen4ExpQsaError::Invalid("selected packed QSA raw-query band end overflow".into())
        })?;
        if raw_query_end > capacity {
            return invalid(format!(
                "selected packed QSA raw-query band {raw_query_offset}..{raw_query_end} exceeds capacity {capacity}"
            ));
        }
        let raw_element_offset = raw_query_offset
            .checked_mul(geometry.index_query_width())
            .ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(
                    "selected packed QSA raw-query band offset overflow".into(),
                )
            })?;
        self.validate(geometry, capacity, query_tile)?;
        let view =
            |name: &str, tensor: &MetalTensor, offset: usize, elements: usize, shape: Vec<u64>| {
                let view = tensor.view_subrange(offset as u64, shape);
                if view.n_elements() as usize != elements {
                    return invalid(format!(
                        "selected packed QSA {name} view has the wrong element count"
                    ));
                }
                Ok(view)
            };
        Ok(QwenSparseAttentionSelectedPackedViews {
            index_query_raw: view(
                "raw index query",
                &self.index_query_raw,
                raw_element_offset,
                geometry.index_query_width() * query_count,
                vec![
                    geometry.index_head_dim as u64,
                    geometry.index_query_heads as u64,
                    query_count as u64,
                ],
            )?,
            index_query: view(
                "index query",
                &self.index_query,
                0,
                geometry.index_query_width() * query_count,
                vec![
                    geometry.index_head_dim as u64,
                    geometry.index_query_heads as u64,
                    query_count as u64,
                ],
            )?,
            scores: view(
                "scores",
                &self.scores,
                0,
                geometry.block_capacity() * query_count,
                vec![geometry.block_capacity() as u64, query_count as u64],
            )?,
            visible_blocks: view(
                "visible blocks",
                &self.visible_blocks,
                0,
                query_count,
                vec![query_count as u64],
            )?,
            selected_blocks: view(
                "selected blocks",
                &self.selected_blocks,
                0,
                geometry.block_budget() * query_count,
                vec![geometry.block_budget() as u64, query_count as u64],
            )?,
            selected_count: view(
                "selected count",
                &self.selected_count,
                0,
                query_count,
                vec![query_count as u64],
            )?,
            selector_status: view(
                "selector status",
                &self.selector_status,
                0,
                query_count,
                vec![query_count as u64],
            )?,
            token_ids: view(
                "token IDs",
                &self.token_ids,
                0,
                geometry.output_width() * query_count,
                vec![geometry.output_width() as u64, query_count as u64],
            )?,
            attention_logits: view(
                "attention logits",
                &self.attention_logits,
                0,
                geometry.output_width() * geometry.query_heads * query_count,
                vec![
                    geometry.output_width() as u64,
                    geometry.query_heads as u64,
                    query_count as u64,
                ],
            )?,
        })
    }
}

impl QwenSparseAttentionMetalWorkspace {
    pub fn new(
        ctx: &MetalContext,
        geometry: QwenSparseAttentionMetalGeometry,
    ) -> Result<Self, Qwen4ExpQsaError> {
        geometry.validate(geometry.capacity)?;
        Ok(Self {
            geometry,
            index_query_raw: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.index_head_dim as u64,
                    geometry.index_query_heads as u64,
                    1,
                ],
            )?,
            index_query: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.index_head_dim as u64,
                    geometry.index_query_heads as u64,
                    1,
                ],
            )?,
            index_key_raw: MetalTensor::zeros_f32(ctx, vec![geometry.index_head_dim as u64])?,
            pending_index_keys: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.index_head_dim as u64, geometry.ratio as u64],
            )?,
            compressed_index_keys: MetalTensor::zeros_f16(
                ctx,
                vec![
                    geometry.index_head_dim as u64,
                    geometry.block_capacity() as u64,
                ],
            )?,
            scores: MetalTensor::zeros_f32(ctx, vec![geometry.block_capacity() as u64, 1])?,
            visible_blocks: MetalTensor::zeros_i32(ctx, vec![1])?,
            selected_blocks: MetalTensor::zeros_i32(ctx, vec![geometry.block_budget() as u64, 1])?,
            selected_count: MetalTensor::zeros_i32(ctx, vec![1])?,
            selector_status: MetalTensor::zeros_i32(ctx, vec![1])?,
            token_ids: MetalTensor::zeros_i32(ctx, vec![geometry.output_width() as u64])?,
            attention_logits: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.output_width() as u64, geometry.query_heads as u64],
            )?,
            query_gate_projection: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.query_projection_width() as u64],
            )?,
            query: MetalTensor::zeros_f32(ctx, vec![geometry.query_width() as u64])?,
            raw_gate: MetalTensor::zeros_f32(ctx, vec![geometry.query_width() as u64])?,
            key_raw: MetalTensor::zeros_f32(ctx, vec![geometry.kv_width() as u64])?,
            key: MetalTensor::zeros_f32(ctx, vec![geometry.kv_width() as u64])?,
            value: MetalTensor::zeros_f32(ctx, vec![geometry.kv_width() as u64])?,
            key_cache: MetalTensor::zeros_f16(
                ctx,
                vec![
                    geometry.head_dim as u64,
                    geometry.kv_heads as u64,
                    geometry.capacity as u64,
                ],
            )?,
            value_cache: MetalTensor::zeros_f16(
                ctx,
                vec![
                    geometry.head_dim as u64,
                    geometry.kv_heads as u64,
                    geometry.capacity as u64,
                ],
            )?,
            attention: MetalTensor::zeros_f32(ctx, vec![geometry.query_width() as u64])?,
            output: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            #[cfg(test)]
            split_decode_scratch: None,
            committed_length: 0,
            pending_length: None,
            pending_selected_bands: None,
            active_command: None,
            state_poisoned: false,
        })
    }

    pub fn geometry(&self) -> QwenSparseAttentionMetalGeometry {
        self.geometry
    }

    #[cfg(test)]
    pub(crate) fn validate_split_binding(
        &self,
        ctx: &MetalContext,
        scratch: Option<&MetalTensor>,
    ) -> Result<(), Qwen4ExpQsaError> {
        self.require_idle()?;
        if self.state_poisoned
            || self.pending_length.is_some()
            || self.pending_selected_bands.is_some()
        {
            return invalid("split binding requires a healthy released QSA workspace");
        }
        if let Some(scratch) = scratch {
            if !self.geometry.supports_split_decode() {
                return invalid("split QSA requires released 24/2/256 geometry");
            }
            require_tensor(
                "QSA split scratch",
                scratch,
                GgmlType::F32,
                &[split_decode::SCRATCH_FLOATS as u64],
                true,
            )?;
            require_same_device(ctx, &[("QSA split scratch", scratch)])?;
            let mut tensors = workspace_tensors(self);
            tensors.retain(|(name, _)| *name != "QSA split scratch");
            tensors.push(("QSA split scratch", scratch));
            require_disjoint(&tensors)?;
            split_decode::preflight(ctx)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn bind_split_scratch(&mut self, scratch: Option<&MetalTensor>) {
        self.split_decode_scratch = scratch.cloned();
    }

    #[cfg(test)]
    pub(crate) fn split_scratch_identity(&self) -> Option<usize> {
        self.split_decode_scratch
            .as_ref()
            .map(|s| s.buffer.contents().as_ptr() as usize)
    }

    pub fn committed_length(&self) -> usize {
        self.committed_length
    }

    #[cfg(test)]
    pub(crate) fn restore_length_for_tests(&mut self, length: usize) {
        self.require_idle().unwrap();
        assert!(!self.state_poisoned);
        assert!(self.pending_length.is_none() && self.pending_selected_bands.is_none());
        assert!(length <= self.committed_length);
        self.committed_length = length;
    }

    #[cfg(test)]
    pub(crate) fn persistent_state_tensors(&self) -> Vec<MetalTensor> {
        vec![
            self.pending_index_keys.clone(),
            self.compressed_index_keys.clone(),
            self.key_cache.clone(),
            self.value_cache.clone(),
        ]
    }

    pub fn is_poisoned(&self) -> bool {
        self.state_poisoned
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpQsaError> {
        self.require_idle()?;
        self.committed_length = 0;
        self.pending_length = None;
        self.pending_selected_bands = None;
        self.state_poisoned = false;
        Ok(())
    }

    pub fn release_after(&mut self) -> Result<(), Qwen4ExpQsaError> {
        let Some(command) = self.active_command.clone() else {
            if self.pending_length.is_some() || self.pending_selected_bands.is_some() {
                self.pending_length = None;
                self.pending_selected_bands = None;
                self.state_poisoned = true;
                return invalid("QSA pending metadata has no owning command buffer");
            }
            return Ok(());
        };
        let status = command.status();
        if matches!(
            status,
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued
        ) {
            return invalid(format!(
                "workspace owner is not committed (status {status:?}); commit it or abandon the uncommitted command"
            ));
        }
        command.waitUntilCompleted();
        let status = command.status();
        let error = command.error().map(|error| error.to_string());
        let scalar_result = (|| {
            Ok((
                read_i32_scalar(&self.selector_status)?,
                read_i32_scalar(&self.selected_count)?,
                read_i32_scalar(&self.visible_blocks)?,
            ))
        })();
        self.active_command = None;
        let pending = self.pending_length.take();
        let pending_selected_bands = self.pending_selected_bands.take();
        let (selector_status, selected_count, audited_bands) = match scalar_result {
            Ok(scalars) => scalars,
            Err(error) => {
                self.state_poisoned = true;
                return Err(error);
            }
        };
        let (pending, pending_selected_bands) = match (pending, pending_selected_bands) {
            (Some(pending), Some(pending_selected_bands)) => (pending, pending_selected_bands),
            _ => {
                self.state_poisoned = true;
                return invalid("completed QSA command had incomplete pending metadata");
            }
        };
        let expected_count = (pending / self.geometry.ratio).min(self.geometry.block_budget());
        let expected_audited_bands = i32::try_from(pending_selected_bands).map_err(|_| {
            self.state_poisoned = true;
            Qwen4ExpQsaError::Invalid(format!(
                "pending QSA selected-band count {pending_selected_bands} exceeds i32"
            ))
        })?;
        let audit_complete = pending_selected_bands == 0 || audited_bands == expected_audited_bands;
        if status == MTLCommandBufferStatus::Completed
            && error.is_none()
            && !self.state_poisoned
            && selector_status == 0
            && selected_count == expected_count as i32
            && audit_complete
        {
            self.committed_length = pending;
            Ok(())
        } else {
            self.state_poisoned = true;
            Err(Qwen4ExpQsaError::CommandBuffer(format!(
                "status={status:?}, error={error:?}, selector_status={selector_status}, selected_count={selected_count}, expected_count={expected_count}, audited_bands={audited_bands}, expected_audited_bands={expected_audited_bands}"
            )))
        }
    }

    /// Release a workspace from a command buffer that will never be committed.
    ///
    /// # Safety
    ///
    /// The caller must end and permanently discard every reference to the
    /// owning command buffer. Committing it later can mutate causal state out
    /// of order or race a subsequent owner.
    pub unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpQsaError> {
        let Some(command) = self.active_command.as_ref() else {
            if self.pending_length.is_some() || self.pending_selected_bands.is_some() {
                self.pending_length = None;
                self.pending_selected_bands = None;
                self.state_poisoned = true;
                return invalid("QSA pending metadata has no command buffer to abandon");
            }
            return Ok(());
        };
        let status = command.status();
        if status != MTLCommandBufferStatus::NotEnqueued {
            return invalid(format!(
                "only a NotEnqueued workspace owner can be abandoned, got {status:?}"
            ));
        }
        self.active_command = None;
        self.pending_length = None;
        self.pending_selected_bands = None;
        self.state_poisoned = false;
        Ok(())
    }

    fn require_idle(&self) -> Result<(), Qwen4ExpQsaError> {
        if self.active_command.is_some()
            || self.pending_length.is_some()
            || self.pending_selected_bands.is_some()
        {
            invalid("workspace causal state is still owned by a command buffer")
        } else {
            Ok(())
        }
    }
}

#[must_use = "consume or copy the QSA output in its owning command, then release the workspace"]
pub struct QwenSparseAttentionMetalRead<'a> {
    workspace: &'a mut QwenSparseAttentionMetalWorkspace,
    position: usize,
}

pub struct QwenSparseAttentionMetalOutput<'a> {
    workspace: &'a QwenSparseAttentionMetalWorkspace,
    position: usize,
}

impl QwenSparseAttentionMetalRead<'_> {
    pub fn output(&self) -> QwenSparseAttentionMetalOutput<'_> {
        QwenSparseAttentionMetalOutput {
            workspace: self.workspace,
            position: self.position,
        }
    }
}

impl QwenSparseAttentionMetalOutput<'_> {
    pub fn n_elements(&self) -> u64 {
        self.workspace.geometry.hidden_size as u64
    }

    pub fn dtype(&self) -> GgmlType {
        GgmlType::F32
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn encode_copy_to(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        destination: &MetalTensor,
    ) -> Result<(), Qwen4ExpQsaError> {
        validate_encoder(ctx, enc)?;
        let command = enc.parent_command_buffer();
        let Some(owner) = self.workspace.active_command.as_ref() else {
            return invalid("QSA output has no owning command buffer");
        };
        if !std::ptr::addr_eq(Retained::as_ptr(owner), Retained::as_ptr(&command)) {
            return invalid("QSA output must be copied by its owning command buffer");
        }
        require_tensor(
            "QSA copied output destination",
            destination,
            GgmlType::F32,
            &[self.workspace.geometry.hidden_size as u64],
            true,
        )?;
        require_same_device(
            ctx,
            &[
                ("QSA output", &self.workspace.output),
                ("QSA copied output destination", destination),
            ],
        )?;
        let mut tensors = workspace_tensors(self.workspace);
        tensors.push(("QSA copied output destination", destination));
        require_disjoint(&tensors)?;
        ctx.pipeline("kernel_copy_offset_f32")?;
        encode_copy_offset_f32(
            ctx,
            enc,
            &self.workspace.output,
            0,
            destination,
            self.workspace.geometry.hidden_size,
        )?;
        Ok(())
    }
}

/// Encode one text-only QSA token. Multimodal IMRoPE positions are unsupported.
pub fn encode_qwen_sparse_attention_text<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &'a mut QwenSparseAttentionMetalWorkspace,
) -> Result<QwenSparseAttentionMetalRead<'a>, Qwen4ExpQsaError> {
    validate_encoder(ctx, enc)?;
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if weights.geometry != workspace.geometry {
        return invalid("QSA weight and workspace geometry differ");
    }
    if workspace.committed_length >= workspace.geometry.capacity {
        return invalid(format!(
            "QSA token capacity {} is exhausted",
            workspace.geometry.capacity
        ));
    }
    validate_contract(ctx, input, weights, workspace)?;
    preflight(ctx, weights)?;

    let (position, sequence_length) = prepare_control_scalars(workspace)?;
    reserve_command(workspace, enc, sequence_length, 0)?;

    let encoded = encode_step(
        ctx,
        enc,
        input,
        weights,
        workspace,
        position,
        sequence_length,
    );
    if let Err(error) = encoded {
        workspace.state_poisoned = true;
        return Err(error);
    }
    Ok(QwenSparseAttentionMetalRead {
        workspace,
        position,
    })
}

/// Encode a dense, consecutive text chunk into the scalar QSA cache owner.
/// Multi-token chunks reject any query that requires block selection; `N=1`
/// delegates to the scalar path at every position.
///
/// # Safety
///
/// The caller must retain every tensor and exclusive logical ownership of
/// `workspace` and `scratch` until the command completes successfully or is
/// permanently abandoned. Commands touching this workspace must execute in
/// causal order. Any encode or command failure makes mutable state
/// indeterminate and requires poisoning the enclosing transaction.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) unsafe fn encode_qwen_sparse_attention_text_dense_packed_motor(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &mut QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    tokens: usize,
) -> Result<MetalTensor, Qwen4ExpQsaError> {
    unsafe {
        encode_qwen_sparse_attention_text_dense_packed_motor_inner(
            ctx,
            enc,
            input,
            weights,
            workspace,
            scratch,
            start_position,
            tokens,
            0,
            None,
            false,
        )
    }
}

/// Encode a consecutive text chunk, using block selection for queries beyond
/// the dense QSA range, into the scalar QSA cache owner.
///
/// # Safety
///
/// The caller must retain every tensor and exclusive logical ownership of
/// `workspace` and `scratch` until the command completes successfully or is
/// permanently abandoned. Commands touching this workspace must execute in
/// causal order. Any encode or command failure makes mutable state
/// indeterminate and requires poisoning the enclosing transaction.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn encode_qwen_sparse_attention_text_packed_motor(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &mut QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    tokens: usize,
) -> Result<MetalTensor, Qwen4ExpQsaError> {
    unsafe {
        encode_qwen_sparse_attention_text_dense_packed_motor_inner(
            ctx,
            enc,
            input,
            weights,
            workspace,
            scratch,
            start_position,
            tokens,
            0,
            None,
            true,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) unsafe fn encode_qwen_sparse_attention_text_dense_packed_motor_profiled(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &mut QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    tokens: usize,
    layer: u32,
    recorder: &mut Qwen4ExpPackedProfileRecorder<'_>,
) -> Result<MetalTensor, Qwen4ExpQsaError> {
    unsafe {
        encode_qwen_sparse_attention_text_dense_packed_motor_inner(
            ctx,
            enc,
            input,
            weights,
            workspace,
            scratch,
            start_position,
            tokens,
            layer,
            Some(recorder),
            false,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn encode_qwen_sparse_attention_text_packed_motor_profiled(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &mut QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    tokens: usize,
    layer: u32,
    recorder: &mut Qwen4ExpPackedProfileRecorder<'_>,
) -> Result<MetalTensor, Qwen4ExpQsaError> {
    unsafe {
        encode_qwen_sparse_attention_text_dense_packed_motor_inner(
            ctx,
            enc,
            input,
            weights,
            workspace,
            scratch,
            start_position,
            tokens,
            layer,
            Some(recorder),
            true,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn encode_qwen_sparse_attention_text_dense_packed_motor_inner(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &mut QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    tokens: usize,
    layer: u32,
    profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
    selected_enabled: bool,
) -> Result<MetalTensor, Qwen4ExpQsaError> {
    if tokens == 0 {
        return invalid("dense packed QSA token count must be nonzero");
    }
    if start_position != workspace.committed_length {
        return invalid(format!(
            "dense packed QSA start {start_position} differs from committed length {}",
            workspace.committed_length
        ));
    }
    require_tensor(
        "dense packed QSA input",
        input,
        GgmlType::F32,
        &[workspace.geometry.hidden_size as u64, tokens as u64],
        false,
    )?;

    if tokens == 1 {
        let scalar_input = input.view_subrange(0, vec![workspace.geometry.hidden_size as u64]);
        let read = encode_qwen_sparse_attention_text(ctx, enc, &scalar_input, weights, workspace)?;
        let output = read
            .workspace
            .output
            .view_subrange(0, vec![read.workspace.geometry.hidden_size as u64, 1]);
        drop(read);
        return Ok(output);
    }

    let plan = validate_packed_contract(
        ctx,
        enc,
        input,
        weights,
        workspace,
        scratch,
        start_position,
        tokens,
        selected_enabled,
    )?;
    preflight_packed(ctx, weights, plan)?;
    prepare_packed_control_scalars(workspace, scratch, start_position, tokens, plan)?;
    reserve_command(workspace, enc, plan.end_position, plan.selected_bands)?;

    let encoded = (|| {
        encode_packed_step(
            ctx,
            enc,
            input,
            weights,
            workspace,
            scratch,
            start_position,
            tokens,
            plan,
            layer,
            profile,
        )?;
        scratch.views(tokens).map(|views| views.output)
    })();
    match encoded {
        Ok(output) => Ok(output),
        Err(error) => {
            workspace.state_poisoned = true;
            Err(error)
        }
    }
}

pub(crate) fn prepare_control_scalars(
    workspace: &QwenSparseAttentionMetalWorkspace,
) -> Result<(usize, usize), Qwen4ExpQsaError> {
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if workspace.committed_length >= workspace.geometry.capacity {
        return invalid(format!(
            "QSA token capacity {} is exhausted",
            workspace.geometry.capacity
        ));
    }
    let position = workspace.committed_length;
    let sequence_length = position + 1;
    let visible_blocks = sequence_length / workspace.geometry.ratio;
    write_i32_scalar(&workspace.visible_blocks, visible_blocks as i32)?;
    write_i32_scalar(&workspace.selector_status, 0)?;
    write_i32_scalar(&workspace.selected_count, 0)?;
    Ok((position, sequence_length))
}

#[allow(dead_code)]
fn prepare_packed_control_scalars(
    workspace: &QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    tokens: usize,
    plan: QwenSparseAttentionPackedRangePlan,
) -> Result<(), Qwen4ExpQsaError> {
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if start_position != workspace.committed_length {
        return invalid(format!(
            "dense packed QSA start {start_position} differs from committed length {}",
            workspace.committed_length
        ));
    }
    if tokens <= 1 {
        return invalid(format!(
            "dense packed QSA requires at least two tokens, got {tokens}"
        ));
    }
    let expected_plan = workspace
        .geometry
        .plan_packed_range(start_position, tokens)?;
    if plan != expected_plan {
        return invalid("packed QSA control plan differs from validated range");
    }
    if plan.selected_tokens > 0 {
        let selected = scratch.selected.as_ref().ok_or_else(|| {
            Qwen4ExpQsaError::Invalid("packed QSA selected scratch was not admitted".into())
        })?;
        let first_band_rows = plan.selected_tokens.min(scratch.query_tile);
        let views = selected.band_views(
            workspace.geometry,
            scratch.capacity,
            scratch.query_tile,
            0,
            first_band_rows,
        )?;
        fill_i32_tensor(&views.visible_blocks, -1)?;
        fill_i32_tensor(&views.selected_count, -1)?;
        fill_i32_tensor(&views.selector_status, -1)?;
        write_i32_scalar(&workspace.visible_blocks, 0)?;
        write_i32_scalar(&workspace.selector_status, 0)?;
        write_i32_scalar(&workspace.selected_count, -1)?;
    } else {
        let visible_blocks = plan.end_position / workspace.geometry.ratio;
        write_i32_scalar(&workspace.visible_blocks, visible_blocks as i32)?;
        write_i32_scalar(&workspace.selector_status, 0)?;
        write_i32_scalar(&workspace.selected_count, 0)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn validate_dense_packed_contract(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    tokens: usize,
) -> Result<(), Qwen4ExpQsaError> {
    validate_packed_contract(
        ctx,
        enc,
        input,
        weights,
        workspace,
        scratch,
        start_position,
        tokens,
        false,
    )
    .map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_and_preflight_packed_contract(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    tokens: usize,
) -> Result<(), Qwen4ExpQsaError> {
    let plan = validate_packed_contract(
        ctx,
        enc,
        input,
        weights,
        workspace,
        scratch,
        start_position,
        tokens,
        true,
    )?;
    preflight_packed(ctx, weights, plan)
}

#[allow(clippy::too_many_arguments)]
fn validate_packed_contract(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    tokens: usize,
    selected_enabled: bool,
) -> Result<QwenSparseAttentionPackedRangePlan, Qwen4ExpQsaError> {
    validate_encoder(ctx, enc)?;
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    let g = workspace.geometry;
    if weights.geometry != g || scratch.geometry != g {
        return invalid("dense packed QSA weight, workspace, and scratch geometry differ");
    }
    if tokens <= 1 || tokens > scratch.capacity {
        return invalid(format!(
            "dense packed QSA token count {tokens} is outside packed capacity {}",
            scratch.capacity
        ));
    }
    if start_position != workspace.committed_length {
        return invalid(format!(
            "dense packed QSA start {start_position} differs from committed length {}",
            workspace.committed_length
        ));
    }
    let plan = g.plan_packed_range(start_position, tokens)?;
    if plan.selected_tokens != 0 && !selected_enabled {
        return invalid(format!(
            "dense packed QSA end {} exceeds dense limit {}",
            plan.end_position,
            g.output_width()
        ));
    }
    if plan.selected_tokens > 0 {
        if !scratch.selected_capable() {
            return invalid("packed QSA selected suffix requires selected-capable scratch");
        }
        if scratch.query_tile == 0 || scratch.query_tile > DENSE_PACKED_QUERY_TILE {
            return invalid(format!(
                "packed QSA selected query tile {} is outside 1..={DENSE_PACKED_QUERY_TILE}",
                scratch.query_tile
            ));
        }
    }
    let sequence_length = plan.end_position;
    let dense_end = start_position
        .checked_add(plan.dense_tokens)
        .ok_or_else(|| Qwen4ExpQsaError::Invalid("packed QSA dense end overflow".into()))?;
    if g.theta <= 1.0 {
        return invalid(format!(
            "dense packed QSA requires RoPE theta greater than one, got {}",
            g.theta
        ));
    }
    for (name, value) in [
        ("start position", start_position),
        ("token count", tokens),
        ("sequence length", sequence_length),
        ("dense end", dense_end),
        ("query tile", scratch.query_tile),
    ] {
        if u32::try_from(value).is_err() {
            return invalid(format!("dense packed QSA {name} {value} exceeds u32"));
        }
    }
    if plan.selected_tokens > 0 {
        for (name, value) in [
            ("block budget", g.block_budget()),
            ("maximum visible blocks", sequence_length / g.ratio),
            ("selected token count", plan.selected_tokens),
            ("selected band count", plan.selected_bands),
        ] {
            if i32::try_from(value).is_err() {
                return invalid(format!("packed QSA {name} {value} exceeds i32"));
            }
        }
    }
    let first_input = input.view_subrange(0, vec![g.hidden_size as u64]);
    validate_contract(ctx, &first_input, weights, workspace)?;
    require_tensor(
        "dense packed QSA input",
        input,
        GgmlType::F32,
        &[g.hidden_size as u64, tokens as u64],
        false,
    )?;
    validate_dense_packed_scratch(scratch)?;
    let views = scratch.views(tokens)?;
    for (dtype, n_in, n_out) in [
        (weights.index_key.dtype, g.hidden_size, g.index_head_dim),
        (
            weights.query.dtype,
            g.hidden_size,
            g.query_projection_width(),
        ),
        (weights.key.dtype, g.hidden_size, g.kv_width()),
        (weights.value.dtype, g.hidden_size, g.kv_width()),
        (weights.output.dtype, g.query_width(), g.hidden_size),
    ] {
        validate_f32_q8_mat_mat_addressing(dtype, n_in, n_out, tokens)?;
    }
    if plan.selected_tokens > 0 {
        validate_f32_q8_mat_mat_addressing(
            weights.index_query.dtype,
            g.hidden_size,
            g.index_query_width(),
            plan.selected_tokens,
        )?;
    }
    let dense_score_length = if plan.dense_tokens > 0 { dense_end } else { 0 };
    for (name, elements) in [
        ("input", g.hidden_size.checked_mul(tokens)),
        ("index key", g.index_head_dim.checked_mul(tokens)),
        ("query/gate", g.query_projection_width().checked_mul(tokens)),
        ("query", g.query_width().checked_mul(tokens)),
        ("key/value", g.kv_width().checked_mul(tokens)),
        ("output", g.hidden_size.checked_mul(tokens)),
        (
            "largest score tile",
            scratch
                .query_tile
                .checked_mul(g.query_heads)
                .and_then(|elements| elements.checked_mul(dense_score_length)),
        ),
        (
            "cache append offset",
            start_position.checked_mul(g.kv_width()),
        ),
        (
            "cache append end",
            sequence_length.checked_mul(g.kv_width()),
        ),
    ] {
        let elements = elements.ok_or_else(|| {
            Qwen4ExpQsaError::Invalid(format!("dense packed QSA {name} arithmetic overflow"))
        })?;
        if u32::try_from(elements).is_err() {
            return invalid(format!(
                "dense packed QSA {name} element span {elements} exceeds u32"
            ));
        }
    }
    if g.kv_heads == 0 || !g.query_heads.is_multiple_of(g.kv_heads) {
        return invalid("dense packed QSA requires a nonzero integral GQA group");
    }
    if plan.selected_tokens > 0
        && (g.head_dim != MAIN_HEAD_DIM
            || !(g.query_heads / g.kv_heads).is_multiple_of(PACKED_ATTENTION_HEADS_PER_TG))
    {
        return invalid(format!(
            "selected packed QSA requires head dim {MAIN_HEAD_DIM} and GQA groups divisible by {PACKED_ATTENTION_HEADS_PER_TG}"
        ));
    }

    let weights_named = [
        ("dense packed QSA query projection", weights.query),
        ("dense packed QSA key projection", weights.key),
        ("dense packed QSA value projection", weights.value),
        ("dense packed QSA output projection", weights.output),
        ("dense packed QSA query norm", weights.query_norm),
        ("dense packed QSA key norm", weights.key_norm),
        (
            "dense packed QSA index query projection",
            weights.index_query,
        ),
        ("dense packed QSA index key projection", weights.index_key),
        (
            "dense packed QSA index query norm",
            weights.index_query_norm,
        ),
        ("dense packed QSA index key norm", weights.index_key_norm),
    ];
    let mut tensors = weights_named.to_vec();
    tensors.push(("dense packed QSA input", input));
    tensors.extend(workspace_tensors(workspace));
    tensors.extend(dense_packed_scratch_tensors(scratch));
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)?;

    for (name, tensor, elements) in [
        (
            "dense packed QSA index-key view",
            &views.index_key_raw,
            g.index_head_dim * tokens,
        ),
        (
            "dense packed QSA query/gate view",
            &views.query_gate_projection,
            g.query_projection_width() * tokens,
        ),
        (
            "dense packed QSA query view",
            &views.query,
            g.query_width() * tokens,
        ),
        (
            "dense packed QSA raw-key view",
            &views.key_raw,
            g.kv_width() * tokens,
        ),
        (
            "dense packed QSA key view",
            &views.key,
            g.kv_width() * tokens,
        ),
        (
            "dense packed QSA value view",
            &views.value,
            g.kv_width() * tokens,
        ),
        (
            "dense packed QSA attention view",
            &views.attention,
            g.query_width() * tokens,
        ),
        (
            "dense packed QSA output view",
            &views.output,
            g.hidden_size * tokens,
        ),
    ] {
        if tensor.n_elements() as usize != elements {
            return invalid(format!("{name} has the wrong element count"));
        }
    }
    if plan.dense_tokens > 0 {
        scratch.score_view(scratch.query_tile.min(plan.dense_tokens), dense_end)?;
    }
    if plan.selected_tokens > 0 {
        let selected = scratch.selected.as_ref().ok_or_else(|| {
            Qwen4ExpQsaError::Invalid("packed QSA selected scratch disappeared".into())
        })?;
        let raw_query_projection = selected.raw_query_projection_view(
            g,
            scratch.capacity,
            scratch.query_tile,
            plan.selected_tokens,
        )?;
        let input_offset = plan
            .selected_offset
            .checked_mul(g.hidden_size)
            .ok_or_else(|| {
                Qwen4ExpQsaError::Invalid("packed QSA selected input offset overflow".into())
            })?;
        let query_offset = plan
            .selected_offset
            .checked_mul(g.query_width())
            .ok_or_else(|| {
                Qwen4ExpQsaError::Invalid("packed QSA selected query offset overflow".into())
            })?;
        let projected_offset = plan
            .selected_offset
            .checked_mul(g.query_projection_width())
            .ok_or_else(|| {
                Qwen4ExpQsaError::Invalid("packed QSA selected query/gate offset overflow".into())
            })?;
        let selected_input = input.view_subrange(
            input_offset as u64,
            vec![g.hidden_size as u64, plan.selected_tokens as u64],
        );
        let selected_query = views.query.view_subrange(
            query_offset as u64,
            vec![g.query_width() as u64, plan.selected_tokens as u64],
        );
        let selected_query_gate = views.query_gate_projection.view_subrange(
            projected_offset as u64,
            vec![
                g.query_projection_width() as u64,
                plan.selected_tokens as u64,
            ],
        );
        let selected_attention = views.attention.view_subrange(
            query_offset as u64,
            vec![g.query_width() as u64, plan.selected_tokens as u64],
        );
        for (name, tensor, shape, writable) in [
            (
                "packed QSA selected input view",
                &selected_input,
                vec![g.hidden_size as u64, plan.selected_tokens as u64],
                false,
            ),
            (
                "packed QSA selected query view",
                &selected_query,
                vec![g.query_width() as u64, plan.selected_tokens as u64],
                false,
            ),
            (
                "packed QSA selected query/gate view",
                &selected_query_gate,
                vec![
                    g.query_projection_width() as u64,
                    plan.selected_tokens as u64,
                ],
                false,
            ),
            (
                "packed QSA selected attention view",
                &selected_attention,
                vec![g.query_width() as u64, plan.selected_tokens as u64],
                true,
            ),
        ] {
            require_tensor(name, tensor, GgmlType::F32, &shape, writable)?;
        }
        require_tensor(
            "packed QSA selected raw index-query projection view",
            &raw_query_projection,
            GgmlType::F32,
            &[g.index_query_width() as u64, plan.selected_tokens as u64],
            true,
        )?;
        let mut band_offset = 0;
        while band_offset < plan.selected_tokens {
            let band_rows = (plan.selected_tokens - band_offset).min(scratch.query_tile);
            selected.band_views(
                g,
                scratch.capacity,
                scratch.query_tile,
                band_offset,
                band_rows,
            )?;
            let local_offset = plan
                .selected_offset
                .checked_add(band_offset)
                .ok_or_else(|| {
                    Qwen4ExpQsaError::Invalid(
                        "packed QSA selected band local offset overflow".into(),
                    )
                })?;
            for (name, width, tensor) in [
                ("input", g.hidden_size, input),
                ("query", g.query_width(), &views.query),
                (
                    "query/gate",
                    g.query_projection_width(),
                    &views.query_gate_projection,
                ),
                ("attention", g.query_width(), &views.attention),
            ] {
                let offset = local_offset.checked_mul(width).ok_or_else(|| {
                    Qwen4ExpQsaError::Invalid(format!(
                        "packed QSA selected {name} band offset overflow"
                    ))
                })?;
                let band =
                    tensor.view_subrange(offset as u64, vec![width as u64, band_rows as u64]);
                if band.n_elements() as usize != width * band_rows {
                    return invalid(format!(
                        "packed QSA selected {name} band view has the wrong size"
                    ));
                }
            }
            band_offset += band_rows;
        }
    }
    Ok(plan)
}

fn validate_dense_packed_scratch(
    scratch: &QwenSparseAttentionPackedScratch,
) -> Result<(), Qwen4ExpQsaError> {
    let g = scratch.geometry;
    for (name, tensor, shape) in [
        (
            "dense packed QSA index key scratch",
            &scratch.index_key_raw,
            vec![g.index_head_dim as u64, scratch.capacity as u64],
        ),
        (
            "dense packed QSA query/gate scratch",
            &scratch.query_gate_projection,
            vec![g.query_projection_width() as u64, scratch.capacity as u64],
        ),
        (
            "dense packed QSA query scratch",
            &scratch.query,
            vec![g.query_width() as u64, scratch.capacity as u64],
        ),
        (
            "dense packed QSA raw-key scratch",
            &scratch.key_raw,
            vec![g.kv_width() as u64, scratch.capacity as u64],
        ),
        (
            "dense packed QSA key scratch",
            &scratch.key,
            vec![g.kv_width() as u64, scratch.capacity as u64],
        ),
        (
            "dense packed QSA value scratch",
            &scratch.value,
            vec![g.kv_width() as u64, scratch.capacity as u64],
        ),
        (
            "dense packed QSA attention scratch",
            &scratch.attention,
            vec![g.query_width() as u64, scratch.capacity as u64],
        ),
        (
            "dense packed QSA output scratch",
            &scratch.output,
            vec![g.hidden_size as u64, scratch.capacity as u64],
        ),
    ] {
        require_tensor(name, tensor, GgmlType::F32, &shape, true)?;
    }
    require_tensor(
        "dense packed QSA attention-score scratch",
        &scratch.attention_scores,
        GgmlType::F32,
        &[
            g.output_width() as u64,
            g.query_heads as u64,
            scratch.query_tile as u64,
        ],
        true,
    )?;
    if let Some(selected) = scratch.selected.as_ref() {
        selected.validate(g, scratch.capacity, scratch.query_tile)?;
    }
    Ok(())
}

fn dense_packed_scratch_tensors(
    scratch: &QwenSparseAttentionPackedScratch,
) -> Vec<(&'static str, &MetalTensor)> {
    let mut tensors = vec![
        ("dense packed QSA index key scratch", &scratch.index_key_raw),
        (
            "dense packed QSA query/gate scratch",
            &scratch.query_gate_projection,
        ),
        ("dense packed QSA query scratch", &scratch.query),
        ("dense packed QSA raw-key scratch", &scratch.key_raw),
        ("dense packed QSA key scratch", &scratch.key),
        ("dense packed QSA value scratch", &scratch.value),
        (
            "dense packed QSA attention-score scratch",
            &scratch.attention_scores,
        ),
        ("dense packed QSA attention scratch", &scratch.attention),
        ("dense packed QSA output scratch", &scratch.output),
    ];
    if let Some(selected) = scratch.selected.as_ref() {
        tensors.extend([
            (
                "selected packed QSA raw index query scratch",
                &selected.index_query_raw,
            ),
            (
                "selected packed QSA index query scratch",
                &selected.index_query,
            ),
            ("selected packed QSA score scratch", &selected.scores),
            (
                "selected packed QSA visible-block scratch",
                &selected.visible_blocks,
            ),
            (
                "selected packed QSA selected-block scratch",
                &selected.selected_blocks,
            ),
            (
                "selected packed QSA selected-count scratch",
                &selected.selected_count,
            ),
            (
                "selected packed QSA selector-status scratch",
                &selected.selector_status,
            ),
            ("selected packed QSA token-ID scratch", &selected.token_ids),
            (
                "selected packed QSA attention-logit scratch",
                &selected.attention_logits,
            ),
        ]);
    }
    tensors
}

pub(crate) fn preflight_dense_packed(
    ctx: &MetalContext,
    weights: QwenSparseAttentionMetalWeights<'_>,
) -> Result<(), Qwen4ExpQsaError> {
    preflight(ctx, weights)?;
    for dtype in [
        weights.query.dtype,
        weights.key.dtype,
        weights.value.dtype,
        weights.output.dtype,
        weights.index_key.dtype,
    ] {
        preflight_dense_packed_projection(ctx, dtype)?;
    }
    for kernel in [
        "kernel_qwen4exp_qsa_pool_publish_packed_f16",
        "kernel_qwen4exp_qsa_commit_pending_packed_f32",
        "kernel_qwen4exp_qsa_fill_block_ids_i32",
        "kernel_qk_rms_norm_rope_f32_packed_consecutive",
        "kernel_scatter_offset_f32_to_f16_kv",
        "kernel_attn_matrix_kq_f32",
        "kernel_attn_matrix_kq_f32_full_tiles",
        "kernel_attn_matrix_softmax_f32",
        "kernel_attn_matrix_kqv_direct_v_f32",
        "kernel_sigmoid_mul_gate_strided_f32",
    ] {
        ctx.pipeline(kernel)?;
    }
    for (name, threads, dynamic_memory) in [
        ("kernel_attn_matrix_kq_f32", 128, 8192),
        ("kernel_attn_matrix_kq_f32_full_tiles", 128, 8192),
        (
            "kernel_attn_matrix_softmax_f32",
            ATTENTION_THREADS,
            8 * size_of::<f32>(),
        ),
        ("kernel_attn_matrix_kqv_direct_v_f32", 128, 8192),
    ] {
        let pipeline = ctx.pipeline(name)?;
        if pipeline.maxTotalThreadsPerThreadgroup() < threads {
            return invalid(format!(
                "dense packed QSA pipeline {name} supports {} threads, needs {threads}",
                pipeline.maxTotalThreadsPerThreadgroup()
            ));
        }
        let memory = pipeline
            .staticThreadgroupMemoryLength()
            .checked_add(dynamic_memory)
            .ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(format!(
                    "dense packed QSA pipeline {name} threadgroup memory overflow"
                ))
            })?;
        if memory > ctx.device.maxThreadgroupMemoryLength() {
            return invalid(format!(
                "dense packed QSA pipeline {name} needs {memory} threadgroup bytes, device has {}",
                ctx.device.maxThreadgroupMemoryLength()
            ));
        }
    }
    Ok(())
}

fn preflight_packed(
    ctx: &MetalContext,
    weights: QwenSparseAttentionMetalWeights<'_>,
    plan: QwenSparseAttentionPackedRangePlan,
) -> Result<(), Qwen4ExpQsaError> {
    preflight_dense_packed(ctx, weights)?;
    if plan.selected_tokens > 0 {
        preflight_dense_packed_projection(ctx, weights.index_query.dtype)?;
        preflight_selected_index_primitives(ctx)?;
        preflight_selected_attention_primitives(ctx)?;
        let tokens = plan.dense_tokens + plan.selected_tokens;
        if mat_vec_q8_0_lcpp_enabled()
            && requires_exact_selected_q8_output(
                weights.geometry,
                weights.output.dtype,
                plan,
                tokens,
            )
        {
            let kernel = "kernel_mat_vec_q8_0_f32_lcpp_batch";
            let pipeline = ctx.pipeline(kernel)?;
            validate_cooperative_pipeline_threads(
                kernel,
                pipeline.threadExecutionWidth(),
                pipeline.maxTotalThreadsPerThreadgroup(),
                128,
                pipeline.staticThreadgroupMemoryLength(),
                32 * 2 * size_of::<f32>(),
                ctx.device.maxThreadgroupMemoryLength(),
            )?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn preflight_dense_packed_projection(
    ctx: &MetalContext,
    dtype: GgmlType,
) -> Result<(), Qwen4ExpQsaError> {
    let kernels: &[&str] = match dtype {
        GgmlType::F32 => &["kernel_mat_mat_f32_f32"],
        GgmlType::Q8_0 => &[
            "kernel_mat_mat_q8_0_f32",
            "kernel_mat_mat_q8_0_f32_n16",
            "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32",
        ],
        GgmlType::BF16 => &["kernel_mat_mat_bf16_f32"],
        _ => {
            return invalid(format!(
                "unsupported dense packed QSA projection dtype {dtype:?}"
            ));
        }
    };
    for kernel in kernels {
        ctx.pipeline(kernel)?;
    }
    Ok(())
}

#[allow(dead_code)]
fn preflight_selected_index_primitives(ctx: &MetalContext) -> Result<(), Qwen4ExpQsaError> {
    for kernel in [
        "kernel_qwen4exp_qsa_norm_rope_packed_f32",
        "kernel_qwen4exp_qsa_expand_ids_packed_i32",
    ] {
        ctx.pipeline(kernel)?;
    }
    for (kernel, dynamic_memory) in [
        ("kernel_qwen4exp_qsa_index_scores_packed_4x128_f16", 0),
        (
            "kernel_deepseek_v4_select_top_k_radix4_ids_f32",
            SELECTOR_SCRATCH_BYTES,
        ),
    ] {
        let pipeline = ctx.pipeline(kernel)?;
        validate_cooperative_pipeline(
            kernel,
            pipeline.threadExecutionWidth(),
            pipeline.maxTotalThreadsPerThreadgroup(),
            pipeline.staticThreadgroupMemoryLength(),
            dynamic_memory,
            ctx.device.maxThreadgroupMemoryLength(),
        )?;
    }
    Ok(())
}

#[allow(dead_code)]
fn preflight_selected_attention_primitives(ctx: &MetalContext) -> Result<(), Qwen4ExpQsaError> {
    ctx.pipeline("kernel_qwen4exp_qsa_audit_selected_i32")?;
    let reset = ctx.pipeline("kernel_qwen4exp_qsa_reset_selected_controls_i32")?;
    validate_selected_reset_pipeline(reset.maxTotalThreadsPerThreadgroup())?;
    for (kernel, threads, dynamic_memory) in [
        (
            "kernel_qwen4exp_qsa_attention_logits_packed_f16",
            PACKED_ATTENTION_THREADS,
            MAIN_HEAD_DIM * size_of::<u16>(),
        ),
        (
            "kernel_qwen4exp_qsa_attention_softmax_value_packed_f16",
            ATTENTION_THREADS,
            ATTENTION_SCRATCH_BYTES,
        ),
    ] {
        let pipeline = ctx.pipeline(kernel)?;
        validate_cooperative_pipeline_threads(
            kernel,
            pipeline.threadExecutionWidth(),
            pipeline.maxTotalThreadsPerThreadgroup(),
            threads,
            pipeline.staticThreadgroupMemoryLength(),
            dynamic_memory,
            ctx.device.maxThreadgroupMemoryLength(),
        )?;
    }
    for (enabled, kernel, threads, dynamic_memory) in [
        (
            configured_qwen4exp_qsa_gqa4_logits_enabled(),
            "kernel_qwen4exp_qsa_attention_logits_packed_gqa4_f16",
            PACKED_ATTENTION_THREADS,
            0,
        ),
        (
            configured_qwen4exp_qsa_gqa4_value_enabled(),
            "kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16",
            ATTENTION_THREADS,
            4 * ATTENTION_SCRATCH_FLOATS * size_of::<f32>(),
        ),
    ] {
        if !enabled {
            continue;
        }
        let pipeline = ctx.pipeline(kernel)?;
        validate_cooperative_pipeline_threads(
            kernel,
            pipeline.threadExecutionWidth(),
            pipeline.maxTotalThreadsPerThreadgroup(),
            threads,
            pipeline.staticThreadgroupMemoryLength(),
            dynamic_memory,
            ctx.device.maxThreadgroupMemoryLength(),
        )?;
    }
    Ok(())
}

fn validate_selected_reset_pipeline(max_threads: usize) -> Result<(), Qwen4ExpQsaError> {
    if max_threads < 32 {
        return invalid(format!(
            "selected QSA reset pipeline supports {max_threads} threads, needs 32"
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn validate_selected_attention_packet(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    norm_weight: &MetalTensor,
    compressed_keys: &MetalTensor,
    query: &MetalTensor,
    query_gate_projection: &MetalTensor,
    key_cache: &MetalTensor,
    value_cache: &MetalTensor,
    attention: &MetalTensor,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    raw_query_offset: usize,
    query_count: usize,
) -> Result<(), Qwen4ExpQsaError> {
    validate_encoder(ctx, enc)?;
    let g = scratch.geometry;
    let plan = g.plan_packed_range(start_position, query_count)?;
    if plan.dense_tokens != 0 || plan.selected_tokens != query_count || plan.selected_bands != 1 {
        return invalid(format!(
            "selected packed QSA attention range start={start_position} queries={query_count} is not one fully selected band"
        ));
    }
    if g.head_dim != MAIN_HEAD_DIM
        || g.kv_heads == 0
        || !g.query_heads.is_multiple_of(g.kv_heads)
        || !(g.query_heads / g.kv_heads).is_multiple_of(PACKED_ATTENTION_HEADS_PER_TG)
    {
        return invalid(format!(
            "selected packed QSA attention requires head dim {MAIN_HEAD_DIM} and GQA groups divisible by {PACKED_ATTENTION_HEADS_PER_TG}"
        ));
    }
    if g.ratio == 0 || g.block_budget() == 0 {
        return invalid("selected packed QSA attention requires nonzero ratio and block budget");
    }
    let selected = scratch.selected.as_ref().ok_or_else(|| {
        Qwen4ExpQsaError::Invalid("selected packed QSA scratch was not admitted".into())
    })?;
    let views = selected.band_views(
        g,
        scratch.capacity,
        scratch.query_tile,
        raw_query_offset,
        query_count,
    )?;
    require_tensor(
        "selected packed QSA index-query norm",
        norm_weight,
        GgmlType::F32,
        &[g.index_head_dim as u64],
        false,
    )?;
    require_read_only_weights(&[("selected packed QSA index-query norm", norm_weight)])?;
    for (name, tensor, dtype, shape, writable) in [
        (
            "selected packed QSA compressed index keys",
            compressed_keys,
            GgmlType::F16,
            vec![g.index_head_dim as u64, g.block_capacity() as u64],
            false,
        ),
        (
            "selected packed QSA query",
            query,
            GgmlType::F32,
            vec![g.query_width() as u64, query_count as u64],
            false,
        ),
        (
            "selected packed QSA query/gate projection",
            query_gate_projection,
            GgmlType::F32,
            vec![g.query_projection_width() as u64, query_count as u64],
            false,
        ),
        (
            "selected packed QSA key cache",
            key_cache,
            GgmlType::F16,
            vec![g.head_dim as u64, g.kv_heads as u64, g.capacity as u64],
            false,
        ),
        (
            "selected packed QSA value cache",
            value_cache,
            GgmlType::F16,
            vec![g.head_dim as u64, g.kv_heads as u64, g.capacity as u64],
            false,
        ),
        (
            "selected packed QSA attention",
            attention,
            GgmlType::F32,
            vec![g.query_width() as u64, query_count as u64],
            true,
        ),
    ] {
        require_tensor(name, tensor, dtype, &shape, writable)?;
    }
    for (name, value) in [
        ("start position", start_position),
        ("query count", query_count),
        ("query end", plan.end_position),
        ("query heads", g.query_heads),
        ("KV heads", g.kv_heads),
        ("head dimension", g.head_dim),
        ("block budget", g.block_budget()),
        ("ratio", g.ratio),
        ("output width", g.output_width()),
        ("cache capacity", g.capacity),
    ] {
        if u32::try_from(value).is_err() {
            return invalid(format!(
                "selected packed QSA attention {name} {value} exceeds u32"
            ));
        }
    }
    for (name, elements) in [
        ("query", g.query_width().checked_mul(query_count)),
        (
            "query/gate projection",
            g.query_projection_width().checked_mul(query_count),
        ),
        (
            "attention logits",
            g.output_width()
                .checked_mul(g.query_heads)
                .and_then(|elements| elements.checked_mul(query_count)),
        ),
        (
            "cache",
            g.head_dim
                .checked_mul(g.kv_heads)
                .and_then(|elements| elements.checked_mul(g.capacity)),
        ),
    ] {
        if elements.is_none() {
            return invalid(format!(
                "selected packed QSA attention {name} element span overflows"
            ));
        }
    }
    let tensors = vec![
        ("selected packed QSA index-query norm", norm_weight),
        ("selected packed QSA compressed index keys", compressed_keys),
        ("selected packed QSA query", query),
        (
            "selected packed QSA query/gate projection",
            query_gate_projection,
        ),
        ("selected packed QSA key cache", key_cache),
        ("selected packed QSA value cache", value_cache),
        ("selected packed QSA attention", attention),
        (
            "selected packed QSA raw index query",
            &views.index_query_raw,
        ),
        ("selected packed QSA index query", &views.index_query),
        ("selected packed QSA scores", &views.scores),
        ("selected packed QSA visible blocks", &views.visible_blocks),
        (
            "selected packed QSA selected blocks",
            &views.selected_blocks,
        ),
        ("selected packed QSA selected count", &views.selected_count),
        (
            "selected packed QSA selector status",
            &views.selector_status,
        ),
        ("selected packed QSA token IDs", &views.token_ids),
        (
            "selected packed QSA attention logits",
            &views.attention_logits,
        ),
    ];
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)?;
    preflight_selected_index_primitives(ctx)?;
    preflight_selected_attention_primitives(ctx)
}

#[allow(dead_code)]
fn encode_selected_index_primitives(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    norm_weight: &MetalTensor,
    compressed_keys: &MetalTensor,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    raw_query_offset: usize,
    query_count: usize,
) -> Result<QwenSparseAttentionSelectedPackedViews, Qwen4ExpQsaError> {
    validate_encoder(ctx, enc)?;
    let g = scratch.geometry;
    let plan = g.plan_packed_range(start_position, query_count)?;
    if plan.dense_tokens != 0 || plan.selected_tokens != query_count || plan.selected_bands != 1 {
        return invalid(format!(
            "selected packed QSA primitive range start={start_position} queries={query_count} is not one fully selected band"
        ));
    }
    let selected = scratch.selected.as_ref().ok_or_else(|| {
        Qwen4ExpQsaError::Invalid("selected packed QSA scratch was not admitted".into())
    })?;
    let views = selected.band_views(
        g,
        scratch.capacity,
        scratch.query_tile,
        raw_query_offset,
        query_count,
    )?;
    require_tensor(
        "selected packed QSA index-query norm",
        norm_weight,
        GgmlType::F32,
        &[g.index_head_dim as u64],
        false,
    )?;
    require_read_only_weights(&[("selected packed QSA index-query norm", norm_weight)])?;
    require_tensor(
        "selected packed QSA compressed index keys",
        compressed_keys,
        GgmlType::F16,
        &[g.index_head_dim as u64, g.block_capacity() as u64],
        false,
    )?;
    for (name, value) in [
        ("start position", start_position),
        ("query count", query_count),
        ("query end", plan.end_position),
        ("block capacity", g.block_capacity()),
        ("block budget", g.block_budget()),
        ("output width", g.output_width()),
    ] {
        if u32::try_from(value).is_err() {
            return invalid(format!("selected packed QSA {name} {value} exceeds u32"));
        }
    }
    let tensors = [
        ("selected packed QSA index-query norm", norm_weight),
        ("selected packed QSA compressed index keys", compressed_keys),
        (
            "selected packed QSA raw index query",
            &views.index_query_raw,
        ),
        ("selected packed QSA index query", &views.index_query),
        ("selected packed QSA scores", &views.scores),
        ("selected packed QSA visible blocks", &views.visible_blocks),
        (
            "selected packed QSA selected blocks",
            &views.selected_blocks,
        ),
        ("selected packed QSA selected count", &views.selected_count),
        (
            "selected packed QSA selector status",
            &views.selector_status,
        ),
        ("selected packed QSA token IDs", &views.token_ids),
        (
            "selected packed QSA attention logits",
            &views.attention_logits,
        ),
    ];
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)?;
    preflight_selected_index_primitives(ctx)?;

    encode_norm_rope_packed(
        ctx,
        enc,
        &views.index_query_raw,
        norm_weight,
        &views.index_query,
        start_position,
        query_count,
        g.index_query_heads,
        g.index_head_dim,
        g.rotary_dim,
        g.theta,
        g.eps,
    )?;
    encode_index_scores_packed(
        ctx,
        enc,
        &views.index_query,
        compressed_keys,
        &views.scores,
        &views.visible_blocks,
        start_position,
        query_count,
        g.ratio,
        g.block_capacity(),
    )?;
    encode_select_blocks_tensors(
        ctx,
        enc,
        &views.scores,
        &views.visible_blocks,
        &views.selected_blocks,
        &views.selected_count,
        &views.selector_status,
        g.block_capacity(),
        g.block_budget(),
        query_count,
    )?;
    encode_expand_ids_packed(
        ctx,
        enc,
        &views.visible_blocks,
        &views.selected_blocks,
        &views.selected_count,
        &views.selector_status,
        &views.token_ids,
        start_position,
        query_count,
        g.block_budget(),
        g.ratio,
        g.output_width(),
    )?;
    Ok(views)
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_selected_attention_packet(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    norm_weight: &MetalTensor,
    compressed_keys: &MetalTensor,
    query: &MetalTensor,
    query_gate_projection: &MetalTensor,
    key_cache: &MetalTensor,
    value_cache: &MetalTensor,
    attention: &MetalTensor,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    raw_query_offset: usize,
    query_count: usize,
) -> Result<QwenSparseAttentionSelectedPackedViews, Qwen4ExpQsaError> {
    validate_selected_attention_packet(
        ctx,
        enc,
        norm_weight,
        compressed_keys,
        query,
        query_gate_projection,
        key_cache,
        value_cache,
        attention,
        scratch,
        start_position,
        raw_query_offset,
        query_count,
    )?;
    let views = encode_selected_index_primitives(
        ctx,
        enc,
        norm_weight,
        compressed_keys,
        scratch,
        start_position,
        raw_query_offset,
        query_count,
    )?;
    encode_attention_logits_packed(
        ctx,
        enc,
        query,
        key_cache,
        &views.token_ids,
        &views.selected_count,
        &views.selector_status,
        &views.attention_logits,
        scratch.geometry,
        start_position,
        query_count,
    )?;
    encode_attention_softmax_value_packed(
        ctx,
        enc,
        query_gate_projection,
        value_cache,
        &views.token_ids,
        &views.selected_count,
        &views.selector_status,
        &views.attention_logits,
        attention,
        scratch.geometry,
        start_position,
        query_count,
    )?;
    Ok(views)
}

#[allow(clippy::too_many_arguments)]
fn encode_projection_scalar_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    rows: usize,
) -> Result<(), Qwen4ExpQsaError> {
    for row in 0..rows {
        let input_offset = row
            .checked_mul(n_in)
            .ok_or_else(|| Qwen4ExpQsaError::Invalid("scalar-row input offset overflow".into()))?;
        let output_offset = row
            .checked_mul(n_out)
            .ok_or_else(|| Qwen4ExpQsaError::Invalid("scalar-row output offset overflow".into()))?;
        let input_row = input.view_subrange(input_offset as u64, vec![n_in as u64]);
        let output_row = output.view_subrange(output_offset as u64, vec![n_out as u64]);
        encode_mat_vec_dispatch(ctx, enc, weight, &input_row, &output_row, n_in, n_out)?;
    }
    Ok(())
}

fn requires_exact_selected_q8_output(
    geometry: QwenSparseAttentionMetalGeometry,
    output_dtype: GgmlType,
    plan: QwenSparseAttentionPackedRangePlan,
    tokens: usize,
) -> bool {
    // The generic Q8_0 mat-mat half-stages a physical 32-column tile. At the
    // released QSA shape its N=2..4 error is amplified by selected continuation.
    plan.selected_tokens > 0
        && (2..=4).contains(&tokens)
        && output_dtype == GgmlType::Q8_0
        && geometry.query_width() == 6_144
        && geometry.hidden_size == 2_560
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_packed_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    tokens: usize,
    plan: QwenSparseAttentionPackedRangePlan,
    layer: u32,
    mut profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
) -> Result<(), Qwen4ExpQsaError> {
    let g = workspace.geometry;
    let sequence_length = plan.end_position;
    let views = scratch.views(tokens)?;
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::detail(
            "qsa.index_projection",
            layer,
            MixerKind::QwenSparseAttention,
        ),
    )?;
    if weights.index_key.dtype == GgmlType::BF16 {
        // These values become persistent selector state. Keep F32 activations
        // instead of taking the global BF16-activation mat-mat shortcut.
        encode_mat_mat_bf16_f32(
            ctx,
            enc,
            weights.index_key,
            input,
            &views.index_key_raw,
            g.hidden_size,
            g.index_head_dim,
            tokens,
        )?;
    } else {
        encode_mat_mat_dispatch(
            ctx,
            enc,
            weights.index_key,
            input,
            &views.index_key_raw,
            g.hidden_size,
            g.index_head_dim,
            tokens,
        )?;
    }
    end_optional(&mut profile, enc, marker)?;
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::detail(
            "qsa.index_state",
            layer,
            MixerKind::QwenSparseAttention,
        ),
    )?;
    encode_packed_index_state(
        ctx,
        enc,
        &views.index_key_raw,
        workspace,
        weights.index_key_norm,
        start_position,
        tokens,
        sequence_length,
    )?;
    end_optional(&mut profile, enc, marker)?;
    if plan.selected_tokens == 0 {
        let visible_blocks = sequence_length / g.ratio;
        let marker = begin_optional(
            &mut profile,
            enc,
            Qwen4ExpPackedProfileLabel::detail(
                "qsa.visible_blocks",
                layer,
                MixerKind::QwenSparseAttention,
            ),
        )?;
        encode_fill_blocks(ctx, enc, workspace, visible_blocks, sequence_length)?;
        end_optional(&mut profile, enc, marker)?;
    }

    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::detail(
            "qsa.qkv_projections",
            layer,
            MixerKind::QwenSparseAttention,
        ),
    )?;
    encode_mat_mat_dispatch(
        ctx,
        enc,
        weights.query,
        input,
        &views.query_gate_projection,
        g.hidden_size,
        g.query_projection_width(),
        tokens,
    )?;
    encode_mat_mat_dispatch(
        ctx,
        enc,
        weights.key,
        input,
        &views.key_raw,
        g.hidden_size,
        g.kv_width(),
        tokens,
    )?;
    encode_mat_mat_dispatch(
        ctx,
        enc,
        weights.value,
        input,
        &views.value,
        g.hidden_size,
        g.kv_width(),
        tokens,
    )?;
    end_optional(&mut profile, enc, marker)?;
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::detail("qsa.norm_rope", layer, MixerKind::QwenSparseAttention),
    )?;
    encode_qk_rms_norm_rope_f32_packed_consecutive(
        ctx,
        enc,
        &views.query_gate_projection,
        weights.query_norm,
        &views.query,
        &views.key_raw,
        weights.key_norm,
        &views.key,
        tokens,
        g.query_heads,
        g.kv_heads,
        g.head_dim,
        g.rotary_dim,
        start_position as u32,
        g.eps,
        g.theta,
    )?;
    end_optional(&mut profile, enc, marker)?;
    let cache_offset = start_position * g.kv_width();
    let cache_elements = tokens * g.kv_width();
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::detail(
            "qsa.cache_scatter",
            layer,
            MixerKind::QwenSparseAttention,
        ),
    )?;
    encode_scatter_offset_f32_to_f16_kv(
        ctx,
        enc,
        &views.key,
        &views.value,
        &workspace.key_cache,
        &workspace.value_cache,
        cache_offset,
        cache_elements,
    )?;
    end_optional(&mut profile, enc, marker)?;

    let group = g.query_heads / g.kv_heads;
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::detail("qsa.attention", layer, MixerKind::QwenSparseAttention),
    )?;
    if plan.dense_tokens > 0 {
        let dense_end = start_position + plan.dense_tokens;
        let mut tile_start = 0;
        while tile_start < plan.dense_tokens {
            let rows = (plan.dense_tokens - tile_start).min(scratch.query_tile);
            let query = views.query.view_subrange(
                (tile_start * g.query_width()) as u64,
                vec![g.query_width() as u64, rows as u64],
            );
            let attention = views.attention.view_subrange(
                (tile_start * g.query_width()) as u64,
                vec![g.query_width() as u64, rows as u64],
            );
            let scores = scratch.score_view(rows, dense_end)?;
            let base_position = start_position + tile_start;
            encode_attn_matrix_kq_f32(
                ctx,
                enc,
                &query,
                &workspace.key_cache,
                &scores,
                rows,
                base_position,
                dense_end,
                g.kv_width(),
                g.query_heads,
                g.kv_heads,
                group,
                g.head_dim,
                true,
            )?;
            encode_attn_matrix_softmax_f32(
                ctx,
                enc,
                &scores,
                rows,
                base_position,
                dense_end,
                g.query_heads,
                g.kv_heads,
                group,
                g.head_dim,
            )?;
            encode_attn_matrix_kqv_direct_v_f32(
                ctx,
                enc,
                &scores,
                &workspace.value_cache,
                &attention,
                rows,
                base_position,
                dense_end,
                g.kv_width(),
                g.query_heads,
                g.kv_heads,
                group,
                g.head_dim,
                true,
            )?;
            tile_start += rows;
        }
    }
    if plan.selected_tokens > 0 {
        let selected = scratch.selected.as_ref().ok_or_else(|| {
            Qwen4ExpQsaError::Invalid("packed QSA selected scratch was not admitted".into())
        })?;
        let selected_raw_query = selected.raw_query_projection_view(
            g,
            scratch.capacity,
            scratch.query_tile,
            plan.selected_tokens,
        )?;
        let input_offset = plan
            .selected_offset
            .checked_mul(g.hidden_size)
            .ok_or_else(|| {
                Qwen4ExpQsaError::Invalid("packed QSA selected input offset overflow".into())
            })?;
        let selected_input = input.view_subrange(
            input_offset as u64,
            vec![g.hidden_size as u64, plan.selected_tokens as u64],
        );
        let projection_tag = crate::metal::dispatch_census_tag_scope(|| {
            "qwen4exp.qsa.selected_index_projection".into()
        });
        if weights.index_query.dtype == GgmlType::BF16 {
            encode_mat_mat_bf16_f32(
                ctx,
                enc,
                weights.index_query,
                &selected_input,
                &selected_raw_query,
                g.hidden_size,
                g.index_query_width(),
                plan.selected_tokens,
            )?;
        } else {
            encode_mat_mat_dispatch(
                ctx,
                enc,
                weights.index_query,
                &selected_input,
                &selected_raw_query,
                g.hidden_size,
                g.index_query_width(),
                plan.selected_tokens,
            )?;
        }
        drop(projection_tag);

        let mut band_offset = 0;
        let mut band_ordinal = 0;
        while band_offset < plan.selected_tokens {
            let band_rows = (plan.selected_tokens - band_offset).min(scratch.query_tile);
            let band_views = selected.band_views(
                g,
                scratch.capacity,
                scratch.query_tile,
                band_offset,
                band_rows,
            )?;
            let local_offset = plan
                .selected_offset
                .checked_add(band_offset)
                .ok_or_else(|| {
                    Qwen4ExpQsaError::Invalid(
                        "packed QSA selected band local offset overflow".into(),
                    )
                })?;
            let query_offset = local_offset.checked_mul(g.query_width()).ok_or_else(|| {
                Qwen4ExpQsaError::Invalid("packed QSA selected query offset overflow".into())
            })?;
            let projected_offset = local_offset
                .checked_mul(g.query_projection_width())
                .ok_or_else(|| {
                    Qwen4ExpQsaError::Invalid(
                        "packed QSA selected query/gate offset overflow".into(),
                    )
                })?;
            let selected_query = views.query.view_subrange(
                query_offset as u64,
                vec![g.query_width() as u64, band_rows as u64],
            );
            let selected_query_gate = views.query_gate_projection.view_subrange(
                projected_offset as u64,
                vec![g.query_projection_width() as u64, band_rows as u64],
            );
            let selected_attention = views.attention.view_subrange(
                query_offset as u64,
                vec![g.query_width() as u64, band_rows as u64],
            );
            let band_tag = crate::metal::dispatch_census_tag_scope(|| {
                format!("qwen4exp.qsa.selected_band.{band_ordinal}")
            });
            encode_selected_control_reset(
                ctx,
                enc,
                &band_views.visible_blocks,
                &band_views.selected_count,
                &band_views.selector_status,
                band_rows,
            )?;
            let packet = encode_selected_attention_packet(
                ctx,
                enc,
                weights.index_query_norm,
                &workspace.compressed_index_keys,
                &selected_query,
                &selected_query_gate,
                &workspace.key_cache,
                &workspace.value_cache,
                &selected_attention,
                scratch,
                start_position + local_offset,
                band_offset,
                band_rows,
            )?;
            #[cfg(test)]
            {
                let selected_input = input.view_subrange(
                    (local_offset * g.hidden_size) as u64,
                    vec![g.hidden_size as u64, band_rows as u64],
                );
                encode_qwen4exp_qsa_decision_capture(
                    ctx,
                    enc,
                    &selected_input,
                    &packet.index_query,
                    &packet.scores,
                    &packet.visible_blocks,
                    &packet.selected_blocks,
                    &packet.selected_count,
                    &packet.selector_status,
                    g,
                    start_position + local_offset,
                    band_rows,
                )?;
            }
            encode_selected_audit(
                ctx,
                enc,
                &packet.selected_count,
                &packet.selector_status,
                &workspace.selected_count,
                &workspace.selector_status,
                &workspace.visible_blocks,
                band_rows,
                g.block_budget(),
                band_ordinal,
            )?;
            drop(band_tag);
            band_offset += band_rows;
            band_ordinal += 1;
        }
    }
    end_optional(&mut profile, enc, marker)?;
    if plan.dense_tokens > 0 {
        let marker = begin_optional(
            &mut profile,
            enc,
            Qwen4ExpPackedProfileLabel::detail("qsa.gate", layer, MixerKind::QwenSparseAttention),
        )?;
        let dense_query_gate = views.query_gate_projection.view_subrange(
            0,
            vec![g.query_projection_width() as u64, plan.dense_tokens as u64],
        );
        let dense_attention = views
            .attention
            .view_subrange(0, vec![g.query_width() as u64, plan.dense_tokens as u64]);
        encode_sigmoid_mul_gate_strided_f32(
            ctx,
            enc,
            &dense_query_gate,
            &dense_attention,
            &dense_attention,
            plan.dense_tokens * g.query_heads,
            g.head_dim,
            2 * g.head_dim,
            g.head_dim,
        )?;
        end_optional(&mut profile, enc, marker)?;
    }
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::detail("qsa.output", layer, MixerKind::QwenSparseAttention),
    )?;
    let output_tag =
        crate::metal::dispatch_census_tag_scope(|| "qwen4exp.qsa.output_projection".into());
    if requires_exact_selected_q8_output(g, weights.output.dtype, plan, tokens) {
        if mat_vec_q8_0_lcpp_enabled() {
            encode_mat_vec_q8_0_batch_f32(
                ctx,
                enc,
                weights.output,
                &views.attention,
                &views.output,
                g.query_width(),
                g.hidden_size,
                tokens,
            )?;
        } else {
            encode_projection_scalar_rows(
                ctx,
                enc,
                weights.output,
                &views.attention,
                &views.output,
                g.query_width(),
                g.hidden_size,
                tokens,
            )?;
        }
    } else {
        encode_mat_mat_dispatch(
            ctx,
            enc,
            weights.output,
            &views.attention,
            &views.output,
            g.query_width(),
            g.hidden_size,
            tokens,
        )?;
    }
    drop(output_tag);
    end_optional(&mut profile, enc, marker)?;
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedIndexStateArgs {
    start_position: u32,
    n_tokens: u32,
    ratio: u32,
    head_dim: u32,
    rotary_dim: u32,
    first_block: u32,
    block_count: u32,
    block_capacity: u32,
    theta: f32,
    eps: f32,
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_packed_index_state(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    raw_keys: &MetalTensor,
    workspace: &QwenSparseAttentionMetalWorkspace,
    weight: &MetalTensor,
    start_position: usize,
    tokens: usize,
    sequence_length: usize,
) -> Result<(), Qwen4ExpQsaError> {
    let g = workspace.geometry;
    let first_block = start_position / g.ratio;
    let completed_end = sequence_length / g.ratio;
    let block_count = completed_end.checked_sub(first_block).ok_or_else(|| {
        Qwen4ExpQsaError::Invalid("packed QSA completed block range underflow".into())
    })?;
    let published_end = first_block.checked_add(block_count).ok_or_else(|| {
        Qwen4ExpQsaError::Invalid("packed QSA completed block range overflow".into())
    })?;
    if published_end > g.block_capacity() {
        return invalid(format!(
            "packed QSA completed block end {published_end} exceeds capacity {}",
            g.block_capacity()
        ));
    }
    let args = PackedIndexStateArgs {
        start_position: start_position as u32,
        n_tokens: tokens as u32,
        ratio: g.ratio as u32,
        head_dim: g.index_head_dim as u32,
        rotary_dim: g.rotary_dim as u32,
        first_block: first_block as u32,
        block_count: block_count as u32,
        block_capacity: g.block_capacity() as u32,
        theta: g.theta,
        eps: g.eps,
    };
    if block_count > 0 {
        let publish = ctx.pipeline("kernel_qwen4exp_qsa_pool_publish_packed_f16")?;
        enc.set_pipeline(&publish);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, &workspace.pending_index_keys);
        enc.set_tensor(2, raw_keys);
        enc.set_tensor(3, weight);
        enc.set_tensor(4, &workspace.compressed_index_keys);
        dispatch_1d(enc, &publish, block_count);
    }
    let pending = ctx.pipeline("kernel_qwen4exp_qsa_commit_pending_packed_f32")?;
    enc.set_pipeline(&pending);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, raw_keys);
    enc.set_tensor(2, &workspace.pending_index_keys);
    dispatch_1d(enc, &pending, g.ratio * g.index_head_dim);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_qwen4exp_qsa_decision_capture(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    index_queries: &MetalTensor,
    scores: &MetalTensor,
    visible_blocks: &MetalTensor,
    selected_blocks: &MetalTensor,
    selected_count: &MetalTensor,
    selector_status: &MetalTensor,
    geometry: QwenSparseAttentionMetalGeometry,
    start_position: usize,
    query_count: usize,
) -> Result<(), Qwen4ExpQsaError> {
    QWEN4EXP_QSA_DECISION_CAPTURE.with(|slot| {
        let binding = slot.borrow();
        let Some(binding) = binding.as_ref() else {
            return Ok(());
        };
        let layer = QWEN4EXP_QSA_CAPTURE_LAYER.with(|slot| slot.get()).ok_or_else(|| {
            Qwen4ExpQsaError::Invalid(
                "QSA decision capture is active without an integrated layer scope".into(),
            )
        })?;
        let banks = &binding.banks;
        let layer_ordinal = banks.layer_ordinal(layer).ok_or_else(|| {
            Qwen4ExpQsaError::Invalid(format!(
                "QSA decision capture received unexpected layer {layer}"
            ))
        })?;
        if geometry.hidden_size() != banks.hidden_size
            || geometry.index_query_width() != banks.index_query_width
            || geometry.block_capacity() != banks.block_capacity
            || geometry.block_budget() != banks.block_budget
        {
            return invalid(format!(
                "QSA decision capture layer {layer} geometry differs from its banks"
            ));
        }
        if query_count == 0 {
            return invalid("QSA decision capture source query count must be nonzero");
        }
        let end_position = start_position.checked_add(query_count).ok_or_else(|| {
            Qwen4ExpQsaError::Invalid("QSA decision capture source range overflow".into())
        })?;
        let capture_end = banks
            .start_position
            .checked_add(banks.tokens)
            .expect("validated QSA decision capture range");
        let overlap_start = start_position.max(banks.start_position);
        let overlap_end = end_position.min(capture_end);
        if overlap_start >= overlap_end {
            return Ok(());
        }
        let source_row = overlap_start - start_position;
        let destination_row = overlap_start - banks.start_position;
        let rows = overlap_end - overlap_start;
        let expected_elements = [
            ("input", input, GgmlType::F32, geometry.hidden_size()),
            (
                "index queries",
                index_queries,
                GgmlType::F32,
                geometry.index_query_width(),
            ),
            (
                "scores",
                scores,
                GgmlType::F32,
                geometry.block_capacity(),
            ),
            ("visible blocks", visible_blocks, GgmlType::I32, 1),
            (
                "selected blocks",
                selected_blocks,
                GgmlType::I32,
                geometry.block_budget(),
            ),
            ("selected count", selected_count, GgmlType::I32, 1),
            ("selector status", selector_status, GgmlType::I32, 1),
        ];
        for (name, tensor, dtype, row_width) in expected_elements {
            let elements = row_width.checked_mul(query_count).ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(format!(
                    "QSA decision capture {name} source size overflow"
                ))
            })?;
            if tensor.dtype != dtype || tensor.n_elements() as usize != elements {
                return invalid(format!(
                    "QSA decision capture {name} must be {dtype:?} with {elements} elements, got {:?} {}",
                    tensor.dtype,
                    tensor.n_elements()
                ));
            }
            require_range(name, tensor)?;
        }

        let layer_row = layer_ordinal
            .checked_mul(banks.tokens)
            .and_then(|offset| offset.checked_add(destination_row))
            .ok_or_else(|| {
                Qwen4ExpQsaError::Invalid("QSA decision capture destination row overflow".into())
            })?;
        let destination = |name: &str,
                           tensor: &MetalTensor,
                           row_width: usize,
                           dtype: GgmlType|
         -> Result<MetalTensor, Qwen4ExpQsaError> {
            let element_offset = layer_row.checked_mul(row_width).ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(format!(
                    "QSA decision capture {name} destination offset overflow"
                ))
            })?;
            let elements = rows.checked_mul(row_width).ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(format!(
                    "QSA decision capture {name} destination size overflow"
                ))
            })?;
            let view = tensor.view_subrange(element_offset as u64, vec![elements as u64]);
            require_tensor(name, &view, dtype, &[elements as u64], true)?;
            Ok(view)
        };
        let input_destination = destination(
            "QSA decision capture inputs",
            &banks.inputs,
            banks.hidden_size,
            GgmlType::F32,
        )?;
        let query_destination = destination(
            "QSA decision capture index queries",
            &banks.index_queries,
            banks.index_query_width,
            GgmlType::F32,
        )?;
        let score_destination = destination(
            "QSA decision capture scores",
            &banks.scores,
            banks.block_capacity,
            GgmlType::F32,
        )?;
        let visible_destination = destination(
            "QSA decision capture visible blocks",
            &banks.visible_blocks,
            1,
            GgmlType::I32,
        )?;
        let selected_destination = destination(
            "QSA decision capture selected blocks",
            &banks.selected_blocks,
            banks.block_budget,
            GgmlType::I32,
        )?;
        let count_destination = destination(
            "QSA decision capture selected count",
            &banks.selected_count,
            1,
            GgmlType::I32,
        )?;
        let status_destination = destination(
            "QSA decision capture selector status",
            &banks.selector_status,
            1,
            GgmlType::I32,
        )?;
        let tensors = [
            ("QSA decision input source", input),
            ("QSA decision query source", index_queries),
            ("QSA decision score source", scores),
            ("QSA decision visibility source", visible_blocks),
            ("QSA decision ID source", selected_blocks),
            ("QSA decision count source", selected_count),
            ("QSA decision status source", selector_status),
            ("QSA decision input destination", &input_destination),
            ("QSA decision query destination", &query_destination),
            ("QSA decision score destination", &score_destination),
            ("QSA decision visibility destination", &visible_destination),
            ("QSA decision ID destination", &selected_destination),
            ("QSA decision count destination", &count_destination),
            ("QSA decision status destination", &status_destination),
        ];
        require_same_device(ctx, &tensors)?;
        require_disjoint(&tensors)?;

        {
            let seen = binding.seen.borrow();
            for row in destination_row..destination_row + rows {
                let index = layer_ordinal * banks.tokens + row;
                if seen[index] {
                    return invalid(format!(
                        "QSA decision capture duplicated layer {layer} position {}",
                        banks.start_position + row
                    ));
                }
            }
        }

        let copy_f32 = |kind: &'static str,
                        source: &MetalTensor,
                        source_width: usize,
                        destination: &MetalTensor|
         -> Result<(), Qwen4ExpQsaError> {
            let _tag = crate::metal::dispatch_census_tag_scope(|| {
                format!(
                    "qwen4exp.qsa_decision_capture.layer{layer}.position{overlap_start}.{kind}"
                )
            });
            encode_copy_offset_f32(
                ctx,
                enc,
                source,
                source_row * source_width,
                destination,
                rows * source_width,
            )?;
            Ok(())
        };
        let copy_i32 = |kind: &'static str,
                        source: &MetalTensor,
                        source_width: usize,
                        destination: &MetalTensor|
         -> Result<(), Qwen4ExpQsaError> {
            let _tag = crate::metal::dispatch_census_tag_scope(|| {
                format!(
                    "qwen4exp.qsa_decision_capture.layer{layer}.position{overlap_start}.{kind}"
                )
            });
            encode_copy_offset_i32(
                ctx,
                enc,
                source,
                source_row * source_width,
                destination,
                rows * source_width,
            )?;
            Ok(())
        };
        copy_f32("input", input, banks.hidden_size, &input_destination)?;
        copy_f32(
            "index_query",
            index_queries,
            banks.index_query_width,
            &query_destination,
        )?;
        copy_f32("scores", scores, banks.block_capacity, &score_destination)?;
        copy_i32("visible", visible_blocks, 1, &visible_destination)?;
        copy_i32(
            "selected_ids",
            selected_blocks,
            banks.block_budget,
            &selected_destination,
        )?;
        copy_i32("selected_count", selected_count, 1, &count_destination)?;
        copy_i32("status", selector_status, 1, &status_destination)?;

        let mut seen = binding.seen.borrow_mut();
        for row in destination_row..destination_row + rows {
            let index = layer_ordinal * banks.tokens + row;
            debug_assert!(!seen[index]);
            seen[index] = true;
        }
        binding
            .records
            .borrow_mut()
            .push(Qwen4ExpQsaDecisionCaptureRecord {
                layer,
                start_position: overlap_start,
                query_count: rows,
            });
        Ok(())
    })
}

fn encode_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &QwenSparseAttentionMetalWorkspace,
    position: usize,
    sequence_length: usize,
) -> Result<(), Qwen4ExpQsaError> {
    let g = workspace.geometry;
    #[cfg(test)]
    let child_index = crate::qwen4exp_child_profile::span(enc, "index_projection_norm_pending");
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.index_query,
        input,
        &workspace.index_query_raw,
        g.hidden_size,
        g.index_query_width(),
    )?;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.index_key,
        input,
        &workspace.index_key_raw,
        g.hidden_size,
        g.index_head_dim,
    )?;
    encode_norm_rope(
        ctx,
        enc,
        &workspace.index_query_raw,
        weights.index_query_norm,
        &workspace.index_query,
        g.index_query_heads,
        g.index_head_dim,
        g.rotary_dim,
        position,
        g.theta,
        g.eps,
    )?;
    encode_write_pending(ctx, enc, workspace, position % g.ratio)?;
    #[cfg(test)]
    drop(child_index);
    if sequence_length.is_multiple_of(g.ratio) {
        #[cfg(test)]
        let _child_pool = crate::qwen4exp_child_profile::span(enc, "pool_publish");
        encode_pool_publish(
            ctx,
            enc,
            workspace,
            weights.index_key_norm,
            sequence_length / g.ratio - 1,
        )?;
    }

    #[cfg(test)]
    let child_selection = crate::qwen4exp_child_profile::span(enc, "index_score_select_expand");
    let visible_blocks = sequence_length / g.ratio;
    if visible_blocks > 0 {
        encode_index_scores(ctx, enc, workspace, visible_blocks)?;
    }
    if visible_blocks > g.block_budget() {
        encode_select_blocks(ctx, enc, workspace, visible_blocks)?;
    } else {
        encode_fill_blocks(ctx, enc, workspace, visible_blocks, sequence_length)?;
    }
    encode_expand_ids(ctx, enc, workspace, visible_blocks, sequence_length)?;
    #[cfg(test)]
    drop(child_selection);
    #[cfg(test)]
    if visible_blocks > g.block_budget() {
        encode_qwen4exp_qsa_decision_capture(
            ctx,
            enc,
            input,
            &workspace.index_query,
            &workspace.scores,
            &workspace.visible_blocks,
            &workspace.selected_blocks,
            &workspace.selected_count,
            &workspace.selector_status,
            g,
            position,
            1,
        )?;
    }

    #[cfg(test)]
    let child_qkv = crate::qwen4exp_child_profile::span(enc, "qkv_norm_publish");
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.query,
        input,
        &workspace.query_gate_projection,
        g.hidden_size,
        g.query_projection_width(),
    )?;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.key,
        input,
        &workspace.key_raw,
        g.hidden_size,
        g.kv_width(),
    )?;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.value,
        input,
        &workspace.value,
        g.hidden_size,
        g.kv_width(),
    )?;
    encode_qgate_norm_rope(ctx, enc, workspace, weights.query_norm, position)?;
    encode_norm_rope(
        ctx,
        enc,
        &workspace.key_raw,
        weights.key_norm,
        &workspace.key,
        g.kv_heads,
        g.head_dim,
        g.rotary_dim,
        position,
        g.theta,
        g.eps,
    )?;
    encode_publish_kv(ctx, enc, workspace, position)?;
    #[cfg(test)]
    drop(child_qkv);
    #[cfg(test)]
    let child_attention = crate::qwen4exp_child_profile::span(enc, "attention_logits_value");
    let active_id_count =
        visible_blocks.min(g.block_budget()) * g.ratio + sequence_length % g.ratio;
    #[cfg(test)]
    let override_split =
        split_decode_probe::try_encode(ctx, enc, workspace, position, active_id_count)?;
    #[cfg(test)]
    let split = if let Some(split) = override_split {
        split
    } else if let Some(scratch) = workspace.split_decode_scratch.as_ref()
        && split_decode::eligible(g, active_id_count)
    {
        split_decode::encode(ctx, enc, workspace, scratch, active_id_count)?;
        true
    } else {
        false
    };
    #[cfg(not(test))]
    let split = false;
    if !split {
        encode_attention_logits(ctx, enc, workspace, active_id_count)?;
        encode_attention_softmax_value(ctx, enc, workspace, active_id_count)?;
    }
    #[cfg(test)]
    drop(child_attention);
    #[cfg(test)]
    let _child_output = crate::qwen4exp_child_profile::span(enc, "qsa_output_projection");
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.output,
        &workspace.attention,
        &workspace.output,
        g.query_width(),
        g.hidden_size,
    )?;
    Ok(())
}

pub(crate) fn validate_contract(
    ctx: &MetalContext,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &QwenSparseAttentionMetalWorkspace,
) -> Result<(), Qwen4ExpQsaError> {
    let g = workspace.geometry;
    require_tensor(
        "QSA input",
        input,
        GgmlType::F32,
        &[g.hidden_size as u64],
        false,
    )?;
    require_projection(
        "QSA query projection",
        weights.query,
        g.hidden_size,
        g.query_projection_width(),
        ProjectionRole::Main,
    )?;
    require_projection(
        "QSA key projection",
        weights.key,
        g.hidden_size,
        g.kv_width(),
        ProjectionRole::Main,
    )?;
    require_projection(
        "QSA value projection",
        weights.value,
        g.hidden_size,
        g.kv_width(),
        ProjectionRole::Main,
    )?;
    require_projection(
        "QSA output projection",
        weights.output,
        g.query_width(),
        g.hidden_size,
        ProjectionRole::Main,
    )?;
    require_projection(
        "QSA index query projection",
        weights.index_query,
        g.hidden_size,
        g.index_query_width(),
        ProjectionRole::Index,
    )?;
    require_projection(
        "QSA index key projection",
        weights.index_key,
        g.hidden_size,
        g.index_head_dim,
        ProjectionRole::Index,
    )?;
    for (name, tensor, width) in [
        ("QSA query norm", weights.query_norm, g.head_dim),
        ("QSA key norm", weights.key_norm, g.head_dim),
        (
            "QSA index query norm",
            weights.index_query_norm,
            g.index_head_dim,
        ),
        (
            "QSA index key norm",
            weights.index_key_norm,
            g.index_head_dim,
        ),
    ] {
        require_tensor(name, tensor, GgmlType::F32, &[width as u64], false)?;
    }
    require_read_only_weights(&[
        ("QSA query projection", weights.query),
        ("QSA key projection", weights.key),
        ("QSA value projection", weights.value),
        ("QSA output projection", weights.output),
        ("QSA query norm", weights.query_norm),
        ("QSA key norm", weights.key_norm),
        ("QSA index query projection", weights.index_query),
        ("QSA index key projection", weights.index_key),
        ("QSA index query norm", weights.index_query_norm),
        ("QSA index key norm", weights.index_key_norm),
    ])?;
    validate_workspace_tensors(workspace)?;
    let mut tensors = vec![
        ("QSA input", input),
        ("QSA query projection", weights.query),
        ("QSA key projection", weights.key),
        ("QSA value projection", weights.value),
        ("QSA output projection", weights.output),
        ("QSA query norm", weights.query_norm),
        ("QSA key norm", weights.key_norm),
        ("QSA index query projection", weights.index_query),
        ("QSA index key projection", weights.index_key),
        ("QSA index query norm", weights.index_query_norm),
        ("QSA index key norm", weights.index_key_norm),
    ];
    tensors.extend(workspace_tensors(workspace));
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)
}

fn validate_workspace_tensors(
    workspace: &QwenSparseAttentionMetalWorkspace,
) -> Result<(), Qwen4ExpQsaError> {
    let g = workspace.geometry;
    #[cfg(test)]
    if let Some(scratch) = &workspace.split_decode_scratch {
        if !g.supports_split_decode() {
            return invalid("split scratch on unsupported QSA geometry");
        }
        require_tensor(
            "QSA split scratch",
            scratch,
            GgmlType::F32,
            &[split_decode::SCRATCH_FLOATS as u64],
            true,
        )?;
    }
    for (name, tensor, dtype, shape) in [
        (
            "QSA index query raw",
            &workspace.index_query_raw,
            GgmlType::F32,
            vec![g.index_head_dim as u64, g.index_query_heads as u64, 1],
        ),
        (
            "QSA index query",
            &workspace.index_query,
            GgmlType::F32,
            vec![g.index_head_dim as u64, g.index_query_heads as u64, 1],
        ),
        (
            "QSA index key raw",
            &workspace.index_key_raw,
            GgmlType::F32,
            vec![g.index_head_dim as u64],
        ),
        (
            "QSA pending index keys",
            &workspace.pending_index_keys,
            GgmlType::F32,
            vec![g.index_head_dim as u64, g.ratio as u64],
        ),
        (
            "QSA compressed index keys",
            &workspace.compressed_index_keys,
            GgmlType::F16,
            vec![g.index_head_dim as u64, g.block_capacity() as u64],
        ),
        (
            "QSA scores",
            &workspace.scores,
            GgmlType::F32,
            vec![g.block_capacity() as u64, 1],
        ),
        (
            "QSA visible blocks",
            &workspace.visible_blocks,
            GgmlType::I32,
            vec![1],
        ),
        (
            "QSA selected blocks",
            &workspace.selected_blocks,
            GgmlType::I32,
            vec![g.block_budget() as u64, 1],
        ),
        (
            "QSA selected count",
            &workspace.selected_count,
            GgmlType::I32,
            vec![1],
        ),
        (
            "QSA selector status",
            &workspace.selector_status,
            GgmlType::I32,
            vec![1],
        ),
        (
            "QSA token IDs",
            &workspace.token_ids,
            GgmlType::I32,
            vec![g.output_width() as u64],
        ),
        (
            "QSA attention logits",
            &workspace.attention_logits,
            GgmlType::F32,
            vec![g.output_width() as u64, g.query_heads as u64],
        ),
        (
            "QSA query/gate projection",
            &workspace.query_gate_projection,
            GgmlType::F32,
            vec![g.query_projection_width() as u64],
        ),
        (
            "QSA query",
            &workspace.query,
            GgmlType::F32,
            vec![g.query_width() as u64],
        ),
        (
            "QSA raw gate",
            &workspace.raw_gate,
            GgmlType::F32,
            vec![g.query_width() as u64],
        ),
        (
            "QSA key raw",
            &workspace.key_raw,
            GgmlType::F32,
            vec![g.kv_width() as u64],
        ),
        (
            "QSA key",
            &workspace.key,
            GgmlType::F32,
            vec![g.kv_width() as u64],
        ),
        (
            "QSA value",
            &workspace.value,
            GgmlType::F32,
            vec![g.kv_width() as u64],
        ),
        (
            "QSA key cache",
            &workspace.key_cache,
            GgmlType::F16,
            vec![g.head_dim as u64, g.kv_heads as u64, g.capacity as u64],
        ),
        (
            "QSA value cache",
            &workspace.value_cache,
            GgmlType::F16,
            vec![g.head_dim as u64, g.kv_heads as u64, g.capacity as u64],
        ),
        (
            "QSA attention",
            &workspace.attention,
            GgmlType::F32,
            vec![g.query_width() as u64],
        ),
        (
            "QSA output",
            &workspace.output,
            GgmlType::F32,
            vec![g.hidden_size as u64],
        ),
    ] {
        require_tensor(name, tensor, dtype, &shape, true)?;
    }
    Ok(())
}

fn workspace_tensors(
    workspace: &QwenSparseAttentionMetalWorkspace,
) -> Vec<(&'static str, &MetalTensor)> {
    let tensors = vec![
        ("QSA index query raw", &workspace.index_query_raw),
        ("QSA index query", &workspace.index_query),
        ("QSA index key raw", &workspace.index_key_raw),
        ("QSA pending index keys", &workspace.pending_index_keys),
        (
            "QSA compressed index keys",
            &workspace.compressed_index_keys,
        ),
        ("QSA scores", &workspace.scores),
        ("QSA visible blocks", &workspace.visible_blocks),
        ("QSA selected blocks", &workspace.selected_blocks),
        ("QSA selected count", &workspace.selected_count),
        ("QSA selector status", &workspace.selector_status),
        ("QSA token IDs", &workspace.token_ids),
        ("QSA attention logits", &workspace.attention_logits),
        (
            "QSA query/gate projection",
            &workspace.query_gate_projection,
        ),
        ("QSA query", &workspace.query),
        ("QSA raw gate", &workspace.raw_gate),
        ("QSA key raw", &workspace.key_raw),
        ("QSA key", &workspace.key),
        ("QSA value", &workspace.value),
        ("QSA key cache", &workspace.key_cache),
        ("QSA value cache", &workspace.value_cache),
        ("QSA attention", &workspace.attention),
        ("QSA output", &workspace.output),
    ];
    #[cfg(test)]
    let tensors = {
        let mut tensors = tensors;
        if let Some(scratch) = &workspace.split_decode_scratch {
            tensors.push(("QSA split scratch", scratch));
        }
        tensors
    };
    tensors
}

pub(crate) fn preflight(
    ctx: &MetalContext,
    weights: QwenSparseAttentionMetalWeights<'_>,
) -> Result<(), Qwen4ExpQsaError> {
    for dtype in [
        weights.query.dtype,
        weights.key.dtype,
        weights.value.dtype,
        weights.output.dtype,
        weights.index_query.dtype,
        weights.index_key.dtype,
    ] {
        preflight_projection(ctx, dtype)?;
    }
    for kernel in [
        "kernel_qwen4exp_qsa_norm_rope_f32",
        "kernel_qwen4exp_qsa_write_pending_f32",
        "kernel_qwen4exp_qsa_pool_publish_f16",
        "kernel_qwen4exp_qsa_fill_block_ids_i32",
        "kernel_qwen4exp_qsa_expand_ids_i32",
        "kernel_qwen4exp_qsa_qgate_norm_rope_f32",
        "kernel_qwen4exp_qsa_publish_kv_f16",
    ] {
        ctx.pipeline(kernel)?;
    }
    for (name, dynamic_memory) in [
        ("kernel_qwen4exp_qsa_index_scores_4x128_f16", 0),
        ("kernel_qwen4exp_qsa_attention_logits_f16", 0),
        (
            "kernel_qwen4exp_qsa_attention_softmax_value_f16",
            ATTENTION_SCRATCH_BYTES,
        ),
        (
            "kernel_deepseek_v4_select_top_k_radix4_ids_f32",
            SELECTOR_SCRATCH_BYTES,
        ),
    ] {
        let pipeline = ctx.pipeline(name)?;
        validate_cooperative_pipeline(
            name,
            pipeline.threadExecutionWidth(),
            pipeline.maxTotalThreadsPerThreadgroup(),
            pipeline.staticThreadgroupMemoryLength(),
            dynamic_memory,
            ctx.device.maxThreadgroupMemoryLength(),
        )?;
    }
    Ok(())
}

fn validate_cooperative_pipeline(
    name: &str,
    simd_width: usize,
    max_threads: usize,
    static_threadgroup_memory: usize,
    dynamic_threadgroup_memory: usize,
    max_threadgroup_memory: usize,
) -> Result<(), Qwen4ExpQsaError> {
    validate_cooperative_pipeline_threads(
        name,
        simd_width,
        max_threads,
        ATTENTION_THREADS,
        static_threadgroup_memory,
        dynamic_threadgroup_memory,
        max_threadgroup_memory,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_cooperative_pipeline_threads(
    name: &str,
    simd_width: usize,
    max_threads: usize,
    required_threads: usize,
    static_threadgroup_memory: usize,
    dynamic_threadgroup_memory: usize,
    max_threadgroup_memory: usize,
) -> Result<(), Qwen4ExpQsaError> {
    if simd_width != 32 {
        return invalid(format!(
            "QSA pipeline {name} requires SIMD width 32, got {simd_width}"
        ));
    }
    if max_threads < required_threads {
        return invalid(format!(
            "QSA pipeline {name} requires at least {required_threads} threads per threadgroup, got {max_threads}"
        ));
    }
    let required_memory = static_threadgroup_memory
        .checked_add(dynamic_threadgroup_memory)
        .ok_or_else(|| Qwen4ExpQsaError::Invalid("QSA threadgroup memory overflow".into()))?;
    if max_threadgroup_memory < required_memory {
        return invalid(format!(
            "QSA pipeline {name} requires {required_memory} threadgroup bytes ({static_threadgroup_memory} static plus {dynamic_threadgroup_memory} dynamic), device exposes {max_threadgroup_memory}"
        ));
    }
    Ok(())
}

fn preflight_projection(ctx: &MetalContext, dtype: GgmlType) -> Result<(), Qwen4ExpQsaError> {
    if !crate::qwen4exp_metal::preflight_projection_pipelines(ctx, dtype, false, true)? {
        return invalid(format!("unsupported QSA projection dtype {dtype:?}"));
    }
    Ok(())
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeArgs {
    head_count: u32,
    head_dim: u32,
    rotary_dim: u32,
    position: u32,
    theta: f32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedQueryArgs {
    start_position: u32,
    query_count: u32,
    head_count: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    eps: f32,
}

#[allow(clippy::too_many_arguments)]
fn encode_norm_rope(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weight: &MetalTensor,
    output: &MetalTensor,
    head_count: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
    theta: f32,
    eps: f32,
) -> Result<(), MetalError> {
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_norm_rope_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &RopeArgs {
            head_count: head_count as u32,
            head_dim: head_dim as u32,
            rotary_dim: rotary_dim as u32,
            position: position as u32,
            theta,
            eps,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, output);
    dispatch_1d(enc, &pso, head_count * head_dim);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_norm_rope_packed(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weight: &MetalTensor,
    output: &MetalTensor,
    start_position: usize,
    query_count: usize,
    head_count: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    eps: f32,
) -> Result<(), MetalError> {
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_norm_rope_packed_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &PackedQueryArgs {
            start_position: start_position as u32,
            query_count: query_count as u32,
            head_count: head_count as u32,
            head_dim: head_dim as u32,
            rotary_dim: rotary_dim as u32,
            theta,
            eps,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, output);
    dispatch_1d(enc, &pso, query_count * head_count * head_dim);
    Ok(())
}

fn encode_write_pending(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    slot: usize,
) -> Result<(), MetalError> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_dim: u32,
        slot: u32,
    }
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_write_pending_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_dim: workspace.geometry.index_head_dim as u32,
            slot: slot as u32,
        },
    );
    enc.set_tensor(1, &workspace.index_key_raw);
    enc.set_tensor(2, &workspace.pending_index_keys);
    dispatch_1d(enc, &pso, workspace.geometry.index_head_dim);
    Ok(())
}

fn encode_pool_publish(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    weight: &MetalTensor,
    block: usize,
) -> Result<(), MetalError> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ratio: u32,
        head_dim: u32,
        rotary_dim: u32,
        block: u32,
        theta: f32,
        eps: f32,
    }
    let g = workspace.geometry;
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_pool_publish_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            ratio: g.ratio as u32,
            head_dim: g.index_head_dim as u32,
            rotary_dim: g.rotary_dim as u32,
            block: block as u32,
            theta: g.theta,
            eps: g.eps,
        },
    );
    enc.set_tensor(1, &workspace.pending_index_keys);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, &workspace.compressed_index_keys);
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

fn encode_index_scores(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    visible_blocks: usize,
) -> Result<(), MetalError> {
    encode_index_scores_tensors(
        ctx,
        enc,
        &workspace.index_query,
        &workspace.compressed_index_keys,
        &workspace.scores,
        visible_blocks,
    )
}

fn encode_index_scores_tensors(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    compressed_keys: &MetalTensor,
    scores: &MetalTensor,
    visible_blocks: usize,
) -> Result<(), MetalError> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        visible_blocks: u32,
    }
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_index_scores_4x128_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            visible_blocks: visible_blocks as u32,
        },
    );
    enc.set_tensor(1, query);
    enc.set_tensor(2, compressed_keys);
    enc.set_tensor(3, scores);
    enc.dispatch(
        MTLSize {
            width: visible_blocks.div_ceil(LOGITS_SIMDGROUPS_PER_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: ATTENTION_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedScoreArgs {
    start_position: u32,
    query_count: u32,
    ratio: u32,
    block_capacity: u32,
}

#[allow(dead_code)]
fn encode_index_scores_packed(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    compressed_keys: &MetalTensor,
    scores: &MetalTensor,
    visible_counts: &MetalTensor,
    start_position: usize,
    query_count: usize,
    ratio: usize,
    block_capacity: usize,
) -> Result<(), MetalError> {
    let maximum_visible = (start_position + query_count) / ratio;
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_index_scores_packed_4x128_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &PackedScoreArgs {
            start_position: start_position as u32,
            query_count: query_count as u32,
            ratio: ratio as u32,
            block_capacity: block_capacity as u32,
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, compressed_keys);
    enc.set_tensor(3, scores);
    enc.set_tensor(4, visible_counts);
    enc.dispatch(
        MTLSize {
            width: maximum_visible.div_ceil(LOGITS_SIMDGROUPS_PER_TG),
            height: query_count,
            depth: 1,
        },
        MTLSize {
            width: ATTENTION_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct IdArgs {
    block_budget: u32,
    visible_blocks: u32,
    ratio: u32,
    sequence_length: u32,
    output_width: u32,
}

fn id_args(g: QwenSparseAttentionMetalGeometry, visible: usize, length: usize) -> IdArgs {
    IdArgs {
        block_budget: g.block_budget() as u32,
        visible_blocks: visible as u32,
        ratio: g.ratio as u32,
        sequence_length: length as u32,
        output_width: g.output_width() as u32,
    }
}

fn encode_fill_blocks(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    visible: usize,
    sequence_length: usize,
) -> Result<(), MetalError> {
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_fill_block_ids_i32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &id_args(workspace.geometry, visible, sequence_length));
    enc.set_tensor(1, &workspace.selected_blocks);
    enc.set_tensor(2, &workspace.selected_count);
    enc.set_tensor(3, &workspace.selector_status);
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

fn encode_select_blocks(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    _visible: usize,
) -> Result<(), MetalError> {
    let g = workspace.geometry;
    encode_select_blocks_tensors(
        ctx,
        enc,
        &workspace.scores,
        &workspace.visible_blocks,
        &workspace.selected_blocks,
        &workspace.selected_count,
        &workspace.selector_status,
        g.block_capacity(),
        g.block_budget(),
        1,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_select_blocks_tensors(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    scores: &MetalTensor,
    visible_blocks: &MetalTensor,
    selected_blocks: &MetalTensor,
    selected_count: &MetalTensor,
    selector_status: &MetalTensor,
    block_capacity: usize,
    block_budget: usize,
    query_count: usize,
) -> Result<(), MetalError> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        query_count: u32,
        emit_ranked: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_select_top_k_radix4_ids_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_capacity: block_capacity as u32,
            top_k: block_budget as u32,
            query_count: query_count as u32,
            emit_ranked: 0,
        },
    );
    enc.set_tensor(1, scores);
    enc.set_tensor(2, visible_blocks);
    enc.set_tensor(3, selected_blocks);
    enc.set_tensor(4, selected_count);
    enc.set_tensor(5, selector_status);
    enc.set_threadgroup_memory(0, ATTENTION_THREADS * size_of::<u32>());
    enc.set_threadgroup_memory(1, ATTENTION_THREADS * size_of::<u32>());
    enc.dispatch(
        MTLSize {
            width: query_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: ATTENTION_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_expand_ids(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    visible: usize,
    length: usize,
) -> Result<(), MetalError> {
    encode_expand_ids_tensors(
        ctx,
        enc,
        &workspace.selected_blocks,
        &workspace.token_ids,
        workspace.geometry,
        visible,
        length,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_expand_ids_tensors(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    selected_blocks: &MetalTensor,
    token_ids: &MetalTensor,
    geometry: QwenSparseAttentionMetalGeometry,
    visible: usize,
    length: usize,
) -> Result<(), MetalError> {
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_expand_ids_i32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &id_args(geometry, visible, length));
    enc.set_tensor(1, selected_blocks);
    enc.set_tensor(2, token_ids);
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedIdsArgs {
    start_position: u32,
    query_count: u32,
    block_budget: u32,
    ratio: u32,
    output_width: u32,
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_expand_ids_packed(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    visible_blocks: &MetalTensor,
    selected_blocks: &MetalTensor,
    selected_count: &MetalTensor,
    selector_status: &MetalTensor,
    token_ids: &MetalTensor,
    start_position: usize,
    query_count: usize,
    block_budget: usize,
    ratio: usize,
    output_width: usize,
) -> Result<(), MetalError> {
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_expand_ids_packed_i32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &PackedIdsArgs {
            start_position: start_position as u32,
            query_count: query_count as u32,
            block_budget: block_budget as u32,
            ratio: ratio as u32,
            output_width: output_width as u32,
        },
    );
    enc.set_tensor(1, visible_blocks);
    enc.set_tensor(2, selected_blocks);
    enc.set_tensor(3, selected_count);
    enc.set_tensor(4, selector_status);
    enc.set_tensor(5, token_ids);
    dispatch_1d(enc, &pso, query_count * output_width);
    Ok(())
}

fn encode_qgate_norm_rope(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    weight: &MetalTensor,
    position: usize,
) -> Result<(), MetalError> {
    let g = workspace.geometry;
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_qgate_norm_rope_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &RopeArgs {
            head_count: g.query_heads as u32,
            head_dim: g.head_dim as u32,
            rotary_dim: g.rotary_dim as u32,
            position: position as u32,
            theta: g.theta,
            eps: g.eps,
        },
    );
    enc.set_tensor(1, &workspace.query_gate_projection);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, &workspace.query);
    enc.set_tensor(4, &workspace.raw_gate);
    dispatch_1d(enc, &pso, g.query_width());
    Ok(())
}

fn encode_publish_kv(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    position: usize,
) -> Result<(), MetalError> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        width: u32,
        position: u32,
    }
    let g = workspace.geometry;
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_publish_kv_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            width: g.kv_width() as u32,
            position: position as u32,
        },
    );
    enc.set_tensor(1, &workspace.key);
    enc.set_tensor(2, &workspace.value);
    enc.set_tensor(3, &workspace.key_cache);
    enc.set_tensor(4, &workspace.value_cache);
    dispatch_1d(enc, &pso, g.kv_width());
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttentionArgs {
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    id_count: u32,
    row_stride: u32,
    cache_capacity: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedAttentionArgs {
    start_position: u32,
    query_count: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_budget: u32,
    ratio: u32,
    row_stride: u32,
    cache_capacity: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SelectedAuditArgs {
    query_count: u32,
    band_ordinal: u32,
    expected_selected_count: i32,
    count_mismatch_status: i32,
    order_mismatch_status: i32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SelectedResetArgs {
    query_count: u32,
}

fn attention_args(g: QwenSparseAttentionMetalGeometry, id_count: usize) -> AttentionArgs {
    AttentionArgs {
        query_heads: g.query_heads as u32,
        kv_heads: g.kv_heads as u32,
        head_dim: g.head_dim as u32,
        id_count: id_count as u32,
        row_stride: g.output_width() as u32,
        cache_capacity: g.capacity as u32,
        scale: 1.0 / (g.head_dim as f32).sqrt(),
    }
}

fn packed_attention_args(
    g: QwenSparseAttentionMetalGeometry,
    start_position: usize,
    query_count: usize,
) -> PackedAttentionArgs {
    PackedAttentionArgs {
        start_position: start_position as u32,
        query_count: query_count as u32,
        query_heads: g.query_heads as u32,
        kv_heads: g.kv_heads as u32,
        head_dim: g.head_dim as u32,
        block_budget: g.block_budget() as u32,
        ratio: g.ratio as u32,
        row_stride: g.output_width() as u32,
        cache_capacity: g.capacity as u32,
        scale: 1.0 / (g.head_dim as f32).sqrt(),
    }
}

fn encode_attention_logits(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    id_count: usize,
) -> Result<(), MetalError> {
    let g = workspace.geometry;
    encode_attention_logits_tensors(
        ctx,
        enc,
        &workspace.query,
        &workspace.key_cache,
        &workspace.token_ids,
        &workspace.attention_logits,
        g,
        id_count,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_attention_logits_tensors(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key_cache: &MetalTensor,
    token_ids: &MetalTensor,
    logits: &MetalTensor,
    geometry: QwenSparseAttentionMetalGeometry,
    id_count: usize,
) -> Result<(), MetalError> {
    let g = geometry;
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_attention_logits_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &attention_args(g, id_count));
    enc.set_tensor(1, query);
    enc.set_tensor(2, key_cache);
    enc.set_tensor(3, token_ids);
    enc.set_tensor(4, logits);
    let pairs = id_count * g.query_heads;
    enc.dispatch(
        MTLSize {
            width: pairs.div_ceil(LOGITS_SIMDGROUPS_PER_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: ATTENTION_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn encode_attention_softmax_value(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    id_count: usize,
) -> Result<(), MetalError> {
    let g = workspace.geometry;
    encode_attention_softmax_value_tensors(
        ctx,
        enc,
        &workspace.raw_gate,
        &workspace.value_cache,
        &workspace.token_ids,
        &workspace.attention_logits,
        &workspace.attention,
        g,
        id_count,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_attention_softmax_value_tensors(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    raw_gate: &MetalTensor,
    value_cache: &MetalTensor,
    token_ids: &MetalTensor,
    logits: &MetalTensor,
    attention: &MetalTensor,
    geometry: QwenSparseAttentionMetalGeometry,
    id_count: usize,
) -> Result<(), MetalError> {
    let g = geometry;
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_attention_softmax_value_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &attention_args(g, id_count));
    enc.set_tensor(1, raw_gate);
    enc.set_tensor(2, value_cache);
    enc.set_tensor(3, token_ids);
    enc.set_tensor(4, logits);
    enc.set_tensor(5, attention);
    enc.set_threadgroup_memory(0, ATTENTION_SCRATCH_BYTES);
    enc.dispatch(
        MTLSize {
            width: g.query_heads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: ATTENTION_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_attention_logits_packed(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key_cache: &MetalTensor,
    token_ids: &MetalTensor,
    selected_count: &MetalTensor,
    selector_status: &MetalTensor,
    logits: &MetalTensor,
    geometry: QwenSparseAttentionMetalGeometry,
    start_position: usize,
    query_count: usize,
) -> Result<(), MetalError> {
    if configured_qwen4exp_qsa_gqa4_logits_enabled() {
        return encode_attention_logits_packed_gqa4(
            ctx,
            enc,
            query,
            key_cache,
            token_ids,
            selected_count,
            selector_status,
            logits,
            geometry,
            start_position,
            query_count,
        );
    }
    let g = geometry;
    let heads_per_kv = g.query_heads / g.kv_heads;
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_attention_logits_packed_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &packed_attention_args(g, start_position, query_count));
    enc.set_tensor(1, query);
    enc.set_tensor(2, key_cache);
    enc.set_tensor(3, token_ids);
    enc.set_tensor(4, selected_count);
    enc.set_tensor(5, selector_status);
    enc.set_tensor(6, logits);
    enc.set_threadgroup_memory(0, g.head_dim * size_of::<u16>());
    enc.dispatch(
        MTLSize {
            width: heads_per_kv / PACKED_ATTENTION_HEADS_PER_TG,
            height: g.kv_heads,
            depth: query_count,
        },
        MTLSize {
            width: PACKED_ATTENTION_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_attention_logits_packed_gqa4(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key_cache: &MetalTensor,
    token_ids: &MetalTensor,
    selected_count: &MetalTensor,
    selector_status: &MetalTensor,
    logits: &MetalTensor,
    geometry: QwenSparseAttentionMetalGeometry,
    start_position: usize,
    query_count: usize,
) -> Result<(), MetalError> {
    const SLOTS_PER_THREADGROUP: usize = 32;
    let g = geometry;
    let heads_per_kv = g.query_heads / g.kv_heads;
    let head_groups = heads_per_kv / PACKED_ATTENTION_HEADS_PER_TG;
    let slot_tiles = g.output_width().div_ceil(SLOTS_PER_THREADGROUP);
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_attention_logits_packed_gqa4_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &packed_attention_args(g, start_position, query_count));
    enc.set_tensor(1, query);
    enc.set_tensor(2, key_cache);
    enc.set_tensor(3, token_ids);
    enc.set_tensor(4, selected_count);
    enc.set_tensor(5, selector_status);
    enc.set_tensor(6, logits);
    enc.dispatch(
        MTLSize {
            width: head_groups * slot_tiles,
            height: g.kv_heads,
            depth: query_count,
        },
        MTLSize {
            width: PACKED_ATTENTION_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_attention_softmax_value_packed(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query_gate_projection: &MetalTensor,
    value_cache: &MetalTensor,
    token_ids: &MetalTensor,
    selected_count: &MetalTensor,
    selector_status: &MetalTensor,
    logits: &MetalTensor,
    attention: &MetalTensor,
    geometry: QwenSparseAttentionMetalGeometry,
    start_position: usize,
    query_count: usize,
) -> Result<(), MetalError> {
    if configured_qwen4exp_qsa_gqa4_value_enabled() {
        return encode_attention_softmax_value_packed_gqa4(
            ctx,
            enc,
            query_gate_projection,
            value_cache,
            token_ids,
            selected_count,
            selector_status,
            logits,
            attention,
            geometry,
            start_position,
            query_count,
        );
    }
    let g = geometry;
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_attention_softmax_value_packed_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &packed_attention_args(g, start_position, query_count));
    enc.set_tensor(1, query_gate_projection);
    enc.set_tensor(2, value_cache);
    enc.set_tensor(3, token_ids);
    enc.set_tensor(4, selected_count);
    enc.set_tensor(5, selector_status);
    enc.set_tensor(6, logits);
    enc.set_tensor(7, attention);
    enc.set_threadgroup_memory(0, ATTENTION_SCRATCH_BYTES);
    enc.dispatch(
        MTLSize {
            width: g.query_heads,
            height: query_count,
            depth: 1,
        },
        MTLSize {
            width: ATTENTION_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_attention_softmax_value_packed_gqa4(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query_gate_projection: &MetalTensor,
    value_cache: &MetalTensor,
    token_ids: &MetalTensor,
    selected_count: &MetalTensor,
    selector_status: &MetalTensor,
    logits: &MetalTensor,
    attention: &MetalTensor,
    geometry: QwenSparseAttentionMetalGeometry,
    start_position: usize,
    query_count: usize,
) -> Result<(), MetalError> {
    const HEADS_PER_GROUP: usize = 4;
    const SCRATCH_FLOATS: usize = HEADS_PER_GROUP * ATTENTION_SCRATCH_FLOATS;
    let g = geometry;
    let heads_per_kv = g.query_heads / g.kv_heads;
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_attention_softmax_value_packed_gqa4_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &packed_attention_args(g, start_position, query_count));
    enc.set_tensor(1, query_gate_projection);
    enc.set_tensor(2, value_cache);
    enc.set_tensor(3, token_ids);
    enc.set_tensor(4, selected_count);
    enc.set_tensor(5, selector_status);
    enc.set_tensor(6, logits);
    enc.set_tensor(7, attention);
    enc.set_threadgroup_memory(0, SCRATCH_FLOATS * size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: heads_per_kv / HEADS_PER_GROUP,
            height: g.kv_heads,
            depth: query_count,
        },
        MTLSize {
            width: ATTENTION_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_selected_audit(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    selected_count: &MetalTensor,
    selector_status: &MetalTensor,
    workspace_selected_count: &MetalTensor,
    workspace_selector_status: &MetalTensor,
    audited_bands: &MetalTensor,
    query_count: usize,
    expected_selected_count: usize,
    band_ordinal: usize,
) -> Result<(), MetalError> {
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_audit_selected_i32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &SelectedAuditArgs {
            query_count: query_count as u32,
            band_ordinal: band_ordinal as u32,
            expected_selected_count: expected_selected_count as i32,
            count_mismatch_status: SELECTED_COUNT_MISMATCH_STATUS,
            order_mismatch_status: SELECTED_AUDIT_ORDER_MISMATCH_STATUS,
        },
    );
    enc.set_tensor(1, selected_count);
    enc.set_tensor(2, selector_status);
    enc.set_tensor(3, workspace_selector_status);
    enc.set_tensor(4, workspace_selected_count);
    enc.set_tensor(5, audited_bands);
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

#[allow(dead_code)]
fn encode_selected_control_reset(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    visible_blocks: &MetalTensor,
    selected_count: &MetalTensor,
    selector_status: &MetalTensor,
    query_count: usize,
) -> Result<(), MetalError> {
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_reset_selected_controls_i32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &SelectedResetArgs {
            query_count: query_count as u32,
        },
    );
    enc.set_tensor(1, visible_blocks);
    enc.set_tensor(2, selected_count);
    enc.set_tensor(3, selector_status);
    enc.dispatch(
        MTLSize {
            width: query_count.div_ceil(32),
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

fn dispatch_1d(
    enc: &KernelEncoder,
    pso: &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    count: usize,
) {
    let threads = pso.maxTotalThreadsPerThreadgroup().min(256);
    enc.dispatch(
        MTLSize {
            width: count.div_ceil(threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

fn validate_encoder(ctx: &MetalContext, enc: &KernelEncoder) -> Result<(), Qwen4ExpQsaError> {
    let command = enc.parent_command_buffer();
    let actual = command.device().registryID();
    let expected = ctx.device.registryID();
    if actual != expected {
        return invalid(format!(
            "encoder belongs to Metal device registry {actual}, context is {expected}"
        ));
    }
    if enc.is_concurrent() {
        return invalid("dependent QSA dispatches require a serial encoder");
    }
    let status = command.status();
    if status != MTLCommandBufferStatus::NotEnqueued {
        return invalid(format!(
            "QSA encoding requires a NotEnqueued command buffer, got {status:?}"
        ));
    }
    Ok(())
}

fn reserve_command(
    workspace: &mut QwenSparseAttentionMetalWorkspace,
    enc: &KernelEncoder,
    pending_length: usize,
    pending_selected_bands: usize,
) -> Result<(), Qwen4ExpQsaError> {
    workspace.require_idle()?;
    workspace.active_command = Some(enc.parent_command_buffer());
    workspace.pending_length = Some(pending_length);
    workspace.pending_selected_bands = Some(pending_selected_bands);
    Ok(())
}

#[derive(Clone, Copy)]
enum ProjectionRole {
    Main,
    Index,
}

fn require_projection(
    name: &str,
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
    role: ProjectionRole,
) -> Result<(), Qwen4ExpQsaError> {
    if tensor.shape != [n_in as u64, n_out as u64] {
        return invalid(format!(
            "{name} shape {:?} does not match [{n_in}, {n_out}]",
            tensor.shape
        ));
    }
    match (role, tensor.dtype) {
        (_, GgmlType::F32) | (ProjectionRole::Index, GgmlType::BF16) => {}
        (ProjectionRole::Main, GgmlType::Q8_0) if n_in.is_multiple_of(32) => {}
        _ => {
            return invalid(format!("{name} has unsupported dtype {:?}", tensor.dtype));
        }
    }
    require_range(name, tensor)
}

fn require_tensor(
    name: &str,
    tensor: &MetalTensor,
    dtype: GgmlType,
    shape: &[u64],
    writable: bool,
) -> Result<(), Qwen4ExpQsaError> {
    if tensor.dtype != dtype || tensor.shape != shape {
        return invalid(format!(
            "{name} must be {dtype:?} with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    if writable && !tensor.is_writable() {
        return invalid(format!("{name} must be writable"));
    }
    require_range(name, tensor)
}

fn storage_bytes(tensor: &MetalTensor) -> Result<u64, Qwen4ExpQsaError> {
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .ok_or_else(|| Qwen4ExpQsaError::Invalid("tensor element count overflow".into()))?;
    let (block, bytes) = ggml_type_layout(tensor.dtype).ok_or_else(|| {
        Qwen4ExpQsaError::Invalid(format!("unsupported tensor dtype {:?}", tensor.dtype))
    })?;
    if block == 0 || !elements.is_multiple_of(block) {
        return invalid(format!(
            "tensor shape {:?} is not block-aligned for {:?}",
            tensor.shape, tensor.dtype
        ));
    }
    (elements / block)
        .checked_mul(bytes)
        .ok_or_else(|| Qwen4ExpQsaError::Invalid("tensor byte count overflow".into()))
}

fn require_range(name: &str, tensor: &MetalTensor) -> Result<(), Qwen4ExpQsaError> {
    let alignment = match tensor.dtype {
        GgmlType::F16 | GgmlType::BF16 | GgmlType::Q8_0 => 2,
        _ => 4,
    };
    if !tensor.offset.is_multiple_of(alignment) {
        return invalid(format!(
            "{name} offset {} is not {alignment}-byte aligned",
            tensor.offset
        ));
    }
    let bytes = storage_bytes(tensor)?;
    let end = tensor
        .offset
        .checked_add(bytes)
        .ok_or_else(|| Qwen4ExpQsaError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range offset={} bytes={bytes} exceeds buffer={}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

fn require_read_only_weights(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpQsaError> {
    for (name, tensor) in tensors {
        if tensor.provenance() == MetalTensorProvenance::OwnedWritable {
            return invalid(format!("{name} must have read-only weight provenance"));
        }
    }
    Ok(())
}

fn require_same_device(
    ctx: &MetalContext,
    tensors: &[(&str, &MetalTensor)],
) -> Result<(), Qwen4ExpQsaError> {
    let expected = ctx.device.registryID();
    for (name, tensor) in tensors {
        let actual = tensor.buffer.device().registryID();
        if actual != expected {
            return invalid(format!(
                "{name} belongs to Metal device registry {actual}, expected {expected}"
            ));
        }
    }
    Ok(())
}

fn require_disjoint(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpQsaError> {
    for left in 0..tensors.len() {
        let left_bytes = storage_bytes(tensors[left].1)?;
        for right in left + 1..tensors.len() {
            if Retained::as_ptr(&tensors[left].1.buffer)
                != Retained::as_ptr(&tensors[right].1.buffer)
            {
                continue;
            }
            let right_bytes = storage_bytes(tensors[right].1)?;
            let left_end = tensors[left].1.offset.saturating_add(left_bytes);
            let right_end = tensors[right].1.offset.saturating_add(right_bytes);
            if tensors[left].1.offset < right_end && tensors[right].1.offset < left_end {
                return invalid(format!("{} overlaps {}", tensors[left].0, tensors[right].0));
            }
        }
    }
    Ok(())
}

fn write_i32_scalar(tensor: &MetalTensor, value: i32) -> Result<(), Qwen4ExpQsaError> {
    require_tensor("QSA host scalar", tensor, GgmlType::I32, &[1], true)?;
    unsafe {
        tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<i32>()
            .write(value)
    };
    Ok(())
}

fn fill_i32_tensor(tensor: &MetalTensor, value: i32) -> Result<(), Qwen4ExpQsaError> {
    if tensor.dtype != GgmlType::I32 || !tensor.is_writable() {
        return invalid("QSA host fill requires a writable I32 tensor");
    }
    require_range("QSA host fill", tensor)?;
    let count = usize::try_from(tensor.n_elements())
        .map_err(|_| Qwen4ExpQsaError::Invalid("QSA host fill count exceeds usize".into()))?;
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<i32>();
        std::slice::from_raw_parts_mut(destination, count).fill(value);
    }
    Ok(())
}

fn read_i32_scalar(tensor: &MetalTensor) -> Result<i32, Qwen4ExpQsaError> {
    require_tensor("QSA host scalar", tensor, GgmlType::I32, &[1], true)?;
    Ok(unsafe {
        tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<i32>()
            .read()
    })
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpQsaError> {
    Err(Qwen4ExpQsaError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests;
