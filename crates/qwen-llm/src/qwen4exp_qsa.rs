//! Stateful one-token, text-only Metal Qwen Sparse Attention for
//! Qwen3.8-Flash-Next.
//!
//! Plain NEOX RoPE is valid only because all IMRoPE axes coincide for text.
//! F32 pending index keys with an explicit pooled F16 rounding point, F16
//! compressed index keys, and F16 main K/V caches are this Metal checkpoint's
//! numerical contract. It intentionally differs from BF16 serving backends.
//! Dense packed attention preserves that state contract while qualifying the
//! F16-staged matrix kernels against chronological scalar execution.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, MetalTensorProvenance,
    encode_attn_matrix_kq_f32, encode_attn_matrix_kqv_direct_v_f32, encode_attn_matrix_softmax_f32,
    encode_copy_offset_f32, encode_mat_mat_bf16_f32,
    encode_qk_rms_norm_rope_f32_packed_consecutive, encode_scatter_offset_f32_to_f16_kv,
    encode_sigmoid_mul_gate_strided_f32,
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
const LOGITS_SIMDGROUPS_PER_TG: usize = 8;
const SELECTOR_SCRATCH_BYTES: usize = 2 * ATTENTION_THREADS * size_of::<u32>();
// Staged for the all-layer packed composer in the next checkpoint.
#[allow(dead_code)]
const DENSE_PACKED_QUERY_TILE: usize = 32;

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
        let query_tile = capacity.min(DENSE_PACKED_QUERY_TILE);
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
                &[self.token_budget, self.query_heads, query_tile],
            )?,
            f32_bytes("attention", &[self.query_width(), capacity])?,
            f32_bytes("output", &[self.hidden_size, capacity])?,
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
        let score_elements = geometry
            .token_budget
            .checked_mul(geometry.query_heads)
            .and_then(|elements| elements.checked_mul(query_tile))
            .ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(
                    "dense packed QSA attention-score capacity overflow".into(),
                )
            })?;
        if u32::try_from(score_elements).is_err() {
            return invalid(format!(
                "dense packed QSA attention-score scratch has {score_elements} elements, exceeding u32"
            ));
        }
        score_elements
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| {
                Qwen4ExpQsaError::Invalid(
                    "dense packed QSA attention-score byte count overflow".into(),
                )
            })?;

        let shape = |width: usize| vec![width as u64, capacity as u64];
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
                    geometry.token_budget as u64,
                    geometry.query_heads as u64,
                    query_tile as u64,
                ],
            )?,
            attention: MetalTensor::zeros_f32(ctx, shape(geometry.query_width()))?,
            output: MetalTensor::zeros_f32(ctx, shape(geometry.hidden_size))?,
        })
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
        if rows == 0 || rows > self.query_tile || sequence_length > self.geometry.token_budget {
            return invalid(format!(
                "dense packed QSA score view rows={rows} length={sequence_length} exceeds tile={} budget={}",
                self.query_tile, self.geometry.token_budget
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
            committed_length: 0,
            pending_length: None,
            active_command: None,
            state_poisoned: false,
        })
    }

    pub fn geometry(&self) -> QwenSparseAttentionMetalGeometry {
        self.geometry
    }

    pub fn committed_length(&self) -> usize {
        self.committed_length
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
        self.state_poisoned = false;
        Ok(())
    }

    pub fn release_after(&mut self) -> Result<(), Qwen4ExpQsaError> {
        let Some(command) = self.active_command.clone() else {
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
            ))
        })();
        self.active_command = None;
        let pending = self.pending_length.take();
        let (selector_status, selected_count) = match scalar_result {
            Ok(scalars) => scalars,
            Err(error) => {
                self.state_poisoned = true;
                return Err(error);
            }
        };
        let expected_count = pending
            .map(|length| (length / self.geometry.ratio).min(self.geometry.block_budget()))
            .unwrap_or(0);
        if status == MTLCommandBufferStatus::Completed
            && error.is_none()
            && selector_status == 0
            && selected_count == expected_count as i32
        {
            self.committed_length = pending.ok_or_else(|| {
                Qwen4ExpQsaError::Invalid("completed QSA command had no pending length".into())
            })?;
            Ok(())
        } else {
            self.state_poisoned = true;
            Err(Qwen4ExpQsaError::CommandBuffer(format!(
                "status={status:?}, error={error:?}, selector_status={selector_status}, selected_count={selected_count}, expected_count={expected_count}"
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
        self.state_poisoned = false;
        Ok(())
    }

    fn require_idle(&self) -> Result<(), Qwen4ExpQsaError> {
        if self.active_command.is_some() || self.pending_length.is_some() {
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
    reserve_command(workspace, enc, sequence_length)?;

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
///
/// # Safety
///
/// The caller must retain every tensor and exclusive logical ownership of
/// `workspace` and `scratch` until the command completes successfully or is
/// permanently abandoned. Commands touching this workspace must execute in
/// causal order. Any encode or command failure makes mutable state
/// indeterminate and requires poisoning the enclosing transaction.
#[allow(clippy::too_many_arguments)]
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
        )
    }
}

#[allow(clippy::too_many_arguments)]
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

    validate_dense_packed_contract(
        ctx,
        enc,
        input,
        weights,
        workspace,
        scratch,
        start_position,
        tokens,
    )?;
    preflight_dense_packed(ctx, weights)?;
    let sequence_length = prepare_dense_packed_control_scalars(workspace, start_position, tokens)?;
    reserve_command(workspace, enc, sequence_length)?;

    let encoded = encode_dense_packed_step(
        ctx,
        enc,
        input,
        weights,
        workspace,
        scratch,
        start_position,
        tokens,
        sequence_length,
        layer,
        profile,
    );
    if let Err(error) = encoded {
        workspace.state_poisoned = true;
        return Err(error);
    }
    scratch.views(tokens).map(|views| views.output)
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
fn prepare_dense_packed_control_scalars(
    workspace: &QwenSparseAttentionMetalWorkspace,
    start_position: usize,
    tokens: usize,
) -> Result<usize, Qwen4ExpQsaError> {
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
    let sequence_length = start_position.checked_add(tokens).ok_or_else(|| {
        Qwen4ExpQsaError::Invalid("dense packed QSA sequence length overflow".into())
    })?;
    if tokens <= 1
        || tokens > workspace.geometry.token_budget
        || sequence_length > workspace.geometry.capacity
        || sequence_length > workspace.geometry.token_budget
    {
        return invalid(format!(
            "dense packed QSA range start={start_position} tokens={tokens} end={sequence_length} exceeds capacity={} or dense budget={}",
            workspace.geometry.capacity, workspace.geometry.token_budget
        ));
    }
    let visible_blocks = sequence_length / workspace.geometry.ratio;
    write_i32_scalar(&workspace.visible_blocks, visible_blocks as i32)?;
    write_i32_scalar(&workspace.selector_status, 0)?;
    write_i32_scalar(&workspace.selected_count, 0)?;
    Ok(sequence_length)
}

#[allow(clippy::too_many_arguments)]
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
    let sequence_length = start_position.checked_add(tokens).ok_or_else(|| {
        Qwen4ExpQsaError::Invalid("dense packed QSA sequence length overflow".into())
    })?;
    if sequence_length > g.capacity || sequence_length > g.token_budget {
        return invalid(format!(
            "dense packed QSA end {sequence_length} exceeds cache capacity {} or dense budget {}",
            g.capacity, g.token_budget
        ));
    }
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
        ("query tile", scratch.query_tile),
    ] {
        if u32::try_from(value).is_err() {
            return invalid(format!("dense packed QSA {name} {value} exceeds u32"));
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
                .and_then(|elements| elements.checked_mul(sequence_length)),
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
    scratch.score_view(scratch.query_tile.min(tokens), sequence_length)?;
    Ok(())
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
            g.token_budget as u64,
            g.query_heads as u64,
            scratch.query_tile as u64,
        ],
        true,
    )?;
    Ok(())
}

fn dense_packed_scratch_tensors(
    scratch: &QwenSparseAttentionPackedScratch,
) -> Vec<(&'static str, &MetalTensor)> {
    vec![
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
    ]
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

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn encode_dense_packed_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: QwenSparseAttentionMetalWeights<'_>,
    workspace: &QwenSparseAttentionMetalWorkspace,
    scratch: &QwenSparseAttentionPackedScratch,
    start_position: usize,
    tokens: usize,
    sequence_length: usize,
    layer: u32,
    mut profile: Option<&mut Qwen4ExpPackedProfileRecorder<'_>>,
) -> Result<(), Qwen4ExpQsaError> {
    let g = workspace.geometry;
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
    let mut tile_start = 0;
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::detail("qsa.attention", layer, MixerKind::QwenSparseAttention),
    )?;
    while tile_start < tokens {
        let rows = (tokens - tile_start).min(scratch.query_tile);
        let query = views.query.view_subrange(
            (tile_start * g.query_width()) as u64,
            vec![g.query_width() as u64, rows as u64],
        );
        let attention = views.attention.view_subrange(
            (tile_start * g.query_width()) as u64,
            vec![g.query_width() as u64, rows as u64],
        );
        let scores = scratch.score_view(rows, sequence_length)?;
        let base_position = start_position + tile_start;
        encode_attn_matrix_kq_f32(
            ctx,
            enc,
            &query,
            &workspace.key_cache,
            &scores,
            rows,
            base_position,
            sequence_length,
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
            sequence_length,
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
            sequence_length,
            g.kv_width(),
            g.query_heads,
            g.kv_heads,
            group,
            g.head_dim,
            true,
        )?;
        tile_start += rows;
    }
    end_optional(&mut profile, enc, marker)?;
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::detail("qsa.gate", layer, MixerKind::QwenSparseAttention),
    )?;
    encode_sigmoid_mul_gate_strided_f32(
        ctx,
        enc,
        &views.query_gate_projection,
        &views.attention,
        &views.attention,
        tokens * g.query_heads,
        g.head_dim,
        2 * g.head_dim,
        g.head_dim,
    )?;
    end_optional(&mut profile, enc, marker)?;
    let marker = begin_optional(
        &mut profile,
        enc,
        Qwen4ExpPackedProfileLabel::detail("qsa.output", layer, MixerKind::QwenSparseAttention),
    )?;
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
    if sequence_length.is_multiple_of(g.ratio) {
        encode_pool_publish(
            ctx,
            enc,
            workspace,
            weights.index_key_norm,
            sequence_length / g.ratio - 1,
        )?;
    }

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
    let active_id_count =
        visible_blocks.min(g.block_budget()) * g.ratio + sequence_length % g.ratio;
    encode_attention_logits(ctx, enc, workspace, active_id_count)?;
    encode_attention_softmax_value(ctx, enc, workspace, active_id_count)?;
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
    vec![
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
    ]
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
            ATTENTION_SCRATCH_FLOATS * size_of::<f32>(),
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
    if simd_width != 32 {
        return invalid(format!(
            "QSA pipeline {name} requires SIMD width 32, got {simd_width}"
        ));
    }
    if max_threads < ATTENTION_THREADS {
        return invalid(format!(
            "QSA pipeline {name} requires at least {ATTENTION_THREADS} threads per threadgroup, got {max_threads}"
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
    let kernels: &[&str] = match dtype {
        GgmlType::F32 => &["kernel_mat_vec_f32_f32", "kernel_mat_vec_f32_f32_lcpp_r2"],
        GgmlType::Q8_0 => &["kernel_mat_vec_q8_0_f32", "kernel_mat_vec_q8_0_f32_lcpp"],
        GgmlType::BF16 => &["kernel_mat_vec_bf16_f32"],
        _ => return invalid(format!("unsupported QSA projection dtype {dtype:?}")),
    };
    for kernel in kernels {
        ctx.pipeline(kernel)?;
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
    enc.set_tensor(1, &workspace.index_query);
    enc.set_tensor(2, &workspace.compressed_index_keys);
    enc.set_tensor(3, &workspace.scores);
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
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        query_count: u32,
        emit_ranked: u32,
    }
    let g = workspace.geometry;
    let pso = ctx.pipeline("kernel_deepseek_v4_select_top_k_radix4_ids_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_capacity: g.block_capacity() as u32,
            top_k: g.block_budget() as u32,
            query_count: 1,
            emit_ranked: 0,
        },
    );
    enc.set_tensor(1, &workspace.scores);
    enc.set_tensor(2, &workspace.visible_blocks);
    enc.set_tensor(3, &workspace.selected_blocks);
    enc.set_tensor(4, &workspace.selected_count);
    enc.set_tensor(5, &workspace.selector_status);
    enc.set_threadgroup_memory(0, ATTENTION_THREADS * size_of::<u32>());
    enc.set_threadgroup_memory(1, ATTENTION_THREADS * size_of::<u32>());
    enc.dispatch(
        MTLSize {
            width: 1,
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
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_expand_ids_i32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &id_args(workspace.geometry, visible, length));
    enc.set_tensor(1, &workspace.selected_blocks);
    enc.set_tensor(2, &workspace.token_ids);
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

fn encode_attention_logits(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &QwenSparseAttentionMetalWorkspace,
    id_count: usize,
) -> Result<(), MetalError> {
    let g = workspace.geometry;
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_attention_logits_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &attention_args(g, id_count));
    enc.set_tensor(1, &workspace.query);
    enc.set_tensor(2, &workspace.key_cache);
    enc.set_tensor(3, &workspace.token_ids);
    enc.set_tensor(4, &workspace.attention_logits);
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
    let pso = ctx.pipeline("kernel_qwen4exp_qsa_attention_softmax_value_f16")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &attention_args(g, id_count));
    enc.set_tensor(1, &workspace.raw_gate);
    enc.set_tensor(2, &workspace.value_cache);
    enc.set_tensor(3, &workspace.token_ids);
    enc.set_tensor(4, &workspace.attention_logits);
    enc.set_tensor(5, &workspace.attention);
    enc.set_threadgroup_memory(0, ATTENTION_SCRATCH_FLOATS * size_of::<f32>());
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
) -> Result<(), Qwen4ExpQsaError> {
    workspace.require_idle()?;
    workspace.active_command = Some(enc.parent_command_buffer());
    workspace.pending_length = Some(pending_length);
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
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::qwen4exp_gdn::GatedDeltaNetMetalWeights;
    use crate::qwen4exp_residency::Qwen4ExpMetalWeightPlan;
    use half::f16;
    use objc2_metal::MTLCommandQueue;
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    const QSA_ORACLE_JSON: &str = include_str!("../tests/fixtures/qwen4exp_qsa_text_f16_v1.json");
    const QSA_ORACLE_F32: &[u8] = include_bytes!("../tests/fixtures/qwen4exp_qsa_text_f16_v1.f32");

    struct TestWeights {
        geometry: QwenSparseAttentionMetalGeometry,
        query: MetalTensor,
        key: MetalTensor,
        value: MetalTensor,
        output: MetalTensor,
        query_norm: MetalTensor,
        key_norm: MetalTensor,
        index_query: MetalTensor,
        index_key: MetalTensor,
        index_query_norm: MetalTensor,
        index_key_norm: MetalTensor,
    }

    impl TestWeights {
        fn borrowed(&self) -> QwenSparseAttentionMetalWeights<'_> {
            QwenSparseAttentionMetalWeights {
                geometry: self.geometry,
                query: &self.query,
                key: &self.key,
                value: &self.value,
                output: &self.output,
                query_norm: &self.query_norm,
                key_norm: &self.key_norm,
                index_query: &self.index_query,
                index_key: &self.index_key,
                index_query_norm: &self.index_query_norm,
                index_key_norm: &self.index_key_norm,
            }
        }
    }

    fn context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(ctx) => Some(ctx),
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => None,
            Err(error) => panic!("Metal initialization failed: {error}"),
        }
    }

    fn test_geometry(capacity: usize) -> QwenSparseAttentionMetalGeometry {
        let mut config = Qwen4ExpConfig::flash_next_reference();
        config.context_length = 32;
        config.hidden_size = 16;
        config.attention.query_heads = 4;
        config.attention.kv_heads = 2;
        config.qsa.token_budget = 8;
        config.ple = None;
        config.validate().unwrap();
        QwenSparseAttentionMetalGeometry::from_config(&config, 3, capacity).unwrap()
    }

    fn packed_test_geometry(capacity: usize) -> QwenSparseAttentionMetalGeometry {
        let mut config = Qwen4ExpConfig::flash_next_reference();
        config.context_length = 64;
        config.hidden_size = 16;
        config.qsa.token_budget = 64;
        config.ple = None;
        config.validate().unwrap();
        QwenSparseAttentionMetalGeometry::from_config(&config, 3, capacity).unwrap()
    }

    fn values(count: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|index| {
                let raw = ((index * 37 + seed * 19 + index / 7 * 3 + 5) % 101) as f32 - 50.0;
                raw * scale
            })
            .collect()
    }

    fn real_mat_vec(weight: &[f32], input: &[f32], n_in: usize) -> Vec<f32> {
        assert_eq!(weight.len() % n_in, 0);
        weight
            .chunks_exact(n_in)
            .map(|row| row.iter().zip(input).map(|(&w, &x)| w * x).sum())
            .collect()
    }

    fn rmsnorm_heads_position_zero(
        input: &[f32],
        heads: usize,
        head_dim: usize,
        weight: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        assert_eq!(input.len(), heads * head_dim);
        assert_eq!(weight.len(), head_dim);
        let mut output = vec![0.0; input.len()];
        for head in 0..heads {
            let start = head * head_dim;
            let row = &input[start..start + head_dim];
            let scale = 1.0
                / (row.iter().map(|value| value * value).sum::<f32>() / head_dim as f32 + eps)
                    .sqrt();
            for lane in 0..head_dim {
                output[start + lane] = row[lane] * scale * weight[lane];
            }
        }
        output
    }

    fn weight(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        let mut tensor =
            MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32)
                .unwrap();
        tensor.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        tensor
    }

    fn test_weights(ctx: &MetalContext, geometry: QwenSparseAttentionMetalGeometry) -> TestWeights {
        let g = geometry;
        let query_cpu = values(g.hidden_size * g.query_projection_width(), 1, 0.0013);
        let key_cpu = values(g.hidden_size * g.kv_width(), 2, 0.0011);
        let value_cpu = values(g.hidden_size * g.kv_width(), 3, 0.0015);
        let output_cpu = values(g.query_width() * g.hidden_size, 4, 0.0009);
        let index_query_cpu = values(g.hidden_size * g.index_query_width(), 5, 0.0017);
        let index_key_cpu = values(g.hidden_size * g.index_head_dim, 6, 0.0019);
        let query_norm_cpu = (0..g.head_dim)
            .map(|lane| 0.75 + (lane % 13) as f32 * 0.025)
            .collect::<Vec<_>>();
        let key_norm_cpu = (0..g.head_dim)
            .map(|lane| 0.8 + (lane % 11) as f32 * 0.021)
            .collect::<Vec<_>>();
        let index_query_norm_cpu = (0..g.index_head_dim)
            .map(|lane| 0.7 + (lane % 9) as f32 * 0.031)
            .collect::<Vec<_>>();
        let index_key_norm_cpu = (0..g.index_head_dim)
            .map(|lane| 0.78 + (lane % 7) as f32 * 0.027)
            .collect::<Vec<_>>();
        TestWeights {
            geometry,
            query: weight(
                ctx,
                &query_cpu,
                vec![g.hidden_size as u64, g.query_projection_width() as u64],
            ),
            key: weight(
                ctx,
                &key_cpu,
                vec![g.hidden_size as u64, g.kv_width() as u64],
            ),
            value: weight(
                ctx,
                &value_cpu,
                vec![g.hidden_size as u64, g.kv_width() as u64],
            ),
            output: weight(
                ctx,
                &output_cpu,
                vec![g.query_width() as u64, g.hidden_size as u64],
            ),
            query_norm: weight(ctx, &query_norm_cpu, vec![g.head_dim as u64]),
            key_norm: weight(ctx, &key_norm_cpu, vec![g.head_dim as u64]),
            index_query: weight(
                ctx,
                &index_query_cpu,
                vec![g.hidden_size as u64, g.index_query_width() as u64],
            ),
            index_key: weight(
                ctx,
                &index_key_cpu,
                vec![g.hidden_size as u64, g.index_head_dim as u64],
            ),
            index_query_norm: weight(ctx, &index_query_norm_cpu, vec![g.index_head_dim as u64]),
            index_key_norm: weight(ctx, &index_key_norm_cpu, vec![g.index_head_dim as u64]),
        }
    }

    fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>();
            std::slice::from_raw_parts(source, tensor.shape.iter().product::<u64>() as usize)
                .to_vec()
        }
    }

    fn read_i32(tensor: &MetalTensor) -> Vec<i32> {
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<i32>();
            std::slice::from_raw_parts(source, tensor.shape.iter().product::<u64>() as usize)
                .to_vec()
        }
    }

    fn read_f16(tensor: &MetalTensor) -> Vec<f32> {
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<u16>();
            std::slice::from_raw_parts(source, tensor.shape.iter().product::<u64>() as usize)
                .iter()
                .map(|&bits| f16::from_bits(bits).to_f32())
                .collect()
        }
    }

    fn write_f32_tensor(tensor: &MetalTensor, values: &[f32]) {
        assert!(tensor.is_writable());
        assert_eq!(tensor.dtype, GgmlType::F32);
        assert_eq!(tensor.n_elements() as usize, values.len());
        let offset = tensor.offset as usize;
        let bytes = std::mem::size_of_val(values);
        let end = offset
            .checked_add(bytes)
            .expect("F32 test write range overflow");
        assert!(end <= tensor.buffer.length());
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(offset)
                    .cast::<f32>(),
                values.len(),
            );
        }
    }

    fn write_i32_tensor(tensor: &MetalTensor, values: &[i32]) {
        assert!(tensor.is_writable());
        assert_eq!(tensor.dtype, GgmlType::I32);
        assert_eq!(tensor.n_elements() as usize, values.len());
        let offset = tensor.offset as usize;
        let bytes = std::mem::size_of_val(values);
        let end = offset
            .checked_add(bytes)
            .expect("I32 test write range overflow");
        assert!(end <= tensor.buffer.length());
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(offset)
                    .cast::<i32>(),
                values.len(),
            );
        }
    }

    fn write_f16_prefix(tensor: &MetalTensor, values: &[f32]) {
        assert!(tensor.is_writable());
        assert_eq!(tensor.dtype, GgmlType::F16);
        assert!(values.len() <= tensor.n_elements() as usize);
        let offset = tensor.offset as usize;
        let bytes = values
            .len()
            .checked_mul(size_of::<u16>())
            .expect("F16 test write byte count overflow");
        let end = offset
            .checked_add(bytes)
            .expect("F16 test write range overflow");
        assert!(end <= tensor.buffer.length());
        unsafe {
            let destination = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset)
                .cast::<u16>();
            for (index, &value) in values.iter().enumerate() {
                destination.add(index).write(f16::from_f32(value).to_bits());
            }
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = atol + rtol * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: expected {expected}, got {actual}, tolerance={tolerance}"
            );
        }
    }

    fn assert_similarity(
        label: &str,
        actual: &[f32],
        expected: &[f32],
        maximum_relative_rms: f64,
        minimum_cosine: f64,
        maximum_absolute: f32,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        assert!(actual.iter().all(|value| value.is_finite()), "{label}");
        let dot = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| *actual as f64 * *expected as f64)
            .sum::<f64>();
        let actual_square = actual
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>();
        let expected_square = expected
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>();
        let difference_square = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (*actual as f64 - *expected as f64).powi(2))
            .sum::<f64>();
        let relative_rms = (difference_square / expected_square.max(1e-30)).sqrt();
        let cosine = dot / (actual_square * expected_square).sqrt().max(1e-30);
        let observed_max = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "[{label}] relative_rms={relative_rms:.3e} cosine={cosine:.9} max_abs={observed_max:.3e}"
        );
        assert!(
            relative_rms <= maximum_relative_rms,
            "{label} relative_rms={relative_rms}"
        );
        assert!(cosine >= minimum_cosine, "{label} cosine={cosine}");
        assert!(
            observed_max <= maximum_absolute,
            "{label} max_abs={observed_max}"
        );
    }

    #[derive(Clone)]
    struct DenseQsaStateSnapshot {
        committed_length: usize,
        pending_index_keys: Vec<f32>,
        compressed_index_keys: Vec<f32>,
        key_cache: Vec<f32>,
        value_cache: Vec<f32>,
        selected_count: i32,
    }

    struct DenseQsaSerialTrace {
        outputs: Vec<f32>,
        states: Vec<DenseQsaStateSnapshot>,
    }

    fn serial_dense_trace(
        ctx: &MetalContext,
        weights: &TestWeights,
        inputs: &[f32],
        tokens: usize,
    ) -> DenseQsaSerialTrace {
        let g = weights.geometry;
        assert_eq!(inputs.len(), g.hidden_size * tokens);
        let mut workspace = QwenSparseAttentionMetalWorkspace::new(ctx, g).unwrap();
        let mut outputs = Vec::with_capacity(g.hidden_size * tokens);
        let mut states = Vec::with_capacity(tokens);
        for token in 0..tokens {
            let start = token * g.hidden_size;
            outputs.extend(encode_one(
                ctx,
                weights,
                &mut workspace,
                &inputs[start..start + g.hidden_size],
            ));
            let sequence_length = token + 1;
            let completed_index_elements = sequence_length / g.ratio * g.index_head_dim;
            let cache_elements = sequence_length * g.kv_width();
            states.push(DenseQsaStateSnapshot {
                committed_length: sequence_length,
                pending_index_keys: read_f32(&workspace.pending_index_keys),
                compressed_index_keys: read_f16(&workspace.compressed_index_keys)
                    [..completed_index_elements]
                    .to_vec(),
                key_cache: read_f16(&workspace.key_cache)[..cache_elements].to_vec(),
                value_cache: read_f16(&workspace.value_cache)[..cache_elements].to_vec(),
                selected_count: read_i32_scalar(&workspace.selected_count).unwrap(),
            });
        }
        DenseQsaSerialTrace { outputs, states }
    }

    fn encode_dense_packed_chunk(
        ctx: &MetalContext,
        weights: &TestWeights,
        workspace: &mut QwenSparseAttentionMetalWorkspace,
        scratch: &QwenSparseAttentionPackedScratch,
        inputs: &[f32],
        start_position: usize,
        tokens: usize,
    ) -> (Vec<f32>, Vec<crate::metal::DispatchCensusRow>) {
        let g = weights.geometry;
        assert_eq!(inputs.len(), g.hidden_size * tokens);
        let input = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(inputs),
            vec![g.hidden_size as u64, tokens as u64],
            GgmlType::F32,
        )
        .unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        let output = unsafe {
            encode_qwen_sparse_attention_text_dense_packed_motor(
                ctx,
                &encoder,
                &input,
                weights.borrowed(),
                workspace,
                scratch,
                start_position,
                tokens,
            )
        }
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        (read_f32(&output), census)
    }

    fn assert_dense_state_matches(
        label: &str,
        workspace: &QwenSparseAttentionMetalWorkspace,
        expected: &DenseQsaStateSnapshot,
    ) {
        let g = workspace.geometry;
        assert_eq!(workspace.committed_length(), expected.committed_length);
        assert_eq!(
            read_i32_scalar(&workspace.selected_count).unwrap(),
            expected.selected_count
        );
        assert_close(
            &read_f32(&workspace.pending_index_keys),
            &expected.pending_index_keys,
            1e-5,
            1e-4,
        );
        let completed_index_elements = expected.compressed_index_keys.len();
        assert_close(
            &read_f16(&workspace.compressed_index_keys)[..completed_index_elements],
            &expected.compressed_index_keys,
            5e-4,
            0.0,
        );
        let cache_elements = expected.committed_length * g.kv_width();
        assert_eq!(
            cache_elements,
            expected.key_cache.len(),
            "{label} key cache"
        );
        assert_close(
            &read_f16(&workspace.key_cache)[..cache_elements],
            &expected.key_cache,
            1e-3,
            0.0,
        );
        assert_close(
            &read_f16(&workspace.value_cache)[..cache_elements],
            &expected.value_cache,
            1e-3,
            0.0,
        );
    }

    fn oracle_values() -> Vec<f32> {
        assert!(QSA_ORACLE_F32.len().is_multiple_of(4));
        QSA_ORACLE_F32
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn oracle_section<'a>(fixture: &Value, values: &'a [f32], name: &str) -> &'a [f32] {
        let section = &fixture["binary"]["sections"][name];
        let offset = section["offset_f32"].as_u64().unwrap() as usize;
        let count = section["count_f32"].as_u64().unwrap() as usize;
        let shape_count = section["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|dimension| dimension.as_u64().unwrap() as usize)
            .product::<usize>();
        assert_eq!(count, shape_count, "bad oracle section shape for {name}");
        &values[offset..offset + count]
    }

    fn formula_from_json(formula: &Value, rows: usize, columns: usize) -> Vec<f32> {
        let multipliers = formula["multipliers"].as_array().unwrap();
        let m0 = multipliers[0].as_i64().unwrap();
        let m1 = multipliers[1].as_i64().unwrap();
        let add = formula["add"].as_i64().unwrap();
        let modulus = formula["modulus"].as_i64().unwrap();
        let center = formula["center"].as_i64().unwrap();
        let scale = formula["scale"].as_f64().unwrap() as f32;
        let mut output = Vec::with_capacity(rows * columns);
        for row in 0..rows {
            for column in 0..columns {
                let raw = (row as i64 * m0 + column as i64 * m1 + add).rem_euclid(modulus) - center;
                output.push(raw as f32 * scale);
            }
        }
        output
    }

    fn norm_from_json(recipe: &Value, width: usize) -> Vec<f32> {
        let base = recipe["base"].as_f64().unwrap() as f32;
        let step = recipe["step"].as_f64().unwrap() as f32;
        let modulus = recipe["modulus"].as_u64().unwrap() as usize;
        (0..width)
            .map(|lane| base + (lane % modulus) as f32 * step)
            .collect()
    }

    fn oracle_weights(
        ctx: &MetalContext,
        geometry: QwenSparseAttentionMetalGeometry,
        fixture: &Value,
    ) -> TestWeights {
        let g = geometry;
        let formulas = &fixture["recipe"]["weights_and_inputs"];
        let norms = &fixture["recipe"]["norms"];
        let query = formula_from_json(
            &formulas["query_gate"],
            g.query_projection_width(),
            g.hidden_size,
        );
        let key = formula_from_json(&formulas["key"], g.kv_width(), g.hidden_size);
        let value = formula_from_json(&formulas["value"], g.kv_width(), g.hidden_size);
        let output = formula_from_json(&formulas["output"], g.hidden_size, g.query_width());
        let index_query = formula_from_json(
            &formulas["index_query"],
            g.index_query_width(),
            g.hidden_size,
        );
        let index_key = formula_from_json(&formulas["index_key"], g.index_head_dim, g.hidden_size);
        let query_norm = norm_from_json(&norms["query"], g.head_dim);
        let key_norm = norm_from_json(&norms["key"], g.head_dim);
        let index_query_norm = norm_from_json(&norms["index_query"], g.index_head_dim);
        let index_key_norm = norm_from_json(&norms["index_key"], g.index_head_dim);
        TestWeights {
            geometry,
            query: weight(
                ctx,
                &query,
                vec![g.hidden_size as u64, g.query_projection_width() as u64],
            ),
            key: weight(ctx, &key, vec![g.hidden_size as u64, g.kv_width() as u64]),
            value: weight(ctx, &value, vec![g.hidden_size as u64, g.kv_width() as u64]),
            output: weight(
                ctx,
                &output,
                vec![g.query_width() as u64, g.hidden_size as u64],
            ),
            query_norm: weight(ctx, &query_norm, vec![g.head_dim as u64]),
            key_norm: weight(ctx, &key_norm, vec![g.head_dim as u64]),
            index_query: weight(
                ctx,
                &index_query,
                vec![g.hidden_size as u64, g.index_query_width() as u64],
            ),
            index_key: weight(
                ctx,
                &index_key,
                vec![g.hidden_size as u64, g.index_head_dim as u64],
            ),
            index_query_norm: weight(ctx, &index_query_norm, vec![g.index_head_dim as u64]),
            index_key_norm: weight(ctx, &index_key_norm, vec![g.index_head_dim as u64]),
        }
    }

    fn assert_source_identity(
        fixture: &Value,
        name: &str,
        revision: &str,
        tree: &str,
        files: &[(&str, &str)],
    ) {
        let source = &fixture["sources"][name];
        assert_eq!(source["revision"], revision);
        assert_eq!(source["tree"], tree);
        let actual = source["files"].as_array().unwrap();
        assert_eq!(actual.len(), files.len());
        for &(path, digest) in files {
            let entry = actual
                .iter()
                .find(|entry| entry["path"] == path)
                .unwrap_or_else(|| panic!("missing pinned source {path}"));
            assert_eq!(entry["sha256"], digest);
        }
    }

    fn encode_one(
        ctx: &MetalContext,
        weights: &TestWeights,
        workspace: &mut QwenSparseAttentionMetalWorkspace,
        input_values: &[f32],
    ) -> Vec<f32> {
        let input = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(input_values),
            vec![weights.geometry.hidden_size as u64],
            GgmlType::F32,
        )
        .unwrap();
        let copied =
            MetalTensor::zeros_f32(ctx, vec![weights.geometry.hidden_size as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read =
            encode_qwen_sparse_attention_text(ctx, &encoder, &input, weights.borrowed(), workspace)
                .unwrap();
        read.output()
            .encode_copy_to(ctx, &encoder, &copied)
            .unwrap();
        encoder.end();
        command.commit();
        drop(read);
        workspace.release_after().unwrap();
        read_f32(&copied)
    }

    #[test]
    fn dense_packed_qsa_matches_scalar_rows_state_and_topology() {
        const MAX_TOKENS: usize = 64;
        let Some(ctx) = context() else { return };
        let geometry = packed_test_geometry(64);
        assert_eq!(geometry.query_heads, 24);
        assert_eq!(geometry.kv_heads, 2);
        let weights = test_weights(&ctx, geometry);
        let inputs = values(MAX_TOKENS * geometry.hidden_size, 1_701, 0.002_1);
        let serial = serial_dense_trace(&ctx, &weights, &inputs, MAX_TOKENS);

        for tokens in [1_usize, 2, 8, 16, 33, 64] {
            let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
            let scratch = QwenSparseAttentionPackedScratch::new(&ctx, geometry, tokens).unwrap();
            let (actual, census) = encode_dense_packed_chunk(
                &ctx,
                &weights,
                &mut workspace,
                &scratch,
                &inputs[..tokens * geometry.hidden_size],
                0,
                tokens,
            );
            let expected = &serial.outputs[..tokens * geometry.hidden_size];
            if tokens == 1 {
                assert_eq!(actual, expected, "N=1 must delegate exactly");
                assert_eq!(
                    read_f32(&workspace.pending_index_keys),
                    serial.states[0].pending_index_keys
                );
                assert_eq!(
                    read_f16(&workspace.key_cache)[..geometry.kv_width()],
                    serial.states[0].key_cache
                );
                assert_eq!(
                    read_f16(&workspace.value_cache)[..geometry.kv_width()],
                    serial.states[0].value_cache
                );
            } else {
                for token in 0..tokens {
                    let start = token * geometry.hidden_size;
                    assert_similarity(
                        &format!("dense packed QSA N={tokens} token={token}"),
                        &actual[start..start + geometry.hidden_size],
                        &expected[start..start + geometry.hidden_size],
                        1e-3,
                        0.9999997,
                        1e-5,
                    );
                }
            }
            assert_dense_state_matches(
                &format!("dense packed QSA N={tokens}"),
                &workspace,
                &serial.states[tokens - 1],
            );

            let names = census
                .iter()
                .map(|row| row.kernel.as_str())
                .collect::<Vec<_>>();
            if tokens == 1 {
                assert!(names.contains(&"kernel_qwen4exp_qsa_write_pending_f32"));
                assert!(names.contains(&"kernel_qwen4exp_qsa_attention_logits_f16"));
                assert!(!names.iter().any(|name| name.contains("packed")));
                continue;
            }
            for required in [
                "kernel_qwen4exp_qsa_commit_pending_packed_f32",
                "kernel_qwen4exp_qsa_fill_block_ids_i32",
                "kernel_qk_rms_norm_rope_f32_packed_consecutive",
                "kernel_scatter_offset_f32_to_f16_kv",
                "kernel_attn_matrix_softmax_f32",
                "kernel_attn_matrix_kqv_direct_v_f32",
                "kernel_sigmoid_mul_gate_strided_f32",
            ] {
                assert!(names.contains(&required), "N={tokens} missing {required}");
            }
            assert_eq!(
                names
                    .iter()
                    .filter(|&&name| name == "kernel_qwen4exp_qsa_pool_publish_packed_f16")
                    .count(),
                usize::from(tokens >= geometry.ratio)
            );
            let attention_tiles = tokens.div_ceil(DENSE_PACKED_QUERY_TILE);
            assert_eq!(
                names
                    .iter()
                    .filter(|&&name| {
                        matches!(
                            name,
                            "kernel_attn_matrix_kq_f32" | "kernel_attn_matrix_kq_f32_full_tiles"
                        )
                    })
                    .count(),
                attention_tiles
            );
            if tokens == 64 {
                assert_eq!(
                    names
                        .iter()
                        .filter(|&&name| name == "kernel_attn_matrix_kq_f32_full_tiles")
                        .count(),
                    attention_tiles
                );
            }
            assert_eq!(
                names
                    .iter()
                    .filter(|&&name| name == "kernel_attn_matrix_softmax_f32")
                    .count(),
                attention_tiles
            );
            assert_eq!(
                names
                    .iter()
                    .filter(|&&name| name == "kernel_attn_matrix_kqv_direct_v_f32")
                    .count(),
                attention_tiles
            );
            for absent in [
                "qsa_index_scores",
                "select_top_k",
                "qsa_expand_ids",
                "qsa_attention_logits",
                "qsa_attention_softmax_value",
            ] {
                assert!(
                    !names.iter().any(|name| name.contains(absent)),
                    "N={tokens} unexpectedly dispatched {absent}"
                );
            }
        }
    }

    #[test]
    fn dense_packed_qsa_cross_block_continuation_matches_scalar() {
        const TOKENS: usize = 33;
        let Some(ctx) = context() else { return };
        let geometry = packed_test_geometry(64);
        let weights = test_weights(&ctx, geometry);
        let inputs = values(TOKENS * geometry.hidden_size, 1_919, 0.001_9);
        let serial = serial_dense_trace(&ctx, &weights, &inputs, TOKENS);
        let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
        let scratch = QwenSparseAttentionPackedScratch::new(&ctx, geometry, TOKENS).unwrap();
        let mut start_position = 0;

        for chunk in [2_usize, 8, 16, 7] {
            let input_start = start_position * geometry.hidden_size;
            let input_end = (start_position + chunk) * geometry.hidden_size;
            let (actual, _) = encode_dense_packed_chunk(
                &ctx,
                &weights,
                &mut workspace,
                &scratch,
                &inputs[input_start..input_end],
                start_position,
                chunk,
            );
            let expected = &serial.outputs[input_start..input_end];
            for token in 0..chunk {
                let row = token * geometry.hidden_size;
                assert_similarity(
                    &format!("dense packed QSA continuation start={start_position} token={token}"),
                    &actual[row..row + geometry.hidden_size],
                    &expected[row..row + geometry.hidden_size],
                    1e-3,
                    0.9999997,
                    1e-5,
                );
            }
            start_position += chunk;
            assert_dense_state_matches(
                &format!("dense packed QSA continuation end={start_position}"),
                &workspace,
                &serial.states[start_position - 1],
            );
        }
        assert_eq!(start_position, TOKENS);

        for (prefix, chunk) in [(0_usize, 4_usize), (1, 3), (2, 2), (3, 2)] {
            let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
            for token in 0..prefix {
                let start = token * geometry.hidden_size;
                encode_one(
                    &ctx,
                    &weights,
                    &mut workspace,
                    &inputs[start..start + geometry.hidden_size],
                );
            }
            let scratch = QwenSparseAttentionPackedScratch::new(&ctx, geometry, chunk).unwrap();
            let input_start = prefix * geometry.hidden_size;
            let input_end = (prefix + chunk) * geometry.hidden_size;
            let (actual, _) = encode_dense_packed_chunk(
                &ctx,
                &weights,
                &mut workspace,
                &scratch,
                &inputs[input_start..input_end],
                prefix,
                chunk,
            );
            let expected = &serial.outputs[input_start..input_end];
            for token in 0..chunk {
                let row = token * geometry.hidden_size;
                assert_similarity(
                    &format!(
                        "dense packed QSA residue={} token={token}",
                        prefix % geometry.ratio
                    ),
                    &actual[row..row + geometry.hidden_size],
                    &expected[row..row + geometry.hidden_size],
                    1e-3,
                    0.9999997,
                    1e-5,
                );
            }
            assert_dense_state_matches(
                &format!("dense packed QSA residue={}", prefix % geometry.ratio),
                &workspace,
                &serial.states[prefix + chunk - 1],
            );
        }
    }

    #[test]
    fn dense_packed_qsa_preflight_ownership_poison_and_reset_are_strict() {
        let Some(ctx) = context() else { return };
        let geometry = packed_test_geometry(64);
        let weights = test_weights(&ctx, geometry);
        let input_values = values(2 * geometry.hidden_size, 2_113, 0.002_3);
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input_values),
            vec![geometry.hidden_size as u64, 2],
            GgmlType::F32,
        )
        .unwrap();
        let scratch = QwenSparseAttentionPackedScratch::new(&ctx, geometry, 2).unwrap();
        let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();

        let mut writable_query = weights.query.clone();
        writable_query.provenance = MetalTensorProvenance::OwnedWritable;
        let bad_weights = QwenSparseAttentionMetalWeights {
            query: &writable_query,
            ..weights.borrowed()
        };
        let bad_command = ctx.queue.commandBuffer().unwrap();
        let bad_encoder = KernelEncoder::begin(&bad_command);
        crate::metal::dispatch_census_begin();
        let error = match unsafe {
            encode_qwen_sparse_attention_text_dense_packed_motor(
                &ctx,
                &bad_encoder,
                &input,
                bad_weights,
                &mut workspace,
                &scratch,
                0,
                2,
            )
        } {
            Ok(_) => panic!("writable QSA weight was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("read-only weight provenance"));
        assert!(crate::metal::dispatch_census_take().is_empty());
        bad_encoder.end();
        assert!(workspace.active_command.is_none());
        assert_eq!(workspace.committed_length(), 0);
        assert!(!workspace.is_poisoned());

        let alias = scratch
            .output
            .view_subrange(0, vec![geometry.hidden_size as u64, 2]);
        let alias_command = ctx.queue.commandBuffer().unwrap();
        let alias_encoder = KernelEncoder::begin(&alias_command);
        crate::metal::dispatch_census_begin();
        let error = match unsafe {
            encode_qwen_sparse_attention_text_dense_packed_motor(
                &ctx,
                &alias_encoder,
                &alias,
                weights.borrowed(),
                &mut workspace,
                &scratch,
                0,
                2,
            )
        } {
            Ok(_) => panic!("aliased packed QSA input was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("overlap"));
        assert!(crate::metal::dispatch_census_take().is_empty());
        alias_encoder.end();
        assert!(workspace.active_command.is_none());
        assert!(!workspace.is_poisoned());

        let abandoned_command = ctx.queue.commandBuffer().unwrap();
        let abandoned_encoder = KernelEncoder::begin(&abandoned_command);
        unsafe {
            encode_qwen_sparse_attention_text_dense_packed_motor(
                &ctx,
                &abandoned_encoder,
                &input,
                weights.borrowed(),
                &mut workspace,
                &scratch,
                0,
                2,
            )
        }
        .unwrap();
        abandoned_encoder.end();
        assert!(workspace.reset().is_err());
        unsafe { workspace.abandon_uncommitted().unwrap() };
        drop(abandoned_command);
        assert_eq!(workspace.committed_length(), 0);
        assert!(!workspace.is_poisoned());

        let poison_command = ctx.queue.commandBuffer().unwrap();
        let poison_encoder = KernelEncoder::begin(&poison_command);
        unsafe {
            encode_qwen_sparse_attention_text_dense_packed_motor(
                &ctx,
                &poison_encoder,
                &input,
                weights.borrowed(),
                &mut workspace,
                &scratch,
                0,
                2,
            )
        }
        .unwrap();
        poison_encoder.end();
        poison_command.commit();
        poison_command.waitUntilCompleted();
        write_i32_scalar(&workspace.selector_status, 17).unwrap();
        assert!(workspace.release_after().is_err());
        assert!(workspace.is_poisoned());
        workspace.reset().unwrap();
        let (output, _) = encode_dense_packed_chunk(
            &ctx,
            &weights,
            &mut workspace,
            &scratch,
            &input_values,
            0,
            2,
        );
        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(workspace.committed_length(), 2);
    }

    #[test]
    fn thirteen_tokens_match_independent_pytorch_oracle() {
        let fixture: Value = serde_json::from_str(QSA_ORACLE_JSON).unwrap();
        assert_eq!(fixture["schema"], "qwen4exp-qsa-text-f16-oracle");
        assert_eq!(fixture["schema_version"], 1);
        assert_eq!(fixture["generator_version"], 1);
        assert_eq!(fixture["binary"]["file"], "qwen4exp_qsa_text_f16_v1.f32");
        assert_eq!(fixture["binary"]["dtype"], "f32");
        assert_eq!(fixture["binary"]["byte_order"], "little");
        assert_eq!(
            format!("{:x}", Sha256::digest(QSA_ORACLE_F32)),
            fixture["binary"]["sha256"].as_str().unwrap()
        );
        assert_source_identity(
            &fixture,
            "vllm",
            "02f2b4c15dd987d9436e125aab29604447c77405",
            "72eaeeeb9bfff19494a9d19ee85a6a83745d0602",
            &[
                (
                    "vllm/models/qwen4_exp/nvidia/indexer_qsa.py",
                    "668706c3a59c51e2c1ed51d19bd9a0e1564a0aad68c5e2bc20d5fb9e65cc2f98",
                ),
                (
                    "vllm/models/qwen4_exp/nvidia/ops/qsa.py",
                    "faa8d358c79745f304edd363e4da21992e4cf015a22316b14980500bd199a0ad",
                ),
                (
                    "tests/models/qwen4_exp/test_qsa_reference.py",
                    "7396d4482c2e7a0529bd927b2922925a709916910b651b2ed5da790ec2d385f1",
                ),
            ],
        );
        assert_source_identity(
            &fixture,
            "sglang",
            "73a255206f916366c8d26d4022f82ddfb0ab558d",
            "dc134c86f21a7396d89bdb01a8019e8db81d763a",
            &[
                (
                    "python/sglang/srt/layers/attention/qsa/qsa_indexer.py",
                    "bb57ce1e9abc4fbfcba2c9aaaf125b9e625983966497df57165b6d4c6461afe2",
                ),
                (
                    "python/sglang/srt/layers/attention/qsa/kernel.py",
                    "5482e38d30bfaf1624ec0625b4896cbb395a1637f75c183c8ca723c9f6055ff8",
                ),
                (
                    "python/sglang/srt/layers/attention/qsa/mqa.py",
                    "af36d5c8f4fbda5b0e82b7f31046a95c9a709fcc57b3600c6473c49e87b7629f",
                ),
                (
                    "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py",
                    "c959835d05d0f395ad7eae4330cf264af9f6f7c1bff3d45a39bb953d2536f5f2",
                ),
            ],
        );
        assert!(fixture["minimum_sparse_score_margin"].as_f64().unwrap() > 0.05);
        for fault in [
            "modulo_gqa_mapping",
            "silu_gate",
            "partial_extra_block_tail_residue_1",
        ] {
            assert!(
                fixture["fault_sensitivity_max_abs_output"][fault]
                    .as_f64()
                    .unwrap()
                    > 1e-2
            );
        }

        let Some(ctx) = context() else { return };
        let geometry = test_geometry(16);
        assert_eq!(geometry.query_heads, 4);
        assert_eq!(geometry.kv_heads, 2);
        assert_eq!(geometry.hidden_size, 16);
        let weights = oracle_weights(&ctx, geometry, &fixture);
        let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
        let oracle = oracle_values();
        let inputs = formula_from_json(
            &fixture["recipe"]["weights_and_inputs"]["inputs"],
            13,
            geometry.hidden_size,
        );
        let mut observed_residues = [false; 4];
        let mut saw_true_sparse = false;
        for position in 0..13 {
            let input =
                &inputs[position * geometry.hidden_size..(position + 1) * geometry.hidden_size];
            let actual = encode_one(&ctx, &weights, &mut workspace, input);
            let row = |name: &str, width: usize| {
                let section = oracle_section(&fixture, &oracle, name);
                &section[position * width..(position + 1) * width]
            };
            assert_close(&actual, row("output", geometry.hidden_size), 4e-4, 5e-4);
            assert_close(
                &read_f32(&workspace.attention),
                row("attention", geometry.query_width()),
                4e-4,
                4e-4,
            );
            assert_close(
                &read_f32(&workspace.index_query),
                row("index_query", geometry.index_query_width()),
                8e-5,
                8e-5,
            );
            assert_close(
                &read_f32(&workspace.query),
                row("query", geometry.query_width()),
                8e-5,
                8e-5,
            );
            assert_close(
                &read_f32(&workspace.raw_gate),
                row("raw_gate", geometry.query_width()),
                2e-5,
                2e-5,
            );
            assert_close(
                &read_f32(&workspace.key),
                row("key", geometry.kv_width()),
                8e-5,
                8e-5,
            );
            assert_close(
                &read_f32(&workspace.value),
                row("value", geometry.kv_width()),
                2e-5,
                2e-5,
            );

            let step = &fixture["steps"][position];
            let expected_ids = step["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_i64().unwrap() as i32)
                .collect::<Vec<_>>();
            assert_eq!(read_i32(&workspace.token_ids), expected_ids);
            let expected_blocks = step["selected_blocks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_i64().unwrap() as i32)
                .collect::<Vec<_>>();
            assert_eq!(
                read_i32_scalar(&workspace.selected_count).unwrap(),
                expected_blocks.len() as i32
            );
            let mut expected_block_slots = expected_blocks.clone();
            expected_block_slots.resize(geometry.block_budget(), -1);
            assert_eq!(read_i32(&workspace.selected_blocks), expected_block_slots);
            let visible = step["visible_blocks"].as_u64().unwrap() as usize;
            if visible > 0 {
                assert_close(
                    &read_f32(&workspace.scores)[..visible],
                    &row("scores", geometry.block_capacity())[..visible],
                    3e-4,
                    3e-4,
                );
            }
            observed_residues[(position + 1) % geometry.ratio] = true;
            if visible > geometry.block_budget() {
                saw_true_sparse = true;
                assert!(expected_blocks.len() < visible);
            }
        }
        assert!(observed_residues.into_iter().all(|seen| seen));
        assert!(saw_true_sparse);
        assert_eq!(workspace.committed_length(), 13);
        assert_close(
            &read_f16(&workspace.compressed_index_keys),
            oracle_section(&fixture, &oracle, "compressed_cache"),
            5e-4,
            0.0,
        );
        assert_close(
            &read_f16(&workspace.key_cache),
            oracle_section(&fixture, &oracle, "key_cache"),
            1e-3,
            0.0,
        );
        assert_close(
            &read_f16(&workspace.value_cache),
            oracle_section(&fixture, &oracle, "value_cache"),
            1e-3,
            0.0,
        );
        assert_eq!(
            fixture["steps"][11]["newly_completed_third_block_selected"],
            true
        );
        assert_eq!(
            fixture["steps"][12]["newly_completed_third_block_selected"],
            true
        );
    }

    #[test]
    fn command_ownership_abandon_poison_reset_and_capacity_are_strict() {
        let Some(ctx) = context() else { return };
        let geometry = test_geometry(4);
        let weights = test_weights(&ctx, geometry);
        let input_values = values(geometry.hidden_size, 700, 0.01);
        let input = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input_values),
            vec![geometry.hidden_size as u64],
            GgmlType::F32,
        )
        .unwrap();
        let destination = MetalTensor::zeros_f32(&ctx, vec![geometry.hidden_size as u64]).unwrap();
        let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();

        let first_command = ctx.queue.commandBuffer().unwrap();
        let first_encoder = KernelEncoder::begin(&first_command);
        let read = encode_qwen_sparse_attention_text(
            &ctx,
            &first_encoder,
            &input,
            weights.borrowed(),
            &mut workspace,
        )
        .unwrap();
        let other_command = ctx.queue.commandBuffer().unwrap();
        let other_encoder = KernelEncoder::begin(&other_command);
        assert!(
            read.output()
                .encode_copy_to(&ctx, &other_encoder, &destination)
                .is_err()
        );
        other_encoder.end();
        drop(other_command);
        first_encoder.end();
        drop(read);
        assert!(workspace.release_after().is_err());
        workspace.state_poisoned = true;
        unsafe { workspace.abandon_uncommitted().unwrap() };
        assert_eq!(workspace.committed_length(), 0);
        assert!(!workspace.is_poisoned());

        let concurrent_command = ctx.queue.commandBuffer().unwrap();
        let concurrent_encoder = KernelEncoder::begin_concurrent(&concurrent_command);
        assert!(
            encode_qwen_sparse_attention_text(
                &ctx,
                &concurrent_encoder,
                &input,
                weights.borrowed(),
                &mut workspace,
            )
            .is_err()
        );
        concurrent_encoder.end();

        let owner_command = ctx.queue.commandBuffer().unwrap();
        let owner_encoder = KernelEncoder::begin(&owner_command);
        let owner_read = encode_qwen_sparse_attention_text(
            &ctx,
            &owner_encoder,
            &input,
            weights.borrowed(),
            &mut workspace,
        )
        .unwrap();
        owner_encoder.end();
        drop(owner_read);
        let second_command = ctx.queue.commandBuffer().unwrap();
        let second_encoder = KernelEncoder::begin(&second_command);
        assert!(
            encode_qwen_sparse_attention_text(
                &ctx,
                &second_encoder,
                &input,
                weights.borrowed(),
                &mut workspace,
            )
            .is_err()
        );
        second_encoder.end();
        unsafe { workspace.abandon_uncommitted().unwrap() };

        let poison_command = ctx.queue.commandBuffer().unwrap();
        let poison_encoder = KernelEncoder::begin(&poison_command);
        let poison_read = encode_qwen_sparse_attention_text(
            &ctx,
            &poison_encoder,
            &input,
            weights.borrowed(),
            &mut workspace,
        )
        .unwrap();
        poison_encoder.end();
        poison_command.commit();
        poison_command.waitUntilCompleted();
        drop(poison_read);
        write_i32_scalar(&workspace.selector_status, 17).unwrap();
        assert!(workspace.release_after().is_err());
        assert!(workspace.is_poisoned());
        assert!(encode_one_result(&ctx, &weights, &mut workspace, &input_values).is_err());
        workspace.reset().unwrap();
        assert!(!workspace.is_poisoned());

        for token in 0..geometry.capacity {
            let values = values(geometry.hidden_size, 900 + token, 0.01);
            encode_one(&ctx, &weights, &mut workspace, &values);
        }
        assert_eq!(workspace.committed_length(), geometry.capacity);
        assert!(encode_one_result(&ctx, &weights, &mut workspace, &input_values).is_err());
        assert_eq!(workspace.committed_length(), geometry.capacity);
    }

    fn encode_one_result(
        ctx: &MetalContext,
        weights: &TestWeights,
        workspace: &mut QwenSparseAttentionMetalWorkspace,
        input_values: &[f32],
    ) -> Result<(), Qwen4ExpQsaError> {
        let input = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(input_values),
            vec![weights.geometry.hidden_size as u64],
            GgmlType::F32,
        )
        .unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let result =
            encode_qwen_sparse_attention_text(ctx, &encoder, &input, weights.borrowed(), workspace);
        encoder.end();
        match result {
            Ok(read) => {
                command.commit();
                drop(read);
                workspace.release_after()
            }
            Err(error) => Err(error),
        }
    }

    #[test]
    fn released_workspace_estimate_prices_large_f16_caches_and_logits() {
        let config = Qwen4ExpConfig::flash_next_reference();
        let geometry = QwenSparseAttentionMetalGeometry::from_config(
            &config,
            3,
            config.context_length as usize,
        )
        .unwrap();
        let estimate = geometry.checked_workspace_byte_estimate().unwrap();
        let allocations = geometry.workspace_logical_allocations().unwrap();
        assert_eq!(allocations.len(), 22);
        assert_eq!(
            allocations.iter().sum::<usize>(),
            estimate.total_workspace_bytes
        );
        let expected_cache = config.context_length as usize
            * config.attention.kv_heads as usize
            * config.attention.key_head_dim as usize
            * 2;
        assert_eq!(estimate.main_key_cache_bytes, expected_cache);
        assert_eq!(estimate.main_value_cache_bytes, expected_cache);
        assert_eq!(
            estimate.logits_bytes,
            geometry.output_width() * config.attention.query_heads as usize * 4
        );
        assert!(estimate.total_workspace_bytes > 2 * expected_cache);
    }

    #[test]
    fn released_attention_launch_geometry_and_large_width_reduction_match_cpu() {
        let Some(ctx) = context() else { return };
        let config = Qwen4ExpConfig::flash_next_reference();
        let geometry = QwenSparseAttentionMetalGeometry::from_config(&config, 3, 2_052).unwrap();
        assert_eq!(geometry.query_heads, 24);
        assert_eq!(geometry.kv_heads, 2);
        assert_eq!(geometry.head_dim, 256);
        assert_eq!(geometry.output_width(), 2_051);
        let workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();

        let query = (0..geometry.query_width())
            .map(|index| ((index * 17 + 3) % 97) as f32 * 0.003 - 0.144)
            .collect::<Vec<_>>();
        let raw_gate = (0..geometry.query_width())
            .map(|index| ((index * 11 + 5) % 61) as f32 * 0.07 - 2.1)
            .collect::<Vec<_>>();
        let token_ids = (0..geometry.output_width())
            .map(|position| position as i32)
            .collect::<Vec<_>>();
        let cache_elements = geometry.output_width() * geometry.kv_width();
        let mut key_cache = Vec::with_capacity(cache_elements);
        let mut value_cache = Vec::with_capacity(cache_elements);
        for position in 0..geometry.output_width() {
            for kv_head in 0..geometry.kv_heads {
                for lane in 0..geometry.head_dim {
                    let key =
                        ((position * 7 + kv_head * 13 + lane * 3 + 1) % 113) as f32 * 0.004 - 0.224;
                    let value = ((position * 5 + kv_head * 19 + lane * 11 + 2) % 127) as f32
                        * 0.003
                        - 0.189;
                    key_cache.push(f16::from_f32(key).to_f32());
                    value_cache.push(f16::from_f32(value).to_f32());
                }
            }
        }
        write_f32_tensor(&workspace.query, &query);
        write_f32_tensor(&workspace.raw_gate, &raw_gate);
        write_i32_tensor(&workspace.token_ids, &token_ids);
        write_f16_prefix(&workspace.key_cache, &key_cache);
        write_f16_prefix(&workspace.value_cache, &value_cache);

        let mut expected = vec![0.0; geometry.query_width()];
        let queries_per_kv = geometry.query_heads / geometry.kv_heads;
        let scale = 1.0 / (geometry.head_dim as f32).sqrt();
        for query_head in 0..geometry.query_heads {
            let kv_head = query_head / queries_per_kv;
            let query_row =
                &query[query_head * geometry.head_dim..(query_head + 1) * geometry.head_dim];
            let mut logits = Vec::with_capacity(geometry.output_width());
            for position in 0..geometry.output_width() {
                let key_start = position * geometry.kv_width() + kv_head * geometry.head_dim;
                let dot = query_row
                    .iter()
                    .zip(&key_cache[key_start..key_start + geometry.head_dim])
                    .map(|(&q, &k)| q * k)
                    .sum::<f32>();
                logits.push(dot * scale);
            }
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let masses = logits
                .iter()
                .map(|&logit| (logit - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = masses.iter().sum::<f32>();
            for lane in 0..geometry.head_dim {
                let accumulator = masses
                    .iter()
                    .enumerate()
                    .map(|(position, &mass)| {
                        let value_start =
                            position * geometry.kv_width() + kv_head * geometry.head_dim;
                        value_cache[value_start + lane] * mass
                    })
                    .sum::<f32>();
                let index = query_head * geometry.head_dim + lane;
                expected[index] = accumulator / denominator / (1.0 + (-raw_gate[index]).exp());
            }
        }
        assert!(expected.iter().all(|value| value.is_finite()));

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        encode_attention_logits(&ctx, &encoder, &workspace, geometry.output_width()).unwrap();
        encode_attention_softmax_value(&ctx, &encoder, &workspace, geometry.output_width())
            .unwrap();
        let census = crate::metal::dispatch_census_take();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "command failed: {:?}",
            command.error()
        );

        assert_eq!(census.len(), 2);
        let logits_launch = &census[0];
        assert_eq!(
            logits_launch.kernel,
            "kernel_qwen4exp_qsa_attention_logits_f16"
        );
        assert_eq!(logits_launch.grid_width, 6_153);
        assert_eq!(logits_launch.grid_height, 1);
        assert_eq!(logits_launch.threads_width, 256);
        let value_launch = &census[1];
        assert_eq!(
            value_launch.kernel,
            "kernel_qwen4exp_qsa_attention_softmax_value_f16"
        );
        assert_eq!(value_launch.grid_width, 24);
        assert_eq!(value_launch.grid_height, 1);
        assert_eq!(value_launch.threads_width, 256);

        let actual = read_f32(&workspace.attention);
        assert!(actual.iter().all(|value| value.is_finite()));
        assert_close(&actual, &expected, 3e-5, 3e-5);
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_QSA_GGUF to the pinned full release"]
    fn released_layer_three_dense_packed_matches_scalar_rows_and_state() {
        const MAX_TOKENS: usize = 33;
        let path = std::env::var_os("QWEN4EXP_Q3_K_XL_QSA_GGUF")
            .expect("QWEN4EXP_Q3_K_XL_QSA_GGUF must point to the first Q3 shard");
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let ctx = MetalContext::new().expect("initialize Metal");
        let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
        let admitted = plan.admit(ctx.memory_signals()).unwrap();
        let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted).unwrap();
        let bound = QwenSparseAttentionMetalWeights::bind(realized.weights(), 3, 64).unwrap();
        assert_eq!(bound.index_key.dtype, GgmlType::BF16);
        for tensor in [bound.query, bound.key, bound.value, bound.output] {
            assert_eq!(tensor.dtype, GgmlType::Q8_0);
        }
        let geometry = bound.geometry;
        let weights = TestWeights {
            geometry,
            query: bound.query.clone(),
            key: bound.key.clone(),
            value: bound.value.clone(),
            output: bound.output.clone(),
            query_norm: bound.query_norm.clone(),
            key_norm: bound.key_norm.clone(),
            index_query: bound.index_query.clone(),
            index_key: bound.index_key.clone(),
            index_query_norm: bound.index_query_norm.clone(),
            index_key_norm: bound.index_key_norm.clone(),
        };
        let inputs = values(MAX_TOKENS * geometry.hidden_size, 2_311, 0.001_7);
        let serial = serial_dense_trace(&ctx, &weights, &inputs, MAX_TOKENS);

        for tokens in [2_usize, 8, 16, 33] {
            let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
            let scratch = QwenSparseAttentionPackedScratch::new(&ctx, geometry, tokens).unwrap();
            let (actual, census) =
                crate::metal_forward::with_matmat_bf16_bfloat_act_override(true, || {
                    encode_dense_packed_chunk(
                        &ctx,
                        &weights,
                        &mut workspace,
                        &scratch,
                        &inputs[..tokens * geometry.hidden_size],
                        0,
                        tokens,
                    )
                });
            let expected = &serial.outputs[..tokens * geometry.hidden_size];
            for token in 0..tokens {
                let start = token * geometry.hidden_size;
                assert_similarity(
                    &format!("released dense packed QSA N={tokens} token={token}"),
                    &actual[start..start + geometry.hidden_size],
                    &expected[start..start + geometry.hidden_size],
                    9e-4,
                    0.99999965,
                    1e-4,
                );
            }
            let expected_state = &serial.states[tokens - 1];
            assert_similarity(
                &format!("released dense packed QSA N={tokens} pending index state"),
                &read_f32(&workspace.pending_index_keys),
                &expected_state.pending_index_keys,
                2e-6,
                0.99999999,
                1e-6,
            );
            let completed_index_elements = expected_state.compressed_index_keys.len();
            assert_close(
                &read_f16(&workspace.compressed_index_keys)[..completed_index_elements],
                &expected_state.compressed_index_keys,
                3e-3,
                3e-3,
            );
            let cache_elements = expected_state.key_cache.len();
            assert_close(
                &read_f16(&workspace.key_cache)[..cache_elements],
                &expected_state.key_cache,
                3e-3,
                3e-3,
            );
            assert_close(
                &read_f16(&workspace.value_cache)[..cache_elements],
                &expected_state.value_cache,
                3e-3,
                3e-3,
            );
            assert_eq!(workspace.committed_length(), tokens);
            assert_eq!(
                read_i32_scalar(&workspace.selected_count).unwrap(),
                (tokens / geometry.ratio) as i32
            );

            let names = census
                .iter()
                .map(|row| row.kernel.as_str())
                .collect::<Vec<_>>();
            let expected_q8 = match tokens {
                8 => "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32",
                16 => "kernel_mat_mat_q8_0_f32_n16",
                _ => "kernel_mat_mat_q8_0_f32",
            };
            assert_eq!(
                names.iter().filter(|&&name| name == expected_q8).count(),
                4,
                "released N={tokens} Q8 route: {names:?}"
            );
            assert_eq!(
                names
                    .iter()
                    .filter(|&&name| name == "kernel_mat_mat_bf16_f32")
                    .count(),
                1,
                "released N={tokens} BF16 route: {names:?}"
            );
        }
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_QSA_GGUF to the pinned full release"]
    fn released_layer_three_position_zero_matches_cpu_quantized_oracle() {
        let path = std::env::var_os("QWEN4EXP_Q3_K_XL_QSA_GGUF")
            .expect("QWEN4EXP_Q3_K_XL_QSA_GGUF must point to the first Q3 shard");
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let ctx = MetalContext::new().expect("initialize Metal");
        let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
        let admitted = plan.admit(ctx.memory_signals()).unwrap();
        let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted).unwrap();
        let metal_weights = realized.weights();

        let mut qsa_layers = 0;
        for layer in 0..48 {
            let qsa = QwenSparseAttentionMetalWeights::bind(metal_weights, layer, 4);
            if layer % 4 == 3 {
                qsa.unwrap();
                assert!(
                    GatedDeltaNetMetalWeights::bind(metal_weights, layer).is_err(),
                    "QSA layer {layer} accepted as GDN"
                );
                qsa_layers += 1;
            } else {
                assert!(qsa.is_err(), "GDN layer {layer} accepted as QSA");
            }
        }
        assert_eq!(qsa_layers, 12);

        let weights = QwenSparseAttentionMetalWeights::bind(metal_weights, 3, 4).unwrap();
        let geometry = weights.geometry;
        let input = values(geometry.hidden_size, 1_003, 0.01);
        let dequant = |name: &str| {
            let desc = gguf.find(name).unwrap();
            crate::codec::dequant_to_f32(desc, gguf.try_slice(desc).unwrap()).unwrap()
        };

        let index_query_weight = dequant("blk.3.indexer.q_proj.weight");
        let index_query_raw = real_mat_vec(&index_query_weight, &input, geometry.hidden_size);
        drop(index_query_weight);
        let index_key_weight = dequant("blk.3.indexer.k_proj.weight");
        let index_key_raw = real_mat_vec(&index_key_weight, &input, geometry.hidden_size);
        drop(index_key_weight);
        let index_query_norm = dequant("blk.3.indexer.q_norm.weight");
        let index_query = rmsnorm_heads_position_zero(
            &index_query_raw,
            geometry.index_query_heads,
            geometry.index_head_dim,
            &index_query_norm,
            geometry.eps,
        );
        drop(index_query_norm);
        let index_key_norm = dequant("blk.3.indexer.k_norm.weight");
        assert_eq!(index_key_norm.len(), geometry.index_head_dim);
        drop(index_key_norm);

        let query_weight = dequant("blk.3.attn_q.weight");
        let query_gate = real_mat_vec(&query_weight, &input, geometry.hidden_size);
        drop(query_weight);
        let mut query_raw = Vec::with_capacity(geometry.query_width());
        let mut raw_gate = Vec::with_capacity(geometry.query_width());
        for head in 0..geometry.query_heads {
            let start = head * geometry.head_dim * 2;
            query_raw.extend_from_slice(&query_gate[start..start + geometry.head_dim]);
            raw_gate.extend_from_slice(
                &query_gate[start + geometry.head_dim..start + 2 * geometry.head_dim],
            );
        }
        drop(query_gate);
        let query_norm = dequant("blk.3.attn_q_norm.weight");
        let query = rmsnorm_heads_position_zero(
            &query_raw,
            geometry.query_heads,
            geometry.head_dim,
            &query_norm,
            geometry.eps,
        );
        drop(query_norm);

        let key_weight = dequant("blk.3.attn_k.weight");
        let key_raw = real_mat_vec(&key_weight, &input, geometry.hidden_size);
        drop(key_weight);
        let key_norm = dequant("blk.3.attn_k_norm.weight");
        let key = rmsnorm_heads_position_zero(
            &key_raw,
            geometry.kv_heads,
            geometry.head_dim,
            &key_norm,
            geometry.eps,
        );
        drop(key_norm);
        let value_weight = dequant("blk.3.attn_v.weight");
        let value = real_mat_vec(&value_weight, &input, geometry.hidden_size);
        drop(value_weight);

        let key_cache = key
            .iter()
            .map(|&value| f16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        let value_cache = value
            .iter()
            .map(|&value| f16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        let mut attention = vec![0.0; geometry.query_width()];
        let queries_per_kv = geometry.query_heads / geometry.kv_heads;
        for query_head in 0..geometry.query_heads {
            let kv_head = query_head / queries_per_kv;
            for lane in 0..geometry.head_dim {
                let query_index = query_head * geometry.head_dim + lane;
                attention[query_index] = value_cache[kv_head * geometry.head_dim + lane]
                    / (1.0 + (-raw_gate[query_index]).exp());
            }
        }
        let output_weight = dequant("blk.3.attn_output.weight");
        let expected_output = real_mat_vec(&output_weight, &attention, geometry.query_width());
        drop(output_weight);

        let input_gpu = MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&input),
            vec![geometry.hidden_size as u64],
            GgmlType::F32,
        )
        .unwrap();
        let copied_output =
            MetalTensor::zeros_f32(&ctx, vec![geometry.hidden_size as u64]).unwrap();
        let mut workspace = QwenSparseAttentionMetalWorkspace::new(&ctx, geometry).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read =
            encode_qwen_sparse_attention_text(&ctx, &encoder, &input_gpu, weights, &mut workspace)
                .unwrap();
        read.output()
            .encode_copy_to(&ctx, &encoder, &copied_output)
            .unwrap();
        encoder.end();
        command.commit();
        drop(read);
        workspace.release_after().unwrap();

        assert_close(&read_f32(&copied_output), &expected_output, 5e-2, 4e-3);
        assert_close(
            &read_f32(&workspace.index_query_raw),
            &index_query_raw,
            2e-2,
            3e-3,
        );
        assert_close(&read_f32(&workspace.index_query), &index_query, 3e-3, 3e-3);
        assert_close(
            &read_f32(&workspace.pending_index_keys)[..geometry.index_head_dim],
            &index_key_raw,
            2e-2,
            3e-3,
        );
        assert_close(&read_f32(&workspace.query), &query, 3e-3, 3e-3);
        assert_close(&read_f32(&workspace.raw_gate), &raw_gate, 2e-2, 3e-3);
        assert_close(&read_f32(&workspace.key), &key, 3e-3, 3e-3);
        assert_close(&read_f32(&workspace.value), &value, 2e-2, 3e-3);
        let expected_ids = std::iter::once(0)
            .chain(std::iter::repeat_n(-1, geometry.output_width() - 1))
            .collect::<Vec<_>>();
        assert_eq!(read_i32(&workspace.token_ids), expected_ids);
        assert_close(
            &read_f16(&workspace.key_cache)[..geometry.kv_width()],
            &key_cache,
            3e-3,
            3e-3,
        );
        assert_close(
            &read_f16(&workspace.value_cache)[..geometry.kv_width()],
            &value_cache,
            3e-3,
            3e-3,
        );
    }
}
