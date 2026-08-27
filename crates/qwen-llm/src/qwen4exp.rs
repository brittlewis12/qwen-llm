//! Qwen3.8-Flash-Next (`qwen4exp`) architecture contract and CPU references.
//!
//! The architecture contract and CPU references remain independent from the
//! runnable Metal family so metadata, PLE, and QSA semantics stay testable
//! without model residency.

use crate::gguf::{GgufError, GgufFile};
use std::collections::HashSet;

pub const ARCHITECTURE_NAME: &str = "qwen4exp";

pub const FLASH_NEXT_PLE_MULTIPLIERS: [u64; 3] =
    [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];

pub const FLASH_NEXT_PLE_HEAD_VOCAB_SIZES: [u64; 16] = [
    20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069, 20_000_077,
    20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159, 20_000_161, 20_000_171,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixerKind {
    GatedDeltaNet,
    QwenSparseAttention,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HyperConnectionConfig {
    pub count: u32,
    pub low_rank: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionConfig {
    pub query_heads: u32,
    pub kv_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub rotary_dim: u32,
    pub rope_sections: [u32; 4],
    pub rope_theta: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GatedDeltaNetConfig {
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel: u32,
    pub inner_size: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    pub expert_count: u32,
    pub experts_per_token: u32,
    pub expert_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QsaConfig {
    pub query_heads: u32,
    pub key_heads: u32,
    pub head_dim: u32,
    pub token_budget: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PleConfig {
    pub token_vocab_size: u32,
    pub layers: Vec<u32>,
    pub ngram_size: u32,
    pub heads_per_ngram: u32,
    pub embedding_head_dim: u32,
    pub conv_kernel: u32,
    pub eos_token_id: u32,
    pub image_token_id: Option<u32>,
    pub multipliers: Vec<u64>,
    pub head_offsets: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4ExpConfig {
    pub context_length: u32,
    pub layer_count: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub rms_norm_eps: f32,
    pub full_attention_interval: u32,
    pub hyper_connection: HyperConnectionConfig,
    pub attention: AttentionConfig,
    pub gated_delta_net: GatedDeltaNetConfig,
    pub moe: MoeConfig,
    pub qsa: QsaConfig,
    pub compress_ratios: Vec<u32>,
    pub ple: Option<PleConfig>,
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpError {
    #[error("unsupported architecture: {0:?}")]
    UnsupportedArchitecture(Option<String>),
    #[error("missing required metadata key: {0}")]
    MissingMetadata(&'static str),
    #[error("invalid metadata key {key:?}: {reason}")]
    InvalidMetadata { key: &'static str, reason: String },
    #[error(transparent)]
    Gguf(#[from] GgufError),
}

impl Qwen4ExpConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, Qwen4ExpError> {
        if gguf.architecture().as_deref() != Some(ARCHITECTURE_NAME) {
            return Err(Qwen4ExpError::UnsupportedArchitecture(gguf.architecture()));
        }

        let rope_sections =
            required_nonnegative_u32_array(gguf, "qwen4exp.rope.dimension_sections")?;
        let rope_sections: [u32; 4] = rope_sections.try_into().map_err(|values: Vec<u32>| {
            invalid(
                "qwen4exp.rope.dimension_sections",
                format!("expected 4 entries, got {}", values.len()),
            )
        })?;

        let compress_ratios =
            required_nonnegative_u32_array(gguf, "qwen4exp.attention.compress_ratios")?;
        let vocab_size = required_array_len_u32(gguf, "tokenizer.ggml.tokens")?;
        let ple_layers = optional_nonnegative_u32_array(gguf, "qwen4exp.ple.layers")?;
        let ple = match ple_layers {
            Some(layers) if !layers.is_empty() => Some(PleConfig {
                token_vocab_size: vocab_size,
                layers,
                ngram_size: required_u32(gguf, "qwen4exp.ple.ngram_size")?,
                heads_per_ngram: required_u32(gguf, "qwen4exp.ple.heads_per_ngram")?,
                embedding_head_dim: required_u32(
                    gguf,
                    "qwen4exp.embedding_length_per_layer_input",
                )?,
                conv_kernel: required_u32(gguf, "qwen4exp.ple.conv_kernel")?,
                eos_token_id: required_u32(gguf, "qwen4exp.ple.eos_token_id")?,
                image_token_id: optional_u32(gguf, "qwen4exp.ple.image_token_id")?,
                multipliers: required_u64_array(gguf, "qwen4exp.ple.layer_multipliers")?,
                head_offsets: required_u64_array(gguf, "qwen4exp.ple.head_offsets")?,
                head_vocab_sizes: required_u64_array(gguf, "qwen4exp.ple.head_vocab_sizes")?,
            }),
            Some(_) | None => None,
        };

        let config = Self {
            context_length: required_u32(gguf, "qwen4exp.context_length")?,
            layer_count: required_u32(gguf, "qwen4exp.block_count")?,
            hidden_size: required_u32(gguf, "qwen4exp.embedding_length")?,
            vocab_size,
            rms_norm_eps: required_f32(gguf, "qwen4exp.attention.layer_norm_rms_epsilon")?,
            full_attention_interval: required_u32(gguf, "qwen4exp.full_attention_interval")?,
            hyper_connection: HyperConnectionConfig {
                count: required_u32(gguf, "qwen4exp.hyper_connection.count")?,
                low_rank: required_u32(gguf, "qwen4exp.hyper_connection.low_rank")?,
            },
            attention: AttentionConfig {
                query_heads: required_u32(gguf, "qwen4exp.attention.head_count")?,
                kv_heads: required_u32(gguf, "qwen4exp.attention.head_count_kv")?,
                key_head_dim: required_u32(gguf, "qwen4exp.attention.key_length")?,
                value_head_dim: required_u32(gguf, "qwen4exp.attention.value_length")?,
                rotary_dim: required_u32(gguf, "qwen4exp.rope.dimension_count")?,
                rope_sections,
                rope_theta: required_f32(gguf, "qwen4exp.rope.freq_base")?,
            },
            gated_delta_net: GatedDeltaNetConfig {
                key_heads: required_u32(gguf, "qwen4exp.ssm.group_count")?,
                value_heads: required_u32(gguf, "qwen4exp.ssm.time_step_rank")?,
                key_head_dim: required_u32(gguf, "qwen4exp.ssm.state_size")?,
                value_head_dim: required_u32(gguf, "qwen4exp.ssm.state_size")?,
                conv_kernel: required_u32(gguf, "qwen4exp.ssm.conv_kernel")?,
                inner_size: required_u32(gguf, "qwen4exp.ssm.inner_size")?,
            },
            moe: MoeConfig {
                expert_count: required_u32(gguf, "qwen4exp.expert_count")?,
                experts_per_token: required_u32(gguf, "qwen4exp.expert_used_count")?,
                expert_intermediate_size: required_u32(
                    gguf,
                    "qwen4exp.expert_feed_forward_length",
                )?,
                shared_expert_intermediate_size: required_u32(
                    gguf,
                    "qwen4exp.expert_shared_feed_forward_length",
                )?,
            },
            qsa: QsaConfig {
                query_heads: required_u32(gguf, "qwen4exp.attention.indexer.head_count")?,
                key_heads: 1,
                head_dim: required_u32(gguf, "qwen4exp.attention.indexer.key_length")?,
                token_budget: required_u32(gguf, "qwen4exp.attention.indexer.top_k")?,
            },
            compress_ratios,
            ple,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn flash_next_reference() -> Self {
        let mut offsets = Vec::with_capacity(FLASH_NEXT_PLE_HEAD_VOCAB_SIZES.len());
        let mut offset = 0_u64;
        for size in FLASH_NEXT_PLE_HEAD_VOCAB_SIZES {
            offsets.push(offset);
            offset += size;
        }

        Self {
            context_length: 262_144,
            layer_count: 48,
            hidden_size: 2_560,
            vocab_size: 248_320,
            rms_norm_eps: 1e-6,
            full_attention_interval: 4,
            hyper_connection: HyperConnectionConfig {
                count: 4,
                low_rank: 320,
            },
            attention: AttentionConfig {
                query_heads: 24,
                kv_heads: 2,
                key_head_dim: 256,
                value_head_dim: 256,
                rotary_dim: 64,
                rope_sections: [11, 11, 10, 0],
                rope_theta: 10_000_000.0,
            },
            gated_delta_net: GatedDeltaNetConfig {
                key_heads: 16,
                value_heads: 48,
                key_head_dim: 128,
                value_head_dim: 128,
                conv_kernel: 4,
                inner_size: 6_144,
            },
            moe: MoeConfig {
                expert_count: 512,
                experts_per_token: 10,
                expert_intermediate_size: 640,
                shared_expert_intermediate_size: 640,
            },
            qsa: QsaConfig {
                query_heads: 4,
                key_heads: 1,
                head_dim: 128,
                token_budget: 2_048,
            },
            compress_ratios: (0..48)
                .map(|layer| if layer % 4 == 3 { 4 } else { 0 })
                .collect(),
            ple: Some(PleConfig {
                token_vocab_size: 248_320,
                layers: vec![1],
                ngram_size: 3,
                heads_per_ngram: 8,
                embedding_head_dim: 160,
                conv_kernel: 4,
                eos_token_id: 248_044,
                image_token_id: Some(248_056),
                multipliers: FLASH_NEXT_PLE_MULTIPLIERS.to_vec(),
                head_offsets: offsets,
                head_vocab_sizes: FLASH_NEXT_PLE_HEAD_VOCAB_SIZES.to_vec(),
            }),
        }
    }

    pub fn validate(&self) -> Result<(), Qwen4ExpError> {
        require_nonzero("qwen4exp.context_length", self.context_length)?;
        if self.context_length as u64 > i32::MAX as u64 + 1 {
            return Err(invalid(
                "qwen4exp.context_length",
                "must fit the I32 sparse-attention position contract",
            ));
        }
        require_nonzero("qwen4exp.block_count", self.layer_count)?;
        require_nonzero("qwen4exp.embedding_length", self.hidden_size)?;
        require_nonzero("tokenizer.ggml.tokens", self.vocab_size)?;
        require_nonzero(
            "qwen4exp.full_attention_interval",
            self.full_attention_interval,
        )?;
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(invalid(
                "qwen4exp.attention.layer_norm_rms_epsilon",
                "must be finite and positive",
            ));
        }

        require_nonzero(
            "qwen4exp.hyper_connection.count",
            self.hyper_connection.count,
        )?;
        if self.hyper_connection.count < 2 {
            return Err(invalid(
                "qwen4exp.hyper_connection.count",
                "must contain at least two residual streams",
            ));
        }
        require_nonzero(
            "qwen4exp.hyper_connection.low_rank",
            self.hyper_connection.low_rank,
        )?;
        require_nonzero("qwen4exp.attention.head_count", self.attention.query_heads)?;
        require_nonzero("qwen4exp.attention.head_count_kv", self.attention.kv_heads)?;
        require_nonzero("qwen4exp.attention.key_length", self.attention.key_head_dim)?;
        require_nonzero(
            "qwen4exp.attention.value_length",
            self.attention.value_head_dim,
        )?;
        if !self
            .attention
            .query_heads
            .is_multiple_of(self.attention.kv_heads)
        {
            return Err(invalid(
                "qwen4exp.attention.head_count",
                "must be divisible by attention.head_count_kv",
            ));
        }
        if self.attention.key_head_dim != self.attention.value_head_dim {
            return Err(invalid(
                "qwen4exp.attention.value_length",
                "must equal attention.key_length",
            ));
        }
        if self.attention.rotary_dim > self.attention.key_head_dim
            || !self.attention.rotary_dim.is_multiple_of(2)
        {
            return Err(invalid(
                "qwen4exp.rope.dimension_count",
                "must be even and no larger than attention.key_length",
            ));
        }
        let section_sum = self
            .attention
            .rope_sections
            .iter()
            .try_fold(0_u32, |sum, &section| sum.checked_add(section))
            .ok_or_else(|| invalid("qwen4exp.rope.dimension_sections", "section sum overflow"))?;
        if section_sum.checked_mul(2) != Some(self.attention.rotary_dim) {
            return Err(invalid(
                "qwen4exp.rope.dimension_sections",
                "sum must equal half of rope.dimension_count",
            ));
        }
        if !self.attention.rope_theta.is_finite() || self.attention.rope_theta <= 0.0 {
            return Err(invalid(
                "qwen4exp.rope.freq_base",
                "must be finite and positive",
            ));
        }

        let gdn = &self.gated_delta_net;
        require_nonzero("qwen4exp.ssm.group_count", gdn.key_heads)?;
        require_nonzero("qwen4exp.ssm.time_step_rank", gdn.value_heads)?;
        require_nonzero("qwen4exp.ssm.state_size", gdn.key_head_dim)?;
        require_nonzero("qwen4exp.ssm.conv_kernel", gdn.conv_kernel)?;
        if !gdn.value_heads.is_multiple_of(gdn.key_heads) {
            return Err(invalid(
                "qwen4exp.ssm.time_step_rank",
                "must be divisible by ssm.group_count",
            ));
        }
        let expected_inner = gdn
            .value_heads
            .checked_mul(gdn.value_head_dim)
            .ok_or_else(|| invalid("qwen4exp.ssm.inner_size", "dimension overflow"))?;
        if gdn.inner_size != expected_inner {
            return Err(invalid(
                "qwen4exp.ssm.inner_size",
                format!("expected {expected_inner}, got {}", gdn.inner_size),
            ));
        }

        require_nonzero("qwen4exp.expert_count", self.moe.expert_count)?;
        require_nonzero("qwen4exp.expert_used_count", self.moe.experts_per_token)?;
        if self.moe.experts_per_token > self.moe.expert_count {
            return Err(invalid(
                "qwen4exp.expert_used_count",
                "must be no larger than expert_count",
            ));
        }
        require_nonzero(
            "qwen4exp.expert_feed_forward_length",
            self.moe.expert_intermediate_size,
        )?;
        require_nonzero(
            "qwen4exp.expert_shared_feed_forward_length",
            self.moe.shared_expert_intermediate_size,
        )?;

        require_nonzero(
            "qwen4exp.attention.indexer.head_count",
            self.qsa.query_heads,
        )?;
        if self.qsa.key_heads != 1 {
            return Err(invalid(
                "qwen4exp.attention.indexer.head_count_kv",
                "QSA requires one shared index key head",
            ));
        }
        require_nonzero("qwen4exp.attention.indexer.key_length", self.qsa.head_dim)?;
        if self.qsa.head_dim < self.attention.rotary_dim {
            return Err(invalid(
                "qwen4exp.attention.indexer.key_length",
                "must be at least rope.dimension_count",
            ));
        }
        require_nonzero("qwen4exp.attention.indexer.top_k", self.qsa.token_budget)?;

        if self.compress_ratios.len() != self.layer_count as usize {
            return Err(invalid(
                "qwen4exp.attention.compress_ratios",
                format!(
                    "expected {} entries, got {}",
                    self.layer_count,
                    self.compress_ratios.len()
                ),
            ));
        }
        for (layer, &ratio) in self.compress_ratios.iter().enumerate() {
            let scheduled_qsa =
                layer as u32 % self.full_attention_interval == self.full_attention_interval - 1;
            if (ratio > 0) != scheduled_qsa {
                return Err(invalid(
                    "qwen4exp.attention.compress_ratios",
                    format!("layer {layer} disagrees with full_attention_interval"),
                ));
            }
            if ratio > 0 && !self.qsa.token_budget.is_multiple_of(ratio) {
                return Err(invalid(
                    "qwen4exp.attention.indexer.top_k",
                    format!("token budget is not divisible by layer {layer} ratio {ratio}"),
                ));
            }
        }

        if let Some(ple) = &self.ple {
            if ple.token_vocab_size != self.vocab_size {
                return Err(invalid(
                    "tokenizer.ggml.tokens",
                    "PLE and model vocabulary sizes disagree",
                ));
            }
            ple.validate(self.hidden_size, self.layer_count)?;
        }
        Ok(())
    }

    pub fn mixer_kind(&self, layer: u32) -> Option<MixerKind> {
        self.compress_ratios.get(layer as usize).map(|ratio| {
            if *ratio == 0 {
                MixerKind::GatedDeltaNet
            } else {
                MixerKind::QwenSparseAttention
            }
        })
    }

    pub fn qsa_layer_count(&self) -> usize {
        self.compress_ratios
            .iter()
            .filter(|&&ratio| ratio > 0)
            .count()
    }
}

impl PleConfig {
    pub fn head_count(&self) -> Result<u32, Qwen4ExpError> {
        self.ngram_size
            .checked_sub(1)
            .and_then(|orders| orders.checked_mul(self.heads_per_ngram))
            .ok_or_else(|| invalid("qwen4exp.ple.ngram_size", "head count overflow"))
    }

    pub fn logical_row_count(&self) -> Result<u64, Qwen4ExpError> {
        let head_count = self.head_count()? as usize;
        if self.head_offsets.len() != head_count || self.head_vocab_sizes.len() != head_count {
            return Err(invalid(
                "qwen4exp.ple.head_offsets",
                format!("expected {head_count} offset and vocabulary entries"),
            ));
        }
        let mut expected_offset = 0_u64;
        for (&offset, &size) in self.head_offsets.iter().zip(&self.head_vocab_sizes) {
            if size == 0
                || size > i64::MAX as u64
                || offset > i64::MAX as u64
                || offset != expected_offset
            {
                return Err(invalid(
                    "qwen4exp.ple.head_offsets",
                    "head tables must be non-empty and contiguous",
                ));
            }
            expected_offset = expected_offset
                .checked_add(size)
                .ok_or_else(|| invalid("qwen4exp.ple.head_offsets", "table row count overflow"))?;
        }
        Ok(expected_offset)
    }

    pub fn validate(&self, hidden_size: u32, layer_count: u32) -> Result<(), Qwen4ExpError> {
        require_nonzero("tokenizer.ggml.tokens", self.token_vocab_size)?;
        if self.ngram_size < 2 {
            return Err(invalid("qwen4exp.ple.ngram_size", "must be at least 2"));
        }
        require_nonzero("qwen4exp.ple.heads_per_ngram", self.heads_per_ngram)?;
        require_nonzero(
            "qwen4exp.embedding_length_per_layer_input",
            self.embedding_head_dim,
        )?;
        require_nonzero("qwen4exp.ple.conv_kernel", self.conv_kernel)?;
        if self.layers.is_empty() || self.layers.iter().any(|&layer| layer >= layer_count) {
            return Err(invalid(
                "qwen4exp.ple.layers",
                "must contain in-range layer indices",
            ));
        }
        let mut unique_layers = HashSet::with_capacity(self.layers.len());
        if self
            .layers
            .iter()
            .any(|&layer| !unique_layers.insert(layer))
        {
            return Err(invalid(
                "qwen4exp.ple.layers",
                "layer indices must not contain duplicates",
            ));
        }
        if self.eos_token_id >= self.token_vocab_size
            || self
                .image_token_id
                .is_some_and(|token| token >= self.token_vocab_size)
        {
            return Err(invalid(
                "qwen4exp.ple.eos_token_id",
                "PLE special tokens must be inside the tokenizer vocabulary",
            ));
        }

        let head_count = self.head_count()?;
        if head_count.checked_mul(self.embedding_head_dim) != Some(hidden_size) {
            return Err(invalid(
                "qwen4exp.embedding_length_per_layer_input",
                "PLE head count times head width must equal embedding_length",
            ));
        }
        if self.multipliers.len() != self.ngram_size as usize {
            return Err(invalid(
                "qwen4exp.ple.layer_multipliers",
                format!(
                    "expected {} entries, got {}",
                    self.ngram_size,
                    self.multipliers.len()
                ),
            ));
        }
        if self
            .multipliers
            .iter()
            .any(|&value| value % 2 == 0 || value > i64::MAX as u64)
        {
            return Err(invalid(
                "qwen4exp.ple.layer_multipliers",
                "hash multipliers must be odd positive i64 values",
            ));
        }
        let expected_offset = self.logical_row_count()?;
        if expected_offset > i32::MAX as u64 + 1 {
            return Err(invalid(
                "qwen4exp.ple.head_offsets",
                "table row ids do not fit the I32 gather contract",
            ));
        }
        Ok(())
    }

    /// Compute the bigram-through-N-gram row ids for one token.
    /// `history` is chronological and must exclude `current_token`.
    pub fn row_ids(&self, current_token: u32, history: &[u32]) -> Result<Vec<u32>, Qwen4ExpError> {
        let head_count = self.head_count()? as usize;
        if self.multipliers.len() != self.ngram_size as usize
            || self.head_offsets.len() != head_count
            || self.head_vocab_sizes.len() != head_count
        {
            return Err(invalid(
                "qwen4exp.ple.layer_multipliers",
                "PLE hash metadata has inconsistent lengths",
            ));
        }
        if current_token >= self.token_vocab_size
            || history.iter().any(|&token| token >= self.token_vocab_size)
        {
            return Err(invalid(
                "tokenizer.ggml.tokens",
                "PLE hash token is outside the tokenizer vocabulary",
            ));
        }

        let mut context = Vec::with_capacity(self.ngram_size as usize);
        context.push(current_token);
        let mut cut = false;
        for shift in 1..self.ngram_size as usize {
            let token = if cut || shift > history.len() {
                self.eos_token_id
            } else {
                history[history.len() - shift]
            };
            context.push(token);
            cut |= token == self.eos_token_id;
        }

        let mut rows = Vec::with_capacity(head_count);
        for ngram in 2..=self.ngram_size as usize {
            let mut mixed = (context[0] as i64).wrapping_mul(self.multipliers[0] as i64);
            for (&token, &multiplier) in context.iter().zip(&self.multipliers).take(ngram).skip(1) {
                mixed ^= (token as i64).wrapping_mul(multiplier as i64);
            }
            let first_head = (ngram - 2) * self.heads_per_ngram as usize;
            let end_head = first_head + self.heads_per_ngram as usize;
            for head in first_head..end_head {
                let size = i64::try_from(self.head_vocab_sizes[head]).map_err(|_| {
                    invalid(
                        "qwen4exp.ple.head_vocab_sizes",
                        format!("head {head} size does not fit i64"),
                    )
                })?;
                if size == 0 {
                    return Err(invalid(
                        "qwen4exp.ple.head_vocab_sizes",
                        format!("head {head} has zero rows"),
                    ));
                }
                let row = self.head_offsets[head]
                    .checked_add(mixed.rem_euclid(size) as u64)
                    .ok_or_else(|| invalid("qwen4exp.ple.head_offsets", "row id overflow"))?;
                rows.push(u32::try_from(row).map_err(|_| {
                    invalid(
                        "qwen4exp.ple.head_offsets",
                        format!("row id {row} does not fit u32"),
                    )
                })?);
            }
        }
        Ok(rows)
    }
}

/// Rolling PLE state for exactly one sequence. Clone this value to fork a
/// sequence; do not share one instance across interleaved sequences.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PleHistory {
    tokens: Vec<u32>,
    next_position: Option<u64>,
    contract: Option<(u32, u32, u32)>,
}

impl PleHistory {
    pub fn reset(&mut self) {
        self.tokens.clear();
        self.next_position = None;
    }

    pub fn row_ids_for_token(
        &mut self,
        config: &PleConfig,
        token: u32,
        position: u64,
    ) -> Result<Vec<u32>, Qwen4ExpError> {
        let (next, rows) = self.advanced(config, token, position)?;
        *self = next;
        Ok(rows)
    }

    pub fn advanced(
        &self,
        config: &PleConfig,
        token: u32,
        position: u64,
    ) -> Result<(Self, Vec<u32>), Qwen4ExpError> {
        let mut next = self.clone();
        let rows = next.advance_in_place(config, token, position)?;
        Ok((next, rows))
    }

    fn advance_in_place(
        &mut self,
        config: &PleConfig,
        token: u32,
        position: u64,
    ) -> Result<Vec<u32>, Qwen4ExpError> {
        let contract = (
            config.ngram_size,
            config.eos_token_id,
            config.token_vocab_size,
        );
        match self.contract {
            Some(existing) if existing != contract => {
                return Err(invalid(
                    "qwen4exp.ple.ngram_size",
                    "PLE history cannot change model contracts",
                ));
            }
            None => self.contract = Some(contract),
            Some(_) => {}
        }
        let next_position = position.checked_add(1).ok_or_else(|| {
            invalid(
                "qwen4exp.ple.layers",
                "sequence position cannot advance beyond u64::MAX",
            )
        })?;
        if self.next_position != Some(position) {
            self.tokens.clear();
        }
        let rows = config.row_ids(token, &self.tokens)?;
        self.tokens.push(token);
        let keep = config.ngram_size.saturating_sub(1) as usize;
        if self.tokens.len() > keep {
            self.tokens.drain(..self.tokens.len() - keep);
        }
        self.next_position = Some(next_position);
        Ok(rows)
    }

    pub fn prior_tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn next_position(&self) -> Option<u64> {
        self.next_position
    }
}

impl QsaConfig {
    pub fn block_budget(&self, compress_ratio: u32) -> Result<u32, Qwen4ExpError> {
        require_nonzero("qwen4exp.attention.compress_ratios", compress_ratio)?;
        if !self.token_budget.is_multiple_of(compress_ratio) {
            return Err(invalid(
                "qwen4exp.attention.indexer.top_k",
                "token budget must be divisible by the compression ratio",
            ));
        }
        Ok(self.token_budget / compress_ratio)
    }

    pub fn output_width(&self, compress_ratio: u32) -> Result<usize, Qwen4ExpError> {
        require_nonzero("qwen4exp.attention.compress_ratios", compress_ratio)?;
        let width = self
            .token_budget
            .checked_add(compress_ratio - 1)
            .ok_or_else(|| {
                invalid(
                    "qwen4exp.attention.indexer.top_k",
                    "selection width overflow",
                )
            })?;
        usize::try_from(width).map_err(|_| {
            invalid(
                "qwen4exp.attention.indexer.top_k",
                "selection width does not fit usize",
            )
        })
    }

    pub fn visible_complete_blocks(
        query_position: u64,
        sequence_length: u64,
        compress_ratio: u32,
    ) -> Result<u64, Qwen4ExpError> {
        require_nonzero("qwen4exp.attention.compress_ratios", compress_ratio)?;
        let visible_tokens = query_position.checked_add(1).ok_or_else(|| {
            invalid(
                "qwen4exp.attention.compress_ratios",
                "query position overflow",
            )
        })?;
        let ratio = compress_ratio as u64;
        Ok((visible_tokens / ratio).min(sequence_length / ratio))
    }

    /// Expand selected compressed blocks to full-resolution token positions and
    /// append the current incomplete block. Missing entries are `-1`.
    pub fn expand_block_indices(
        &self,
        selected_blocks: &[i32],
        query_position: u64,
        sequence_length: u64,
        compress_ratio: u32,
    ) -> Result<Vec<i32>, Qwen4ExpError> {
        if query_position >= sequence_length {
            return Err(invalid(
                "qwen4exp.attention.indexer.top_k",
                "query position must be inside the sequence",
            ));
        }
        let block_budget = self.block_budget(compress_ratio)? as usize;
        let width = self.output_width(compress_ratio)?;
        let visible_blocks =
            Self::visible_complete_blocks(query_position, sequence_length, compress_ratio)?;
        let ratio = compress_ratio as u64;
        let mut positions = Vec::with_capacity(width);
        if selected_blocks.len() != block_budget {
            return Err(invalid(
                "qwen4exp.attention.indexer.top_k",
                format!(
                    "expected {block_budget} selected block entries, got {}",
                    selected_blocks.len()
                ),
            ));
        }
        let mut seen = HashSet::with_capacity(block_budget);

        for &block in selected_blocks {
            if block == -1 {
                continue;
            }
            if block < 0 || block as u64 >= visible_blocks {
                return Err(invalid(
                    "qwen4exp.attention.indexer.top_k",
                    format!("selected block {block} is not visible"),
                ));
            }
            if !seen.insert(block) {
                return Err(invalid(
                    "qwen4exp.attention.indexer.top_k",
                    format!("selected block {block} is duplicated"),
                ));
            }
            let start = (block as u64).checked_mul(ratio).ok_or_else(|| {
                invalid(
                    "qwen4exp.attention.compress_ratios",
                    "expanded block position overflow",
                )
            })?;
            for offset in 0..ratio {
                let position = start + offset;
                if position < sequence_length {
                    positions.push(i32::try_from(position).map_err(|_| {
                        invalid(
                            "qwen4exp.attention.indexer.top_k",
                            "token position does not fit i32",
                        )
                    })?);
                }
            }
        }

        let visible_tokens = query_position + 1;
        let tail_start = visible_tokens / ratio * ratio;
        for position in tail_start..visible_tokens {
            if position < sequence_length {
                positions.push(i32::try_from(position).map_err(|_| {
                    invalid(
                        "qwen4exp.attention.indexer.top_k",
                        "tail position does not fit i32",
                    )
                })?);
            }
        }
        positions.truncate(width);
        positions.resize(width, -1);
        Ok(positions)
    }

    /// Score one pooled index key. ReLU is applied per query head before the
    /// head scores are summed.
    pub fn index_score(
        &self,
        query_heads: &[f32],
        pooled_key: &[f32],
    ) -> Result<f32, Qwen4ExpError> {
        let expected_query = (self.query_heads as usize)
            .checked_mul(self.head_dim as usize)
            .ok_or_else(|| {
                invalid(
                    "qwen4exp.attention.indexer.head_count",
                    "query shape overflow",
                )
            })?;
        if query_heads.len() != expected_query || pooled_key.len() != self.head_dim as usize {
            return Err(invalid(
                "qwen4exp.attention.indexer.key_length",
                format!(
                    "expected query/key lengths {expected_query}/{}, got {}/{}",
                    self.head_dim,
                    query_heads.len(),
                    pooled_key.len()
                ),
            ));
        }

        let dim = self.head_dim as usize;
        let mut score = 0.0_f32;
        for head in 0..self.query_heads as usize {
            let query = &query_heads[head * dim..(head + 1) * dim];
            let dot = query
                .iter()
                .zip(pooled_key)
                .map(|(&q, &k)| q * k)
                .sum::<f32>();
            score += dot.max(0.0);
        }
        Ok(score / (self.head_dim as f32).sqrt())
    }
}

fn required_u32(gguf: &GgufFile, key: &'static str) -> Result<u32, Qwen4ExpError> {
    let value = gguf
        .get_u64(key)
        .ok_or(Qwen4ExpError::MissingMetadata(key))?;
    u32::try_from(value).map_err(|_| invalid(key, format!("{value} does not fit u32")))
}

fn optional_u32(gguf: &GgufFile, key: &'static str) -> Result<Option<u32>, Qwen4ExpError> {
    gguf.get_u64(key)
        .map(|value| {
            u32::try_from(value).map_err(|_| invalid(key, format!("{value} does not fit u32")))
        })
        .transpose()
}

fn required_f32(gguf: &GgufFile, key: &'static str) -> Result<f32, Qwen4ExpError> {
    let value = gguf
        .get_f64(key)?
        .ok_or(Qwen4ExpError::MissingMetadata(key))?;
    if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
        return Err(invalid(key, format!("{value} does not fit finite f32")));
    }
    Ok(value as f32)
}

fn required_u64_array(gguf: &GgufFile, key: &'static str) -> Result<Vec<u64>, Qwen4ExpError> {
    gguf.get_u64_array(key)?
        .ok_or(Qwen4ExpError::MissingMetadata(key))
}

fn required_array_len_u32(gguf: &GgufFile, key: &'static str) -> Result<u32, Qwen4ExpError> {
    let length = gguf
        .get_array_len(key)?
        .ok_or(Qwen4ExpError::MissingMetadata(key))?;
    u32::try_from(length).map_err(|_| invalid(key, format!("length {length} does not fit u32")))
}

fn required_nonnegative_u32_array(
    gguf: &GgufFile,
    key: &'static str,
) -> Result<Vec<u32>, Qwen4ExpError> {
    let values = gguf
        .get_i64_array(key)?
        .ok_or(Qwen4ExpError::MissingMetadata(key))?;
    nonnegative_u32_values(key, values)
}

fn optional_nonnegative_u32_array(
    gguf: &GgufFile,
    key: &'static str,
) -> Result<Option<Vec<u32>>, Qwen4ExpError> {
    gguf.get_i64_array(key)?
        .map(|values| nonnegative_u32_values(key, values))
        .transpose()
}

fn nonnegative_u32_values(key: &'static str, values: Vec<i64>) -> Result<Vec<u32>, Qwen4ExpError> {
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            u32::try_from(value).map_err(|_| {
                invalid(
                    key,
                    format!("entry {index} value {value} does not fit nonnegative u32"),
                )
            })
        })
        .collect()
}

fn require_nonzero(key: &'static str, value: u32) -> Result<(), Qwen4ExpError> {
    if value == 0 {
        Err(invalid(key, "must be nonzero"))
    } else {
        Ok(())
    }
}

fn invalid(key: &'static str, reason: impl Into<String>) -> Qwen4ExpError {
    Qwen4ExpError::InvalidMetadata {
        key,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_contract_is_self_consistent() {
        let config = Qwen4ExpConfig::flash_next_reference();
        config.validate().unwrap();
        assert_eq!(config.qsa_layer_count(), 12);
        assert_eq!(config.mixer_kind(0), Some(MixerKind::GatedDeltaNet));
        assert_eq!(config.mixer_kind(3), Some(MixerKind::QwenSparseAttention));
        assert_eq!(config.ple.as_ref().unwrap().head_count().unwrap(), 16);
        assert_eq!(config.qsa.block_budget(4).unwrap(), 512);
        assert_eq!(config.qsa.output_width(4).unwrap(), 2_051);
    }

    #[test]
    fn ple_hash_matches_released_checkpoint_vector() {
        let ple = Qwen4ExpConfig::flash_next_reference().ple.unwrap();
        let rows = ple.row_ids(42, &[]).unwrap();
        assert_eq!(
            rows,
            [
                4_064_167,
                21_428_206,
                43_851_727,
                63_435_096,
                89_826_284,
                106_088_217,
                121_229_381,
                156_147_803,
                175_176_707,
                188_463_481,
                216_828_809,
                228_933_352,
                248_448_227,
                269_188_956,
                283_041_664,
                314_347_972,
            ]
        );
    }

    #[test]
    fn ple_eos_cuts_only_following_tokens() {
        let ple = Qwen4ExpConfig::flash_next_reference().ple.unwrap();
        let eos = ple.eos_token_id;
        let eos_with_context = ple.row_ids(eos, &[42, 1_337]).unwrap();
        let eos_without_context = ple.row_ids(eos, &[]).unwrap();
        assert_ne!(eos_with_context, eos_without_context);

        let after_eos = ple.row_ids(7, &[42, eos]).unwrap();
        let fresh = ple.row_ids(7, &[]).unwrap();
        assert_eq!(after_eos, fresh);
    }

    #[test]
    fn ple_history_resets_on_position_discontinuity() {
        let ple = Qwen4ExpConfig::flash_next_reference().ple.unwrap();
        let mut history = PleHistory::default();
        history.row_ids_for_token(&ple, 42, 0).unwrap();
        history.row_ids_for_token(&ple, 1_337, 1).unwrap();

        let discontinuous = history.row_ids_for_token(&ple, 7, 10).unwrap();
        assert_eq!(discontinuous, ple.row_ids(7, &[]).unwrap());
        assert_eq!(history.prior_tokens(), &[7]);
        assert_eq!(history.next_position(), Some(11));
    }

    #[test]
    fn ple_hash_rejects_tokens_outside_the_signed_hash_contract() {
        let ple = Qwen4ExpConfig::flash_next_reference().ple.unwrap();
        assert!(ple.row_ids(ple.token_vocab_size, &[]).is_err());
        assert!(ple.row_ids(7, &[ple.token_vocab_size]).is_err());
    }

    #[test]
    fn ple_history_forks_without_mutating_the_parent() {
        let ple = Qwen4ExpConfig::flash_next_reference().ple.unwrap();
        let mut history = PleHistory::default();
        history.row_ids_for_token(&ple, 42, 0).unwrap();
        let (fork, rows) = history.advanced(&ple, 1_337, 1).unwrap();
        assert_eq!(rows, ple.row_ids(1_337, &[42]).unwrap());
        assert_eq!(history.prior_tokens(), &[42]);
        assert_eq!(fork.prior_tokens(), &[42, 1_337]);
    }

    #[test]
    fn ple_history_failure_is_transactional() {
        let ple = Qwen4ExpConfig::flash_next_reference().ple.unwrap();
        let mut history = PleHistory::default();
        history.row_ids_for_token(&ple, 42, 0).unwrap();
        let before = history.clone();
        assert!(
            history
                .row_ids_for_token(&ple, ple.token_vocab_size, 10)
                .is_err()
        );
        assert_eq!(history, before);
    }

    #[test]
    fn qsa_expands_blocks_and_appends_incomplete_tail() {
        let qsa = QsaConfig {
            query_heads: 4,
            key_heads: 1,
            head_dim: 128,
            token_budget: 8,
        };
        let positions = qsa.expand_block_indices(&[1, 0], 10, 20, 4).unwrap();
        assert_eq!(positions, [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10]);
    }

    #[test]
    fn qsa_handles_complete_boundaries_and_rejects_bad_topk_rows() {
        let qsa = QsaConfig {
            query_heads: 4,
            key_heads: 1,
            head_dim: 128,
            token_budget: 8,
        };
        assert_eq!(
            qsa.expand_block_indices(&[-1, -1], 2, 20, 4).unwrap(),
            [0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1]
        );
        assert_eq!(
            qsa.expand_block_indices(&[0, -1], 3, 20, 4).unwrap(),
            [0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1]
        );
        assert!(qsa.expand_block_indices(&[0], 10, 20, 4).is_err());
        assert!(qsa.expand_block_indices(&[0, 0], 10, 20, 4).is_err());
        assert!(qsa.expand_block_indices(&[2, -1], 10, 20, 4).is_err());
    }

    #[test]
    fn qsa_rectifies_each_head_before_summing() {
        let qsa = QsaConfig {
            query_heads: 2,
            key_heads: 1,
            head_dim: 2,
            token_budget: 8,
        };
        let score = qsa
            .index_score(&[1.0, 0.0, -1.0, 0.0], &[2.0, 0.0])
            .unwrap();
        assert!((score - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn malformed_schedule_is_rejected() {
        let mut config = Qwen4ExpConfig::flash_next_reference();
        config.compress_ratios[0] = 4;
        assert!(config.validate().is_err());
    }

    #[test]
    fn ple_is_optional_but_qsa_rope_width_is_not() {
        let mut config = Qwen4ExpConfig::flash_next_reference();
        config.ple = None;
        config.validate().unwrap();
        config.qsa.head_dim = config.attention.rotary_dim - 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn ple_layer_order_is_irrelevant_but_duplicates_are_rejected() {
        let mut config = Qwen4ExpConfig::flash_next_reference();
        let ple = config.ple.as_mut().unwrap();
        ple.layers = vec![2, 1];
        config.validate().unwrap();
        config.ple.as_mut().unwrap().layers = vec![1, 1];
        assert!(config.validate().is_err());
    }
}
