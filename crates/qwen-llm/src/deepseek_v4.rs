//! Strict DeepSeek V4 GGUF schema binding.
//!
//! This module deliberately stops before execution. It establishes a separate
//! model-family boundary, validates the frozen 0731 target schema, and binds
//! every target tensor without widening the Qwen-specific forward types.

use crate::gguf::GgufFile;
use crate::tensor::{GgmlType, TensorDesc};
use std::collections::HashSet;

const ARCHITECTURE: &str = "deepseek4";
const FLASH_0731_FULL_EXPERT_COUNT: u32 = 256;
const FLASH_0731_REAP_K160_EXPERT_COUNT: u32 = 160;
const FLASH_0731_REAP_K216_EXPERT_COUNT: u32 = 216;

pub(crate) fn flash_0731_expert_count_supported(expert_count: u32) -> bool {
    matches!(
        expert_count,
        FLASH_0731_FULL_EXPERT_COUNT
            | FLASH_0731_REAP_K160_EXPERT_COUNT
            | FLASH_0731_REAP_K216_EXPERT_COUNT
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionKind {
    SlidingWindow,
    CompressedSparse,
    HeavilyCompressed,
}

impl AttentionKind {
    pub fn from_ratio(layer: usize, ratio: u64) -> Result<Self, DeepSeekV4Error> {
        match ratio {
            0 => Ok(Self::SlidingWindow),
            4 => Ok(Self::CompressedSparse),
            128 => Ok(Self::HeavilyCompressed),
            _ => Err(DeepSeekV4Error::InvalidMetadata {
                key: "deepseek4.attention.compress_ratios".into(),
                detail: format!("layer {layer} has unsupported ratio {ratio}"),
            }),
        }
    }

    pub fn ratio(self) -> u32 {
        match self {
            Self::SlidingWindow => 0,
            Self::CompressedSparse => 4,
            Self::HeavilyCompressed => 128,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekV4Config {
    pub layer_count: u32,
    pub context_length: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub attention_head_count: u32,
    pub kv_head_count: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub rope_dimension_count: u32,
    pub rope_freq_base: f32,
    pub attention_rms_epsilon: f32,
    pub rope_scaling_type: String,
    pub rope_scaling_factor: f32,
    pub rope_original_context_length: u32,
    pub rope_yarn_beta_fast: f32,
    pub rope_yarn_beta_slow: f32,
    pub q_lora_rank: u32,
    pub sliding_window: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_feed_forward_length: u32,
    pub shared_expert_count: u32,
    pub expert_weights_scale: f32,
    pub expert_weights_norm: bool,
    pub expert_gating_func: u32,
    pub indexer_head_count: u32,
    pub indexer_key_length: u32,
    pub indexer_top_k: u32,
    pub output_group_count: u32,
    pub output_lora_rank: u32,
    pub attention_kinds: Vec<AttentionKind>,
    pub compress_ratio_tail: Vec<u32>,
    pub compress_rope_freq_base: f32,
    pub hyper_connection_count: u32,
    pub sinkhorn_iterations: u32,
    pub hyper_connection_epsilon: f32,
    pub hash_layer_count: u32,
    pub swiglu_clamp_experts: Vec<f32>,
    pub swiglu_clamp_shared: Vec<f32>,
    pub tokenizer_model: String,
    pub tokenizer_pre: String,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    pub tokenizer_token_type_count: usize,
    pub tokenizer_merge_count: usize,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
}

impl DeepSeekV4Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, DeepSeekV4Error> {
        let architecture = gguf.architecture();
        if architecture.as_deref() != Some(ARCHITECTURE) {
            return Err(DeepSeekV4Error::UnsupportedArchitecture(architecture));
        }

        let layer_count = required_u32(gguf, "deepseek4.block_count")?;
        let context_length = required_u32(gguf, "deepseek4.context_length")?;
        let hidden_size = required_u32(gguf, "deepseek4.embedding_length")?;
        let attention_head_count = required_u32(gguf, "deepseek4.attention.head_count")?;
        let kv_head_count = required_u32(gguf, "deepseek4.attention.head_count_kv")?;
        let key_length = required_u32(gguf, "deepseek4.attention.key_length")?;
        let value_length = required_u32(gguf, "deepseek4.attention.value_length")?;
        let rope_dimension_count = required_u32(gguf, "deepseek4.rope.dimension_count")?;
        let rope_freq_base = required_f32(gguf, "deepseek4.rope.freq_base")?;
        let attention_rms_epsilon =
            required_f32(gguf, "deepseek4.attention.layer_norm_rms_epsilon")?;
        let rope_scaling_type = required_str(gguf, "deepseek4.rope.scaling.type")?.to_string();
        let rope_scaling_factor = required_f32(gguf, "deepseek4.rope.scaling.factor")?;
        let rope_original_context_length =
            required_u32(gguf, "deepseek4.rope.scaling.original_context_length")?;
        let rope_yarn_beta_fast = required_f32(gguf, "deepseek4.rope.scaling.yarn_beta_fast")?;
        let rope_yarn_beta_slow = required_f32(gguf, "deepseek4.rope.scaling.yarn_beta_slow")?;
        let q_lora_rank = required_u32(gguf, "deepseek4.attention.q_lora_rank")?;
        let sliding_window = required_u32(gguf, "deepseek4.attention.sliding_window")?;
        let expert_count = required_u32(gguf, "deepseek4.expert_count")?;
        let expert_used_count = required_u32(gguf, "deepseek4.expert_used_count")?;
        let expert_feed_forward_length =
            required_u32(gguf, "deepseek4.expert_feed_forward_length")?;
        let shared_expert_count = required_u32(gguf, "deepseek4.expert_shared_count")?;
        let expert_weights_scale = required_f32(gguf, "deepseek4.expert_weights_scale")?;
        let expert_weights_norm = required_bool(gguf, "deepseek4.expert_weights_norm")?;
        let expert_gating_func = required_u32(gguf, "deepseek4.expert_gating_func")?;
        let indexer_head_count = required_u32(gguf, "deepseek4.attention.indexer.head_count")?;
        let indexer_key_length = required_u32(gguf, "deepseek4.attention.indexer.key_length")?;
        let indexer_top_k = required_u32(gguf, "deepseek4.attention.indexer.top_k")?;
        let output_group_count = required_u32(gguf, "deepseek4.attention.output_group_count")?;
        let output_lora_rank = required_u32(gguf, "deepseek4.attention.output_lora_rank")?;
        let compress_rope_freq_base =
            required_f32(gguf, "deepseek4.attention.compress_rope_freq_base")?;
        let hyper_connection_count = required_u32(gguf, "deepseek4.hyper_connection.count")?;
        let sinkhorn_iterations =
            required_u32(gguf, "deepseek4.hyper_connection.sinkhorn_iterations")?;
        let hyper_connection_epsilon = required_f32(gguf, "deepseek4.hyper_connection.epsilon")?;
        let hash_layer_count = required_u32(gguf, "deepseek4.hash_layer_count")?;

        let ratios = gguf
            .get_u64_array("deepseek4.attention.compress_ratios")?
            .ok_or_else(|| {
                DeepSeekV4Error::MissingMetadata("deepseek4.attention.compress_ratios".into())
            })?;
        if ratios.len() < layer_count as usize {
            return Err(DeepSeekV4Error::InvalidMetadata {
                key: "deepseek4.attention.compress_ratios".into(),
                detail: format!(
                    "has {} entries for {layer_count} target layers",
                    ratios.len()
                ),
            });
        }
        let attention_kinds = ratios[..layer_count as usize]
            .iter()
            .copied()
            .enumerate()
            .map(|(layer, ratio)| AttentionKind::from_ratio(layer, ratio))
            .collect::<Result<Vec<_>, _>>()?;
        let compress_ratio_tail = ratios[layer_count as usize..]
            .iter()
            .enumerate()
            .map(|(idx, &ratio)| {
                u32::try_from(ratio).map_err(|_| DeepSeekV4Error::InvalidMetadata {
                    key: "deepseek4.attention.compress_ratios".into(),
                    detail: format!("trailing entry {idx} exceeds u32"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let swiglu_clamp_experts = required_f32_array(gguf, "deepseek4.swiglu_clamp_exp")?;
        let swiglu_clamp_shared = optional_f32_array(gguf, "deepseek4.swiglu_clamp_shexp")?
            .unwrap_or_else(|| swiglu_clamp_experts.clone());
        validate_layer_array(
            "deepseek4.swiglu_clamp_exp",
            &swiglu_clamp_experts,
            layer_count,
        )?;
        validate_layer_array(
            "deepseek4.swiglu_clamp_shexp",
            &swiglu_clamp_shared,
            layer_count,
        )?;

        let token_embd = gguf
            .find("token_embd.weight")
            .ok_or_else(|| DeepSeekV4Error::MissingTensor("token_embd.weight".into()))?;
        if token_embd.shape.len() != 2 || token_embd.shape[0] != hidden_size as u64 {
            return Err(DeepSeekV4Error::ShapeMismatch {
                tensor: token_embd.name.clone(),
                expected: vec![hidden_size as u64, 0],
                actual: token_embd.shape.clone(),
            });
        }
        let vocab_size =
            u32::try_from(token_embd.shape[1]).map_err(|_| DeepSeekV4Error::InvalidMetadata {
                key: "token_embd.weight".into(),
                detail: format!("vocabulary dimension {} exceeds u32", token_embd.shape[1]),
            })?;
        if let Some(declared_vocab) = gguf.get_u64("deepseek4.vocab_size")
            && declared_vocab != vocab_size as u64
        {
            return Err(DeepSeekV4Error::InvalidMetadata {
                key: "deepseek4.vocab_size".into(),
                detail: format!("declares {declared_vocab}, embedding has {vocab_size} rows"),
            });
        }
        let tokenizer_vocab = gguf
            .get_array_len("tokenizer.ggml.tokens")?
            .ok_or_else(|| DeepSeekV4Error::MissingMetadata("tokenizer.ggml.tokens".into()))?;
        if tokenizer_vocab != vocab_size as usize {
            return Err(DeepSeekV4Error::InvalidMetadata {
                key: "tokenizer.ggml.tokens".into(),
                detail: format!("has {tokenizer_vocab} entries, embedding has {vocab_size} rows"),
            });
        }

        let tokenizer_model = required_str(gguf, "tokenizer.ggml.model")?.to_string();
        let tokenizer_pre = required_str(gguf, "tokenizer.ggml.pre")?.to_string();
        let tokenizer_token_type_count = gguf
            .get_array_len("tokenizer.ggml.token_type")?
            .ok_or_else(|| DeepSeekV4Error::MissingMetadata("tokenizer.ggml.token_type".into()))?;
        let tokenizer_merge_count = gguf
            .get_array_len("tokenizer.ggml.merges")?
            .ok_or_else(|| DeepSeekV4Error::MissingMetadata("tokenizer.ggml.merges".into()))?;

        let config = Self {
            layer_count,
            context_length,
            hidden_size,
            vocab_size,
            attention_head_count,
            kv_head_count,
            key_length,
            value_length,
            rope_dimension_count,
            rope_freq_base,
            attention_rms_epsilon,
            rope_scaling_type,
            rope_scaling_factor,
            rope_original_context_length,
            rope_yarn_beta_fast,
            rope_yarn_beta_slow,
            q_lora_rank,
            sliding_window,
            expert_count,
            expert_used_count,
            expert_feed_forward_length,
            shared_expert_count,
            expert_weights_scale,
            expert_weights_norm,
            expert_gating_func,
            indexer_head_count,
            indexer_key_length,
            indexer_top_k,
            output_group_count,
            output_lora_rank,
            attention_kinds,
            compress_ratio_tail,
            compress_rope_freq_base,
            hyper_connection_count,
            sinkhorn_iterations,
            hyper_connection_epsilon,
            hash_layer_count,
            swiglu_clamp_experts,
            swiglu_clamp_shared,
            tokenizer_model,
            tokenizer_pre,
            bos_token_id: optional_u32(gguf, "tokenizer.ggml.bos_token_id")?,
            eos_token_id: optional_u32(gguf, "tokenizer.ggml.eos_token_id")?,
            padding_token_id: optional_u32(gguf, "tokenizer.ggml.padding_token_id")?,
            tokenizer_token_type_count,
            tokenizer_merge_count,
            add_bos_token: optional_bool(gguf, "tokenizer.ggml.add_bos_token")?.unwrap_or(false),
            add_eos_token: optional_bool(gguf, "tokenizer.ggml.add_eos_token")?.unwrap_or(false),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn attention_counts(&self) -> (usize, usize, usize) {
        self.attention_kinds
            .iter()
            .fold((0, 0, 0), |(local, csa, hca), kind| match kind {
                AttentionKind::SlidingWindow => (local + 1, csa, hca),
                AttentionKind::CompressedSparse => (local, csa + 1, hca),
                AttentionKind::HeavilyCompressed => (local, csa, hca + 1),
            })
    }

    fn validate(&self) -> Result<(), DeepSeekV4Error> {
        require_nonzero("deepseek4.block_count", self.layer_count)?;
        require_nonzero("deepseek4.context_length", self.context_length)?;
        require_nonzero("deepseek4.embedding_length", self.hidden_size)?;
        require_nonzero("deepseek4.attention.head_count", self.attention_head_count)?;
        require_nonzero("deepseek4.attention.key_length", self.key_length)?;
        require_nonzero("deepseek4.attention.value_length", self.value_length)?;
        require_nonzero("deepseek4.rope.dimension_count", self.rope_dimension_count)?;
        require_nonzero("deepseek4.attention.q_lora_rank", self.q_lora_rank)?;
        require_nonzero("deepseek4.attention.sliding_window", self.sliding_window)?;
        require_nonzero("deepseek4.expert_count", self.expert_count)?;
        require_nonzero("deepseek4.expert_used_count", self.expert_used_count)?;
        require_nonzero(
            "deepseek4.expert_feed_forward_length",
            self.expert_feed_forward_length,
        )?;
        require_nonzero("deepseek4.expert_shared_count", self.shared_expert_count)?;
        require_nonzero(
            "deepseek4.attention.indexer.head_count",
            self.indexer_head_count,
        )?;
        require_nonzero(
            "deepseek4.attention.indexer.key_length",
            self.indexer_key_length,
        )?;
        require_nonzero("deepseek4.attention.indexer.top_k", self.indexer_top_k)?;
        require_nonzero(
            "deepseek4.attention.output_lora_rank",
            self.output_lora_rank,
        )?;
        require_nonzero(
            "deepseek4.hyper_connection.count",
            self.hyper_connection_count,
        )?;
        require_nonzero(
            "deepseek4.hyper_connection.sinkhorn_iterations",
            self.sinkhorn_iterations,
        )?;
        require_positive_f32("deepseek4.rope.freq_base", self.rope_freq_base)?;
        require_positive_f32(
            "deepseek4.attention.layer_norm_rms_epsilon",
            self.attention_rms_epsilon,
        )?;
        require_positive_f32("deepseek4.rope.scaling.factor", self.rope_scaling_factor)?;
        require_positive_f32(
            "deepseek4.rope.scaling.yarn_beta_fast",
            self.rope_yarn_beta_fast,
        )?;
        require_positive_f32(
            "deepseek4.rope.scaling.yarn_beta_slow",
            self.rope_yarn_beta_slow,
        )?;
        require_positive_f32(
            "deepseek4.attention.compress_rope_freq_base",
            self.compress_rope_freq_base,
        )?;
        require_positive_f32(
            "deepseek4.hyper_connection.epsilon",
            self.hyper_connection_epsilon,
        )?;
        require_positive_f32("deepseek4.expert_weights_scale", self.expert_weights_scale)?;
        if self.kv_head_count != 1 {
            return invalid("deepseek4.attention.head_count_kv", "must be one");
        }
        if self.key_length != self.value_length {
            return invalid(
                "deepseek4.attention.key_length/value_length",
                "shared K=V dimensions differ",
            );
        }
        if self.rope_dimension_count > self.key_length {
            return invalid(
                "deepseek4.rope.dimension_count",
                "exceeds attention key length",
            );
        }
        if !self.rope_dimension_count.is_multiple_of(2) {
            return invalid("deepseek4.rope.dimension_count", "must be even");
        }
        if self.rope_dimension_count > self.indexer_key_length {
            return invalid(
                "deepseek4.rope.dimension_count",
                "exceeds indexer key length",
            );
        }
        if self.rope_original_context_length > self.context_length {
            return invalid(
                "deepseek4.rope.scaling.original_context_length",
                "exceeds context length",
            );
        }
        if self.expert_used_count > self.expert_count {
            return invalid("deepseek4.expert_used_count", "exceeds expert count");
        }
        if self.hash_layer_count > self.layer_count {
            return invalid("deepseek4.hash_layer_count", "exceeds block count");
        }
        if self.output_group_count == 0
            || !self
                .attention_head_count
                .is_multiple_of(self.output_group_count)
        {
            return invalid(
                "deepseek4.attention.output_group_count",
                "must divide attention head count",
            );
        }
        if self.expert_gating_func != 4 {
            return invalid(
                "deepseek4.expert_gating_func",
                "only sqrt-softplus gating function 4 is supported",
            );
        }
        if self.attention_kinds.len() != self.layer_count as usize {
            return invalid(
                "deepseek4.attention.compress_ratios",
                "target schedule length differs from block count",
            );
        }
        if self.tokenizer_token_type_count != self.vocab_size as usize {
            return invalid(
                "tokenizer.ggml.token_type",
                "length differs from vocabulary size",
            );
        }
        if self.tokenizer_merge_count == 0 {
            return invalid("tokenizer.ggml.merges", "must be nonempty");
        }
        for (key, id) in [
            ("tokenizer.ggml.bos_token_id", self.bos_token_id),
            ("tokenizer.ggml.eos_token_id", self.eos_token_id),
            ("tokenizer.ggml.padding_token_id", self.padding_token_id),
        ] {
            if id.is_some_and(|id| id >= self.vocab_size) {
                return invalid(key, "token id is outside the vocabulary");
            }
        }
        if self.add_bos_token && self.bos_token_id.is_none() {
            return invalid(
                "tokenizer.ggml.add_bos_token",
                "enabled without a BOS token id",
            );
        }
        if self.add_eos_token && self.eos_token_id.is_none() {
            return invalid(
                "tokenizer.ggml.add_eos_token",
                "enabled without an EOS token id",
            );
        }
        if self
            .swiglu_clamp_experts
            .iter()
            .chain(&self.swiglu_clamp_shared)
            .any(|&value| !value.is_finite() || value <= 0.0)
        {
            return invalid("deepseek4.swiglu_clamp_*", "clamps must be positive");
        }
        Ok(())
    }

    pub fn validate_flash_0731_profile(&self) -> Result<(), DeepSeekV4Error> {
        macro_rules! exact {
            ($field:ident, $expected:expr) => {
                if self.$field != $expected {
                    return Err(DeepSeekV4Error::ProfileMismatch {
                        field: stringify!($field),
                        expected: format!("{:?}", $expected),
                        actual: format!("{:?}", self.$field),
                    });
                }
            };
        }

        exact!(layer_count, 43);
        exact!(context_length, 1_048_576);
        exact!(hidden_size, 4_096);
        exact!(vocab_size, 129_280);
        exact!(attention_head_count, 64);
        exact!(kv_head_count, 1);
        exact!(key_length, 512);
        exact!(value_length, 512);
        exact!(rope_dimension_count, 64);
        exact!(rope_freq_base, 10_000.0);
        exact!(attention_rms_epsilon, 1e-6);
        exact!(rope_scaling_type, "yarn");
        exact!(rope_scaling_factor, 16.0);
        exact!(rope_original_context_length, 65_536);
        exact!(rope_yarn_beta_fast, 32.0);
        exact!(rope_yarn_beta_slow, 1.0);
        exact!(q_lora_rank, 1_024);
        exact!(sliding_window, 128);
        if !flash_0731_expert_count_supported(self.expert_count) {
            return Err(DeepSeekV4Error::ProfileMismatch {
                field: "expert_count",
                expected: "160, 216, or 256".into(),
                actual: self.expert_count.to_string(),
            });
        }
        exact!(expert_used_count, 6);
        exact!(expert_feed_forward_length, 2_048);
        exact!(shared_expert_count, 1);
        exact!(expert_weights_scale, 1.5);
        exact!(expert_weights_norm, true);
        exact!(expert_gating_func, 4);
        exact!(indexer_head_count, 64);
        exact!(indexer_key_length, 128);
        exact!(indexer_top_k, 512);
        exact!(output_group_count, 8);
        exact!(output_lora_rank, 1_024);
        exact!(compress_rope_freq_base, 160_000.0);
        exact!(hyper_connection_count, 4);
        exact!(sinkhorn_iterations, 20);
        exact!(hyper_connection_epsilon, 1e-6);
        exact!(hash_layer_count, 3);
        exact!(tokenizer_model, "gpt2");
        exact!(tokenizer_pre, "joyai-llm");
        exact!(bos_token_id, Some(0));
        exact!(eos_token_id, Some(1));
        // AtomicChat's AD-* line uses Some(1) (same as eos_token_id); Unsloth's UD-* uses Some(2).
        // padding_token_id is not used at inference time, so accept either.
        if !matches!(self.padding_token_id, Some(1) | Some(2)) {
            return Err(DeepSeekV4Error::ProfileMismatch {
                field: "padding_token_id",
                expected: "Some(1) or Some(2)".into(),
                actual: format!("{:?}", self.padding_token_id),
            });
        }
        exact!(tokenizer_token_type_count, 129_280);
        exact!(tokenizer_merge_count, 127_741);
        exact!(add_bos_token, false);
        exact!(add_eos_token, false);
        if self.compress_ratio_tail.as_slice() != [0, 0, 0] {
            return Err(DeepSeekV4Error::ProfileMismatch {
                field: "compress_ratio_tail",
                expected: "[0, 0, 0]".into(),
                actual: format!("{:?}", self.compress_ratio_tail),
            });
        }
        for (layer, &actual) in self.attention_kinds.iter().enumerate() {
            let expected = flash_0731_attention_kind(layer);
            if actual != expected {
                return Err(DeepSeekV4Error::ProfileMismatch {
                    field: "attention_kinds",
                    expected: format!("layer {layer}: {expected:?}"),
                    actual: format!("layer {layer}: {actual:?}"),
                });
            }
        }
        if self
            .swiglu_clamp_experts
            .iter()
            .chain(&self.swiglu_clamp_shared)
            .any(|&value| value != 10.0)
        {
            return Err(DeepSeekV4Error::ProfileMismatch {
                field: "swiglu_clamp",
                expected: "10.0 for every target layer".into(),
                actual: "one or more values differ".into(),
            });
        }
        Ok(())
    }
}

fn flash_0731_attention_kind(layer: usize) -> AttentionKind {
    if layer < 2 {
        AttentionKind::SlidingWindow
    } else if layer.is_multiple_of(2) {
        AttentionKind::CompressedSparse
    } else {
        AttentionKind::HeavilyCompressed
    }
}

#[derive(Clone, Debug)]
pub struct HyperConnectionWeights<'a> {
    pub function: &'a TensorDesc,
    pub scale: &'a TensorDesc,
    pub base: &'a TensorDesc,
}

#[derive(Clone, Debug)]
pub struct CompressorWeights<'a> {
    pub kv: &'a TensorDesc,
    pub gate: &'a TensorDesc,
    pub ape: &'a TensorDesc,
    pub norm: &'a TensorDesc,
}

#[derive(Clone, Debug)]
pub struct IndexerWeights<'a> {
    pub q: &'a TensorDesc,
    pub projection: &'a TensorDesc,
    pub compressor: CompressorWeights<'a>,
}

#[derive(Clone, Debug)]
pub enum AttentionLane<'a> {
    SlidingWindow,
    CompressedSparse {
        compressor: CompressorWeights<'a>,
        indexer: IndexerWeights<'a>,
    },
    HeavilyCompressed {
        compressor: CompressorWeights<'a>,
    },
}

#[derive(Clone, Debug)]
pub struct AttentionWeights<'a> {
    pub norm: &'a TensorDesc,
    pub sinks: &'a TensorDesc,
    pub q_a: &'a TensorDesc,
    pub q_a_norm: &'a TensorDesc,
    pub q_b: &'a TensorDesc,
    pub kv: &'a TensorDesc,
    pub kv_norm: &'a TensorDesc,
    pub output_a: &'a TensorDesc,
    pub output_b: &'a TensorDesc,
    pub lane: AttentionLane<'a>,
}

#[derive(Clone, Debug)]
pub enum RouterWeights<'a> {
    TokenHash { token_to_expert: &'a TensorDesc },
    Learned { correction_bias: &'a TensorDesc },
}

#[derive(Clone, Debug)]
pub struct MoeWeights<'a> {
    pub gate_input: &'a TensorDesc,
    pub gate_experts: &'a TensorDesc,
    pub up_experts: &'a TensorDesc,
    pub down_experts: &'a TensorDesc,
    pub gate_shared: &'a TensorDesc,
    pub up_shared: &'a TensorDesc,
    pub down_shared: &'a TensorDesc,
    pub router: RouterWeights<'a>,
}

#[derive(Clone, Debug)]
pub struct DeepSeekV4Block<'a> {
    pub attention: AttentionWeights<'a>,
    pub attention_hyper_connection: HyperConnectionWeights<'a>,
    pub ffn_norm: &'a TensorDesc,
    pub moe: MoeWeights<'a>,
    pub ffn_hyper_connection: HyperConnectionWeights<'a>,
}

#[derive(Debug)]
pub struct DeepSeekV4Model<'a> {
    pub config: DeepSeekV4Config,
    pub token_embedding: &'a TensorDesc,
    pub output_norm: &'a TensorDesc,
    pub output: &'a TensorDesc,
    pub output_hyper_connection: HyperConnectionWeights<'a>,
    pub blocks: Vec<DeepSeekV4Block<'a>>,
    pub source_tensor_count: usize,
}

impl<'a> DeepSeekV4Model<'a> {
    pub fn from_gguf_flash_0731(gguf: &'a GgufFile) -> Result<Self, DeepSeekV4Error> {
        let config = DeepSeekV4Config::from_gguf(gguf)?;
        config.validate_flash_0731_profile()?;
        let mut expected = HashSet::with_capacity(gguf.tensors.len());
        let h = config.hidden_size as u64;
        let vocab = config.vocab_size as u64;
        let hc = config.hyper_connection_count as u64;
        let hc_width = checked_mul(h, hc, "hidden_size * hyper_connection_count")?;
        let hc_params = checked_mul(hc, hc + 2, "hyper-connection parameter count")?;

        let token_embedding = bind(gguf, &mut expected, "token_embd.weight", &[h, vocab])?;
        let output_norm = bind(gguf, &mut expected, "output_norm.weight", &[h])?;
        let output = bind(gguf, &mut expected, "output.weight", &[h, vocab])?;
        let output_hyper_connection = HyperConnectionWeights {
            function: bind(gguf, &mut expected, "output_hc_fn.weight", &[hc_width, hc])?,
            scale: bind(gguf, &mut expected, "output_hc_scale.weight", &[1])?,
            base: bind(gguf, &mut expected, "output_hc_base.weight", &[hc])?,
        };

        let q_width = checked_mul(
            config.attention_head_count as u64,
            config.key_length as u64,
            "attention_head_count * key_length",
        )?;
        let output_low_rank_width = checked_mul(
            config.output_group_count as u64,
            config.output_lora_rank as u64,
            "output_group_count * output_lora_rank",
        )?;
        let output_group_input_width = grouped_output_input_width(
            config.attention_head_count,
            config.key_length,
            config.output_group_count,
        )?;
        let shared_width = checked_mul(
            config.shared_expert_count as u64,
            config.expert_feed_forward_length as u64,
            "shared_expert_count * expert_feed_forward_length",
        )?;
        let mut blocks = Vec::with_capacity(config.layer_count as usize);

        for layer in 0..config.layer_count as usize {
            let prefix = format!("blk.{layer}");
            let attention = AttentionWeights {
                norm: bind_owned(
                    gguf,
                    &mut expected,
                    format!("{prefix}.attn_norm.weight"),
                    &[h],
                )?,
                sinks: bind_owned(
                    gguf,
                    &mut expected,
                    format!("{prefix}.attn_sinks.weight"),
                    &[config.attention_head_count as u64],
                )?,
                q_a: bind_owned(
                    gguf,
                    &mut expected,
                    format!("{prefix}.attn_q_a.weight"),
                    &[h, config.q_lora_rank as u64],
                )?,
                q_a_norm: bind_owned(
                    gguf,
                    &mut expected,
                    format!("{prefix}.attn_q_a_norm.weight"),
                    &[config.q_lora_rank as u64],
                )?,
                q_b: bind_owned(
                    gguf,
                    &mut expected,
                    format!("{prefix}.attn_q_b.weight"),
                    &[config.q_lora_rank as u64, q_width],
                )?,
                kv: bind_owned(
                    gguf,
                    &mut expected,
                    format!("{prefix}.attn_kv.weight"),
                    &[h, config.key_length as u64],
                )?,
                kv_norm: bind_owned(
                    gguf,
                    &mut expected,
                    format!("{prefix}.attn_kv_a_norm.weight"),
                    &[config.key_length as u64],
                )?,
                output_a: bind_owned(
                    gguf,
                    &mut expected,
                    format!("{prefix}.attn_output_a.weight"),
                    &[output_group_input_width, output_low_rank_width],
                )?,
                output_b: bind_owned(
                    gguf,
                    &mut expected,
                    format!("{prefix}.attn_output_b.weight"),
                    &[output_low_rank_width, h],
                )?,
                lane: bind_attention_lane(gguf, &mut expected, &config, layer, &prefix)?,
            };
            let attention_hyper_connection =
                bind_hyper_connection(gguf, &mut expected, &prefix, "attn", hc_width, hc_params)?;
            let ffn_norm = bind_owned(
                gguf,
                &mut expected,
                format!("{prefix}.ffn_norm.weight"),
                &[h],
            )?;
            let gate_input = bind_owned(
                gguf,
                &mut expected,
                format!("{prefix}.ffn_gate_inp.weight"),
                &[h, config.expert_count as u64],
            )?;
            let gate_experts = bind_owned(
                gguf,
                &mut expected,
                format!("{prefix}.ffn_gate_exps.weight"),
                &[
                    h,
                    config.expert_feed_forward_length as u64,
                    config.expert_count as u64,
                ],
            )?;
            let up_experts = bind_owned(
                gguf,
                &mut expected,
                format!("{prefix}.ffn_up_exps.weight"),
                &[
                    h,
                    config.expert_feed_forward_length as u64,
                    config.expert_count as u64,
                ],
            )?;
            let down_experts = bind_owned(
                gguf,
                &mut expected,
                format!("{prefix}.ffn_down_exps.weight"),
                &[
                    config.expert_feed_forward_length as u64,
                    h,
                    config.expert_count as u64,
                ],
            )?;
            let gate_shared = bind_owned(
                gguf,
                &mut expected,
                format!("{prefix}.ffn_gate_shexp.weight"),
                &[h, shared_width],
            )?;
            let up_shared = bind_owned(
                gguf,
                &mut expected,
                format!("{prefix}.ffn_up_shexp.weight"),
                &[h, shared_width],
            )?;
            let down_shared = bind_owned(
                gguf,
                &mut expected,
                format!("{prefix}.ffn_down_shexp.weight"),
                &[shared_width, h],
            )?;
            let router = if layer < config.hash_layer_count as usize {
                let token_to_expert = bind_owned(
                    gguf,
                    &mut expected,
                    format!("{prefix}.ffn_gate_tid2eid.weight"),
                    &[config.expert_used_count as u64, vocab],
                )?;
                if token_to_expert.dtype != GgmlType::I32 {
                    return Err(DeepSeekV4Error::DtypeMismatch {
                        tensor: token_to_expert.name.clone(),
                        expected: GgmlType::I32,
                        actual: token_to_expert.dtype,
                    });
                }
                RouterWeights::TokenHash { token_to_expert }
            } else {
                RouterWeights::Learned {
                    correction_bias: bind_owned(
                        gguf,
                        &mut expected,
                        format!("{prefix}.exp_probs_b.bias"),
                        &[config.expert_count as u64],
                    )?,
                }
            };
            let ffn_hyper_connection =
                bind_hyper_connection(gguf, &mut expected, &prefix, "ffn", hc_width, hc_params)?;
            blocks.push(DeepSeekV4Block {
                attention,
                attention_hyper_connection,
                ffn_norm,
                moe: MoeWeights {
                    gate_input,
                    gate_experts,
                    up_experts,
                    down_experts,
                    gate_shared,
                    up_shared,
                    down_shared,
                    router,
                },
                ffn_hyper_connection,
            });
        }

        let mut unexpected = gguf
            .tensors
            .iter()
            .filter(|tensor| !expected.contains(tensor.name.as_str()))
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        unexpected.sort();
        if !unexpected.is_empty() {
            return Err(DeepSeekV4Error::UnexpectedTensors(unexpected));
        }
        if gguf.tensors.len() != 1_328 {
            return Err(DeepSeekV4Error::ProfileMismatch {
                field: "target_tensor_count",
                expected: "1328".into(),
                actual: gguf.tensors.len().to_string(),
            });
        }

        validate_bound_dtypes(gguf)?;
        for block in &blocks {
            validate_dtype_pairs(block)?;
            if let RouterWeights::TokenHash { token_to_expert } = &block.moe.router {
                validate_hash_expert_ids(gguf, token_to_expert, config.expert_count)?;
            }
        }

        Ok(Self {
            config,
            token_embedding,
            output_norm,
            output,
            output_hyper_connection,
            blocks,
            source_tensor_count: gguf.tensors.len(),
        })
    }
}

fn bind_attention_lane<'a>(
    gguf: &'a GgufFile,
    expected: &mut HashSet<String>,
    config: &DeepSeekV4Config,
    layer: usize,
    prefix: &str,
) -> Result<AttentionLane<'a>, DeepSeekV4Error> {
    match config.attention_kinds[layer] {
        AttentionKind::SlidingWindow => Ok(AttentionLane::SlidingWindow),
        AttentionKind::CompressedSparse => {
            let compressor = bind_compressor(
                gguf,
                expected,
                prefix,
                "attn_compressor",
                config.hidden_size as u64,
                config.key_length as u64,
                4,
                true,
            )?;
            let indexer = IndexerWeights {
                q: bind_owned(
                    gguf,
                    expected,
                    format!("{prefix}.indexer.attn_q_b.weight"),
                    &[
                        config.q_lora_rank as u64,
                        checked_mul(
                            config.indexer_head_count as u64,
                            config.indexer_key_length as u64,
                            "indexer_head_count * indexer_key_length",
                        )?,
                    ],
                )?,
                projection: bind_owned(
                    gguf,
                    expected,
                    format!("{prefix}.indexer.proj.weight"),
                    &[config.hidden_size as u64, config.indexer_head_count as u64],
                )?,
                compressor: bind_compressor(
                    gguf,
                    expected,
                    prefix,
                    "indexer_compressor",
                    config.hidden_size as u64,
                    config.indexer_key_length as u64,
                    4,
                    true,
                )?,
            };
            Ok(AttentionLane::CompressedSparse {
                compressor,
                indexer,
            })
        }
        AttentionKind::HeavilyCompressed => Ok(AttentionLane::HeavilyCompressed {
            compressor: bind_compressor(
                gguf,
                expected,
                prefix,
                "attn_compressor",
                config.hidden_size as u64,
                config.key_length as u64,
                128,
                false,
            )?,
        }),
    }
}

fn bind_compressor<'a>(
    gguf: &'a GgufFile,
    expected: &mut HashSet<String>,
    prefix: &str,
    stem: &str,
    hidden_size: u64,
    output_size: u64,
    ratio: u64,
    overlapping: bool,
) -> Result<CompressorWeights<'a>, DeepSeekV4Error> {
    let state_width = if overlapping {
        checked_mul(output_size, 2, "overlapping compressor width")?
    } else {
        output_size
    };
    Ok(CompressorWeights {
        kv: bind_owned(
            gguf,
            expected,
            format!("{prefix}.{stem}_kv.weight"),
            &[hidden_size, state_width],
        )?,
        gate: bind_owned(
            gguf,
            expected,
            format!("{prefix}.{stem}_gate.weight"),
            &[hidden_size, state_width],
        )?,
        ape: bind_owned(
            gguf,
            expected,
            format!("{prefix}.{stem}_ape.weight"),
            &[state_width, ratio],
        )?,
        norm: bind_owned(
            gguf,
            expected,
            format!("{prefix}.{stem}_norm.weight"),
            &[output_size],
        )?,
    })
}

fn bind_hyper_connection<'a>(
    gguf: &'a GgufFile,
    expected: &mut HashSet<String>,
    prefix: &str,
    boundary: &str,
    input_width: u64,
    parameter_count: u64,
) -> Result<HyperConnectionWeights<'a>, DeepSeekV4Error> {
    Ok(HyperConnectionWeights {
        function: bind_owned(
            gguf,
            expected,
            format!("{prefix}.hc_{boundary}_fn.weight"),
            &[input_width, parameter_count],
        )?,
        scale: bind_owned(
            gguf,
            expected,
            format!("{prefix}.hc_{boundary}_scale.weight"),
            &[3],
        )?,
        base: bind_owned(
            gguf,
            expected,
            format!("{prefix}.hc_{boundary}_base.weight"),
            &[parameter_count],
        )?,
    })
}

fn bind<'a>(
    gguf: &'a GgufFile,
    expected: &mut HashSet<String>,
    name: &str,
    shape: &[u64],
) -> Result<&'a TensorDesc, DeepSeekV4Error> {
    bind_owned(gguf, expected, name.to_string(), shape)
}

fn bind_owned<'a>(
    gguf: &'a GgufFile,
    expected: &mut HashSet<String>,
    name: String,
    shape: &[u64],
) -> Result<&'a TensorDesc, DeepSeekV4Error> {
    expected.insert(name.clone());
    let tensor = gguf
        .find(&name)
        .ok_or_else(|| DeepSeekV4Error::MissingTensor(name.clone()))?;
    if tensor.shape.as_slice() != shape {
        return Err(DeepSeekV4Error::ShapeMismatch {
            tensor: name,
            expected: shape.to_vec(),
            actual: tensor.shape.clone(),
        });
    }
    Ok(tensor)
}

fn validate_bound_dtypes(gguf: &GgufFile) -> Result<(), DeepSeekV4Error> {
    for tensor in &gguf.tensors {
        if tensor.name.ends_with(".ffn_gate_tid2eid.weight") {
            if tensor.dtype != GgmlType::I32 {
                return Err(DeepSeekV4Error::DtypeMismatch {
                    tensor: tensor.name.clone(),
                    expected: GgmlType::I32,
                    actual: tensor.dtype,
                });
            }
        } else if requires_f32_storage(&tensor.name) {
            if tensor.dtype != GgmlType::F32 {
                return Err(DeepSeekV4Error::DtypeMismatch {
                    tensor: tensor.name.clone(),
                    expected: GgmlType::F32,
                    actual: tensor.dtype,
                });
            }
        } else if !is_supported_weight_dtype(tensor.dtype) {
            return Err(DeepSeekV4Error::UnsupportedWeightDtype {
                tensor: tensor.name.clone(),
                actual: tensor.dtype,
            });
        }
    }
    Ok(())
}

fn requires_f32_storage(name: &str) -> bool {
    name == "output_norm.weight"
        || matches!(name, "output_hc_scale.weight" | "output_hc_base.weight")
        || [
            ".attn_norm.weight",
            ".attn_sinks.weight",
            ".attn_q_a_norm.weight",
            ".attn_kv_a_norm.weight",
            ".attn_compressor_ape.weight",
            ".attn_compressor_norm.weight",
            ".indexer.proj.weight",
            ".indexer_compressor_ape.weight",
            ".indexer_compressor_norm.weight",
            ".exp_probs_b.bias",
            ".ffn_norm.weight",
            ".hc_attn_scale.weight",
            ".hc_attn_base.weight",
            ".hc_ffn_scale.weight",
            ".hc_ffn_base.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn is_supported_weight_dtype(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_K
            | GgmlType::IQ2_XXS
            | GgmlType::IQ2_XS
            | GgmlType::IQ3_XXS
            | GgmlType::IQ1_S
            | GgmlType::IQ4_NL
            | GgmlType::IQ3_S
            | GgmlType::IQ2_S
            | GgmlType::IQ4_XS
            | GgmlType::IQ1_M
            | GgmlType::MXFP4
    )
}

fn validate_dtype_pairs(block: &DeepSeekV4Block<'_>) -> Result<(), DeepSeekV4Error> {
    require_same_dtype(block.moe.gate_experts, block.moe.up_experts)?;
    require_same_dtype(block.moe.gate_shared, block.moe.up_shared)?;
    match &block.attention.lane {
        AttentionLane::SlidingWindow => {}
        AttentionLane::CompressedSparse {
            compressor,
            indexer,
        } => {
            validate_compressor_dtype_pair(compressor)?;
            validate_compressor_dtype_pair(&indexer.compressor)?;
        }
        AttentionLane::HeavilyCompressed { compressor } => {
            validate_compressor_dtype_pair(compressor)?;
        }
    }
    Ok(())
}

fn validate_compressor_dtype_pair(
    compressor: &CompressorWeights<'_>,
) -> Result<(), DeepSeekV4Error> {
    require_same_dtype(compressor.kv, compressor.gate)
}

fn require_same_dtype(left: &TensorDesc, right: &TensorDesc) -> Result<(), DeepSeekV4Error> {
    if left.dtype != right.dtype {
        return Err(DeepSeekV4Error::PairedDtypeMismatch {
            left: left.name.clone(),
            left_dtype: left.dtype,
            right: right.name.clone(),
            right_dtype: right.dtype,
        });
    }
    Ok(())
}

fn validate_hash_expert_ids(
    gguf: &GgufFile,
    tensor: &TensorDesc,
    expert_count: u32,
) -> Result<(), DeepSeekV4Error> {
    let bytes = gguf.try_slice(tensor)?;
    let topk = tensor.shape.first().copied().unwrap_or(0) as usize;
    validate_hash_expert_payload(&tensor.name, bytes, topk, expert_count)
}

fn validate_hash_expert_payload(
    tensor_name: &str,
    bytes: &[u8],
    topk: usize,
    expert_count: u32,
) -> Result<(), DeepSeekV4Error> {
    if topk == 0 {
        return Err(DeepSeekV4Error::InvalidTensorValue {
            tensor: tensor_name.into(),
            detail: "hash router top-k must be nonzero".into(),
        });
    }
    let mut chunks = bytes.chunks_exact(4);
    let mut entries = 0usize;
    for (index, chunk) in chunks.by_ref().enumerate() {
        let expert = i32::from_le_bytes(chunk.try_into().expect("four-byte I32 chunk"));
        if expert < 0 || expert as u32 >= expert_count {
            return Err(DeepSeekV4Error::InvalidTensorValue {
                tensor: tensor_name.into(),
                detail: format!(
                    "entry {index} contains expert id {expert}, expected 0..{expert_count}"
                ),
            });
        }
        entries += 1;
    }
    if !chunks.remainder().is_empty() {
        return Err(DeepSeekV4Error::InvalidTensorValue {
            tensor: tensor_name.into(),
            detail: "I32 payload has a non-four-byte remainder".into(),
        });
    }
    if !entries.is_multiple_of(topk) {
        return Err(DeepSeekV4Error::InvalidTensorValue {
            tensor: tensor_name.into(),
            detail: format!("I32 payload has {entries} entries, not divisible by top-k {topk}"),
        });
    }
    Ok(())
}

fn required_u32(gguf: &GgufFile, key: &str) -> Result<u32, DeepSeekV4Error> {
    let value = gguf
        .get_u64(key)
        .ok_or_else(|| DeepSeekV4Error::MissingMetadata(key.into()))?;
    u32::try_from(value).map_err(|_| DeepSeekV4Error::InvalidMetadata {
        key: key.into(),
        detail: format!("value {value} exceeds u32"),
    })
}

fn optional_u32(gguf: &GgufFile, key: &str) -> Result<Option<u32>, DeepSeekV4Error> {
    gguf.get_u64(key)
        .map(|value| {
            u32::try_from(value).map_err(|_| DeepSeekV4Error::InvalidMetadata {
                key: key.into(),
                detail: format!("value {value} exceeds u32"),
            })
        })
        .transpose()
}

fn required_f32(gguf: &GgufFile, key: &str) -> Result<f32, DeepSeekV4Error> {
    let value = gguf
        .get_f64(key)?
        .ok_or_else(|| DeepSeekV4Error::MissingMetadata(key.into()))?;
    if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
        return Err(DeepSeekV4Error::InvalidMetadata {
            key: key.into(),
            detail: format!("value {value} is not a finite f32"),
        });
    }
    let narrowed = value as f32;
    if value != 0.0 && narrowed == 0.0 {
        return Err(DeepSeekV4Error::InvalidMetadata {
            key: key.into(),
            detail: format!("value {value} underflows f32"),
        });
    }
    Ok(narrowed)
}

fn required_f32_array(gguf: &GgufFile, key: &str) -> Result<Vec<f32>, DeepSeekV4Error> {
    optional_f32_array(gguf, key)?.ok_or_else(|| DeepSeekV4Error::MissingMetadata(key.into()))
}

fn optional_f32_array(gguf: &GgufFile, key: &str) -> Result<Option<Vec<f32>>, DeepSeekV4Error> {
    gguf.get_f64_array(key)?
        .map(|values| {
            values
                .into_iter()
                .enumerate()
                .map(|(idx, value)| {
                    if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
                        return Err(DeepSeekV4Error::InvalidMetadata {
                            key: key.into(),
                            detail: format!("entry {idx} value {value} is not a finite f32"),
                        });
                    }
                    let narrowed = value as f32;
                    if value != 0.0 && narrowed == 0.0 {
                        return Err(DeepSeekV4Error::InvalidMetadata {
                            key: key.into(),
                            detail: format!("entry {idx} value {value} underflows f32"),
                        });
                    }
                    Ok(narrowed)
                })
                .collect()
        })
        .transpose()
}

fn required_bool(gguf: &GgufFile, key: &str) -> Result<bool, DeepSeekV4Error> {
    gguf.get_bool(key)?
        .ok_or_else(|| DeepSeekV4Error::MissingMetadata(key.into()))
}

fn optional_bool(gguf: &GgufFile, key: &str) -> Result<Option<bool>, DeepSeekV4Error> {
    Ok(gguf.get_bool(key)?)
}

fn required_str<'a>(gguf: &'a GgufFile, key: &str) -> Result<&'a str, DeepSeekV4Error> {
    gguf.get_str(key)
        .ok_or_else(|| DeepSeekV4Error::MissingMetadata(key.into()))
}

fn validate_layer_array(
    key: &str,
    values: &[f32],
    layer_count: u32,
) -> Result<(), DeepSeekV4Error> {
    if values.len() != layer_count as usize {
        return Err(DeepSeekV4Error::InvalidMetadata {
            key: key.into(),
            detail: format!(
                "has {} entries for {layer_count} target layers",
                values.len()
            ),
        });
    }
    Ok(())
}

fn checked_mul(left: u64, right: u64, label: &str) -> Result<u64, DeepSeekV4Error> {
    left.checked_mul(right)
        .ok_or_else(|| DeepSeekV4Error::DimensionOverflow(label.into()))
}

fn grouped_output_input_width(
    attention_head_count: u32,
    key_length: u32,
    output_group_count: u32,
) -> Result<u64, DeepSeekV4Error> {
    let q_width = checked_mul(
        attention_head_count as u64,
        key_length as u64,
        "attention_head_count * key_length",
    )?;
    let groups = output_group_count as u64;
    if groups == 0 || q_width % groups != 0 {
        return invalid(
            "deepseek4.attention.output_group_count",
            "does not divide projected query width",
        );
    }
    Ok(q_width / groups)
}

fn require_nonzero(key: &str, value: u32) -> Result<(), DeepSeekV4Error> {
    if value == 0 {
        return invalid(key, "must be nonzero");
    }
    Ok(())
}

fn require_positive_f32(key: &str, value: f32) -> Result<(), DeepSeekV4Error> {
    if !value.is_finite() || value <= 0.0 {
        return invalid(key, "must be finite and positive");
    }
    Ok(())
}

fn invalid<T>(key: &str, detail: &str) -> Result<T, DeepSeekV4Error> {
    Err(DeepSeekV4Error::InvalidMetadata {
        key: key.into(),
        detail: detail.into(),
    })
}

#[derive(Debug, thiserror::Error)]
pub enum DeepSeekV4Error {
    #[error("unsupported architecture {0:?}; expected deepseek4")]
    UnsupportedArchitecture(Option<String>),
    #[error("missing GGUF metadata {0:?}")]
    MissingMetadata(String),
    #[error("invalid GGUF metadata {key:?}: {detail}")]
    InvalidMetadata { key: String, detail: String },
    #[error("missing DeepSeek V4 tensor {0:?}")]
    MissingTensor(String),
    #[error("tensor {tensor:?} shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        tensor: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("tensor {tensor:?} dtype mismatch: expected {expected}, got {actual}")]
    DtypeMismatch {
        tensor: String,
        expected: GgmlType,
        actual: GgmlType,
    },
    #[error("tensor {tensor:?} uses unsupported weight dtype {actual}")]
    UnsupportedWeightDtype { tensor: String, actual: GgmlType },
    #[error("paired tensor dtypes differ: {left:?} is {left_dtype}, {right:?} is {right_dtype}")]
    PairedDtypeMismatch {
        left: String,
        left_dtype: GgmlType,
        right: String,
        right_dtype: GgmlType,
    },
    #[error("tensor {tensor:?} contains an invalid value: {detail}")]
    InvalidTensorValue { tensor: String, detail: String },
    #[error("unexpected tensors outside the target DeepSeek V4 schema: {0:?}")]
    UnexpectedTensors(Vec<String>),
    #[error(
        "DeepSeek V4 Flash-0731 profile mismatch for {field}: expected {expected}, got {actual}"
    )]
    ProfileMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("DeepSeek V4 dimension overflow: {0}")]
    DimensionOverflow(String),
    #[error(transparent)]
    Gguf(#[from] crate::gguf::GgufError),
}

#[cfg(test)]
pub(crate) fn flash_0731_config_fixture() -> DeepSeekV4Config {
    DeepSeekV4Config {
        layer_count: 43,
        context_length: 1_048_576,
        hidden_size: 4_096,
        vocab_size: 129_280,
        attention_head_count: 64,
        kv_head_count: 1,
        key_length: 512,
        value_length: 512,
        rope_dimension_count: 64,
        rope_freq_base: 10_000.0,
        attention_rms_epsilon: 1e-6,
        rope_scaling_type: "yarn".into(),
        rope_scaling_factor: 16.0,
        rope_original_context_length: 65_536,
        rope_yarn_beta_fast: 32.0,
        rope_yarn_beta_slow: 1.0,
        q_lora_rank: 1_024,
        sliding_window: 128,
        expert_count: 256,
        expert_used_count: 6,
        expert_feed_forward_length: 2_048,
        shared_expert_count: 1,
        expert_weights_scale: 1.5,
        expert_weights_norm: true,
        expert_gating_func: 4,
        indexer_head_count: 64,
        indexer_key_length: 128,
        indexer_top_k: 512,
        output_group_count: 8,
        output_lora_rank: 1_024,
        attention_kinds: (0..43).map(flash_0731_attention_kind).collect(),
        compress_ratio_tail: vec![0, 0, 0],
        compress_rope_freq_base: 160_000.0,
        hyper_connection_count: 4,
        sinkhorn_iterations: 20,
        hyper_connection_epsilon: 1e-6,
        hash_layer_count: 3,
        swiglu_clamp_experts: vec![10.0; 43],
        swiglu_clamp_shared: vec![10.0; 43],
        tokenizer_model: "gpt2".into(),
        tokenizer_pre: "joyai-llm".into(),
        bos_token_id: Some(0),
        eos_token_id: Some(1),
        padding_token_id: Some(2),
        tokenizer_token_type_count: 129_280,
        tokenizer_merge_count: 127_741,
        add_bos_token: false,
        add_eos_token: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek_v4_census::DeepSeekV4CensusV1;
    use std::path::Path;

    const DS4_0731_CURRENT: &str = crate::test_fixtures::DEEPSEEK_V4_IQ3_XXS.path();
    const DS4_0731_REAP_K160: &str = "/Users/tito/models/deepseek-v4-flash-0731-reap-k160/DeepSeek-V4-Flash-0731-REAP-K160-Q3_K_Q4_K-00001-of-00004.gguf";
    const DS4_0731_REAP_K216: &str = "/Users/tito/models/deepseek-v4-flash-0731-reap-k216/DeepSeek-V4-Flash-0731-REAP-K216-UD-IQ3_XXS-00001-of-00003.gguf";

    #[test]
    fn compression_ratios_are_closed() {
        assert_eq!(
            AttentionKind::from_ratio(0, 0).unwrap(),
            AttentionKind::SlidingWindow
        );
        assert_eq!(
            AttentionKind::from_ratio(2, 4).unwrap(),
            AttentionKind::CompressedSparse
        );
        assert_eq!(
            AttentionKind::from_ratio(3, 128).unwrap(),
            AttentionKind::HeavilyCompressed
        );
        assert!(AttentionKind::from_ratio(4, 2).is_err());
    }

    #[test]
    fn flash_0731_schedule_is_two_local_then_alternating() {
        let schedule = (0..43).map(flash_0731_attention_kind).collect::<Vec<_>>();
        let counts = schedule
            .iter()
            .fold((0, 0, 0), |(local, csa, hca), kind| match kind {
                AttentionKind::SlidingWindow => (local + 1, csa, hca),
                AttentionKind::CompressedSparse => (local, csa + 1, hca),
                AttentionKind::HeavilyCompressed => (local, csa, hca + 1),
            });
        assert_eq!(counts, (2, 21, 20));
        assert_eq!(schedule[2], AttentionKind::CompressedSparse);
        assert_eq!(schedule[3], AttentionKind::HeavilyCompressed);
        assert_eq!(schedule[42], AttentionKind::CompressedSparse);
    }

    #[test]
    fn flash_0731_profile_rejects_geometry_and_schedule_drift() {
        let config = flash_0731_config_fixture();
        config.validate().expect("valid generic config");
        config
            .validate_flash_0731_profile()
            .expect("valid frozen profile");

        let mut wrong_hidden = config.clone();
        wrong_hidden.hidden_size += 1;
        assert!(wrong_hidden.validate_flash_0731_profile().is_err());

        let mut reap_k160 = config.clone();
        reap_k160.expert_count = 160;
        reap_k160
            .validate_flash_0731_profile()
            .expect("valid K160 REAP profile");

        let mut reap_k216 = config.clone();
        reap_k216.expert_count = 216;
        reap_k216
            .validate_flash_0731_profile()
            .expect("valid K216 REAP profile");

        let mut unsupported_experts = config.clone();
        unsupported_experts.expert_count = 200;
        assert!(unsupported_experts.validate_flash_0731_profile().is_err());

        let mut wrong_schedule = config;
        wrong_schedule.attention_kinds[3] = AttentionKind::CompressedSparse;
        assert!(wrong_schedule.validate_flash_0731_profile().is_err());
    }

    #[test]
    fn semantic_and_weight_dtype_classes_are_distinct() {
        assert!(requires_f32_storage("blk.2.attn_compressor_ape.weight"));
        assert!(requires_f32_storage("blk.3.exp_probs_b.bias"));
        assert!(requires_f32_storage("blk.2.hc_attn_scale.weight"));
        assert!(!requires_f32_storage("blk.2.hc_attn_fn.weight"));
        assert!(!requires_f32_storage("output_hc_fn.weight"));
        assert!(!requires_f32_storage("blk.2.attn_compressor_kv.weight"));
        assert!(is_supported_weight_dtype(GgmlType::IQ2_S));
        assert!(!is_supported_weight_dtype(GgmlType::I32));
    }

    #[test]
    fn grouped_output_width_comes_from_attention_geometry() {
        assert_eq!(grouped_output_input_width(64, 512, 8).unwrap(), 4096);
        assert_eq!(grouped_output_input_width(32, 512, 8).unwrap(), 2048);
        assert!(grouped_output_input_width(63, 511, 8).is_err());
    }

    #[test]
    fn hash_router_payload_accepts_duplicate_slots_and_rejects_out_of_range_experts() {
        fn bytes(values: &[i32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        let valid = bytes(&[0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]);
        validate_hash_expert_payload("hash", &valid, 6, 256).expect("valid hash rows");

        let duplicate = bytes(&[0, 1, 2, 2, 4, 5]);
        validate_hash_expert_payload("hash", &duplicate, 6, 256)
            .expect("duplicate hash slots retain independent route weights");

        let out_of_range = bytes(&[0, 1, 2, 3, 4, 256]);
        assert!(validate_hash_expert_payload("hash", &out_of_range, 6, 256).is_err());
    }

    #[test]
    #[ignore = "requires the current local DeepSeek V4 Flash-0731 fixture"]
    fn live_0731_iq3_schema_binds_every_tensor() {
        assert!(Path::new(DS4_0731_CURRENT).exists(), "missing DS4 fixture");
        let gguf = GgufFile::open(DS4_0731_CURRENT).expect("open DS4 fixture");
        let model = DeepSeekV4Model::from_gguf_flash_0731(&gguf).expect("bind DS4 fixture");
        assert_eq!(model.config.layer_count, 43);
        assert_eq!(model.config.hidden_size, 4096);
        assert_eq!(model.config.vocab_size, 129_280);
        assert_eq!(model.config.attention_counts(), (2, 21, 20));
        assert_eq!(model.config.hash_layer_count, 3);
        assert_eq!(model.config.compress_ratio_tail, [0, 0, 0]);
        assert_eq!(model.config.bos_token_id, Some(0));
        assert_eq!(model.config.eos_token_id, Some(1));
        assert_eq!(model.config.padding_token_id, Some(2));
        assert_eq!(model.blocks.len(), 43);
        assert_eq!(model.source_tensor_count, 1328);
        assert!(matches!(
            model.blocks[0].moe.router,
            RouterWeights::TokenHash { .. }
        ));
        assert!(matches!(
            model.blocks[3].attention.lane,
            AttentionLane::HeavilyCompressed { .. }
        ));
    }

    #[test]
    #[ignore = "requires the local DeepSeek V4 Flash-0731 REAP K160 fixture"]
    fn live_0731_reap_k160_schema_binds_every_tensor() {
        assert!(
            Path::new(DS4_0731_REAP_K160).exists(),
            "missing K160 REAP fixture"
        );
        let gguf = GgufFile::open(DS4_0731_REAP_K160).expect("open K160 REAP fixture");
        let model = DeepSeekV4Model::from_gguf_flash_0731(&gguf).expect("bind K160 REAP fixture");
        assert_eq!(model.config.expert_count, 160);
        assert_eq!(model.config.swiglu_clamp_experts, vec![10.0; 43]);
        assert_eq!(model.config.swiglu_clamp_shared, vec![10.0; 43]);
        assert_eq!(model.token_embedding.dtype, GgmlType::Q8_0);
        assert_eq!(model.output.dtype, GgmlType::Q8_0);
        assert_eq!(model.blocks[0].moe.gate_experts.dtype, GgmlType::Q3_K);
        assert_eq!(model.blocks[0].moe.down_experts.dtype, GgmlType::Q4_K);
        assert_eq!(model.source_tensor_count, 1_328);
    }

    #[test]
    #[ignore = "requires the local DeepSeek V4 Flash-0731 REAP K216 fixture"]
    fn live_0731_reap_k216_schema_binds_every_tensor() {
        assert!(
            Path::new(DS4_0731_REAP_K216).exists(),
            "missing K216 REAP fixture"
        );
        let gguf = GgufFile::open(DS4_0731_REAP_K216).expect("open K216 REAP fixture");
        let model = DeepSeekV4Model::from_gguf_flash_0731(&gguf).expect("bind K216 REAP fixture");
        let census =
            DeepSeekV4CensusV1::from_gguf_flash_0731(&gguf).expect("census K216 REAP fixture");
        assert_eq!(gguf.shard_count(), 3);
        assert_eq!(census.totals.shard_count, 3);
        assert_eq!(census.totals.tensor_count, 1_328);
        assert_eq!(census.totals.tensor_bytes, 89_060_075_612);
        assert_eq!(
            gguf.tensors
                .iter()
                .map(|tensor| tensor.n_bytes)
                .sum::<u64>(),
            89_060_075_612
        );
        assert_eq!(model.config.expert_count, 216);
        assert_eq!(model.config.swiglu_clamp_experts, vec![10.0; 43]);
        assert_eq!(model.config.swiglu_clamp_shared, vec![10.0; 43]);
        assert_eq!(model.token_embedding.dtype, GgmlType::Q6_K);
        assert_eq!(model.output.dtype, GgmlType::Q6_K);
        assert_eq!(model.blocks[0].moe.gate_experts.dtype, GgmlType::IQ2_XS);
        assert_eq!(model.blocks[0].moe.down_experts.dtype, GgmlType::IQ3_XXS);
        assert_eq!(model.blocks[42].moe.gate_experts.dtype, GgmlType::IQ3_XXS);
        assert_eq!(model.blocks[42].moe.down_experts.dtype, GgmlType::MXFP4);
        let cohort_count = |gate, up, down| {
            model
                .blocks
                .iter()
                .filter(|block| {
                    block.moe.gate_experts.dtype == gate
                        && block.moe.up_experts.dtype == up
                        && block.moe.down_experts.dtype == down
                })
                .count()
        };
        assert_eq!(
            cohort_count(GgmlType::IQ2_XS, GgmlType::IQ2_XS, GgmlType::IQ3_XXS),
            25
        );
        assert_eq!(
            cohort_count(GgmlType::IQ3_XXS, GgmlType::IQ3_XXS, GgmlType::IQ3_XXS,),
            16
        );
        assert_eq!(
            cohort_count(GgmlType::IQ3_S, GgmlType::IQ3_S, GgmlType::MXFP4),
            1
        );
        assert_eq!(
            cohort_count(GgmlType::IQ3_XXS, GgmlType::IQ3_XXS, GgmlType::MXFP4),
            1
        );
        assert_eq!(model.source_tensor_count, 1_328);
    }
}
