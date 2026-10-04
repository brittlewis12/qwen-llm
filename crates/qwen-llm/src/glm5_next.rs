//! GLM-5.3-Flash (`glm5-next`) configuration, structural GGUF binding, kernel
//! coverage and memory ledger.
//!
//! This is an inspection API, not runtime admission. Binding a model enables no
//! family dispatch, chat capability or Metal execution. Reference semantics are
//! llama.cpp `src/models/glm5-next.cpp`; see `docs/GLM53-FLASH-PLAN.md`.
//!
//! The release stores 46 blocks: 45 executed trunk blocks plus the NextN (MTP)
//! block. NextN tensors are recognized and counted, but are never retained.

use crate::gguf::{GgufError, GgufFile};
use crate::metal::{RetainedStorageDisposition, RetainedStoragePlan, plan_retained_storage};
use crate::tensor::TensorDesc;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

pub mod coverage;
pub mod memory;

pub use coverage::{CoverageRow, ExecutionMode, RoleCoverage, Support, TensorRole};
pub use memory::{Glm5NextMemoryLedger, Glm5NextPhasePeaks};

/// Architecture spelled by llama.cpp and unsloth's rewritten shard.
pub const ARCHITECTURE_NAME: &str = "glm5-next";
/// Spelling in the original unsloth shard; tensor bytes are identical.
pub const LEGACY_ARCHITECTURE_NAME: &str = "glm5next";
pub const RELEASE_TENSOR_COUNT: usize = 1412;
/// GPT-2 byte-BPE vocabulary entries, including unused padding rows.
pub const RELEASE_VOCAB_SIZE: u32 = 154_880;

#[derive(Debug, thiserror::Error)]
pub enum Glm5NextError {
    #[error("missing required GLM-5.3 metadata: {0}")]
    MissingMetadata(String),
    #[error("invalid GLM-5.3 metadata {key:?}: {detail}")]
    InvalidMetadata { key: String, detail: String },
    #[error("unsupported GLM-5.3 feature {key:?}: {detail}")]
    Unsupported { key: String, detail: String },
    #[error("invalid GLM-5.3 tensor {name:?}: {detail}")]
    Tensor { name: String, detail: String },
    #[error("GLM-5.3 size arithmetic overflow: {0}")]
    Overflow(&'static str),
    #[error(transparent)]
    Gguf(#[from] GgufError),
}

pub type Result<T> = std::result::Result<T, Glm5NextError>;

/// Token mixer of one executed block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixerKind {
    /// Kimi delta attention: gated delta rule with per-channel decay.
    Kda,
    /// NoPE multi-head latent attention with pooled DSA selection.
    Mla,
}

/// Feed-forward of one executed block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FfnKind {
    Dense,
    Moe,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockKind {
    pub mixer: MixerKind,
    pub ffn: FfnKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Glm5NextConfig {
    /// `general.architecture` as stored; selects the metadata namespace.
    pub architecture: String,
    pub stored_block_count: u32,
    pub nextn_layer_count: u32,
    pub context_length: u32,
    pub hidden_size: u32,
    pub dense_ffn_size: u32,
    pub head_count: u32,
    pub vocab_size: u32,
    pub rms_epsilon: f32,
    pub layer_norm_epsilon: f32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_ffn_size: u32,
    pub shared_expert_count: u32,
    pub shared_expert_ffn_size: u32,
    pub leading_dense_block_count: u32,
    pub expert_weights_scale: f32,
    pub swiglu_clamp: f32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub mla_key_head_dim: u32,
    pub mla_value_head_dim: u32,
    pub kda_head_dim: u32,
    pub kda_conv_kernel: u32,
    pub kda_gate_lower_bound: f32,
    pub indexer_head_count: u32,
    pub indexer_head_dim: u32,
    pub indexer_top_k: u32,
    pub indexer_pool: u32,
    pub hc_streams: u32,
    pub hc_sinkhorn_iterations: u32,
    pub hc_epsilon: f32,
    /// Executed blocks only (NextN excluded), in order.
    pub blocks: Vec<BlockKind>,
}

impl Glm5NextConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        Self::from_metadata(gguf.model.metadata())
    }

    pub(crate) fn from_metadata(metadata: &BTreeMap<String, Value>) -> Result<Self> {
        let architecture = required_str(metadata, "general.architecture")?;
        if architecture != ARCHITECTURE_NAME && architecture != LEGACY_ARCHITECTURE_NAME {
            return Err(unsupported("general.architecture", architecture));
        }
        let ns = Namespace(architecture);
        validate_metadata_features(metadata, ns)?;
        for (key, expected) in [
            ("tokenizer.ggml.model", "gpt2"),
            ("tokenizer.ggml.pre", "glm4"),
        ] {
            let actual = required_str(metadata, key)?;
            if actual != expected {
                return Err(unsupported(
                    key,
                    format!("expected {expected}, got {actual}"),
                ));
            }
        }
        let tokens = required(metadata, "tokenizer.ggml.tokens")?
            .as_array()
            .ok_or_else(|| invalid("tokenizer.ggml.tokens", "expected string array"))?;
        let vocab_size = required_u32(metadata, &ns.key("vocab_size"))?;
        if tokens.len() != vocab_size as usize {
            return Err(invalid(
                "tokenizer.ggml.tokens",
                format!("{} entries, metadata vocab_size {vocab_size}", tokens.len()),
            ));
        }
        if ns.u32(metadata, "expert_gating_func")? != 2 {
            return Err(unsupported(
                &ns.key("expert_gating_func"),
                "only sigmoid routing (2) is implemented",
            ));
        }
        for suffix in ["expert_group_count", "expert_group_used_count"] {
            if ns.u32(metadata, suffix)? != 1 {
                return Err(unsupported(&ns.key(suffix), "grouped routing"));
            }
        }
        if !required_bool(metadata, &ns.key("expert_weights_norm"))? {
            return Err(unsupported(
                &ns.key("expert_weights_norm"),
                "unnormalized expert weights",
            ));
        }
        if ns.u32(metadata, "rope.dimension_count")? != 0 {
            return Err(unsupported(
                &ns.key("rope.dimension_count"),
                "the release graph has no RoPE",
            ));
        }
        for (suffix, expected) in [
            ("attention.key_length", "attention.kv_lora_rank"),
            ("attention.value_length", "attention.kv_lora_rank"),
        ] {
            if ns.u32(metadata, suffix)? != ns.u32(metadata, expected)? {
                return Err(unsupported(
                    &ns.key(suffix),
                    "latent K/V width must equal kv_lora_rank",
                ));
            }
        }

        let stored_block_count = ns.u32(metadata, "block_count")?;
        let nextn_layer_count = ns.u32(metadata, "nextn_predict_layers")?;
        let leading_dense_block_count = ns.u32(metadata, "leading_dense_block_count")?;
        let executed = stored_block_count
            .checked_sub(nextn_layer_count)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid(&ns.key("nextn_predict_layers"), "exceeds block_count"))?;
        let kv_heads = required_u64_array(metadata, &ns.key("attention.head_count_kv"))?;
        if kv_heads.len() != stored_block_count as usize {
            return Err(invalid(
                &ns.key("attention.head_count_kv"),
                format!("{} entries for {stored_block_count} blocks", kv_heads.len()),
            ));
        }
        let mut blocks = Vec::with_capacity(executed as usize);
        for (index, &kv) in kv_heads.iter().enumerate().take(executed as usize) {
            let mixer = match kv {
                0 => MixerKind::Kda,
                1 => MixerKind::Mla,
                other => {
                    return Err(invalid(
                        &ns.key("attention.head_count_kv"),
                        format!("block {index} has {other} KV heads; expected 0 (KDA) or 1 (MLA)"),
                    ));
                }
            };
            let ffn = if (index as u32) < leading_dense_block_count {
                FfnKind::Dense
            } else {
                FfnKind::Moe
            };
            blocks.push(BlockKind { mixer, ffn });
        }
        let swiglu_clamp = uniform_f32_array(
            metadata,
            &ns.key("swiglu_clamp_exp"),
            stored_block_count as usize,
        )?;
        let shared_clamp = uniform_f32_array(
            metadata,
            &ns.key("swiglu_clamp_shexp"),
            stored_block_count as usize,
        )?;
        if swiglu_clamp.to_bits() != shared_clamp.to_bits() {
            return Err(unsupported(
                &ns.key("swiglu_clamp_shexp"),
                "routed and shared clamps differ",
            ));
        }

        let config = Self {
            architecture: architecture.to_string(),
            stored_block_count,
            nextn_layer_count,
            context_length: ns.u32(metadata, "context_length")?,
            hidden_size: ns.u32(metadata, "embedding_length")?,
            dense_ffn_size: ns.u32(metadata, "feed_forward_length")?,
            head_count: ns.u32(metadata, "attention.head_count")?,
            vocab_size,
            rms_epsilon: ns.f32(metadata, "attention.layer_norm_rms_epsilon")?,
            layer_norm_epsilon: ns.f32(metadata, "attention.layer_norm_epsilon")?,
            expert_count: ns.u32(metadata, "expert_count")?,
            expert_used_count: ns.u32(metadata, "expert_used_count")?,
            expert_ffn_size: ns.u32(metadata, "expert_feed_forward_length")?,
            shared_expert_count: ns.u32(metadata, "expert_shared_count")?,
            shared_expert_ffn_size: ns.u32(metadata, "expert_shared_feed_forward_length")?,
            leading_dense_block_count,
            expert_weights_scale: ns.f32(metadata, "expert_weights_scale")?,
            swiglu_clamp,
            q_lora_rank: ns.u32(metadata, "attention.q_lora_rank")?,
            kv_lora_rank: ns.u32(metadata, "attention.kv_lora_rank")?,
            mla_key_head_dim: ns.u32(metadata, "attention.key_length_mla")?,
            mla_value_head_dim: ns.u32(metadata, "attention.value_length_mla")?,
            kda_head_dim: ns.u32(metadata, "kda.head_dim")?,
            kda_conv_kernel: ns.u32(metadata, "ssm.conv_kernel")?,
            kda_gate_lower_bound: ns.f32(metadata, "kda.gate_lower_bound")?,
            indexer_head_count: ns.u32(metadata, "attention.indexer.head_count")?,
            indexer_head_dim: ns.u32(metadata, "attention.indexer.key_length")?,
            indexer_top_k: ns.u32(metadata, "attention.indexer.top_k")?,
            indexer_pool: ns.u32(metadata, "attention.indexer.kpool")?,
            hc_streams: ns.u32(metadata, "hyper_connection.count")?,
            hc_sinkhorn_iterations: ns.u32(metadata, "hyper_connection.sinkhorn_iterations")?,
            hc_epsilon: ns.f32(metadata, "hyper_connection.epsilon")?,
            blocks,
        };
        config.validate_release()?;
        Ok(config)
    }

    /// Pins the released GLM-5.3-Flash geometry. Other geometries need their own
    /// reference evidence before they are admitted.
    pub fn validate_release(&self) -> Result<()> {
        let ns = Namespace(&self.architecture);
        for (suffix, actual, expected) in [
            ("block_count", self.stored_block_count, 46),
            ("nextn_predict_layers", self.nextn_layer_count, 1),
            ("embedding_length", self.hidden_size, 4096),
            ("feed_forward_length", self.dense_ffn_size, 12288),
            ("attention.head_count", self.head_count, 64),
            ("vocab_size", self.vocab_size, RELEASE_VOCAB_SIZE),
            ("expert_count", self.expert_count, 288),
            ("expert_used_count", self.expert_used_count, 8),
            ("expert_feed_forward_length", self.expert_ffn_size, 2048),
            ("expert_shared_count", self.shared_expert_count, 1),
            (
                "expert_shared_feed_forward_length",
                self.shared_expert_ffn_size,
                2048,
            ),
            (
                "leading_dense_block_count",
                self.leading_dense_block_count,
                3,
            ),
            ("attention.q_lora_rank", self.q_lora_rank, 1536),
            ("attention.kv_lora_rank", self.kv_lora_rank, 512),
            ("attention.key_length_mla", self.mla_key_head_dim, 256),
            ("attention.value_length_mla", self.mla_value_head_dim, 256),
            ("kda.head_dim", self.kda_head_dim, 128),
            ("ssm.conv_kernel", self.kda_conv_kernel, 4),
            ("attention.indexer.head_count", self.indexer_head_count, 32),
            ("attention.indexer.key_length", self.indexer_head_dim, 128),
            ("attention.indexer.top_k", self.indexer_top_k, 2048),
            ("attention.indexer.kpool", self.indexer_pool, 4),
            ("hyper_connection.count", self.hc_streams, 4),
            (
                "hyper_connection.sinkhorn_iterations",
                self.hc_sinkhorn_iterations,
                20,
            ),
        ] {
            if actual != expected {
                return Err(unsupported(
                    &ns.key(suffix),
                    format!("release expects {expected}, got {actual}"),
                ));
            }
        }
        for (suffix, actual, expected) in [
            ("attention.layer_norm_rms_epsilon", self.rms_epsilon, 1e-5),
            (
                "attention.layer_norm_epsilon",
                self.layer_norm_epsilon,
                1e-6,
            ),
            ("hyper_connection.epsilon", self.hc_epsilon, 1e-6),
            ("expert_weights_scale", self.expert_weights_scale, 2.5),
            ("swiglu_clamp_exp", self.swiglu_clamp, 10.0),
            ("kda.gate_lower_bound", self.kda_gate_lower_bound, -5.0),
        ] {
            if actual.to_bits() != f32::to_bits(expected) {
                return Err(unsupported(
                    &ns.key(suffix),
                    format!("release expects {expected}, got {actual}"),
                ));
            }
        }
        if self.context_length == 0 {
            return Err(invalid(&ns.key("context_length"), "must be positive"));
        }
        if self.blocks.len() != self.executed_block_count() as usize {
            return Err(invalid(&ns.key("block_count"), "schedule length mismatch"));
        }
        // Release schedule: MLA at every fourth block (3, 7, ..., 43), KDA
        // elsewhere; the first three FFNs are dense.
        for (index, block) in self.blocks.iter().enumerate() {
            let mixer = if index % 4 == 3 {
                MixerKind::Mla
            } else {
                MixerKind::Kda
            };
            let ffn = if index < 3 {
                FfnKind::Dense
            } else {
                FfnKind::Moe
            };
            if *block != (BlockKind { mixer, ffn }) {
                return Err(unsupported(
                    &ns.key("attention.head_count_kv"),
                    format!("block {index} is {block:?}; release expects {mixer:?}/{ffn:?}"),
                ));
            }
        }
        Ok(())
    }

    pub fn executed_block_count(&self) -> u32 {
        self.stored_block_count - self.nextn_layer_count
    }

    pub fn block_count(&self, mixer: MixerKind) -> usize {
        self.blocks.iter().filter(|b| b.mixer == mixer).count()
    }

    /// KDA value width (heads x head dim), also its q/k width.
    pub fn kda_width(&self) -> u32 {
        self.head_count * self.kda_head_dim
    }

    /// Concatenated MLA query/output width before `attn_output`.
    pub fn mla_width(&self) -> u32 {
        self.head_count * self.mla_key_head_dim
    }

    /// Flattened hyper-connection residual width (hidden x streams).
    pub fn hc_width(&self) -> u32 {
        self.hidden_size * self.hc_streams
    }

    /// pre (streams) + post (streams) + comb (streams^2).
    pub fn hc_mix_count(&self) -> u32 {
        self.hc_streams * (2 + self.hc_streams)
    }

    /// Selected pools before the tail.
    pub fn selected_pool_count(&self) -> u32 {
        self.indexer_top_k / self.indexer_pool
    }

    /// Maximum selected positions per query: whole pools plus the tail.
    pub fn selection_width(&self) -> u32 {
        self.indexer_top_k + self.indexer_pool - 1
    }

    /// First visible length (current token included) at which sparse selection
    /// can exclude a position. Below it, attention is exactly dense.
    pub fn sparse_frontier(&self) -> u32 {
        self.indexer_top_k + self.indexer_pool
    }
}

#[derive(Debug)]
pub struct HyperConnectionTensors<'a> {
    pub mix: &'a TensorDesc,
    pub base: &'a TensorDesc,
    pub scale: &'a TensorDesc,
}

#[derive(Debug)]
pub struct KdaTensors<'a> {
    pub query: &'a TensorDesc,
    pub key: &'a TensorDesc,
    pub value: &'a TensorDesc,
    pub query_conv: &'a TensorDesc,
    pub key_conv: &'a TensorDesc,
    pub value_conv: &'a TensorDesc,
    pub decay_a: &'a TensorDesc,
    pub decay_b: &'a TensorDesc,
    pub decay_bias: &'a TensorDesc,
    /// Stores `-exp(A_log)` per head, not `A_log`.
    pub neg_exp_a_log: &'a TensorDesc,
    pub beta: &'a TensorDesc,
    pub gate_a: &'a TensorDesc,
    pub gate_b: &'a TensorDesc,
    pub output_norm: &'a TensorDesc,
    pub output: &'a TensorDesc,
}

#[derive(Debug)]
pub struct IndexerTensors<'a> {
    pub query: &'a TensorDesc,
    pub key: &'a TensorDesc,
    pub key_norm: &'a TensorDesc,
    pub key_norm_bias: &'a TensorDesc,
    pub head_weights: &'a TensorDesc,
    pub pool_gate: &'a TensorDesc,
    pub pool_position: &'a TensorDesc,
}

#[derive(Debug)]
pub struct MlaTensors<'a> {
    pub query_a: &'a TensorDesc,
    pub query_a_norm: &'a TensorDesc,
    pub query_b: &'a TensorDesc,
    pub latent: &'a TensorDesc,
    pub latent_norm: &'a TensorDesc,
    /// [head_dim, kv_lora_rank, heads]: absorbs queries into the latent.
    pub key_absorb: &'a TensorDesc,
    /// [kv_lora_rank, head_dim, heads]: expands latent outputs per head.
    pub value_expand: &'a TensorDesc,
    pub output: &'a TensorDesc,
    pub indexer: IndexerTensors<'a>,
}

#[derive(Debug)]
pub enum MixerTensors<'a> {
    Kda(KdaTensors<'a>),
    Mla(MlaTensors<'a>),
}

#[derive(Debug)]
pub struct DenseFfnTensors<'a> {
    pub gate: &'a TensorDesc,
    pub up: &'a TensorDesc,
    pub down: &'a TensorDesc,
}

#[derive(Debug)]
pub struct MoeTensors<'a> {
    pub router: &'a TensorDesc,
    pub selection_bias: &'a TensorDesc,
    pub gate_experts: &'a TensorDesc,
    pub up_experts: &'a TensorDesc,
    pub down_experts: &'a TensorDesc,
    pub shared: DenseFfnTensors<'a>,
}

#[derive(Debug)]
pub enum FfnTensors<'a> {
    Dense(DenseFfnTensors<'a>),
    Moe(MoeTensors<'a>),
}

#[derive(Debug)]
pub struct Glm5NextBlock<'a> {
    pub attention_hc: HyperConnectionTensors<'a>,
    pub ffn_hc: HyperConnectionTensors<'a>,
    pub attention_norm: &'a TensorDesc,
    pub ffn_norm: &'a TensorDesc,
    pub mixer: MixerTensors<'a>,
    pub ffn: FfnTensors<'a>,
}

/// One bound trunk tensor and the execution role that consumes it.
#[derive(Clone, Copy, Debug)]
pub struct BoundTensor<'a> {
    pub tensor: &'a TensorDesc,
    pub role: TensorRole,
}

#[derive(Debug)]
pub struct Glm5NextModel<'a> {
    pub config: Glm5NextConfig,
    pub token_embedding: &'a TensorDesc,
    pub output_norm: &'a TensorDesc,
    pub output: &'a TensorDesc,
    pub blocks: Vec<Glm5NextBlock<'a>>,
    /// Every executed tensor exactly once, in binding order.
    pub trunk: Vec<BoundTensor<'a>>,
    /// NextN block tensors: recognized, never retained or executed.
    pub nextn: Vec<&'a TensorDesc>,
    pub trunk_bytes: u64,
    pub nextn_bytes: u64,
}

impl<'a> Glm5NextModel<'a> {
    /// Binds descriptors without reading weight values or establishing identity.
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self> {
        let config = Glm5NextConfig::from_gguf(gguf)?;
        let model = Self::bind(config, &gguf.tensors)?;
        for tensor in &gguf.tensors {
            gguf.try_slice(tensor)?;
        }
        Ok(model)
    }

    pub(crate) fn bind(config: Glm5NextConfig, tensors: &'a [TensorDesc]) -> Result<Self> {
        config.validate_release()?;
        let mut binder = Binder::new(tensors)?;
        let c = &config;
        let h = u64::from(c.hidden_size);
        let vocab = u64::from(c.vocab_size);
        let token_embedding =
            binder.take("token_embd.weight", &[h, vocab], TensorRole::Embedding)?;
        let output_norm = binder.take("output_norm.weight", &[h], TensorRole::Vector)?;
        let output = binder.take("output.weight", &[h, vocab], TensorRole::Head)?;

        let mut blocks = Vec::with_capacity(c.blocks.len());
        for (i, kind) in c.blocks.iter().enumerate() {
            blocks.push(bind_block(&mut binder, c, i, *kind)?);
        }
        let nextn_start = binder.trunk.len();
        for i in c.executed_block_count()..c.stored_block_count {
            bind_nextn_block(&mut binder, c, i as usize)?;
        }
        binder.finish()?;
        let nextn = binder.trunk.split_off(nextn_start);
        let nextn = nextn.into_iter().map(|b| b.tensor).collect::<Vec<_>>();
        let trunk_bytes = sum_bytes(binder.trunk.iter().map(|b| b.tensor))?;
        let nextn_bytes = sum_bytes(nextn.iter().copied())?;
        Ok(Self {
            config,
            token_embedding,
            output_norm,
            output,
            blocks,
            trunk: binder.trunk,
            nextn,
            trunk_bytes,
            nextn_bytes,
        })
    }

    /// Tensors to retain for execution (NextN excluded).
    pub fn retained_tensors(&self) -> Vec<&'a TensorDesc> {
        self.trunk
            .iter()
            .filter(|b| b.role != TensorRole::NextN)
            .map(|b| b.tensor)
            .collect()
    }

    /// Retained no-copy windows for the executed tensors, and the Metal bytes
    /// they cost (window lengths plus copy fallbacks). The planner bridges gaps
    /// between requested tensors when a window fits, so the caller must check
    /// actual ranges, not only which tensors were requested. Planning only:
    /// residency must first pass [`Self::validate_execution`].
    pub fn plan_retained(
        &self,
        gguf: &GgufFile,
        page_size: usize,
        max_buffer_length: usize,
    ) -> Result<(RetainedStoragePlan, u64)> {
        let requests = self.retained_tensors();
        let plan = plan_retained_storage(
            &gguf.shard_mapped_lengths(),
            &requests,
            page_size,
            max_buffer_length,
            32,
        )
        .map_err(|e| tensor_error("retained storage plan", e.to_string()))?;
        if plan.entries.len() != requests.len() {
            return Err(tensor_error(
                "retained storage plan",
                "entry count differs from requests",
            ));
        }
        let windows = plan.windows.iter().map(|w| w.length as u64);
        let fallbacks = plan.entries.iter().filter_map(|e| match e.disposition {
            RetainedStorageDisposition::CopyFallback { .. } => Some(e.n_bytes),
            _ => None,
        });
        let bytes = windows
            .chain(fallbacks)
            .try_fold(0u64, |acc, b| acc.checked_add(b))
            .ok_or(Glm5NextError::Overflow("retained bytes"))?;
        Ok((plan, bytes))
    }
}

fn bind_block<'a>(
    b: &mut Binder<'a>,
    c: &Glm5NextConfig,
    i: usize,
    kind: BlockKind,
) -> Result<Glm5NextBlock<'a>> {
    let h = u64::from(c.hidden_size);
    let attention_hc = bind_hc(b, c, i, "attn")?;
    let ffn_hc = bind_hc(b, c, i, "ffn")?;
    let attention_norm = b.take(
        &format!("blk.{i}.attn_norm.weight"),
        &[h],
        TensorRole::Vector,
    )?;
    let ffn_norm = b.take(
        &format!("blk.{i}.ffn_norm.weight"),
        &[h],
        TensorRole::Vector,
    )?;
    let mixer = match kind.mixer {
        MixerKind::Kda => MixerTensors::Kda(bind_kda(b, c, i)?),
        MixerKind::Mla => MixerTensors::Mla(bind_mla(b, c, i)?),
    };
    let ffn = match kind.ffn {
        FfnKind::Dense => FfnTensors::Dense(bind_dense(b, i, "", h, u64::from(c.dense_ffn_size))?),
        FfnKind::Moe => FfnTensors::Moe(bind_moe(b, c, i)?),
    };
    Ok(Glm5NextBlock {
        attention_hc,
        ffn_hc,
        attention_norm,
        ffn_norm,
        mixer,
        ffn,
    })
}

fn bind_hc<'a>(
    b: &mut Binder<'a>,
    c: &Glm5NextConfig,
    i: usize,
    site: &str,
) -> Result<HyperConnectionTensors<'a>> {
    let mixes = u64::from(c.hc_mix_count());
    Ok(HyperConnectionTensors {
        mix: b.take(
            &format!("blk.{i}.hc_{site}_fn.weight"),
            &[u64::from(c.hc_width()), mixes],
            TensorRole::HyperMix,
        )?,
        base: b.take(
            &format!("blk.{i}.hc_{site}_base.weight"),
            &[mixes],
            TensorRole::Vector,
        )?,
        scale: b.take(
            &format!("blk.{i}.hc_{site}_scale.weight"),
            &[3],
            TensorRole::Vector,
        )?,
    })
}

fn bind_kda<'a>(b: &mut Binder<'a>, c: &Glm5NextConfig, i: usize) -> Result<KdaTensors<'a>> {
    let h = u64::from(c.hidden_size);
    let d = u64::from(c.kda_width());
    let heads = u64::from(c.head_count);
    let head_dim = u64::from(c.kda_head_dim);
    let rank = head_dim;
    let conv = [u64::from(c.kda_conv_kernel), 1, d];
    let p = TensorRole::Projection;
    let n = |suffix: &str| format!("blk.{i}.{suffix}");
    Ok(KdaTensors {
        query: b.take(&n("attn_q.weight"), &[h, d], p)?,
        key: b.take(&n("attn_k.weight"), &[h, d], p)?,
        value: b.take(&n("attn_v.weight"), &[h, d], p)?,
        query_conv: b.take(&n("ssm_conv1d_q.weight"), &conv, TensorRole::Conv)?,
        key_conv: b.take(&n("ssm_conv1d_k.weight"), &conv, TensorRole::Conv)?,
        value_conv: b.take(&n("ssm_conv1d_v.weight"), &conv, TensorRole::Conv)?,
        decay_a: b.take(&n("ssm_f_a.weight"), &[h, rank], p)?,
        decay_b: b.take(&n("ssm_f_b.weight"), &[rank, d], p)?,
        decay_bias: b.take(&n("ssm_dt.bias"), &[d], TensorRole::Vector)?,
        neg_exp_a_log: b.take(&n("ssm_a"), &[heads], TensorRole::Vector)?,
        beta: b.take(&n("ssm_beta.weight"), &[h, heads], p)?,
        gate_a: b.take(&n("ssm_g_a.weight"), &[h, rank], p)?,
        gate_b: b.take(&n("ssm_g_b.weight"), &[rank, d], p)?,
        output_norm: b.take(&n("ssm_norm.weight"), &[head_dim], TensorRole::Vector)?,
        output: b.take(&n("attn_output.weight"), &[d, h], p)?,
    })
}

fn bind_mla<'a>(b: &mut Binder<'a>, c: &Glm5NextConfig, i: usize) -> Result<MlaTensors<'a>> {
    let h = u64::from(c.hidden_size);
    let q_rank = u64::from(c.q_lora_rank);
    let kv_rank = u64::from(c.kv_lora_rank);
    let heads = u64::from(c.head_count);
    let k_dim = u64::from(c.mla_key_head_dim);
    let v_dim = u64::from(c.mla_value_head_dim);
    let ih = u64::from(c.indexer_head_count);
    let id = u64::from(c.indexer_head_dim);
    let pool = u64::from(c.indexer_pool);
    let p = TensorRole::Projection;
    let v = TensorRole::Vector;
    let n = |suffix: &str| format!("blk.{i}.{suffix}");
    Ok(MlaTensors {
        query_a: b.take(&n("attn_q_a.weight"), &[h, q_rank], p)?,
        query_a_norm: b.take(&n("attn_q_a_norm.weight"), &[q_rank], v)?,
        query_b: b.take(&n("attn_q_b.weight"), &[q_rank, heads * k_dim], p)?,
        latent: b.take(&n("attn_kv_a_mqa.weight"), &[h, kv_rank], p)?,
        latent_norm: b.take(&n("attn_kv_a_norm.weight"), &[kv_rank], v)?,
        key_absorb: b.take(
            &n("attn_k_b.weight"),
            &[k_dim, kv_rank, heads],
            TensorRole::LatentAbsorb,
        )?,
        value_expand: b.take(
            &n("attn_v_b.weight"),
            &[kv_rank, v_dim, heads],
            TensorRole::LatentAbsorb,
        )?,
        output: b.take(&n("attn_output.weight"), &[heads * v_dim, h], p)?,
        indexer: IndexerTensors {
            query: b.take(&n("indexer.attn_q_b.weight"), &[q_rank, ih * id], p)?,
            key: b.take(&n("indexer.attn_k.weight"), &[h, id], p)?,
            key_norm: b.take(&n("indexer.k_norm.weight"), &[id], v)?,
            key_norm_bias: b.take(&n("indexer.k_norm.bias"), &[id], v)?,
            head_weights: b.take(
                &n("indexer.proj.weight"),
                &[h, ih],
                TensorRole::F32Projection,
            )?,
            pool_gate: b.take(&n("indexer_compressor_gate.weight"), &[h, id], p)?,
            pool_position: b.take(&n("indexer_compressor_ape.weight"), &[id, pool], v)?,
        },
    })
}

fn bind_dense<'a>(
    b: &mut Binder<'a>,
    i: usize,
    suffix: &str,
    h: u64,
    f: u64,
) -> Result<DenseFfnTensors<'a>> {
    let p = TensorRole::Projection;
    Ok(DenseFfnTensors {
        gate: b.take(&format!("blk.{i}.ffn_gate{suffix}.weight"), &[h, f], p)?,
        up: b.take(&format!("blk.{i}.ffn_up{suffix}.weight"), &[h, f], p)?,
        down: b.take(&format!("blk.{i}.ffn_down{suffix}.weight"), &[f, h], p)?,
    })
}

fn bind_moe<'a>(b: &mut Binder<'a>, c: &Glm5NextConfig, i: usize) -> Result<MoeTensors<'a>> {
    let h = u64::from(c.hidden_size);
    let e = u64::from(c.expert_count);
    let f = u64::from(c.expert_ffn_size);
    let gate_up = TensorRole::ExpertGateUp;
    let n = |suffix: &str| format!("blk.{i}.{suffix}");
    Ok(MoeTensors {
        router: b.take(
            &n("ffn_gate_inp.weight"),
            &[h, e],
            TensorRole::F32Projection,
        )?,
        selection_bias: b.take(&n("exp_probs_b.bias"), &[e], TensorRole::Vector)?,
        gate_experts: b.take(&n("ffn_gate_exps.weight"), &[h, f, e], gate_up)?,
        up_experts: b.take(&n("ffn_up_exps.weight"), &[h, f, e], gate_up)?,
        down_experts: b.take(
            &n("ffn_down_exps.weight"),
            &[f, h, e],
            TensorRole::ExpertDown,
        )?,
        shared: bind_dense(b, i, "_shexp", h, u64::from(c.shared_expert_ffn_size))?,
    })
}

/// NextN: an MLA block without hyper-connections, a MoE FFN and the four
/// `nextn.*` tensors. Bound for census completeness only; the binder records
/// every tensor from here on as [`TensorRole::NextN`].
fn bind_nextn_block(b: &mut Binder<'_>, c: &Glm5NextConfig, i: usize) -> Result<()> {
    b.nextn = true;
    let h = u64::from(c.hidden_size);
    b.take(
        &format!("blk.{i}.attn_norm.weight"),
        &[h],
        TensorRole::NextN,
    )?;
    b.take(&format!("blk.{i}.ffn_norm.weight"), &[h], TensorRole::NextN)?;
    bind_mla(b, c, i)?;
    bind_moe(b, c, i)?;
    b.take(
        &format!("blk.{i}.nextn.eh_proj.weight"),
        &[2 * h, h],
        TensorRole::NextN,
    )?;
    for norm in ["enorm", "hnorm", "shared_head_norm"] {
        b.take(
            &format!("blk.{i}.nextn.{norm}.weight"),
            &[h],
            TensorRole::NextN,
        )?;
    }
    Ok(())
}

struct Binder<'a> {
    inventory: BTreeMap<&'a str, &'a TensorDesc>,
    used: HashSet<&'a str>,
    trunk: Vec<BoundTensor<'a>>,
    /// Once set, every subsequent tensor is NextN regardless of its role.
    nextn: bool,
}

impl<'a> Binder<'a> {
    fn new(tensors: &'a [TensorDesc]) -> Result<Self> {
        let mut inventory = BTreeMap::new();
        for tensor in tensors {
            if inventory.insert(tensor.name.as_str(), tensor).is_some() {
                return Err(tensor_error(&tensor.name, "duplicate tensor name"));
            }
        }
        Ok(Self {
            inventory,
            used: HashSet::new(),
            trunk: Vec::with_capacity(tensors.len()),
            nextn: false,
        })
    }

    fn take(&mut self, name: &str, shape: &[u64], role: TensorRole) -> Result<&'a TensorDesc> {
        let (&key, &tensor) = self
            .inventory
            .get_key_value(name)
            .ok_or_else(|| tensor_error(name, "missing required tensor"))?;
        if tensor.shape != shape {
            return Err(tensor_error(
                name,
                format!("expected shape {shape:?}, got {:?}", tensor.shape),
            ));
        }
        let role = if self.nextn { TensorRole::NextN } else { role };
        validate_storage(tensor)?;
        if role != TensorRole::NextN && coverage::coverage(role, tensor.dtype).is_none() {
            return Err(tensor_error(
                name,
                format!(
                    "no executable path for {role:?} stored as {:?}",
                    tensor.dtype
                ),
            ));
        }
        if !self.used.insert(key) {
            return Err(tensor_error(name, "bound twice"));
        }
        self.trunk.push(BoundTensor { tensor, role });
        Ok(tensor)
    }

    fn finish(&self) -> Result<()> {
        for name in self.inventory.keys() {
            if !self.used.contains(name) {
                return Err(tensor_error(
                    name,
                    "unexpected tensor; unsupported graph or optional feature",
                ));
            }
        }
        Ok(())
    }
}

fn sum_bytes<'a>(mut tensors: impl Iterator<Item = &'a TensorDesc>) -> Result<u64> {
    tensors.try_fold(0u64, |acc, t| {
        acc.checked_add(t.n_bytes)
            .ok_or(Glm5NextError::Overflow("tensor bytes"))
    })
}

fn validate_storage(tensor: &TensorDesc) -> Result<()> {
    let (block, size) = tensor
        .dtype
        .storage_layout()
        .ok_or_else(|| tensor_error(&tensor.name, "unknown storage layout"))?;
    let width = tensor.shape.first().copied().unwrap_or(0);
    if width == 0 || !width.is_multiple_of(block) {
        return Err(tensor_error(
            &tensor.name,
            "row width is zero or not block-aligned",
        ));
    }
    let elements = tensor
        .checked_n_elements()
        .filter(|&n| n > 0)
        .ok_or_else(|| tensor_error(&tensor.name, "zero or overflowing element count"))?;
    let bytes = (elements / block)
        .checked_mul(size)
        .ok_or(Glm5NextError::Overflow("tensor bytes"))?;
    if bytes != tensor.n_bytes || tensor.data_offset.checked_add(bytes).is_none() {
        return Err(tensor_error(
            &tensor.name,
            "byte count or data range is inconsistent",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Namespace<'a>(&'a str);

impl Namespace<'_> {
    fn key(self, suffix: &str) -> String {
        format!("{}.{suffix}", self.0)
    }

    fn u32(self, metadata: &BTreeMap<String, Value>, suffix: &str) -> Result<u32> {
        required_u32(metadata, &self.key(suffix))
    }

    fn f32(self, metadata: &BTreeMap<String, Value>, suffix: &str) -> Result<f32> {
        required_f32(metadata, &self.key(suffix))
    }
}

/// Every key in the architecture namespace must be understood. MTP-only keys
/// are accepted because NextN is not executed.
fn validate_metadata_features(metadata: &BTreeMap<String, Value>, ns: Namespace<'_>) -> Result<()> {
    const KNOWN: &[&str] = &[
        "block_count",
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "vocab_size",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.layer_norm_rms_epsilon",
        "attention.layer_norm_epsilon",
        "attention.q_lora_rank",
        "attention.kv_lora_rank",
        "attention.key_length",
        "attention.value_length",
        "attention.key_length_mla",
        "attention.value_length_mla",
        "attention.indexer.head_count",
        "attention.indexer.key_length",
        "attention.indexer.top_k",
        "attention.indexer.kpool",
        "rope.dimension_count",
        "expert_count",
        "expert_used_count",
        "expert_group_count",
        "expert_group_used_count",
        "expert_gating_func",
        "expert_feed_forward_length",
        "expert_shared_feed_forward_length",
        "expert_shared_count",
        "expert_weights_scale",
        "expert_weights_norm",
        "leading_dense_block_count",
        "swiglu_clamp_exp",
        "swiglu_clamp_shexp",
        "ssm.conv_kernel",
        "kda.head_dim",
        "kda.gate_lower_bound",
        "hyper_connection.count",
        "hyper_connection.sinkhorn_iterations",
        "hyper_connection.epsilon",
        "nextn_predict_layers",
        // MTP only.
        "attention.indexer.index_share_mtp",
    ];
    let prefix = format!("{}.", ns.0);
    for (key, value) in metadata {
        if key == "general.type" && value.as_str() != Some("model") {
            return Err(unsupported(key, "expected model, not adapter/projector"));
        }
        if let Some(suffix) = key.strip_prefix(&prefix)
            && !KNOWN.contains(&suffix)
        {
            return Err(unsupported(
                key,
                "unrecognized or unsupported architecture metadata",
            ));
        }
    }
    Ok(())
}

fn uniform_f32_array(metadata: &BTreeMap<String, Value>, key: &str, len: usize) -> Result<f32> {
    let values = required(metadata, key)?
        .as_array()
        .ok_or_else(|| invalid(key, "expected number array"))?;
    if values.len() != len {
        return Err(invalid(
            key,
            format!("{} entries, expected {len}", values.len()),
        ));
    }
    let mut first = None;
    for value in values {
        let v = value
            .as_f64()
            .ok_or_else(|| invalid(key, "expected numbers"))? as f32;
        match first {
            None => first = Some(v),
            Some(f) if f32::to_bits(f) == v.to_bits() => {}
            Some(_) => return Err(unsupported(key, "per-layer values differ")),
        }
    }
    first.ok_or_else(|| invalid(key, "empty array"))
}

fn required<'a>(metadata: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a Value> {
    metadata
        .get(key)
        .ok_or_else(|| Glm5NextError::MissingMetadata(key.into()))
}

fn required_u32(metadata: &BTreeMap<String, Value>, key: &str) -> Result<u32> {
    let value = required(metadata, key)?
        .as_u64()
        .ok_or_else(|| invalid(key, "expected unsigned integer"))?;
    u32::try_from(value).map_err(|_| invalid(key, "exceeds u32"))
}

fn required_u64_array(metadata: &BTreeMap<String, Value>, key: &str) -> Result<Vec<u64>> {
    required(metadata, key)?
        .as_array()
        .ok_or_else(|| invalid(key, "expected array"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .ok_or_else(|| invalid(key, "expected unsigned integers"))
        })
        .collect()
}

fn required_f32(metadata: &BTreeMap<String, Value>, key: &str) -> Result<f32> {
    let value = required(metadata, key)?
        .as_f64()
        .ok_or_else(|| invalid(key, "expected number"))? as f32;
    if !value.is_finite() {
        return Err(invalid(key, "must fit finite f32"));
    }
    Ok(value)
}

fn required_bool(metadata: &BTreeMap<String, Value>, key: &str) -> Result<bool> {
    required(metadata, key)?
        .as_bool()
        .ok_or_else(|| invalid(key, "expected bool"))
}

fn required_str<'a>(metadata: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a str> {
    required(metadata, key)?
        .as_str()
        .ok_or_else(|| invalid(key, "expected string"))
}

fn invalid(key: &str, detail: impl Into<String>) -> Glm5NextError {
    Glm5NextError::InvalidMetadata {
        key: key.into(),
        detail: detail.into(),
    }
}

fn unsupported(key: &str, detail: impl Into<String>) -> Glm5NextError {
    Glm5NextError::Unsupported {
        key: key.into(),
        detail: detail.into(),
    }
}

fn tensor_error(name: &str, detail: impl Into<String>) -> Glm5NextError {
    Glm5NextError::Tensor {
        name: name.into(),
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests;
