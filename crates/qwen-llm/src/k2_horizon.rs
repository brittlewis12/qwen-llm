//! Checkpoint-aware K2 Horizon 7B configuration and structural GGUF binding.
//!
//! This is an explicit inspection API, not runtime admission. No family dispatch,
//! tokenizer, Metal execution, or chat capability is enabled by binding a model.
//! Checkpoint context and RoPE metadata are preserved independently of names and
//! templates. Artifact provenance and execution qualification remain separate.

use crate::gguf::{GgufError, GgufFile};
use crate::tensor::{GgmlType, TensorDesc};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

pub const ARCHITECTURE_NAME: &str = "k2-horizon";
pub const DENSE_7B_TENSOR_COUNT: usize = 327;

#[derive(Debug, thiserror::Error)]
pub enum K2HorizonError {
    #[error("missing required K2 metadata: {0}")]
    MissingMetadata(String),
    #[error("invalid K2 metadata {key:?}: {detail}")]
    InvalidMetadata { key: String, detail: String },
    #[error("unsupported K2 feature {key:?}: {detail}")]
    Unsupported { key: String, detail: String },
    #[error("invalid K2 tensor {name:?}: {detail}")]
    Tensor { name: String, detail: String },
    #[error("K2 size arithmetic overflow: {0}")]
    Overflow(&'static str),
    #[error("K2 request capacity {requested} is outside checkpoint context 1..={declared}")]
    Capacity { requested: u64, declared: u32 },
    #[error(transparent)]
    Gguf(#[from] GgufError),
}

type Result<T> = std::result::Result<T, K2HorizonError>;

#[derive(Clone, Debug, PartialEq)]
pub struct K2HorizonConfig {
    pub layer_count: u32,
    pub context_length: u32,
    pub hidden_size: u32,
    pub feed_forward_size: u32,
    pub vocab_size: u32,
    pub query_head_count: u32,
    pub kv_head_count: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub norm_groups: u32,
    pub rms_epsilon: f32,
    pub rope_dimension_count: u32,
    pub rope_theta: f32,
}

impl K2HorizonConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        Self::from_metadata(gguf.model.metadata())
    }

    fn from_metadata(metadata: &BTreeMap<String, Value>) -> Result<Self> {
        let architecture = required_str(metadata, "general.architecture")?;
        if architecture != ARCHITECTURE_NAME {
            return Err(unsupported("general.architecture", architecture));
        }
        validate_metadata_features(metadata)?;
        let tokens = required(metadata, "tokenizer.ggml.tokens")?
            .as_array()
            .ok_or_else(|| invalid("tokenizer.ggml.tokens", "expected string array"))?;
        if tokens.len() != 250_624 || tokens.iter().any(|token| !token.is_string()) {
            return Err(invalid(
                "tokenizer.ggml.tokens",
                "7B requires 250624 string entries; tokenizer conformance is a separate check",
            ));
        }
        for (key, expected) in [
            ("tokenizer.ggml.model", "gpt2"),
            ("tokenizer.ggml.pre", "k2-horizon"),
        ] {
            let actual = required_str(metadata, key)?;
            if actual != expected {
                return Err(unsupported(
                    key,
                    format!("expected {expected}, got {actual}"),
                ));
            }
        }
        let config = Self {
            layer_count: required_u32(metadata, "k2-horizon.block_count")?,
            context_length: required_u32(metadata, "k2-horizon.context_length")?,
            hidden_size: required_u32(metadata, "k2-horizon.embedding_length")?,
            feed_forward_size: required_u32(metadata, "k2-horizon.feed_forward_length")?,
            vocab_size: tokens.len() as u32,
            query_head_count: required_u32(metadata, "k2-horizon.attention.head_count")?,
            kv_head_count: required_u32(metadata, "k2-horizon.attention.head_count_kv")?,
            key_head_dim: required_u32(metadata, "k2-horizon.attention.key_length")?,
            value_head_dim: required_u32(metadata, "k2-horizon.attention.value_length")?,
            norm_groups: required_u32(metadata, "k2-horizon.attention.group_norm_groups")?,
            rms_epsilon: required_f32(metadata, "k2-horizon.attention.layer_norm_rms_epsilon")?,
            rope_dimension_count: required_u32(metadata, "k2-horizon.rope.dimension_count")?,
            rope_theta: required_f32(metadata, "k2-horizon.rope.freq_base")?,
        };
        config.validate_7b()?;
        Ok(config)
    }

    pub fn validate_7b(&self) -> Result<()> {
        for (key, actual, expected) in [
            ("k2-horizon.block_count", self.layer_count, 36),
            ("k2-horizon.embedding_length", self.hidden_size, 4096),
            (
                "k2-horizon.feed_forward_length",
                self.feed_forward_size,
                12288,
            ),
            ("tokenizer.ggml.tokens", self.vocab_size, 250624),
            ("k2-horizon.attention.head_count", self.query_head_count, 32),
            ("k2-horizon.attention.head_count_kv", self.kv_head_count, 8),
            ("k2-horizon.attention.key_length", self.key_head_dim, 128),
            (
                "k2-horizon.attention.value_length",
                self.value_head_dim,
                128,
            ),
            (
                "k2-horizon.attention.group_norm_groups",
                self.norm_groups,
                4,
            ),
            (
                "k2-horizon.rope.dimension_count",
                self.rope_dimension_count,
                128,
            ),
        ] {
            if actual != expected {
                return Err(unsupported(
                    key,
                    format!("7B expects {expected}, got {actual}"),
                ));
            }
        }
        if self.context_length == 0 {
            return Err(invalid("k2-horizon.context_length", "must be positive"));
        }
        if self.context_length > 524_288 {
            return Err(unsupported(
                "k2-horizon.context_length",
                "exceeds initial 7B profile ceiling",
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(invalid(
                "k2-horizon.rope.freq_base",
                "must be finite and positive",
            ));
        }
        if self.rms_epsilon.to_bits() != 1e-6_f32.to_bits() {
            return Err(unsupported(
                "k2-horizon.attention.layer_norm_rms_epsilon",
                "expected 1e-6",
            ));
        }
        Ok(())
    }

    /// Logical retained K/V only: no weights, scratch, paging, or snapshots.
    /// Sizing is not a guarantee of runtime or checkpoint qualification.
    pub fn kv_storage_bytes(&self, capacity: u64, storage: K2KvStorage) -> Result<u64> {
        self.validate_7b()?;
        if capacity == 0 || capacity > u64::from(self.context_length) {
            return Err(K2HorizonError::Capacity {
                requested: capacity,
                declared: self.context_length,
            });
        }
        let k = storage.row_bytes(u64::from(self.kv_head_count) * u64::from(self.key_head_dim))?;
        let v =
            storage.row_bytes(u64::from(self.kv_head_count) * u64::from(self.value_head_dim))?;
        k.checked_add(v)
            .and_then(|bytes| bytes.checked_mul(u64::from(self.layer_count)))
            .and_then(|bytes| bytes.checked_mul(capacity))
            .ok_or(K2HorizonError::Overflow("retained KV"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum K2KvStorage {
    F16,
    Q8_0,
}

impl K2KvStorage {
    pub(crate) fn row_bytes(self, elements: u64) -> Result<u64> {
        let (block, size) = match self {
            Self::F16 => (1, 2),
            Self::Q8_0 => (32, 34),
        };
        if elements == 0 || !elements.is_multiple_of(block) {
            return Err(invalid(
                "KV row",
                format!("{elements} values not aligned to block {block}"),
            ));
        }
        (elements / block)
            .checked_mul(size)
            .ok_or(K2HorizonError::Overflow("KV row"))
    }
}

#[derive(Debug)]
pub struct K2HorizonLayer<'a> {
    pub attention_norm: &'a TensorDesc,
    pub query: &'a TensorDesc,
    pub key: &'a TensorDesc,
    pub value: &'a TensorDesc,
    pub attention_output: &'a TensorDesc,
    pub feed_forward_norm: &'a TensorDesc,
    pub feed_forward_gate: &'a TensorDesc,
    pub feed_forward_up: &'a TensorDesc,
    pub feed_forward_down: &'a TensorDesc,
}

#[derive(Debug)]
pub struct K2HorizonModel<'a> {
    pub config: K2HorizonConfig,
    pub token_embedding: &'a TensorDesc,
    pub output_norm: &'a TensorDesc,
    pub output: &'a TensorDesc,
    pub layers: Vec<K2HorizonLayer<'a>>,
    pub weight_payload_bytes: u64,
}

impl<'a> K2HorizonModel<'a> {
    /// Binds descriptors without reading weight values or establishing identity.
    /// Successful inspection does not enable runtime model selection.
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self> {
        let config = K2HorizonConfig::from_gguf(gguf)?;
        let model = Self::bind(config, &gguf.tensors)?;
        for tensor in &gguf.tensors {
            gguf.try_slice(tensor)?;
        }
        Ok(model)
    }

    fn bind(config: K2HorizonConfig, tensors: &'a [TensorDesc]) -> Result<Self> {
        config.validate_7b()?;
        let mut inventory = BTreeMap::new();
        for tensor in tensors {
            if inventory.insert(tensor.name.as_str(), tensor).is_some() {
                return Err(tensor_error(&tensor.name, "duplicate tensor name"));
            }
        }
        let mut used = HashSet::new();
        let mut weight_payload_bytes = 0u64;
        let mut take = |name: String, shape: &[u64], norm: bool| -> Result<&'a TensorDesc> {
            let tensor = inventory
                .get(name.as_str())
                .copied()
                .ok_or_else(|| tensor_error(&name, "missing required tensor"))?;
            if tensor.shape != shape {
                return Err(tensor_error(
                    &name,
                    format!("expected shape {shape:?}, got {:?}", tensor.shape),
                ));
            }
            validate_storage(tensor, norm)?;
            weight_payload_bytes = weight_payload_bytes
                .checked_add(tensor.n_bytes)
                .ok_or(K2HorizonError::Overflow("weight payload"))?;
            used.insert(name);
            Ok(tensor)
        };
        let h = u64::from(config.hidden_size);
        let f = u64::from(config.feed_forward_size);
        let q = u64::from(config.query_head_count) * u64::from(config.key_head_dim);
        let k = u64::from(config.kv_head_count) * u64::from(config.key_head_dim);
        let v = u64::from(config.kv_head_count) * u64::from(config.value_head_dim);
        let token_embedding = take(
            "token_embd.weight".into(),
            &[h, u64::from(config.vocab_size)],
            false,
        )?;
        let output_norm = take("output_norm.weight".into(), &[h], true)?;
        let output = take(
            "output.weight".into(),
            &[h, u64::from(config.vocab_size)],
            false,
        )?;
        let mut layers = Vec::with_capacity(config.layer_count as usize);
        for i in 0..config.layer_count {
            layers.push(K2HorizonLayer {
                attention_norm: take(format!("blk.{i}.attn_norm.weight"), &[h], true)?,
                query: take(format!("blk.{i}.attn_q.weight"), &[h, q], false)?,
                key: take(format!("blk.{i}.attn_k.weight"), &[h, k], false)?,
                value: take(format!("blk.{i}.attn_v.weight"), &[h, v], false)?,
                attention_output: take(format!("blk.{i}.attn_output.weight"), &[q, h], false)?,
                feed_forward_norm: take(format!("blk.{i}.ffn_norm.weight"), &[h], true)?,
                feed_forward_gate: take(format!("blk.{i}.ffn_gate.weight"), &[h, f], false)?,
                feed_forward_up: take(format!("blk.{i}.ffn_up.weight"), &[h, f], false)?,
                feed_forward_down: take(format!("blk.{i}.ffn_down.weight"), &[f, h], false)?,
            });
        }
        for name in inventory.keys() {
            if !used.contains(*name) {
                return Err(tensor_error(
                    name,
                    "unexpected tensor; unsupported graph or optional feature",
                ));
            }
        }
        Ok(Self {
            config,
            token_embedding,
            output_norm,
            output,
            layers,
            weight_payload_bytes,
        })
    }
}

fn validate_storage(tensor: &TensorDesc, norm: bool) -> Result<()> {
    let admitted = if norm {
        tensor.dtype == GgmlType::F32
    } else {
        matches!(
            tensor.dtype,
            GgmlType::F32
                | GgmlType::F16
                | GgmlType::BF16
                | GgmlType::Q8_0
                | GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
        )
    };
    if !admitted {
        return Err(tensor_error(
            &tensor.name,
            format!(
                "unsupported {} storage {:?}",
                if norm { "norm" } else { "matrix" },
                tensor.dtype
            ),
        ));
    }
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
        .ok_or(K2HorizonError::Overflow("tensor bytes"))?;
    if bytes != tensor.n_bytes || tensor.data_offset.checked_add(bytes).is_none() {
        return Err(tensor_error(
            &tensor.name,
            "byte count or data range is inconsistent",
        ));
    }
    Ok(())
}

fn validate_metadata_features(metadata: &BTreeMap<String, Value>) -> Result<()> {
    const REQUIRED: &[&str] = &[
        "block_count",
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "attention.group_norm_groups",
        "attention.layer_norm_rms_epsilon",
        "rope.dimension_count",
        "rope.freq_base",
    ];
    for (key, value) in metadata {
        if key.starts_with("prism.hadamard.") {
            return Err(unsupported(
                key,
                "rotated weights require a different execution contract",
            ));
        }
        if key == "general.type" && value.as_str() != Some("model") {
            return Err(unsupported(key, "expected model, not adapter/projector"));
        }
        let Some(suffix) = key.strip_prefix("k2-horizon.") else {
            continue;
        };
        if REQUIRED.contains(&suffix) {
            continue;
        }
        match suffix {
            "expert_count"
            | "expert_used_count"
            | "attention.value_expert_count"
            | "attention.value_expert_used_count"
            | "attention.sliding_window" => {
                let count = value
                    .as_u64()
                    .ok_or_else(|| invalid(key, "expected unsigned integer"))?;
                if count != 0 {
                    return Err(unsupported(key, "dense 7B requires zero"));
                }
            }
            "rope.scaling.type" if value.as_str() == Some("none") => {}
            _ => {
                return Err(unsupported(
                    key,
                    "unrecognized or unsupported architecture metadata",
                ));
            }
        }
    }
    Ok(())
}

fn required<'a>(metadata: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a Value> {
    metadata
        .get(key)
        .ok_or_else(|| K2HorizonError::MissingMetadata(key.into()))
}

fn required_u32(metadata: &BTreeMap<String, Value>, key: &str) -> Result<u32> {
    let value = required(metadata, key)?
        .as_u64()
        .ok_or_else(|| invalid(key, "expected unsigned integer"))?;
    u32::try_from(value).map_err(|_| invalid(key, "exceeds u32"))
}

fn required_f32(metadata: &BTreeMap<String, Value>, key: &str) -> Result<f32> {
    let value = required(metadata, key)?
        .as_f64()
        .ok_or_else(|| invalid(key, "expected number"))?;
    let value = value as f32;
    if !value.is_finite() {
        return Err(invalid(key, "must fit finite f32"));
    }
    Ok(value)
}

fn required_str<'a>(metadata: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a str> {
    required(metadata, key)?
        .as_str()
        .ok_or_else(|| invalid(key, "expected string"))
}

fn invalid(key: &str, detail: impl Into<String>) -> K2HorizonError {
    K2HorizonError::InvalidMetadata {
        key: key.into(),
        detail: detail.into(),
    }
}

fn unsupported(key: &str, detail: impl Into<String>) -> K2HorizonError {
    K2HorizonError::Unsupported {
        key: key.into(),
        detail: detail.into(),
    }
}

fn tensor_error(name: &str, detail: impl Into<String>) -> K2HorizonError {
    K2HorizonError::Tensor {
        name: name.into(),
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests;
