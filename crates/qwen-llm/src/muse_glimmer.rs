//! Muse Glimmer 30B text architecture and strict GGUF tensor binding.
//!
//! This module deliberately stops before execution. Muse is an independent
//! model family: its all-attention residual graph, mixed sliding/full cache,
//! and norm placement do not fit the Qwen hybrid [`crate::model::Arch`].

use crate::gguf::{GgufError, GgufFile};
use crate::tensor::{GgmlType, TensorDesc};
use crate::tokenizer::{
    MUSE_GLIMMER_RELEASE_TOKENIZER_IDENTITY_SHA256, Tokenize,
    muse_glimmer_tokenizer_identity_sha256,
};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const ARCHITECTURE_NAME: &str = "muse-glimmer";
pub const POST_NORM_EPSILON: f32 = 1e-8;

const RELEASE_LAYER_COUNT: u32 = 52;
const RELEASE_CONTEXT_LENGTH: u32 = 131_072;
const RELEASE_HIDDEN_SIZE: u32 = 6_656;
const RELEASE_FEED_FORWARD_SIZE: u32 = 19_968;
const RELEASE_VOCAB_SIZE: u32 = 202_048;
const RELEASE_QUERY_HEADS: u32 = 32;
const RELEASE_KV_HEADS: u32 = 2;
const RELEASE_HEAD_DIM: u32 = 128;
const RELEASE_SLIDING_WINDOW: u32 = 2_048;
const RELEASE_ROPE_THETA: f32 = 500_000.0;
const RELEASE_RMS_EPSILON: f32 = 1e-5;
const RELEASE_LOGIT_SCALE: f32 = 0.196_116_13;
const RELEASE_LOGIT_SOFTCAP: f32 = 20.0;
const RELEASE_QK_SCALE: f32 = 3.87;
const RELEASE_MERGE_COUNT: usize = 300_000;
pub const RELEASE_TENSOR_COUNT: usize = 731;
pub const RELEASE_MATRIX_TENSOR_COUNT: usize = 418;
pub const RELEASE_NORM_TENSOR_COUNT: usize = 313;

const META_FIXED_CHAT_SHA256: [u8; 32] = [
    0xcf, 0xc6, 0x7e, 0x5f, 0x34, 0x9f, 0x37, 0x69, 0x0d, 0xfd, 0x31, 0xed, 0x1f, 0x18, 0xbc, 0x44,
    0x42, 0xa9, 0xdd, 0x32, 0xfe, 0x39, 0xa6, 0x48, 0xf9, 0x93, 0xcb, 0x4e, 0xb3, 0xca, 0xe6, 0x78,
];
const UNSLOTH_LAUNCH_CHAT_SHA256: [u8; 32] = [
    0x11, 0x4f, 0x55, 0xeb, 0xdc, 0x18, 0x04, 0xc1, 0xaf, 0x37, 0x11, 0x97, 0xb9, 0xfd, 0xf2, 0xd6,
    0xbb, 0x92, 0x59, 0x66, 0xc9, 0xdf, 0xe4, 0x6b, 0x73, 0x78, 0x2a, 0x71, 0xbc, 0x07, 0x96, 0x5e,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MuseGlimmerChatTemplateProfile {
    MetaFixed,
    UnslothLaunch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MuseGlimmerArtifactProfile {
    UnslothQ8_0,
    UnslothBf16,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerConfig {
    pub layer_count: u32,
    pub context_length: u32,
    pub hidden_size: u32,
    pub feed_forward_size: u32,
    pub vocab_size: u32,
    pub query_head_count: u32,
    pub kv_head_count: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub rope_theta: f32,
    pub rms_epsilon: f32,
    pub post_norm_epsilon: f32,
    pub sliding_window: u32,
    pub sliding_layers: Vec<bool>,
    pub logit_scale: f32,
    pub final_logit_softcap: f32,
    pub tokenizer_model: String,
    pub tokenizer_pre: String,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub eot_token_id: u32,
    pub padding_token_id: u32,
    pub add_bos_token: bool,
    pub add_sep_token: bool,
    pub tokenizer_identity_sha256: [u8; 32],
    pub chat_template_sha256: [u8; 32],
    pub chat_template_profile: MuseGlimmerChatTemplateProfile,
}

impl MuseGlimmerConfig {
    /// The tokenizer must agree with the model on vocabulary size, BOS and
    /// EOS; every lane that opens a Muse tokenizer checks this before use.
    pub fn validate_tokenizer(&self, tokenizer: &impl Tokenize) -> Result<(), MuseGlimmerError> {
        let mismatch = |field, tokenizer: String, model: String| MuseGlimmerError::TokenizerContract {
            field,
            tokenizer,
            model,
        };
        if tokenizer.n_vocab() != self.vocab_size {
            return Err(mismatch(
                "vocabulary",
                tokenizer.n_vocab().to_string(),
                self.vocab_size.to_string(),
            ));
        }
        if tokenizer.bos() != Some(self.bos_token_id as i32) {
            return Err(mismatch(
                "BOS",
                format!("{:?}", tokenizer.bos()),
                self.bos_token_id.to_string(),
            ));
        }
        if tokenizer.eos() != Some(self.eos_token_id as i32) {
            return Err(mismatch(
                "EOS",
                format!("{:?}", tokenizer.eos()),
                self.eos_token_id.to_string(),
            ));
        }
        Ok(())
    }

    /// The release's producer-declared stop set is exactly `[EOS, EOT]`.
    pub fn expected_stop_tokens(&self) -> [i32; 2] {
        [self.eos_token_id as i32, self.eot_token_id as i32]
    }

    pub fn validate_stop_tokens(&self, stop_tokens: &[i32]) -> Result<(), MuseGlimmerError> {
        let expected = self.expected_stop_tokens();
        if stop_tokens != expected {
            return Err(MuseGlimmerError::StopTokens {
                expected: expected.to_vec(),
                actual: stop_tokens.to_vec(),
            });
        }
        Ok(())
    }

    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, MuseGlimmerError> {
        let architecture = gguf.architecture();
        if architecture.as_deref() != Some(ARCHITECTURE_NAME) {
            return Err(MuseGlimmerError::UnsupportedArchitecture(architecture));
        }

        let layer_count = required_u32(gguf, "muse-glimmer.block_count")?;
        let sliding_layers = parse_sliding_pattern(gguf, layer_count)?;
        let tokens = required_string_array(gguf, "tokenizer.ggml.tokens")?;
        let vocab_size = u32::try_from(tokens.len()).map_err(|_| {
            invalid(
                "tokenizer.ggml.tokens",
                format!("array length {} exceeds u32", tokens.len()),
            )
        })?;
        validate_tokenizer_arrays(gguf, &tokens)?;
        let chat_template = required_str(gguf, "tokenizer.chat_template")?;
        for marker in ["<|start|>assistant", "<|eom|>", "<atem:function_calls>"] {
            if !chat_template.contains(marker) {
                return Err(invalid(
                    "tokenizer.chat_template",
                    format!("missing Muse ATEM marker {marker:?}"),
                ));
            }
        }
        let chat_template_sha256: [u8; 32] = Sha256::digest(chat_template.as_bytes()).into();
        let chat_template_profile = classify_chat_template(chat_template_sha256)?;
        let tokenizer_identity_sha256 = muse_glimmer_tokenizer_identity_sha256(gguf)?;

        let config = Self {
            layer_count,
            context_length: required_u32(gguf, "muse-glimmer.context_length")?,
            hidden_size: required_u32(gguf, "muse-glimmer.embedding_length")?,
            feed_forward_size: required_u32(gguf, "muse-glimmer.feed_forward_length")?,
            vocab_size,
            query_head_count: required_u32(gguf, "muse-glimmer.attention.head_count")?,
            kv_head_count: required_u32(gguf, "muse-glimmer.attention.head_count_kv")?,
            key_head_dim: required_u32(gguf, "muse-glimmer.attention.key_length")?,
            value_head_dim: required_u32(gguf, "muse-glimmer.attention.value_length")?,
            rope_theta: required_f32(gguf, "muse-glimmer.rope.freq_base")?,
            rms_epsilon: required_f32(gguf, "muse-glimmer.attention.layer_norm_rms_epsilon")?,
            post_norm_epsilon: POST_NORM_EPSILON,
            sliding_window: required_u32(gguf, "muse-glimmer.attention.sliding_window")?,
            sliding_layers,
            logit_scale: required_f32(gguf, "muse-glimmer.logit_scale")?,
            final_logit_softcap: required_f32(gguf, "muse-glimmer.final_logit_softcapping")?,
            tokenizer_model: required_str(gguf, "tokenizer.ggml.model")?.to_owned(),
            tokenizer_pre: required_str(gguf, "tokenizer.ggml.pre")?.to_owned(),
            bos_token_id: required_u32(gguf, "tokenizer.ggml.bos_token_id")?,
            eos_token_id: required_u32(gguf, "tokenizer.ggml.eos_token_id")?,
            eot_token_id: required_u32(gguf, "tokenizer.ggml.eot_token_id")?,
            padding_token_id: required_u32(gguf, "tokenizer.ggml.padding_token_id")?,
            add_bos_token: required_bool(gguf, "tokenizer.ggml.add_bos_token")?,
            add_sep_token: required_bool(gguf, "tokenizer.ggml.add_sep_token")?,
            tokenizer_identity_sha256,
            chat_template_sha256,
            chat_template_profile,
        };
        config.validate_release_profile()?;
        Ok(config)
    }

    pub fn release_reference() -> Self {
        Self {
            layer_count: RELEASE_LAYER_COUNT,
            context_length: RELEASE_CONTEXT_LENGTH,
            hidden_size: RELEASE_HIDDEN_SIZE,
            feed_forward_size: RELEASE_FEED_FORWARD_SIZE,
            vocab_size: RELEASE_VOCAB_SIZE,
            query_head_count: RELEASE_QUERY_HEADS,
            kv_head_count: RELEASE_KV_HEADS,
            key_head_dim: RELEASE_HEAD_DIM,
            value_head_dim: RELEASE_HEAD_DIM,
            rope_theta: RELEASE_ROPE_THETA,
            rms_epsilon: RELEASE_RMS_EPSILON,
            post_norm_epsilon: POST_NORM_EPSILON,
            sliding_window: RELEASE_SLIDING_WINDOW,
            sliding_layers: release_sliding_pattern(),
            logit_scale: RELEASE_LOGIT_SCALE,
            final_logit_softcap: RELEASE_LOGIT_SOFTCAP,
            tokenizer_model: "gpt2".into(),
            tokenizer_pre: "llama4".into(),
            bos_token_id: 200_000,
            eos_token_id: 200_001,
            eot_token_id: 200_008,
            padding_token_id: 200_018,
            add_bos_token: true,
            add_sep_token: false,
            tokenizer_identity_sha256: MUSE_GLIMMER_RELEASE_TOKENIZER_IDENTITY_SHA256,
            chat_template_sha256: META_FIXED_CHAT_SHA256,
            chat_template_profile: MuseGlimmerChatTemplateProfile::MetaFixed,
        }
    }

    pub fn unsloth_release_reference() -> Self {
        Self {
            chat_template_sha256: UNSLOTH_LAUNCH_CHAT_SHA256,
            chat_template_profile: MuseGlimmerChatTemplateProfile::UnslothLaunch,
            ..Self::release_reference()
        }
    }

    pub fn validate_release_profile(&self) -> Result<(), MuseGlimmerError> {
        require_exact(
            "muse-glimmer.block_count",
            self.layer_count,
            RELEASE_LAYER_COUNT,
        )?;
        require_exact(
            "muse-glimmer.context_length",
            self.context_length,
            RELEASE_CONTEXT_LENGTH,
        )?;
        require_exact(
            "muse-glimmer.embedding_length",
            self.hidden_size,
            RELEASE_HIDDEN_SIZE,
        )?;
        require_exact(
            "muse-glimmer.feed_forward_length",
            self.feed_forward_size,
            RELEASE_FEED_FORWARD_SIZE,
        )?;
        require_exact("tokenizer.ggml.tokens", self.vocab_size, RELEASE_VOCAB_SIZE)?;
        require_exact(
            "muse-glimmer.attention.head_count",
            self.query_head_count,
            RELEASE_QUERY_HEADS,
        )?;
        require_exact(
            "muse-glimmer.attention.head_count_kv",
            self.kv_head_count,
            RELEASE_KV_HEADS,
        )?;
        require_exact(
            "muse-glimmer.attention.key_length",
            self.key_head_dim,
            RELEASE_HEAD_DIM,
        )?;
        require_exact(
            "muse-glimmer.attention.value_length",
            self.value_head_dim,
            RELEASE_HEAD_DIM,
        )?;
        require_exact(
            "muse-glimmer.attention.sliding_window",
            self.sliding_window,
            RELEASE_SLIDING_WINDOW,
        )?;
        require_f32(
            "muse-glimmer.rope.freq_base",
            self.rope_theta,
            RELEASE_ROPE_THETA,
        )?;
        require_f32(
            "muse-glimmer.attention.layer_norm_rms_epsilon",
            self.rms_epsilon,
            RELEASE_RMS_EPSILON,
        )?;
        require_f32(
            "muse-glimmer.post_norm_epsilon",
            self.post_norm_epsilon,
            POST_NORM_EPSILON,
        )?;
        require_f32(
            "muse-glimmer.logit_scale",
            self.logit_scale,
            RELEASE_LOGIT_SCALE,
        )?;
        require_f32(
            "muse-glimmer.final_logit_softcapping",
            self.final_logit_softcap,
            RELEASE_LOGIT_SOFTCAP,
        )?;

        if !self
            .query_head_count
            .is_multiple_of(self.kv_head_count.max(1))
        {
            return Err(invalid(
                "muse-glimmer.attention.head_count",
                "must be divisible by attention.head_count_kv",
            ));
        }
        if self.sliding_layers != release_sliding_pattern() {
            return Err(invalid(
                "muse-glimmer.attention.sliding_window_pattern",
                "expected [sliding, sliding, sliding, full] repeated 13 times",
            ));
        }
        if self.tokenizer_model != "gpt2" || self.tokenizer_pre != "llama4" {
            return Err(invalid(
                "tokenizer.ggml.pre",
                format!(
                    "expected model/pre gpt2/llama4, got {}/{}",
                    self.tokenizer_model, self.tokenizer_pre
                ),
            ));
        }
        for (key, token, expected) in [
            ("tokenizer.ggml.bos_token_id", self.bos_token_id, 200_000),
            ("tokenizer.ggml.eos_token_id", self.eos_token_id, 200_001),
            ("tokenizer.ggml.eot_token_id", self.eot_token_id, 200_008),
            (
                "tokenizer.ggml.padding_token_id",
                self.padding_token_id,
                200_018,
            ),
        ] {
            require_exact(key, token, expected)?;
        }
        if !self.add_bos_token || self.add_sep_token {
            return Err(invalid(
                "tokenizer.ggml.add_bos_token",
                "released Muse tokenizer requires add_bos=true and add_sep=false",
            ));
        }
        if self.tokenizer_identity_sha256 != MUSE_GLIMMER_RELEASE_TOKENIZER_IDENTITY_SHA256 {
            return Err(invalid(
                "Muse tokenizer identity",
                format!(
                    "unrecognized SHA-256 {}",
                    hex_digest(self.tokenizer_identity_sha256)
                ),
            ));
        }
        let classified = classify_chat_template(self.chat_template_sha256)?;
        if classified != self.chat_template_profile {
            return Err(invalid(
                "tokenizer.chat_template",
                "template hash/profile mismatch",
            ));
        }
        Ok(())
    }

    pub fn query_width(&self) -> Result<u64, MuseGlimmerError> {
        checked_mul(
            self.query_head_count as u64,
            self.key_head_dim as u64,
            "query heads * key head dimension",
        )
    }

    pub fn kv_width(&self) -> Result<u64, MuseGlimmerError> {
        checked_mul(
            self.kv_head_count as u64,
            self.key_head_dim as u64,
            "KV heads * key head dimension",
        )
    }

    pub fn is_sliding_layer(&self, layer: usize) -> Option<bool> {
        self.sliding_layers.get(layer).copied()
    }
}

#[derive(Clone)]
pub struct MuseGlimmerLayer<'a> {
    pub attention_norm: &'a TensorDesc,
    pub attention_query: &'a TensorDesc,
    pub attention_key: &'a TensorDesc,
    pub attention_value: &'a TensorDesc,
    pub attention_gate: &'a TensorDesc,
    pub attention_output: &'a TensorDesc,
    pub query_norm: &'a TensorDesc,
    pub key_norm: &'a TensorDesc,
    pub post_attention_norm: &'a TensorDesc,
    pub feed_forward_norm: &'a TensorDesc,
    pub feed_forward_gate: &'a TensorDesc,
    pub feed_forward_up: &'a TensorDesc,
    pub feed_forward_down: &'a TensorDesc,
    pub post_feed_forward_norm: &'a TensorDesc,
    pub sliding_attention: bool,
}

pub struct MuseGlimmerModel<'a> {
    pub config: MuseGlimmerConfig,
    pub artifact_profile: MuseGlimmerArtifactProfile,
    pub token_embedding: &'a TensorDesc,
    pub output_norm: &'a TensorDesc,
    pub output: &'a TensorDesc,
    pub layers: Vec<MuseGlimmerLayer<'a>>,
}

impl<'a> MuseGlimmerModel<'a> {
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, MuseGlimmerError> {
        let config = MuseGlimmerConfig::from_gguf(gguf)?;
        let hidden = config.hidden_size as u64;
        let feed_forward = config.feed_forward_size as u64;
        let vocab = config.vocab_size as u64;
        let query = config.query_width()?;
        let kv = config.kv_width()?;
        let head = config.key_head_dim as u64;
        let mut used = HashSet::with_capacity(expected_tensor_count(config.layer_count));

        let token_embedding = bind_matrix(
            gguf,
            &mut used,
            "token_embd.weight".into(),
            &[hidden, vocab],
        )?;
        let output_norm = bind_norm(gguf, &mut used, "output_norm.weight".into(), &[hidden])?;
        let output = bind_matrix(gguf, &mut used, "output.weight".into(), &[hidden, vocab])?;

        let mut layers = Vec::with_capacity(config.layer_count as usize);
        for layer in 0..config.layer_count {
            let prefix = format!("blk.{layer}");
            let norm = |suffix: &str, expected: &[u64], used: &mut HashSet<String>| {
                bind_norm(gguf, used, format!("{prefix}.{suffix}"), expected)
            };
            let matrix = |suffix: &str, expected: &[u64], used: &mut HashSet<String>| {
                bind_matrix(gguf, used, format!("{prefix}.{suffix}"), expected)
            };
            let query_norm = norm("attn_q_norm.weight", &[head], &mut used)?;
            let key_norm = norm("attn_k_norm.weight", &[head], &mut used)?;
            validate_constant_f32(gguf, query_norm, RELEASE_QK_SCALE)?;
            validate_constant_f32(gguf, key_norm, 1.0)?;
            layers.push(MuseGlimmerLayer {
                attention_norm: norm("attn_norm.weight", &[hidden], &mut used)?,
                attention_query: matrix("attn_q.weight", &[hidden, query], &mut used)?,
                attention_key: matrix("attn_k.weight", &[hidden, kv], &mut used)?,
                attention_value: matrix("attn_v.weight", &[hidden, kv], &mut used)?,
                attention_gate: matrix("attn_gate.weight", &[hidden, query], &mut used)?,
                attention_output: matrix("attn_output.weight", &[query, hidden], &mut used)?,
                query_norm,
                key_norm,
                post_attention_norm: norm("post_attention_norm.weight", &[hidden], &mut used)?,
                feed_forward_norm: norm("ffn_norm.weight", &[hidden], &mut used)?,
                feed_forward_gate: matrix("ffn_gate.weight", &[hidden, feed_forward], &mut used)?,
                feed_forward_up: matrix("ffn_up.weight", &[hidden, feed_forward], &mut used)?,
                feed_forward_down: matrix("ffn_down.weight", &[feed_forward, hidden], &mut used)?,
                post_feed_forward_norm: norm("post_ffw_norm.weight", &[hidden], &mut used)?,
                sliding_attention: config.sliding_layers[layer as usize],
            });
        }

        let mut unexpected: Vec<_> = gguf
            .tensors
            .iter()
            .filter(|tensor| !used.contains(tensor.name.as_str()))
            .map(|tensor| tensor.name.clone())
            .collect();
        unexpected.sort();
        if !unexpected.is_empty() {
            return Err(MuseGlimmerError::UnexpectedTensors(unexpected));
        }
        if used.len() != expected_tensor_count(config.layer_count) {
            return Err(invalid(
                "muse-glimmer tensor inventory",
                format!(
                    "bound {} unique tensors, expected {}",
                    used.len(),
                    expected_tensor_count(config.layer_count)
                ),
            ));
        }
        let artifact_profile = classify_artifact_profile(
            token_embedding,
            output,
            &layers,
            config.chat_template_profile,
        )?;

        Ok(Self {
            config,
            artifact_profile,
            token_embedding,
            output_norm,
            output,
            layers,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MuseGlimmerError {
    #[error("unsupported architecture: {0:?}")]
    UnsupportedArchitecture(Option<String>),
    #[error("missing required metadata key: {0}")]
    MissingMetadata(&'static str),
    #[error("invalid metadata key {key:?}: {reason}")]
    InvalidMetadata { key: &'static str, reason: String },
    #[error("missing required tensor: {0}")]
    MissingTensor(String),
    #[error("shape mismatch for {name}: expected {expected:?}, got {actual:?}")]
    Shape {
        name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("tensor {name} must use F32 norm storage, got {dtype:?}")]
    NormDtype { name: String, dtype: GgmlType },
    #[error("tensor {name} uses unsupported or unknown GGML dtype {dtype:?}")]
    UnknownDtype { name: String, dtype: GgmlType },
    #[error(
        "tensor {name} row width {row_width} is not aligned to {block_size}-element {dtype:?} blocks"
    )]
    RowAlignment {
        name: String,
        row_width: u64,
        block_size: u64,
        dtype: GgmlType,
    },
    #[error("metadata-derived tensor dimension overflow: {0}")]
    DimensionOverflow(&'static str),
    #[error("invalid tensor payload for {tensor}: {detail}")]
    InvalidTensorValue { tensor: String, detail: String },
    #[error("unsupported Muse Glimmer artifact dtype profile: {0}")]
    UnsupportedArtifactProfile(String),
    #[error("Muse Glimmer tokenizer {field} {tokenizer} differs from model {field} {model}")]
    TokenizerContract {
        field: &'static str,
        tokenizer: String,
        model: String,
    },
    #[error("Muse Glimmer release stop tokens must be EOS/EOT {expected:?}, got {actual:?}")]
    StopTokens { expected: Vec<i32>, actual: Vec<i32> },
    #[error("unexpected Muse Glimmer tensor set: {0:?}")]
    UnexpectedTensors(Vec<String>),
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Tokenizer(#[from] crate::tokenizer::TokError),
}

fn parse_sliding_pattern(gguf: &GgufFile, layer_count: u32) -> Result<Vec<bool>, MuseGlimmerError> {
    const KEY: &str = "muse-glimmer.attention.sliding_window_pattern";
    if let Some(period) = gguf.get_u64(KEY) {
        let period = u32::try_from(period).map_err(|_| invalid(KEY, "period exceeds u32"))?;
        return normalize_sliding_pattern(layer_count, Some(period), None);
    }
    let pattern = gguf
        .get_bool_array(KEY)?
        .ok_or(MuseGlimmerError::MissingMetadata(KEY))?;
    normalize_sliding_pattern(layer_count, None, Some(pattern))
}

fn normalize_sliding_pattern(
    layer_count: u32,
    period: Option<u32>,
    explicit: Option<Vec<bool>>,
) -> Result<Vec<bool>, MuseGlimmerError> {
    const KEY: &str = "muse-glimmer.attention.sliding_window_pattern";
    match (period, explicit) {
        (Some(0), _) => Err(invalid(KEY, "period must be nonzero")),
        (Some(period), None) => Ok((0..layer_count)
            .map(|layer| layer % period != period - 1)
            .collect()),
        (None, Some(pattern)) if pattern.len() == layer_count as usize => Ok(pattern),
        (None, Some(pattern)) => Err(invalid(
            KEY,
            format!(
                "has {} entries for {layer_count} decoder layers",
                pattern.len()
            ),
        )),
        (Some(_), Some(_)) => Err(invalid(
            KEY,
            "cannot declare scalar and array forms together",
        )),
        (None, None) => Err(MuseGlimmerError::MissingMetadata(KEY)),
    }
}

fn release_sliding_pattern() -> Vec<bool> {
    (0..RELEASE_LAYER_COUNT)
        .map(|layer| layer % 4 != 3)
        .collect()
}

fn expected_tensor_count(layer_count: u32) -> usize {
    3 + layer_count as usize * 14
}

fn classify_chat_template(
    sha256: [u8; 32],
) -> Result<MuseGlimmerChatTemplateProfile, MuseGlimmerError> {
    match sha256 {
        META_FIXED_CHAT_SHA256 => Ok(MuseGlimmerChatTemplateProfile::MetaFixed),
        UNSLOTH_LAUNCH_CHAT_SHA256 => Ok(MuseGlimmerChatTemplateProfile::UnslothLaunch),
        other => Err(invalid(
            "tokenizer.chat_template",
            format!("unrecognized SHA-256 {}", hex_digest(other)),
        )),
    }
}

fn classify_artifact_profile(
    token_embedding: &TensorDesc,
    output: &TensorDesc,
    layers: &[MuseGlimmerLayer<'_>],
    chat_profile: MuseGlimmerChatTemplateProfile,
) -> Result<MuseGlimmerArtifactProfile, MuseGlimmerError> {
    let mut dtypes = Vec::with_capacity(RELEASE_MATRIX_TENSOR_COUNT);
    dtypes.extend([token_embedding.dtype, output.dtype]);
    for layer in layers {
        dtypes.extend([
            layer.attention_query.dtype,
            layer.attention_key.dtype,
            layer.attention_value.dtype,
            layer.attention_gate.dtype,
            layer.attention_output.dtype,
            layer.feed_forward_gate.dtype,
            layer.feed_forward_up.dtype,
            layer.feed_forward_down.dtype,
        ]);
    }
    classify_matrix_dtypes(&dtypes, chat_profile)
}

fn classify_matrix_dtypes(
    dtypes: &[GgmlType],
    chat_profile: MuseGlimmerChatTemplateProfile,
) -> Result<MuseGlimmerArtifactProfile, MuseGlimmerError> {
    if dtypes.len() != RELEASE_MATRIX_TENSOR_COUNT {
        return Err(MuseGlimmerError::UnsupportedArtifactProfile(format!(
            "expected {RELEASE_MATRIX_TENSOR_COUNT} matrices, got {}",
            dtypes.len()
        )));
    }
    if chat_profile == MuseGlimmerChatTemplateProfile::UnslothLaunch
        && dtypes.iter().all(|&dtype| dtype == GgmlType::Q8_0)
    {
        return Ok(MuseGlimmerArtifactProfile::UnslothQ8_0);
    }
    if chat_profile == MuseGlimmerChatTemplateProfile::UnslothLaunch
        && dtypes.iter().all(|&dtype| dtype == GgmlType::BF16)
    {
        return Ok(MuseGlimmerArtifactProfile::UnslothBf16);
    }
    let mut census = std::collections::BTreeMap::<String, usize>::new();
    for dtype in dtypes {
        *census.entry(format!("{dtype:?}")).or_default() += 1;
    }
    Err(MuseGlimmerError::UnsupportedArtifactProfile(format!(
        "initial lane requires UnslothLaunch chat plus uniform Q8_0 or BF16 matrices; got chat={chat_profile:?}, matrices={census:?}"
    )))
}

fn validate_constant_f32(
    gguf: &GgufFile,
    tensor: &TensorDesc,
    expected: f32,
) -> Result<(), MuseGlimmerError> {
    let bytes = gguf.try_slice(tensor)?;
    let width = std::mem::size_of::<f32>();
    let mut chunks = bytes.chunks_exact(width);
    for (index, chunk) in chunks.by_ref().enumerate() {
        let value = f32::from_le_bytes(chunk.try_into().expect("four-byte F32 chunk"));
        if value.to_bits() != expected.to_bits() {
            return Err(MuseGlimmerError::InvalidTensorValue {
                tensor: tensor.name.clone(),
                detail: format!("entry {index} expected {expected}, got {value}"),
            });
        }
    }
    if !chunks.remainder().is_empty() {
        return Err(MuseGlimmerError::InvalidTensorValue {
            tensor: tensor.name.clone(),
            detail: "F32 payload has a non-four-byte remainder".into(),
        });
    }
    Ok(())
}

fn validate_tokenizer_arrays(gguf: &GgufFile, tokens: &[&str]) -> Result<(), MuseGlimmerError> {
    if tokens.len() != RELEASE_VOCAB_SIZE as usize {
        return Err(invalid(
            "tokenizer.ggml.tokens",
            format!(
                "expected {RELEASE_VOCAB_SIZE} strings, got {}",
                tokens.len()
            ),
        ));
    }
    let token_types = gguf.get_i64_array("tokenizer.ggml.token_type")?.ok_or(
        MuseGlimmerError::MissingMetadata("tokenizer.ggml.token_type"),
    )?;
    if token_types.len() != tokens.len() || token_types.iter().any(|&kind| !(0..=6).contains(&kind))
    {
        return Err(invalid(
            "tokenizer.ggml.token_type",
            "must contain one valid GGUF token type per vocabulary entry",
        ));
    }
    let merges = required_string_array(gguf, "tokenizer.ggml.merges")?;
    if merges.len() != RELEASE_MERGE_COUNT {
        return Err(invalid(
            "tokenizer.ggml.merges",
            format!(
                "expected {RELEASE_MERGE_COUNT} strings, got {}",
                merges.len()
            ),
        ));
    }
    for (id, expected) in [
        (200_000, "<|begin_of_text|>"),
        (200_001, "<|end_of_text|>"),
        (200_007, "<|eom|>"),
        (200_008, "<|eot|>"),
        (200_018, "<|finetune_right_pad|>"),
        (200_022, "<|start|>"),
        (200_023, "<|message|>"),
        (200_090, "<|image|>"),
        (200_091, "<|video|>"),
        (200_092, "<|patch|>"),
    ] {
        if tokens[id] != expected {
            return Err(invalid(
                "tokenizer.ggml.tokens",
                format!("token {id} expected {expected:?}, got {:?}", tokens[id]),
            ));
        }
    }
    Ok(())
}

fn required_string_array<'a>(
    gguf: &'a GgufFile,
    key: &'static str,
) -> Result<Vec<&'a str>, MuseGlimmerError> {
    let value = gguf
        .model
        .metadata()
        .get(key)
        .ok_or(MuseGlimmerError::MissingMetadata(key))?;
    let values = value
        .as_array()
        .ok_or_else(|| invalid(key, "is not an array"))?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .ok_or_else(|| invalid(key, format!("entry {index} is not a string")))
        })
        .collect()
}

fn hex_digest(bytes: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn bind_norm<'a>(
    gguf: &'a GgufFile,
    used: &mut HashSet<String>,
    name: String,
    expected: &[u64],
) -> Result<&'a TensorDesc, MuseGlimmerError> {
    let tensor = bind_shape(gguf, used, name, expected)?;
    if tensor.dtype != GgmlType::F32 {
        return Err(MuseGlimmerError::NormDtype {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
        });
    }
    Ok(tensor)
}

fn bind_matrix<'a>(
    gguf: &'a GgufFile,
    used: &mut HashSet<String>,
    name: String,
    expected: &[u64],
) -> Result<&'a TensorDesc, MuseGlimmerError> {
    let tensor = bind_shape(gguf, used, name, expected)?;
    validate_weight_storage(tensor)?;
    Ok(tensor)
}

fn bind_shape<'a>(
    gguf: &'a GgufFile,
    used: &mut HashSet<String>,
    name: String,
    expected: &[u64],
) -> Result<&'a TensorDesc, MuseGlimmerError> {
    let tensor = gguf
        .find(&name)
        .ok_or_else(|| MuseGlimmerError::MissingTensor(name.clone()))?;
    if tensor.shape != expected {
        return Err(MuseGlimmerError::Shape {
            name,
            expected: expected.to_vec(),
            actual: tensor.shape.clone(),
        });
    }
    used.insert(tensor.name.clone());
    Ok(tensor)
}

fn validate_weight_storage(tensor: &TensorDesc) -> Result<(), MuseGlimmerError> {
    if matches!(
        tensor.dtype,
        GgmlType::Unknown
            | GgmlType::I8
            | GgmlType::I16
            | GgmlType::I32
            | GgmlType::I64
            | GgmlType::F64
    ) {
        return Err(MuseGlimmerError::UnknownDtype {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
        });
    }
    let (block_size, _) =
        tensor
            .dtype
            .storage_layout()
            .ok_or_else(|| MuseGlimmerError::UnknownDtype {
                name: tensor.name.clone(),
                dtype: tensor.dtype,
            })?;
    if block_size == 0 || !tensor.shape[0].is_multiple_of(block_size) {
        return Err(MuseGlimmerError::RowAlignment {
            name: tensor.name.clone(),
            row_width: tensor.shape[0],
            block_size,
            dtype: tensor.dtype,
        });
    }
    Ok(())
}

fn required_u32(gguf: &GgufFile, key: &'static str) -> Result<u32, MuseGlimmerError> {
    let value = gguf
        .get_u64(key)
        .ok_or(MuseGlimmerError::MissingMetadata(key))?;
    u32::try_from(value).map_err(|_| invalid(key, format!("value {value} exceeds u32")))
}

fn required_f32(gguf: &GgufFile, key: &'static str) -> Result<f32, MuseGlimmerError> {
    let value = gguf
        .get_f64(key)?
        .ok_or(MuseGlimmerError::MissingMetadata(key))?;
    let value = value as f32;
    if !value.is_finite() {
        return Err(invalid(key, "must be finite"));
    }
    Ok(value)
}

fn required_bool(gguf: &GgufFile, key: &'static str) -> Result<bool, MuseGlimmerError> {
    gguf.get_bool(key)?
        .ok_or(MuseGlimmerError::MissingMetadata(key))
}

fn required_str<'a>(gguf: &'a GgufFile, key: &'static str) -> Result<&'a str, MuseGlimmerError> {
    gguf.get_str(key)
        .ok_or(MuseGlimmerError::MissingMetadata(key))
}

fn require_exact(key: &'static str, actual: u32, expected: u32) -> Result<(), MuseGlimmerError> {
    if actual != expected {
        return Err(invalid(key, format!("expected {expected}, got {actual}")));
    }
    Ok(())
}

fn require_f32(key: &'static str, actual: f32, expected: f32) -> Result<(), MuseGlimmerError> {
    if !actual.is_finite() || actual.to_bits() != expected.to_bits() {
        return Err(invalid(key, format!("expected {expected}, got {actual}")));
    }
    Ok(())
}

fn checked_mul(a: u64, b: u64, label: &'static str) -> Result<u64, MuseGlimmerError> {
    a.checked_mul(b)
        .ok_or(MuseGlimmerError::DimensionOverflow(label))
}

fn invalid(key: &'static str, reason: impl Into<String>) -> MuseGlimmerError {
    MuseGlimmerError::InvalidMetadata {
        key,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_contract_is_exact_and_ordered() {
        let config = MuseGlimmerConfig::release_reference();
        let [eos, eot] = config.expected_stop_tokens();
        assert!(config.validate_stop_tokens(&[eos, eot]).is_ok());
        for invalid in [vec![eos], vec![eot, eos], vec![eos, eot, 3]] {
            assert!(matches!(
                config.validate_stop_tokens(&invalid),
                Err(MuseGlimmerError::StopTokens { .. })
            ));
        }
    }

    #[test]
    fn release_profile_has_exact_geometry_and_schedule() {
        let config = MuseGlimmerConfig::release_reference();
        config.validate_release_profile().unwrap();
        assert_eq!(config.query_width().unwrap(), 4_096);
        assert_eq!(config.kv_width().unwrap(), 256);
        assert_eq!(config.query_head_count / config.kv_head_count, 16);
        assert_eq!(config.sliding_layers.iter().filter(|&&v| v).count(), 39);
        assert_eq!(config.sliding_layers.iter().filter(|&&v| !v).count(), 13);
        assert!(config.is_sliding_layer(2).unwrap());
        assert!(!config.is_sliding_layer(3).unwrap());
        assert!(!config.is_sliding_layer(51).unwrap());
        assert_eq!(
            expected_tensor_count(config.layer_count),
            RELEASE_TENSOR_COUNT
        );

        let unsloth = MuseGlimmerConfig::unsloth_release_reference();
        unsloth.validate_release_profile().unwrap();
        assert_eq!(
            unsloth.chat_template_profile,
            MuseGlimmerChatTemplateProfile::UnslothLaunch
        );
    }

    #[test]
    fn scalar_and_explicit_sliding_patterns_normalize_identically() {
        let expected = release_sliding_pattern();
        assert_eq!(
            normalize_sliding_pattern(RELEASE_LAYER_COUNT, Some(4), None).unwrap(),
            expected
        );
        assert_eq!(
            normalize_sliding_pattern(RELEASE_LAYER_COUNT, None, Some(expected.clone())).unwrap(),
            expected
        );
    }

    #[test]
    fn malformed_sliding_patterns_fail_closed() {
        assert!(normalize_sliding_pattern(RELEASE_LAYER_COUNT, Some(0), None).is_err());
        assert!(
            normalize_sliding_pattern(RELEASE_LAYER_COUNT, None, Some(vec![true; 51])).is_err()
        );
        assert!(normalize_sliding_pattern(RELEASE_LAYER_COUNT, Some(4), Some(vec![])).is_err());
    }

    #[test]
    fn release_profile_rejects_semantic_drift() {
        let mut config = MuseGlimmerConfig::release_reference();
        config.sliding_layers[3] = true;
        assert!(config.validate_release_profile().is_err());

        let mut config = MuseGlimmerConfig::release_reference();
        config.tokenizer_pre = "qwen35".into();
        assert!(config.validate_release_profile().is_err());

        let mut config = MuseGlimmerConfig::release_reference();
        config.eot_token_id = config.vocab_size;
        assert!(config.validate_release_profile().is_err());
    }

    #[test]
    fn quantized_matrix_storage_requires_row_alignment() {
        let tensor = |width| TensorDesc {
            name: "test.weight".into(),
            shape: vec![width, 64],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: 0,
        };
        assert!(validate_weight_storage(&tensor(6_656)).is_ok());
        assert!(validate_weight_storage(&tensor(6_655)).is_err());
    }

    #[test]
    fn only_initial_q8_and_bf16_matrix_profiles_are_admitted() {
        assert_eq!(
            classify_matrix_dtypes(
                &vec![GgmlType::Q8_0; RELEASE_MATRIX_TENSOR_COUNT],
                MuseGlimmerChatTemplateProfile::UnslothLaunch,
            )
            .unwrap(),
            MuseGlimmerArtifactProfile::UnslothQ8_0
        );
        assert_eq!(
            classify_matrix_dtypes(
                &vec![GgmlType::BF16; RELEASE_MATRIX_TENSOR_COUNT],
                MuseGlimmerChatTemplateProfile::UnslothLaunch,
            )
            .unwrap(),
            MuseGlimmerArtifactProfile::UnslothBf16
        );
        let mut mixed = vec![GgmlType::Q8_0; RELEASE_MATRIX_TENSOR_COUNT];
        mixed[17] = GgmlType::Q6_K;
        assert!(
            classify_matrix_dtypes(&mixed, MuseGlimmerChatTemplateProfile::UnslothLaunch).is_err()
        );
        assert!(
            classify_matrix_dtypes(
                &vec![GgmlType::Q8_0; RELEASE_MATRIX_TENSOR_COUNT],
                MuseGlimmerChatTemplateProfile::MetaFixed,
            )
            .is_err()
        );
    }

    #[test]
    fn released_chat_template_hashes_are_pinned() {
        assert_eq!(
            classify_chat_template(META_FIXED_CHAT_SHA256).unwrap(),
            MuseGlimmerChatTemplateProfile::MetaFixed
        );
        assert_eq!(
            classify_chat_template(UNSLOTH_LAUNCH_CHAT_SHA256).unwrap(),
            MuseGlimmerChatTemplateProfile::UnslothLaunch
        );
        assert!(classify_chat_template([0; 32]).is_err());
    }

    #[test]
    #[ignore = "requires the pinned local Unsloth Muse Glimmer Q8_0 GGUF"]
    fn binds_pinned_q8_target() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF").unwrap_or_else(|_| {
            crate::test_fixtures::MUSE_GLIMMER_Q8_0.path().into()
        });
        let gguf = GgufFile::open(path).expect("open Muse Q8 target");
        let model = MuseGlimmerModel::from_gguf(&gguf).expect("bind Muse Q8 target");
        assert_eq!(
            model.artifact_profile,
            MuseGlimmerArtifactProfile::UnslothQ8_0
        );
        assert_eq!(model.layers.len(), RELEASE_LAYER_COUNT as usize);
        assert_eq!(gguf.tensors.len(), 731);
    }

    #[test]
    #[ignore = "requires the pinned local two-shard Unsloth Muse Glimmer BF16 GGUF"]
    fn binds_pinned_bf16_target() {
        let path = std::env::var("MUSE_GLIMMER_BF16_GGUF").unwrap_or_else(|_| {
            "/Users/tito/models/muse-glimmer/BF16/Muse-Glimmer-30B-BF16-00001-of-00002.gguf".into()
        });
        let gguf = GgufFile::open(path).expect("open Muse BF16 target");
        let model = MuseGlimmerModel::from_gguf(&gguf).expect("bind Muse BF16 target");
        assert_eq!(
            model.artifact_profile,
            MuseGlimmerArtifactProfile::UnslothBf16
        );
        assert_eq!(gguf.shards.len(), 2);
        assert_eq!(gguf.tensors.len(), 731);
    }
}
