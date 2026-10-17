//! DeepSeek V4.1 Flash 1017 metadata/schedule contract, separate from V4.
//!
//! Not a runnable model registration, GGUF admission, tensor binding, or memory
//! admission. Validation covers the complete text_config, architecture/model
//! tags, text special-token IDs and quantization_config. Other top-level metadata
//! and vision_config are deliberately outside this text-only scope. NextN
//! metadata is validated but no NextN execution plan is provided.

use serde_json::{Value, json};
use std::ops::Range;

const KV_OWNERS: [usize; 4] = [2, 8, 14, 20];
const SELECTION_PRODUCERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];
const MAX_CONTEXT: usize = 1_048_576;
const CANDIDATE_LIMIT: usize = 16_384;
const HIDDEN_SIZE: u64 = 5120;
const EXPERT_WIDTH: u64 = 2304;

/// Geometry only: REAP variants do not qualify any artifact or kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertGeometry {
    Base384,
    PotentialReap272,
    PotentialReap256,
}

impl ExpertGeometry {
    pub fn expert_count(self) -> usize {
        match self {
            Self::Base384 => 384,
            Self::PotentialReap272 => 272,
            Self::PotentialReap256 => 256,
        }
    }

    /// Packed expert payload only, including one scale byte per 32 values.
    /// Excludes metadata, alignment, all other weights, and runtime allocations.
    /// Equal byte counts do not make HF and GGML layouts interchangeable.
    pub fn projection_bytes(self, projection: ExpertProjection) -> u64 {
        let (output, input) = projection.logical_shape();
        self.expert_count() as u64 * output * (input / 32) * 17
    }
}

/// Logical source matrices are [output, input], before any GGUF conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertProjection {
    Gate,
    Up,
    Down,
}

impl ExpertProjection {
    pub fn logical_shape(self) -> (u64, u64) {
        match self {
            Self::Gate | Self::Up => (EXPERT_WIDTH, HIDDEN_SIZE),
            Self::Down => (HIDDEN_SIZE, EXPERT_WIDTH),
        }
    }

    /// HF stores packed values and E8M0 scales separately; these are not
    /// checkpoint header descriptors or an admitted converter layout.
    pub fn source_shapes(self) -> ([u64; 2], [u64; 2]) {
        let (output, input) = self.logical_shape();
        ([output, input / 2], [output, input / 32])
    }
}

/// Distinct arithmetic contracts, not interchangeable numerical implementations.
/// Neither is implemented or admitted by this metadata module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivationArithmetic {
    /// Native f32 activations, accumulation and outputs over decoded weights,
    /// without reference FP8 activation quantization or BF16 output rounding.
    NativeF32,
    /// Reference dynamic E4M3 activation quantization in blocks of 32 with
    /// UE8M0 scales, FP32 accumulation and BF16 output; not an f32 parity claim.
    ReferenceFp8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerKind {
    Swa,
    Full,
    Reindex,
    Reuse,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerPlan {
    layer: usize,
    kind: LayerKind,
    ratio: usize,
    kv_index_owner: Option<usize>,
    selection_producer: Option<usize>,
    candidate_owner: Option<usize>,
}

impl LayerPlan {
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn kind(&self) -> LayerKind {
        self.kind
    }
    pub fn compression_ratio(&self) -> usize {
        self.ratio
    }
    /// KV and index-key caches share these four owners. Reindex creates a
    /// selection, not another persistent index-key cache.
    pub fn kv_index_owner(&self) -> Option<usize> {
        self.kv_index_owner
    }
    pub fn selection_producer(&self) -> Option<usize> {
        self.selection_producer
    }
    pub fn candidate_owner(&self) -> Option<usize> {
        self.candidate_owner
    }
}

#[derive(Clone, Debug)]
pub struct TextConfig {
    geometry: ExpertGeometry,
    layers: Vec<LayerPlan>,
}

impl TextConfig {
    /// Strict pinned text-schema validation; unknown text fields are rejected.
    /// The caller must explicitly select any potential REAP geometry.
    pub fn from_hf_json(input: &str, geometry: ExpertGeometry) -> Result<Self, String> {
        let value: Value = serde_json::from_str(input).map_err(|e| e.to_string())?;
        for (key, expected) in [
            ("architectures", json!(["DeepseekV41ForCausalLM"])),
            ("model_type", json!("deepseek_v41")),
            ("dtype", json!("bfloat16")),
            ("bos_token_id", json!(0)),
            ("eos_token_id", json!(1)),
            ("pad_token_id", json!(2)),
            (
                "quantization_config",
                json!({
                    "quant_method":"fp8", "activation_scheme":"dynamic",
                    "weight_block_size":[32,32], "scale_fmt":"ue8m0", "expert_dtype":"fp4"
                }),
            ),
            ("text_config", expected_text(geometry)),
        ] {
            validate_value(key, value.get(key), &expected)?;
        }
        let layers = (0..40)
            .map(|layer| {
                let owner = KV_OWNERS.iter().copied().filter(|&id| id <= layer).last();
                let producer = SELECTION_PRODUCERS
                    .iter()
                    .copied()
                    .filter(|&id| id <= layer)
                    .last();
                let kind = if layer < 2 {
                    LayerKind::Swa
                } else if KV_OWNERS.contains(&layer) {
                    LayerKind::Full
                } else if SELECTION_PRODUCERS.contains(&layer) {
                    LayerKind::Reindex
                } else {
                    LayerKind::Reuse
                };
                LayerPlan {
                    layer,
                    kind,
                    ratio: if layer < 2 {
                        0
                    } else if layer < 20 {
                        2
                    } else {
                        1
                    },
                    kv_index_owner: owner,
                    selection_producer: producer,
                    candidate_owner: (layer >= 20).then_some(20),
                }
            })
            .collect();
        Ok(Self { geometry, layers })
    }

    pub fn geometry(&self) -> ExpertGeometry {
        self.geometry
    }
    pub fn layers(&self) -> &[LayerPlan] {
        &self.layers
    }

    /// Plan a contiguous append in absolute, zero-based positions [start, end).
    /// This is eligibility only, independent of completion. A pair row becomes
    /// eligible at its odd position; a caller must separately ensure its even
    /// predecessor and owner writes exist before publishing anything.
    pub fn publication(&self, positions: Range<usize>) -> Result<PublicationPlan, String> {
        if positions.start > positions.end || positions.end > MAX_CONTEXT {
            return Err("invalid context position range".into());
        }
        let owners = KV_OWNERS
            .iter()
            .map(|&owner| {
                let rows = if owner < 20 {
                    positions.start / 2..positions.end / 2
                } else {
                    positions.clone()
                };
                OwnerRows { owner, rows }
            })
            .collect();
        Ok(PublicationPlan {
            swa_context: positions.end.saturating_sub(128)..positions.end,
            candidate_capacity: positions.end.min(CANDIDATE_LIMIT),
            owners,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnerRows {
    owner: usize,
    rows: Range<usize>,
}
impl OwnerRows {
    pub fn owner(&self) -> usize {
        self.owner
    }
    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicationPlan {
    swa_context: Range<usize>,
    candidate_capacity: usize,
    owners: Vec<OwnerRows>,
}
impl PublicationPlan {
    /// Retained SWA context after this append, not a per-query attention mask.
    pub fn swa_context(&self) -> Range<usize> {
        self.swa_context.clone()
    }
    /// Upper bound only; not actual selected candidates or memory admission.
    pub fn candidate_capacity(&self) -> usize {
        self.candidate_capacity
    }
    pub fn owners(&self) -> &[OwnerRows] {
        &self.owners
    }
}

fn validate_value(path: &str, actual: Option<&Value>, expected: &Value) -> Result<(), String> {
    match (actual, expected) {
        (Some(Value::Object(actual)), Value::Object(expected))
            if actual.len() == expected.len() =>
        {
            for (key, value) in expected {
                validate_value(&format!("{path}.{key}"), actual.get(key), value)?;
            }
            Ok(())
        }
        (Some(Value::Array(actual)), Value::Array(expected)) if actual.len() == expected.len() => {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                validate_value(&format!("{path}[{index}]"), Some(actual), expected)?;
            }
            Ok(())
        }
        // Lossless JSON preserves exponent spelling (1e-06 versus 1e-6).
        // Compare real-valued parameters numerically, but keep integer fields
        // strictly integer so malformed dimensions cannot be rounded into shape.
        (Some(Value::Number(actual)), Value::Number(expected)) if expected.is_f64() => {
            if actual.as_f64().is_some_and(|n| n.is_finite())
                && actual.as_f64() == expected.as_f64()
            {
                Ok(())
            } else {
                Err(format!("{path} differs from pinned V4.1 text contract"))
            }
        }
        (Some(actual), expected) if actual == expected => Ok(()),
        _ => Err(format!("{path} differs from pinned V4.1 text contract")),
    }
}

fn expected_text(geometry: ExpertGeometry) -> Value {
    let mut text = json!({
        "model_type":"deepseek_v41_text", "vocab_size":129280,
        "hidden_size":HIDDEN_SIZE, "moe_intermediate_size":EXPERT_WIDTH, "num_hidden_layers":40,
        "num_attention_heads":64, "num_key_value_heads":1, "head_dim":512,
        "qk_rope_head_dim":64, "q_lora_rank":1280, "o_lora_rank":1024, "o_groups":8,
        "hidden_act":"silu", "swiglu_limit":10.0, "rms_norm_eps":1e-20,
        "attention_bias":false, "attention_dropout":0.0, "initializer_range":0.02,
        "use_cache":true, "tie_word_embeddings":false, "max_position_embeddings":MAX_CONTEXT
    });
    let attention = json!({
        "rope_theta":10000, "rope_scaling":{
            "rope_type":"yarn", "factor":16, "beta_fast":32, "beta_slow":1,
            "original_max_position_embeddings":65536
        },
        "n_routed_experts":geometry.expert_count(), "n_shared_experts":1,
        "num_experts_per_tok":6, "scoring_func":"sqrtsoftplus", "topk_method":"noaux_tc",
        "norm_topk_prob":true, "routed_scaling_factor":1.5, "sliding_window":128,
        "compress_ratios":(0..43).map(|i| if i < 2 || i >= 40 { 0 } else if i < 20 { 2 } else { 1 }).collect::<Vec<_>>(),
        "compress_rope_theta":160000, "kv_source_layer_ids":KV_OWNERS,
        "index_source_layer_ids":SELECTION_PRODUCERS, "index_n_heads":32,
        "index_head_dim":128, "index_topk":512, "candidate_source_layer_id":20,
        "candidate_topk_blocks":2048, "candidate_block_size":8
    });
    let auxiliary = json!({
        "hc_mult":4, "hc_sinkhorn_iters":20, "hc_eps":1e-6,
        "engram_layer_ids":[1,14], "engram_num_embeddings":[384006168,384016682],
        "engram_max_ngram_size":4, "engram_vocab_size":16000000, "engram_n_heads":8,
        "engram_head_dim":256, "engram_pad_token_id":2, "engram_compressed_vocab_size":99092,
        "num_nextn_predict_layers":3, "dspark_block_size":5, "dspark_noise_token_id":128799,
        "dspark_target_layer_ids":[37,38,39], "dspark_markov_rank":256,
        "dspark_n_routed_experts":128, "dspark_num_experts_per_tok":3
    });
    for section in [attention, auxiliary] {
        text.as_object_mut()
            .unwrap()
            .extend(section.as_object().unwrap().clone());
    }
    text
}

#[cfg(test)]
mod tests;
