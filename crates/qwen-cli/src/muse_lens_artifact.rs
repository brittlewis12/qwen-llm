use anyhow::{Context, Result, ensure};
use qwen_llm::muse_glimmer::{MuseGlimmerArtifactProfile, MuseGlimmerConfig};
use qwen_llm::muse_glimmer_lens::MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS;
use qwen_llm::tokenizer::Tokenize;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub(crate) const SCHEMA: &str = "muse_glimmer.selected_token_transport";
pub(crate) const SCHEMA_VERSION: u32 = 1;
pub(crate) const RULE_CONTRACT: &str = "muse_glimmer_one_full_attention_block_v1";
pub(crate) const COORDINATE: &str = "post_block_residual_hugging_face_block_output";
pub(crate) const ESTIMATOR: &str =
    "selected_score_covector_vjp_through_one_adjacent_full_attention_block";
pub(crate) const REDUCTION: &str =
    "place_covector_at_each_position_skip_first..T-1; mean_matching_source_rows; mean_used_prompts";
pub(crate) const REPLAY_SEMANTICS: &str = "smooth_f32_model_level_block_replay_and_vjp";
pub(crate) const PRODUCTION_SEMANTICS: &str = "scalar_forward_with_f16_attention_kv_cache";
pub(crate) const SCORE: &str =
    "pre_final_softcap_selected_token_linear_numerator_without_output_rms_denominator";
pub(crate) const PAYLOAD_NAME: &str = "readouts.f32le";
pub(crate) const MANIFEST_NAME: &str = "readouts.json";
pub(crate) const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;

pub(crate) fn is_muse_architecture(architecture: Option<&str>) -> bool {
    architecture == Some(qwen_llm::muse_glimmer::ARCHITECTURE_NAME)
}

pub(crate) fn validate_tokenizer(
    tokenizer: &impl Tokenize,
    config: &MuseGlimmerConfig,
) -> Result<()> {
    ensure!(
        tokenizer.n_vocab() == config.vocab_size,
        "Muse Glimmer tokenizer vocabulary {} differs from model vocabulary {}",
        tokenizer.n_vocab(),
        config.vocab_size
    );
    ensure!(
        tokenizer.bos() == Some(config.bos_token_id as i32),
        "Muse Glimmer tokenizer BOS {:?} differs from model BOS {}",
        tokenizer.bos(),
        config.bos_token_id
    );
    ensure!(
        tokenizer.eos() == Some(config.eos_token_id as i32),
        "Muse Glimmer tokenizer EOS {:?} differs from model EOS {}",
        tokenizer.eos(),
        config.eos_token_id
    );
    Ok(())
}

pub(crate) fn serialize_manifest(manifest: &Manifest) -> Result<Vec<u8>> {
    let bytes = serde_json::to_vec_pretty(manifest).context("serialize Muse readout manifest")?;
    ensure!(
        bytes.len() <= MAX_MANIFEST_BYTES,
        "Muse readout manifest is {} bytes; limit is {}",
        bytes.len(),
        MAX_MANIFEST_BYTES
    );
    Ok(bytes)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub schema: String,
    pub schema_version: u32,
    pub architecture: String,
    pub artifact_profile: String,
    pub model_content_blake3: String,
    pub geometry: Geometry,
    pub transport: Transport,
    pub selected: Selected,
    pub payload: Payload,
    pub corpus: Corpus,
    pub replay: Replay,
    pub provenance: Provenance,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Geometry {
    pub layer_count: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub sliding_window: u32,
    pub sliding_layers: Vec<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Transport {
    pub method: String,
    pub rule_contract: String,
    pub target_layer: u32,
    pub source_layers: Vec<u32>,
    pub coordinate: String,
    pub estimator: String,
    pub reduction: String,
    pub replay_semantics: String,
    pub production_semantics: String,
    pub skip_first: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Selected {
    pub token_ids: Vec<u32>,
    pub score: String,
    pub covector_formula: String,
    pub logit_scale: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Payload {
    pub path: String,
    pub dtype: String,
    pub shape: [usize; 3],
    pub byte_length: u64,
    pub blake3: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Corpus {
    pub selected_records: usize,
    pub used_prompts: u64,
    pub skipped_prompts: Vec<super::SkippedPrompt>,
    pub truncated_prompts: u64,
    pub ordered_token_ids_blake3: String,
    pub add_special_tokens: bool,
    pub max_tokens: usize,
    pub prompt_reduction: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Replay {
    pub f32_vs_production_f16_kv_post_attention_max_abs: f32,
    pub f32_vs_production_f16_kv_post_block_max_abs: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Provenance {
    pub build_commit: String,
    pub build_dirty: String,
    pub build_source_state: String,
}

pub(crate) fn profile_name(profile: MuseGlimmerArtifactProfile) -> &'static str {
    match profile {
        MuseGlimmerArtifactProfile::UnslothQ8_0 => "unsloth_q8_0",
        MuseGlimmerArtifactProfile::UnslothBf16 => "unsloth_bf16",
    }
}

pub(crate) fn geometry(config: &MuseGlimmerConfig) -> Geometry {
    Geometry {
        layer_count: config.layer_count,
        hidden_size: config.hidden_size,
        vocab_size: config.vocab_size,
        query_heads: config.query_head_count,
        kv_heads: config.kv_head_count,
        head_dim: config.key_head_dim,
        sliding_window: config.sliding_window,
        sliding_layers: config.sliding_layers.clone(),
    }
}

pub(crate) fn validate(
    manifest: &Manifest,
    config: &MuseGlimmerConfig,
    profile: MuseGlimmerArtifactProfile,
    content_id: &str,
) -> Result<()> {
    ensure!(
        manifest.schema == SCHEMA,
        "artifact is not a Muse selected-token artifact"
    );
    ensure!(
        manifest.schema_version == SCHEMA_VERSION,
        "unsupported Muse artifact schema version"
    );
    ensure!(
        manifest.architecture == qwen_llm::muse_glimmer::ARCHITECTURE_NAME,
        "wrong Muse artifact architecture"
    );
    ensure!(
        manifest.artifact_profile == profile_name(profile),
        "Muse artifact profile differs from running GGUF"
    );
    ensure!(
        manifest.model_content_blake3 == content_id,
        "Muse artifact GGUF content identity differs from running model"
    );
    let expected = geometry(config);
    ensure!(
        manifest.geometry.layer_count == expected.layer_count
            && manifest.geometry.hidden_size == expected.hidden_size
            && manifest.geometry.vocab_size == expected.vocab_size
            && manifest.geometry.query_heads == expected.query_heads
            && manifest.geometry.kv_heads == expected.kv_heads
            && manifest.geometry.head_dim == expected.head_dim
            && manifest.geometry.sliding_window == expected.sliding_window
            && manifest.geometry.sliding_layers == expected.sliding_layers,
        "Muse artifact geometry differs from running model"
    );
    ensure!(
        manifest.transport.rule_contract == RULE_CONTRACT,
        "unsupported Muse lens rule contract"
    );
    ensure!(
        matches!(manifest.transport.method.as_str(), "J" | "R"),
        "unsupported Muse transport method"
    );
    ensure!(
        manifest.transport.coordinate == COORDINATE,
        "unsupported Muse readout coordinate"
    );
    ensure!(
        manifest.transport.estimator == ESTIMATOR
            && manifest.transport.reduction == REDUCTION
            && manifest.transport.replay_semantics == REPLAY_SEMANTICS
            && manifest.transport.production_semantics == PRODUCTION_SEMANTICS,
        "unsupported Muse estimator or replay semantics"
    );
    ensure!(
        manifest.transport.target_layer > 0 && manifest.transport.target_layer < config.layer_count,
        "Muse target layer is out of range"
    );
    ensure!(
        !config.sliding_layers[manifest.transport.target_layer as usize],
        "Muse target must be a full-attention layer"
    );
    ensure!(
        manifest.transport.source_layers == [manifest.transport.target_layer - 1],
        "Muse artifact must contain exactly the adjacent source layer"
    );
    ensure!(
        !manifest.selected.token_ids.is_empty()
            && manifest.selected.token_ids.len() <= MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS,
        "Muse artifact selected-token count must be in 1..={MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS}"
    );
    let mut ids = HashSet::new();
    ensure!(
        manifest
            .selected
            .token_ids
            .iter()
            .all(|&id| id < config.vocab_size && ids.insert(id)),
        "Muse artifact has duplicate or invalid selected token IDs"
    );
    ensure!(
        manifest.selected.logit_scale.to_bits() == config.logit_scale.to_bits(),
        "Muse artifact logit scale differs from running model"
    );
    ensure!(
        manifest.selected.score == SCORE,
        "unsupported Muse selected-token score"
    );
    ensure!(
        manifest.selected.covector_formula
            == "logit_scale * output_norm_gamma * output_weight[token_id]",
        "unsupported Muse selected-token covector formula"
    );
    ensure!(
        manifest.payload.path == PAYLOAD_NAME && manifest.payload.dtype == "f32_le",
        "unsupported Muse payload encoding"
    );
    let shape = [
        1,
        manifest.selected.token_ids.len(),
        config.hidden_size as usize,
    ];
    ensure!(
        manifest.payload.shape == shape,
        "malformed Muse payload shape"
    );
    let expected_bytes = shape
        .into_iter()
        .try_fold(4usize, |n, d| n.checked_mul(d))
        .context("Muse payload size overflow")?;
    ensure!(
        manifest.payload.byte_length == expected_bytes as u64,
        "malformed Muse payload byte length"
    );
    ensure!(
        manifest
            .replay
            .f32_vs_production_f16_kv_post_attention_max_abs
            .is_finite()
            && manifest
                .replay
                .f32_vs_production_f16_kv_post_attention_max_abs
                >= 0.0,
        "invalid Muse attention replay drift"
    );
    ensure!(
        manifest
            .replay
            .f32_vs_production_f16_kv_post_block_max_abs
            .is_finite()
            && manifest.replay.f32_vs_production_f16_kv_post_block_max_abs >= 0.0,
        "invalid Muse block replay drift"
    );
    Ok(())
}

pub(crate) fn decode_payload(bytes: &[u8], manifest: &Manifest) -> Result<Vec<f32>> {
    ensure!(
        blake3::hash(bytes).to_hex().as_str() == manifest.payload.blake3,
        "Muse payload digest mismatch"
    );
    ensure!(
        bytes.len() as u64 == manifest.payload.byte_length,
        "Muse payload length mismatch"
    );
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    ensure!(
        bytes.len().is_multiple_of(4) && values.iter().all(|value| value.is_finite()),
        "Muse payload contains malformed or non-finite F32 values"
    );
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest(config: &MuseGlimmerConfig) -> Manifest {
        let byte_length = config.hidden_size as u64 * 4;
        Manifest {
            schema: SCHEMA.into(),
            schema_version: SCHEMA_VERSION,
            architecture: "muse-glimmer".into(),
            artifact_profile: "unsloth_q8_0".into(),
            model_content_blake3: "00".repeat(32),
            geometry: geometry(config),
            transport: Transport {
                method: "J".into(),
                rule_contract: RULE_CONTRACT.into(),
                target_layer: 51,
                source_layers: vec![50],
                coordinate: COORDINATE.into(),
                estimator: ESTIMATOR.into(),
                reduction: REDUCTION.into(),
                replay_semantics: REPLAY_SEMANTICS.into(),
                production_semantics: PRODUCTION_SEMANTICS.into(),
                skip_first: 0,
            },
            selected: Selected {
                token_ids: vec![1],
                score: SCORE.into(),
                covector_formula: "logit_scale * output_norm_gamma * output_weight[token_id]"
                    .into(),
                logit_scale: config.logit_scale,
            },
            payload: Payload {
                path: PAYLOAD_NAME.into(),
                dtype: "f32_le".into(),
                shape: [1, 1, config.hidden_size as usize],
                byte_length,
                blake3: "00".repeat(32),
            },
            corpus: Corpus {
                selected_records: 1,
                used_prompts: 1,
                skipped_prompts: vec![],
                truncated_prompts: 0,
                ordered_token_ids_blake3: String::new(),
                add_special_tokens: false,
                max_tokens: 2,
                prompt_reduction: "arithmetic_mean_over_used_prompts".into(),
            },
            replay: Replay {
                f32_vs_production_f16_kv_post_attention_max_abs: 0.0,
                f32_vs_production_f16_kv_post_block_max_abs: 0.0,
            },
            provenance: Provenance {
                build_commit: String::new(),
                build_dirty: String::new(),
                build_source_state: String::new(),
            },
        }
    }

    #[test]
    fn dispatch_is_exactly_muse_glimmer() {
        assert!(is_muse_architecture(Some("muse-glimmer")));
        assert!(!is_muse_architecture(Some("qwen3")));
        assert!(!is_muse_architecture(None));
    }

    #[test]
    fn manifest_validation_rejects_ordinary_schema_and_coordinate_drift() {
        let config = MuseGlimmerConfig::unsloth_release_reference();
        let mut manifest = valid_manifest(&config);
        validate(
            &manifest,
            &config,
            MuseGlimmerArtifactProfile::UnslothQ8_0,
            &"00".repeat(32),
        )
        .unwrap();
        manifest.schema = "qwen.workspace_lens_token_readouts".into();
        assert!(
            validate(
                &manifest,
                &config,
                MuseGlimmerArtifactProfile::UnslothQ8_0,
                &"00".repeat(32)
            )
            .is_err()
        );
        manifest = valid_manifest(&config);
        manifest.transport.coordinate = "post_attention_residual".into();
        assert!(
            validate(
                &manifest,
                &config,
                MuseGlimmerArtifactProfile::UnslothQ8_0,
                &"00".repeat(32)
            )
            .is_err()
        );
        manifest = valid_manifest(&config);
        manifest.selected.token_ids =
            vec![1; MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS.saturating_add(1)];
        assert!(
            validate(
                &manifest,
                &config,
                MuseGlimmerArtifactProfile::UnslothQ8_0,
                &"00".repeat(32)
            )
            .is_err()
        );
    }

    #[test]
    fn payload_indexing_is_source_token_hidden() {
        let hidden = 3;
        let tokens = 2;
        let values = [0., 1., 2., 3., 4., 5.];
        let offset = (0 * tokens + 1) * hidden;
        assert_eq!(&values[offset..offset + hidden], &[3., 4., 5.]);
    }

    #[test]
    fn payload_decoder_rejects_non_finite() {
        let manifest = Manifest {
            schema: SCHEMA.into(),
            schema_version: 1,
            architecture: "muse-glimmer".into(),
            artifact_profile: "unsloth_q8_0".into(),
            model_content_blake3: "00".repeat(32),
            geometry: geometry(&MuseGlimmerConfig::unsloth_release_reference()),
            transport: Transport {
                method: "J".into(),
                rule_contract: RULE_CONTRACT.into(),
                target_layer: 51,
                source_layers: vec![50],
                coordinate: COORDINATE.into(),
                estimator: String::new(),
                reduction: String::new(),
                replay_semantics: String::new(),
                production_semantics: String::new(),
                skip_first: 0,
            },
            selected: Selected {
                token_ids: vec![1],
                score: String::new(),
                covector_formula: "logit_scale * output_norm_gamma * output_weight[token_id]"
                    .into(),
                logit_scale: MuseGlimmerConfig::unsloth_release_reference().logit_scale,
            },
            payload: Payload {
                path: PAYLOAD_NAME.into(),
                dtype: "f32_le".into(),
                shape: [1, 1, 1],
                byte_length: 4,
                blake3: blake3::hash(&f32::NAN.to_le_bytes()).to_hex().to_string(),
            },
            corpus: Corpus {
                selected_records: 1,
                used_prompts: 1,
                skipped_prompts: vec![],
                truncated_prompts: 0,
                ordered_token_ids_blake3: String::new(),
                add_special_tokens: false,
                max_tokens: 2,
                prompt_reduction: String::new(),
            },
            replay: Replay {
                f32_vs_production_f16_kv_post_attention_max_abs: 0.0,
                f32_vs_production_f16_kv_post_block_max_abs: 0.0,
            },
            provenance: Provenance {
                build_commit: String::new(),
                build_dirty: String::new(),
                build_source_state: String::new(),
            },
        };
        assert!(decode_payload(&f32::NAN.to_le_bytes(), &manifest).is_err());
    }
}
