use super::muse_lens_artifact;
use super::muse_lens_rows_artifact as rows;
use anyhow::{Context, Result, ensure};
use qwen_llm::muse_glimmer::{ARCHITECTURE_NAME, MuseGlimmerConfig};
use serde::{Deserialize, Serialize};

pub(crate) const SCHEMA: &str = "muse_glimmer.full_transport";
pub(crate) const SCHEMA_VERSION: u32 = 3;
pub(crate) const MANIFEST_NAME: &str = "lens.json";
pub(crate) const PAYLOAD_NAME: &str = "transport.f16le";
pub(crate) const PARTIAL_PAYLOAD_NAME: &str = "transport.f16le.partial";
pub(crate) const ASSEMBLY_STATE_SCHEMA: &str = "muse_glimmer.full_transport_assembly";
pub(crate) const ASSEMBLY_STATE_NAME: &str = "assembly.json";
pub(crate) const CONVERSION: &str = "ieee754_binary32_to_binary16_round_ties_to_even";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    pub architecture: String,
    pub artifact_profile: String,
    pub model_content_blake3: String,
    pub content_identity_policy: String,
    pub geometry: muse_lens_artifact::Geometry,
    pub method: String,
    pub rule_contract: String,
    pub target_layer: u32,
    pub source_layers: Vec<u32>,
    pub coordinate: String,
    pub estimator: String,
    pub orientation: String,
    pub reduction: String,
    pub replay_semantics: String,
    pub production_semantics: String,
    pub skip_first: usize,
    pub query_batch_size: usize,
    pub max_tokens: usize,
    pub add_special_tokens: bool,
    pub corpus_blake3: String,
    pub selected_records: usize,
    pub fit_build_source_state: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InputShard {
    pub row_start: u32,
    pub row_end: u32,
    pub config_blake3: String,
    pub payload_blake3: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Assembly {
    pub input_schema: String,
    pub input_schema_version: u32,
    pub input_dtype: String,
    pub conversion: String,
    pub row_coverage: [u32; 2],
    pub shards: Vec<InputShard>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AssemblyState {
    pub schema: String,
    pub schema_version: u32,
    pub config_blake3: String,
    pub assembly_blake3: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MatrixDescriptor {
    pub source_layer: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub blake3: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Payload {
    pub path: String,
    pub dtype: String,
    pub shape: [usize; 3],
    pub byte_length: u64,
    pub blake3: String,
    pub matrices: Vec<MatrixDescriptor>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Provenance {
    pub build_commit: String,
    pub build_dirty: String,
    pub build_source_state: String,
    pub build_stamp_source: String,
    pub build_stamp_error: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct IdentitySummary {
    pub policy: String,
    pub content_blake3: String,
    pub content_authenticated: bool,
    pub input_outcomes: Vec<String>,
    pub weight_bytes_hashed: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub schema: String,
    pub schema_version: u32,
    pub status: String,
    pub config_blake3: String,
    pub config: Config,
    pub identity: IdentitySummary,
    pub corpus: rows::CorpusSummary,
    pub assembly: Assembly,
    pub payload: Payload,
    pub replay: Vec<rows::ReplayDiagnostic>,
    pub provenance: Provenance,
}

pub(crate) fn config_from_row(value: &rows::Config) -> Config {
    Config {
        architecture: value.architecture.clone(),
        artifact_profile: value.artifact_profile.clone(),
        model_content_blake3: value.model_content_blake3.clone(),
        content_identity_policy: value.content_identity_policy.clone(),
        geometry: value.geometry.clone(),
        method: value.method.clone(),
        rule_contract: value.rule_contract.clone(),
        target_layer: value.target_layer,
        source_layers: value.source_layers.clone(),
        coordinate: value.coordinate.clone(),
        estimator: value.estimator.clone(),
        orientation: value.orientation.clone(),
        reduction: value.reduction.clone(),
        replay_semantics: value.replay_semantics.clone(),
        production_semantics: value.production_semantics.clone(),
        skip_first: value.skip_first,
        query_batch_size: value.query_batch_size,
        max_tokens: value.max_tokens,
        add_special_tokens: value.add_special_tokens,
        corpus_blake3: value.corpus_blake3.clone(),
        selected_records: value.selected_records,
        fit_build_source_state: value.build_source_state.clone(),
    }
}

pub(crate) fn validate_config(config: &Config) -> Result<()> {
    let reference = MuseGlimmerConfig::unsloth_release_reference();
    let expected_geometry = muse_lens_artifact::geometry(&reference);
    let expected_sources = (0..config.target_layer).collect::<Vec<_>>();
    ensure!(
        config.architecture == ARCHITECTURE_NAME
            && matches!(
                config.artifact_profile.as_str(),
                "unsloth_q8_0" | "unsloth_bf16"
            )
            && config.geometry == expected_geometry
            && is_digest(&config.model_content_blake3)
            && config.content_identity_policy == rows::CONTENT_IDENTITY_POLICY
            && is_digest(&config.corpus_blake3),
        "Muse full transport has unsupported model binding"
    );
    ensure!(
        config.target_layer == config.geometry.layer_count - 1
            && config.source_layers == expected_sources,
        "Muse full transport requires every source layer below the final target layer"
    );
    ensure!(
        matches!(config.method.as_str(), "J" | "R")
            && config.rule_contract == muse_lens_artifact::RULE_CONTRACT_V2
            && config.coordinate == muse_lens_artifact::COORDINATE
            && config.estimator == rows::ESTIMATOR
            && config.orientation == rows::ORIENTATION
            && config.reduction == rows::REDUCTION
            && config.replay_semantics == muse_lens_artifact::REPLAY_SEMANTICS
            && config.production_semantics == muse_lens_artifact::PRODUCTION_SEMANTICS,
        "Muse full transport has unsupported fitting semantics"
    );
    ensure!(
        config.query_batch_size > 0
            && config.query_batch_size
                <= qwen_llm::muse_glimmer_lens_fit::MUSE_GLIMMER_QUERY_BATCH_MAX
            && config.max_tokens > 0
            && config.max_tokens
                <= qwen_llm::muse_glimmer_lens::MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS
            && config
                .skip_first
                .checked_add(2)
                .is_some_and(|minimum| minimum <= config.max_tokens)
            && config.selected_records > 0
            && config.selected_records <= super::MAX_PROMPT_RECORDS
            && valid_build_source_state(&config.fit_build_source_state),
        "Muse full transport has invalid fit bounds or source identity"
    );
    Ok(())
}

pub(crate) fn validate_manifest(manifest: &Manifest) -> Result<()> {
    ensure!(
        manifest.schema == SCHEMA
            && manifest.schema_version == SCHEMA_VERSION
            && manifest.status == "complete",
        "artifact is not a complete Muse full transport"
    );
    validate_config(&manifest.config)?;
    ensure!(
        manifest.config_blake3 == super::digest_json(&manifest.config)?,
        "Muse full-transport config digest is inconsistent"
    );
    validate_corpus(&manifest.corpus, &manifest.config)?;
    validate_identity(&manifest.identity, &manifest.config)?;
    validate_assembly(&manifest.assembly, &manifest.config)?;
    validate_payload(&manifest.payload, &manifest.config)?;
    validate_replay(&manifest.replay, &manifest.config)?;
    ensure!(
        valid_build_source_state(&manifest.provenance.build_source_state)
            && manifest.provenance.build_stamp_error == "none",
        "Muse full-transport assembly provenance is invalid"
    );
    Ok(())
}

fn validate_identity(identity: &IdentitySummary, config: &Config) -> Result<()> {
    ensure!(
        identity.policy == config.content_identity_policy
            && identity.content_blake3 == config.model_content_blake3
            && identity.content_authenticated
            && identity.weight_bytes_hashed == 0
            && !identity.input_outcomes.is_empty()
            && identity
                .input_outcomes
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            && identity.input_outcomes.iter().all(|outcome| matches!(
                outcome.as_str(),
                "Hit" | "DeclaredAndStored" | "DeclaredUncached"
            )),
        "Muse full-transport identity summary is inconsistent"
    );
    Ok(())
}

fn validate_corpus(corpus: &rows::CorpusSummary, config: &Config) -> Result<()> {
    let skipped = u64::try_from(corpus.skipped_prompts.len())
        .context("Muse full-transport skipped prompt count")?;
    ensure!(
        corpus.selected_records == config.selected_records
            && corpus.used_prompts > 0
            && corpus.used_prompts.checked_add(skipped)
                == u64::try_from(corpus.selected_records).ok()
            && corpus.truncated_prompts <= corpus.used_prompts
            && corpus.ordered_token_ids_blake3 == config.corpus_blake3
            && corpus.add_special_tokens == config.add_special_tokens
            && corpus.max_tokens == config.max_tokens
            && corpus.prompt_reduction == rows::PROMPT_REDUCTION,
        "Muse full-transport corpus summary is inconsistent"
    );
    Ok(())
}

fn validate_assembly(assembly: &Assembly, config: &Config) -> Result<()> {
    ensure!(
        assembly.input_schema == rows::SCHEMA
            && assembly.input_schema_version == rows::SCHEMA_VERSION
            && assembly.input_dtype == "f32_le"
            && assembly.conversion == CONVERSION
            && assembly.row_coverage == [0, config.geometry.hidden_size]
            && !assembly.shards.is_empty(),
        "Muse full-transport assembly contract is invalid"
    );
    let mut cursor = 0u32;
    for shard in &assembly.shards {
        ensure!(
            shard.row_start == cursor
                && shard.row_end > shard.row_start
                && shard.row_end <= config.geometry.hidden_size
                && shard.row_end - shard.row_start
                    <= qwen_llm::muse_glimmer_lens_fit::MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD as u32
                && is_digest(&shard.config_blake3)
                && is_digest(&shard.payload_blake3),
            "Muse full-transport shard coverage or digest is invalid at row {cursor}"
        );
        cursor = shard.row_end;
    }
    ensure!(
        cursor == config.geometry.hidden_size,
        "Muse full-transport shard coverage ends at {cursor}"
    );
    Ok(())
}

fn validate_payload(payload: &Payload, config: &Config) -> Result<()> {
    let hidden = config.geometry.hidden_size as usize;
    let source_count = config.source_layers.len();
    let matrix_bytes = matrix_bytes(config)?;
    let total_bytes = matrix_bytes
        .checked_mul(u64::try_from(source_count).context("Muse full source count")?)
        .context("Muse full payload byte count overflow")?;
    ensure!(
        payload.path == PAYLOAD_NAME
            && payload.dtype == "f16_le"
            && payload.shape == [source_count, hidden, hidden]
            && payload.byte_length == total_bytes
            && is_digest(&payload.blake3)
            && payload.matrices.len() == source_count,
        "Muse full-transport payload descriptor is invalid"
    );
    for (slot, matrix) in payload.matrices.iter().enumerate() {
        ensure!(
            matrix.source_layer == config.source_layers[slot]
                && matrix.byte_offset
                    == u64::try_from(slot)
                        .context("Muse full matrix slot")?
                        .checked_mul(matrix_bytes)
                        .context("Muse full matrix offset overflow")?
                && matrix.byte_length == matrix_bytes
                && is_digest(&matrix.blake3),
            "Muse full-transport matrix descriptor is invalid at slot {slot}"
        );
    }
    Ok(())
}

fn validate_replay(replay: &[rows::ReplayDiagnostic], config: &Config) -> Result<()> {
    ensure!(
        replay.len() == config.target_layer as usize,
        "Muse full-transport replay schedule length is invalid"
    );
    for (offset, diagnostic) in replay.iter().enumerate() {
        let block =
            config.target_layer - u32::try_from(offset).context("Muse full replay offset")?;
        let kind = if config.geometry.sliding_layers[block as usize] {
            "sliding"
        } else {
            "full"
        };
        ensure!(
            diagnostic.block == block
                && diagnostic.kind == kind
                && diagnostic.post_attention_replay_max_abs_error.is_finite()
                && diagnostic.post_attention_replay_max_abs_error >= 0.0
                && diagnostic.post_block_replay_max_abs_error.is_finite()
                && diagnostic.post_block_replay_max_abs_error >= 0.0,
            "Muse full-transport replay diagnostic is invalid at offset {offset}"
        );
    }
    Ok(())
}

pub(crate) fn matrix_bytes(config: &Config) -> Result<u64> {
    let hidden = u64::from(config.geometry.hidden_size);
    hidden
        .checked_mul(hidden)
        .and_then(|values| values.checked_mul(2))
        .context("Muse full matrix byte count overflow")
}

pub(crate) fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_build_source_state(value: &str) -> bool {
    value
        .strip_prefix("git-source-sha256-v2:")
        .is_some_and(is_digest)
}

#[cfg(test)]
mod tests {
    use super::super::FitMethod;
    use super::*;
    use qwen_llm::muse_glimmer::MuseGlimmerArtifactProfile;

    fn row_config() -> rows::Config {
        let model = MuseGlimmerConfig::unsloth_release_reference();
        rows::make_config(
            &model,
            MuseGlimmerArtifactProfile::UnslothQ8_0,
            "11".repeat(32),
            FitMethod::R,
            51,
            (0..51).collect(),
            0,
            32,
            0,
            4,
            2,
            false,
            "22".repeat(32),
            1,
            format!("git-source-sha256-v2:{}", "33".repeat(32)),
        )
    }

    #[test]
    fn full_config_requires_complete_sources_and_final_target() {
        let mut config = config_from_row(&row_config());
        validate_config(&config).unwrap();
        assert_eq!(matrix_bytes(&config).unwrap(), 88_604_672);
        config.source_layers.pop();
        assert!(validate_config(&config).is_err());
        config = config_from_row(&row_config());
        config.target_layer = 50;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn identity_summary_requires_no_hashing_and_canonical_outcomes() {
        let config = config_from_row(&row_config());
        let mut identity = IdentitySummary {
            policy: rows::CONTENT_IDENTITY_POLICY.into(),
            content_blake3: config.model_content_blake3.clone(),
            content_authenticated: true,
            input_outcomes: vec!["DeclaredAndStored".into(), "Hit".into()],
            weight_bytes_hashed: 0,
        };
        validate_identity(&identity, &config).unwrap();
        identity.weight_bytes_hashed = 1;
        assert!(validate_identity(&identity, &config).is_err());
        identity.weight_bytes_hashed = 0;
        identity.input_outcomes.reverse();
        assert!(validate_identity(&identity, &config).is_err());
    }

    #[test]
    fn assembly_coverage_rejects_gaps_and_overlaps() {
        let config = config_from_row(&row_config());
        let mut assembly = Assembly {
            input_schema: rows::SCHEMA.into(),
            input_schema_version: rows::SCHEMA_VERSION,
            input_dtype: "f32_le".into(),
            conversion: CONVERSION.into(),
            row_coverage: [0, 6_656],
            shards: (0..26)
                .map(|slot| InputShard {
                    row_start: slot * 256,
                    row_end: ((slot + 1) * 256).min(6_656),
                    config_blake3: "44".repeat(32),
                    payload_blake3: "55".repeat(32),
                })
                .collect(),
        };
        validate_assembly(&assembly, &config).unwrap();
        assembly.shards[1].row_start += 1;
        assert!(validate_assembly(&assembly, &config).is_err());
    }
}
