use super::muse_lens_artifact;
use super::{FitMethod, PreparedPrompt, SkippedPrompt};
use anyhow::{Context, Result, ensure};
use qwen_llm::muse_glimmer::{ARCHITECTURE_NAME, MuseGlimmerArtifactProfile, MuseGlimmerConfig};
use qwen_llm::muse_glimmer_lens::{MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS, MuseGlimmerLensRule};
use qwen_llm::muse_glimmer_lens_fit::{
    MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD, MUSE_GLIMMER_QUERY_BATCH_MAX,
};
use serde::{Deserialize, Serialize};

pub(crate) const SCHEMA: &str = "muse_glimmer.full_transport_row_shard";
pub(crate) const CHECKPOINT_SCHEMA: &str = "muse_glimmer.full_transport_row_checkpoint";
pub(crate) const SCHEMA_VERSION: u32 = 1;
pub(crate) const PAYLOAD_NAME: &str = "rows.f32le";
pub(crate) const MANIFEST_NAME: &str = "shard.json";
pub(crate) const CHECKPOINT_NAME: &str = "checkpoint.json";
pub(crate) const ORIENTATION: &str = "source_layer_target_output_coordinate_source_coordinate";
pub(crate) const ESTIMATOR: &str =
    "hidden_basis_covector_composed_vjp_to_arbitrary_post_block_sources_v1";
pub(crate) const REDUCTION: &str = "place_basis_covector_at_each_position_skip_first..T-1; mean_matching_source_rows; mean_used_prompts";
pub(crate) const PROMPT_REDUCTION: &str = "arithmetic_mean_over_used_prompts";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    pub architecture: String,
    pub artifact_profile: String,
    pub model_content_blake3: String,
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
    pub row_start: u32,
    pub row_end: u32,
    pub skip_first: usize,
    pub query_batch_size: usize,
    pub max_tokens: usize,
    pub add_special_tokens: bool,
    pub corpus_blake3: String,
    pub selected_records: usize,
    pub build_source_state: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Payload {
    pub path: String,
    pub dtype: String,
    pub shape: [usize; 3],
    pub byte_length: u64,
    pub blake3: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct VjpTimings {
    pub replay_seconds: f64,
    pub feed_forward_reverse_seconds: f64,
    pub attention_output_reverse_seconds: f64,
    pub attention_cpu_reverse_seconds: f64,
    pub attention_input_reverse_seconds: f64,
    pub total_seconds: f64,
}

impl VjpTimings {
    pub(crate) fn add_assign(&mut self, other: &Self) {
        self.replay_seconds += other.replay_seconds;
        self.feed_forward_reverse_seconds += other.feed_forward_reverse_seconds;
        self.attention_output_reverse_seconds += other.attention_output_reverse_seconds;
        self.attention_cpu_reverse_seconds += other.attention_cpu_reverse_seconds;
        self.attention_input_reverse_seconds += other.attention_input_reverse_seconds;
        self.total_seconds += other.total_seconds;
    }

    pub(crate) fn is_valid(&self) -> bool {
        [
            self.replay_seconds,
            self.feed_forward_reverse_seconds,
            self.attention_output_reverse_seconds,
            self.attention_cpu_reverse_seconds,
            self.attention_input_reverse_seconds,
            self.total_seconds,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
    }

    pub(crate) fn is_zero(&self) -> bool {
        self == &Self::default()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReplayDiagnostic {
    pub block: u32,
    pub kind: String,
    pub post_attention_replay_max_abs_error: f32,
    pub post_block_replay_max_abs_error: f32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Checkpoint {
    pub schema: String,
    pub schema_version: u32,
    pub config_blake3: String,
    pub generation: u64,
    pub next_record: usize,
    pub used_prompts: u64,
    pub truncated_prompts: u64,
    pub skipped_prompts: Vec<SkippedPrompt>,
    pub forward_seconds: f64,
    pub vjp_wall_seconds: f64,
    pub vjp_timings: VjpTimings,
    pub sums: Payload,
    pub diagnostics: Vec<ReplayDiagnostic>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelSummary {
    pub path: String,
    pub architecture: String,
    pub artifact_profile: String,
    pub content_blake3: String,
    pub content_identity_outcome: String,
    pub content_bytes_hashed: u64,
    pub content_authenticated: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CorpusSummary {
    pub selected_records: usize,
    pub used_prompts: u64,
    pub skipped_prompts: Vec<SkippedPrompt>,
    pub truncated_prompts: u64,
    pub ordered_token_ids_blake3: String,
    pub add_special_tokens: bool,
    pub max_tokens: usize,
    pub prompt_reduction: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FitSummary {
    pub method: String,
    pub rule_contract: String,
    pub target_layer: u32,
    pub source_layers: Vec<u32>,
    pub coordinate: String,
    pub estimator: String,
    pub orientation: String,
    pub reduction: String,
    pub row_start: u32,
    pub row_end: u32,
    pub skip_first: usize,
    pub query_batch_size: usize,
    pub valid_position_denominator: String,
    pub accumulator_dtype: String,
    pub storage_dtype: String,
    pub forward_seconds: f64,
    pub vjp_wall_seconds: f64,
    pub vjp_timings: VjpTimings,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub schema: String,
    pub schema_version: u32,
    pub status: String,
    pub config_blake3: String,
    pub config: Config,
    pub model: ModelSummary,
    pub corpus: CorpusSummary,
    pub fit: FitSummary,
    pub payload: Payload,
    pub diagnostics: Vec<ReplayDiagnostic>,
    pub provenance: Provenance,
}

pub(crate) fn make_config(
    model: &MuseGlimmerConfig,
    profile: MuseGlimmerArtifactProfile,
    model_content_blake3: String,
    method: FitMethod,
    target_layer: u32,
    source_layers: Vec<u32>,
    row_start: u32,
    row_end: u32,
    skip_first: usize,
    query_batch_size: usize,
    max_tokens: usize,
    add_special_tokens: bool,
    corpus_blake3: String,
    selected_records: usize,
    build_source_state: String,
) -> Config {
    Config {
        architecture: ARCHITECTURE_NAME.into(),
        artifact_profile: muse_lens_artifact::profile_name(profile).into(),
        model_content_blake3,
        geometry: muse_lens_artifact::geometry(model),
        method: rule(method).as_str().into(),
        rule_contract: muse_lens_artifact::RULE_CONTRACT_V2.into(),
        target_layer,
        source_layers,
        coordinate: muse_lens_artifact::COORDINATE.into(),
        estimator: ESTIMATOR.into(),
        orientation: ORIENTATION.into(),
        reduction: REDUCTION.into(),
        replay_semantics: muse_lens_artifact::REPLAY_SEMANTICS.into(),
        production_semantics: muse_lens_artifact::PRODUCTION_SEMANTICS.into(),
        row_start,
        row_end,
        skip_first,
        query_batch_size,
        max_tokens,
        add_special_tokens,
        corpus_blake3,
        selected_records,
        build_source_state,
    }
}

pub(crate) fn rule(method: FitMethod) -> MuseGlimmerLensRule {
    match method {
        FitMethod::J => MuseGlimmerLensRule::J,
        FitMethod::R => MuseGlimmerLensRule::R,
    }
}

pub(crate) fn validate_config(
    config: &Config,
    model: &MuseGlimmerConfig,
    profile: MuseGlimmerArtifactProfile,
    content_id: &str,
) -> Result<()> {
    validate_stored_config(config)?;
    ensure!(
        config.architecture == ARCHITECTURE_NAME
            && config.artifact_profile == muse_lens_artifact::profile_name(profile)
            && config.model_content_blake3 == content_id,
        "Muse row-shard model binding is inconsistent"
    );
    ensure!(
        config.geometry == muse_lens_artifact::geometry(model),
        "Muse row-shard geometry differs from the running model"
    );
    Ok(())
}

pub(crate) fn validate_stored_config(config: &Config) -> Result<()> {
    let profile = profile_from_name(&config.artifact_profile)
        .context("Muse row-shard artifact profile is unsupported")?;
    let reference = MuseGlimmerConfig::unsloth_release_reference();
    ensure!(
        config.architecture == ARCHITECTURE_NAME
            && config.artifact_profile == muse_lens_artifact::profile_name(profile)
            && config.geometry == muse_lens_artifact::geometry(&reference),
        "Muse row-shard stored model geometry or profile is unsupported"
    );
    ensure!(
        is_blake3(&config.model_content_blake3) && is_blake3(&config.corpus_blake3),
        "Muse row-shard content or corpus digest is malformed"
    );
    ensure!(
        config.rule_contract == muse_lens_artifact::RULE_CONTRACT_V2
            && matches!(config.method.as_str(), "J" | "R")
            && config.coordinate == muse_lens_artifact::COORDINATE
            && config.estimator == ESTIMATOR
            && config.orientation == ORIENTATION
            && config.reduction == REDUCTION
            && config.replay_semantics == muse_lens_artifact::REPLAY_SEMANTICS
            && config.production_semantics == muse_lens_artifact::PRODUCTION_SEMANTICS,
        "Muse row-shard transport contract is unsupported"
    );
    ensure!(
        config.target_layer > 0 && config.target_layer < config.geometry.layer_count,
        "Muse row-shard target layer is out of range"
    );
    ensure!(
        !config.source_layers.is_empty()
            && config
                .source_layers
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            && config
                .source_layers
                .iter()
                .all(|&source| source < config.target_layer),
        "Muse row-shard sources must be strictly increasing and below target"
    );
    let row_count = config
        .row_end
        .checked_sub(config.row_start)
        .context("Muse row-shard row range underflow")? as usize;
    ensure!(
        row_count > 0
            && row_count <= MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD
            && config.row_end <= config.geometry.hidden_size,
        "Muse row-shard rows are empty, too large, or outside hidden size"
    );
    ensure!(
        config.query_batch_size > 0 && config.query_batch_size <= MUSE_GLIMMER_QUERY_BATCH_MAX,
        "Muse row-shard query batch is outside the supported range"
    );
    ensure!(
        config.max_tokens > 0
            && config.max_tokens <= MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS
            && config
                .skip_first
                .checked_add(2)
                .is_some_and(|minimum| minimum <= config.max_tokens),
        "Muse row-shard prompt bounds are invalid"
    );
    ensure!(
        config.selected_records > 0 && config.selected_records <= super::MAX_PROMPT_RECORDS,
        "Muse row-shard selected record count is invalid"
    );
    ensure!(
        valid_build_source_state(&config.build_source_state),
        "Muse row-shard build source state is malformed"
    );
    Ok(())
}

pub(crate) fn validate_stored(manifest: &Manifest) -> Result<()> {
    ensure!(
        manifest.schema == SCHEMA
            && manifest.schema_version == SCHEMA_VERSION
            && manifest.status == "complete",
        "artifact is not a complete Muse full-transport row shard"
    );
    validate_stored_config(&manifest.config)?;
    ensure!(
        manifest.config_blake3 == super::digest_json(&manifest.config)?,
        "stored Muse row-shard config digest is inconsistent"
    );
    ensure!(
        manifest.model.architecture == manifest.config.architecture
            && manifest.model.artifact_profile == manifest.config.artifact_profile
            && manifest.model.content_blake3 == manifest.config.model_content_blake3
            && manifest.model.content_authenticated
            && !manifest.model.path.is_empty()
            && !manifest.model.content_identity_outcome.is_empty(),
        "stored Muse row-shard model summary is inconsistent"
    );
    let skipped = u64::try_from(manifest.corpus.skipped_prompts.len())
        .context("stored Muse row-shard skipped prompt count")?;
    ensure!(
        manifest.corpus.selected_records == manifest.config.selected_records
            && manifest.corpus.used_prompts > 0
            && manifest.corpus.used_prompts.checked_add(skipped)
                == u64::try_from(manifest.corpus.selected_records).ok()
            && manifest.corpus.truncated_prompts <= manifest.corpus.used_prompts
            && manifest.corpus.ordered_token_ids_blake3 == manifest.config.corpus_blake3
            && manifest.corpus.add_special_tokens == manifest.config.add_special_tokens
            && manifest.corpus.max_tokens == manifest.config.max_tokens
            && manifest.corpus.prompt_reduction == PROMPT_REDUCTION,
        "stored Muse row-shard corpus summary is inconsistent"
    );
    let mut skipped_ids = std::collections::HashSet::new();
    ensure!(
        manifest
            .corpus
            .skipped_prompts
            .iter()
            .all(|prompt| !prompt.id.is_empty() && skipped_ids.insert(&prompt.id)),
        "stored Muse row-shard skipped prompt IDs are empty or duplicated"
    );
    ensure!(
        manifest.fit.method == manifest.config.method
            && manifest.fit.rule_contract == manifest.config.rule_contract
            && manifest.fit.target_layer == manifest.config.target_layer
            && manifest.fit.source_layers == manifest.config.source_layers
            && manifest.fit.coordinate == manifest.config.coordinate
            && manifest.fit.estimator == manifest.config.estimator
            && manifest.fit.orientation == manifest.config.orientation
            && manifest.fit.reduction == manifest.config.reduction
            && manifest.fit.row_start == manifest.config.row_start
            && manifest.fit.row_end == manifest.config.row_end
            && manifest.fit.skip_first == manifest.config.skip_first
            && manifest.fit.query_batch_size == manifest.config.query_batch_size
            && manifest.fit.valid_position_denominator == "number_of_valid_source_positions"
            && manifest.fit.accumulator_dtype == "f32"
            && manifest.fit.storage_dtype == "f32_le"
            && valid_seconds(manifest.fit.forward_seconds)
            && valid_seconds(manifest.fit.vjp_wall_seconds)
            && manifest.fit.vjp_timings.is_valid(),
        "stored Muse row-shard fit summary is inconsistent"
    );
    let shape = expected_shape(&manifest.config)?;
    let expected_bytes = shape
        .into_iter()
        .try_fold(4usize, |bytes, dimension| bytes.checked_mul(dimension))
        .context("stored Muse row-shard payload size overflow")?;
    ensure!(
        manifest.payload.path == PAYLOAD_NAME
            && manifest.payload.dtype == "f32_le"
            && manifest.payload.shape == shape
            && manifest.payload.byte_length == expected_bytes as u64
            && is_blake3(&manifest.payload.blake3),
        "stored Muse row-shard payload descriptor is inconsistent"
    );
    validate_diagnostics(&manifest.diagnostics, &manifest.config)?;
    ensure!(
        manifest.provenance.build_source_state == manifest.config.build_source_state
            && manifest.provenance.build_stamp_error == "none",
        "stored Muse row-shard provenance is inconsistent"
    );
    Ok(())
}

pub(crate) fn validate_complete(
    manifest: &Manifest,
    expected_config: &Config,
    prompts: &[PreparedPrompt],
    expected_config_blake3: &str,
) -> Result<()> {
    validate_stored(manifest)?;
    ensure!(
        &manifest.config == expected_config,
        "completed Muse row-shard config differs from the requested fit"
    );
    ensure!(
        manifest.config_blake3 == expected_config_blake3,
        "completed Muse row-shard config digest is inconsistent"
    );

    let expected_skipped = prompts
        .iter()
        .filter_map(|prompt| super::skipped_prompt(prompt, expected_config.skip_first))
        .collect::<Vec<_>>();
    let expected_used = u64::try_from(prompts.len() - expected_skipped.len())
        .context("Muse row-shard used prompt count")?;
    let expected_truncated = u64::try_from(
        prompts
            .iter()
            .filter(|prompt| {
                super::skipped_prompt(prompt, expected_config.skip_first).is_none()
                    && prompt.truncated
            })
            .count(),
    )
    .context("Muse row-shard truncated prompt count")?;
    ensure!(
        expected_used > 0
            && manifest.corpus.selected_records == expected_config.selected_records
            && manifest.corpus.selected_records == prompts.len()
            && manifest.corpus.used_prompts == expected_used
            && manifest.corpus.skipped_prompts == expected_skipped
            && manifest.corpus.truncated_prompts == expected_truncated
            && manifest.corpus.ordered_token_ids_blake3 == expected_config.corpus_blake3,
        "completed Muse row-shard corpus summary is inconsistent"
    );
    Ok(())
}

pub(crate) fn validate_diagnostics(
    diagnostics: &[ReplayDiagnostic],
    config: &Config,
) -> Result<()> {
    let earliest_source = config
        .source_layers
        .first()
        .copied()
        .context("Muse row-shard has no source layers")?;
    let expected_count = usize::try_from(config.target_layer - earliest_source)
        .context("Muse row-shard diagnostic count")?;
    ensure!(
        diagnostics.len() == expected_count,
        "Muse row-shard replay diagnostic count is inconsistent"
    );
    for (offset, diagnostic) in diagnostics.iter().enumerate() {
        let block = config.target_layer
            - u32::try_from(offset).context("Muse row-shard diagnostic offset")?;
        let expected_kind = if config.geometry.sliding_layers[block as usize] {
            "sliding"
        } else {
            "full"
        };
        ensure!(
            diagnostic.block == block
                && diagnostic.kind == expected_kind
                && diagnostic.post_attention_replay_max_abs_error.is_finite()
                && diagnostic.post_attention_replay_max_abs_error >= 0.0
                && diagnostic.post_block_replay_max_abs_error.is_finite()
                && diagnostic.post_block_replay_max_abs_error >= 0.0,
            "invalid Muse row-shard replay diagnostic at offset {offset}"
        );
    }
    Ok(())
}

pub(crate) fn expected_shape(config: &Config) -> Result<[usize; 3]> {
    let rows =
        usize::try_from(config.row_end - config.row_start).context("Muse row-shard row count")?;
    Ok([
        config.source_layers.len(),
        rows,
        config.geometry.hidden_size as usize,
    ])
}

pub(crate) fn payload(path: &str, bytes: &[u8], shape: [usize; 3]) -> Payload {
    Payload {
        path: path.into(),
        dtype: "f32_le".into(),
        shape,
        byte_length: bytes.len() as u64,
        blake3: blake3::hash(bytes).to_hex().to_string(),
    }
}

pub(crate) fn valid_seconds(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn is_blake3(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_build_source_state(value: &str) -> bool {
    value
        .strip_prefix("git-source-sha256-v2:")
        .is_some_and(is_blake3)
}

fn profile_from_name(value: &str) -> Option<MuseGlimmerArtifactProfile> {
    match value {
        "unsloth_q8_0" => Some(MuseGlimmerArtifactProfile::UnslothQ8_0),
        "unsloth_bf16" => Some(MuseGlimmerArtifactProfile::UnslothBf16),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        let model = MuseGlimmerConfig::unsloth_release_reference();
        make_config(
            &model,
            MuseGlimmerArtifactProfile::UnslothQ8_0,
            "11".repeat(32),
            FitMethod::R,
            51,
            vec![0, 50],
            0,
            3,
            0,
            2,
            2,
            false,
            "22".repeat(32),
            1,
            format!("git-source-sha256-v2:{}", "33".repeat(32)),
        )
    }

    #[test]
    fn config_binds_rows_query_batch_and_full_muse_geometry() {
        let model = MuseGlimmerConfig::unsloth_release_reference();
        let mut value = config();
        assert_eq!(
            serde_json::to_value(&value).unwrap()["method"],
            serde_json::Value::String("R".into())
        );
        validate_config(
            &value,
            &model,
            MuseGlimmerArtifactProfile::UnslothQ8_0,
            &"11".repeat(32),
        )
        .unwrap();

        value.row_end = 33;
        assert!(
            validate_config(
                &value,
                &model,
                MuseGlimmerArtifactProfile::UnslothQ8_0,
                &"11".repeat(32)
            )
            .is_err()
        );
        value = config();
        value.query_batch_size = MUSE_GLIMMER_QUERY_BATCH_MAX + 1;
        assert!(
            validate_config(
                &value,
                &model,
                MuseGlimmerArtifactProfile::UnslothQ8_0,
                &"11".repeat(32)
            )
            .is_err()
        );
        value = config();
        value.geometry.sliding_layers[51] = true;
        assert!(
            validate_config(
                &value,
                &model,
                MuseGlimmerArtifactProfile::UnslothQ8_0,
                &"11".repeat(32)
            )
            .is_err()
        );
    }

    #[test]
    fn diagnostic_schedule_distinguishes_full_and_sliding_blocks() {
        let config = config();
        let diagnostics = (1..=51)
            .rev()
            .map(|block| ReplayDiagnostic {
                block,
                kind: if config.geometry.sliding_layers[block as usize] {
                    "sliding".into()
                } else {
                    "full".into()
                },
                post_attention_replay_max_abs_error: 0.0,
                post_block_replay_max_abs_error: 0.0,
            })
            .collect::<Vec<_>>();
        validate_diagnostics(&diagnostics, &config).unwrap();
        let mut invalid = diagnostics;
        invalid[0].kind = "sliding".into();
        assert!(validate_diagnostics(&invalid, &config).is_err());
    }
}
