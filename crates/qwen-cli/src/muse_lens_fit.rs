use super::muse_lens_artifact as artifact;
use super::{FitMethod, FitTokensArgs};
use anyhow::{Context, Result, bail, ensure};
use qwen_llm::checkpoint_identity::{CheckpointIdentityCache, checkpoint_content_identity};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::muse_glimmer::{ARCHITECTURE_NAME, MuseGlimmerModel};
use qwen_llm::muse_glimmer_lens::{
    MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS, MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS, MuseGlimmerLensRule,
};
use qwen_llm::muse_glimmer_runtime::MuseGlimmerLoadedModel;
use qwen_llm::tokenizer::LlamaCppTokenizer;
use std::fs::{DirBuilder, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn fit_tokens(mut args: FitTokensArgs, gguf: GgufFile) -> Result<()> {
    validate_args(&args)?;
    super::validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    let bound = MuseGlimmerModel::from_gguf(&gguf)
        .context("bind Muse Glimmer artifact profile and geometry")?;
    let config = bound.config.clone();
    let profile = bound.artifact_profile;
    drop(bound);
    ensure!(
        args.target_layer < config.layer_count,
        "--target-layer is outside Muse layer geometry"
    );
    if let Some(&id) = args.token_ids.iter().find(|&&id| id >= config.vocab_size) {
        bail!(
            "--token-ids entry {id} is outside Muse vocab {}",
            config.vocab_size
        );
    }

    args.output = super::resolve_output_path(&args.output)?;
    ensure!(
        !args.output.exists(),
        "Muse fitting requires a new --output directory"
    );

    let requests = super::read_prompt_requests(&args.prompts, args.max_prompts)?;
    let tokenizer = LlamaCppTokenizer::from_gguf(&gguf, &args.model)
        .context("load Muse llama.cpp tokenizer")?;
    artifact::validate_tokenizer(&tokenizer, &config)?;
    let add_special_tokens = !args.no_special_tokens;
    let prompts = super::prepare_prompts(
        requests,
        &tokenizer,
        add_special_tokens,
        args.max_tokens,
        config.vocab_size,
    )?;
    let corpus_blake3 = super::corpus_digest(&prompts);
    let content =
        checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(&args.identity_cache))
            .with_context(|| {
                format!(
                    "resolve Muse GGUF content identity using {}",
                    args.identity_cache.display()
                )
            })?;
    let content_id = super::hex(&content.content_id);

    let context = MetalContext::new().context("initialize Metal for Muse lens fitting")?;
    let mut loaded = MuseGlimmerLoadedModel::load(&context, &gguf, args.max_tokens)
        .context("load Muse Glimmer lens model")?;
    let covectors = loaded
        .selected_token_lens_covectors(&context, &args.token_ids)
        .context("derive Muse selected-token covectors")?;
    ensure!(
        covectors.token_ids() == args.token_ids && covectors.values().iter().all(|v| v.is_finite()),
        "invalid Muse selected-token covectors"
    );
    let logit_scale = covectors.logit_scale();
    let mut runner = loaded
        .create_runner(&context)
        .context("create Muse lens runner")?;
    let hidden = config.hidden_size as usize;
    let count = args
        .source_layers
        .len()
        .checked_mul(args.token_ids.len())
        .and_then(|count| count.checked_mul(hidden))
        .context("Muse accumulator size overflow")?;
    let mut sums = vec![0.0f32; count];
    let mut used_prompts = 0u64;
    let mut truncated_prompts = 0u64;
    let mut skipped_prompts = Vec::new();
    let mut attention_drift = 0.0f32;
    let mut block_drift = 0.0f32;
    let rule = match args.method {
        FitMethod::J => MuseGlimmerLensRule::J,
        FitMethod::R => MuseGlimmerLensRule::R,
    };
    let traversed_blocks = ((args.source_layers[0] + 1)..=args.target_layer).collect::<Vec<_>>();

    for (index, prompt) in prompts.iter().enumerate() {
        if let Some(skipped) = super::skipped_prompt(prompt, args.skip_first) {
            skipped_prompts.push(skipped);
            continue;
        }
        eprintln!(
            "fit Muse prompt {}/{} id={} tokens={} selected_tokens={} sources={:?}",
            index + 1,
            prompts.len(),
            prompt.id,
            prompt.token_ids.len(),
            args.token_ids.len(),
            args.source_layers
        );
        runner
            .reset()
            .with_context(|| format!("reset Muse session for prompt {}", prompt.id))?;
        let tokens = prompt
            .token_ids
            .iter()
            .map(|&id| id as u32)
            .collect::<Vec<_>>();
        let captures = runner
            .capture_fresh_lens_prompt_blocks(&tokens, &traversed_blocks)
            .with_context(|| format!("capture Muse prompt {}", prompt.id))?;
        let fit = runner
            .fit_selected_tokens_to_sources(
                &captures,
                args.target_layer,
                &args.source_layers,
                &covectors,
                args.skip_first,
                rule,
            )
            .with_context(|| format!("fit Muse prompt {}", prompt.id))?;
        ensure!(
            fit.source_layers == args.source_layers
                && fit.target_block == args.target_layer
                && fit.method == rule
                && fit.token_ids == args.token_ids
                && fit.hidden_size == hidden
                && fit.values.len() == sums.len(),
            "Muse fit returned inconsistent metadata"
        );
        for (sum, value) in sums.iter_mut().zip(fit.values) {
            *sum += value;
        }
        ensure!(
            sums.iter().all(|v| v.is_finite()),
            "Muse fit accumulator became non-finite"
        );
        for diagnostic in fit.diagnostics {
            attention_drift = attention_drift.max(diagnostic.post_attention_replay_max_abs_error);
            block_drift = block_drift.max(diagnostic.post_block_replay_max_abs_error);
        }
        used_prompts += 1;
        truncated_prompts += u64::from(prompt.truncated);
    }
    ensure!(
        used_prompts > 0,
        "no Muse corpus prompt had valid fit positions"
    );
    let scale = (used_prompts as f32).recip();
    sums.iter_mut().for_each(|value| *value *= scale);
    ensure!(
        sums.iter().all(|v| v.is_finite()),
        "averaged Muse readouts are non-finite"
    );
    let payload_bytes = super::encode_f32_le_fallible(&sums)?;
    let payload = artifact::Payload {
        path: artifact::PAYLOAD_NAME.into(),
        dtype: "f32_le".into(),
        shape: [args.source_layers.len(), args.token_ids.len(), hidden],
        byte_length: payload_bytes.len() as u64,
        blake3: blake3::hash(&payload_bytes).to_hex().to_string(),
    };
    let manifest = artifact::Manifest {
        schema: artifact::SCHEMA.into(),
        schema_version: artifact::SCHEMA_VERSION,
        architecture: ARCHITECTURE_NAME.into(),
        artifact_profile: artifact::profile_name(profile).into(),
        model_content_blake3: content_id.clone(),
        geometry: artifact::geometry(&config),
        transport: artifact::Transport {
            method: rule.as_str().into(),
            rule_contract: artifact::RULE_CONTRACT_V2.into(),
            target_layer: args.target_layer,
            source_layers: args.source_layers.clone(),
            coordinate: artifact::COORDINATE.into(),
            estimator: artifact::ESTIMATOR_V2.into(),
            reduction: artifact::REDUCTION.into(),
            replay_semantics: artifact::REPLAY_SEMANTICS.into(),
            production_semantics: artifact::PRODUCTION_SEMANTICS.into(),
            skip_first: args.skip_first,
        },
        selected: artifact::Selected {
            token_ids: args.token_ids.clone(),
            score: artifact::SCORE.into(),
            covector_formula: "logit_scale * output_norm_gamma * output_weight[token_id]".into(),
            logit_scale,
        },
        payload,
        corpus: artifact::Corpus {
            selected_records: prompts.len(),
            used_prompts,
            skipped_prompts,
            truncated_prompts,
            ordered_token_ids_blake3: corpus_blake3,
            add_special_tokens,
            max_tokens: args.max_tokens,
            prompt_reduction: "arithmetic_mean_over_used_prompts".into(),
        },
        replay: artifact::Replay {
            f32_vs_production_f16_kv_post_attention_max_abs: attention_drift,
            f32_vs_production_f16_kv_post_block_max_abs: block_drift,
        },
        provenance: artifact::Provenance {
            build_commit: env!("QWEN_BUILD_COMMIT").into(),
            build_dirty: env!("QWEN_BUILD_DIRTY").into(),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
        },
    };
    artifact::validate(&manifest, &config, profile, &content_id)?;
    let manifest_bytes = artifact::serialize_manifest(&manifest)?;
    publish_artifact_directory(&args.output, &payload_bytes, &manifest_bytes)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn publish_artifact_directory(output: &Path, payload: &[u8], manifest: &[u8]) -> Result<()> {
    ensure!(
        !output.exists(),
        "Muse fitting requires a new --output directory"
    );
    let parent = output.parent().context("Muse output has no parent")?;
    let leaf = output
        .file_name()
        .context("Muse output has no directory name")?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_nanos();
    let staging = parent.join(format!(
        ".{}.stage.{}.{}",
        leaf.to_string_lossy(),
        std::process::id(),
        nonce
    ));
    DirBuilder::new()
        .mode(0o700)
        .create(&staging)
        .with_context(|| format!("create Muse staging directory {}", staging.display()))?;
    let staged = (|| -> Result<()> {
        write_staged_file(&staging.join(artifact::PAYLOAD_NAME), payload)?;
        write_staged_file(&staging.join(artifact::MANIFEST_NAME), manifest)?;
        super::sync_directory(&staging)?;
        std::fs::rename(&staging, output).with_context(|| {
            format!(
                "publish Muse staging directory {} to {}",
                staging.display(),
                output.display()
            )
        })?;
        if let Err(error) = super::sync_directory(parent) {
            let _ = std::fs::remove_dir_all(output);
            let _ = super::sync_directory(parent);
            return Err(error).context("sync Muse artifact parent after publication");
        }
        Ok(())
    })();
    if staged.is_err() && staging.exists() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    staged
}

fn write_staged_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("create staged Muse artifact {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write staged Muse artifact {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync staged Muse artifact {}", path.display()))
}

fn validate_args(args: &FitTokensArgs) -> Result<()> {
    ensure!(
        !args.resume,
        "--resume is not supported by the first Muse fitting contract"
    );
    ensure!(args.dim_batch == 1, "Muse fitting requires --dim-batch 1");
    ensure!(
        args.target_layer > 0,
        "Muse fitting requires a nonzero --target-layer"
    );
    ensure!(
        !args.source_layers.is_empty()
            && args.source_layers.windows(2).all(|pair| pair[0] < pair[1])
            && args
                .source_layers
                .iter()
                .all(|&source| source < args.target_layer),
        "Muse --source-layers must be nonempty, strictly increasing, and below --target-layer"
    );
    ensure!(
        args.max_tokens > 0 && args.max_tokens <= MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS,
        "Muse --max-tokens must be in 1..={MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS}"
    );
    ensure!(
        args.skip_first
            .checked_add(2)
            .is_some_and(|n| n <= args.max_tokens),
        "Muse --max-tokens must be at least --skip-first + 2"
    );
    ensure!(
        !args.token_ids.is_empty() && args.token_ids.len() <= MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS,
        "Muse selected-token count must be in 1..={MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS}"
    );
    let mut ids = std::collections::HashSet::new();
    ensure!(
        args.token_ids.iter().all(|id| ids.insert(*id)),
        "Muse --token-ids must be unique"
    );
    ensure!(
        args.max_prompts > 0 && args.max_prompts <= super::MAX_PROMPT_RECORDS,
        "Muse --max-prompts is outside the supported bound"
    );
    ensure!(
        args.prompts != std::path::Path::new("-"),
        "Muse fitting requires a replayable prompt corpus file"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn args() -> FitTokensArgs {
        FitTokensArgs {
            model: PathBuf::from("m.gguf"),
            prompts: PathBuf::from("p.jsonl"),
            output: PathBuf::from("out"),
            identity_cache: PathBuf::from("cache"),
            method: FitMethod::J,
            target_layer: 51,
            source_layers: vec![50],
            token_ids: vec![1],
            dim_batch: 1,
            skip_first: 0,
            max_tokens: 2,
            max_prompts: 1,
            no_special_tokens: true,
            resume: false,
        }
    }

    #[test]
    fn multi_source_contract_accepts_arbitrary_sorted_sources_and_rejects_bad_bounds() {
        let mut value = args();
        value.dim_batch = 2;
        assert!(validate_args(&value).is_err());
        let mut value = args();
        value.source_layers = vec![49, 50];
        validate_args(&value).unwrap();
        value.source_layers = vec![50, 49];
        assert!(validate_args(&value).is_err());
        value.source_layers = vec![50, 51];
        assert!(validate_args(&value).is_err());
        let mut value = args();
        value.resume = true;
        assert!(validate_args(&value).is_err());
    }

    #[test]
    fn artifact_directory_is_published_by_one_final_rename() {
        let root = std::env::temp_dir().join(format!(
            "qwen-muse-publish-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let output = root.join("artifact");
        publish_artifact_directory(&output, b"payload", b"manifest").unwrap();
        assert_eq!(
            std::fs::read(output.join(artifact::PAYLOAD_NAME)).unwrap(),
            b"payload"
        );
        assert_eq!(
            std::fs::read(output.join(artifact::MANIFEST_NAME)).unwrap(),
            b"manifest"
        );
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }
}
