use super::lens_run::{LensDefinition, LensPlan, LensRunArgs, Scope, Selector};
use super::muse_lens_artifact as artifact;
use anyhow::{Context, Result, bail, ensure};
use qwen_llm::checkpoint_identity::{CheckpointIdentityCache, checkpoint_content_identity};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::muse_glimmer::{MuseGlimmerArtifactProfile, MuseGlimmerConfig, MuseGlimmerModel};
use qwen_llm::muse_glimmer_runtime::MuseGlimmerLoadedModel;
use qwen_llm::sampling::{Sampler, SamplingConfig};
use qwen_llm::tokenizer::{LlamaCppTokenizer, Tokenize};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

struct LoadedMuseLens {
    method: String,
    target_layer: u32,
    source_layer: u32,
    token_ids: Vec<u32>,
    hidden_size: usize,
    values: Vec<f32>,
}

#[derive(Clone, Copy)]
enum Event {
    Prefill(usize),
    Decode(usize),
}

impl Event {
    fn label(self) -> &'static str {
        match self {
            Self::Prefill(_) => "prefill",
            Self::Decode(_) => "decode",
        }
    }
    fn index(self) -> usize {
        match self {
            Self::Prefill(i) | Self::Decode(i) => i,
        }
    }
}

#[derive(Serialize)]
struct Output {
    prompt_token_ids: Vec<i32>,
    generated_token_ids: Vec<i32>,
    decoded_text: String,
    stop_reason: String,
    operation_applications: Vec<serde_json::Value>,
    live_readouts: Vec<LiveReadout>,
}

#[derive(Debug, Serialize)]
struct LiveReadout {
    id: String,
    lens: String,
    method: String,
    source_layer: u32,
    target_layer: Option<u32>,
    phase: &'static str,
    index: usize,
    scores: Vec<LiveScore>,
}

#[derive(Debug, Serialize, PartialEq)]
struct LiveScore {
    token_id: Option<i32>,
    row_id: usize,
    word_id: Option<i64>,
    label: Option<String>,
    score: f32,
}

pub(crate) fn run(
    args: &LensRunArgs,
    plan: LensPlan,
    plan_dir: &Path,
    gguf: GgufFile,
) -> Result<()> {
    validate_plan(&plan, args)?;
    let cache = args
        .identity_cache
        .as_ref()
        .context("Muse Glimmer qwen-lens run requires --identity-cache")?;
    let bound = MuseGlimmerModel::from_gguf(&gguf).context("bind running Muse Glimmer model")?;
    let config = bound.config.clone();
    let profile = bound.artifact_profile;
    drop(bound);
    let tokenizer = LlamaCppTokenizer::from_gguf(&gguf, &args.model)
        .context("load Muse llama.cpp tokenizer")?;
    artifact::validate_tokenizer(&tokenizer, &config)?;
    let prompt_ids = input_tokens(args, &tokenizer)?;
    ensure!(
        !prompt_ids.is_empty(),
        "prompt must encode to at least one token"
    );
    ensure!(
        prompt_ids.len() <= 4096 * 16,
        "prompt is too long for the bounded Lens runner"
    );
    ensure!(
        prompt_ids
            .iter()
            .all(|&id| id >= 0 && (id as u32) < config.vocab_size),
        "Muse prompt contains an invalid token ID"
    );
    validate_schedule(
        &plan,
        prompt_ids.len(),
        args.max_new_tokens,
        config.layer_count,
    )?;

    let content = checkpoint_content_identity(&gguf, &CheckpointIdentityCache::new(cache))
        .with_context(|| {
            format!(
                "resolve running Muse GGUF identity using {}",
                cache.display()
            )
        })?;
    let content_id = super::hex(&content.content_id);
    let mut lenses = HashMap::new();
    for lens in &plan.lenses {
        let LensDefinition::NativeSelected { id, artifact: path } = lens else {
            bail!("Muse plans support native_selected lenses only");
        };
        let loaded = load_muse_artifact(&resolve(plan_dir, path), &config, profile, &content_id)?;
        ensure!(
            lenses.insert(id.clone(), loaded).is_none(),
            "duplicate Muse lens id"
        );
    }
    let capture_layers = lenses
        .values()
        .map(|lens| lens.source_layer)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let layer_slots = capture_layers
        .iter()
        .enumerate()
        .map(|(slot, &layer)| (layer, slot))
        .collect::<HashMap<_, _>>();
    for readout in &plan.readouts {
        let lens = &lenses[&readout.lens];
        ensure!(
            selector_values(&readout.scope.layers, config.layer_count)?.as_slice()
                == [lens.source_layer],
            "Muse readout {} must select exactly artifact source layer {}",
            readout.id,
            lens.source_layer
        );
    }

    let forward_count = prompt_ids
        .len()
        .checked_add(args.max_new_tokens.saturating_sub(1))
        .context("Muse forward count overflow")?;
    let context = MetalContext::new().context("initialize Metal for Muse Lens run")?;
    let mut loaded = MuseGlimmerLoadedModel::load(&context, &gguf, forward_count)
        .context("load Muse Lens runner model")?;
    let mut runner = loaded
        .create_runner(&context)
        .context("create Muse Lens runner")?;
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
    })?;
    let stops = [config.eos_token_id as i32, config.eot_token_id as i32]
        .into_iter()
        .collect::<HashSet<_>>();
    let mut live_readouts = Vec::new();
    let mut logits = Vec::new();
    for (index, &token) in prompt_ids.iter().enumerate() {
        logits = forward_event(
            &plan,
            &lenses,
            &capture_layers,
            &layer_slots,
            &mut runner,
            token as u32,
            Event::Prefill(index),
            &mut live_readouts,
        )?;
    }
    let mut generated = Vec::new();
    let mut stop_reason = "max_new_tokens".to_string();
    for index in 0..args.max_new_tokens {
        let token = sampler.sample(&logits)?.token;
        generated.push(token);
        if stops.contains(&token) {
            stop_reason = "stop_token".into();
            break;
        }
        if index + 1 == args.max_new_tokens {
            break;
        }
        logits = forward_event(
            &plan,
            &lenses,
            &capture_layers,
            &layer_slots,
            &mut runner,
            token as u32,
            Event::Decode(index),
            &mut live_readouts,
        )?;
    }
    println!(
        "{}",
        serde_json::to_string(&Output {
            prompt_token_ids: prompt_ids,
            generated_token_ids: generated.clone(),
            decoded_text: tokenizer.decode(&generated),
            stop_reason,
            operation_applications: Vec::new(),
            live_readouts
        })?
    );
    Ok(())
}

fn load_muse_artifact(
    artifact_path: &Path,
    config: &MuseGlimmerConfig,
    profile: MuseGlimmerArtifactProfile,
    content_id: &str,
) -> Result<LoadedMuseLens> {
    let manifest_path = if artifact_path.is_dir() {
        artifact_path.join(artifact::MANIFEST_NAME)
    } else {
        artifact_path.to_path_buf()
    };
    let manifest: artifact::Manifest = serde_json::from_slice(&super::read_regular_file_bounded(
        &manifest_path,
        artifact::MAX_MANIFEST_BYTES,
    )?)
    .with_context(|| format!("parse Muse artifact {}", manifest_path.display()))?;
    artifact::validate(&manifest, config, profile, content_id)
        .with_context(|| format!("validate Muse artifact {}", manifest_path.display()))?;
    let expected_bytes = usize::try_from(manifest.payload.byte_length)
        .context("Muse payload byte length does not fit this platform")?;
    let directory = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let bytes =
        super::read_regular_file_exact(&directory.join(&manifest.payload.path), expected_bytes)?;
    let values = artifact::decode_payload(&bytes, &manifest)?;
    Ok(LoadedMuseLens {
        method: manifest.transport.method,
        target_layer: manifest.transport.target_layer,
        source_layer: manifest.transport.source_layers[0],
        token_ids: manifest.selected.token_ids,
        hidden_size: config.hidden_size as usize,
        values,
    })
}

fn validate_plan(plan: &LensPlan, args: &LensRunArgs) -> Result<()> {
    ensure!(
        args.messages.is_none(),
        "Muse Lens run does not support --messages"
    );
    ensure!(
        plan.directions.is_empty(),
        "Muse Lens plans do not support directions"
    );
    ensure!(
        plan.operations.is_empty(),
        "Muse Lens plans do not support operations"
    );
    ensure!(
        !plan.readouts.is_empty(),
        "Muse Lens plans require readouts"
    );
    ensure!(
        plan.lenses
            .iter()
            .all(|lens| matches!(lens, LensDefinition::NativeSelected { .. })),
        "Muse Lens plans support native_selected lenses only"
    );
    Ok(())
}

fn input_tokens(args: &LensRunArgs, tokenizer: &impl Tokenize) -> Result<Vec<i32>> {
    match (&args.prompt, &args.token_ids, &args.messages) {
        (Some(text), None, None) => Ok(tokenizer.encode(text, !args.no_special_tokens)?),
        (None, Some(ids), None) => {
            ensure!(!ids.is_empty(), "--token-ids must not be empty");
            Ok(ids.clone())
        }
        _ => bail!("Muse Lens run requires exactly one of --prompt or --token-ids"),
    }
}

fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.into()
    } else {
        base.join(path)
    }
}

fn selector_values(selector: &Selector, bound: u32) -> Result<Vec<u32>> {
    let values = match selector {
        Selector::All => (0..bound).collect(),
        Selector::Values { values } => values.clone(),
        Selector::Range { start, end } => {
            ensure!(start <= end, "invalid selector range");
            (*start..=*end).collect()
        }
    };
    ensure!(
        values.iter().all(|&v| v < bound),
        "selector value is outside event bound"
    );
    Ok(values)
}

fn validate_schedule(
    plan: &LensPlan,
    prompt_len: usize,
    max_new: usize,
    layers: u32,
) -> Result<()> {
    for readout in &plan.readouts {
        ensure!(
            !selector_values(&readout.scope.layers, layers)?.is_empty(),
            "Muse readout has empty layer schedule"
        );
        if let Some(prefill) = &readout.scope.prefill {
            ensure!(
                !selector_values(prefill, prompt_len as u32)?.is_empty(),
                "Muse readout has empty prefill schedule"
            );
        }
        if let Some(decode) = &readout.scope.decode {
            let bound = max_new.saturating_sub(1);
            ensure!(
                bound > 0,
                "Muse decode readout cannot run when --max-new-tokens is 1"
            );
            ensure!(
                !selector_values(decode, bound as u32)?.is_empty(),
                "Muse readout has empty decode schedule"
            );
        }
    }
    Ok(())
}

fn matches(scope: &Scope, event: Event, layer: u32) -> bool {
    let contains = |selector: &Selector, value| match selector {
        Selector::All => true,
        Selector::Values { values } => values.binary_search(&value).is_ok(),
        Selector::Range { start, end } => (*start..=*end).contains(&value),
    };
    contains(&scope.layers, layer)
        && match event {
            Event::Prefill(index) => scope
                .prefill
                .as_ref()
                .is_some_and(|s| contains(s, index as u32)),
            Event::Decode(index) => scope
                .decode
                .as_ref()
                .is_some_and(|s| contains(s, index as u32)),
        }
}

fn score(lens: &LoadedMuseLens, row: &[f32], top_k: usize) -> Vec<LiveScore> {
    let mut scores = lens
        .token_ids
        .iter()
        .enumerate()
        .map(|(slot, &token)| {
            let offset = slot * lens.hidden_size;
            let value = row
                .iter()
                .zip(&lens.values[offset..offset + lens.hidden_size])
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>();
            (slot, token, value)
        })
        .collect::<Vec<_>>();
    scores.sort_unstable_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.1.cmp(&b.1)));
    scores.truncate(top_k.min(scores.len()));
    scores
        .into_iter()
        .map(|(row_id, token, value)| LiveScore {
            token_id: Some(token as i32),
            row_id,
            word_id: None,
            label: None,
            score: value as f32,
        })
        .collect()
}

fn forward_event(
    plan: &LensPlan,
    lenses: &HashMap<String, LoadedMuseLens>,
    capture_layers: &[u32],
    layer_slots: &HashMap<u32, usize>,
    runner: &mut qwen_llm::muse_glimmer_runtime::MuseGlimmerTextRunner<'_, '_>,
    token: u32,
    event: Event,
    output: &mut Vec<LiveReadout>,
) -> Result<Vec<f32>> {
    let captured = runner
        .forward_token_capture_post_blocks(token, capture_layers)
        .with_context(|| format!("forward Muse {} event {}", event.label(), event.index()))?;
    for readout in &plan.readouts {
        let lens = &lenses[&readout.lens];
        if !matches(&readout.scope, event, lens.source_layer) {
            continue;
        }
        let row = captured
            .layer_values(layer_slots[&lens.source_layer])
            .context("Muse captured source layer is missing")?;
        output.push(LiveReadout {
            id: readout.id.clone(),
            lens: readout.lens.clone(),
            method: lens.method.clone(),
            source_layer: lens.source_layer,
            target_layer: Some(lens.target_layer),
            phase: event.label(),
            index: event.index(),
            scores: score(lens, row, readout.top_k),
        });
    }
    Ok(captured.logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scoring_is_f64_dot_and_stable_token_tie_break() {
        let lens = LoadedMuseLens {
            method: "J".into(),
            target_layer: 51,
            source_layer: 50,
            token_ids: vec![9, 3],
            hidden_size: 2,
            values: vec![1.0, 2.0, 1.0, 2.0],
        };
        let scores = score(&lens, &[3.0, 4.0], 2);
        assert_eq!(
            scores.iter().map(|s| s.token_id).collect::<Vec<_>>(),
            vec![Some(3), Some(9)]
        );
        assert_eq!(scores[0].score, 11.0);
    }

    #[test]
    fn muse_plan_rejects_operations_and_templates() {
        let plan: LensPlan = serde_json::from_value(json!({"version":1,"lenses":[{"kind":"workspace_template","id":"x","weights":"w","labels":"l"}],"directions":[],"operations":[],"readouts":[{"id":"r","lens":"x","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"all"}},"top_k":1}]})).unwrap();
        let args = LensRunArgs {
            model: "m".into(),
            plan: "p".into(),
            identity_cache: Some("c".into()),
            prompt: Some("x".into()),
            token_ids: None,
            messages: None,
            no_special_tokens: false,
            max_new_tokens: 1,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        };
        assert!(validate_plan(&plan, &args).is_err());
    }

    #[test]
    fn decode_schedule_excludes_unforwarded_last_sample() {
        let selector = Selector::Values { values: vec![1] };
        assert!(selector_values(&selector, 2).is_ok());
        assert!(selector_values(&selector, 1).is_err());
    }

    #[test]
    #[ignore = "requires MUSE_GLIMMER_Q8_GGUF and loads the real 30B Q8 model twice"]
    fn real_q8_fit_payload_fresh_runner_score_matches_direct_f64_layer_50_dot() -> Result<()> {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF")
            .expect("set MUSE_GLIMMER_Q8_GGUF to the authenticated Unsloth Q8 GGUF");
        let root = std::env::temp_dir().join(format!(
            "qwen-muse-real-proof-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&root)?;
        let result = real_q8_proof(Path::new(&path), &root);
        let cleanup = std::fs::remove_dir_all(&root);
        result?;
        cleanup?;
        Ok(())
    }

    fn real_q8_proof(path: &Path, root: &Path) -> Result<()> {
        let gguf = GgufFile::open(path)?;
        let bound = MuseGlimmerModel::from_gguf(&gguf)?;
        let config = bound.config.clone();
        let profile = bound.artifact_profile;
        drop(bound);
        let content = checkpoint_content_identity(
            &gguf,
            &CheckpointIdentityCache::new(root.join("identity")),
        )?;
        let content_id = super::super::hex(&content.content_id);
        let selected_id = config.eos_token_id;
        let prompt = [config.bos_token_id, config.eos_token_id];

        let context = MetalContext::new()?;
        let mut model = MuseGlimmerLoadedModel::load(&context, &gguf, prompt.len())?;
        let covectors = model.selected_token_lens_covectors(&context, &[selected_id])?;
        let mut runner = model.create_runner(&context)?;
        let capture = runner.capture_fresh_lens_prompt(&prompt, 51)?;
        let mut artifacts = Vec::new();
        for rule in [
            qwen_llm::muse_glimmer_lens::MuseGlimmerLensRule::J,
            qwen_llm::muse_glimmer_lens::MuseGlimmerLensRule::R,
        ] {
            let fit = runner
                .fit_adjacent_full_attention_selected_tokens(&capture, &covectors, 0, rule)?;
            let directory = root.join(rule.as_str().to_ascii_lowercase());
            std::fs::create_dir(&directory)?;
            let payload = fit
                .values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let manifest = proof_manifest(
                &config,
                profile,
                &content_id,
                selected_id,
                rule,
                &fit,
                &payload,
            );
            artifact::validate(&manifest, &config, profile, &content_id)?;
            std::fs::write(directory.join(artifact::PAYLOAD_NAME), &payload)?;
            std::fs::write(
                directory.join(artifact::MANIFEST_NAME),
                artifact::serialize_manifest(&manifest)?,
            )?;
            let load_path = if rule == qwen_llm::muse_glimmer_lens::MuseGlimmerLensRule::J {
                directory
            } else {
                directory.join(artifact::MANIFEST_NAME)
            };
            artifacts.push(load_muse_artifact(
                &load_path,
                &config,
                profile,
                &content_id,
            )?);
        }
        drop(runner);
        drop(model);
        drop(context);

        let context = MetalContext::new()?;
        let mut model = MuseGlimmerLoadedModel::load(&context, &gguf, 1)?;
        let mut runner = model.create_runner(&context)?;
        let live = runner.forward_token_capture_post_blocks(config.bos_token_id, &[50])?;
        let residual = live
            .layer_values(0)
            .context("missing layer-50 proof capture")?;
        for lens in artifacts {
            let emitted = score(&lens, residual, 1)[0].score;
            let manual = residual
                .iter()
                .zip(&lens.values)
                .map(|(&left, &right)| f64::from(left) * f64::from(right))
                .sum::<f64>();
            eprintln!(
                "muse_method={} layer50_emitted={emitted:.9e} manual_f64={manual:.17e}",
                lens.method
            );
            ensure!(
                emitted.to_bits() == (manual as f32).to_bits(),
                "{} emitted score differs from direct F64 dot",
                lens.method
            );
        }
        Ok(())
    }

    fn proof_manifest(
        config: &MuseGlimmerConfig,
        profile: MuseGlimmerArtifactProfile,
        content_id: &str,
        selected_id: u32,
        rule: qwen_llm::muse_glimmer_lens::MuseGlimmerLensRule,
        fit: &qwen_llm::muse_glimmer_lens_fit::MuseGlimmerAdjacentSelectedTokenFit,
        payload: &[u8],
    ) -> artifact::Manifest {
        artifact::Manifest {
            schema: artifact::SCHEMA.into(),
            schema_version: artifact::SCHEMA_VERSION,
            architecture: qwen_llm::muse_glimmer::ARCHITECTURE_NAME.into(),
            artifact_profile: artifact::profile_name(profile).into(),
            model_content_blake3: content_id.into(),
            geometry: artifact::geometry(config),
            transport: artifact::Transport {
                method: rule.as_str().into(),
                rule_contract: artifact::RULE_CONTRACT.into(),
                target_layer: 51,
                source_layers: vec![50],
                coordinate: artifact::COORDINATE.into(),
                estimator: artifact::ESTIMATOR.into(),
                reduction: artifact::REDUCTION.into(),
                replay_semantics: artifact::REPLAY_SEMANTICS.into(),
                production_semantics: artifact::PRODUCTION_SEMANTICS.into(),
                skip_first: 0,
            },
            selected: artifact::Selected {
                token_ids: vec![selected_id],
                score: artifact::SCORE.into(),
                covector_formula: "logit_scale * output_norm_gamma * output_weight[token_id]"
                    .into(),
                logit_scale: config.logit_scale,
            },
            payload: artifact::Payload {
                path: artifact::PAYLOAD_NAME.into(),
                dtype: "f32_le".into(),
                shape: [1, 1, config.hidden_size as usize],
                byte_length: payload.len() as u64,
                blake3: blake3::hash(payload).to_hex().to_string(),
            },
            corpus: artifact::Corpus {
                selected_records: 1,
                used_prompts: 1,
                skipped_prompts: vec![],
                truncated_prompts: 0,
                ordered_token_ids_blake3: "real_q8_literal_bos_eos".into(),
                add_special_tokens: false,
                max_tokens: 2,
                prompt_reduction: "arithmetic_mean_over_used_prompts".into(),
            },
            replay: artifact::Replay {
                f32_vs_production_f16_kv_post_attention_max_abs: fit
                    .post_attention_replay_max_abs_error,
                f32_vs_production_f16_kv_post_block_max_abs: fit.post_block_replay_max_abs_error,
            },
            provenance: artifact::Provenance {
                build_commit: env!("QWEN_BUILD_COMMIT").into(),
                build_dirty: env!("QWEN_BUILD_DIRTY").into(),
                build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
            },
        }
    }
}
