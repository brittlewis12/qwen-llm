use super::lens_run::{
    Action, DirectionDefinition, DirectionRow, LensDefinition, LensPlan, LensRunArgs, LiveReadout,
    LiveScore, OperationApplication, RunExecutionBinding, RunPublishedLensBinding,
    RunPublishedMatrixBinding, RunResult, Scope, Selector, emit_run_output,
};
use super::muse_lens_artifact as artifact;
use super::muse_published_full_lens_artifact as published;
use anyhow::{Context, Result, bail, ensure};
use qwen_llm::checkpoint_identity::{
    CheckpointIdentityCache, checkpoint_content_identity_without_weight_hashing,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::{MetalContext, MetalTensor, PostBlockIntervention};
use qwen_llm::muse_glimmer::{MuseGlimmerArtifactProfile, MuseGlimmerConfig, MuseGlimmerModel};
use qwen_llm::muse_glimmer_runtime::MuseGlimmerLoadedModel;
use qwen_llm::sampling::{Sampler, SamplingConfig};
use qwen_llm::tensor::GgmlType;
use qwen_llm::tokenizer::{LlamaCppTokenizer, Tokenize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

struct LoadedMuseLens {
    method: String,
    candidate_universe: &'static str,
    target_layer: u32,
    source_layers: Vec<u32>,
    token_ids: Vec<u32>,
    hidden_size: usize,
    values: Vec<f32>,
}

struct PreparedMuseDirection {
    rows: BTreeMap<u32, MetalTensor>,
}

struct MuseExecutionPlan {
    plan: LensPlan,
    lenses: HashMap<String, LoadedMuseLens>,
    directions: HashMap<String, PreparedMuseDirection>,
    coordinate_swaps: HashMap<String, PreparedMuseDirection>,
    capture_layers: Vec<u32>,
    layer_slots: HashMap<u32, usize>,
    layer_count: u32,
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

pub(crate) fn run(
    args: &LensRunArgs,
    plan: LensPlan,
    plan_path: &Path,
    plan_dir: &Path,
    gguf: GgufFile,
    output_path: Option<&Path>,
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
    super::lens_run::validate_reachable_scopes(&plan, prompt_ids.len(), args.max_new_tokens)?;

    let content = checkpoint_content_identity_without_weight_hashing(
        &gguf,
        &CheckpointIdentityCache::new(cache),
    )
    .with_context(|| {
        format!(
            "resolve running Muse GGUF identity without hashing weights using {}",
            cache.display()
        )
    })?;
    ensure!(
        content.bytes_hashed == 0,
        "Muse lens execution must not hash model weights"
    );
    let content_id = super::hex(&content.content_id);
    let forward_count = prompt_ids
        .len()
        .checked_add(args.max_new_tokens.saturating_sub(1))
        .context("Muse forward count overflow")?;
    let context = MetalContext::new().context("initialize Metal for Muse Lens run")?;
    let mut loaded = MuseGlimmerLoadedModel::load(&context, &gguf, forward_count)
        .context("load Muse Lens runner model")?;
    let mut lenses = HashMap::new();
    let mut published_lenses = Vec::new();
    for lens in &plan.lenses {
        let (id, loaded_lens) = match lens {
            LensDefinition::NativeSelected { id, artifact: path } => (
                id,
                load_muse_artifact(&resolve(plan_dir, path), &config, profile, &content_id)?,
            ),
            LensDefinition::PublishedFullTransport {
                id,
                artifact: path,
                token_ids,
                allow_unvalidated_transfer,
            } => {
                let source_layers = required_lens_layers(&plan, id, config.layer_count)?;
                let (loaded_lens, binding) = load_published_muse_artifact(
                    id,
                    &resolve(plan_dir, path),
                    token_ids,
                    &source_layers,
                    *allow_unvalidated_transfer,
                    &config,
                    &context,
                    &loaded,
                )?;
                published_lenses.push(binding);
                (id, loaded_lens)
            }
            LensDefinition::WorkspaceTemplate { .. } => {
                bail!("Muse plans do not support workspace_template lenses")
            }
        };
        ensure!(
            lenses.insert(id.clone(), loaded_lens).is_none(),
            "duplicate Muse lens id"
        );
    }
    let execution_binding = (!published_lenses.is_empty()).then(|| RunExecutionBinding {
        deployed_model_content_blake3: content_id.clone(),
        content_identity_outcome: format!("{:?}", content.outcome),
        weight_bytes_hashed: content.bytes_hashed,
        published_lenses,
    });
    let execution = prepare_execution_plan(plan, lenses, &config, &context)?;
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
    let mut operation_applications = Vec::new();
    let mut live_readouts = Vec::new();
    let mut logits = Vec::new();
    for (index, &token) in prompt_ids.iter().enumerate() {
        logits = forward_event(
            &execution,
            &mut runner,
            token as u32,
            Event::Prefill(index),
            &mut operation_applications,
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
            &execution,
            &mut runner,
            token as u32,
            Event::Decode(index),
            &mut operation_applications,
            &mut live_readouts,
        )?;
    }
    let plan = execution.plan.clone();
    emit_run_output(
        args,
        "muse_glimmer",
        plan_path,
        plan,
        RunResult {
            prompt_token_ids: prompt_ids,
            generated_token_ids: generated.clone(),
            decoded_text: tokenizer.decode(&generated),
            stop_reason,
            operation_applications,
            live_readouts,
            native_hyper_captures: Vec::new(),
        },
        execution_binding,
        output_path,
    )
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
        candidate_universe: "lens_artifact_selected_token_rows",
        target_layer: manifest.transport.target_layer,
        source_layers: manifest.transport.source_layers,
        token_ids: manifest.selected.token_ids,
        hidden_size: config.hidden_size as usize,
        values,
    })
}

fn required_lens_layers(plan: &LensPlan, lens_id: &str, layer_count: u32) -> Result<Vec<u32>> {
    let direction_lenses = plan
        .directions
        .iter()
        .filter_map(|definition| match definition {
            DirectionDefinition::LensRow(direction) => {
                Some((direction.id.as_str(), direction.lens.as_str()))
            }
            DirectionDefinition::NativeHyper(_) => None,
        })
        .collect::<HashMap<_, _>>();
    let mut layers = BTreeSet::new();
    for readout in &plan.readouts {
        if readout.lens == lens_id {
            layers.extend(selector_values(&readout.scope.layers, layer_count)?);
        }
    }
    for operation in &plan.operations {
        if operation.action.direction_ids().any(|direction| {
            direction_lenses
                .get(direction)
                .is_some_and(|candidate| *candidate == lens_id)
        }) {
            layers.extend(selector_values(&operation.scope.layers, layer_count)?);
        }
    }
    Ok(layers.into_iter().collect())
}

fn load_published_muse_artifact(
    lens_id: &str,
    artifact_path: &Path,
    token_ids: &[u32],
    source_layers: &[u32],
    allow_unvalidated_transfer: bool,
    config: &MuseGlimmerConfig,
    context: &MetalContext,
    loaded: &MuseGlimmerLoadedModel,
) -> Result<(LoadedMuseLens, RunPublishedLensBinding)> {
    ensure!(
        allow_unvalidated_transfer,
        "published Muse full transport requires allow_unvalidated_transfer=true"
    );
    let manifest_path = if artifact_path.is_dir() {
        artifact_path.join(published::MANIFEST_NAME)
    } else {
        artifact_path.to_path_buf()
    };
    let manifest: published::Manifest = super::read_json_file(&manifest_path)?;
    published::validate_manifest(&manifest).with_context(|| {
        format!(
            "validate Muse published artifact {}",
            manifest_path.display()
        )
    })?;
    let manifest_canonical_json_blake3 = super::digest_json(&manifest)?;
    ensure!(
        manifest.model.geometry == artifact::geometry(config)
            && manifest.model.architecture == qwen_llm::muse_glimmer::ARCHITECTURE_NAME,
        "Muse published transport geometry differs from the running model"
    );
    ensure!(
        !source_layers.is_empty()
            && source_layers.windows(2).all(|pair| pair[0] < pair[1])
            && source_layers.iter().all(|layer| manifest
                .transport
                .source_layers
                .binary_search(layer)
                .is_ok()),
        "published Muse source layers must be sorted unique artifact layers"
    );
    let mut unique_tokens = BTreeSet::new();
    ensure!(
        !token_ids.is_empty()
            && token_ids.len() <= 32
            && token_ids
                .iter()
                .all(|token| *token < config.vocab_size && unique_tokens.insert(*token)),
        "published Muse token IDs must be 1..=32 unique model-vocabulary IDs"
    );

    let covectors = loaded
        .selected_token_lens_covectors(context, token_ids)
        .context("derive Muse deployed-model selected-token covectors")?;
    ensure!(
        covectors.token_ids() == token_ids
            && covectors.hidden_size() == config.hidden_size as usize,
        "Muse selected-token covector metadata is inconsistent"
    );
    let projected_count = source_layers
        .len()
        .checked_mul(token_ids.len())
        .and_then(|count| count.checked_mul(config.hidden_size as usize))
        .context("published Muse projected direction count overflow")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(projected_count)
        .context("allocate published Muse projected directions")?;
    let directory = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    for &layer in source_layers {
        let descriptor = manifest
            .payload
            .matrices
            .iter()
            .find(|matrix| matrix.source_layer == layer)
            .context("Muse published transport omitted a requested source matrix")?;
        let matrix = super::muse_full_lens::read_matrix(
            directory,
            &manifest.payload.path,
            manifest.payload.byte_length,
            descriptor,
        )?;
        let projected = loaded
            .project_f16_transport_lens_covectors(context, &matrix, &covectors)
            .with_context(|| format!("project Muse published source layer {layer}"))?;
        ensure!(
            projected.len() == token_ids.len() * config.hidden_size as usize,
            "Muse published projection returned an invalid shape"
        );
        values.extend(projected);
    }
    ensure!(
        values.len() == projected_count && values.iter().all(|value| value.is_finite()),
        "Muse published projection returned invalid values"
    );
    let selected_matrices = source_layers
        .iter()
        .map(|&layer| {
            let descriptor = manifest
                .payload
                .matrices
                .iter()
                .find(|matrix| matrix.source_layer == layer)
                .expect("selected matrix was validated before projection");
            RunPublishedMatrixBinding {
                source_layer: layer,
                blake3: descriptor.blake3.clone(),
            }
        })
        .collect();
    let canonical_manifest_path = std::fs::canonicalize(&manifest_path).with_context(|| {
        format!(
            "resolve Muse published manifest {}",
            manifest_path.display()
        )
    })?;
    let binding = RunPublishedLensBinding {
        lens_id: lens_id.into(),
        manifest: canonical_manifest_path,
        manifest_canonical_json_blake3,
        profile: manifest.profile.clone(),
        method: manifest.transport.method.clone(),
        target_layer: manifest.transport.target_layer,
        fitted_checkpoint: manifest.model.fitted_checkpoint.clone(),
        fitted_checkpoint_revision: manifest.model.fitted_checkpoint_revision.clone(),
        source_repository: manifest.source.repository.clone(),
        source_revision: manifest.source.revision.clone(),
        source_sha256: manifest.source.sha256.clone(),
        payload_blake3: manifest.payload.blake3.clone(),
        claims_basis: manifest.fit.claims_basis.clone(),
        transfer_validation_status: manifest.transfer.validation_status.clone(),
        selected_token_ids: token_ids.to_vec(),
        selected_matrices,
    };
    Ok((
        LoadedMuseLens {
            method: format!(
                "published_{}_selected_token_numerator",
                manifest.transport.method
            ),
            candidate_universe: "plan_selected_published_token_rows",
            target_layer: manifest.transport.target_layer,
            source_layers: source_layers.to_vec(),
            token_ids: token_ids.to_vec(),
            hidden_size: config.hidden_size as usize,
            values,
        },
        binding,
    ))
}

fn prepare_execution_plan(
    plan: LensPlan,
    lenses: HashMap<String, LoadedMuseLens>,
    config: &MuseGlimmerConfig,
    context: &MetalContext,
) -> Result<MuseExecutionPlan> {
    let mut readout_layers = BTreeSet::new();
    for readout in &plan.readouts {
        let selected =
            validate_readout_source_subset(readout, &lenses[&readout.lens], config.layer_count)?;
        readout_layers.extend(selected);
    }
    let capture_layers = readout_layers.into_iter().collect::<Vec<_>>();
    let layer_slots = capture_layers
        .iter()
        .enumerate()
        .map(|(slot, &layer)| (layer, slot))
        .collect::<HashMap<_, _>>();

    let mut direction_layers: BTreeMap<&str, BTreeSet<u32>> = BTreeMap::new();
    for operation in &plan.operations {
        let layers = selector_values(&operation.scope.layers, config.layer_count)?;
        for direction in operation.action.direction_ids() {
            direction_layers
                .entry(direction)
                .or_default()
                .extend(layers.iter().copied());
        }
    }
    let definitions = plan
        .directions
        .iter()
        .filter_map(|definition| match definition {
            DirectionDefinition::LensRow(direction) => Some((direction.id.as_str(), direction)),
            DirectionDefinition::NativeHyper(_) => None,
        })
        .collect::<HashMap<_, _>>();
    let mut directions = HashMap::new();
    for (id, layers) in direction_layers {
        let definition = definitions[id];
        let token_id = match &definition.row {
            DirectionRow::TokenId { token_id } => *token_id,
            _ => bail!("Muse direction {id} requires row.kind=token_id"),
        };
        let token_id = u32::try_from(token_id)
            .with_context(|| format!("Muse direction {id} token ID must be nonnegative"))?;
        let lens = &lenses[&definition.lens];
        let mut rows = BTreeMap::new();
        for layer in layers {
            let raw = muse_lens_row(lens, layer, token_id, id)?;
            let normalized =
                super::lens_run::normalize_direction(raw, definition.normalization, id)?;
            let tensor = MetalTensor::from_bytes(
                context,
                bytemuck::cast_slice(&normalized),
                vec![config.hidden_size as u64],
                GgmlType::F32,
            )?;
            ensure!(rows.insert(layer, tensor).is_none());
        }
        ensure!(
            directions
                .insert(id.to_string(), PreparedMuseDirection { rows })
                .is_none()
        );
    }
    let coordinate_swaps =
        prepare_coordinate_swaps(&plan.operations, &directions, config, context)?;
    Ok(MuseExecutionPlan {
        plan,
        lenses,
        directions,
        coordinate_swaps,
        capture_layers,
        layer_slots,
        layer_count: config.layer_count,
    })
}

fn prepare_coordinate_swaps(
    operations: &[super::lens_run::OperationDefinition],
    directions: &HashMap<String, PreparedMuseDirection>,
    config: &MuseGlimmerConfig,
    context: &MetalContext,
) -> Result<HashMap<String, PreparedMuseDirection>> {
    let hidden_size = config.hidden_size as usize;
    let mut swaps = HashMap::new();
    let mut reflection_cache = HashMap::<(String, String, u32), MetalTensor>::new();
    for operation in operations {
        let Action::CoordinateSwap { source, target, .. } = &operation.action else {
            continue;
        };
        let pair = if source <= target {
            (source.clone(), target.clone())
        } else {
            (target.clone(), source.clone())
        };
        let layers = selector_values(&operation.scope.layers, config.layer_count)?;
        let source = directions
            .get(source)
            .with_context(|| format!("coordinate swap {} has no source direction", operation.id))?;
        let target = directions
            .get(target)
            .with_context(|| format!("coordinate swap {} has no target direction", operation.id))?;
        let mut rows = BTreeMap::new();
        for layer in layers {
            let cache_key = (pair.0.clone(), pair.1.clone(), layer);
            if let Some(reflection) = reflection_cache.get(&cache_key) {
                ensure!(rows.insert(layer, reflection.clone()).is_none());
                continue;
            }
            let source = source.rows.get(&layer).with_context(|| {
                format!(
                    "coordinate swap {} source is unavailable at layer {layer}",
                    operation.id
                )
            })?;
            let target = target.rows.get(&layer).with_context(|| {
                format!(
                    "coordinate swap {} target is unavailable at layer {layer}",
                    operation.id
                )
            })?;
            let reflection = super::lens_run::coordinate_swap_reflection_direction(
                &super::lens_run::read_f32_tensor(source, hidden_size),
                &super::lens_run::read_f32_tensor(target, hidden_size),
                &format!("{} at layer {layer}", operation.id),
            )?;
            let reflection = MetalTensor::from_bytes(
                context,
                bytemuck::cast_slice(&reflection),
                vec![config.hidden_size as u64],
                GgmlType::F32,
            )?;
            ensure!(
                reflection_cache
                    .insert(cache_key, reflection.clone())
                    .is_none()
            );
            ensure!(rows.insert(layer, reflection).is_none());
        }
        ensure!(
            swaps
                .insert(operation.id.clone(), PreparedMuseDirection { rows })
                .is_none(),
            "duplicate coordinate-swap operation {}",
            operation.id
        );
    }
    Ok(swaps)
}

fn muse_lens_row(
    lens: &LoadedMuseLens,
    layer: u32,
    token_id: u32,
    direction_id: &str,
) -> Result<Vec<f32>> {
    let source_slot = lens.source_layers.binary_search(&layer).map_err(|_| {
        anyhow::anyhow!("Muse direction {direction_id} lens has no fitted source layer {layer}")
    })?;
    let token_slot = lens
        .token_ids
        .iter()
        .position(|&candidate| candidate == token_id)
        .with_context(|| {
            format!("Muse direction {direction_id} lens has no selected token {token_id}")
        })?;
    let offset = (source_slot * lens.token_ids.len() + token_slot) * lens.hidden_size;
    Ok(lens.values[offset..offset + lens.hidden_size].to_vec())
}

fn validate_readout_source_subset(
    readout: &super::lens_run::ReadoutDefinition,
    lens: &LoadedMuseLens,
    layer_count: u32,
) -> Result<Vec<u32>> {
    let selected = selector_values(&readout.scope.layers, layer_count)?;
    ensure!(
        !selected.is_empty()
            && selected
                .iter()
                .all(|layer| lens.source_layers.binary_search(layer).is_ok()),
        "Muse readout {} layers must be a nonempty subset of lens {} source layers {:?}",
        readout.id,
        readout.lens,
        lens.source_layers
    );
    Ok(selected)
}

fn validate_plan(plan: &LensPlan, args: &LensRunArgs) -> Result<()> {
    ensure!(
        args.messages.is_none(),
        "Muse Lens run does not support --messages"
    );
    ensure!(
        !plan.operations.is_empty() || !plan.readouts.is_empty(),
        "Muse Lens plans require at least one operation or readout"
    );
    ensure!(
        plan.lenses.iter().all(|lens| matches!(
            lens,
            LensDefinition::NativeSelected { .. } | LensDefinition::PublishedFullTransport { .. }
        )),
        "Muse Lens plans support native_selected and published_full_transport lenses only"
    );
    for direction in &plan.directions {
        ensure!(
            matches!(
                direction,
                DirectionDefinition::LensRow(definition)
                    if matches!(&definition.row, DirectionRow::TokenId { .. })
            ),
            "Muse directions require a selected token row"
        );
    }
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

fn score(lens: &LoadedMuseLens, layer: u32, row: &[f32], top_k: usize) -> Result<Vec<LiveScore>> {
    let source_slot = lens
        .source_layers
        .binary_search(&layer)
        .map_err(|_| anyhow::anyhow!("Muse lens has no fitted source layer {layer}"))?;
    let source_offset = source_slot * lens.token_ids.len() * lens.hidden_size;
    let mut scores = lens
        .token_ids
        .iter()
        .enumerate()
        .map(|(slot, &token)| {
            let offset = source_offset + slot * lens.hidden_size;
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
    Ok(scores
        .into_iter()
        .map(|(row_id, token, value)| LiveScore {
            token_id: Some(token as i32),
            row_id,
            word_id: None,
            label: None,
            score: value as f32,
        })
        .collect())
}

fn forward_event(
    execution: &MuseExecutionPlan,
    runner: &mut qwen_llm::muse_glimmer_runtime::MuseGlimmerTextRunner<'_, '_>,
    token: u32,
    event: Event,
    operation_applications: &mut Vec<OperationApplication>,
    output: &mut Vec<LiveReadout>,
) -> Result<Vec<f32>> {
    let interventions = event_interventions(execution, event)?;
    let borrowed = interventions
        .iter()
        .map(|(_, _, intervention)| *intervention)
        .collect::<Vec<_>>();
    let (logits, captured) = if execution.capture_layers.is_empty() {
        let logits = if borrowed.is_empty() {
            runner.forward_token(token)
        } else {
            runner.forward_token_with_post_block_interventions(token, &borrowed)
        }
        .with_context(|| format!("forward Muse {} event {}", event.label(), event.index()))?;
        (Some(logits), None)
    } else {
        let captured = runner
            .forward_token_capture_post_blocks_with_interventions(
                token,
                &execution.capture_layers,
                &borrowed,
            )
            .with_context(|| format!("forward Muse {} event {}", event.label(), event.index()))?;
        (None, Some(captured))
    };
    operation_applications.extend(interventions.iter().map(|(id, layer, _)| {
        OperationApplication {
            id: id.clone(),
            layer: *layer,
            phase: event.label(),
            index: event.index(),
        }
    }));
    if let Some(captured) = captured {
        append_live_readouts(execution, &captured, event, output)?;
        Ok(captured.logits)
    } else {
        Ok(logits.expect("no-capture Muse forward produced logits"))
    }
}

fn event_interventions<'a>(
    execution: &'a MuseExecutionPlan,
    event: Event,
) -> Result<Vec<(String, u32, PostBlockIntervention<'a>)>> {
    let mut interventions = Vec::new();
    for layer in 0..execution.layer_count {
        for operation in &execution.plan.operations {
            if matches(&operation.scope, event, layer) {
                interventions.push((
                    operation.id.clone(),
                    layer,
                    action_to_intervention(
                        &operation.id,
                        &operation.action,
                        layer,
                        &execution.directions,
                        &execution.coordinate_swaps,
                    )?,
                ));
            }
        }
    }
    Ok(interventions)
}

fn action_to_intervention<'a>(
    operation_id: &str,
    action: &Action,
    layer: u32,
    directions: &'a HashMap<String, PreparedMuseDirection>,
    coordinate_swaps: &'a HashMap<String, PreparedMuseDirection>,
) -> Result<PostBlockIntervention<'a>> {
    let direction = |id: &str| -> Result<&'a MetalTensor> {
        directions
            .get(id)
            .and_then(|prepared| prepared.rows.get(&layer))
            .with_context(|| format!("Muse direction {id} has no uploaded row for layer {layer}"))
    };
    Ok(match action {
        Action::FixedAdd {
            direction: id,
            coefficient,
        } => PostBlockIntervention::Fixed {
            layer,
            direction: direction(id)?,
            coefficient: *coefficient,
        },
        Action::ResidualL2Fraction {
            direction: id,
            coefficient,
        } => PostBlockIntervention::ResidualL2Relative {
            layer,
            direction: direction(id)?,
            coefficient: *coefficient,
        },
        Action::ProjectionAblate {
            direction: id,
            coefficient,
        } => PostBlockIntervention::Projection {
            layer,
            direction: direction(id)?,
            coefficient: *coefficient,
        },
        Action::SourceToTarget {
            source,
            target,
            coefficient,
        } => PostBlockIntervention::SourceToTarget {
            layer,
            source: direction(source)?,
            target: direction(target)?,
            coefficient: *coefficient,
        },
        Action::CoordinateSwap { coefficient, .. } => PostBlockIntervention::Projection {
            layer,
            direction: coordinate_swaps
                .get(operation_id)
                .and_then(|prepared| prepared.rows.get(&layer))
                .with_context(|| {
                    format!(
                        "coordinate swap {operation_id} has no reflection direction at layer {layer}"
                    )
                })?,
            coefficient: 2.0 * *coefficient,
        },
    })
}

fn append_live_readouts(
    execution: &MuseExecutionPlan,
    captured: &qwen_llm::muse_glimmer_text_session::MuseGlimmerPostBlockForward,
    event: Event,
    output: &mut Vec<LiveReadout>,
) -> Result<()> {
    for readout in &execution.plan.readouts {
        let lens = &execution.lenses[&readout.lens];
        for &layer in &execution.capture_layers {
            if !matches(&readout.scope, event, layer) {
                continue;
            }
            let row = captured
                .layer_values(execution.layer_slots[&layer])
                .context("Muse captured source layer is missing")?;
            output.push(LiveReadout {
                id: readout.id.clone(),
                lens: readout.lens.clone(),
                method: lens.method.clone(),
                score_kind: "selected_row_projection_numerator",
                candidate_universe: lens.candidate_universe,
                source_layer: layer,
                target_layer: Some(lens.target_layer),
                phase: event.label(),
                index: event.index(),
                scores: score(lens, layer, row, readout.top_k)?,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scoring_is_f64_dot_and_stable_token_tie_break() {
        let lens = LoadedMuseLens {
            method: "J".into(),
            candidate_universe: "lens_artifact_selected_token_rows",
            target_layer: 51,
            source_layers: vec![49, 50],
            token_ids: vec![9, 3],
            hidden_size: 2,
            values: vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0],
        };
        let scores = score(&lens, 50, &[3.0, 4.0], 2).unwrap();
        assert_eq!(
            scores.iter().map(|s| s.token_id).collect::<Vec<_>>(),
            vec![Some(3), Some(9)]
        );
        assert_eq!(scores[0].score, 11.0);
    }

    #[test]
    fn source_selector_must_be_a_nonempty_artifact_subset() {
        let lens = LoadedMuseLens {
            method: "J".into(),
            candidate_universe: "lens_artifact_selected_token_rows",
            target_layer: 51,
            source_layers: vec![49, 50],
            token_ids: vec![1],
            hidden_size: 1,
            values: vec![1.0, 2.0],
        };
        let plan: LensPlan = serde_json::from_value(json!({"version":1,"lenses":[{"kind":"native_selected","id":"x","artifact":"a"}],"directions":[],"operations":[],"readouts":[{"id":"r","lens":"x","scope":{"layers":{"kind":"values","values":[49,50]},"prefill":{"kind":"values","values":[0]}},"top_k":1}]})).unwrap();
        assert_eq!(
            validate_readout_source_subset(&plan.readouts[0], &lens, 52).unwrap(),
            vec![49, 50]
        );
        let bad: LensPlan = serde_json::from_value(json!({"version":1,"lenses":[{"kind":"native_selected","id":"x","artifact":"a"}],"directions":[],"operations":[],"readouts":[{"id":"r","lens":"x","scope":{"layers":{"kind":"values","values":[48,50]},"prefill":{"kind":"values","values":[0]}},"top_k":1}]})).unwrap();
        assert!(validate_readout_source_subset(&bad.readouts[0], &lens, 52).is_err());
    }

    #[test]
    fn event_layer_matching_scores_each_selected_source_offset() {
        let lens = LoadedMuseLens {
            method: "R".into(),
            candidate_universe: "lens_artifact_selected_token_rows",
            target_layer: 51,
            source_layers: vec![49, 50],
            token_ids: vec![7],
            hidden_size: 2,
            values: vec![1.0, 0.0, 0.0, 2.0],
        };
        let plan: LensPlan = serde_json::from_value(json!({"version":1,"lenses":[{"kind":"native_selected","id":"x","artifact":"a"}],"directions":[],"operations":[],"readouts":[{"id":"r","lens":"x","scope":{"layers":{"kind":"values","values":[49,50]},"prefill":{"kind":"values","values":[1]}},"top_k":1}]})).unwrap();
        let scope = &plan.readouts[0].scope;
        assert!(matches(scope, Event::Prefill(1), 49));
        assert!(matches(scope, Event::Prefill(1), 50));
        assert!(!matches(scope, Event::Prefill(0), 49));
        assert_eq!(score(&lens, 49, &[3.0, 4.0], 1).unwrap()[0].score, 3.0);
        assert_eq!(score(&lens, 50, &[3.0, 4.0], 1).unwrap()[0].score, 8.0);
    }

    #[test]
    fn muse_plan_accepts_selected_token_operations_and_rejects_other_direction_kinds() {
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
            output: None,
            format: Some(super::super::lens_run::RunStdoutFormat::Summary),
        };
        let plan: LensPlan = serde_json::from_value(json!({
            "version": 1,
            "lenses": [{"kind":"native_selected","id":"x","artifact":"a"}],
            "directions": [
                {"id":"a","lens":"x","row":{"kind":"token_id","token_id":7},"normalization":"unit_l2"},
                {"id":"u","lens":"x","row":{"kind":"token_id","token_id":8},"normalization":"unit_l2"}
            ],
            "operations": [
                {"id":"fixed","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"all"}},"action":{"kind":"fixed_add","direction":"a","coefficient":1.0}},
                {"id":"relative","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"all"}},"action":{"kind":"residual_l2_fraction","direction":"u","coefficient":0.1}},
                {"id":"ablate","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"all"}},"action":{"kind":"projection_ablate","direction":"u","coefficient":1.0}},
                {"id":"displace","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"all"}},"action":{"kind":"source_to_target","source":"u","target":"u","coefficient":0.5}},
                {"id":"swap","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"all"}},"action":{"kind":"coordinate_swap","source":"u","target":"a","coefficient":1.0}}
            ],
            "readouts": []
        }))
        .unwrap();
        validate_plan(&plan, &args).unwrap();

        let published: LensPlan = serde_json::from_value(json!({
            "version": 1,
            "lenses": [{
                "kind":"published_full_transport",
                "id":"p",
                "artifact":"published",
                "token_ids":[7,8],
                "allow_unvalidated_transfer":true
            }],
            "directions": [
                {"id":"p7","lens":"p","row":{"kind":"token_id","token_id":7},"normalization":"unit_l2"}
            ],
            "operations": [
                {"id":"steer","scope":{"layers":{"kind":"values","values":[25]},"prefill":{"kind":"all"}},"action":{"kind":"fixed_add","direction":"p7","coefficient":1.0}}
            ],
            "readouts": [
                {"id":"live","lens":"p","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"all"}},"top_k":2}
            ]
        }))
        .unwrap();
        validate_plan(&published, &args).unwrap();
        assert_eq!(required_lens_layers(&published, "p", 52).unwrap(), [25, 50]);

        let template: LensPlan = serde_json::from_value(json!({"version":1,"lenses":[{"kind":"workspace_template","id":"x","weights":"w","labels":"l"}],"directions":[],"operations":[],"readouts":[{"id":"r","lens":"x","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"all"}},"top_k":1}]})).unwrap();
        assert!(validate_plan(&template, &args).is_err());
        let label: LensPlan = serde_json::from_value(json!({"version":1,"lenses":[{"kind":"native_selected","id":"x","artifact":"a"}],"directions":[{"id":"d","lens":"x","row":{"kind":"label","label":"no"},"normalization":"unit_l2"}],"operations":[{"id":"o","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"all"}},"action":{"kind":"fixed_add","direction":"d","coefficient":1.0}}],"readouts":[]})).unwrap();
        assert!(validate_plan(&label, &args).is_err());
    }

    #[test]
    fn operation_only_execution_prepares_all_actions_in_file_order() {
        let config = MuseGlimmerConfig::unsloth_release_reference();
        let hidden = config.hidden_size as usize;
        let mut values = vec![0.0_f32; 2 * 2 * hidden];
        values[2 * hidden] = 2.0;
        values[3 * hidden + 1] = 3.0;
        let lenses = HashMap::from([(
            "x".to_string(),
            LoadedMuseLens {
                method: "J".into(),
                candidate_universe: "lens_artifact_selected_token_rows",
                target_layer: 51,
                source_layers: vec![49, 50],
                token_ids: vec![7, 8],
                hidden_size: hidden,
                values,
            },
        )]);
        let plan: LensPlan = serde_json::from_value(json!({
            "version": 1,
            "lenses": [{"kind":"native_selected","id":"x","artifact":"a"}],
            "directions": [
                {"id":"a","lens":"x","row":{"kind":"token_id","token_id":7},"normalization":"unit_l2"},
                {"id":"u","lens":"x","row":{"kind":"token_id","token_id":8},"normalization":"unit_l2"}
            ],
            "operations": [
                {"id":"fixed","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"values","values":[0]}},"action":{"kind":"fixed_add","direction":"a","coefficient":1.0}},
                {"id":"relative","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"values","values":[0]}},"action":{"kind":"residual_l2_fraction","direction":"u","coefficient":0.1}},
                {"id":"ablate","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"values","values":[0]}},"action":{"kind":"projection_ablate","direction":"u","coefficient":1.0}},
                {"id":"displace","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"values","values":[0]}},"action":{"kind":"source_to_target","source":"u","target":"a","coefficient":0.5}},
                {"id":"swap","scope":{"layers":{"kind":"values","values":[50]},"prefill":{"kind":"values","values":[0]}},"action":{"kind":"coordinate_swap","source":"u","target":"a","coefficient":1.0}}
            ],
            "readouts": []
        }))
        .unwrap();
        let context = MetalContext::new().unwrap();
        let execution = prepare_execution_plan(plan, lenses, &config, &context).unwrap();
        assert!(execution.capture_layers.is_empty());
        let interventions = event_interventions(&execution, Event::Prefill(0)).unwrap();
        assert_eq!(
            interventions
                .iter()
                .map(|(id, layer, _)| (id.as_str(), *layer))
                .collect::<Vec<_>>(),
            vec![
                ("fixed", 50),
                ("relative", 50),
                ("ablate", 50),
                ("displace", 50),
                ("swap", 50)
            ]
        );
        assert!(matches!(
            interventions[0].2,
            PostBlockIntervention::Fixed { .. }
        ));
        assert!(matches!(
            interventions[1].2,
            PostBlockIntervention::ResidualL2Relative { .. }
        ));
        assert!(matches!(
            interventions[2].2,
            PostBlockIntervention::Projection { .. }
        ));
        assert!(matches!(
            interventions[3].2,
            PostBlockIntervention::SourceToTarget { .. }
        ));
        assert!(matches!(
            interventions[4].2,
            PostBlockIntervention::Projection {
                coefficient: 2.0,
                ..
            }
        ));
        assert!(
            event_interventions(&execution, Event::Prefill(1))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn decode_schedule_excludes_unforwarded_last_sample() {
        let selector = Selector::Values { values: vec![1] };
        assert!(selector_values(&selector, 2).is_ok());
        assert!(selector_values(&selector, 1).is_err());
    }

    #[test]
    #[ignore = "requires MUSE_GLIMMER_Q8_GGUF and loads the real 30B Q8 model twice"]
    fn real_q8_multi_source_scores_match_direct_f64_dots() -> Result<()> {
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
        let content = checkpoint_content_identity_without_weight_hashing(
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
        let captures = runner.capture_fresh_lens_prompt_blocks(&prompt, &[50, 51])?;
        let source_layers = [49, 50];
        let mut artifacts = HashMap::new();
        for rule in [
            qwen_llm::muse_glimmer_lens::MuseGlimmerLensRule::J,
            qwen_llm::muse_glimmer_lens::MuseGlimmerLensRule::R,
        ] {
            let fit = runner.fit_selected_tokens_to_sources(
                &captures,
                51,
                &source_layers,
                &covectors,
                0,
                rule,
            )?;
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
            artifacts.insert(
                rule.as_str().to_string(),
                load_muse_artifact(&load_path, &config, profile, &content_id)?,
            );
        }
        drop(runner);
        drop(model);
        drop(context);

        let context = MetalContext::new()?;
        let mut model = MuseGlimmerLoadedModel::load(&context, &gguf, 1)?;
        let mut runner = model.create_runner(&context)?;
        let live = runner.forward_token_capture_post_blocks(config.bos_token_id, &source_layers)?;
        let plan: LensPlan = serde_json::from_value(json!({
            "version": 1,
            "lenses": [
                {"kind":"native_selected","id":"J","artifact":"j"},
                {"kind":"native_selected","id":"R","artifact":"r"}
            ],
            "directions": [],
            "operations": [],
            "readouts": [
                {"id":"j","lens":"J","scope":{"layers":{"kind":"values","values":[49,50]},"prefill":{"kind":"values","values":[0]}},"top_k":1},
                {"id":"r","lens":"R","scope":{"layers":{"kind":"values","values":[49,50]},"prefill":{"kind":"values","values":[0]}},"top_k":1}
            ]
        }))?;
        let execution = prepare_execution_plan(plan, artifacts, &config, &context)?;
        let mut emitted = Vec::new();
        append_live_readouts(&execution, &live, Event::Prefill(0), &mut emitted)?;
        ensure!(
            emitted.len() == 4,
            "real proof did not emit four source scores"
        );
        for readout in emitted {
            let lens = &execution.lenses[&readout.lens];
            let source_slot = lens
                .source_layers
                .binary_search(&readout.source_layer)
                .unwrap();
            let residual = live
                .layer_values(execution.layer_slots[&readout.source_layer])
                .with_context(|| format!("missing layer-{} proof capture", readout.source_layer))?;
            let offset = source_slot * lens.token_ids.len() * lens.hidden_size;
            let row = &lens.values[offset..offset + lens.hidden_size];
            let manual = residual
                .iter()
                .zip(row)
                .map(|(&left, &right)| f64::from(left) * f64::from(right))
                .sum::<f64>();
            let score = readout.scores[0].score;
            eprintln!(
                "muse_method={} source_layer={} emitted={score:.9e} manual_f64={manual:.17e}",
                lens.method, readout.source_layer
            );
            ensure!(
                score.to_bits() == (manual as f32).to_bits(),
                "{} layer {} emitted score differs from direct F64 dot",
                lens.method,
                readout.source_layer
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
        fit: &qwen_llm::muse_glimmer_lens_fit::MuseGlimmerMultiSourceSelectedTokenFit,
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
                rule_contract: artifact::RULE_CONTRACT_V2.into(),
                target_layer: 51,
                source_layers: fit.source_layers.clone(),
                coordinate: artifact::COORDINATE.into(),
                estimator: artifact::ESTIMATOR_V2.into(),
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
                shape: [fit.source_layers.len(), 1, config.hidden_size as usize],
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
                    .diagnostics
                    .iter()
                    .map(|diagnostic| diagnostic.post_attention_replay_max_abs_error)
                    .fold(0.0, f32::max),
                f32_vs_production_f16_kv_post_block_max_abs: fit
                    .diagnostics
                    .iter()
                    .map(|diagnostic| diagnostic.post_block_replay_max_abs_error)
                    .fold(0.0, f32::max),
            },
            provenance: artifact::Provenance {
                build_commit: env!("QWEN_BUILD_COMMIT").into(),
                build_dirty: env!("QWEN_BUILD_DIRTY").into(),
                build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
            },
        }
    }
}
