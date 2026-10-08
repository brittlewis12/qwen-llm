//! GLM-5.3-Flash `qwen-lens run`: raw residual directions applied and read at
//! GLM module sites (embedding, mixer output, routed experts output, shared
//! expert output, FFN output), each a single hidden-size vector before
//! GLM's four-stream mHC post. A unit projection with coefficient `a` at a
//! module's output is the activation form of `W' = W - a v v^T W` on that
//! module's output projection.
//!
//! Every prompt and generated token runs through the serial decode path
//! (the Exact lineage), with the token's operations applied and its
//! captures read in that forward. Lenses, live lens readouts, transports,
//! post-block sites (four mixed streams), coordinate swaps, message
//! documents, Open Responses inputs, cohorts and sweeps are refused.

use super::*;
use crate::lens_input::LensMessageMode;
use crate::lens_intervention::OperationSite;
use crate::model_request::prefill::AssistantPrefillChannel;
use qwen_llm::glm5_next::Glm5NextPreparedArtifact;
use qwen_llm::glm5_next_chat::{self as chat, Effort, Message, RenderOptions};
use qwen_llm::glm5_next_metal::{
    Glm5NextCapturePoint, Glm5NextModuleIntervention, Glm5NextSession, Glm5NextSiteCapture,
    Glm5NextWeights, prefetch_retained_with_cancel, preflight_session,
};

const FAMILY: &str = "GLM-5.3-Flash";

/// Everything the plan may use, refused before any model work.
fn validate_glm5_next_plan(plan: &LensPlan) -> Result<()> {
    ensure!(
        plan.lenses.is_empty(),
        "{FAMILY} Lens plans cannot use lenses; use raw_residual_f32le directions"
    );
    ensure!(
        plan.readouts.is_empty(),
        "{FAMILY} Lens plans cannot use live lens readouts; use direction_readouts"
    );
    ensure!(
        !plan.operations.is_empty() || !plan.direction_readouts.is_empty(),
        "{FAMILY} Lens plans need an operation or a direction readout"
    );
    for direction in &plan.directions {
        ensure!(
            direction.raw().is_some(),
            "{FAMILY} direction {} must be raw_residual_f32le",
            direction.id()
        );
    }
    for operation in &plan.operations {
        operation
            .site
            .glm5_next_site()
            .with_context(|| format!("{FAMILY} operation {}", operation.id))?;
        ensure!(
            !matches!(operation.action, Action::CoordinateSwap { .. }),
            "{FAMILY} operation {} cannot use coordinate_swap",
            operation.id
        );
    }
    for readout in &plan.direction_readouts {
        readout
            .site
            .glm5_next_site()
            .with_context(|| format!("{FAMILY} direction readout {}", readout.id))?;
        ensure!(
            readout.point.is_some(),
            "{FAMILY} direction readout {} needs a point (before_operations or after_operations)",
            readout.id
        );
    }
    Ok(())
}

/// Whether `site` exists in block `layer` of this model.
fn site_exists(
    config: &qwen_llm::glm5_next::Glm5NextConfig,
    site: OperationSite,
    layer: u32,
) -> bool {
    let Some(block) = config.blocks.get(layer as usize) else {
        return false;
    };
    match site {
        OperationSite::Embedding => layer == 0,
        OperationSite::MixerOutput | OperationSite::FfnOutput => true,
        OperationSite::RoutedExpertsOutput | OperationSite::SharedExpertOutput => {
            block.ffn == qwen_llm::glm5_next::FfnKind::Moe
        }
        OperationSite::PostBlock => false,
    }
}

/// Every selected (site, block) exists, before any model work.
fn validate_sites(plan: &LensPlan, config: &qwen_llm::glm5_next::Glm5NextConfig) -> Result<()> {
    let blocks = config.executed_block_count();
    let check = |kind: &str, id: &str, site: OperationSite, scope: &Scope| -> Result<()> {
        for layer in scope
            .layers
            .expand(blocks, &format!("{kind} {id} layers"))?
        {
            ensure!(
                site_exists(config, site, layer),
                "{kind} {id}: site {} does not exist in block {layer} (embedding is block 0 only; expert sites need a MoE block, 3 and later)",
                site.as_str()
            );
        }
        Ok(())
    };
    for operation in &plan.operations {
        check("operation", &operation.id, operation.site, &operation.scope)?;
    }
    for readout in &plan.direction_readouts {
        check(
            "direction readout",
            &readout.id,
            readout.site,
            &readout.scope,
        )?;
    }
    Ok(())
}

fn effort(mode: Option<LensMessageMode>) -> Result<Effort> {
    Ok(match mode {
        None | Some(LensMessageMode::Auto | LensMessageMode::Thinking) => Effort::Max,
        Some(LensMessageMode::Low) => Effort::Low,
        Some(LensMessageMode::High) => Effort::High,
        Some(other) => bail!(
            "{FAMILY} --message-mode {other:?} is not supported; use low, high or auto (max); there is no non-thinking mode"
        ),
    })
}

/// The prompt tokens and their rendering record: raw text, literal ids, or a
/// user turn rendered by the pinned GLM template (`--message-mode` effort),
/// optionally followed by an assistant prefill: a `final` prefill closes the
/// pre-opened reasoning with exactly `</think>` and appends its text; a
/// `reasoning` prefill appends its text inside the reasoning.
fn prepare_input(
    args: &LensRunArgs,
    artifact: &Glm5NextPreparedArtifact<'_>,
) -> Result<PreparedLensInput> {
    ensure!(
        args.messages.is_none() && args.open_responses.is_none() && args.requests_jsonl.is_none(),
        "{FAMILY} Lens runs accept --prompt, --token-ids or --user (with --system); message documents, Open Responses and cohorts are not supported"
    );
    let tokenizer = artifact.tokenizer();
    let rendering = |renderer: &str, mode: Option<String>| LensInputRendering {
        renderer: renderer.into(),
        generation_mode: mode,
        spans: Vec::new(),
    };
    if let Some(ids) = &args.token_ids {
        return Ok(PreparedLensInput {
            source: "token_ids",
            add_special_tokens: None,
            token_ids: ids.clone(),
            rendering: rendering("token_ids", None),
        });
    }
    if let Some(prompt) = &args.prompt {
        // The glm4 tokenizer inserts nothing; the flag governs special parsing.
        let add_special = !args.no_special_tokens;
        return Ok(PreparedLensInput {
            source: "prompt",
            add_special_tokens: Some(add_special),
            token_ids: tokenizer
                .encode(prompt, add_special)
                .with_context(|| format!("tokenize {FAMILY} prompt"))?,
            rendering: rendering("raw_prompt", None),
        });
    }
    let user = args
        .user
        .as_deref()
        .context("--prompt, --token-ids or --user is required")?;
    let user = if user == "-" {
        let mut text = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)?;
        text
    } else {
        user.to_string()
    };
    let effort = effort(args.message_mode)?;
    let mut messages = Vec::new();
    if let Some(system) = &args.system {
        messages.push(Message::System(system.clone()));
    }
    messages.push(Message::User(user));
    let mut text = chat::render(&messages, RenderOptions::generate(effort, false))
        .map_err(|e| anyhow::anyhow!(e.to_string()))
        .with_context(|| format!("render {FAMILY} chat input"))?;
    let transition = match &args.assistant_prefill {
        None => "none",
        Some(prefill) => {
            for marker in [
                "<|",
                "|>",
                chat::THINK_OPEN,
                chat::THINK_CLOSE,
                chat::TOOL_CALL_OPEN,
            ] {
                ensure!(
                    !prefill.text.contains(marker),
                    "{FAMILY} assistant prefill text must not contain template markers ({marker:?})"
                );
            }
            match prefill.channel {
                AssistantPrefillChannel::Final => {
                    text.push_str(chat::THINK_CLOSE);
                    text.push_str(&prefill.text);
                    "final"
                }
                AssistantPrefillChannel::Reasoning => {
                    text.push_str(&prefill.text);
                    "reasoning"
                }
            }
        }
    };
    Ok(PreparedLensInput {
        source: "user",
        add_special_tokens: Some(false),
        token_ids: tokenizer
            .encode(&text, false)
            .with_context(|| format!("tokenize rendered {FAMILY} chat input"))?,
        rendering: rendering(
            chat::RENDERER,
            Some(format!("effort={};prefill={transition}", effort.as_str())),
        ),
    })
}

fn readout_point(point: ReadoutPoint) -> Glm5NextCapturePoint {
    match point {
        ReadoutPoint::BeforeOperations => Glm5NextCapturePoint::BeforeOperations,
        ReadoutPoint::AfterOperations => Glm5NextCapturePoint::AfterOperations,
    }
}

struct Glm5NextExecution<'p> {
    schedule: BoundEventSchedule<'p, 'p>,
    directions: HashMap<String, PreparedDirection>,
    raw: HashMap<String, LoadedRawDirection>,
    blocks: u32,
}

/// One token through the serial path with its event's operations and
/// captures; appends applications and direction readouts.
#[allow(clippy::too_many_arguments)]
fn forward_event(
    execution: &Glm5NextExecution<'_>,
    session: &mut Glm5NextSession<'_>,
    ctx: &MetalContext,
    token: u32,
    phase: Phase,
    event: &mut CompiledEvent,
    operation_applications: &mut Vec<OperationApplication>,
    direction_readouts: &mut Vec<LiveDirectionReadout>,
) -> Result<Vec<f32>> {
    crate::shutdown::checkpoint()?;
    let plan = execution.schedule.plan();
    execution.schedule.populate(phase, event)?;
    let no_swaps = HashMap::new();
    let mut module = Vec::new();
    let mut applied = Vec::new();
    for layer in 0..execution.blocks {
        for &index in event.operation_indices() {
            if !execution.schedule.operation_selects_layer(index, layer) {
                continue;
            }
            let operation = &plan.operations[index];
            let op = action_to_intervention(
                &operation.id,
                &operation.action,
                layer,
                &execution.directions,
                &no_swaps,
            )?;
            module.push(Glm5NextModuleIntervention {
                site: operation.site.glm5_next_site()?,
                op,
            });
            applied.push((operation.id.clone(), layer, operation.site));
        }
    }
    let mut wanted: Vec<(usize, u32, Glm5NextSiteCapture)> = Vec::new();
    for &index in event.direction_readout_indices() {
        let readout = &plan.direction_readouts[index];
        let point = readout.point.context("module readouts carry a point")?;
        for layer in 0..execution.blocks {
            if execution
                .schedule
                .direction_readout_selects_layer(index, layer)
            {
                wanted.push((
                    index,
                    layer,
                    Glm5NextSiteCapture {
                        site: readout.site.glm5_next_site()?,
                        block: layer as usize,
                        point: readout_point(point),
                    },
                ));
            }
        }
    }
    let mut captures: Vec<Glm5NextSiteCapture> = wanted.iter().map(|w| w.2).collect();
    captures.sort_by_key(|c| (c.block, c.site as u8, c.point as u8));
    captures.dedup();
    let mut captured: HashMap<Glm5NextSiteCapture, Vec<f32>> = HashMap::new();
    let logits = session
        .forward_with_interventions(ctx, token, &module, &captures, &mut |capture, values| {
            captured.insert(capture, values.to_vec());
        })
        .with_context(|| format!("{FAMILY} {} token {}", phase.label(), phase.index()))?;
    for (id, layer, site) in applied {
        operation_applications.push(OperationApplication {
            id,
            layer,
            phase: phase.label(),
            index: phase.index(),
            site,
        });
    }
    for (index, layer, capture) in wanted {
        let readout = &plan.direction_readouts[index];
        let values = captured
            .get(&capture)
            .with_context(|| format!("direction readout {} was not captured", readout.id))?;
        let row = execution
            .raw
            .get(&readout.direction)
            .and_then(|raw| raw.row(layer))
            .with_context(|| {
                format!(
                    "direction readout {} has no row at block {layer}",
                    readout.id
                )
            })?;
        let (dot, h_norm_l2, v_norm_l2) = direction_readout_scalars(values, row)?;
        direction_readouts.push(LiveDirectionReadout {
            id: readout.id.clone(),
            direction: readout.direction.clone(),
            site: readout.site,
            point: readout.point,
            source_layer: layer,
            phase: phase.label(),
            index: phase.index(),
            dot,
            h_norm_l2,
            v_norm_l2,
        });
    }
    Ok(logits)
}

pub(super) fn run_glm5_next(
    args: &LensRunArgs,
    plan: LensPlan,
    plan_path: &Path,
    plan_dir: &Path,
    gguf: GgufFile,
    output_path: Option<&Path>,
) -> Result<()> {
    validate_glm5_next_plan(&plan)?;
    let artifact = Glm5NextPreparedArtifact::inspect(&gguf)
        .with_context(|| format!("admit {FAMILY} artifact"))?;
    let config = artifact.config().clone();
    let blocks = config.executed_block_count();
    let hidden = config.hidden_size as usize;
    validate_sites(&plan, &config)?;
    let raw = load_raw_directions(&plan, plan_dir, blocks, hidden)?;
    let prepared_input = prepare_input(args, &artifact)?;
    let prompt_token_ids = prepared_input.token_ids.clone();
    ensure!(
        !prompt_token_ids.is_empty(),
        "prompt must encode to at least one token"
    );
    ensure!(
        prompt_token_ids
            .iter()
            .all(|&t| t >= 0 && (t as u32) < config.vocab_size),
        "prompt contains a token outside the {FAMILY} vocabulary"
    );
    let capacity = ensure_request_fits_context(
        prompt_token_ids.len(),
        args.max_new_tokens,
        config.context_length as usize,
    )?;
    let bound_plan = bind_plan_positions(&plan, &prepared_input.rendering, prompt_token_ids.len())?;
    validate_reachable_scopes(
        &bound_plan.resolved,
        prompt_token_ids.len(),
        args.max_new_tokens,
    )?;
    let compiled = CompiledEventSchedule::compile(&bound_plan.resolved, blocks)?;
    let raw_directions = bound_plan
        .resolved
        .directions
        .iter()
        .filter_map(DirectionDefinition::raw)
        .map(|direction| {
            raw.get(&direction.id)
                .map(|loaded| loaded.binding.clone())
                .with_context(|| format!("raw direction {} was not loaded", direction.id))
        })
        .collect::<Result<Vec<_>>>()?;

    crate::shutdown::checkpoint()?;
    let ctx =
        MetalContext::new().with_context(|| format!("initialize Metal for {FAMILY} Lens run"))?;
    preflight_session(&ctx, &gguf, artifact.model(), capacity, 0)
        .with_context(|| format!("admit {FAMILY} serial Lens session"))?;
    prefetch_retained_with_cancel(&ctx, &gguf, 0.98, &|| {
        crate::shutdown::checkpoint().is_err()
    })
    .with_context(|| format!("prefetch {FAMILY} weights"))?;
    let weights = Glm5NextWeights::load(&ctx, &gguf).with_context(|| format!("load {FAMILY}"))?;
    let mut session = Glm5NextSession::new(&ctx, &weights, capacity)
        .with_context(|| format!("create {FAMILY} serial Lens session"))?;

    // Upload each operation direction's rows at the blocks it is used.
    let mut directions: HashMap<String, PreparedDirection> = HashMap::new();
    for operation in &bound_plan.resolved.operations {
        let layers = operation
            .scope
            .layers
            .expand(blocks, &format!("operation {} layers", operation.id))?;
        for id in operation.action.direction_ids() {
            let loaded = raw
                .get(id)
                .with_context(|| format!("operation {} direction {id} is not raw", operation.id))?;
            let rows = &mut directions
                .entry(id.to_string())
                .or_insert_with(|| PreparedDirection {
                    rows: BTreeMap::new(),
                })
                .rows;
            for &layer in &layers {
                if rows.contains_key(&layer) {
                    continue;
                }
                let row = loaded
                    .row(layer)
                    .with_context(|| format!("direction {id} has no row at block {layer}"))?;
                rows.insert(
                    layer,
                    MetalTensor::from_bytes(
                        &ctx,
                        bytemuck::cast_slice(row),
                        vec![hidden as u64],
                        GgmlType::F32,
                    )?,
                );
            }
        }
    }
    let execution = Glm5NextExecution {
        schedule: compiled.bind(&bound_plan.resolved)?,
        directions,
        raw,
        blocks,
    };

    let tokenizer = artifact.tokenizer();
    let stops: HashSet<i32> = chat::CHAT_STOPS.into_iter().collect();
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
    })?;
    let record_logprobs = args.logprobs_top_k > 0 || !args.logprobs_token_ids.is_empty();
    let mut event = execution.schedule.new_event()?;
    let mut operation_applications = Vec::new();
    let mut direction_readouts = Vec::new();
    let mut generation_logprobs = Vec::new();
    let mut logits = Vec::new();
    for (index, &token) in prompt_token_ids.iter().enumerate() {
        logits = forward_event(
            &execution,
            &mut session,
            &ctx,
            token as u32,
            Phase::Prefill(index),
            &mut event,
            &mut operation_applications,
            &mut direction_readouts,
        )?;
    }
    let mut generated_token_ids = Vec::new();
    let mut stop_reason = String::from("max_new_tokens");
    for generated_index in 0..args.max_new_tokens {
        let sampled = sampler.sample(&logits)?.token;
        if record_logprobs {
            generation_logprobs.push(super::generation_logprobs(
                &logits,
                generated_index,
                sampled,
                args.logprobs_top_k,
                &args.logprobs_token_ids,
            )?);
        }
        generated_token_ids.push(sampled);
        if stops.contains(&sampled) {
            stop_reason = String::from("stop_token");
            break;
        }
        if generated_index + 1 == args.max_new_tokens {
            break;
        }
        logits = forward_event(
            &execution,
            &mut session,
            &ctx,
            sampled as u32,
            Phase::Decode(generated_index),
            &mut event,
            &mut operation_applications,
            &mut direction_readouts,
        )?;
    }

    let result = RunResult {
        linear_transports: Vec::new(),
        prompt_token_ids,
        decoded_text: tokenizer.decode(&generated_token_ids),
        generated_token_ids,
        stop_reason,
        operation_applications,
        live_readouts: Vec::new(),
        native_hyper_captures: Vec::new(),
        raw_directions,
        direction_readouts,
        generation_logprobs,
    };
    emit_run_output(
        args,
        "glm5_next",
        plan_path,
        bound_plan,
        &prepared_input,
        None,
        result,
        RunExecution::runtime_serial(
            args.prefill_execution,
            RunSerialReason::Glm5NextSerialInterventions,
        ),
        None,
        output_path,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logprobs_are_full_vocabulary_and_ordered() {
        let logits = [1.0f32, 3.0, 2.0, 3.0];
        let record = super::super::generation_logprobs(&logits, 0, 1, 3, &[0, 2]).unwrap();
        let ids: Vec<u32> = record.top.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, [1, 3, 2], "ties break by lower id");
        let total: f64 = logits.iter().map(|v| (f64::from(*v)).exp()).sum();
        let expect = |v: f32| f64::from(v) - total.ln();
        assert!((record.top[0].1 - expect(3.0)).abs() < 1e-12);
        assert!((record.tracked[0].1 - expect(1.0)).abs() < 1e-12);
        assert!((record.tracked[1].1 - expect(2.0)).abs() < 1e-12);
        assert!(super::super::generation_logprobs(&logits, 0, 1, 2, &[9]).is_err());
        assert!(super::super::generation_logprobs(&[f32::NAN], 0, 0, 1, &[]).is_err());
    }

    #[test]
    fn effort_maps_message_modes_and_refuses_the_rest() {
        assert_eq!(effort(None).unwrap(), Effort::Max);
        assert_eq!(effort(Some(LensMessageMode::Low)).unwrap(), Effort::Low);
        assert_eq!(effort(Some(LensMessageMode::High)).unwrap(), Effort::High);
        assert!(effort(Some(LensMessageMode::NoThinking)).is_err());
        assert!(effort(Some(LensMessageMode::Medium)).is_err());
    }
}
