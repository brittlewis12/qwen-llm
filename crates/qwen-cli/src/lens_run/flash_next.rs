//! Flash-Next (qwen4exp) lens execution on hyper state.

use super::*;

pub(super) struct Qwen4ExpExecutionPlan {
    pub(super) plan: LensPlan,
    pub(super) directions: HashMap<String, PreparedNativeHyperDirection>,
    pub(super) operation_layers: HashMap<String, u32>,
    pub(super) branch_count: usize,
    pub(super) hidden_size: usize,
}

pub(super) fn run_qwen4exp(
    args: &LensRunArgs,
    plan: LensPlan,
    plan_path: &Path,
    plan_dir: &Path,
    gguf: GgufFile,
    output_path: Option<&Path>,
) -> Result<()> {
    let config = Qwen4ExpConfig::from_gguf(&gguf).context("bind Flash-Next model geometry")?;
    ensure!(
        config == Qwen4ExpConfig::flash_next_reference(),
        "qwen-lens run requires the released Flash-Next architecture contract"
    );
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load Flash-Next tokenizer")?;
    ensure!(
        tokenizer.n_vocab() == config.vocab_size,
        "Flash-Next tokenizer vocabulary {} differs from model {}",
        tokenizer.n_vocab(),
        config.vocab_size
    );
    let prepared_input =
        prepare_qwen_model_input(args.input_spec(), ModelFamily::Qwen4Exp, &gguf, &tokenizer)?;
    let prompt_token_ids = &prepared_input.token_ids;
    ensure!(
        !prompt_token_ids.is_empty(),
        "prompt must encode to at least one token"
    );
    ensure!(
        prompt_token_ids
            .iter()
            .all(|&token| token >= 0 && (token as u32) < config.vocab_size),
        "prompt contains a token outside the Flash-Next vocabulary"
    );
    let required_forwards = ensure_request_fits_context(
        prompt_token_ids.len(),
        args.max_new_tokens,
        config.context_length as usize,
    )?;
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, required_forwards)
        .context("derive Flash-Next serial session capacity")?;
    let bound_plan = bind_plan_positions(&plan, &prepared_input.rendering, prompt_token_ids.len())?;
    validate_reachable_scopes(
        &bound_plan.resolved,
        prompt_token_ids.len(),
        args.max_new_tokens,
    )?;
    let execution =
        prepare_qwen4exp_execution_plan(bound_plan.resolved.clone(), plan_dir, &config)?;
    validate_qwen4exp_event_schedule(&execution, prompt_token_ids.len(), args.max_new_tokens)?;

    let stop_tokens = gguf
        .stop_token_ids()
        .context("load Flash-Next stop tokens")?;
    ensure!(
        stop_tokens
            .iter()
            .all(|&token| token >= 0 && (token as u32) < config.vocab_size),
        "Flash-Next stop-token metadata contains an invalid token"
    );
    let stop_tokens = stop_tokens.into_iter().collect::<HashSet<_>>();

    crate::shutdown::checkpoint()?;
    let context = MetalContext::new().context("initialize Metal for Flash-Next Lens run")?;
    let mut loaded = Qwen4ExpLoadedModel::load(&context, &gguf, capacity)
        .context("load Flash-Next serial Lens session")?;
    let mut runner = loaded
        .create_runner(&context)
        .context("bind Flash-Next serial Lens runner")?;
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
    })?;
    let mut operation_applications = Vec::new();
    let mut native_hyper_captures = Vec::new();
    let mut logits = Vec::new();
    for (index, &token) in prompt_token_ids.iter().enumerate() {
        logits = qwen4exp_forward_event(
            &execution,
            &mut runner,
            u32::try_from(token).context("Flash-Next prompt token is negative")?,
            Phase::Prefill(index),
            &mut operation_applications,
            &mut native_hyper_captures,
        )?;
    }

    let mut generated_token_ids = Vec::new();
    let mut stop_reason = String::from("max_new_tokens");
    for generated_index in 0..args.max_new_tokens {
        let sampled = sampler.sample(&logits)?.token;
        generated_token_ids.push(sampled);
        if stop_tokens.contains(&sampled) {
            stop_reason = String::from("stop_token");
            break;
        }
        if generated_index + 1 == args.max_new_tokens {
            break;
        }
        logits = qwen4exp_forward_event(
            &execution,
            &mut runner,
            u32::try_from(sampled).context("Flash-Next sampled a negative token")?,
            Phase::Decode(generated_index),
            &mut operation_applications,
            &mut native_hyper_captures,
        )?;
    }

    let result = RunResult {
        prompt_token_ids: prompt_token_ids.to_vec(),
        decoded_text: tokenizer.decode(&generated_token_ids),
        generated_token_ids,
        stop_reason,
        operation_applications,
        live_readouts: Vec::new(),
        native_hyper_captures,
    };
    emit_run_output(
        args,
        "flash_next",
        plan_path,
        bound_plan,
        &prepared_input,
        result,
        RunExecution::runtime_serial(
            args.prefill_execution,
            RunSerialReason::FlashNextPackedNotImplemented,
        ),
        None,
        output_path,
    )
}

pub(super) fn prepare_qwen4exp_execution_plan(
    plan: LensPlan,
    plan_dir: &Path,
    config: &Qwen4ExpConfig,
) -> Result<Qwen4ExpExecutionPlan> {
    ensure!(
        plan.lenses.is_empty(),
        "Flash-Next Lens plans cannot use ordinary J/R or template lenses"
    );
    ensure!(
        plan.readouts.is_empty(),
        "Flash-Next live readouts require a real hyper-space lens and are not yet supported"
    );
    ensure!(
        !plan.operations.is_empty(),
        "Flash-Next Lens plans must declare at least one fixed-add operation"
    );
    ensure!(
        plan.directions
            .iter()
            .all(|direction| direction.native_hyper().is_some()),
        "Flash-Next Lens plans require native_hyper_f32 directions"
    );
    let branch_count = config.hyper_connection.count as usize;
    let hidden_size = config.hidden_size as usize;
    let hyper_width = branch_count
        .checked_mul(hidden_size)
        .context("Flash-Next hyper width overflow")?;
    let mut directions = HashMap::new();
    for definition in &plan.directions {
        let definition = definition
            .native_hyper()
            .context("Flash-Next direction is not native hyper")?;
        let (path, layer) = definition.source.path_and_layer();
        ensure!(
            layer > 0 && layer < config.layer_count,
            "native hyper direction {} layer {} is outside 1..{}",
            definition.id,
            layer,
            config.layer_count
        );
        let values = load_native_hyper_direction(
            &resolve_plan_path(plan_dir, path),
            hyper_width,
            &definition.id,
        )?;
        ensure!(
            directions
                .insert(
                    definition.id.clone(),
                    PreparedNativeHyperDirection { layer, values }
                )
                .is_none()
        );
    }

    let mut operation_layers = HashMap::new();
    for operation in &plan.operations {
        let (direction_id, coefficient) = match &operation.action {
            Action::FixedAdd {
                direction,
                coefficient,
            } => (direction.as_str(), *coefficient),
            _ => bail!(
                "Flash-Next operation {} supports fixed_add only",
                operation.id
            ),
        };
        let direction = directions.get(direction_id).with_context(|| {
            format!(
                "Flash-Next operation {} has no native hyper direction {}",
                operation.id, direction_id
            )
        })?;
        let layers = operation.scope.layers.expand(
            config.layer_count,
            &format!("operation {} layers", operation.id),
        )?;
        ensure!(
            layers.len() == 1 && layers[0] == direction.layer,
            "Flash-Next operation {} must select only direction {} layer {}",
            operation.id,
            direction_id,
            direction.layer
        );
        ensure!(
            direction
                .values
                .iter()
                .all(|value| (*value * coefficient).is_finite()),
            "Flash-Next operation {} coefficient overflows its direction",
            operation.id
        );
        operation_layers.insert(operation.id.clone(), direction.layer);
    }
    Ok(Qwen4ExpExecutionPlan {
        plan,
        directions,
        operation_layers,
        branch_count,
        hidden_size,
    })
}

pub(super) fn validate_qwen4exp_event_schedule(
    execution: &Qwen4ExpExecutionPlan,
    prompt_len: usize,
    max_new_tokens: usize,
) -> Result<()> {
    let mut capture_count = 0usize;
    for index in 0..prompt_len {
        if qwen4exp_matching_operation(execution, Phase::Prefill(index))?.is_some() {
            capture_count += 1;
        }
    }
    for index in 0..max_new_tokens.saturating_sub(1) {
        if qwen4exp_matching_operation(execution, Phase::Decode(index))?.is_some() {
            capture_count += 1;
        }
    }
    ensure!(
        capture_count <= MAX_NATIVE_HYPER_CAPTURES,
        "Flash-Next plan can emit {capture_count} native hyper captures, maximum is {MAX_NATIVE_HYPER_CAPTURES}"
    );
    Ok(())
}

pub(super) fn qwen4exp_matching_operation<'a>(
    execution: &'a Qwen4ExpExecutionPlan,
    phase: Phase,
) -> Result<Option<(&'a OperationDefinition, u32)>> {
    let mut matched = None;
    for operation in &execution.plan.operations {
        if !operation_enabled(operation) {
            continue;
        }
        let layer = execution.operation_layers[&operation.id];
        if !scope_matches(&operation.scope, phase, layer)? {
            continue;
        }
        ensure!(
            matched.is_none(),
            "Flash-Next operations overlap at {} index {}; one native hyper probe is supported per token event",
            phase.label(),
            phase.index()
        );
        matched = Some((operation, layer));
    }
    Ok(matched)
}

pub(super) fn qwen4exp_forward_event(
    execution: &Qwen4ExpExecutionPlan,
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    token: u32,
    phase: Phase,
    operation_applications: &mut Vec<OperationApplication>,
    native_hyper_captures: &mut Vec<NativeHyperCapture>,
) -> Result<Vec<f32>> {
    crate::shutdown::checkpoint()?;
    let Some((operation, layer)) = qwen4exp_matching_operation(execution, phase)? else {
        return Ok(runner
            .forward_token(token)
            .with_context(|| {
                format!(
                    "forward Flash-Next {} token {}",
                    phase.label(),
                    phase.index()
                )
            })?
            .to_vec());
    };
    let (direction_id, coefficient) = match &operation.action {
        Action::FixedAdd {
            direction,
            coefficient,
        } => (direction, *coefficient),
        _ => unreachable!("Flash-Next plan validation admits fixed_add only"),
    };
    let direction = &execution.directions[direction_id];
    let capture = runner
        .forward_token_with_post_layer_hyper_capture(
            token,
            Qwen4ExpPostLayerHyperRequest {
                layer,
                fixed_add: Some(Qwen4ExpFixedHyperAdd {
                    direction: &direction.values,
                    coefficient,
                }),
            },
        )
        .with_context(|| {
            format!(
                "apply Flash-Next operation {} at {} index {}",
                operation.id,
                phase.label(),
                phase.index()
            )
        })?;
    let logits = runner.logits()?.to_vec();
    operation_applications.push(OperationApplication {
        id: operation.id.clone(),
        layer,
        phase: phase.label(),
        index: phase.index(),
    });
    native_hyper_captures.push(NativeHyperCapture {
        operation_id: operation.id.clone(),
        layer,
        phase: phase.label(),
        index: phase.index(),
        position: capture.position,
        coordinate: "qwen4exp_persistent_post_layer_hyper_state",
        capture_stage: "after_fixed_add",
        shape: [execution.branch_count, execution.hidden_size],
        flattening: "branch_major_hidden_minor",
        direction_normalization: "as_stored",
        coefficient,
        values: capture.values,
    });
    Ok(logits)
}

pub(super) fn validate_flash_artifact_plan(plan: &LensPlan) -> Result<()> {
    ensure!(
        plan.lenses.is_empty()
            && plan.readouts.is_empty()
            && !plan.operations.is_empty()
            && plan
                .directions
                .iter()
                .all(|direction| direction.native_hyper().is_some()),
        "Flash-Next run artifact plan has unsupported lenses, directions, or readouts"
    );
    let direction_layers = plan
        .directions
        .iter()
        .filter_map(|direction| {
            direction.native_hyper().map(|definition| {
                let (_, layer) = definition.source.path_and_layer();
                (definition.id.as_str(), layer)
            })
        })
        .collect::<HashMap<_, _>>();
    for operation in &plan.operations {
        let Action::FixedAdd { direction, .. } = &operation.action else {
            bail!("Flash-Next run artifact operations require fixed_add")
        };
        let layer = direction_layers.get(direction.as_str()).with_context(|| {
            format!("Flash-Next operation {} lacks its direction", operation.id)
        })?;
        let exact_layer = match &operation.scope.layers {
            Selector::Values { values } => values.as_slice() == [*layer],
            Selector::Range { start, end } => start == layer && end == layer,
            Selector::All | Selector::RenderedSpans { .. } => false,
        };
        ensure!(
            exact_layer,
            "Flash-Next operation {} does not select exactly its direction layer {}",
            operation.id,
            layer
        );
    }
    Ok(())
}
