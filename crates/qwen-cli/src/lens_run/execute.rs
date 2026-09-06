//! Event execution: prefill, decode, capture, and interventions.

use super::*;

pub(super) const MAX_NATIVE_HYPER_CAPTURES: usize = 32;

pub(super) const PACKED_PREFILL_MIN_PASSIVE_SPAN_TOKENS: usize = 65;

pub(super) const PACKED_PREFILL_CHUNK_CAP_TOKENS: usize = 1024;

pub(crate) fn required_forward_count(prompt_tokens: usize, max_new_tokens: usize) -> Result<usize> {
    ensure!(
        prompt_tokens > 0,
        "prompt must encode to at least one token"
    );
    ensure!(max_new_tokens > 0, "--max-new-tokens must be positive");
    prompt_tokens
        .checked_add(max_new_tokens - 1)
        .context("request forward count overflow")
}

pub(super) fn ensure_request_fits_context(
    prompt_tokens: usize,
    max_new_tokens: usize,
    model_context_tokens: usize,
) -> Result<usize> {
    let required_forwards = required_forward_count(prompt_tokens, max_new_tokens)?;
    ensure!(
        required_forwards <= model_context_tokens,
        "request requires {required_forwards} token forwards ({prompt_tokens} prompt + {} maximum decode transitions), exceeding model context {model_context_tokens}",
        max_new_tokens - 1,
    );
    Ok(required_forwards)
}

pub(crate) fn ensure_qwen_sequence_admitted(
    loaded: &qwen_llm::runtime::LoadedModel,
    capacity: usize,
) -> Result<()> {
    let admission = loaded
        .qwen_execution_memory_admission(1, capacity, 0, 0)
        .context("price Lens sequence memory")?;
    ensure!(
        admission.admitted,
        "Lens sequence memory admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
        admission.reason.as_str(),
        admission.required_bytes,
        admission.working_set_headroom_bytes,
        admission.signals.process_limit_remaining_bytes,
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PrefillExecution {
    Auto,
    Serial,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RunEffectivePrefill {
    Serial,
    DensePackedPassiveSpans,
}

impl RunEffectivePrefill {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::DensePackedPassiveSpans => "dense_packed_passive_spans",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunPackedPrefillSpan {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

pub(super) fn expected_packed_prefill_spans(
    plan: &LensPlan,
    prompt_len: usize,
) -> Result<Vec<RunPackedPrefillSpan>> {
    let schedule = CompiledEventSchedule::compile(plan, u32::MAX)?;
    Ok(schedule
        .bind(plan)?
        .passive_prefill_spans(prompt_len, PACKED_PREFILL_MIN_PASSIVE_SPAN_TOKENS)?
        .into_iter()
        .map(|span| RunPackedPrefillSpan {
            start: span.start,
            end: span.end,
        })
        .collect())
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RunSampler {
    pub(super) temperature: f32,
    pub(super) top_k: usize,
    pub(super) top_p: f32,
    pub(super) min_p: f32,
    pub(super) seed: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct NativeHyperCapture {
    pub(super) operation_id: String,
    pub(super) layer: u32,
    pub(super) phase: &'static str,
    pub(super) index: usize,
    pub(super) position: usize,
    pub(super) coordinate: &'static str,
    pub(super) capture_stage: &'static str,
    pub(super) shape: [usize; 2],
    pub(super) flattening: &'static str,
    pub(super) direction_normalization: &'static str,
    pub(super) coefficient: f32,
    pub(super) values: Vec<f32>,
}

pub(super) fn run_sampler(args: &LensRunArgs) -> RunSampler {
    RunSampler {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
    }
}

pub(super) struct PreparedOrdinaryPrefill {
    pub(super) execution: RunExecution,
    pub(super) scratch: Option<PackedPrefillScratch>,
}

impl PreparedOrdinaryPrefill {
    pub(super) fn serial(
        requested: PrefillExecution,
        schedule_basis: RunExecutionScheduleBasis,
        reason: RunSerialReason,
    ) -> Self {
        Self {
            execution: RunExecution::serial(requested, schedule_basis, reason),
            scratch: None,
        }
    }
}

pub(super) fn prepare_ordinary_prefill(
    loaded: &qwen_llm::runtime::LoadedModel,
    schedule: &CompiledEventSchedule,
    schedule_plan: &LensPlan,
    requested: PrefillExecution,
    schedule_basis: RunExecutionScheduleBasis,
    prompt_len: usize,
    max_new_tokens: usize,
) -> Result<PreparedOrdinaryPrefill> {
    if requested == PrefillExecution::Serial {
        return Ok(PreparedOrdinaryPrefill::serial(
            requested,
            schedule_basis,
            RunSerialReason::RequestedSerial,
        ));
    }
    if loaded.arch().kind == ArchKind::Moe {
        return Ok(PreparedOrdinaryPrefill::serial(
            requested,
            schedule_basis,
            RunSerialReason::MoePackedNotQualified,
        ));
    }

    let bound_schedule = schedule.bind(schedule_plan)?;
    let packed_spans = bound_schedule
        .passive_prefill_spans(prompt_len, PACKED_PREFILL_MIN_PASSIVE_SPAN_TOKENS)?
        .into_iter()
        .map(|span| RunPackedPrefillSpan {
            start: span.start,
            end: span.end,
        })
        .collect::<Vec<_>>();
    let Some(longest_span) = packed_spans.iter().map(|span| span.end - span.start).max() else {
        return Ok(PreparedOrdinaryPrefill::serial(
            requested,
            schedule_basis,
            RunSerialReason::NoEligiblePassiveSpan,
        ));
    };
    let block_tokens = u32::try_from(longest_span.min(PACKED_PREFILL_CHUNK_CAP_TOKENS))
        .context("packed Lens prefill block size exceeds u32")?;
    let scratch_plan = loaded
        .plan_packed_prefill_scratch(block_tokens, prompt_len)
        .context("plan dense packed Lens prefill scratch")?;
    let scratch_priced_upper_bytes = scratch_plan.priced_upper_bytes();
    let capacity = required_forward_count(prompt_len, max_new_tokens)?;
    let admission = loaded
        .qwen_execution_memory_admission(1, capacity, scratch_priced_upper_bytes, 0)
        .context("price dense packed Lens prefill memory")?;
    if !admission.admitted {
        return Ok(PreparedOrdinaryPrefill::serial(
            requested,
            schedule_basis,
            RunSerialReason::DensePackedMemoryAdmissionDenied,
        ));
    }
    let block_tokens = scratch_plan.block_size();
    let matrix_max_position = scratch_plan.matrix_max_pos();
    let scratch = loaded
        .allocate_packed_prefill_scratch(scratch_plan)
        .context("allocate dense packed Lens prefill scratch")?;
    let execution = RunExecution::dense_packed(
        schedule_basis,
        block_tokens,
        matrix_max_position,
        scratch_priced_upper_bytes,
        packed_spans,
    );
    execution.validate("ordinary_qwen", prompt_len)?;
    Ok(PreparedOrdinaryPrefill {
        execution,
        scratch: Some(scratch),
    })
}

#[derive(Clone, Copy)]
pub(super) enum Phase {
    Prefill(usize),
    Decode(usize),
}

impl Phase {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Prefill(_) => "prefill",
            Self::Decode(_) => "decode",
        }
    }

    pub(super) fn index(self) -> usize {
        match self {
            Self::Prefill(index) | Self::Decode(index) => index,
        }
    }
}

pub(super) fn phase_needs_logits(phase: Phase, prompt_len: usize) -> bool {
    match phase {
        Phase::Prefill(index) => prompt_len.checked_sub(1) == Some(index),
        Phase::Decode(_) => true,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EventForwardRoute {
    ProductionFullTail,
    ProductionFullTailDiscardLogits,
    SerialFullTailCapture,
    SerialFullTailNoCapture,
    SerialNoTailCapture,
    SerialNoTailNoCapture,
}

pub(super) fn event_forward_route(
    needs_logits: bool,
    has_capture_plan: bool,
    has_active_readouts: bool,
    has_operation_topology: bool,
) -> EventForwardRoute {
    debug_assert!(!has_active_readouts || has_capture_plan);
    if has_active_readouts {
        if needs_logits {
            EventForwardRoute::SerialFullTailCapture
        } else {
            EventForwardRoute::SerialNoTailCapture
        }
    } else if has_capture_plan || has_operation_topology {
        if needs_logits {
            EventForwardRoute::SerialFullTailNoCapture
        } else {
            EventForwardRoute::SerialNoTailNoCapture
        }
    } else if needs_logits {
        EventForwardRoute::ProductionFullTail
    } else {
        EventForwardRoute::ProductionFullTailDiscardLogits
    }
}

pub(super) fn forward_event(
    execution: &ExecutionPlan,
    schedule: &BoundEventSchedule<'_, '_>,
    forward: &qwen_llm::metal_forward::MetalForward<'_>,
    token: i32,
    position: u32,
    sequence: &mut qwen_llm::runtime::Sequence,
    phase: Phase,
    event: &CompiledEvent,
    needs_logits: bool,
    operation_applications: &mut Vec<OperationApplication>,
    live_readouts: &mut Vec<LiveReadout>,
) -> Result<Vec<f32>> {
    let mut interventions = Vec::new();
    for layer in 0..execution.n_layer {
        for &definition_index in event.operation_indices() {
            if !schedule.operation_selects_layer(definition_index, layer) {
                continue;
            }
            let operation = &schedule.plan().operations[definition_index];
            let intervention = action_to_intervention(
                &operation.id,
                &operation.action,
                layer,
                &execution.directions,
                &execution.coordinate_swaps,
            )?;
            interventions.push((operation.id.clone(), layer, intervention));
        }
    }
    let borrowed = interventions
        .iter()
        .map(|(_, _, op)| *op)
        .collect::<Vec<_>>();
    sequence.check_position(position as usize)?;
    sequence.ensure_can_append(1)?;
    let has_readouts = !event.readout_indices().is_empty();
    let route = event_forward_route(
        needs_logits,
        execution.capture.is_some(),
        has_readouts,
        !event.operation_topology_indices().is_empty(),
    );
    let capture = if has_readouts {
        ensure!(
            !event.capture_layers().is_empty(),
            "active Lens readout selected no capture layers"
        );
        let capture_elements = event
            .capture_layers()
            .len()
            .checked_mul(execution.hidden_size)
            .context("event-local Lens capture size overflow")?;
        let capture_elements = u64::try_from(capture_elements)
            .context("event-local Lens capture exceeds Metal addressing")?;
        Some(
            execution
                .capture
                .as_ref()
                .context("active Lens readout has no capture storage")?
                .view_subrange(0, vec![capture_elements]),
        )
    } else {
        None
    };
    let logits = match route {
        EventForwardRoute::SerialFullTailCapture => {
            let capture = capture
                .as_ref()
                .context("active Lens readout has no capture view")?;
            forward.single_token_with_post_block_interventions(
                token,
                position,
                unsafe { sequence.metal_session_mut() },
                event.capture_layers(),
                capture,
                &borrowed,
            )?
        }
        EventForwardRoute::SerialFullTailNoCapture => forward
            .single_token_with_post_block_interventions_no_capture(
                token,
                position,
                unsafe { sequence.metal_session_mut() },
                &borrowed,
            )?,
        EventForwardRoute::ProductionFullTail => {
            forward.single_token(token, position, unsafe { sequence.metal_session_mut() })?
        }
        EventForwardRoute::SerialNoTailCapture => {
            let capture = capture
                .as_ref()
                .context("active Lens readout has no capture view")?;
            forward.single_token_with_post_block_interventions_no_tail(
                token,
                position,
                unsafe { sequence.metal_session_mut() },
                event.capture_layers(),
                capture,
                &borrowed,
            )?;
            Vec::new()
        }
        EventForwardRoute::SerialNoTailNoCapture => {
            forward.single_token_with_post_block_interventions_no_capture_no_tail(
                token,
                position,
                unsafe { sequence.metal_session_mut() },
                &borrowed,
            )?;
            Vec::new()
        }
        EventForwardRoute::ProductionFullTailDiscardLogits => {
            forward.single_token(token, position, unsafe { sequence.metal_session_mut() })?;
            Vec::new()
        }
    };
    sequence.advance_by(1)?;

    for (id, layer, _) in &interventions {
        operation_applications.push(OperationApplication {
            id: id.clone(),
            layer: *layer,
            phase: phase.label(),
            index: phase.index(),
        });
    }
    if let Some(capture) = capture.as_ref() {
        let values = read_f32_tensor(
            capture,
            event.capture_layers().len() * execution.hidden_size,
        );
        for &definition_index in event.readout_indices() {
            let readout = &schedule.plan().readouts[definition_index];
            for (slot, &layer) in event.capture_layers().iter().enumerate() {
                if !schedule.readout_selects_layer(definition_index, layer) {
                    continue;
                }
                let row = &values[slot * execution.hidden_size..(slot + 1) * execution.hidden_size];
                let prepared = &execution.lenses[&readout.lens];
                let (score_kind, candidate_universe) = readout_score_semantics(prepared);
                let scores = score_readout(prepared, layer, row, readout.top_k)?;
                live_readouts.push(LiveReadout {
                    id: readout.id.clone(),
                    lens: readout.lens.clone(),
                    method: scores.0,
                    score_kind,
                    candidate_universe,
                    source_layer: layer,
                    target_layer: scores.1,
                    phase: phase.label(),
                    index: phase.index(),
                    scores: scores.2,
                });
            }
        }
    }
    Ok(logits)
}
