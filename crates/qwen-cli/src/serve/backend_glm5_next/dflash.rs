//! Cold-prefill speculation only: target KV alone cannot restore draft features.

use super::*;
use crate::{GenerationResult, StopReason};
use qwen_llm::glm5_next::memory::DYNAMIC_RESERVE_BYTES;
use qwen_llm::glm5_next_metal::{
    Glm5NextDFlashError, Glm5NextDFlashOptions, Glm5NextDFlashPacket, Glm5NextDFlashRunner,
    Glm5NextDFlashStop, Glm5NextDFlashVerification,
};
use qwen_llm::metal::evaluate_metal_memory_admission;
use qwen_llm::metal_dflash::MetalDFlashHead;
use qwen_llm::sampling::SamplingConfig;
use qwen_llm::tokenizer::Tokenize;

pub(super) fn load_head(
    ctx: &MetalContext,
    prepared: &crate::drafter_policy::PreparedDrafter,
) -> Result<MetalDFlashHead> {
    crate::shutdown::checkpoint()?;
    let admission = evaluate_metal_memory_admission(
        prepared.priced_metal_bytes(ctx)?,
        DYNAMIC_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    if let Some(denied) = admission.refusal() {
        return Err(anyhow::Error::new(denied)).context("admit GLM DFlash head");
    }
    prepared.load(ctx)
}

fn proposal_config(
    target: SamplingConfig,
    selector: bool,
) -> Result<Option<SamplingConfig>, ServeError> {
    if target.temperature == 0.0 {
        return Ok(None);
    }
    if !selector {
        return Err(ServeError::invalid_request(
            Some("temperature"),
            "GLM sampled --drafter requires a DFlash2 selector; use temperature 0 for this head",
        ));
    }
    Ok(Some(SamplingConfig {
        temperature: target.temperature,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: target.seed ^ crate::dflash::DFLASH_PROPOSAL_SEED_XOR,
    }))
}

fn cpu_reserve(transport: u64, maximum: usize) -> Result<u64, ServeError> {
    let output = u64::try_from(maximum)
        .ok()
        .and_then(|n| n.checked_mul(std::mem::size_of::<i32>() as u64))
        .ok_or_else(|| ServeError::server_error("GLM DFlash output reservation overflow"))?;
    // OwnerSink already includes the control/history allowance in transport.
    super::super::transport_memory::combined_reserve(transport, output)
}

fn adapter_serve_error(error: Glm5NextDFlashError) -> ServeError {
    match error {
        Glm5NextDFlashError::Target(error) => session_error(error),
        error => ServeError::server_error(format!("{FAMILY} DFlash execution: {error}")),
    }
}

fn adapter_error(error: Glm5NextDFlashError) -> BackendFailure {
    adapter_serve_error(error).into()
}

fn packed_candidates(maximum: usize, floor: usize) -> Result<Vec<usize>, ServeError> {
    if floor == 0 || maximum < floor {
        return Err(ServeError::server_error(
            "GLM DFlash invalid packed row floor",
        ));
    }
    let mut rows = vec![maximum];
    rows.extend(
        prefill_plan::candidates(maximum, true)
            .into_iter()
            .filter(|&n| n >= floor),
    );
    rows.push(floor);
    rows.dedup();
    Ok(rows)
}

/// Only admission failures belong in the outer Result. Once prefill starts,
/// return its result as T so even a later pressure error cannot replay output.
/// Each attempt owns and drops its target/runner before returning a refusal.
fn with_admission_retry<T>(
    mut rows: &[usize],
    mut plan: impl FnMut(&[usize]) -> Result<prefill_plan::Plan, ServeError>,
    mut attempt: impl FnMut(usize) -> Result<T, ServeError>,
) -> Result<(T, prefill_plan::Plan), ServeError> {
    loop {
        let selected = plan(rows)?;
        match attempt(selected.rows) {
            Ok(result) => return Ok((result, selected)),
            Err(error) if is_pressure_refusal(&error) => {
                let Some(next) = rows.iter().position(|&n| n < selected.rows) else {
                    return Err(error);
                };
                rows = &rows[next..];
            }
            Err(error) => return Err(error),
        }
    }
}

fn fail(message: impl std::fmt::Display) -> BackendFailure {
    ServeError::server_error(format!("{FAMILY} DFlash {message}")).into()
}

/// Preserve the original transport error even if cancellation also fails rollback.
fn with_ticks<T>(
    sink: &mut dyn GenerationSink,
    execute: impl FnOnce(&mut dyn FnMut() -> bool) -> Result<T, Glm5NextDFlashError>,
) -> Result<T, BackendFailure> {
    sink.tick().map_err(BackendFailure::Aborted)?;
    let mut abort = None;
    let result = execute(&mut || {
        if abort.is_none() {
            abort = sink.tick().err();
        }
        abort.is_some()
    });
    if let Some(error) = abort {
        return Err(BackendFailure::Aborted(error));
    }
    let value = result.map_err(adapter_error)?;
    sink.tick().map_err(BackendFailure::Aborted)?;
    Ok(value)
}

impl Glm5NextBackend<'_, '_> {
    pub(super) fn generate_dflash(
        &mut self,
        request: &ServeRequest,
        tokens: &[u32],
        maximum: usize,
        sampler: Sampler,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        let head = self
            .drafter
            .as_ref()
            .ok_or_else(|| fail("head was not loaded"))?;
        let proposal = proposal_config(sampler.config(), head.selector.is_some())?
            .map(Sampler::new)
            .transpose()
            .map_err(|error| fail(format!("proposal sampler: {error}")))?;
        let reserve = cpu_reserve(sink.transport_reserve_bytes(), maximum)?;
        with_transport_snapshot_release(&mut self.cache, None, || {
            super::super::transport_memory::admit_resident_transport_with_headroom(
                reserve,
                qwen_llm::metal::host_memory_headroom(),
            )
        })?;
        sink.tick().map_err(BackendFailure::Aborted)?;

        // No lookup, restore, capture, or successful-history publication in this
        // lane. A fresh runner must capture the actual prompt's hidden features.
        self.session = None;
        self.history.clear();
        self.chunk_origin = None;
        self.live_canonical = false;
        let lineage = request_lineage(request, self.lineage);
        let ctx = self.ctx;
        let weights = self.weights;
        let prepared = &self.prepared;
        let floor = (head.config.block_size as usize).min(prepared.capacity);
        let candidates = packed_candidates(prepared.prefill_rows.max(floor), floor)?;
        let (result, plan) = with_admission_retry(
            &candidates,
            |rows| {
                prefill_plan::select(rows, ctx.memory_signals(), reserve, None, |rows| {
                    let ledger =
                        Glm5NextSession::plan_memory(ctx, weights, prepared.capacity, rows)
                            .map_err(session_error)?;
                    Ok(prefill_plan::Price::from_ledger(&ledger))
                })
            },
            |rows| {
                if let Err(error) = sink.tick() {
                    return Ok(Err(BackendFailure::Aborted(error)));
                }
                let mut session = with_snapshot_release(&mut self.cache, None, || {
                    new_session(ctx, weights, prepared, reserve, lineage, rows)
                })
                .map_err(session_error)?;
                // Session planning alone can fit while aggregate runner scratch
                // cannot. Refuse here, before prefill, drop this target and retry
                // narrower packed rows without shrinking the verification block.
                let mut runner = Glm5NextDFlashRunner::new_with_cpu_reserve(
                    ctx,
                    &mut session,
                    head,
                    Glm5NextDFlashOptions {
                        prompt_tokens: tokens.len(),
                        max_new_tokens: maximum,
                        stop_ids: CHAT_STOPS.to_vec(),
                        verification: Glm5NextDFlashVerification::ConfiguredPacked,
                    },
                    sampler.clone(),
                    proposal.clone(),
                    reserve,
                )
                .map_err(adapter_serve_error)?;
                let result = execute(
                    &mut runner,
                    tokens,
                    maximum,
                    prepared.artifact.tokenizer(),
                    sink,
                );
                let counters = runner.counters();
                let scratch_bytes = runner.priced_bytes();
                if result.is_err() {
                    runner.abort_output();
                }
                drop(runner);
                drop(session);
                Ok(result.map(|(generation, prefill_ms)| {
                    (generation, prefill_ms, counters, scratch_bytes)
                }))
            },
        )?;
        let (generation, prefill_ms, counters, scratch_bytes) = result?;
        tracing::info!(target: "qwen_diag",
            "serve dflash: family=glm5_next mode=cold_prefill matched_tokens=0 reuse_source=none restore_ms=0 snapshot_schedule=none prefill_ms={prefill_ms:.1} decode_ms={:.1} lineage={} fast_precision={} prefill_rows={} verification=configured_packed scratch_bytes={scratch_bytes} counters={counters:?}",
            generation.wall_ms, lineage_name(lineage), self.prepared.fast_precision.name(), plan.rows);
        Ok(super::super::outcome::finish_generation(
            tokens.len(),
            &generation,
            0,
            0.0,
        ))
    }
}

fn execute(
    runner: &mut Glm5NextDFlashRunner<'_, '_>,
    prompt: &[u32],
    maximum: usize,
    tokenizer: &dyn Tokenize,
    sink: &mut dyn GenerationSink,
) -> Result<(GenerationResult, f64), BackendFailure> {
    let started = Instant::now();
    // The adapter polls transport between capture-prefill chunks as well as at
    // packet boundaries, preserving heartbeat and typed cancellation.
    let first = with_ticks(sink, |cancelled| runner.prefill(prompt, cancelled))?;
    let selection_ms = first.costs.selection.as_secs_f64() * 1e3;
    let prefill_ms = (started.elapsed().as_secs_f64() * 1e3 - selection_ms).max(0.0);
    let generation = collect_packets(first, maximum, tokenizer, sink, |carry, sink| {
        with_ticks(sink, |cancelled| runner.step(carry, cancelled))
    })?;
    if runner.target_position() != prompt.len() + generation.transitions {
        return Err(fail("consumed-prefix accounting mismatch"));
    }
    Ok((generation, prefill_ms))
}

fn collect_packets(
    mut packet: Glm5NextDFlashPacket,
    maximum: usize,
    tokenizer: &dyn Tokenize,
    sink: &mut dyn GenerationSink,
    mut step: impl FnMut(i32, &mut dyn GenerationSink) -> Result<Glm5NextDFlashPacket, BackendFailure>,
) -> Result<GenerationResult, BackendFailure> {
    let started = Instant::now();
    let selection_ms = packet.costs.selection.as_secs_f64() * 1e3;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(maximum)
        .map_err(|error| fail(error))?;
    let mut generation = GenerationResult {
        tokens,
        wall_ms: 0.0,
        first_token_selection_ms: selection_ms,
        first_token_ready_ms: Some(selection_ms),
        first_token_callback_ms: None,
        transitions: 0,
        transition_ms: 0.0,
        first_transition_ms: None,
        stop_reason: StopReason::TokenLimit,
    };
    loop {
        let visible = match packet.stop_reason {
            Some(Glm5NextDFlashStop::Eos) => {
                let (terminal, visible) = packet
                    .emitted_tokens
                    .split_last()
                    .filter(|(token, _)| CHAT_STOPS.contains(token))
                    .ok_or_else(|| fail("EOS history omitted the selected stop token"))?;
                if *terminal < 0 || *terminal as u32 >= tokenizer.n_vocab() {
                    return Err(fail("invalid stop token in history"));
                }
                visible
            }
            Some(Glm5NextDFlashStop::Capacity) => {
                return Err(fail("exhausted capacity before the admitted output budget"));
            }
            _ => packet.emitted_tokens.as_slice(),
        };
        if visible != packet.tokens.as_slice() {
            return Err(fail("callback tokens differ from emitted history"));
        }
        let selected = packet.emitted_tokens.len();
        let total = generation
            .tokens
            .len()
            .checked_add(selected)
            .filter(|&n| n <= maximum && selected > 0)
            .ok_or_else(|| fail("empty packet or output budget exceeded"))?;
        let transitions = generation
            .transitions
            .checked_add(packet.retain_input_count)
            .filter(|&n| n == total - 1)
            .ok_or_else(|| fail("packet transitions must equal selected tokens minus one"))?;
        if packet.tokens.iter().any(|&token| {
            token < 0 || token as u32 >= tokenizer.n_vocab() || CHAT_STOPS.contains(&token)
        }) {
            return Err(fail("invalid visible token in packet"));
        }
        let carry = packet.next_carry.map(|carry| carry.token());
        match packet.stop_reason {
            None if total >= maximum
                || carry.is_none()
                || carry != packet.tokens.last().copied() =>
            {
                return Err(fail("nonterminal packet has no valid continuation"));
            }
            Some(Glm5NextDFlashStop::TokenLimit) if total != maximum => {
                return Err(fail("premature token-limit packet"));
            }
            _ => {}
        }
        for token in packet.tokens {
            let bytes = tokenizer
                .try_decode_piece_bytes(token)
                .map_err(|error| fail(format!("decode token {token}: {error}")))?;
            sink.piece(&bytes)
                .and_then(|()| sink.tick())
                .map_err(BackendFailure::Aborted)?;
            generation
                .first_token_callback_ms
                .get_or_insert_with(|| selection_ms + started.elapsed().as_secs_f64() * 1e3);
        }
        generation.tokens.extend_from_slice(&packet.emitted_tokens);
        generation.transitions = transitions;
        if packet.stop_reason == Some(Glm5NextDFlashStop::Eos) {
            generation.stop_reason = StopReason::Eos;
        }
        if packet.stop_reason.is_some() {
            sink.tick().map_err(BackendFailure::Aborted)?;
            break;
        }
        let step_started = Instant::now();
        // next_carry is already in packet.tokens; only feed it back, never emit it.
        packet = step(carry.expect("validated continuation"), sink)?;
        let elapsed = step_started.elapsed().as_secs_f64() * 1e3;
        generation.transition_ms += elapsed;
        generation.first_transition_ms.get_or_insert(elapsed);
    }
    generation.wall_ms = selection_ms + started.elapsed().as_secs_f64() * 1e3;
    Ok(generation)
}

#[cfg(test)]
mod tests;
