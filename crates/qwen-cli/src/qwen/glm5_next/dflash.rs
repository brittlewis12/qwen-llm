//! Native packet execution; ordinary GLM sessions never allocate this scratch.

use super::*;
use qwen_llm::glm5_next::{Glm5NextConfig, memory::DYNAMIC_RESERVE_BYTES};
use qwen_llm::glm5_next_metal::{
    Glm5NextAppendCheckpoint, Glm5NextDFlashOptions, Glm5NextDFlashPacket, Glm5NextDFlashRunner,
    Glm5NextDFlashStop, Glm5NextDFlashVerification, Glm5NextRowBankPlan,
};
use qwen_llm::metal::evaluate_metal_memory_admission;
use qwen_llm::metal_dflash::{
    MetalDFlashHead, MetalDFlashSession, dflash_capture_window_limit, dflash_capture_window_span,
};

pub(super) fn aggregate_bytes(parts: &[u64]) -> Result<u64> {
    parts.iter().try_fold(0u64, |sum, &part| {
        sum.checked_add(part)
            .context("GLM DFlash memory price overflow")
    })
}

pub(super) fn admit_bytes(ctx: &MetalContext, bytes: u64, reserve: u64, label: &str) -> Result<()> {
    // Same policy as native GLM admission, including the documented zero process
    // budget convention. This does not disable the recommended working-set cap.
    let admission = evaluate_metal_memory_admission(bytes, reserve, ctx.memory_signals(), true);
    if let Some(denied) = admission.refusal() {
        return Err(anyhow::Error::new(denied)).with_context(|| format!("admit {label}"));
    }
    Ok(())
}

pub(super) fn load_head(
    ctx: &MetalContext,
    prepared: &drafter_policy::PreparedDrafter,
    sampling: SamplingConfig,
) -> Result<MetalDFlashHead> {
    shutdown::checkpoint()?;
    admit_bytes(
        ctx,
        prepared.priced_metal_bytes(ctx)?,
        DYNAMIC_RESERVE_BYTES,
        "GLM DFlash head",
    )?;
    let head = prepared.load(ctx)?;
    proposal_config(sampling, head.selector.is_some())?;
    Ok(head)
}

pub(super) fn packed_rows(ordinary: usize, capacity: usize, block: usize) -> Result<usize> {
    ensure!(
        capacity > 0 && block > 0,
        "GLM DFlash capacity and block must be positive"
    );
    // A short prompt must not shrink every verification packet. Only the actual
    // request capacity may bound the loaded block (e.g. a one-token request).
    Ok(ordinary.max(block).min(capacity))
}

#[derive(Debug, PartialEq, Eq)]
struct RequestGeometry {
    draft_capacity: usize,
    feature_rows: usize,
    verify_rows: usize,
}

fn request_geometry(
    prompt: usize,
    max_tokens: usize,
    capacity: usize,
    block: usize,
    packed: usize,
    window: usize,
) -> Result<RequestGeometry> {
    ensure!(
        prompt > 0 && prompt <= capacity,
        "invalid GLM DFlash prompt capacity"
    );
    let verify_rows = block.min(packed);
    ensure!(
        verify_rows > 0,
        "GLM DFlash needs packed verification scratch"
    );
    let (_, seed_rows) = dflash_capture_window_span(prompt, window);
    let draft_capacity = seed_rows
        .checked_add(max_tokens.saturating_sub(1).min(capacity - prompt))
        .context("GLM DFlash draft context overflow")?;
    Ok(RequestGeometry {
        draft_capacity,
        feature_rows: seed_rows.max(verify_rows),
        verify_rows,
    })
}

/// Mirrors the runner's allocation plan so refusal precedes target prefetch.
/// The runner also re-admits each allocation against live resident memory.
pub(super) fn request_bytes(
    ctx: &MetalContext,
    config: &Glm5NextConfig,
    head: &MetalDFlashHead,
    prompt: usize,
    max_tokens: usize,
    capacity: usize,
    packed: usize,
) -> Result<u64> {
    let geometry = request_geometry(
        prompt,
        max_tokens,
        capacity,
        head.config.block_size as usize,
        packed,
        dflash_capture_window_limit(head),
    )?;
    let price = |bytes| {
        ctx.price_shared_buffer_upper(bytes)
            .ok()
            .map(|price| price.priced_upper_bytes)
    };
    aggregate_bytes(&[
        MetalDFlashSession::priced_bytes(
            ctx,
            head,
            config.hidden_size as u64,
            config.vocab_size as u64,
            geometry.draft_capacity,
        )?,
        Glm5NextRowBankPlan::new(
            config,
            &head.target_layer_ids,
            geometry.feature_rows,
            geometry.verify_rows,
        )?
        .priced_bytes(&price)?,
        Glm5NextAppendCheckpoint::priced_bytes_for(config, &price)?,
    ])
}

fn proposal_config(target: SamplingConfig, selector: bool) -> Result<Option<SamplingConfig>> {
    if target.temperature == 0.0 {
        return Ok(None);
    }
    ensure!(
        selector,
        "GLM sampled --drafter requires a DFlash2 selector; use --temp 0 for a greedy-only head"
    );
    Ok(Some(SamplingConfig {
        temperature: target.temperature,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: target.seed ^ super::super::dflash::DFLASH_PROPOSAL_SEED_XOR,
    }))
}

pub(super) fn runner<'a, 'w>(
    ctx: &'a MetalContext,
    session: &'a mut Glm5NextSession<'w>,
    head: &'a MetalDFlashHead,
    prompt_tokens: usize,
    max_new_tokens: usize,
    stops: &[i32],
    sampler: Sampler,
) -> Result<Glm5NextDFlashRunner<'a, 'w>> {
    let proposal = proposal_config(sampler.config(), head.selector.is_some())?
        .map(Sampler::new)
        .transpose()?;
    Glm5NextDFlashRunner::new(
        ctx,
        session,
        head,
        Glm5NextDFlashOptions {
            prompt_tokens,
            max_new_tokens,
            stop_ids: stops.to_vec(),
            // Keep the session's default Fast lineage and precision. Qwen's
            // calibrated margin fallback does not apply to native GLM.
            verification: Glm5NextDFlashVerification::ConfiguredPacked,
        },
        sampler,
        proposal,
    )
    .context("create GLM DFlash request")
}

pub(super) fn generate(
    runner: &mut Glm5NextDFlashRunner<'_, '_>,
    prompt: &[u32],
    max_tokens: usize,
    on_token: impl FnMut(i32) -> Result<()>,
) -> Result<(GenerationResult, f64)> {
    let prefill_t0 = Instant::now();
    let first = runner.prefill(prompt, &mut || shutdown::checkpoint().is_err())?;
    let prefill_ms = prefill_t0
        .elapsed()
        .saturating_sub(first.costs.selection)
        .as_secs_f64()
        * 1e3;
    let generation = collect_packets(first, max_tokens, on_token, |carry| {
        shutdown::checkpoint()?;
        runner
            .step(carry, &mut || shutdown::checkpoint().is_err())
            .context("execute GLM DFlash packet")
    });
    let generation = match generation {
        Ok(generation) => generation,
        Err(error) => {
            runner.abort_output();
            return Err(error);
        }
    };
    eprintln!(
        "glm5_next dflash: verification=configured_packed initial_selection_ms={:.3} scratch_bytes={} counters={:?}",
        generation.first_token_selection_ms,
        runner.priced_bytes(),
        runner.counters(),
    );
    Ok((generation, prefill_ms))
}

/// Packet publication is the only output source: never re-emit the old carry,
/// never forward a terminal carry, and count committed inputs rather than GPU
/// verification work as GenerationResult transitions. The actual terminal stop
/// token counts toward generation/budget, but never reaches the output callback.
fn collect_packets(
    mut packet: Glm5NextDFlashPacket,
    max_tokens: usize,
    mut on_token: impl FnMut(i32) -> Result<()>,
    mut step: impl FnMut(i32) -> Result<Glm5NextDFlashPacket>,
) -> Result<GenerationResult> {
    let wall_t0 = Instant::now();
    let first_token_selection_ms = packet.costs.selection.as_secs_f64() * 1e3;
    let mut result = GenerationResult {
        tokens: Vec::new(),
        wall_ms: 0.0,
        first_token_selection_ms,
        first_token_ready_ms: (max_tokens > 0).then_some(first_token_selection_ms),
        first_token_callback_ms: None,
        transitions: 0,
        transition_ms: 0.0,
        first_transition_ms: None,
        stop_reason: StopReason::TokenLimit,
    };
    loop {
        let terminal_count = usize::from(packet.stop_reason == Some(Glm5NextDFlashStop::Eos));
        ensure!(
            packet.emitted_tokens.len().checked_sub(terminal_count) == Some(packet.tokens.len())
                && packet.emitted_tokens.starts_with(&packet.tokens),
            "GLM DFlash callback tokens must equal emitted history except for exactly one terminal EOS on an EOS packet"
        );
        if terminal_count != 0 {
            let terminal = packet.emitted_tokens.last().expect("EOS extent validated");
            ensure!(
                !packet.tokens.contains(terminal),
                "GLM DFlash terminal token must not occur in callback tokens"
            );
        }
        ensure!(
            packet.emitted_tokens.len() <= max_tokens.saturating_sub(result.tokens.len()),
            "GLM DFlash packet exceeds the output budget"
        );
        result.transitions = result
            .transitions
            .checked_add(packet.retain_input_count)
            .context("GLM DFlash transition count overflow")?;
        for token in packet.tokens {
            shutdown::checkpoint()?;
            on_token(token)?;
            result.first_token_callback_ms.get_or_insert_with(|| {
                first_token_selection_ms + wall_t0.elapsed().as_secs_f64() * 1e3
            });
        }
        result.tokens.extend(packet.emitted_tokens);
        if let Some(stop) = packet.stop_reason {
            result.stop_reason = match stop {
                Glm5NextDFlashStop::Eos => StopReason::Eos,
                Glm5NextDFlashStop::TokenLimit => StopReason::TokenLimit,
                Glm5NextDFlashStop::Capacity => {
                    bail!("GLM DFlash exhausted target capacity before the admitted output budget")
                }
            };
            break;
        }
        ensure!(
            result.tokens.len() < max_tokens,
            "GLM DFlash omitted terminal token limit"
        );
        let carry = packet
            .next_carry
            .context("GLM DFlash nonterminal packet has no carry")?;
        let t0 = Instant::now();
        packet = step(carry.token())?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        result.transition_ms += ms;
        result.first_transition_ms.get_or_insert(ms);
    }
    ensure!(
        result.transitions == result.tokens.len().saturating_sub(1),
        "GLM DFlash committed transitions must equal generated tokens minus one"
    );
    ensure!(
        result.stop_reason != StopReason::TokenLimit || result.tokens.len() == max_tokens,
        "GLM DFlash token-limit packet must exhaust the output budget"
    );
    result.wall_ms = first_token_selection_ms + wall_t0.elapsed().as_secs_f64() * 1e3;
    Ok(result)
}

#[cfg(test)]
#[path = "dflash/tests.rs"]
mod tests;
