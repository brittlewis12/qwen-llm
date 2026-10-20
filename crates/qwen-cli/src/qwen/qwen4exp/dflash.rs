//! Flash-Next request admission and committed-packet CLI publication.

use super::*;
use crate::decode::GenerationResult;
use crate::drafter_policy::PreparedDrafter;
use qwen_llm::loader::DFlashHead;
use qwen_llm::metal_dflash::MetalDFlashHead;
use qwen_llm::qwen4exp_residency::Qwen4ExpMetalWeightPlan;
use qwen_llm::qwen4exp_runtime::{
    Qwen4ExpCapturePlan, Qwen4ExpDFlashAdapter, Qwen4ExpDFlashFinish, Qwen4ExpDFlashPacket,
    Qwen4ExpDFlashStats, Qwen4ExpRowLogitsPlan, Qwen4ExpTextRunner,
    select_qwen4exp_prefill_plan_with_floor,
};
use qwen_llm::qwen4exp_text_session::{Qwen4ExpTextSessionMetalGeometry, Qwen4ExpTextSessionPlan};
use std::time::Duration;

pub(super) fn validate_profiles(drafter: bool, layer: bool, packed: bool) -> Result<()> {
    ensure!(
        !drafter || (!layer && !packed),
        "Flash-Next --drafter cannot be combined with {QWEN4EXP_LAYER_PROFILE_ENV} or {QWEN4EXP_PACKED_PREFILL_PROFILE_ENV}"
    );
    Ok(())
}

pub(super) struct RequestPlan {
    pub(super) packed_extent: Option<usize>,
    verification_width: usize,
}

fn verification_width(block: usize, prompt: usize, budget: usize, limit: usize) -> Result<usize> {
    ensure!(
        block > 0 && prompt > 0 && budget > 0,
        "Flash-Next DFlash requires nonzero block, prompt and output budget"
    );
    let forwards = prompt
        .checked_add(budget - 1)
        .context("Flash-Next DFlash request extent overflow")?;
    ensure!(
        forwards <= limit,
        "Flash-Next DFlash request exceeds the forward limit"
    );
    ensure!(
        forwards
            .checked_add(block)
            .is_some_and(|end| end <= i32::MAX as usize),
        "Flash-Next DFlash nominal draft block exceeds signed positions"
    );
    Ok(block.min(budget - 1).max(1))
}

fn packed_extent(prompt: usize, width: usize, forwards: usize, dense_end: usize) -> Option<usize> {
    // Verification can enter selected QSA after a dense-only prompt. The native
    // loader derives both reusable rows and selected scratch from this extent.
    let extent = if width > 1 && forwards > dense_end {
        forwards
    } else {
        prompt.max(width)
    };
    (extent > 1).then_some(extent)
}

impl RequestPlan {
    pub(super) fn minimum_packed_rows(&self) -> usize {
        // Singleton-only execution needs no packed workspace. If one was
        // requested for prefill, keep it packed even with a one-row verifier.
        self.packed_extent
            .map_or(0, |_| self.verification_width.max(2))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        head: &DFlashHead<'_>,
        prompt: usize,
        budget: usize,
        limit: usize,
        temperature: f32,
        config: &Qwen4ExpConfig,
        capacity: Qwen4ExpSessionCapacity,
    ) -> Result<Self> {
        ensure!(
            temperature == 0.0 || head.selector.is_some(),
            "sampled Flash-Next --drafter requires a sparse selector head"
        );
        let width = verification_width(head.config.block_size as usize, prompt, budget, limit)?;
        let geometry = Qwen4ExpTextSessionMetalGeometry::from_config(
            config,
            capacity.qsa_physical_capacity(),
        )?;
        let dense_end = geometry
            .post_ple()
            .iter()
            .filter_map(|block| block.mixer().qsa())
            .map(|qsa| qsa.output_width())
            .min()
            .context("Flash-Next DFlash requires QSA geometry")?;
        let capture_rows =
            if head.layers.iter().all(|layer| layer.is_swa) && head.config.swa_window > 0 {
                prompt.min(head.config.swa_window as usize)
            } else {
                prompt
            };
        Qwen4ExpCapturePlan::new(
            &geometry,
            head.target_layer_ids.clone(),
            capture_rows.max(width),
        )
        .context("validate Flash-Next DFlash capture geometry")?;
        Qwen4ExpRowLogitsPlan::new(&geometry, width)
            .context("validate Flash-Next DFlash verification logits")?;
        Ok(Self {
            packed_extent: packed_extent(prompt, width, prompt + budget - 1, dense_end),
            verification_width: width,
        })
    }

    pub(super) fn validate_packed_capacity(&self, capacity: Option<usize>) -> Result<()> {
        ensure!(
            self.verification_width == 1
                || capacity.is_some_and(|rows| rows >= self.verification_width),
            "Flash-Next DFlash requires packed capacity covering verification width {}; got {capacity:?}",
            self.verification_width
        );
        Ok(())
    }
}

pub(super) struct HeadLoad {
    pub(super) head: MetalDFlashHead,
    pub(super) session: Qwen4ExpTextSessionPlan,
}

pub(super) fn load_head(
    ctx: &MetalContext,
    gguf: &GgufFile,
    capacity: Qwen4ExpSessionCapacity,
    prepared: &PreparedDrafter,
    request: &RequestPlan,
) -> Result<HeadLoad> {
    let weights = Qwen4ExpMetalWeightPlan::for_native(ctx, gguf)
        .context("validate Flash-Next target weights before drafter residency")?;
    let session = if let Some(extent) = request.packed_extent {
        Qwen4ExpTextSessionPlan::for_config_with_packed_prefill_tokens(
            ctx,
            weights.config(),
            capacity.qsa_physical_capacity(),
            weights.memory_plan(),
            extent,
        )?
    } else {
        Qwen4ExpTextSessionPlan::for_config(
            ctx,
            weights.config(),
            capacity.qsa_physical_capacity(),
            weights.memory_plan(),
        )?
    };
    request.validate_packed_capacity(session.memory_plan().packed_prefill_capacity())?;
    let head_bytes = prepared.priced_metal_bytes(ctx)?;
    let (session, admission) = select_qwen4exp_prefill_plan_with_floor(
        ctx,
        session,
        ctx.memory_signals(),
        request.minimum_packed_rows(),
        head_bytes,
    )
    .context("admit Flash-Next target plus drafter")?;
    // Target loading rechecks normal admission with this head already resident.
    // Adapter scratch is separately admitted against both resident models.
    let head = prepared.load(ctx)?;
    eprintln!(
        "qwen4exp dflash_head: path={} block={} verification_width={} packed_extent={:?} packed_rows={:?} priced_bytes={head_bytes} aggregate_required={:?}",
        prepared.path().display(),
        head.config.block_size,
        request.verification_width,
        request.packed_extent,
        session.memory_plan().packed_prefill_capacity(),
        admission.required_bytes,
    );
    Ok(HeadLoad { head, session })
}

pub(super) struct Execution {
    pub(super) generation: GenerationResult,
    pub(super) prefill_ms: f64,
    pub(super) generator_return: Instant,
    pub(super) stats: Qwen4ExpDFlashStats,
}

pub(super) fn generate(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    head: &MetalDFlashHead,
    prompt: &[u32],
    sampler: Sampler,
    budget: usize,
    stops: &[i32],
    on_token: impl FnMut(i32) -> Result<()>,
) -> Result<Execution> {
    shutdown::checkpoint()?;
    let start = Instant::now();
    let (mut adapter, initial) = Qwen4ExpDFlashAdapter::start_with_checkpoint(
        runner,
        head,
        prompt,
        sampler,
        budget,
        stops,
        0,
        || {
            shutdown::checkpoint()
                .map_err(|error| Qwen4ExpRuntimeError::Checkpoint(error.to_string()))
        },
    )
    .context("start Flash-Next DFlash capture prefill")?;
    let startup = start.elapsed();
    let result = drive_packets(
        initial,
        budget,
        stops,
        || adapter.step().context("step Flash-Next DFlash"),
        shutdown::checkpoint,
        on_token,
    );
    let generator_return = Instant::now();
    let stats = adapter.finish();
    let mut generation = result?;
    let prefill_ms =
        attribute_initial_selection(startup, stats.first_token_selection, &mut generation);
    ensure!(
        !runner.is_poisoned() && runner.next_position() == prompt.len() + generation.transitions,
        "Flash-Next DFlash did not preserve the committed T-1 boundary"
    );
    ensure!(
        stats.emitted_tokens == generation.tokens.len()
            && stats.retained_inputs == generation.transitions,
        "Flash-Next DFlash committed accounting differs from CLI publication"
    );
    Ok(Execution {
        generation,
        prefill_ms,
        generator_return,
        stats,
    })
}

fn attribute_initial_selection(
    startup: Duration,
    selection: Duration,
    generation: &mut GenerationResult,
) -> f64 {
    // Packet publication starts after the initial draw. Move that measured
    // work out of startup and rebase generation-relative clocks exactly once.
    let selection_ms = selection.as_secs_f64() * 1e3;
    generation.first_token_selection_ms = selection_ms;
    generation.wall_ms += selection_ms;
    generation.first_token_ready_ms = generation.first_token_ready_ms.map(|ms| ms + selection_ms);
    generation.first_token_callback_ms = generation
        .first_token_callback_ms
        .map(|ms| ms + selection_ms);
    startup.saturating_sub(selection).as_secs_f64() * 1e3
}

fn drive_packets(
    mut packet: Qwen4ExpDFlashPacket,
    budget: usize,
    stops: &[i32],
    mut step: impl FnMut() -> Result<Qwen4ExpDFlashPacket>,
    mut checkpoint: impl FnMut() -> Result<()>,
    mut on_token: impl FnMut(i32) -> Result<()>,
) -> Result<GenerationResult> {
    ensure!(
        budget > 0 && packet.tokens.len() == 1,
        "Flash-Next DFlash requires one initial token and a nonzero budget"
    );
    let wall = Instant::now();
    let mut tokens = Vec::with_capacity(budget);
    let mut transition_ms = 0.0;
    let mut first_transition_ms = None;
    let mut first_token_callback_ms = None;
    let stop_reason = loop {
        ensure!(
            !packet.tokens.is_empty() && packet.tokens.len() <= budget - tokens.len(),
            "invalid Flash-Next DFlash packet extent"
        );
        let mut expected_finish = None;
        for (index, &token) in packet.tokens.iter().enumerate() {
            ensure!(
                expected_finish.is_none(),
                "Flash-Next DFlash emitted tokens after a terminal token"
            );
            expected_finish = if stops.contains(&token) {
                Some(Qwen4ExpDFlashFinish::Eos)
            } else if tokens.len() + index + 1 == budget {
                Some(Qwen4ExpDFlashFinish::TokenLimit)
            } else {
                None
            };
        }
        ensure!(
            packet.finish == expected_finish,
            "Flash-Next DFlash packet finish disagrees with stop IDs or token budget"
        );
        for token in packet.tokens {
            checkpoint()?;
            tokens.push(token);
            if !stops.contains(&token) {
                on_token(token)?;
                first_token_callback_ms.get_or_insert_with(|| wall.elapsed().as_secs_f64() * 1e3);
            }
        }
        match packet.finish {
            Some(Qwen4ExpDFlashFinish::Eos) => break StopReason::Eos,
            Some(Qwen4ExpDFlashFinish::TokenLimit) => break StopReason::TokenLimit,
            Some(Qwen4ExpDFlashFinish::Cancelled) => bail!("Flash-Next DFlash was cancelled"),
            None => {}
        }
        checkpoint()?;
        let started = Instant::now();
        packet = step()?;
        checkpoint()?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        transition_ms += elapsed_ms;
        first_transition_ms.get_or_insert(elapsed_ms);
    };
    Ok(GenerationResult {
        transitions: tokens.len() - 1,
        tokens,
        wall_ms: wall.elapsed().as_secs_f64() * 1e3,
        // Publication-relative clocks are rebased with the adapter's measured
        // first draw by attribute_initial_selection before returning to the CLI.
        first_token_selection_ms: 0.0,
        first_token_ready_ms: Some(0.0),
        first_token_callback_ms,
        transition_ms,
        first_transition_ms,
        stop_reason,
    })
}

pub(super) fn emit_stats(stats: &Qwen4ExpDFlashStats) {
    eprintln!(
        "qwen4exp dflash: emitted_tokens={} packets={} draft_calls={} proposed_tokens={} verified_inputs={} accepted_drafts={} retained_inputs={} replay_inputs={} restores={} full_retention_packets={} target_draws={} proposal_draws={} target_prefill_ms={:.3} first_token_selection_ms={:.3} draft_ms={:.3} checkpoint_ms={:.3} verify_ms={:.3} acceptance_ms={:.3} restore_ms={:.3} replay_ms={:.3} context_append_ms={:.3} adapter_ms={:.3} target_command_timings=unavailable",
        stats.emitted_tokens,
        stats.packets,
        stats.draft_calls,
        stats.proposed_tokens,
        stats.verified_inputs,
        stats.accepted_drafts,
        stats.retained_inputs,
        stats.replay_inputs,
        stats.restores,
        stats.full_retention_packets,
        stats.target_draws,
        stats.proposal_draws,
        stats.prefill.as_secs_f64() * 1e3,
        stats.first_token_selection.as_secs_f64() * 1e3,
        stats.draft.as_secs_f64() * 1e3,
        stats.checkpoint.as_secs_f64() * 1e3,
        stats.verify.as_secs_f64() * 1e3,
        stats.acceptance.as_secs_f64() * 1e3,
        stats.restore.as_secs_f64() * 1e3,
        stats.replay.as_secs_f64() * 1e3,
        stats.context_append.as_secs_f64() * 1e3,
        stats.total.as_secs_f64() * 1e3,
    );
}

#[cfg(test)]
#[path = "dflash/tests.rs"]
mod tests;
