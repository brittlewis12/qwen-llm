//! Prompt-lookup (n-gram) speculative decode loop.

use super::*;

pub(crate) struct PromptLookupGeneration {
    pub(crate) generation: GenerationResult,
    pub(crate) stats: PromptLookupDecodeStats,
    pub(crate) sequence: Sequence,
}

pub(crate) struct PromptLookupScratch {
    pub(crate) verify: MetalDFlashVerifyScratch,
    pub(crate) layer: MetalDFlashLayerMajorScratch,
}

pub(crate) fn generate_prompt_lookup<OnToken>(
    loaded: &LoadedModel,
    forward: &MetalForward<'_>,
    mut sequence: Sequence,
    prompt_ids: &[i32],
    logits: Vec<f32>,
    max_tokens: usize,
    stop_tokens: &[i32],
    mut on_token: OnToken,
) -> Result<PromptLookupGeneration>
where
    OnToken: FnMut(i32) -> Result<()>,
{
    ensure!(max_tokens > 0, "max_tokens must be >= 1");
    let wall_t0 = Instant::now();
    let selection_t0 = Instant::now();
    let mut carry = argmax_i32(&logits);
    let first_token_selection_ms = selection_t0.elapsed().as_secs_f64() * 1e3;
    let first_token_ready_ms = Some(wall_t0.elapsed().as_secs_f64() * 1e3);
    let mut first_token_callback_ms = None;
    let mut first_transition_ms = None;
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut transitions = 0usize;
    let mut proposer = None;
    let mut scratch = None;
    let mut stats = PromptLookupDecodeStats::default();
    let mut transition_wall_ms = 0.0;
    let mut post_callback_policy_ms = 0.0;

    let stop_reason = 'outer: loop {
        shutdown::checkpoint()?;
        tokens.push(carry);

        if stop_tokens.contains(&carry) {
            break StopReason::Eos;
        }
        on_token(carry)?;
        first_token_callback_ms.get_or_insert_with(|| wall_t0.elapsed().as_secs_f64() * 1e3);
        if tokens.len() == max_tokens {
            break StopReason::TokenLimit;
        }

        let transition_t0 = Instant::now();
        if proposer.is_none() {
            let index_t0 = Instant::now();
            proposer = Some(PromptLookupProposer::new(prompt_ids));
            stats.index_build_ms += index_t0.elapsed().as_secs_f64() * 1e3;
        }
        let proposer = proposer
            .as_mut()
            .expect("prompt lookup proposer initialized");
        let update_t0 = Instant::now();
        proposer.commit_verified(&[carry]);
        stats.index_update_ms += update_t0.elapsed().as_secs_f64() * 1e3;

        let lookup_t0 = Instant::now();
        let candidate = proposer.propose();
        stats.lookup_ms += lookup_t0.elapsed().as_secs_f64() * 1e3;
        let Some(candidate) = candidate else {
            stats.abstentions += 1;
            sequence.ensure_can_append(1)?;
            let position =
                u32::try_from(sequence.position()).context("position does not fit u32")?;
            let serial_t0 = Instant::now();
            let next = forward
                .single_token(carry, position, unsafe { sequence.metal_session_mut() })
                .context("prompt-lookup serial decode")?;
            stats.serial_ms += serial_t0.elapsed().as_secs_f64() * 1e3;
            stats.physical_target_positions += 1;
            sequence.advance_by(1)?;
            transitions += 1;
            carry = argmax_i32(&next);
            let elapsed_ms = transition_t0.elapsed().as_secs_f64() * 1e3;
            transition_wall_ms += elapsed_ms;
            first_transition_ms.get_or_insert(elapsed_ms);
            continue;
        };

        stats.attempts += 1;
        let terminal_window =
            terminal_draft_window(&candidate.proposal, tokens.len(), max_tokens, stop_tokens);
        let mut verify_input = Vec::with_capacity(DRAFT_TOKENS + 1);
        verify_input.push(carry);
        let (n_eff, n_drafts_scored) = if let Some(window) = terminal_window {
            verify_input.extend_from_slice(&candidate.proposal[..window.count.saturating_sub(1)]);
            (window.count, window.count)
        } else {
            verify_input.extend_from_slice(&candidate.proposal);
            (DRAFT_TOKENS + 1, DRAFT_TOKENS)
        };
        ensure!(n_eff > 0, "prompt-lookup verifier planned an empty chain");
        sequence.ensure_can_append(n_eff)?;
        let start_position =
            u32::try_from(sequence.position()).context("position does not fit u32")?;

        if scratch.is_none() {
            let before = loaded.context().current_allocated_size();
            let allocation_t0 = Instant::now();
            let verify = MetalDFlashVerifyScratch::fresh(
                loaded.context(),
                loaded.metal_model(),
                (DRAFT_TOKENS + 1) as u32,
                0,
            )
            .context("allocate prompt-lookup verify scratch")?;
            let layer = MetalDFlashLayerMajorScratch::fresh(
                loaded.context(),
                loaded.metal_model(),
                (DRAFT_TOKENS + 1) as u32,
            )
            .context("allocate prompt-lookup layer scratch")?;
            stats.scratch_allocation_ms += allocation_t0.elapsed().as_secs_f64() * 1e3;
            let after = loaded.context().current_allocated_size();
            stats.scratch_allocated_bytes = after.saturating_sub(before);
            stats.scratch_peak_allocated_bytes = after;
            scratch = Some(PromptLookupScratch { verify, layer });
        }
        let scratch = scratch.as_mut().expect("prompt lookup scratch initialized");
        let verify_t0 = Instant::now();
        let verify_argmax = qwen_llm::metal_dflash::encode_packed_verify_layer_major_inner(
            forward,
            &[],
            &verify_input,
            start_position,
            &mut scratch.verify,
            &mut scratch.layer,
            unsafe { sequence.metal_session_mut() },
            None,
            Some(n_eff as u32),
        )
        .context("prompt-lookup packed verify")?;
        stats.verify_ms += verify_t0.elapsed().as_secs_f64() * 1e3;
        stats.verify_calls += 1;
        stats.drafts_scored += n_drafts_scored;
        stats.physical_target_positions += n_eff;

        let mut accepted = Vec::with_capacity(n_drafts_scored);
        let mut terminal = None;
        for (&draft, &target) in candidate.proposal[..n_drafts_scored]
            .iter()
            .zip(&verify_argmax)
        {
            if draft != target {
                break;
            }
            accepted.push(draft);
            if stop_tokens.contains(&draft) {
                terminal = Some(StopReason::Eos);
                break;
            }
            if tokens.len() + accepted.len() == max_tokens {
                terminal = Some(StopReason::TokenLimit);
                break;
            }
        }
        let n_accepted = accepted.len();
        let n_keep = if terminal.is_some() {
            n_accepted
        } else {
            1 + n_accepted
        };
        ensure!(
            n_keep > 0,
            "prompt-lookup terminal plan retained no target state"
        );
        if n_keep < n_eff {
            let restore_t0 = Instant::now();
            qwen_llm::metal_dflash::encode_restore_after_partial_accept_inner(
                forward,
                &scratch.verify,
                n_keep as u32,
                start_position,
                unsafe { sequence.metal_session_mut() },
                Some(n_eff as u32),
            )
            .context("prompt-lookup restore after partial accept")?;
            stats.restore_ms += restore_t0.elapsed().as_secs_f64() * 1e3;
            stats.restore_calls += 1;
        }
        sequence.advance_by(n_keep)?;
        transitions += n_keep;
        stats.accepted_drafts += n_accepted;
        let elapsed_ms = transition_t0.elapsed().as_secs_f64() * 1e3;
        transition_wall_ms += elapsed_ms;
        first_transition_ms.get_or_insert(elapsed_ms);

        for token in accepted {
            tokens.push(token);
            if !stop_tokens.contains(&token) {
                on_token(token)?;
            }
            let update_t0 = Instant::now();
            proposer.commit_verified(&[token]);
            let update_ms = update_t0.elapsed().as_secs_f64() * 1e3;
            stats.index_update_ms += update_ms;
            post_callback_policy_ms += update_ms;
        }
        if let Some(reason) = terminal {
            break 'outer reason;
        }
        carry = verify_argmax[n_accepted];
    };

    ensure!(
        transitions.checked_add(1) == Some(tokens.len()),
        "prompt-lookup generation violated N-1 transition semantics"
    );
    let transition_ms = transition_wall_ms + post_callback_policy_ms;
    Ok(PromptLookupGeneration {
        generation: GenerationResult {
            tokens,
            wall_ms: wall_t0.elapsed().as_secs_f64() * 1e3,
            first_token_selection_ms,
            first_token_ready_ms,
            first_token_callback_ms,
            transitions,
            transition_ms,
            first_transition_ms,
            stop_reason,
        },
        stats,
        sequence,
    })
}
