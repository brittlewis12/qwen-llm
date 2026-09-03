//! Prompt-lookup decode bench.

use super::*;

#[derive(Default)]
pub(crate) struct PldStats {
    pub(crate) index_build_ms: f64,
    pub(crate) index_update_ms: f64,
    pub(crate) lookup_ms: f64,
    pub(crate) serial_ms: f64,
    pub(crate) verify_ms: f64,
    pub(crate) restore_ms: f64,
    pub(crate) decode_ms: f64,
    pub(crate) attempts: u32,
    pub(crate) abstentions: u32,
    pub(crate) verify_calls: u32,
    pub(crate) restore_calls: u32,
    pub(crate) accepted: u32,
    pub(crate) drafts_scored: u32,
    pub(crate) prompt_attempts: u32,
    pub(crate) self_attempts: u32,
    pub(crate) target_transitions: u32,
    pub(crate) final_effective_verify_n: u32,
    pub(crate) accept_histogram: [u32; DRAFT_TOKENS + 1],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PldTerminalCause {
    StopToken,
    OutputLimit,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct PldEvent {
    pub(crate) event: usize,
    pub(crate) carry_index: usize,
    pub(crate) carry_token: i32,
    pub(crate) action: &'static str,
    pub(crate) source: Option<&'static str>,
    pub(crate) source_start: Option<usize>,
    pub(crate) source_end: Option<usize>,
    pub(crate) proposal: Option<[i32; DRAFT_TOKENS]>,
    pub(crate) accepted_prefix: usize,
    pub(crate) terminal_cause: Option<PldTerminalCause>,
    pub(crate) effective_verify_n: Option<usize>,
    pub(crate) n_keep: Option<usize>,
    pub(crate) restored: bool,
    pub(crate) resulting_position: usize,
    pub(crate) emitted_after: usize,
}

pub(crate) fn run_pld(args: PldArgs) -> Result<()> {
    let PldArgs {
        model,
        prompt,
        qwen_chat,
        system,
        disable_thinking,
        tokens,
        stop_tokens,
        no_warmup,
        output,
        trace_events,
    } = args;
    anyhow::ensure!(tokens > 0, "--tokens must be positive");
    if !qwen_chat && (system.is_some() || disable_thinking) {
        anyhow::bail!("`--system` and `--disable-thinking` require `--qwen-chat`");
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[pld] device: {}", ctx.describe());
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let stops = resolve_stop_tokens(&g, stop_tokens)?;
    let m = Model::from_gguf(&g).context("parse model arch")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load weights")?;
    let tok = Tokenizer::from_gguf(&g).context("open tokenizer")?;
    let rendered_prompt = if qwen_chat {
        messages::render_qwen_single_turn_prompt(
            &prompt,
            system.as_deref().filter(|system| !system.is_empty()),
            if disable_thinking {
                messages::QwenGenerationMode::NoThinking
            } else {
                messages::QwenGenerationMode::Thinking
            },
        )
    } else {
        prompt.clone()
    };
    let prompt_ids = tok
        .encode(&rendered_prompt, false)
        .context("tokenize prompt")?;
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt tokenized to zero tokens");
    eprintln!(
        "[pld] model={} prompt_tokens={} gen={} stop_tokens={stops:?}",
        model.display(),
        prompt_ids.len(),
        tokens,
    );

    let mf = MetalForward::new(&ctx, &mm);
    let cap = prompt_ids.len() + tokens + 16;
    if !no_warmup {
        let mut session = MetalSession::fresh(&ctx, &mm, cap).context("warmup session")?;
        let _ = mf.single_token(prompt_ids[0], 0, &mut session)?;
    }

    let mut ref_session = MetalSession::fresh(&ctx, &mm, cap).context("ref session")?;
    let mut ref_scratch = MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, 16)
        .context("ref prefill scratch")?;
    let ref_total_start = Instant::now();
    let ref_prefill_start = Instant::now();
    let ref_last_logits = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut ref_session,
        &mut ref_scratch,
        &[],
        None,
    )
    .context("ref packed prefill")?;
    let ref_prefill_ms = ref_prefill_start.elapsed().as_secs_f64() * 1e3;
    let ref_decode_start = Instant::now();
    let mut ref_generated = Vec::with_capacity(tokens);
    let mut ref_carry = argmax_i32(&ref_last_logits);
    let mut ref_processed_pos = (prompt_ids.len() - 1) as u32;
    loop {
        shutdown::checkpoint()?;
        ref_generated.push(ref_carry);
        if stops.contains(&ref_carry) || ref_generated.len() >= tokens {
            break;
        }
        let position = ref_processed_pos + 1;
        let logits = mf.single_token(ref_carry, position, &mut ref_session)?;
        ref_carry = argmax_i32(&logits);
        ref_processed_pos = position;
    }
    let ref_decode_ms = ref_decode_start.elapsed().as_secs_f64() * 1e3;
    let ref_total_ms = ref_total_start.elapsed().as_secs_f64() * 1e3;

    let mut candidate_session = MetalSession::fresh(&ctx, &mm, cap).context("candidate session")?;
    let mut candidate_prefill_scratch = MetalDFlashLayerMajorScratch::fresh_prefill(&ctx, &mm, 16)
        .context("candidate prefill scratch")?;
    let mut verify_scratch =
        MetalDFlashVerifyScratch::fresh(&ctx, &mm, 8, 0).context("PLD verify scratch")?;
    let mut layer_scratch =
        MetalDFlashLayerMajorScratch::fresh(&ctx, &mm, 8).context("PLD layer scratch")?;
    let candidate_total_start = Instant::now();
    let index_start = Instant::now();
    let mut proposer = PromptLookupProposer::new(&prompt_ids);
    let mut stats = PldStats {
        index_build_ms: index_start.elapsed().as_secs_f64() * 1e3,
        ..PldStats::default()
    };
    let candidate_prefill_start = Instant::now();
    let candidate_last_logits = prefill_tokens_with_multi_hidden(
        &mf,
        &prompt_ids,
        0,
        &mut candidate_session,
        &mut candidate_prefill_scratch,
        &[],
        None,
    )
    .context("candidate packed prefill")?;
    let candidate_prefill_ms = candidate_prefill_start.elapsed().as_secs_f64() * 1e3;
    let decode_start = Instant::now();
    let mut candidate_generated = Vec::with_capacity(tokens);
    let mut carry = argmax_i32(&candidate_last_logits);
    let mut processed_pos = (prompt_ids.len() - 1) as u32;
    let mut events = Vec::new();
    let mut event_index = 0usize;
    'outer: loop {
        shutdown::checkpoint()?;
        let carry_index = candidate_generated.len();
        let event_carry = carry;
        candidate_generated.push(carry);
        let update_start = Instant::now();
        proposer.commit_verified(&[carry]);
        stats.index_update_ms += update_start.elapsed().as_secs_f64() * 1e3;
        if stops.contains(&carry) || candidate_generated.len() >= tokens {
            if trace_events {
                events.push(PldEvent {
                    event: event_index,
                    carry_index,
                    carry_token: event_carry,
                    action: "terminal_emit",
                    source: None,
                    source_start: None,
                    source_end: None,
                    proposal: None,
                    accepted_prefix: 0,
                    terminal_cause: Some(if stops.contains(&carry) {
                        PldTerminalCause::StopToken
                    } else {
                        PldTerminalCause::OutputLimit
                    }),
                    effective_verify_n: None,
                    n_keep: None,
                    restored: false,
                    resulting_position: candidate_session.kv_n_pos[0],
                    emitted_after: candidate_generated.len(),
                });
            }
            break;
        }

        let lookup_start = Instant::now();
        let candidate = proposer.propose();
        stats.lookup_ms += lookup_start.elapsed().as_secs_f64() * 1e3;
        let Some(candidate) = candidate else {
            stats.abstentions += 1;
            let serial_start = Instant::now();
            let position = processed_pos + 1;
            let logits = mf.single_token(carry, position, &mut candidate_session)?;
            stats.serial_ms += serial_start.elapsed().as_secs_f64() * 1e3;
            stats.target_transitions += 1;
            carry = argmax_i32(&logits);
            processed_pos = position;
            if trace_events {
                events.push(PldEvent {
                    event: event_index,
                    carry_index,
                    carry_token: event_carry,
                    action: "abstain",
                    source: None,
                    source_start: None,
                    source_end: None,
                    proposal: None,
                    accepted_prefix: 0,
                    terminal_cause: None,
                    effective_verify_n: None,
                    n_keep: Some(1),
                    restored: false,
                    resulting_position: candidate_session.kv_n_pos[0],
                    emitted_after: candidate_generated.len(),
                });
            }
            event_index += 1;
            continue;
        };

        stats.attempts += 1;
        match candidate.source {
            ProposalSource::Prompt => stats.prompt_attempts += 1,
            ProposalSource::SelfOutput => stats.self_attempts += 1,
        }
        let terminal_window = terminal_draft_window(
            &candidate.proposal,
            candidate_generated.len(),
            tokens,
            &stops,
        );
        let mut verify_input = Vec::with_capacity(8);
        verify_input.push(carry);
        let (n_eff, n_drafts_scored) = if let Some(window) = terminal_window {
            let count = window.count;
            verify_input.extend_from_slice(&candidate.proposal[..count.saturating_sub(1)]);
            (count, count)
        } else {
            verify_input.extend_from_slice(&candidate.proposal);
            (8, DRAFT_TOKENS)
        };
        stats.drafts_scored += n_drafts_scored as u32;
        stats.final_effective_verify_n = n_eff as u32;
        let start_position = processed_pos + 1;
        let verify_start = Instant::now();
        let verify_argmax = qwen_llm::metal_dflash::encode_packed_verify_layer_major_inner(
            &mf,
            &[],
            &verify_input,
            start_position,
            &mut verify_scratch,
            &mut layer_scratch,
            &mut candidate_session,
            None,
            Some(n_eff as u32),
        )
        .context("PLD packed verify")?;
        stats.verify_ms += verify_start.elapsed().as_secs_f64() * 1e3;
        stats.verify_calls += 1;
        stats.target_transitions += n_eff as u32;

        let mut accepted_tokens = Vec::with_capacity(n_drafts_scored);
        let mut stop_now = false;
        for (&draft, &target) in candidate.proposal[..n_drafts_scored]
            .iter()
            .zip(&verify_argmax)
        {
            if draft != target {
                break;
            }
            accepted_tokens.push(draft);
            candidate_generated.push(draft);
            stats.accepted += 1;
            if stops.contains(&draft) || candidate_generated.len() >= tokens {
                stop_now = true;
                break;
            }
        }
        let n_accepted = accepted_tokens.len();
        stats.accept_histogram[n_accepted] += 1;
        let n_keep = if stop_now { n_accepted } else { 1 + n_accepted };
        let restored = n_keep < n_eff;
        if restored {
            let restore_start = Instant::now();
            qwen_llm::metal_dflash::encode_restore_after_partial_accept_inner(
                &mf,
                &verify_scratch,
                n_keep as u32,
                start_position,
                &mut candidate_session,
                Some(n_eff as u32),
            )
            .context("PLD restore after partial accept")?;
            stats.restore_ms += restore_start.elapsed().as_secs_f64() * 1e3;
            stats.restore_calls += 1;
        }
        let update_start = Instant::now();
        proposer.commit_verified(&accepted_tokens);
        stats.index_update_ms += update_start.elapsed().as_secs_f64() * 1e3;
        if !stop_now {
            processed_pos += 1 + n_accepted as u32;
            carry = verify_argmax[n_accepted];
        }
        if trace_events {
            events.push(PldEvent {
                event: event_index,
                carry_index,
                carry_token: event_carry,
                action: "attempt",
                source: Some(match candidate.source {
                    ProposalSource::Prompt => "prompt",
                    ProposalSource::SelfOutput => "self",
                }),
                source_start: Some(candidate.source_start),
                source_end: Some(candidate.source_end),
                proposal: Some(candidate.proposal),
                accepted_prefix: n_accepted,
                terminal_cause: terminal_window.map(|window| match window.cause {
                    PromptLookupTerminalCause::StopToken => PldTerminalCause::StopToken,
                    PromptLookupTerminalCause::OutputLimit => PldTerminalCause::OutputLimit,
                }),
                effective_verify_n: Some(n_eff),
                n_keep: Some(n_keep),
                restored,
                resulting_position: candidate_session.kv_n_pos[0],
                emitted_after: candidate_generated.len(),
            });
        }
        event_index += 1;
        if stop_now {
            break 'outer;
        }
    }
    stats.decode_ms = decode_start.elapsed().as_secs_f64() * 1e3;
    let candidate_total_ms = candidate_total_start.elapsed().as_secs_f64() * 1e3;

    let identical = ref_generated == candidate_generated;
    anyhow::ensure!(identical, "PLD generated tokens differ from serial target");
    let pending_terminal_token = *ref_generated.last().context("PLD emitted no tokens")?;
    let expected_position = prompt_ids.len() + ref_generated.len() - 1;
    const PLD_CONTINUATION_AUDIT_STEPS: usize = 16;
    let mut audit = audit_mtp_target_state(
        &mf,
        &mut ref_session,
        &mut candidate_session,
        expected_position,
        pending_terminal_token,
    )?;
    for offset in 1..PLD_CONTINUATION_AUDIT_STEPS {
        let next_token = audit.continuation_token;
        let next = audit_mtp_target_state(
            &mf,
            &mut ref_session,
            &mut candidate_session,
            expected_position + offset,
            next_token,
        )?;
        merge_target_state_audit(&mut audit, next);
    }
    audit.resume_audit_pass &= audit.gdn_state_max_abs <= 1e-2;
    audit.resume_audit_pass &= audit.gdn_conv_max_abs <= 1e-1;
    anyhow::ensure!(audit.resume_audit_pass, "PLD terminal resume audit failed");

    let decode_speedup = ref_decode_ms / stats.decode_ms;
    let total_speedup = ref_total_ms / candidate_total_ms;
    eprintln!(
        "[pld] ref prefill={ref_prefill_ms:.1} decode={ref_decode_ms:.1} total={ref_total_ms:.1} ms"
    );
    eprintln!(
        "[pld] pld index={:.3} prefill={candidate_prefill_ms:.1} decode={:.1} \
         total={candidate_total_ms:.1} ms",
        stats.index_build_ms, stats.decode_ms,
    );
    eprintln!(
        "[pld] speedup decode={decode_speedup:.3}x total={total_speedup:.3}x \
         attempts={} abstentions={} verifies={} restores={} accepted={}/{}",
        stats.attempts,
        stats.abstentions,
        stats.verify_calls,
        stats.restore_calls,
        stats.accepted,
        stats.drafts_scored,
    );
    eprintln!(
        "[pld] phases lookup={:.3} update={:.3} serial={:.1} verify={:.1} restore={:.1} ms",
        stats.lookup_ms, stats.index_update_ms, stats.serial_ms, stats.verify_ms, stats.restore_ms,
    );

    if let Some(output_path) = output {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let event_trace = if trace_events {
            serde_json::to_value(&events)?
        } else {
            serde_json::Value::Null
        };
        let row = serde_json::json!({
            "schema_version": 1,
            "model": model.display().to_string(),
            "prompt_tokens": prompt_ids.len(),
            "generated_requested": tokens,
            "generated_emitted": ref_generated.len(),
            "stop_tokens": stops,
            "policy": {
                "sources": "prompt_and_committed_output",
                "selector": "most_recent",
                "match_tokens": 8,
                "draft_tokens": DRAFT_TOKENS,
                "physical_verify_n": 8,
            },
            "reference": {
                "prefill_ms": ref_prefill_ms,
                "decode_ms": ref_decode_ms,
                "total_ms": ref_total_ms,
            },
            "candidate": {
                "index_build_ms": stats.index_build_ms,
                "prefill_ms": candidate_prefill_ms,
                "decode_ms": stats.decode_ms,
                "total_ms": candidate_total_ms,
                "lookup_ms": stats.lookup_ms,
                "index_update_ms": stats.index_update_ms,
                "serial_ms": stats.serial_ms,
                "verify_ms": stats.verify_ms,
                "restore_ms": stats.restore_ms,
                "attempts": stats.attempts,
                "abstentions": stats.abstentions,
                "verify_calls": stats.verify_calls,
                "restore_calls": stats.restore_calls,
                "accepted": stats.accepted,
                "drafts_scored": stats.drafts_scored,
                "prompt_attempts": stats.prompt_attempts,
                "self_attempts": stats.self_attempts,
                "target_transitions": stats.target_transitions,
                "final_effective_verify_n": (stats.verify_calls > 0)
                    .then_some(stats.final_effective_verify_n),
                "accept_histogram": stats.accept_histogram,
            },
            "decode_speedup": decode_speedup,
            "total_speedup": total_speedup,
            "identical": identical,
            "events": event_trace,
            "target_state": {
                "continuation_steps": PLD_CONTINUATION_AUDIT_STEPS,
                "resume_audit_pass": audit.resume_audit_pass,
                "kv_position_equal": audit.kv_position_equal,
                "kv_payload_exact": audit.kv_payload_exact,
                "kv_payload_max_abs": audit.kv_payload_max_abs,
                "kv_payload_cosine": audit.kv_payload_cosine,
                "gdn_state_max_abs": audit.gdn_state_max_abs,
                "gdn_conv_max_abs": audit.gdn_conv_max_abs,
                "continuation_argmax_equal": audit.continuation_argmax_equal,
                "continuation_logits_max_abs": audit.continuation_logits_max_abs,
                "continuation_logits_cosine": audit.continuation_logits_cosine,
            },
        });
        std::fs::write(&output_path, serde_json::to_string(&row)? + "\n")?;
        eprintln!("[pld] wrote {}", output_path.display());
    }
    Ok(())
}
