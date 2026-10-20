//! Request-local Flash-Next speculative packets. No target workspace ownership transfer.

use super::*;
use crate::metal::{
    evaluate_metal_memory_admission_with_cpu_bytes, host_page_size_bytes, price_shared_buffer_upper,
};
use crate::metal_dflash::{
    DFlashAcceptError, DFlashAcceptStopReason, DFlashError, DFlashNextCarry,
    MetalDFlashDraftDecoder, MetalDFlashDraftTarget, MetalDFlashHead, MetalDFlashSession,
    decide_dflash_accept_prefix, dflash_capture_window_limit, dflash_capture_window_span,
};
use crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES;
use crate::sampling::{Sampler, SamplingConfig, SamplingError};
use std::time::Duration;

#[cfg(test)]
use crate::metal::evaluate_metal_memory_admission;

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpDFlashError {
    #[error("invalid Flash-Next DFlash request: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Target(#[from] Qwen4ExpRuntimeError),
    #[error(transparent)]
    Draft(#[from] DFlashError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Acceptance(#[from] DFlashAcceptError),
    #[error(
        "DFlash packet failed at target position {target_position} (poisoned={poisoned}, restored_entry={restored_entry}, rollback_error={rollback_error:?}): {source}"
    )]
    Packet {
        #[source]
        source: Box<Self>,
        target_position: usize,
        poisoned: bool,
        restored_entry: bool,
        rollback_error: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4ExpDFlashFinish {
    Eos,
    TokenLimit,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4ExpDFlashState {
    Running,
    Finished(Qwen4ExpDFlashFinish),
    Failed,
}

/// Cumulative work, including discarded proposals and replay. Durations are wall time.
#[derive(Clone, Debug, Default)]
pub struct Qwen4ExpDFlashStats {
    pub emitted_tokens: usize,
    pub packets: usize,
    pub draft_calls: usize,
    pub proposed_tokens: usize,
    pub verified_inputs: usize,
    pub accepted_drafts: usize,
    pub retained_inputs: usize,
    pub replay_inputs: usize,
    pub restores: usize,
    pub full_retention_packets: usize,
    pub target_draws: usize,
    pub proposal_draws: usize,
    pub prefill: Duration,
    pub first_token_selection: Duration,
    pub draft: Duration,
    pub checkpoint: Duration,
    pub verify: Duration,
    pub acceptance: Duration,
    pub restore: Duration,
    pub replay: Duration,
    pub context_append: Duration,
    pub total: Duration,
}

/// Publish only after target retention and draft-context append both succeed.
/// Stop IDs are included; a text streamer may suppress their rendering.
#[derive(Clone, Debug)]
pub struct Qwen4ExpDFlashPacket {
    pub tokens: Vec<i32>,
    pub finish: Option<Qwen4ExpDFlashFinish>,
}

/// Borrows the target and resident, target-bound head; owns separately admitted
/// request scratch. Drop (or `finish`) releases the runner borrow without resetting
/// it. A healthy boundary always leaves the most recently emitted token unconsumed.
/// Cancellation is boundary-only: `step` is synchronous and never calls user code.
///
/// Packed verification uses the native packed head, not a serial-logit oracle.
/// Partial retention replays scalar inputs; its cost and numerical schedule are
/// explicit. Projection caches see only committed history, never speculative rows.
/// Context storage is bounded by the prompt capture window plus the requested
/// output budget. This version does not compact that append-only bank mid-request.
pub struct Qwen4ExpDFlashAdapter<'run, 'head, 'ctx, 'model, 'gguf> {
    runner: &'run mut Qwen4ExpTextRunner<'ctx, 'model, 'gguf>,
    head: &'head MetalDFlashHead,
    draft_session: MetalDFlashSession,
    captures: Qwen4ExpCaptureBank,
    row_logits: Qwen4ExpRowLogitsBank,
    checkpoint: Qwen4ExpDeviceCheckpointBank,
    checkpoint_live: bool,
    sampler: Sampler,
    proposal_sampler: Option<Sampler>,
    carry: i32,
    max_new_tokens: usize,
    stop_ids: Vec<i32>,
    state: Qwen4ExpDFlashState,
    stats: Qwen4ExpDFlashStats,
}

fn request_extent(
    prompt: usize,
    max_new_tokens: usize,
    block: usize,
    forward_limit: usize,
    packed_capacity: Option<usize>,
) -> Result<usize, Qwen4ExpDFlashError> {
    if prompt == 0 || max_new_tokens == 0 || block == 0 {
        return Err(Qwen4ExpDFlashError::Invalid(
            "prompt, output budget and block must be nonzero",
        ));
    }
    let forwards = prompt
        .checked_add(max_new_tokens - 1)
        .ok_or(Qwen4ExpDFlashError::Invalid("request extent overflow"))?;
    // The drafter computes a nominal block even at a ragged target boundary.
    if forwards > forward_limit
        || forwards
            .checked_add(block)
            .is_none_or(|n| n > i32::MAX as usize)
    {
        return Err(Qwen4ExpDFlashError::Invalid(
            "request exceeds target capacity or position representation",
        ));
    }
    let verify_width = block.min(max_new_tokens.saturating_sub(1)).max(1);
    if verify_width > 1 && packed_capacity.is_none_or(|n| n < verify_width) {
        return Err(Qwen4ExpDFlashError::Invalid(
            "packed workspace does not cover verification width",
        ));
    }
    Ok(verify_width)
}

fn stop_after(
    token: i32,
    emitted: usize,
    budget: usize,
    stops: &[i32],
) -> Option<Qwen4ExpDFlashFinish> {
    if stops.contains(&token) {
        Some(Qwen4ExpDFlashFinish::Eos)
    } else if emitted >= budget {
        Some(Qwen4ExpDFlashFinish::TokenLimit)
    } else {
        None
    }
}

fn append_features(
    session: &mut MetalDFlashSession,
    ctx: &MetalContext,
    features: &Qwen4ExpCapturedFeatures<'_>,
) -> Result<(), DFlashError> {
    session.append_target_ctx_columns_contiguous_now(
        ctx,
        features.tensor(),
        features.positions().start as u32,
        features.positions().len(),
        features.feature_width(),
    )
}

fn request_bank_bytes(
    capture: &Qwen4ExpCapturePlan,
    hidden: usize,
    rows: &Qwen4ExpRowLogitsPlan,
    checkpoint_bytes: u64,
) -> [u64; 6] {
    // The capture bank has four separately aligned buffers. Keep their
    // inventory explicit here because the bank's allocator is private.
    let c = capture.capacity() as u64;
    let logical = [
        c * capture.feature_width() as u64 * 4,
        c * hidden as u64 * 4,
        c * 16,
        c * capture.layers().len() as u64 * 4,
        rows.logical_bytes(),
        checkpoint_bytes,
    ];
    debug_assert_eq!(logical[..4].iter().sum::<u64>(), capture.logical_bytes());
    logical
}

impl<'run, 'head, 'ctx, 'model, 'gguf> Qwen4ExpDFlashAdapter<'run, 'head, 'ctx, 'model, 'gguf> {
    /// Requires an empty, healthy runner and an already loaded/bound head. All
    /// scratch is admitted before prefill. No ordinary-prefill allocation changes.
    /// On prefill failure the caller still owns the runner; inspect poison/position
    /// and reset or return its workspace, rather than retrying the prompt in place.
    pub fn start(
        runner: &'run mut Qwen4ExpTextRunner<'ctx, 'model, 'gguf>,
        head: &'head MetalDFlashHead,
        prompt: &[u32],
        sampler: Sampler,
        max_new_tokens: usize,
        stop_ids: &[i32],
    ) -> Result<(Self, Qwen4ExpDFlashPacket), Qwen4ExpDFlashError> {
        Self::start_with_checkpoint(
            runner,
            head,
            prompt,
            sampler,
            max_new_tokens,
            stop_ids,
            0,
            || Ok(()),
        )
    }

    /// Price the complete optional request scratch before allocating any of it.
    /// The extra CPU allowance is future control/transport storage, not Metal.
    /// Callers may release optional caches on a typed pressure refusal and retry.
    /// Start repeats admission while holding the allocation transaction.
    pub fn admit_request(
        runner: &Qwen4ExpTextRunner<'_, '_, '_>,
        head: &MetalDFlashHead,
        prompt_len: usize,
        max_new_tokens: usize,
        extra_cpu_reserve: u64,
    ) -> Result<(), Qwen4ExpDFlashError> {
        let width = request_extent(
            prompt_len,
            max_new_tokens,
            head.config.block_size as usize,
            runner.remaining_forwards(),
            runner.workspace.packed_prefill_capacity(),
        )?;
        let geometry = &runner.weights.geometry;
        let (_, tail_len) =
            dflash_capture_window_span(prompt_len, dflash_capture_window_limit(head));
        let capture =
            Qwen4ExpCapturePlan::new(geometry, head.target_layer_ids.clone(), tail_len.max(width))
                .map_err(Qwen4ExpRuntimeError::from)?;
        let rows =
            Qwen4ExpRowLogitsPlan::new(geometry, width).map_err(Qwen4ExpRuntimeError::from)?;
        let capacity = tail_len
            .checked_add(max_new_tokens - 1)
            .ok_or(Qwen4ExpDFlashError::Invalid("context capacity overflow"))?;
        let ctx = runner.ctx;
        let mut price = MetalDFlashSession::priced_bytes(
            ctx,
            head,
            geometry.hidden_size() as u64,
            geometry.vocab_size() as u64,
            capacity,
        )?;
        let page = host_page_size_bytes().map_err(Qwen4ExpRuntimeError::from)? as u64;
        for bytes in request_bank_bytes(
            &capture,
            geometry.hidden_size(),
            &rows,
            geometry.device_checkpoint_bytes(),
        ) {
            let buffer = price_shared_buffer_upper(
                bytes,
                ctx.shared_buffer_size_and_align(bytes)
                    .map_err(Qwen4ExpRuntimeError::from)?,
                page,
                ctx.max_buffer_length() as u64,
            )
            .map_err(|error| {
                Qwen4ExpRuntimeError::Invalid(format!("DFlash scratch price: {error}"))
            })?;
            price = price.checked_add(buffer.priced_upper_bytes).ok_or(
                Qwen4ExpDFlashError::Invalid("aggregate scratch price overflow"),
            )?;
        }
        let admission = evaluate_metal_memory_admission_with_cpu_bytes(
            price,
            extra_cpu_reserve,
            QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        if let Some(denied) = admission.refusal() {
            return Err(Qwen4ExpRuntimeError::MemoryAdmission(denied).into());
        }
        Ok(())
    }

    /// Serving variant: preserves typed cancellation through the caller's
    /// checkpoint closure and admits future CPU storage alongside all scratch.
    /// Checkpoints run before work and between prefill commands, never in flight.
    pub fn start_with_checkpoint(
        runner: &'run mut Qwen4ExpTextRunner<'ctx, 'model, 'gguf>,
        head: &'head MetalDFlashHead,
        prompt: &[u32],
        mut sampler: Sampler,
        max_new_tokens: usize,
        stop_ids: &[i32],
        extra_cpu_reserve: u64,
        mut on_checkpoint: impl FnMut() -> Result<(), Qwen4ExpRuntimeError>,
    ) -> Result<(Self, Qwen4ExpDFlashPacket), Qwen4ExpDFlashError> {
        let started = Instant::now();
        on_checkpoint()?;
        if runner.next_position() != 0 || runner.is_poisoned() {
            return Err(Qwen4ExpDFlashError::Invalid(
                "start requires an empty healthy runner",
            ));
        }
        let block = head.config.block_size as usize;
        let width = request_extent(
            prompt.len(),
            max_new_tokens,
            block,
            runner.remaining_forwards(),
            runner.workspace.packed_prefill_capacity(),
        )?;
        runner.validate_prefill_request(prompt)?;
        let hidden = runner.weights.geometry.hidden_size();
        let vocab = runner.weights.geometry.vocab_size();
        if head.config.hidden_size as usize != hidden
            || head.fc.shape
                != [
                    hidden as u64 * head.target_layer_ids.len() as u64,
                    hidden as u64,
                ]
            || head.config.n_target_features_layers as usize != head.target_layer_ids.len()
            || stop_ids.iter().any(|&id| id < 0 || id as usize >= vocab)
        {
            return Err(Qwen4ExpDFlashError::Invalid(
                "head geometry or stop IDs do not match target",
            ));
        }
        let cfg = sampler.config();
        let proposal_sampler = if cfg.temperature > 0.0 {
            if head.selector.is_none() {
                return Err(Qwen4ExpDFlashError::Invalid(
                    "sampled drafting requires a sparse selector head",
                ));
            }
            Some(Sampler::new(SamplingConfig {
                temperature: cfg.temperature,
                seed: cfg.seed ^ 0x4446_4c41_5348_3251,
                ..SamplingConfig::default()
            })?)
        } else {
            None
        };
        let (tail_start, tail_len) =
            dflash_capture_window_span(prompt.len(), dflash_capture_window_limit(head));
        let context_capacity = tail_len
            .checked_add(max_new_tokens - 1)
            .ok_or(Qwen4ExpDFlashError::Invalid("context capacity overflow"))?;
        let (mut captures, row_logits, checkpoint, mut draft_session) = {
            let ctx = runner.ctx;
            let _allocation = ctx.begin_allocation_transaction();
            Self::admit_request(
                runner,
                head,
                prompt.len(),
                max_new_tokens,
                extra_cpu_reserve,
            )?;
            let captures =
                runner.allocate_capture_bank(head.target_layer_ids.clone(), tail_len.max(width))?;
            let row_logits = runner.allocate_row_logits_bank(width)?;
            let checkpoint = runner.allocate_device_checkpoint_bank()?;
            let session = MetalDFlashSession::fresh(
                ctx,
                head,
                hidden as u64,
                vocab as u64,
                context_capacity,
            )?;
            // Device-wide pool/validation growth is not an owned-buffer size.
            // All scratch is now resident: preserve the future CPU allowance
            // against the actual global footprint, without pricing scratch twice.
            let admission = evaluate_metal_memory_admission_with_cpu_bytes(
                0,
                extra_cpu_reserve,
                QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
                ctx.memory_signals(),
                true,
            );
            if let Some(denied) = admission.refusal() {
                return Err(Qwen4ExpRuntimeError::MemoryAdmission(denied).into());
            }
            (captures, row_logits, checkpoint, session)
        };
        let prefill_start = Instant::now();
        let features = runner.prefill_continuation_with_captures_and_checkpoint(
            prompt,
            &mut captures,
            &mut on_checkpoint,
        )?;
        let prefill = prefill_start.elapsed();
        let append_start = Instant::now();
        let feature_width = features.feature_width();
        let tail = features.tensor().view_subrange(
            ((tail_start - features.positions().start) * feature_width) as u64,
            vec![(tail_len * feature_width) as u64],
        );
        draft_session.append_target_ctx_columns_contiguous_now(
            runner.ctx,
            &tail,
            tail_start as u32,
            tail_len,
            feature_width,
        )?;
        drop(features);
        let context_append = append_start.elapsed();
        on_checkpoint()?;
        let selection_start = Instant::now();
        let carry = sampler.sample(runner.logits()?.as_slice())?.token;
        let first_token_selection = selection_start.elapsed();
        let finish = stop_after(carry, 1, max_new_tokens, stop_ids);
        let stats = Qwen4ExpDFlashStats {
            emitted_tokens: 1,
            target_draws: sampler.draws(),
            prefill,
            first_token_selection,
            context_append,
            total: started.elapsed(),
            ..Default::default()
        };
        Ok((
            Self {
                runner,
                head,
                draft_session,
                captures,
                row_logits,
                checkpoint,
                checkpoint_live: false,
                sampler,
                proposal_sampler,
                carry,
                max_new_tokens,
                stop_ids: stop_ids.to_vec(),
                state: finish.map_or(Qwen4ExpDFlashState::Running, Qwen4ExpDFlashState::Finished),
                stats,
            },
            Qwen4ExpDFlashPacket {
                tokens: vec![carry],
                finish,
            },
        ))
    }

    pub fn state(&self) -> Qwen4ExpDFlashState {
        self.state
    }
    pub fn stats(&self) -> &Qwen4ExpDFlashStats {
        &self.stats
    }
    pub fn target_position(&self) -> usize {
        self.runner.next_position()
    }
    pub fn target_is_poisoned(&self) -> bool {
        self.runner.is_poisoned()
    }
    pub fn committed_context_rows(&self) -> usize {
        self.draft_session.target_ctx_n
    }

    /// Executes one real draft/packed-verify packet. No serial fallback. An error
    /// permanently fails this adapter. Healthy errors before replay attempt a
    /// compact entry restore; actual GPU failure is never unpoisoned. If replay
    /// or context publication fails after consuming the checkpoint, target_position
    /// reports the partial retained prefix and the request must be discarded.
    pub fn step(&mut self) -> Result<Qwen4ExpDFlashPacket, Qwen4ExpDFlashError> {
        if self.state != Qwen4ExpDFlashState::Running {
            return Err(Qwen4ExpDFlashError::Invalid("adapter is not running"));
        }
        let start = Instant::now();
        let result = self.step_inner();
        let result = match result {
            Ok(packet) => Ok(packet),
            Err(source) => {
                self.state = Qwen4ExpDFlashState::Failed;
                let mut restored_entry = false;
                let mut rollback_error = None;
                if self.checkpoint_live && !self.runner.is_poisoned() {
                    let t = Instant::now();
                    match self.runner.restore_device_checkpoint(&mut self.checkpoint) {
                        Ok(_) => {
                            restored_entry = true;
                            self.stats.restores += 1;
                        }
                        Err(err) => rollback_error = Some(err.to_string()),
                    }
                    self.stats.restore += t.elapsed();
                }
                self.checkpoint_live = false;
                Err(Qwen4ExpDFlashError::Packet {
                    source: Box::new(source),
                    target_position: self.runner.next_position(),
                    poisoned: self.runner.is_poisoned(),
                    restored_entry,
                    rollback_error,
                })
            }
        };
        self.stats.total += start.elapsed();
        result
    }

    fn step_inner(&mut self) -> Result<Qwen4ExpDFlashPacket, Qwen4ExpDFlashError> {
        let remaining = self.max_new_tokens - self.stats.emitted_tokens;
        let width = (self.head.config.block_size as usize)
            .min(remaining)
            .min(self.runner.remaining_forwards());
        if width == 0 {
            return Err(Qwen4ExpDFlashError::Invalid(
                "no physical verification capacity",
            ));
        }
        let mut target_sampler = self.sampler.clone();
        let mut proposal_sampler = self.proposal_sampler.clone();
        let t = Instant::now();
        let (mut tokens, proposals) = {
            let target = MetalDFlashDraftTarget {
                ctx: self.runner.ctx,
                hidden_size: self.runner.weights.geometry.hidden_size() as u32,
                vocab_size: self.runner.weights.geometry.vocab_size() as u32,
                token_embedding: self.runner.weights.zero_one.layer_zero.token_embedding,
                output: self.runner.weights.output,
            };
            let mut decoder =
                MetalDFlashDraftDecoder::new(target, self.head, &mut self.draft_session);
            self.stats.draft_calls += 1;
            if let Some(sampler) = &mut proposal_sampler {
                let result = decoder.draft_block_sampled(
                    self.carry,
                    self.runner.next_position() as u32,
                    sampler,
                )?;
                (result.draft_tokens, Some(result.proposals))
            } else {
                (
                    decoder.draft_block(self.carry, self.runner.next_position() as u32)?,
                    None,
                )
            }
        };
        self.stats.draft += t.elapsed();
        self.stats.proposed_tokens += tokens.len().saturating_sub(1);
        if tokens.len() != self.head.config.block_size as usize
            || proposals
                .as_ref()
                .is_some_and(|p| p.len() + 1 != tokens.len())
        {
            return Err(Qwen4ExpDFlashError::Invalid("draft result extent mismatch"));
        }
        tokens.truncate(width);
        tokens[0] = self.carry;
        let inputs: Vec<u32> = tokens
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| Qwen4ExpDFlashError::Invalid("negative draft token"))
            })
            .collect::<Result<_, _>>()?;
        let t = Instant::now();
        self.runner
            .capture_device_checkpoint(&mut self.checkpoint)?;
        self.checkpoint_live = true;
        self.stats.checkpoint += t.elapsed();
        let t = Instant::now();
        let rows = self.runner.forward_rows_with_captures_and_logits(
            &inputs,
            &mut self.captures,
            &mut self.row_logits,
        )?;
        self.stats.verify += t.elapsed();
        self.stats.verified_inputs += width;
        let t = Instant::now();
        let decision = decide_dflash_accept_prefix(
            &tokens,
            rows.logits().as_slice(),
            rows.logits().vocab_size(),
            proposals.as_ref().map(|p| &p[..width - 1]),
            &mut target_sampler,
            remaining,
            &self.stop_ids,
        )?;
        self.stats.acceptance += t.elapsed();
        if decision.retain_input_count == width {
            let t = Instant::now();
            append_features(&mut self.draft_session, self.runner.ctx, rows.features())?;
            self.stats.context_append += t.elapsed();
            self.stats.full_retention_packets += 1;
            drop(rows);
        } else {
            drop(rows);
            let t = Instant::now();
            self.checkpoint_live = false;
            self.runner
                .restore_device_checkpoint(&mut self.checkpoint)?;
            self.stats.restore += t.elapsed();
            self.stats.restores += 1;
            for &token in &inputs[..decision.retain_input_count] {
                let t = Instant::now();
                let features = self
                    .runner
                    .forward_token_with_captures(token, &mut self.captures)?;
                self.stats.replay += t.elapsed();
                self.stats.replay_inputs += 1;
                let t = Instant::now();
                append_features(&mut self.draft_session, self.runner.ctx, &features)?;
                self.stats.context_append += t.elapsed();
            }
        }
        self.checkpoint_live = false;
        self.stats.retained_inputs += decision.retain_input_count;
        self.stats.accepted_drafts += decision.accepted_drafts.len();
        let mut emitted = decision.accepted_drafts;
        let mut finish = decision.stop_reason.map(|reason| match reason {
            DFlashAcceptStopReason::Eos => Qwen4ExpDFlashFinish::Eos,
            DFlashAcceptStopReason::TokenLimit => Qwen4ExpDFlashFinish::TokenLimit,
        });
        if let Some(DFlashNextCarry::Correction(token) | DFlashNextCarry::Bonus(token)) =
            decision.next_carry
        {
            self.carry = token;
            emitted.push(token);
            finish = stop_after(
                token,
                self.stats.emitted_tokens + emitted.len(),
                self.max_new_tokens,
                &self.stop_ids,
            );
        }
        self.sampler = target_sampler;
        self.proposal_sampler = proposal_sampler;
        self.stats.target_draws = self.sampler.draws();
        self.stats.proposal_draws = self.proposal_sampler.as_ref().map_or(0, Sampler::draws);
        self.stats.emitted_tokens += emitted.len();
        self.stats.packets += 1;
        self.state = finish.map_or(Qwen4ExpDFlashState::Running, Qwen4ExpDFlashState::Finished);
        Ok(Qwen4ExpDFlashPacket {
            tokens: emitted,
            finish,
        })
    }

    /// No device work: the previous packet already left a committed boundary.
    pub fn cancel(&mut self) -> Result<Qwen4ExpDFlashPacket, Qwen4ExpDFlashError> {
        if self.state == Qwen4ExpDFlashState::Failed {
            return Err(Qwen4ExpDFlashError::Invalid(
                "failed adapter cannot be cancelled or resumed",
            ));
        }
        let finish = match self.state {
            Qwen4ExpDFlashState::Finished(reason) => reason,
            _ => Qwen4ExpDFlashFinish::Cancelled,
        };
        self.state = Qwen4ExpDFlashState::Finished(finish);
        Ok(Qwen4ExpDFlashPacket {
            tokens: Vec::new(),
            finish: Some(finish),
        })
    }

    /// Releases scratch and the runner borrow. Running requests are abandoned at
    /// their current healthy packet boundary, without forwarding the pending carry.
    pub fn finish(self) -> Qwen4ExpDFlashStats {
        self.stats
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod serve_hook_tests {
    use super::*;

    #[test]
    fn aggregate_bank_inventory_prices_each_capture_allocation_plus_logits_and_checkpoint() {
        let geometry = crate::qwen4exp_text_session::Qwen4ExpTextSessionMetalGeometry::from_config(
            &Qwen4ExpConfig::flash_next_reference(),
            64,
        )
        .unwrap();
        for (capture_rows, verify_rows) in [(1, 1), (8, 3), (32, 8)] {
            let capture =
                Qwen4ExpCapturePlan::new(&geometry, vec![45, 5, 25, 15, 35], capture_rows).unwrap();
            let logits = Qwen4ExpRowLogitsPlan::new(&geometry, verify_rows).unwrap();
            let bytes = request_bank_bytes(
                &capture,
                geometry.hidden_size(),
                &logits,
                geometry.device_checkpoint_bytes(),
            );
            assert_eq!(bytes[..4].iter().sum::<u64>(), capture.logical_bytes());
            assert_eq!(bytes[0], capture_rows as u64 * 5 * 2560 * 4);
            assert_eq!(bytes[1], capture_rows as u64 * 2560 * 4);
            assert_eq!(bytes[2], capture_rows as u64 * 16);
            assert_eq!(bytes[3], capture_rows as u64 * 5 * 4);
            assert_eq!(bytes[4], verify_rows as u64 * 248320 * 4);
            assert_eq!(bytes[5], geometry.device_checkpoint_bytes());
        }
    }
}
