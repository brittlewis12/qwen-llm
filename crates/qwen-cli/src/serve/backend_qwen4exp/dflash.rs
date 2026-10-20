//! Explicit cold-prefill packets; no snapshot reuse or serial fallback.

use super::*;
use qwen_llm::metal::{
    MemoryAdmissionDenied, MetalMemoryAdmissionReason, evaluate_metal_memory_admission,
};
use qwen_llm::qwen4exp_runtime::{
    Qwen4ExpDFlashAdapter, Qwen4ExpDFlashError, Qwen4ExpDFlashFinish, Qwen4ExpDFlashPacket,
    Qwen4ExpDFlashStats,
};
use qwen_llm::qwen4exp_text_session::{
    QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES, Qwen4ExpTextSessionError,
};
use qwen_llm::tokenizer::Tokenize;
use std::time::Duration;

pub(super) fn load_head(
    ctx: &MetalContext,
    prepared: &crate::drafter_policy::PreparedDrafter,
) -> anyhow::Result<MetalDFlashHead> {
    let price = prepared.priced_metal_bytes(ctx)?;
    let admission = evaluate_metal_memory_admission(
        price,
        QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    if let Some(denied) = admission.refusal() {
        return Err(denied.into());
    }
    let started = Instant::now();
    let head = prepared.load(ctx)?;
    tracing::info!(target: "qwen_diag",
        "serve: family=qwen4exp mode=dflash_cold drafter={} block_size={} head_priced_bytes={price} load_ms={:.1}",
        prepared.path().display(), head.config.block_size, started.elapsed().as_secs_f64() * 1e3);
    Ok(head)
}

fn runtime_denial(error: &Qwen4ExpRuntimeError) -> Option<&MemoryAdmissionDenied> {
    match error {
        Qwen4ExpRuntimeError::MemoryAdmission(denied)
        | Qwen4ExpRuntimeError::Session(Qwen4ExpTextSessionError::MemoryAdmission(denied)) => {
            Some(denied)
        }
        Qwen4ExpRuntimeError::Prefill { source, .. } => runtime_denial(source),
        _ => None,
    }
}

fn denial(error: &Qwen4ExpDFlashError) -> Option<&MemoryAdmissionDenied> {
    match error {
        Qwen4ExpDFlashError::Target(error) => runtime_denial(error),
        Qwen4ExpDFlashError::Packet { source, .. } => denial(source),
        _ => None,
    }
}

pub(super) fn relief_request(
    result: &Result<(), Qwen4ExpDFlashError>,
) -> Option<(MetalMemoryAdmissionReason, u64)> {
    let denied = denial(result.as_ref().err()?)?;
    let deficit = denied
        .required_bytes
        .zip(denied.signals.process_limit_remaining_bytes)
        .map_or(u64::MAX, |(required, available)| {
            required.saturating_sub(available)
        });
    Some((denied.reason(), deficit))
}

pub(super) fn map_error(error: Qwen4ExpDFlashError) -> BackendFailure {
    match denial(&error) {
        Some(denied) => {
            super::super::transport_memory::memory_refusal("Flash-Next DFlash request", denied)
                .into()
        }
        None => ServeError::server_error(format!("Flash-Next DFlash: {error}")).into(),
    }
}

fn map_start_error(error: Qwen4ExpDFlashError, abort: Option<io::Error>) -> BackendFailure {
    match abort {
        Some(error) => BackendFailure::Aborted(error),
        None => map_error(error),
    }
}

pub(super) fn generate_cold(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    head: &MetalDFlashHead,
    prompt: &[u32],
    sampler: Sampler,
    max_tokens: usize,
    stops: &[i32],
    tokenizer: &dyn Tokenize,
    sink: &mut dyn GenerationSink,
    cpu_reserve: u64,
) -> Result<(crate::GenerationResult, Qwen4ExpDFlashStats), BackendFailure> {
    let mut abort = None;
    let started = Qwen4ExpDFlashAdapter::start_with_checkpoint(
        runner,
        head,
        prompt,
        sampler,
        max_tokens,
        stops,
        cpu_reserve,
        || {
            sink.tick().map_err(|error| {
                abort = Some(error);
                Qwen4ExpRuntimeError::Checkpoint(
                    "transport aborted during Flash-Next DFlash prefill".into(),
                )
            })
        },
    );
    let (mut adapter, initial) = started.map_err(|error| map_start_error(error, abort))?;
    let mut output = PacketOutput::new(max_tokens, adapter.stats().first_token_selection);
    let result = (|| {
        output.emit(&initial, stops, tokenizer, sink)?;
        while output.stop.is_none() {
            sink.tick().map_err(BackendFailure::Aborted)?;
            let step = Instant::now();
            let packet = adapter.step().map_err(map_error)?;
            output.record_step(step.elapsed());
            output.emit(&packet, stops, tokenizer, sink)?;
        }
        let retained = adapter
            .target_position()
            .checked_sub(prompt.len())
            .ok_or_else(|| ServeError::server_error("DFlash target position precedes prompt"))?;
        output.finish(retained)
    })();
    if result.is_err() {
        // The committed packet can be ahead of the socket on a mid-packet abort.
        // Discard this request; only the outer workspace-return path resets it.
        let _ = adapter.cancel();
    }
    let stats = adapter.finish();
    result.map(|generation| (generation, stats))
}

struct PacketOutput {
    tokens: Vec<i32>,
    budget: usize,
    stop: Option<crate::StopReason>,
    started: Instant,
    selection_ms: f64,
    first_callback_ms: Option<f64>,
    transition_ms: f64,
    first_transition_ms: Option<f64>,
}

impl PacketOutput {
    fn new(budget: usize, selection: Duration) -> Self {
        Self {
            tokens: Vec::new(),
            budget,
            stop: None,
            started: Instant::now(),
            selection_ms: selection.as_secs_f64() * 1e3,
            first_callback_ms: None,
            transition_ms: 0.0,
            first_transition_ms: None,
        }
    }

    fn elapsed_ms(&self) -> f64 {
        // Start includes admission, prefill and context seeding. Decode time is
        // the measured initial selection plus wall time streaming its packets.
        self.selection_ms + self.started.elapsed().as_secs_f64() * 1e3
    }

    fn record_step(&mut self, elapsed: Duration) {
        let ms = elapsed.as_secs_f64() * 1e3;
        self.transition_ms += ms;
        self.first_transition_ms.get_or_insert(ms);
    }

    fn emit(
        &mut self,
        packet: &Qwen4ExpDFlashPacket,
        stops: &[i32],
        tokenizer: &dyn Tokenize,
        sink: &mut dyn GenerationSink,
    ) -> Result<(), BackendFailure> {
        let total = self
            .tokens
            .len()
            .checked_add(packet.tokens.len())
            .ok_or_else(|| ServeError::server_error("DFlash output count overflow"))?;
        let last_stop = packet.tokens.last().is_some_and(|id| stops.contains(id));
        let valid_finish = match packet.finish {
            None => total < self.budget && !last_stop,
            Some(Qwen4ExpDFlashFinish::Eos) => last_stop,
            Some(Qwen4ExpDFlashFinish::TokenLimit) => total == self.budget && !last_stop,
            Some(Qwen4ExpDFlashFinish::Cancelled) => false,
        };
        if self.stop.is_some()
            || packet.tokens.is_empty()
            || total > self.budget
            || !valid_finish
            || packet.tokens[..packet.tokens.len().saturating_sub(1)]
                .iter()
                .any(|id| stops.contains(id))
            || packet
                .tokens
                .iter()
                .any(|&id| id < 0 || id as u32 >= tokenizer.n_vocab())
        {
            return Err(ServeError::server_error("invalid committed DFlash packet").into());
        }
        sink.tick().map_err(BackendFailure::Aborted)?;
        for &token in &packet.tokens {
            if !stops.contains(&token) {
                let bytes = tokenizer.try_decode_piece_bytes(token).map_err(|error| {
                    ServeError::server_error(format!("decode {FAMILY} token {token}: {error}"))
                })?;
                sink.piece(&bytes)
                    .and_then(|()| sink.tick())
                    .map_err(BackendFailure::Aborted)?;
                if self.tokens.is_empty() {
                    self.first_callback_ms = Some(self.elapsed_ms());
                }
            }
            self.tokens.push(token);
        }
        self.stop = match packet.finish {
            Some(Qwen4ExpDFlashFinish::Eos) => Some(crate::StopReason::Eos),
            Some(Qwen4ExpDFlashFinish::TokenLimit) => Some(crate::StopReason::TokenLimit),
            _ => None,
        };
        Ok(())
    }

    fn finish(self, retained: usize) -> Result<crate::GenerationResult, BackendFailure> {
        if retained.checked_add(1) != Some(self.tokens.len()) {
            return Err(
                ServeError::server_error("DFlash retained inputs violate T-1 accounting").into(),
            );
        }
        let stop_reason = self
            .stop
            .ok_or_else(|| ServeError::server_error("DFlash output is not finished"))?;
        let wall_ms = self.elapsed_ms();
        Ok(crate::GenerationResult {
            tokens: self.tokens,
            wall_ms,
            first_token_selection_ms: self.selection_ms,
            first_token_ready_ms: Some(self.selection_ms),
            first_token_callback_ms: self.first_callback_ms,
            transitions: retained,
            transition_ms: self.transition_ms,
            first_transition_ms: self.first_transition_ms,
            stop_reason,
        })
    }
}

pub(super) fn log_stats(stats: &Qwen4ExpDFlashStats) {
    tracing::info!(target: "qwen_diag",
        "serve dflash: family=qwen4exp mode=dflash_cold packets={} draft_calls={} proposed={} accepted={} verified_inputs={} retained_inputs={} replay_inputs={} restores={} full_retention_packets={} target_draws={} proposal_draws={} draft_ms={:.1} checkpoint_ms={:.1} verify_ms={:.1} acceptance_ms={:.1} restore_ms={:.1} replay_ms={:.1} context_append_ms={:.1}",
        stats.packets, stats.draft_calls, stats.proposed_tokens, stats.accepted_drafts,
        stats.verified_inputs, stats.retained_inputs, stats.replay_inputs, stats.restores,
        stats.full_retention_packets, stats.target_draws, stats.proposal_draws,
        stats.draft.as_secs_f64() * 1e3, stats.checkpoint.as_secs_f64() * 1e3,
        stats.verify.as_secs_f64() * 1e3, stats.acceptance.as_secs_f64() * 1e3,
        stats.restore.as_secs_f64() * 1e3, stats.replay.as_secs_f64() * 1e3,
        stats.context_append.as_secs_f64() * 1e3);
}

#[cfg(test)]
mod tests;
