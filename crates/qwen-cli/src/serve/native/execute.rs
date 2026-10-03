use super::{Prepared, Sink, writer};
use crate::ordinary_executor::{
    self, DecodeOptions, ExecutionCancelled, TerminalReason, TokenAllocation,
};
use crate::serve::jobs::state::{Counters, JobError, Phase, StopReason};
use anyhow::{Result, ensure};
use qwen_llm::{
    runtime::{LoadedModel, SequenceConfig},
    sampling::Sampler,
    tokenizer::Tokenizer,
};
use serde::Serialize;
use std::borrow::Cow;
use std::time::Instant;

pub(crate) struct Outcome {
    pub(crate) reason: StopReason,
    pub(crate) counters: Counters,
    pub(crate) error: Option<JobError>,
    pub(crate) wall_ms: Option<f64>,
}
impl Outcome {
    pub(crate) fn preparation_failed(counters: Counters, cause: anyhow::Error) -> Self {
        let mut outcome = classify(counters, cause, None);
        if outcome.reason == StopReason::ExecutionError {
            outcome.error = Some(writer::error(
                "memory_admission_denied",
                "Native preparation failed before any model forward; see server diagnostics.",
            ));
        }
        outcome
    }
    pub(super) fn interrupted(counters: Counters) -> Self {
        Self {
            reason: StopReason::ServerRestart,
            counters,
            error: Some(JobError::restart()),
            wall_ms: None,
        }
    }
    pub(crate) fn failed(counters: Counters, code: &str, message: &str) -> Self {
        Self {
            reason: StopReason::ExecutionError,
            counters,
            error: Some(writer::error(code, message)),
            wall_ms: None,
        }
    }
    pub(super) fn state_name(&self) -> &'static str {
        match self.reason {
            StopReason::StopToken | StopReason::TokenLimit => "completed",
            StopReason::Cancelled => "cancelled",
            StopReason::ServerRestart => "interrupted",
            StopReason::ExecutionError => "failed",
        }
    }
}

#[derive(Debug)]
struct Interrupted;
impl std::fmt::Display for Interrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("native generation interrupted by shutdown")
    }
}
impl std::error::Error for Interrupted {}

pub(crate) fn checkpoint(sink: &Sink) -> Result<()> {
    crate::shutdown::checkpoint()
        .map_err(|cause| anyhow::Error::new(Interrupted).context(cause))?;
    sink.server.checkpoint()?;
    sink.control.checkpoint()?;
    Ok(())
}

pub(crate) fn run_loaded(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    prepared: &Prepared,
    sink: &Sink,
) -> Outcome {
    let setup = (|| -> Result<_> {
        checkpoint(sink)?;
        let capacity = prepared.capacity()?;
        Ok((
            loaded.create_sequence(SequenceConfig::new(capacity))?,
            loaded.gguf().stop_token_ids()?,
        ))
    })();
    let (mut sequence, stops) = match setup {
        Ok(value) => value,
        Err(cause) => return classify(prepared.counters(), cause, None),
    };
    let forward = loaded.forward();
    run_tokens(
        prepared,
        sink,
        &stops,
        |token, position, logits| {
            sequence.check_position(position as usize)?;
            sequence.ensure_can_append(1)?;
            let output = ordinary_executor::post_block_forward(
                &forward,
                unsafe { sequence.metal_session_mut() },
                token,
                position,
                logits,
                None,
                &[],
            )?;
            sequence.advance_by(1)?;
            Ok(output)
        },
        |token| {
            let bytes = tokenizer.try_decode_piece_bytes_exact(token)?;
            ensure!(
                bytes.len() <= super::MAX_TOKEN_PIECE_BYTES,
                "native token piece exceeds retained-record bound"
            );
            Ok(bytes.to_vec())
        },
    )
}

fn classify(counters: Counters, cause: anyhow::Error, wall_ms: Option<f64>) -> Outcome {
    let mut outcome = if cause.is::<ExecutionCancelled>() {
        Outcome {
            reason: StopReason::Cancelled,
            counters,
            error: None,
            wall_ms,
        }
    } else if cause.is::<Interrupted>() || cause.is::<crate::serve::control::ServerStopped>() {
        Outcome::interrupted(counters)
    } else {
        tracing::error!("native generation failed: {cause:#}");
        Outcome::failed(
            counters,
            "execution_failed",
            "Native execution failed; see server diagnostics.",
        )
    };
    outcome.wall_ms = wall_ms;
    outcome
}

struct Pending {
    index: usize,
    token: i32,
    bytes: Option<Vec<u8>>,
}

fn publish_sample(
    sink: &Sink,
    pending: Pending,
    counters: &Counters,
    consumed: bool,
    attempted: bool,
) {
    #[derive(Serialize)]
    struct Record<'a> {
        kind: &'static str,
        index: usize,
        token_id: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        piece_bytes: Option<&'a [u8]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<Cow<'a, str>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        consumed: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<&'static str>,
    }
    let unresolved = attempted || pending.bytes.is_none();
    sink.record(
        Record {
            kind: if unresolved {
                "unresolved_sample"
            } else {
                "sampled_token"
            },
            index: pending.index,
            token_id: pending.token,
            piece_bytes: pending.bytes.as_deref(),
            text: if unresolved {
                None
            } else {
                pending.bytes.as_deref().map(String::from_utf8_lossy)
            },
            consumed: (!unresolved).then_some(consumed),
            reason: if attempted {
                Some("forward_failed_consumption_unknown")
            } else if pending.bytes.is_none() {
                Some("token_piece_unavailable_not_consumed")
            } else {
                None
            },
        },
        Phase::Decode,
        counters,
    );
}

pub(crate) fn run_tokens(
    prepared: &Prepared,
    sink: &Sink,
    stops: &[i32],
    mut forward: impl FnMut(i32, u32, bool) -> Result<Vec<f32>>,
    mut piece: impl FnMut(i32) -> Result<Vec<u8>>,
) -> Outcome {
    struct State {
        sampler: Sampler,
        counters: Counters,
        pending: Option<Pending>,
        attempted: bool,
    }
    let start = Instant::now();
    let sampler = match Sampler::new(prepared.sampling) {
        Ok(value) => value,
        Err(cause) => return classify(prepared.counters(), cause.into(), None),
    };
    let mut state = State {
        sampler,
        counters: prepared.counters(),
        pending: None,
        attempted: false,
    };
    let result = (|| -> Result<TerminalReason> {
        ensure!(
            !prepared.prompt.is_empty() && prepared.max_tokens > 0,
            "empty native generation"
        );
        let mut logits = Vec::new();
        for (position, &token) in prepared.prompt.iter().enumerate() {
            checkpoint(sink)?;
            let last = position + 1 == prepared.prompt.len();
            logits = forward(token, u32::try_from(position)?, last)?;
            state.counters.consumed_prompt_tokens += 1;
            if last || (position + 1) % 16 == 0 {
                sink.progress(Phase::Prefill, &state.counters);
            }
        }
        let summary = ordinary_executor::decode(
            logits,
            DecodeOptions {
                max_tokens: prepared.max_tokens,
                stop_tokens: stops,
                allocation: TokenAllocation::Incremental,
            },
            &mut state,
            || checkpoint(sink),
            |state, logits| {
                let token = state.sampler.sample(logits)?.token;
                let index = usize::try_from(state.counters.sampled_tokens)?;
                state.counters.sampled_tokens += 1;
                state.pending = Some(Pending {
                    index,
                    token,
                    bytes: None,
                });
                state.attempted = false;
                let bytes = piece(token)?;
                ensure!(
                    bytes.len() <= super::MAX_TOKEN_PIECE_BYTES,
                    "native token piece exceeds retained-record bound"
                );
                state.pending.as_mut().expect("pending sample").bytes = Some(bytes);
                Ok(token)
            },
            |_| Ok(()),
            |state, token| {
                let position =
                    state.counters.prompt_tokens + state.counters.consumed_generated_tokens;
                let position = u32::try_from(position)?;
                state.attempted = true;
                let logits = forward(token, position, true)?;
                state.counters.consumed_generated_tokens += 1;
                if let Some(pending) = state.pending.take() {
                    publish_sample(sink, pending, &state.counters, true, false);
                }
                Ok(logits)
            },
        )?;
        Ok(summary.stop_reason)
    })();
    let wall_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
    let outcome = match result {
        Ok(reason) => Outcome {
            reason: match reason {
                TerminalReason::StopToken => StopReason::StopToken,
                TerminalReason::TokenLimit => StopReason::TokenLimit,
            },
            counters: state.counters.clone(),
            error: None,
            wall_ms,
        },
        Err(cause) => classify(state.counters.clone(), cause, wall_ms),
    };
    if let Some(pending) = state.pending.take() {
        publish_sample(sink, pending, &state.counters, false, state.attempted);
    }
    outcome
}
