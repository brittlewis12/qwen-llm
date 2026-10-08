use super::{Prepared, Sink, writer};
use crate::ordinary_executor::{
    self, DecodeOptions, ExecutionCancelled, TerminalReason, TokenAllocation,
};
use crate::serve::jobs::state::{Counters, JobError, Phase, StopReason};
use anyhow::{Context, Result, ensure};
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

pub(super) trait TokenEngine {
    fn forward(&mut self, token: i32, position: u32, logits: bool) -> Result<Vec<f32>>;
    fn observe(
        &mut self,
        _token: i32,
        _position: u32,
        _logits: &[f32],
        _counters: &Counters,
        _sink: &Sink,
    ) -> Result<()> {
        Ok(())
    }
}

fn observe(
    engine: &mut impl TokenEngine,
    token: i32,
    position: u32,
    logits: &[f32],
    counters: &Counters,
    sink: &Sink,
) -> Result<()> {
    if let Err(cause) = engine.observe(token, position, logits, counters, sink) {
        checkpoint(sink)?;
        tracing::error!("native readout failed after successful consumption: {cause:#}");
        sink.fail_recording("readout_failed", "Readout failed after the original forward; consumed tokens remain recorded. See server diagnostics.");
        sink.control.checkpoint()?;
    }
    Ok(())
}
impl Outcome {
    pub(crate) fn diagnostic_preparation_failed(counters: Counters, cause: anyhow::Error) -> Self {
        let mut outcome = classify(counters, cause, None);
        if outcome.reason == StopReason::ExecutionError {
            outcome.error = Some(writer::error(
                "diagnostic_preparation_failed",
                "Diagnostic staging failed before any model forward; see server diagnostics.",
            ));
        }
        outcome
    }
    /// A failure before any model forward. Only a typed memory refusal in
    /// the cause chain is labelled as one (by its kind); anything else is a
    /// preparation failure.
    pub(crate) fn preparation_failed(counters: Counters, cause: anyhow::Error) -> Self {
        let refusal = cause
            .chain()
            .find_map(|e| e.downcast_ref::<qwen_llm::metal::MemoryAdmissionDenied>())
            .map(|denied| super::super::transport_memory::refusal_kind(denied.reason).2);
        let mut outcome = classify(counters, cause, None);
        if outcome.reason == StopReason::ExecutionError {
            outcome.error = Some(match refusal {
                Some(code) => writer::error(
                    code,
                    "Native memory admission refused the job before any model forward; see server diagnostics.",
                ),
                None => writer::error(
                    "native_preparation_failed",
                    "Native preparation failed before any model forward; see server diagnostics.",
                ),
            });
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
    preparation_checkpoint(&sink.control, &sink.server)
}

pub(super) fn preparation_checkpoint(
    control: &ordinary_executor::ExecutionControl,
    server: &crate::serve::control::ExecutionGate,
) -> Result<()> {
    crate::shutdown::checkpoint()
        .map_err(|cause| anyhow::Error::new(Interrupted).context(cause))?;
    server.checkpoint()?;
    control.checkpoint()?;
    Ok(())
}

pub(crate) fn run_loaded(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    prepared: &Prepared,
    sink: &Sink,
    staged: &super::registry::Staged,
) -> Outcome {
    let setup = (|| -> Result<_> {
        checkpoint(sink)?;
        let capacity = prepared.capacity()?;
        Ok((
            loaded.create_sequence(SequenceConfig::new(capacity))?,
            loaded.gguf().stop_token_ids()?,
        ))
    })();
    let (sequence, stops) = match setup {
        Ok(value) => value,
        Err(cause) => return classify(prepared.counters(), cause, None),
    };
    let mut engine =
        match super::observe::Engine::new(loaded, tokenizer, sequence, prepared, staged, sink) {
            Ok(engine) => engine,
            Err(cause) => {
                tracing::error!("native readout preparation failed: {cause:#}");
                return Outcome::diagnostic_preparation_failed(prepared.counters(), cause);
            }
        };
    run_engine(prepared, sink, &stops, &mut engine, |token| {
        let bytes = tokenizer.try_decode_piece_bytes_exact(token)?;
        ensure!(
            bytes.len() <= super::MAX_TOKEN_PIECE_BYTES,
            "native token piece exceeds retained-record bound"
        );
        Ok(bytes.to_vec())
    })
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

#[cfg(test)]
pub(crate) fn run_tokens(
    prepared: &Prepared,
    sink: &Sink,
    stops: &[i32],
    forward: impl FnMut(i32, u32, bool) -> Result<Vec<f32>>,
    piece: impl FnMut(i32) -> Result<Vec<u8>>,
) -> Outcome {
    struct Forward<F>(F);
    impl<F: FnMut(i32, u32, bool) -> Result<Vec<f32>>> TokenEngine for Forward<F> {
        fn forward(&mut self, token: i32, position: u32, logits: bool) -> Result<Vec<f32>> {
            (self.0)(token, position, logits)
        }
    }
    run_engine(prepared, sink, stops, &mut Forward(forward), piece)
}

pub(super) fn run_engine(
    prepared: &Prepared,
    sink: &Sink,
    stops: &[i32],
    engine: &mut impl TokenEngine,
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
        let logits = ordinary_executor::prefill(
            0,
            prepared.prompt.len(),
            |position| {
                let token = prepared.prompt[position];
                checkpoint(sink)?;
                let last = position + 1 == prepared.prompt.len();
                let logits = engine.forward(token, u32::try_from(position)?, last)?;
                state.counters.consumed_prompt_tokens += 1;
                if last || (position + 1) % 16 == 0 {
                    sink.progress(Phase::Prefill, &state.counters);
                }
                observe(
                    engine,
                    token,
                    u32::try_from(position)?,
                    &logits,
                    &state.counters,
                    sink,
                )?;
                Ok((position + 1, Some(logits)))
            },
            || anyhow::anyhow!("native prefill made invalid progress"),
        )?
        .context("native prefill did not produce final logits")?;
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
                let logits = engine.forward(token, position, true)?;
                state.counters.consumed_generated_tokens += 1;
                if let Some(pending) = state.pending.take() {
                    publish_sample(sink, pending, &state.counters, true, false);
                }
                observe(engine, token, position, &logits, &state.counters, sink)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use qwen_llm::metal::{
        MemoryAdmissionDenied, MetalMemoryAdmissionReason as R, MetalMemorySignals,
    };

    fn denied(reason: R) -> MemoryAdmissionDenied {
        MemoryAdmissionDenied {
            reason,
            required_bytes: Some(1 << 30),
            signals: MetalMemorySignals {
                recommended_max_bytes: 1 << 34,
                current_allocated_bytes: 1 << 34,
                process_limit_remaining_bytes: Some(1 << 20),
            },
            working_set_headroom_bytes: Some(0),
        }
    }

    fn code(outcome: &Outcome) -> Option<&str> {
        outcome.error.as_ref().map(|e| e.code.as_str())
    }

    /// Only a typed refusal anywhere in the cause chain (through family
    /// errors and anyhow context) is labelled as a memory refusal, by kind;
    /// other preparation failures and cancellation are not.
    #[test]
    fn preparation_failures_are_labelled_by_their_typed_cause() {
        let glm = qwen_llm::glm5_next_metal::Glm5NextMetalError::MemoryAdmission {
            denied: denied(R::ProcessInsufficient),
            budget_bytes: 1 << 20,
            advice: qwen_llm::glm5_next_metal::CapacityAdvice::NotEvaluated,
        };
        let wrapped = anyhow::Error::new(glm).context("prepare GLM lens job");
        let outcome = Outcome::preparation_failed(Counters::default(), wrapped);
        assert_eq!(outcome.reason, StopReason::ExecutionError);
        assert_eq!(code(&outcome), Some("memory_admission_denied"));

        let k2 = qwen_llm::k2_horizon_runtime::K2RuntimeError::MemoryAdmission(denied(
            R::ProcessSignalUnavailable,
        ));
        let outcome = Outcome::preparation_failed(
            Counters::default(),
            anyhow::Error::new(k2).context("prepare K2 lens job"),
        );
        assert_eq!(code(&outcome), Some("memory_signal_unavailable"));

        let native =
            anyhow::Error::new(denied(R::BothInsufficient)).context("native memory admission");
        let outcome = Outcome::preparation_failed(Counters::default(), native);
        assert_eq!(code(&outcome), Some("memory_admission_denied"));

        let generic = anyhow::anyhow!("direction file is malformed").context("stage directions");
        let outcome = Outcome::preparation_failed(Counters::default(), generic);
        assert_eq!(code(&outcome), Some("native_preparation_failed"));

        let cancelled = anyhow::Error::new(crate::ordinary_executor::ExecutionCancelled);
        let outcome = Outcome::preparation_failed(Counters::default(), cancelled);
        assert_eq!(outcome.reason, StopReason::Cancelled);
        assert_eq!(code(&outcome), None);
    }
}
