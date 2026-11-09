//! Token-at-a-time form of [`decode_serial`](super::decode_loop::decode_serial)
//! for owner-side batch schedulers, which advances several requests one token per
//! forward and therefore cannot hand any of them a closure-driven loop.
//!
//! Each call reproduces the same `ordinary_executor::decode` lifecycle: process
//! checkpoint, sample, retain the token, stop on a stop token (retained, never
//! published), publish the piece and tick, stop at the token limit (published,
//! never consumed), checkpoint, then hand the token to the caller to consume.
//! [`DecodeCursor::on_logits`] completes the transition. Errors map exactly as
//! `decode_serial` maps them: a sink failure is `Aborted`, anything else is a
//! `500` "{family} decode: ..." server error. The equivalence is pinned by
//! `tests::cursor_matches_decode_serial_*` against scripted logits.
use super::decode_loop::DecodeRequest;
use super::http::{BackendFailure, GenerationSink};
use super::items::ServeError;
use anyhow::Context as _;
use qwen_llm::sampling::{SampledToken, Sampler, SamplingError};
use qwen_llm::tokenizer::Tokenize;
use std::time::Instant;

pub(crate) enum Advance {
    /// The token was selected and published; consume it and call
    /// [`DecodeCursor::on_logits`] with the next logits.
    Forward(u32),
    Done(crate::GenerationResult),
}

#[cfg(test)]
mod gpu_burst_tests {
    use super::*;
    use qwen_llm::metal::GpuTokenLoopStats;
    fn cursor(max: usize) -> DecodeCursor {
        DecodeCursor::new(DecodeRequest {
            family: "test",
            logits: vec![0.; 3],
            max_tokens: max,
            stop_tokens: &[2],
            vocab_size: 3,
        })
        .unwrap()
    }
    fn stats(n: usize, eos: bool) -> GpuTokenLoopStats {
        GpuTokenLoopStats {
            sampled_tokens: n,
            forwarded_tokens: n - 1,
            emitted_tokens: n - usize::from(eos),
            stopped_on_eos: eos,
            first_token_ready_ms: Some(0.1),
            first_token_callback_ms: (!eos || n > 1).then_some(0.2),
            transition_ms: (n - 1) as f64,
            first_transition_ms: (n > 1).then_some(1.),
            ..Default::default()
        }
    }
    #[test]
    fn bursts_keep_the_unconsumed_tail_across_native_resume_and_limit() {
        let mut c = cursor(6);
        assert_eq!(c.available_logits(), Some(&[0.; 3][..]));
        assert!(matches!(
            c.append_gpu_burst(&[1, 0, 1], stats(3, false), 0.5),
            Advance::Forward(1)
        ));
        assert_eq!(c.completed_transitions(), 2);
        assert!(c.available_logits().is_none());
        c.on_logits(Ok(vec![0.; 3]), 4., || Ok(())).unwrap();
        assert_eq!(c.available_logits(), Some(&[0.; 3][..]));
        let Advance::Done(g) = c.append_gpu_burst(&[0, 1, 0], stats(3, false), 9.) else {
            panic!("limit not finished");
        };
        assert_eq!(g.tokens, vec![1, 0, 1, 0, 1, 0]);
        assert_eq!(g.transitions, 5);
        assert_eq!(g.transition_ms, 8.);
        assert_eq!(g.first_token_ready_ms, Some(0.6));
        assert_eq!(g.first_token_callback_ms, Some(0.7));
        assert!(matches!(g.stop_reason, crate::StopReason::TokenLimit));
    }
    #[test]
    fn flush_yield_and_eos_consume_no_extra_token() {
        let mut c = cursor(8);
        assert!(matches!(
            c.append_gpu_burst(&[1], stats(1, false), 0.),
            Advance::Forward(1)
        ));
        assert_eq!(c.transitions, 0);
        c.on_logits(Ok(vec![0.; 3]), 2., || Ok(())).unwrap();
        let Advance::Done(g) = c.append_gpu_burst(&[2], stats(1, true), 4.) else {
            panic!("EOS not finished");
        };
        assert_eq!(g.tokens, vec![1, 2]);
        assert_eq!(g.transitions, 1);
        assert!(matches!(g.stop_reason, crate::StopReason::Eos));
        let mut c = cursor(8);
        let Advance::Done(g) = c.append_gpu_burst(&[2], stats(1, true), 0.) else {
            panic!("EOS not finished");
        };
        assert_eq!(g.transitions, 0);
        assert!(g.first_token_callback_ms.is_none());
    }
}

pub(crate) struct DecodeCursor {
    family: &'static str,
    logits: Vec<f32>,
    max_tokens: usize,
    stop_tokens: Vec<i32>,
    vocab_size: u32,
    tokens: Vec<i32>,
    wall_t0: Instant,
    first_token_selection_ms: Option<f64>,
    first_token_ready_ms: Option<f64>,
    first_token_callback_ms: Option<f64>,
    transitions: usize,
    transition_ms: f64,
    first_transition_ms: Option<f64>,
    awaiting_logits: bool,
}

impl DecodeCursor {
    /// A completed head exists only before selection, including after native resume.
    pub(crate) fn available_logits(&self) -> Option<&[f32]> {
        (!self.awaiting_logits && !self.logits.is_empty()).then_some(&self.logits)
    }
    pub(crate) fn remaining_tokens(&self) -> usize {
        self.max_tokens - self.tokens.len()
    }
    pub(crate) fn elapsed_ms(&self) -> f64 {
        self.wall_t0.elapsed().as_secs_f64() * 1e3
    }
    pub(crate) fn completed_transitions(&self) -> usize {
        self.transitions
    }

    /// A completed GPU burst has streamed these tokens already. Its last
    /// selection stays unconsumed when yielding back to native batch decode.
    pub(crate) fn append_gpu_burst(
        &mut self,
        tokens: &[i32],
        stats: qwen_llm::metal::GpuTokenLoopStats,
        offset_ms: f64,
    ) -> Advance {
        assert!(!self.awaiting_logits && !tokens.is_empty());
        assert_eq!(tokens.len(), stats.sampled_tokens);
        assert_eq!(stats.forwarded_tokens, tokens.len() - 1);
        assert!(tokens.len() <= self.remaining_tokens());
        self.first_token_selection_ms
            .get_or_insert(stats.first_token_selection_ms);
        if self.first_token_ready_ms.is_none() {
            self.first_token_ready_ms = stats.first_token_ready_ms.map(|t| t + offset_ms);
        }
        if self.first_token_callback_ms.is_none() {
            self.first_token_callback_ms = stats.first_token_callback_ms.map(|t| t + offset_ms);
        }
        if self.first_transition_ms.is_none() {
            self.first_transition_ms = stats.first_transition_ms;
        }
        self.transition_ms += stats.transition_ms;
        self.transitions += stats.forwarded_tokens;
        self.tokens.extend_from_slice(tokens);
        self.logits.clear();
        if stats.stopped_on_eos {
            return Advance::Done(self.finish(crate::StopReason::Eos));
        }
        if self.tokens.len() == self.max_tokens {
            return Advance::Done(self.finish(crate::StopReason::TokenLimit));
        }
        self.awaiting_logits = true;
        Advance::Forward(*tokens.last().unwrap() as u32)
    }
    /// Starts the decode wall clock, as `decode_serial` does on entry.
    pub(crate) fn new(request: DecodeRequest<'_>) -> Result<Self, BackendFailure> {
        if request.max_tokens == 0 {
            return Err(ServeError::invalid_request(
                Some("max_output_tokens"),
                "max_output_tokens must be >= 1",
            )
            .into());
        }
        Ok(Self {
            family: request.family,
            logits: request.logits,
            max_tokens: request.max_tokens,
            stop_tokens: request.stop_tokens.to_vec(),
            vocab_size: request.vocab_size,
            tokens: Vec::with_capacity(request.max_tokens),
            wall_t0: Instant::now(),
            first_token_selection_ms: None,
            first_token_ready_ms: None,
            first_token_callback_ms: None,
            transitions: 0,
            transition_ms: 0.0,
            first_transition_ms: None,
            awaiting_logits: false,
        })
    }

    fn server(&self, error: anyhow::Error) -> BackendFailure {
        ServeError::server_error(format!("{} decode: {error:#}", self.family)).into()
    }

    /// Select the next token from the current logits and publish it.
    /// `presampled`, when given, is `sampler.sample(logits)` already applied
    /// (by the batch scheduler's sampling pool) and stands in for that call.
    pub(crate) fn advance(
        &mut self,
        sampler: &mut Sampler,
        presampled: Option<Result<SampledToken, SamplingError>>,
        tokenizer: &dyn Tokenize,
        sink: &mut dyn GenerationSink,
        mut checkpoint: impl FnMut() -> anyhow::Result<()>,
    ) -> Result<Advance, BackendFailure> {
        assert!(!self.awaiting_logits, "decode cursor awaits a transition");
        checkpoint().map_err(|error| self.server(error))?;
        let selection_t0 = Instant::now();
        let sampled = presampled.unwrap_or_else(|| sampler.sample(&self.logits));
        let token = match sampled {
            Ok(sampled) => sampled.token,
            Err(error) => return Err(self.server(error.into())),
        };
        self.first_token_selection_ms
            .get_or_insert_with(|| selection_t0.elapsed().as_secs_f64() * 1e3);
        let wall_t0 = self.wall_t0;
        self.first_token_ready_ms
            .get_or_insert_with(|| wall_t0.elapsed().as_secs_f64() * 1e3);
        self.tokens.push(token);
        if self.stop_tokens.contains(&token) {
            return Ok(Advance::Done(self.finish(crate::StopReason::Eos)));
        }
        let family = self.family;
        let bytes = tokenizer
            .try_decode_piece_bytes(token)
            .with_context(|| format!("decode {family} token {token}"))
            .map_err(|error| self.server(error))?;
        sink.piece(&bytes)
            .and_then(|()| sink.tick())
            .map_err(BackendFailure::Aborted)?;
        self.first_token_callback_ms
            .get_or_insert_with(|| wall_t0.elapsed().as_secs_f64() * 1e3);
        if self.tokens.len() == self.max_tokens {
            return Ok(Advance::Done(self.finish(crate::StopReason::TokenLimit)));
        }
        checkpoint().map_err(|error| self.server(error))?;
        let token = crate::checked_token_id(token, self.vocab_size, "generated")
            .map_err(|error| self.server(error))?;
        self.awaiting_logits = true;
        Ok(Advance::Forward(token))
    }

    /// Complete the transition of the token [`Self::advance`] returned.
    /// `elapsed_ms` is the caller's measured forward time.
    pub(crate) fn on_logits(
        &mut self,
        logits: anyhow::Result<Vec<f32>>,
        elapsed_ms: f64,
        mut checkpoint: impl FnMut() -> anyhow::Result<()>,
    ) -> Result<(), BackendFailure> {
        assert!(self.awaiting_logits, "no transition is pending");
        self.awaiting_logits = false;
        let family = self.family;
        self.logits = logits
            .with_context(|| format!("forward {family} token"))
            .map_err(|error| self.server(error))?;
        checkpoint().map_err(|error| self.server(error))?;
        self.transition_ms += elapsed_ms;
        self.first_transition_ms.get_or_insert(elapsed_ms);
        self.transitions += 1;
        Ok(())
    }

    fn finish(&mut self, stop_reason: crate::StopReason) -> crate::GenerationResult {
        crate::GenerationResult {
            tokens: std::mem::take(&mut self.tokens),
            wall_ms: self.wall_t0.elapsed().as_secs_f64() * 1e3,
            first_token_selection_ms: self.first_token_selection_ms.unwrap_or(0.0),
            first_token_ready_ms: self.first_token_ready_ms,
            first_token_callback_ms: self.first_token_callback_ms,
            transitions: self.transitions,
            transition_ms: self.transition_ms,
            first_transition_ms: self.first_transition_ms,
            stop_reason,
        }
    }
}
