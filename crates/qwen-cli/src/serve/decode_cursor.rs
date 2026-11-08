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
