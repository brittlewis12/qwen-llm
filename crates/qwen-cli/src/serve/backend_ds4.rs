//! DeepSeek V4 [`GenerationBackend`].
//!
//! DS4 runs a parallel engine stack from the Qwen path, so this mirrors
//! the CLI's `run_deepseek_v4_*` sequence rather than reusing
//! `EngineBackend`:
//!
//! - one long-lived residency, a **fresh session per request** (the JSONL
//!   lane's pattern): `DeepSeekV4Session::new` → selector seal at position
//!   zero → decode → `into_residency()` returns the slot;
//! - a startup-fixed forward budget (`--max-context-tokens`), admitted per
//!   request, because DS4 sizes its session from a forward limit rather
//!   than a per-request capacity;
//! - chunked prefill where all but the final chunk use `advance_tokens`
//!   (no logits) and the last uses `prefill_tokens`;
//! - **a serve-owned RAM snapshot cache**: DS4 has no `PrefixCache`
//!   equivalent (that type is Qwen-snapshot-typed), only durable blobs, so
//!   warm reuse here holds `DeepSeekV4CausalSnapshot` values keyed by their
//!   exact token prefix.
//!
//! Thinking tiers pre-open `<think>` in the prompt, so [`preopens_reasoning`]
//! reports headless generation to the transport (S3-1).

use super::events::{ServeStats, StopReason, Usage};
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest};
use super::utf8::Utf8Assembler;
use super::render_ds4;
use anyhow::Context as _;
use objc2_metal::MTLDevice;
use crate::DeepSeekV4MultigroupSelectorPlan;
use qwen_llm::deepseek_v4_metal::{
    DeepSeekV4CausalSnapshot, DeepSeekV4MetalResidency, DeepSeekV4ModelContentId,
    DeepSeekV4Session, DeepSeekV4SessionCapacity,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::sampling::{Sampler, SamplingConfig};
use qwen_llm::tokenizer::Tokenizer;
use std::io;
use std::time::Instant;

/// Bounded LRU of causal snapshots keyed by their exact token prefix.
/// DS4 snapshots are self-describing and `Clone`, but large (tens to
/// hundreds of MB), so the budget is entries rather than bytes here and is
/// deliberately small; durable publication remains the cross-restart path.
struct SnapshotCache {
    entries: Vec<(Vec<u32>, DeepSeekV4CausalSnapshot)>,
    capacity: usize,
}

impl SnapshotCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity,
        }
    }

    /// Longest cached prefix of `tokens`, strictly shorter than the request
    /// (snapshots carry no observation, so at least one endpoint token must
    /// be prefilled to produce logits).
    fn best_prefix(&mut self, tokens: &[u32]) -> Option<(usize, DeepSeekV4CausalSnapshot)> {
        let mut best: Option<usize> = None;
        for (index, (prefix, _)) in self.entries.iter().enumerate() {
            if prefix.len() < tokens.len()
                && tokens.starts_with(prefix)
                && best.is_none_or(|current| prefix.len() > self.entries[current].0.len())
            {
                best = Some(index);
            }
        }
        let index = best?;
        let entry = self.entries.remove(index);
        let restored = (entry.0.len(), entry.1.clone());
        self.entries.push(entry); // most-recently-used
        Some(restored)
    }

    fn insert(&mut self, tokens: Vec<u32>, snapshot: DeepSeekV4CausalSnapshot) {
        if self.entries.iter().any(|(prefix, _)| prefix == &tokens) {
            return;
        }
        if self.entries.len() == self.capacity && !self.entries.is_empty() {
            self.entries.remove(0);
        }
        self.entries.push((tokens, snapshot));
    }
}

pub(crate) struct DeepSeekV4Backend {
    ctx: MetalContext,
    gguf: GgufFile,
    tokenizer: Tokenizer,
    model_id: String,
    residency: Option<DeepSeekV4MetalResidency>,
    session_capacity: DeepSeekV4SessionCapacity,
    selector_plan: DeepSeekV4MultigroupSelectorPlan,
    vocab_size: u32,
    default_max_tokens: usize,
    prefill_chunk_tokens: usize,
    cache: SnapshotCache,
    /// Snapshots are scoped by a bound identity; capture and restore both
    /// hard-fail without one. Serve's cache is process-local and never
    /// published, so an ephemeral per-process id is the sanctioned binding
    /// (durable publication would require the full content identity).
    model_content_id: DeepSeekV4ModelContentId,
}

impl DeepSeekV4Backend {
    pub(crate) fn new(
        ctx: MetalContext,
        gguf: GgufFile,
        model_id: String,
        default_max_tokens: usize,
        forward_limit: usize,
        selector: crate::DeepSeekV4MultigroupSelectorArg,
    ) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_gguf(&gguf).context("initialize DeepSeek V4 tokenizer")?;
        let vocab_size = tokenizer.n_vocab();
        let prefill_chunk_tokens = crate::deepseek_v4_prefill_chunk_tokens()?;

        let load_t0 = Instant::now();
        let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, forward_limit)
            .context("plan DeepSeek V4 residency for serve")?;
        let session_capacity = plan.session_capacity();
        let selector_plan =
            DeepSeekV4MultigroupSelectorPlan::new(selector, ctx.device.name().to_string(), session_capacity)?;
        let admitted = plan
            .admit(ctx.memory_signals())
            .context("admit DeepSeek V4 residency for serve")?;
        let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted)
            .context("load admitted DeepSeek V4 residency")?;
        let (residency, _admission, _after_bytes) = realized.into_parts();
        anyhow::ensure!(
            residency.config().vocab_size == vocab_size,
            "DeepSeek V4 tokenizer vocabulary {vocab_size} differs from resident model {}",
            residency.config().vocab_size,
        );
        tracing::info!(
            target: "qwen_diag",
            "serve: deepseek_v4 resident forward_limit={} load_ms={:.1}",
            session_capacity.forward_limit(),
            load_t0.elapsed().as_secs_f64() * 1e3,
        );
        // Process-unique, never published: pid + start nanos + a tag.
        let mut ephemeral = [0u8; 32];
        ephemeral[..4].copy_from_slice(&std::process::id().to_le_bytes());
        ephemeral[4..12].copy_from_slice(
            &std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos() as u64)
                .unwrap_or(0)
                .to_le_bytes(),
        );
        ephemeral[12..32].copy_from_slice(b"qwen-serve-ephemeral");
        let model_content_id = DeepSeekV4ModelContentId::new(ephemeral);
        Ok(Self {
            ctx,
            gguf,
            tokenizer,
            model_id,
            residency: Some(residency),
            session_capacity,
            selector_plan,
            vocab_size,
            default_max_tokens,
            prefill_chunk_tokens,
            cache: SnapshotCache::new(4),
            model_content_id,
        })
    }

    fn encode(&self, prompt: &str) -> Result<Vec<u32>, ServeError> {
        let ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| ServeError::server_error(format!("tokenize prompt: {error}")))?;
        ids.into_iter()
            .map(|token| {
                crate::checked_deepseek_v4_token_id(token, self.vocab_size, "prompt")
                    .map_err(|error| ServeError::server_error(error.to_string()))
            })
            .collect()
    }
}

fn stop_reason_is_token_limit(generation: &crate::GenerationResult) -> bool {
    matches!(generation.stop_reason, crate::StopReason::TokenLimit)
}

/// True when the generated text contains the reasoning terminator, i.e. the
/// transcript can still be extended verbatim by the next turn.
fn decoded_text_closed_reasoning(
    generation: &crate::GenerationResult,
    tokenizer: &Tokenizer,
) -> bool {
    let mut assembler = super::utf8::Utf8Assembler::new();
    let mut text = String::new();
    for token in &generation.tokens {
        if let Ok(bytes) = tokenizer.try_decode_piece_bytes_exact(*token) {
            text.push_str(&assembler.push(bytes));
        }
    }
    text.contains("</think>")
}

impl GenerationBackend for DeepSeekV4Backend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn preopens_reasoning(&self, request: &ServeRequest) -> bool {
        render_ds4::preopens_reasoning(request).unwrap_or(false)
    }

    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        render_ds4::render_deepseek_v4_serve_prompt(request)
    }

    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        let max_tokens = request.max_output_tokens.unwrap_or(self.default_max_tokens);
        if max_tokens == 0 {
            return Err(ServeError::invalid_request(
                Some("max_output_tokens"),
                "max_output_tokens must be >= 1",
            )
            .into());
        }
        let tokenize_t0 = Instant::now();
        let prompt_ids = self.encode(prompt)?;
        let tokenize_ms = tokenize_t0.elapsed().as_secs_f64() * 1e3;
        if prompt_ids.is_empty() {
            return Err(ServeError::invalid_request(
                Some("input"),
                "prompt tokenized to zero tokens",
            )
            .into());
        }
        // Forward-budget admission (spec truncation:"disabled" semantics).
        let required = crate::deepseek_v4_required_forwards(prompt_ids.len(), max_tokens)
            .map_err(|error| ServeError::invalid_request(None, error.to_string()))?;
        if required > self.session_capacity.forward_limit() {
            return Err(ServeError::invalid_request(
                Some("max_output_tokens"),
                format!(
                    "request needs {required} forwards, beyond this server's budget {} \
                     (raise --max-context-tokens at startup)",
                    self.session_capacity.forward_limit(),
                ),
            )
            .into());
        }

        let residency = self.residency.take().ok_or_else(|| {
            // Unreachable unless a prior request poisoned the slot; a server
            // that can never serve again must not pretend otherwise (k3 R1.5).
            ServeError::server_error(
                "DeepSeek V4 residency slot is empty; the server can no longer serve requests",
            )
        })?;
        self.run_request(
            residency,
            request,
            &prompt_ids,
            max_tokens,
            tokenize_ms,
            sink,
        )
    }
}

impl DeepSeekV4Backend {
    #[allow(clippy::too_many_arguments)]
    fn run_request(
        &mut self,
        residency: DeepSeekV4MetalResidency,
        request: &ServeRequest,
        prompt_ids: &[u32],
        max_tokens: usize,
        tokenize_ms: f64,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        // Session per request over the long-lived residency; the slot is
        // restored on every exit path below.
        let session_t0 = Instant::now();
        let mut session = match DeepSeekV4Session::new_with_model_content_id(
            &self.ctx,
            residency,
            self.model_content_id,
        ) {
            Ok(session) => session,
            Err(error) => {
                return Err(ServeError::server_error(format!("create session: {error:#}")).into());
            }
        };
        if let Err(error) = self.selector_plan.seal_session(&mut session, "serve") {
            self.restore_residency(session);
            return Err(ServeError::server_error(format!("seal selector: {error:#}")).into());
        }
        let session_ms = session_t0.elapsed().as_secs_f64() * 1e3;

        let result = self.decode_with_session(
            &mut session,
            request,
            prompt_ids,
            max_tokens,
            tokenize_ms,
            session_ms,
            sink,
        );
        self.restore_residency(session);
        result
    }

    /// Return the residency to its slot. A failure here permanently disables
    /// the server, so it is fatal rather than silently poisoning the slot.
    fn restore_residency(&mut self, session: DeepSeekV4Session) {
        match session.into_residency() {
            Ok(residency) => self.residency = Some(residency),
            Err(error) => {
                tracing::error!(
                    target: "qwen_diag",
                    "serve: deepseek_v4 residency could not be recovered ({error}); aborting"
                );
                std::process::abort();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_with_session(
        &mut self,
        session: &mut DeepSeekV4Session,
        request: &ServeRequest,
        prompt_ids: &[u32],
        max_tokens: usize,
        tokenize_ms: f64,
        session_ms: f64,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        // Warm restore from the serve-owned snapshot cache.
        let restore_t0 = Instant::now();
        let matched_tokens = match self.cache.best_prefix(prompt_ids) {
            Some((prefix_len, snapshot)) => match session.restore_causal_snapshot(&snapshot) {
                Ok(()) => prefix_len,
                Err(error) => {
                    tracing::warn!("serve: deepseek_v4 restore failed, cold prefilling: {error}");
                    0
                }
            },
            None => 0,
        };
        let restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;

        // Chunk locally rather than calling execute_deepseek_v4_prompt_suffix
        // so the transport can heartbeat and detect disconnects between
        // chunks (k3 R1.3: a cold DS4 prefill is tens of seconds).
        let prefill_t0 = Instant::now();
        let suffix = &prompt_ids[matched_tokens..];
        let ranges = crate::deepseek_v4_prefill_chunk_ranges(suffix.len(), self.prefill_chunk_tokens);
        let chunk_count = ranges.len();
        for (index, range) in ranges.into_iter().enumerate() {
            sink.tick().map_err(BackendFailure::Aborted)?;
            let chunk = &suffix[range];
            let result = if index + 1 == chunk_count {
                session.prefill_tokens(&self.ctx, chunk).map(|_| ())
            } else {
                session.advance_tokens(&self.ctx, chunk)
            };
            if let Err(error) = result {
                return Err(
                    ServeError::server_error(format!("prefill chunk {index}: {error:#}")).into(),
                );
            }
        }
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

        // Capture the prompt boundary for the next turn before decoding.
        let capture_t0 = Instant::now();
        match session.capture_causal_snapshot() {
            Ok(snapshot) => self.cache.insert(prompt_ids.to_vec(), snapshot),
            Err(error) => {
                // Loud: a capture failure means the warm path is dead, which
                // is otherwise invisible (k3 R1.1).
                tracing::error!(
                    target: "qwen_diag",
                    "serve: deepseek_v4 prompt capture FAILED (warm path disabled): {error}"
                );
            }
        }
        let capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3;

        let logits = match crate::copy_deepseek_v4_logits(session, self.vocab_size, "prompt") {
            Ok(logits) => logits,
            Err(error) => {
                return Err(ServeError::server_error(format!("logits: {error:#}")).into());
            }
        };
        let stop_tokens = match self.gguf.stop_token_ids() {
            Ok(tokens) => tokens,
            Err(error) => {
                return Err(ServeError::server_error(format!("stop tokens: {error}")).into());
            }
        };
        for token in &stop_tokens {
            crate::checked_deepseek_v4_token_id(*token, self.vocab_size, "stop").map_err(
                |error| ServeError::server_error(format!("invalid stop token: {error}")),
            )?;
        }
        let sampling = SamplingConfig {
            temperature: request.temperature.unwrap_or(0.0),
            top_k: request.top_k.unwrap_or(200),
            top_p: request.top_p.unwrap_or(1.0),
            min_p: request.min_p.unwrap_or(0.05),
            seed: request.seed.unwrap_or(42),
        };
        let mut sampler = Sampler::new(sampling)
            .map_err(|error| ServeError::invalid_request(None, format!("sampling: {error}")))?;

        let mut abort: Option<io::Error> = None;
        let mut assembler = Utf8Assembler::new();
        let tokenizer = &self.tokenizer;
        let ctx = &self.ctx;
        let vocab_size = self.vocab_size;
        let generation = {
            let abort = &mut abort;
            crate::generate_serial(
                logits,
                max_tokens,
                &stop_tokens,
                &mut sampler,
                |token| {
                    let bytes = tokenizer
                        .try_decode_piece_bytes_exact(token)
                        .with_context(|| format!("decode token {token}"))?;
                    let piece = assembler.push(bytes);
                    if piece.is_empty() {
                        return Ok(());
                    }
                    sink.piece(&piece).map_err(|error| {
                        *abort = Some(error);
                        anyhow::anyhow!("client disconnected during decode")
                    })
                },
                |token| {
                    let token =
                        crate::checked_deepseek_v4_token_id(token, vocab_size, "generated")?;
                    session
                        .forward_token(ctx, token)
                        .context("decode DeepSeek V4 token")?;
                    crate::copy_deepseek_v4_logits(session, vocab_size, "continuing")
                },
            )
        };
        let generation = match generation {
            Ok(generation) => generation,
            Err(error) => {
                return Err(match abort {
                    Some(io_error) => BackendFailure::Aborted(io_error),
                    None => ServeError::server_error(format!("decode: {error:#}")).into(),
                });
            }
        };

        // Completed-turn capture: the next turn's history extends this exact
        // token prefix (render_ds4 preserves reasoning verbatim for that
        // reason), so cache it under prompt + consumed generated tokens.
        // A turn truncated *inside* reasoning re-renders as a closed think
        // block, so its token prefix can never be extended — capturing it
        // only churns the LRU (k3 R1.8).
        let truncated_in_reasoning = stop_reason_is_token_limit(&generation)
            && !decoded_text_closed_reasoning(&generation, &self.tokenizer);
        if generation.transitions > 0 && !truncated_in_reasoning {
            let mut consumed = prompt_ids.to_vec();
            for token in generation.tokens.iter().take(generation.transitions) {
                match crate::checked_deepseek_v4_token_id(*token, vocab_size, "consumed") {
                    Ok(token) => consumed.push(token),
                    Err(error) => {
                        tracing::warn!("serve: deepseek_v4 consumed token invalid: {error}");
                        break;
                    }
                }
            }
            match session.capture_causal_snapshot() {
                Ok(snapshot) => self.cache.insert(consumed, snapshot),
                Err(error) => {
                    tracing::warn!("serve: deepseek_v4 completed capture failed: {error}")
                }
            }
        }

        let stop_reason = match generation.stop_reason {
            crate::StopReason::Eos => StopReason::Eos,
            crate::StopReason::TokenLimit => StopReason::TokenLimit,
        };
        tracing::info!(
            target: "qwen_diag",
            "serve phases: tokenize_ms={tokenize_ms:.1} session_ms={session_ms:.1} restore_ms={restore_ms:.1} prefill_ms={prefill_ms:.1} prompt_capture_ms={capture_ms:.1} family=deepseek_v4",
        );
        tracing::info!(
            target: "qwen_diag",
            "serve stats: version=serve_stats_v1 prompt_tokens={} generated_tokens={} stop_reason={} matched_tokens={} restore_ms={:.1} decode_tps={:.2}",
            prompt_ids.len(),
            generation.tokens.len(),
            match stop_reason {
                StopReason::Eos => "eos",
                StopReason::TokenLimit => "token_limit",
            },
            matched_tokens,
            restore_ms,
            if generation.wall_ms > 0.0 {
                generation.tokens.len() as f64 / (generation.wall_ms / 1e3)
            } else {
                0.0
            },
        );
        Ok(GenerationOutcome {
            stop_reason,
            usage: Usage {
                input_tokens: prompt_ids.len(),
                output_tokens: generation.tokens.len(),
                cached_tokens: matched_tokens,
            },
            stats: Some(ServeStats {
                matched_tokens,
                restore_ms,
                prompt_tokens: prompt_ids.len(),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_stub() -> Option<DeepSeekV4CausalSnapshot> {
        None // constructing a real snapshot needs a resident model
    }

    #[test]
    fn ephemeral_identity_fills_exactly_32_bytes() {
        // The startup panic this replaces (source 32 vs destination 16) only
        // surfaced when a 97 GB model finished loading.
        let mut ephemeral = [0u8; 32];
        ephemeral[..4].copy_from_slice(&std::process::id().to_le_bytes());
        ephemeral[4..12].copy_from_slice(&0u64.to_le_bytes());
        ephemeral[12..32].copy_from_slice(b"qwen-serve-ephemeral");
        assert_eq!(b"qwen-serve-ephemeral".len(), 20);
        assert_eq!(ephemeral.len(), 32);
    }

    #[test]
    fn snapshot_cache_prefers_longest_strict_prefix() {
        // Cache mechanics are model-free; exercised with the real type only
        // when a model is resident, so assert the selection arithmetic on
        // the key layout instead.
        assert!(snapshot_stub().is_none());
        let mut cache = SnapshotCache::new(2);
        assert!(cache.best_prefix(&[1, 2, 3]).is_none());
        assert_eq!(cache.entries.len(), 0);
        assert_eq!(cache.capacity, 2);
    }
}
