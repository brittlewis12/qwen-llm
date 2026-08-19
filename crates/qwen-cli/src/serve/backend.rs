//! Engine-backed [`GenerationBackend`]: resident model, RAM prefix cache
//! with dual-boundary capture, chunked prefill with cancellation ticks,
//! and the serial decode loop streaming pieces into the transport sink.
//!
//! Residency is the TTFT mechanism (S0 F2); dual capture — prompt boundary
//! and completed turn — makes the server's next-turn path independent of
//! client echo policy (S0 F1, SERVE.md review R2: RAM always, durable
//! stays on the existing CLI shadowing policy and is out of this slice).

use super::events::{ServeStats, StopReason, Usage};
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest};
use anyhow::Context as _;
use qwen_llm::runtime::LoadedModel;
use qwen_llm::sampling::{Sampler, SamplingConfig};
use qwen_llm::tokenizer::Tokenizer;
use std::io;
use std::time::Instant;

impl ServeError {
    fn server_error(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            error_type: "server_error",
            code: None,
            param: None,
            message: message.into(),
        }
    }
}

pub(crate) struct EngineBackend {
    loaded: LoadedModel,
    tokenizer: Tokenizer,
    model_id: String,
    default_max_tokens: usize,
    max_context_tokens: Option<usize>,
}

impl EngineBackend {
    pub(crate) fn new(
        loaded: LoadedModel,
        model_id: String,
        default_max_tokens: usize,
        max_context_tokens: Option<usize>,
    ) -> anyhow::Result<Self> {
        let tokenizer = loaded.tokenizer().context("initialize serve tokenizer")?;
        Ok(Self {
            loaded,
            tokenizer,
            model_id,
            default_max_tokens,
            max_context_tokens,
        })
    }
}

impl GenerationBackend for EngineBackend {
    fn model_id(&self) -> &str {
        &self.model_id
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
        // Rendered ChatML carries its own special-token markers; matches the
        // legacy messages path (prompt_add_special_tokens = false).
        let tokenize_t0 = Instant::now();
        let prompt_ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| ServeError::server_error(format!("tokenize prompt: {error}")))?;
        let tokenize_ms = tokenize_t0.elapsed().as_secs_f64() * 1e3;
        if prompt_ids.is_empty() {
            return Err(ServeError::invalid_request(
                Some("input"),
                "prompt tokenized to zero tokens",
            )
            .into());
        }

        // Context admission fails closed (S0 F3 == spec truncation:"disabled").
        let min_capacity = prompt_ids
            .len()
            .checked_add(max_tokens)
            .and_then(|value| value.checked_add(16))
            .ok_or_else(|| ServeError::server_error("sequence capacity overflow"))?;
        let capacity = self.max_context_tokens.unwrap_or(min_capacity);
        if capacity < prompt_ids.len() + max_tokens {
            return Err(ServeError::invalid_request(
                Some("max_output_tokens"),
                format!(
                    "max context {capacity} is smaller than prompt {} + generation {max_tokens}",
                    prompt_ids.len(),
                ),
            )
            .into());
        }

        let alloc_t0 = Instant::now();
        let allocated = crate::allocate_prefill_request_state(
            &self.loaded,
            crate::PrefillChunkArg::Auto,
            prompt_ids.len(),
            capacity,
            true,
        )
        .map_err(|error| ServeError::server_error(format!("allocate request state: {error:#}")))?;
        let alloc_ms = alloc_t0.elapsed().as_secs_f64() * 1e3;
        let chunk = allocated.chunk;
        let mut scratch = allocated.scratch;
        let mut sequence = allocated.sequence;
        let forward = self.loaded.forward();

        // RAM prefix cache restore (dual-boundary entries from prior turns).
        let restore_t0 = Instant::now();
        let restore = self
            .loaded
            .restore_cached_prefix(&mut sequence, &prompt_ids)
            .map_err(|error| ServeError::server_error(format!("prefix restore: {error:#}")))?;
        let restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
        let matched_tokens = restore
            .as_ref()
            .map_or(0, |restore| restore.matched_prefix_len);
        let mut prompt_logits = restore
            .as_ref()
            .filter(|restore| restore.exact)
            .and_then(|restore| restore.exact_final_logits.clone());

        // Chunked prefill of whatever the restore left; tick between chunks
        // carries heartbeats and surfaces client disconnects (cancellation).
        // Tails at or below SERIAL_TAIL_THRESHOLD decode token-by-token: the
        // matrix prefill path costs ~500 ms per call regardless of span size
        // (gate-2 phase measurement), while single_token runs ~10 ms/token.
        // Same determinism class as chunk-boundary choice (F7).
        const SERIAL_TAIL_THRESHOLD: usize = 48;
        let prefill_t0 = Instant::now();
        while sequence.position() < prompt_ids.len() {
            sink.tick().map_err(BackendFailure::Aborted)?;
            let start = sequence.position();
            let remaining = prompt_ids.len() - start;
            if remaining <= SERIAL_TAIL_THRESHOLD {
                for (offset, &token) in prompt_ids[start..].iter().enumerate() {
                    let position = start + offset;
                    let logits = forward
                        .single_token(
                            token,
                            u32::try_from(position)
                                .map_err(|_| ServeError::server_error("position overflow"))?,
                            unsafe { sequence.metal_session_mut() },
                        )
                        .map_err(|error| {
                            ServeError::server_error(format!("serial tail prefill: {error:#}"))
                        })?;
                    sequence
                        .advance_by(1)
                        .map_err(|error| ServeError::server_error(format!("advance: {error:#}")))?;
                    prompt_logits = Some(logits);
                }
                break;
            }
            let end = prompt_ids.len().min(start + chunk);
            let (logits, _span_ms) = crate::prefill_span(
                &forward,
                &mut sequence,
                &mut scratch,
                &prompt_ids[start..end],
                start,
            )
            .map_err(|error| ServeError::server_error(format!("prefill: {error:#}")))?;
            prompt_logits = Some(logits);
        }
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;
        let logits = prompt_logits
            .ok_or_else(|| ServeError::server_error("prefill produced no prompt logits"))?;

        // Prompt-boundary capture into the RAM cache (skip when this exact
        // prompt was already an exact hit).
        let capture_t0 = Instant::now();
        if !restore.as_ref().is_some_and(|restore| restore.exact) {
            match self.loaded.prepare_checkpoint_boundary(
                &sequence,
                prompt_ids.clone(),
                None,
                Some(logits.clone()),
            ) {
                Ok(prepared) => {
                    if let Err(error) = self.loaded.cache_prepared_checkpoint(&prepared) {
                        tracing::warn!("serve: prompt-boundary cache insert failed: {error}");
                    }
                }
                Err(error) => {
                    tracing::warn!("serve: prompt-boundary capture failed: {error}");
                }
            }
        }

        let prompt_capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3;
        let sampling = SamplingConfig {
            temperature: request.temperature.unwrap_or(0.0),
            top_k: request.top_k.unwrap_or(200),
            top_p: request.top_p.unwrap_or(1.0),
            min_p: request.min_p.unwrap_or(0.05),
            seed: request.seed.unwrap_or(42),
        };
        let mut sampler = Sampler::new(sampling)
            .map_err(|error| ServeError::invalid_request(None, format!("sampling: {error}")))?;
        let stop_tokens = self
            .loaded
            .gguf()
            .stop_token_ids()
            .map_err(|error| ServeError::server_error(format!("stop tokens: {error}")))?;

        let mut abort: Option<io::Error> = None;
        let tokenizer = &self.tokenizer;
        let generation = {
            let abort = &mut abort;
            crate::generate_serial(
                logits,
                max_tokens,
                &stop_tokens,
                &mut sampler,
                |token| {
                    let piece = tokenizer.decode_piece(token);
                    sink.piece(&piece).map_err(|error| {
                        *abort = Some(error);
                        anyhow::anyhow!("client disconnected during decode")
                    })
                },
                |token| {
                    let position = sequence.position();
                    let next = forward
                        .single_token(
                            token,
                            u32::try_from(position).context("position does not fit u32")?,
                            unsafe { sequence.metal_session_mut() },
                        )
                        .context("decode token")?;
                    sequence.advance_by(1)?;
                    Ok(next)
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

        // Completed-turn capture into the RAM cache (valid for both stop
        // reasons; a truncated-reasoning echo won't reproduce these bytes —
        // S0 F4 caveat — but the entry is harmless and prompt-boundary
        // remains available).
        match crate::derive_completed_checkpoint_boundary(
            prompt_ids.len(),
            &generation.tokens,
            generation.transitions,
            sequence.position(),
        ) {
            Ok(boundary) => {
                let pending_token = boundary.pending_token;
                let consumed = boundary.consumed_tokens(&prompt_ids, &generation.tokens);
                match self.loaded.prepare_checkpoint_boundary(
                    &sequence,
                    consumed,
                    Some(pending_token),
                    None,
                ) {
                    Ok(prepared) => {
                        if let Err(error) = self.loaded.cache_prepared_checkpoint(&prepared) {
                            tracing::warn!("serve: completed cache insert failed: {error}");
                        }
                    }
                    Err(error) => {
                        tracing::warn!("serve: completed capture failed: {error}");
                    }
                }
            }
            Err(error) => tracing::warn!("serve: completed boundary derivation failed: {error}"),
        }

        let stop_reason = match generation.stop_reason {
            crate::StopReason::Eos => StopReason::Eos,
            crate::StopReason::TokenLimit => StopReason::TokenLimit,
        };
        tracing::info!(
            target: "qwen_diag",
            "serve phases: tokenize_ms={tokenize_ms:.1} alloc_ms={alloc_ms:.1} restore_ms={restore_ms:.1} prefill_ms={prefill_ms:.1} prompt_capture_ms={prompt_capture_ms:.1}",
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
