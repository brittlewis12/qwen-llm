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
use super::items::{QwenTemplate, ServeError, ServeRequest};
use super::utf8::Utf8Assembler;
use anyhow::Context as _;
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::{Model, open_dflash_drafter};
use qwen_llm::metal::MetalTensor;
use qwen_llm::metal_dflash::{MetalDFlashHead, MetalDFlashSession};
use qwen_llm::runtime::LoadedModel;
use qwen_llm::sampling::{Sampler, SamplingConfig};
use qwen_llm::tokenizer::Tokenizer;
use std::io;
use std::time::Instant;

impl ServeError {
    pub(crate) fn server_error(message: impl Into<String>) -> Self {
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
    template: super::items::QwenTemplate,
    /// DFlash drafter (v0.77 speculative decode). Speculation requires
    /// captured target hidden states for every context position, which
    /// restored checkpoints do not carry — so a request uses the drafter
    /// only when it cold-prefills the whole prompt. Output is identical
    /// either way (greedy accept-prefix over an exact target verify).
    dflash_head: Option<MetalDFlashHead>,
}

impl EngineBackend {
    pub(crate) fn new(
        loaded: LoadedModel,
        model_id: String,
        default_max_tokens: usize,
        max_context_tokens: Option<usize>,
        drafter: Option<&std::path::Path>,
        template: super::items::QwenTemplate,
    ) -> anyhow::Result<Self> {
        let tokenizer = loaded.tokenizer().context("initialize serve tokenizer")?;
        let dflash_head = match drafter {
            Some(path) => {
                let t0 = Instant::now();
                let drafter_gguf = GgufFile::open(path)
                    .with_context(|| format!("open drafter {}", path.display()))?;
                qwen_llm::runtime::prefetch_opened_gguf(
                    &drafter_gguf,
                    &qwen_llm::runtime::LoadedModelConfig::default(),
                );
                let target_model = Model::from_gguf(loaded.gguf())
                    .context("parse target arch for drafter binding")?;
                let bound = open_dflash_drafter(&drafter_gguf, &target_model)
                    .with_context(|| format!("bind drafter {}", path.display()))?;
                let head = MetalDFlashHead::load(loaded.context(), &drafter_gguf, &bound)
                    .context("metal-load drafter")?;
                tracing::info!(
                    target: "qwen_diag",
                    "serve: dflash drafter loaded path={} block_size={} dflash2={} load_ms={:.1}",
                    path.display(),
                    head.config.block_size,
                    head.config.selector_top_k > 0,
                    t0.elapsed().as_secs_f64() * 1e3,
                );
                Some(head)
            }
            None => None,
        };
        Ok(Self {
            loaded,
            tokenizer,
            model_id,
            default_max_tokens,
            max_context_tokens,
            template,
            dflash_head,
        })
    }
}

fn preopens(template: QwenTemplate, request: &ServeRequest) -> bool {
    template == QwenTemplate::Qwen38
        && !request.no_thinking
        && request.reasoning_effort.as_deref() != Some("none")
}

impl GenerationBackend for EngineBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn preopens_reasoning(&self, request: &ServeRequest) -> bool {
        preopens(self.template, request)
    }

    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        let mut request = request.clone();
        request.template = self.template;
        if self.template == QwenTemplate::Qwen38 {
            match (request.no_thinking, request.reasoning_effort.as_deref()) {
                (true, Some(_)) => {
                    return Err(ServeError::invalid_request(
                        Some("reasoning.effort"),
                        "reasoning.effort cannot be combined with x_qwen.no_thinking",
                    ));
                }
                (_, None | Some("none" | "low" | "medium" | "xhigh")) => {}
                (_, Some(other)) => {
                    return Err(ServeError::invalid_request(
                        Some("reasoning.effort"),
                        format!(
                            "Qwen3.8 supports reasoning.effort none|low|medium|xhigh; got {other:?}"
                        ),
                    ));
                }
            }
        }
        Ok(super::render::render_qwen_serve_prompt(&request))
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
        // Speculation needs captured hiddens for the whole context; a
        // restore leaves earlier positions uncaptured, so the drafter only
        // runs on cold-prefilled requests (CLI parity: --drafter excludes
        // --durable-prefix-cache for the same reason).
        // Greedy-only (CLI parity: T>0 needs the maximal-coupling
        // rejection sampler), and cold-prefill-only. Decided before the
        // capture buffer is allocated so sampled requests pay nothing.
        let speculate = self.dflash_head.is_some()
            && matched_tokens == 0
            && request.temperature.unwrap_or(0.0) == 0.0;
        let mut dflash_capture: Option<(MetalTensor, usize, usize)> = None;
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
            let (logits, _span_ms) = match self.dflash_head.as_ref().filter(|_| speculate) {
                Some(head) => {
                    // One capture buffer for the whole remaining span; the
                    // drafter session consumes it below.
                    let k_layers = head.target_layer_ids.len();
                    let n_features = k_layers * self.loaded.arch().hidden_size as usize;
                    let span = prompt_ids.len() - start;
                    let dst = MetalTensor::zeros_f32(
                        self.loaded.context(),
                        vec![(span * n_features) as u64],
                    )
                    .map_err(|error| {
                        ServeError::server_error(format!("allocate drafter capture: {error:#}"))
                    })?;
                    let out = crate::prefill_span_with_capture(
                        &forward,
                        &mut sequence,
                        &mut scratch,
                        &prompt_ids[start..],
                        start,
                        &head.target_layer_ids,
                        &dst,
                    )
                    .map_err(|error| {
                        ServeError::server_error(format!("capture prefill: {error:#}"))
                    })?;
                    dflash_capture = Some((dst, span, n_features));
                    // The seeding offset below assumes the capture covers the
                    // whole context; true only because speculate ⇒ no restore
                    // ⇒ start == 0 (k3 R3).
                    debug_assert_eq!(start, 0, "speculative capture must start at position zero");
                    out
                }
                None => crate::prefill_span(
                    &forward,
                    &mut sequence,
                    &mut scratch,
                    &prompt_ids[start..end],
                    start,
                )
                .map_err(|error| ServeError::server_error(format!("prefill: {error:#}")))?,
            };
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
        let mut assembler = Utf8Assembler::new();
        let tokenizer = &self.tokenizer;
        // Speculative path: seed the drafter cross-context from the captured
        // prompt hiddens, then greedy accept-prefix over an exact target
        // verify — emitted tokens are identical to serial greedy.
        if let Some(head) = self.dflash_head.as_ref().filter(|_| speculate) {
            let capacity = sequence.position() + max_tokens + 16;
            let mut dsess = MetalDFlashSession::fresh(
                self.loaded.context(),
                head,
                self.loaded.arch().hidden_size as u64,
                self.loaded.arch().vocab_size as u64,
                capacity,
            )
            .map_err(|error| {
                ServeError::server_error(format!("allocate drafter session: {error:#}"))
            })?;
            if let Some((dst, span, n_features)) = dflash_capture.as_ref() {
                dsess
                    .append_target_ctx_columns_contiguous_now(
                        self.loaded.context(),
                        dst,
                        (sequence.position() - span) as u32,
                        *span,
                        *n_features,
                    )
                    .map_err(|error| {
                        ServeError::server_error(format!("seed drafter context: {error:#}"))
                    })?;
            }
            let result = {
                let abort = &mut abort;
                crate::generate_dflash(
                    &self.loaded,
                    &forward,
                    head,
                    dsess,
                    sequence,
                    logits,
                    max_tokens,
                    &stop_tokens,
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
                )
            };
            let result = match result {
                Ok(result) => result,
                Err(error) => {
                    return Err(match abort {
                        Some(io_error) => BackendFailure::Aborted(io_error),
                        None => {
                            ServeError::server_error(format!("dflash decode: {error:#}")).into()
                        }
                    });
                }
            };
            let generation = result.generation;
            let sequence = result.sequence;
            let stats = result.stats;
            return self.finish_generation(
                generation,
                Some(stats),
                sequence,
                prompt_ids,
                matched_tokens,
                restore_ms,
                format!(
                    "tokenize_ms={tokenize_ms:.1} alloc_ms={alloc_ms:.1} restore_ms={restore_ms:.1} prefill_ms={prefill_ms:.1} prompt_capture_ms={prompt_capture_ms:.1} decode_path=dflash"
                ),
            );
        }
        let generation = {
            let abort = &mut abort;
            crate::generate_serial(
                logits,
                max_tokens,
                &stop_tokens,
                &mut sampler,
                |token| {
                    // Exact bytes + incremental UTF-8 assembly: per-token
                    // lossy decode corrupts multibyte characters split
                    // across tokens (k3 R1.6).
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

        self.finish_generation(
            generation,
            None,
            sequence,
            prompt_ids,
            matched_tokens,
            restore_ms,
            format!("tokenize_ms={tokenize_ms:.1} alloc_ms={alloc_ms:.1} restore_ms={restore_ms:.1} prefill_ms={prefill_ms:.1} prompt_capture_ms={prompt_capture_ms:.1} decode_path=serial"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_thinking_prompt_is_headless() {
        let request = ServeRequest::default();
        assert!(preopens(QwenTemplate::Qwen38, &request));
        assert!(!preopens(
            QwenTemplate::Qwen38,
            &ServeRequest {
                no_thinking: true,
                ..request
            }
        ));
        assert!(!preopens(
            QwenTemplate::Qwen38,
            &ServeRequest {
                reasoning_effort: Some("none".into()),
                ..ServeRequest::default()
            }
        ));
        assert!(!preopens(QwenTemplate::Generic, &ServeRequest::default()));
    }
}

impl EngineBackend {
    /// Shared completion for both decode paths: completed-turn capture,
    /// phase/stats lines, and the outcome.
    #[allow(clippy::too_many_arguments)]
    fn finish_generation(
        &self,
        generation: crate::GenerationResult,
        dflash: Option<crate::DflashDecodeStats>,
        sequence: qwen_llm::runtime::Sequence,
        prompt_ids: Vec<i32>,
        matched_tokens: usize,
        restore_ms: f64,
        phases: String,
    ) -> Result<GenerationOutcome, BackendFailure> {
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
                    Err(error) => tracing::warn!("serve: completed capture failed: {error}"),
                }
            }
            Err(error) => tracing::warn!("serve: completed boundary derivation failed: {error}"),
        }

        let stop_reason = match generation.stop_reason {
            crate::StopReason::Eos => StopReason::Eos,
            crate::StopReason::TokenLimit => StopReason::TokenLimit,
        };
        let decode_tps = if generation.wall_ms > 0.0 {
            generation.tokens.len() as f64 / (generation.wall_ms / 1e3)
        } else {
            0.0
        };
        tracing::info!(target: "qwen_diag", "serve phases: {phases}");
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
            decode_tps,
        );
        if let Some(stats) = dflash {
            tracing::info!(
                target: "qwen_diag",
                "serve dflash: spec_steps={} off_steps={} accepted={}/{} alpha_backoff={} draft_ms={:.1} verify_ms={:.1}",
                stats.spec_steps,
                stats.off_steps,
                stats.accepted_drafts,
                stats.drafts_scored,
                stats.alpha_backoff,
                stats.draft_ms,
                stats.verify_ms,
            );
        }
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
