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

use super::backend::request_sampler;
use super::decode_loop;
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest};
use super::output_partition::{OutputProtocol, ToolGrammar};
use super::render_ds4;
use super::snapshot_cache::SnapshotCache;
use crate::DeepSeekV4MultigroupSelectorPlan;
use anyhow::Context as _;
use objc2_metal::MTLDevice;
use qwen_llm::deepseek_v4_metal::{
    DeepSeekV4CausalSnapshot, DeepSeekV4MetalResidency, DeepSeekV4ModelContentId,
    DeepSeekV4Session, DeepSeekV4SessionCapacity,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::sampling::Sampler;
use qwen_llm::snapshot_policy::SnapshotPolicyConfig;
use qwen_llm::tokenizer::Tokenizer;
use std::time::Instant;

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
    cache: SnapshotCache<DeepSeekV4CausalSnapshot>,
    pub(super) snapshot_cache_plan: super::SnapshotCachePlan,
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
        snapshot_cache_mib: Option<u64>,
        snapshot_policy: SnapshotPolicyConfig,
    ) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_gguf(&gguf).context("initialize DeepSeek V4 tokenizer")?;
        let vocab_size = tokenizer.n_vocab();
        let prefill_chunk_tokens = crate::deepseek_v4_prefill_chunk_tokens()?;

        let load_t0 = Instant::now();
        let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, forward_limit)
            .context("plan DeepSeek V4 residency for serve")?;
        let session_capacity = plan.session_capacity();
        let selector_plan = DeepSeekV4MultigroupSelectorPlan::new(
            selector,
            ctx.device.name().to_string(),
            session_capacity,
        )?;
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
        // Sized after load so auto budgets see the resident model.
        let snapshot_cache_plan = super::SnapshotCachePlan::resolve(
            snapshot_cache_mib,
            snapshot_policy,
            ctx.memory_signals(),
        )?;
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
            cache: SnapshotCache::new(snapshot_cache_plan.bytes, snapshot_policy),
            snapshot_cache_plan,
            model_content_id,
        })
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

    fn idle(&mut self) {
        super::log_expired_snapshots("deepseek_v4", &self.cache.sweep());
    }

    fn output_protocol(&self, request: &ServeRequest) -> OutputProtocol {
        OutputProtocol::Qwen {
            preopened_reasoning: render_ds4::preopens_reasoning(request).unwrap_or(false),
            parse_tools: !request.model_request.tools.is_empty(),
            tool_grammar: ToolGrammar::DeepSeekDsml,
        }
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
        // Sampling validation precedes tokenization, admission, residency
        // transfer, session allocation, and all model execution.
        let sampler = request_sampler(request)?;
        let tokenize_t0 = Instant::now();
        let prompt_ids = decode_loop::encode_checked(
            &self.tokenizer,
            prompt,
            false,
            self.vocab_size,
            "DeepSeek V4",
        )?;
        let tokenize_ms = tokenize_t0.elapsed().as_secs_f64() * 1e3;
        // Forward-budget admission (spec truncation:"disabled" semantics):
        // the startup budget is at most the promoted capacity.
        decode_loop::required_forwards(
            "DeepSeek V4",
            prompt_ids.len(),
            max_tokens,
            self.session_capacity.forward_limit(),
        )?;

        let residency = self.residency.take().ok_or_else(|| {
            // Unreachable unless a prior request poisoned the slot; a server
            // that can never serve again must not pretend otherwise (k3 R1.5).
            ServeError::server_error(
                "DeepSeek V4 residency slot is empty; the server can no longer serve requests",
            )
        })?;
        self.run_request(
            residency,
            &prompt_ids,
            max_tokens,
            tokenize_ms,
            sampler,
            sink,
        )
    }
}

impl DeepSeekV4Backend {
    #[allow(clippy::too_many_arguments)]
    fn run_request(
        &mut self,
        residency: DeepSeekV4MetalResidency,
        prompt_ids: &[u32],
        max_tokens: usize,
        tokenize_ms: f64,
        mut sampler: Sampler,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        // Session per request over the long-lived residency; the slot is
        // restored on every exit path below.
        let session_t0 = Instant::now();
        let mut session = match DeepSeekV4Session::new_with_model_content_id_recoverable(
            &self.ctx,
            residency,
            self.model_content_id,
        ) {
            Ok(session) => session,
            Err(failure) => {
                let (residency, error) = failure.into_parts();
                self.residency = Some(residency);
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
            prompt_ids,
            max_tokens,
            tokenize_ms,
            session_ms,
            &mut sampler,
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
        prompt_ids: &[u32],
        max_tokens: usize,
        tokenize_ms: f64,
        session_ms: f64,
        sampler: &mut Sampler,
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
        let ranges =
            crate::deepseek_v4_prefill_chunk_ranges(suffix.len(), self.prefill_chunk_tokens);
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
        let prompt_snapshot_bytes = session.current_causal_snapshot_payload_bytes();
        match prompt_snapshot_bytes {
            Ok(payload_bytes) => {
                self.capture_snapshot(session, prompt_ids, payload_bytes, "prompt")
            }
            Err(error) => tracing::warn!(
                "serve: deepseek_v4 prompt snapshot estimate failed; skipping cache capture: {error}"
            ),
        }
        let capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3;

        let logits = match crate::copy_deepseek_v4_logits(session, self.vocab_size, "prompt") {
            Ok(logits) => logits,
            Err(error) => {
                return Err(ServeError::server_error(format!("logits: {error:#}")).into());
            }
        };
        let stop_tokens = crate::deepseek_v4_generation_stops(&self.gguf, self.vocab_size)
            .map_err(|error| ServeError::server_error(error.to_string()))?;
        let ctx = &self.ctx;
        let vocab_size = self.vocab_size;
        let generation = decode_loop::decode_serial(
            decode_loop::DecodeRequest {
                family: "DeepSeek V4",
                logits,
                max_tokens,
                stop_tokens: &stop_tokens,
                vocab_size,
            },
            sampler,
            &self.tokenizer,
            sink,
            |token| {
                session.forward_token(ctx, token)?;
                crate::copy_deepseek_v4_logits(session, vocab_size, "continuing")
            },
        )?;
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
                match crate::checked_token_id(*token, vocab_size, "consumed") {
                    Ok(token) => consumed.push(token),
                    Err(error) => {
                        tracing::warn!("serve: deepseek_v4 consumed token invalid: {error}");
                        break;
                    }
                }
            }
            match session.current_causal_snapshot_payload_bytes() {
                Ok(payload_bytes) => {
                    self.capture_snapshot(session, &consumed, payload_bytes, "completed")
                }
                Err(error) => tracing::warn!(
                    "serve: deepseek_v4 completed snapshot estimate failed; skipping cache capture: {error}"
                ),
            }
        }

        tracing::info!(
            target: "qwen_diag",
            "serve phases: tokenize_ms={tokenize_ms:.1} session_ms={session_ms:.1} restore_ms={restore_ms:.1} prefill_ms={prefill_ms:.1} prompt_capture_ms={capture_ms:.1} family=deepseek_v4",
        );
        Ok(super::outcome::finish_generation(
            prompt_ids.len(),
            &generation,
            matched_tokens,
            restore_ms,
        ))
    }

    fn capture_snapshot(
        &mut self,
        session: &DeepSeekV4Session,
        tokens: &[u32],
        payload_bytes: u64,
        boundary: &'static str,
    ) {
        let Some(entry_bytes) = self.cache.strict_eligibility(tokens, payload_bytes) else {
            tracing::warn!(
                "serve: deepseek_v4 {boundary} snapshot denied by cache budget; payload_bytes={payload_bytes} cache_bytes={} cache_budget_bytes={}",
                self.cache.indexed_bytes(),
                self.cache.max_bytes(),
            );
            return;
        };
        let ctx = &self.ctx;
        let cache = &mut self.cache;
        if let Err((reason, signals)) = super::admit_snapshot_capture(
            entry_bytes,
            || ctx.memory_signals(),
            |bytes| cache.evict_for(bytes),
        ) {
            tracing::warn!(
                "serve: deepseek_v4 {boundary} snapshot denied by memory headroom; reason={reason:?} payload_bytes={payload_bytes} metal_current_bytes={} metal_recommended_bytes={} process_remaining_bytes={:?}",
                signals.current_allocated_bytes,
                signals.recommended_max_bytes,
                signals.process_limit_remaining_bytes,
            );
            return;
        }
        match session.capture_causal_snapshot() {
            Ok(snapshot) => {
                debug_assert_eq!(snapshot.payload_bytes(), payload_bytes);
                if !self
                    .cache
                    .insert_strict(tokens.to_vec(), snapshot, entry_bytes)
                {
                    tracing::warn!(
                        "serve: deepseek_v4 {boundary} snapshot rejected at strict cache insertion; request continues"
                    );
                }
            }
            Err(error) => tracing::warn!(
                "serve: deepseek_v4 {boundary} snapshot capture failed; request continues: {error}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn invalid_sampling_is_rejected_before_model_work() {
        let request = ServeRequest {
            min_p: Some(f32::NAN),
            ..ServeRequest::default()
        };
        assert!(request_sampler(&request).is_err());
    }
}
