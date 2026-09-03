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
use super::events::{ServeStats, StopReason, Usage};
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest};
use super::output_partition::{GenerationEnd, OutputProtocol};
use super::render_ds4;
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
use qwen_llm::tokenizer::Tokenizer;
use std::io;
use std::sync::Arc;
use std::time::Instant;

struct SnapshotCacheEntry<V> {
    prefix: Vec<u32>,
    value: Arc<V>,
    bytes: u64,
}

/// Byte-bounded LRU keyed by exact token prefixes. Values are shared on
/// lookup so restoring a large snapshot does not clone its state arenas.
struct SnapshotCache<V> {
    entries: Vec<SnapshotCacheEntry<V>>,
    indexed_bytes: u64,
    max_bytes: u64,
}

impl<V> SnapshotCache<V> {
    fn new(max_bytes: u64) -> Self {
        Self {
            entries: Vec::new(),
            indexed_bytes: 0,
            max_bytes,
        }
    }

    /// Longest cached prefix of `tokens`, strictly shorter than the request
    /// (snapshots carry no observation, so at least one endpoint token must
    /// be prefilled to produce logits).
    fn best_prefix(&mut self, tokens: &[u32]) -> Option<(usize, Arc<V>)> {
        let mut best: Option<usize> = None;
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.prefix.len() < tokens.len()
                && tokens.starts_with(&entry.prefix)
                && best
                    .is_none_or(|current| entry.prefix.len() > self.entries[current].prefix.len())
            {
                best = Some(index);
            }
        }
        let index = best?;
        let entry = self.entries.remove(index);
        let restored = (entry.prefix.len(), Arc::clone(&entry.value));
        self.entries.push(entry); // most-recently-used
        Some(restored)
    }

    fn entry_bytes(prefix_len: usize, value_bytes: u64) -> Option<u64> {
        value_bytes.checked_add((prefix_len as u64).checked_mul(size_of::<u32>() as u64)?)
    }

    fn strict_eligibility(&self, tokens: &[u32], value_bytes: u64) -> Option<u64> {
        if self.entries.iter().any(|entry| entry.prefix == tokens) {
            return None;
        }
        let bytes = Self::entry_bytes(tokens.len(), value_bytes)?;
        (bytes <= self.max_bytes).then_some(bytes)
    }

    fn insert_strict(&mut self, tokens: Vec<u32>, value: V, bytes: u64) -> bool {
        if bytes > self.max_bytes || self.entries.iter().any(|entry| entry.prefix == tokens) {
            return false;
        }
        while self.indexed_bytes.saturating_add(bytes) > self.max_bytes {
            let evicted = self.entries.remove(0);
            self.indexed_bytes = self.indexed_bytes.saturating_sub(evicted.bytes);
        }
        self.entries.push(SnapshotCacheEntry {
            prefix: tokens,
            value: Arc::new(value),
            bytes,
        });
        self.indexed_bytes += bytes;
        true
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
    cache: SnapshotCache<DeepSeekV4CausalSnapshot>,
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
        snapshot_cache_max_bytes: u64,
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
            cache: SnapshotCache::new(snapshot_cache_max_bytes),
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
                crate::checked_token_id(token, self.vocab_size, "prompt")
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

    fn output_protocol(&self, request: &ServeRequest) -> OutputProtocol {
        OutputProtocol::Qwen {
            preopened_reasoning: render_ds4::preopens_reasoning(request).unwrap_or(false),
            parse_tools: false,
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
        if max_tokens == 0 {
            return Err(ServeError::invalid_request(
                Some("max_output_tokens"),
                "max_output_tokens must be >= 1",
            )
            .into());
        }
        // Sampling validation precedes tokenization, admission, residency
        // transfer, session allocation, and all model execution.
        let sampler = request_sampler(request)?;
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
        let required = crate::required_forwards(
            "DeepSeek V4",
            prompt_ids.len(),
            max_tokens,
            Some(crate::DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY),
        )
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
        let stop_tokens = match self.gguf.stop_token_ids() {
            Ok(tokens) => tokens,
            Err(error) => {
                return Err(ServeError::server_error(format!("stop tokens: {error}")).into());
            }
        };
        for token in &stop_tokens {
            crate::checked_token_id(*token, self.vocab_size, "stop").map_err(|error| {
                ServeError::server_error(format!("invalid stop token: {error}"))
            })?;
        }
        let mut abort: Option<io::Error> = None;
        let tokenizer = &self.tokenizer;
        let ctx = &self.ctx;
        let vocab_size = self.vocab_size;
        let generation = {
            let abort = &mut abort;
            crate::generate_serial(
                logits,
                max_tokens,
                &stop_tokens,
                sampler,
                |token| {
                    let bytes = tokenizer
                        .try_decode_piece_bytes_exact(token)
                        .with_context(|| format!("decode token {token}"))?;
                    sink.piece(bytes).map_err(|error| {
                        *abort = Some(error);
                        anyhow::anyhow!("client disconnected during decode")
                    })
                },
                |token| {
                    let token = crate::checked_token_id(token, vocab_size, "generated")?;
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

        let (stop_reason, end) = match generation.stop_reason {
            crate::StopReason::Eos => (
                StopReason::Eos,
                GenerationEnd::StopToken(
                    *generation
                        .tokens
                        .last()
                        .expect("EOS generation includes its terminal token"),
                ),
            ),
            crate::StopReason::TokenLimit => (StopReason::TokenLimit, GenerationEnd::TokenLimit),
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
            end,
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
                self.cache.indexed_bytes,
                self.cache.max_bytes,
            );
            return;
        };
        let signals = self.ctx.memory_signals();
        if let Err(reason) = super::snapshot_capture_admission(entry_bytes, signals) {
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
    fn snapshot_cache_prefers_longest_strict_prefix() {
        let mut cache = SnapshotCache::new(1024);
        let bytes = cache.strict_eligibility(&[1], 5).unwrap();
        assert!(cache.insert_strict(vec![1], "short", bytes));
        let bytes = cache.strict_eligibility(&[1, 2], 4).unwrap();
        assert!(cache.insert_strict(vec![1, 2], "long", bytes));
        let (prefix_len, value) = cache.best_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(prefix_len, 2);
        assert_eq!(*value, "long");
        assert!(cache.best_prefix(&[1, 2]).is_some_and(|hit| hit.0 == 1));
    }

    #[test]
    fn snapshot_cache_accounts_bytes_and_evicts_lru() {
        let mut cache = SnapshotCache::new(20);
        let bytes = cache.strict_eligibility(&[1], 6).unwrap();
        assert!(cache.insert_strict(vec![1], "first", bytes)); // 10 bytes with key
        let bytes = cache.strict_eligibility(&[2], 6).unwrap();
        assert!(cache.insert_strict(vec![2], "second", bytes));
        assert_eq!(cache.indexed_bytes, 20);
        assert!(cache.best_prefix(&[1, 9]).is_some()); // first is now MRU
        let bytes = cache.strict_eligibility(&[3], 6).unwrap();
        assert!(cache.insert_strict(vec![3], "third", bytes));
        assert_eq!(cache.indexed_bytes, 20);
        assert!(cache.best_prefix(&[2, 9]).is_none());
        assert!(cache.best_prefix(&[1, 9]).is_some());
        assert!(cache.strict_eligibility(&[4], 17).is_none());
        assert_eq!(cache.indexed_bytes, 20);
    }

    #[test]
    fn snapshot_cache_eligibility_does_not_evict() {
        let mut cache = SnapshotCache::new(20);
        let bytes = cache.strict_eligibility(&[1], 6).unwrap();
        assert!(cache.insert_strict(vec![1], "first", bytes));
        let bytes = cache.strict_eligibility(&[2], 6).unwrap();
        assert!(cache.insert_strict(vec![2], "second", bytes));

        assert!(cache.strict_eligibility(&[3], 6).is_some());
        assert!(cache.strict_eligibility(&[4], 17).is_none());
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.indexed_bytes, 20);
        assert!(cache.best_prefix(&[1, 9]).is_some());
        assert!(cache.best_prefix(&[2, 9]).is_some());
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
