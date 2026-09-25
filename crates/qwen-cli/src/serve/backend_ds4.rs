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
//!   exact token prefix;
//! - **an optional durable tier** (`--durable-snapshot-*`): every captured
//!   boundary of at least the minimum length is written behind on a
//!   background thread (DS4 snapshots are small), and a disk record longer
//!   than the best RAM match is promoted into RAM before restore. Sessions
//!   bind a process-local stand-in identity until the strong content
//!   identity resolves in the background; RAM hits captured under the
//!   stand-in are re-attributed to the strong identity once it is live.
//!
//! Thinking tiers pre-open `<think>` in the prompt, so [`preopens_reasoning`]
//! reports headless generation to the transport (S3-1).

use super::backend::request_sampler;
use super::decode_loop;
use super::durable::{
    DurablePlan, DurableWorker, Resolved, SHUTDOWN_FLUSH_BUDGET, queue_cap_bytes,
    resolve_content_identity,
};
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest, TemplateStyle};
use super::output_partition::{OutputProtocol, ToolGrammar};
use super::render_ds4;
use super::snapshot_cache::SnapshotCache;
use crate::DeepSeekV4MultigroupSelectorPlan;
use anyhow::Context as _;
use objc2_metal::MTLDevice;
use qwen_llm::checkpoint_identity::same_identity_sources;
use qwen_llm::deepseek_v4::DeepSeekV4Config;
use qwen_llm::deepseek_v4_checkpoint_store::{DeepSeekV4CheckpointStore, DeepSeekV4StoreContext};
use qwen_llm::deepseek_v4_metal::{
    DeepSeekV4CausalSnapshot, DeepSeekV4CompatibilityDigest, DeepSeekV4MetalResidency,
    DeepSeekV4ModelContentId, DeepSeekV4Session, DeepSeekV4SessionCapacity,
    DeepSeekV4SnapshotCodecConstraints,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::sampling::Sampler;
use qwen_llm::snapshot_policy::SnapshotPolicyConfig;
use qwen_llm::tokenizer::Tokenizer;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

struct Ds4Durable {
    plan: DurablePlan,
    store: DeepSeekV4CheckpointStore,
    worker: DurableWorker<Arc<DeepSeekV4CausalSnapshot>>,
}

/// A chosen warm start: matched prefix length, the snapshot (bound to the
/// session's identity), and which tier supplied it.
type WarmStart = (usize, Arc<DeepSeekV4CausalSnapshot>, &'static str);

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
    /// hard-fail without one. Until the strong content identity resolves
    /// (or when the durable tier is off) sessions bind this process-local
    /// stand-in, which is never published.
    ephemeral_content_id: DeepSeekV4ModelContentId,
    config: DeepSeekV4Config,
    durable: Option<Ds4Durable>,
    /// Deployment default for `x_qwen.template_style` (`--template-style`).
    pub(super) template_style: TemplateStyle,
}

/// Process-unique, never published: pid + start nanos + a tag.
fn ephemeral_content_id() -> DeepSeekV4ModelContentId {
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
    DeepSeekV4ModelContentId::new(ephemeral)
}

/// The identity a new session binds: the strong content identity once the
/// durable tier resolved it, else the process-local stand-in.
fn session_content_id(
    strong: Option<DeepSeekV4ModelContentId>,
    ephemeral: DeepSeekV4ModelContentId,
) -> DeepSeekV4ModelContentId {
    strong.unwrap_or(ephemeral)
}

/// Write-behind eligibility for one captured boundary.
fn persistable(
    prefix_len: usize,
    min_tokens: usize,
    bound: DeepSeekV4ModelContentId,
    strong: Option<DeepSeekV4ModelContentId>,
) -> bool {
    prefix_len >= min_tokens.max(1) && strong == Some(bound)
}

fn store_context(
    content_id: DeepSeekV4ModelContentId,
    config: &DeepSeekV4Config,
    session_capacity: DeepSeekV4SessionCapacity,
    max_record_bytes: u64,
) -> DeepSeekV4StoreContext<'_> {
    DeepSeekV4StoreContext {
        compatibility_digest: DeepSeekV4CompatibilityDigest::for_model(content_id, config),
        codec_constraints: DeepSeekV4SnapshotCodecConstraints {
            config,
            session_capacity,
            expected_model_content_id: content_id,
            max_record_bytes,
        },
        max_record_bytes,
    }
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
        let config = residency.config().clone();
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
            ephemeral_content_id: ephemeral_content_id(),
            config,
            durable: None,
            template_style: TemplateStyle::House,
        })
    }

    /// Enable write-behind persistence and disk promotion. The strong
    /// content identity resolves on the worker thread over a second open of
    /// the loaded files (hashing every shard on a cold identity cache).
    pub(crate) fn attach_durable(
        &mut self,
        plan: DurablePlan,
        model_path: &Path,
    ) -> anyhow::Result<()> {
        let gguf = GgufFile::open(model_path)
            .with_context(|| format!("reopen {} for identity", model_path.display()))?;
        anyhow::ensure!(
            same_identity_sources(&gguf, &self.gguf),
            "model files changed since load; durable identity would not name the resident weights"
        );
        let store = DeepSeekV4CheckpointStore::new(&plan.root, plan.max_bytes);
        let worker_store = store.clone();
        let config = self.config.clone();
        let capacity = self.session_capacity;
        let max_record_bytes = plan.max_record_bytes;
        let worker = DurableWorker::spawn(
            "deepseek_v4",
            queue_cap_bytes(self.snapshot_cache_plan.bytes),
            move || {
                worker_store
                    .has_managed_blobs()
                    .context("open durable snapshot store")?;
                let (content_id, detail) =
                    resolve_content_identity(&gguf, &worker_store.identity_cache())?;
                drop(gguf);
                let content_id = DeepSeekV4ModelContentId::new(content_id);
                Ok(Resolved {
                    content_id: *content_id.as_bytes(),
                    detail,
                    writer: move |snapshot: Arc<DeepSeekV4CausalSnapshot>| {
                        let report = worker_store.publish(
                            store_context(content_id, &config, capacity, max_record_bytes),
                            &snapshot,
                        )?;
                        Ok(format!(
                            "tokens={} blob_bytes={} outcome={:?} managed_bytes={} evicted_entries={}",
                            snapshot.next_position(),
                            report.blob_bytes,
                            report.outcome,
                            report.managed_bytes_after,
                            report.evicted_entries,
                        ))
                    },
                })
            },
        )?;
        tracing::info!(
            target: "qwen_diag",
            "serve durable: family=deepseek_v4 {plan} queue_cap_bytes={} write_policy=write_behind identity=resolving",
            worker.cap_bytes(),
        );
        self.durable = Some(Ds4Durable {
            plan,
            store,
            worker,
        });
        Ok(())
    }

    fn strong_content_id(&self) -> Option<DeepSeekV4ModelContentId> {
        self.durable
            .as_ref()
            .and_then(|durable| durable.worker.content_id())
            .map(DeepSeekV4ModelContentId::new)
    }

    /// Pick the warm start for `prompt_ids`: the best RAM prefix, replaced by
    /// a strictly longer durable record (promoted into RAM) when one exists.
    /// The returned snapshot is bound to `session_id`.
    fn select_warm_start(
        &mut self,
        prompt_ids: &[u32],
        session_id: DeepSeekV4ModelContentId,
    ) -> Option<WarmStart> {
        let mut chosen = self
            .cache
            .best_prefix(prompt_ids)
            .map(|(len, snapshot)| (len, snapshot, "ram"));
        if let Some(promoted) =
            self.promote_durable_prefix(prompt_ids, chosen.as_ref().map_or(0, |(len, ..)| *len))
        {
            chosen = Some((promoted.0, promoted.1, "disk"));
        }
        let (len, snapshot, source) = chosen?;
        if snapshot.model_content_id() == session_id {
            return Some((len, snapshot, source));
        }
        // A RAM hit captured under the stand-in before the strong identity
        // resolved: same resident weights, so re-attribute a copy.
        match snapshot.rebound_to_model(session_id, &self.config) {
            Ok(rebound) => Some((len, Arc::new(rebound), source)),
            Err(error) => {
                tracing::warn!(
                    "serve: deepseek_v4 snapshot identity rebinding failed; cold prefilling: {error}"
                );
                None
            }
        }
    }

    /// Promote the longest durable prefix strictly longer than `floor`.
    fn promote_durable_prefix(
        &mut self,
        prompt_ids: &[u32],
        floor: usize,
    ) -> Option<(usize, Arc<DeepSeekV4CausalSnapshot>)> {
        let strong = self.strong_content_id()?;
        let Self {
            durable,
            cache,
            ctx,
            config,
            session_capacity,
            ..
        } = self;
        let durable = durable.as_ref()?;
        let min_tokens = durable.plan.min_tokens;
        // Strict prefixes only: a snapshot carries no logits.
        if prompt_ids.len() <= min_tokens.max(floor) {
            return None;
        }
        let t0 = Instant::now();
        let mut denied = None;
        let result = durable.store.lookup_filtered(
            store_context(
                strong,
                config,
                *session_capacity,
                durable.plan.max_record_bytes,
            ),
            prompt_ids,
            |matched, blob_bytes| {
                if matched <= floor || matched < min_tokens {
                    return false;
                }
                let Some(entry_bytes) =
                    cache.strict_eligibility(&prompt_ids[..matched], blob_bytes)
                else {
                    denied = Some("cache_budget");
                    return false;
                };
                if super::admit_snapshot_capture(
                    entry_bytes,
                    || ctx.memory_signals(),
                    |bytes| cache.evict_for(bytes),
                )
                .is_err()
                {
                    denied = Some("memory_headroom");
                    return false;
                }
                true
            },
        );
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let report = match result {
            Ok(report) => report,
            Err(error) => {
                tracing::warn!(
                    target: "qwen_diag",
                    "serve durable: family=deepseek_v4 lookup failed after {ms:.1} ms; continuing without disk: {error}"
                );
                return None;
            }
        };
        if report.candidates_examined > 0 {
            tracing::info!(
                target: "qwen_diag",
                "serve durable: family=deepseek_v4 lookup hit={} matched={} ram_matched={floor} candidates={} corrupt_removed={} unusable_skipped={} denied={} lookup_ms={ms:.1}",
                report.snapshot.is_some(),
                report.matched_prefix_len,
                report.candidates_examined,
                report.corrupt_entries_removed,
                report.unusable_skipped,
                denied.unwrap_or("none"),
            );
        }
        let snapshot = Arc::new(report.snapshot?);
        let tokens = snapshot.prefix_tokens().to_vec();
        if let Some(bytes) = SnapshotCache::<DeepSeekV4CausalSnapshot>::entry_bytes(
            tokens.len(),
            snapshot.payload_bytes(),
        ) && !cache.insert_shared_strict(tokens, Arc::clone(&snapshot), bytes)
        {
            tracing::warn!(
                "serve: deepseek_v4 disk snapshot not retained in RAM; restoring it once"
            );
        }
        Some((report.matched_prefix_len, snapshot))
    }
}

fn stop_reason_is_token_limit(generation: &crate::GenerationResult) -> bool {
    matches!(generation.stop_reason, crate::StopReason::TokenLimit)
}

/// A turn cut off by the token limit while still inside an open `<think>`
/// re-renders as a closed block, so its token prefix can never be extended.
/// Only a generation that began inside open reasoning can end inside it; a
/// chat-mode turn stopped by the limit is still extendable and keeps its
/// completed snapshot.
fn truncated_inside_reasoning(
    preopened_reasoning: bool,
    stopped_at_token_limit: bool,
    closed_reasoning: impl FnOnce() -> bool,
) -> bool {
    preopened_reasoning && stopped_at_token_limit && !closed_reasoning()
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
        if self
            .durable
            .as_ref()
            .is_some_and(|durable| durable.worker.failed())
        {
            self.durable = None;
        }
    }

    /// Snapshots are written behind as they are captured, so shutdown only
    /// waits (bounded) for the queue to drain.
    fn shutdown(&mut self) {
        let Some(durable) = self.durable.as_ref() else {
            return;
        };
        if durable.worker.content_id().is_none() {
            tracing::info!(target: "qwen_diag", "serve durable: family=deepseek_v4 shutdown: identity still resolving; nothing flushed");
            return;
        }
        let started = Instant::now();
        let drained = durable.worker.wait_idle(started + SHUTDOWN_FLUSH_BUDGET);
        let stats = durable.worker.stats();
        tracing::info!(
            target: "qwen_diag",
            "serve durable: family=deepseek_v4 shutdown drained={drained} elapsed_ms={:.1} written_total={} failed_total={} dropped_total={}",
            started.elapsed().as_secs_f64() * 1e3,
            stats.written,
            stats.failed,
            stats.dropped,
        );
    }

    fn template_style_default(&self) -> Option<TemplateStyle> {
        Some(self.template_style)
    }

    /// Under house style an absent reasoning item is a chat turn's
    /// provenance, not lost reasoning, so it is not reported as missing.
    fn normalize_request(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
        if request.template_style != Some(TemplateStyle::Upstream) {
            request.history_reasoning_missing = 0;
        }
        Ok(())
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
        let preopened_reasoning = render_ds4::preopens_reasoning(request).unwrap_or(false);
        self.run_request(
            residency,
            &prompt_ids,
            preopened_reasoning,
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
        preopened_reasoning: bool,
        max_tokens: usize,
        tokenize_ms: f64,
        mut sampler: Sampler,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        // Choose the warm start before binding the session: the session's
        // identity must match the snapshot it restores.
        let restore_t0 = Instant::now();
        let session_id = session_content_id(self.strong_content_id(), self.ephemeral_content_id);
        let warm_start = self.select_warm_start(prompt_ids, session_id);
        let select_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
        // Session per request over the long-lived residency; the slot is
        // restored on every exit path below.
        let session_t0 = Instant::now();
        let mut session = match DeepSeekV4Session::new_with_model_content_id_recoverable(
            &self.ctx, residency, session_id,
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
            preopened_reasoning,
            warm_start,
            select_ms,
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
        preopened_reasoning: bool,
        warm_start: Option<WarmStart>,
        select_ms: f64,
        max_tokens: usize,
        tokenize_ms: f64,
        session_ms: f64,
        sampler: &mut Sampler,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        // Warm restore from the chosen RAM or promoted disk snapshot.
        let restore_t0 = Instant::now();
        let (matched_tokens, restore_source) = match warm_start {
            Some((prefix_len, snapshot, source)) => {
                match session.restore_causal_snapshot(&snapshot) {
                    Ok(()) => (prefix_len, source),
                    Err(error) => {
                        tracing::warn!(
                            "serve: deepseek_v4 {source} restore failed, cold prefilling: {error}"
                        );
                        (0, "none")
                    }
                }
            }
            None => (0, "none"),
        };
        let restore_ms = select_ms + restore_t0.elapsed().as_secs_f64() * 1e3;

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
        let truncated_in_reasoning = truncated_inside_reasoning(
            preopened_reasoning,
            stop_reason_is_token_limit(&generation),
            || decoded_text_closed_reasoning(&generation, &self.tokenizer),
        );
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
            "serve phases: tokenize_ms={tokenize_ms:.1} session_ms={session_ms:.1} restore_ms={restore_ms:.1} restore_source={restore_source} prefill_ms={prefill_ms:.1} prompt_capture_ms={capture_ms:.1} family=deepseek_v4",
        );
        Ok(super::outcome::finish_generation(
            prompt_ids.len(),
            &generation,
            matched_tokens,
            restore_ms,
        ))
    }

    /// Queue a freshly captured boundary for disk when it is long enough and
    /// bound to the strong identity (stand-in captures are never published).
    fn write_behind(&self, snapshot: Arc<DeepSeekV4CausalSnapshot>, payload_bytes: u64) {
        let Some(durable) = self.durable.as_ref() else {
            return;
        };
        if !persistable(
            snapshot.next_position() as usize,
            durable.plan.min_tokens,
            snapshot.model_content_id(),
            self.strong_content_id(),
        ) {
            return;
        }
        durable.worker.try_enqueue(snapshot, payload_bytes);
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
                let snapshot = Arc::new(snapshot);
                if !self.cache.insert_shared_strict(
                    tokens.to_vec(),
                    Arc::clone(&snapshot),
                    entry_bytes,
                ) {
                    tracing::warn!(
                        "serve: deepseek_v4 {boundary} snapshot rejected at strict cache insertion; request continues"
                    );
                    return;
                }
                self.write_behind(snapshot, payload_bytes);
            }
            Err(error) => tracing::warn!(
                "serve: deepseek_v4 {boundary} snapshot capture failed; request continues: {error}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    /// The serve backend over the local DS4 GGUF (`DSV4_GGUF` overrides),
    /// with the 32K forward budget serve probes use.
    fn gpu_backend(model_id: &str) -> DeepSeekV4Backend {
        let path = std::env::var("DSV4_GGUF").unwrap_or_else(|_| {
            "/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf".into()
        });
        let ctx = MetalContext::new().unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let forward_limit = crate::deepseek_v4_forward_budget_for_context_limit(32_768).unwrap();
        DeepSeekV4Backend::new(
            ctx,
            gguf,
            model_id.into(),
            64,
            forward_limit,
            crate::DeepSeekV4MultigroupSelectorArg::Auto,
            Some(0),
            Default::default(),
        )
        .unwrap()
    }

    /// Render and tokenize a chat-mode request exactly as serve does.
    fn render_ids(
        backend: &DeepSeekV4Backend,
        instructions: &str,
        input: serde_json::Value,
    ) -> Vec<u32> {
        let request = crate::open_responses::items::parse_request(&serde_json::json!({
            "model": backend.model_id, "instructions": instructions, "input": input,
            "reasoning": {"effort": "none"}, "max_output_tokens": 8, "temperature": 0,
        }))
        .unwrap();
        let prompt = backend.render_prompt(&request).unwrap();
        decode_loop::encode_checked(
            &backend.tokenizer,
            &prompt,
            false,
            backend.vocab_size,
            "DS4",
        )
        .unwrap()
    }

    /// A sealed session over the backend's residency, as `run_request` makes.
    fn fresh_session(backend: &mut DeepSeekV4Backend) -> DeepSeekV4Session {
        let residency = backend.residency.take().unwrap();
        let id = session_content_id(backend.strong_content_id(), backend.ephemeral_content_id);
        let mut session =
            DeepSeekV4Session::new_with_model_content_id_recoverable(&backend.ctx, residency, id)
                .map_err(|failure| failure.into_parts().1)
                .unwrap();
        backend
            .selector_plan
            .seal_session(&mut session, "test")
            .unwrap();
        session
    }

    fn release_session(backend: &mut DeepSeekV4Backend, session: DeepSeekV4Session) {
        backend.residency = Some(session.into_residency().unwrap());
    }

    /// Serve's chunking at `chunk` tokens: advance all but the final chunk.
    fn serve_prefill(
        backend: &DeepSeekV4Backend,
        session: &mut DeepSeekV4Session,
        tokens: &[u32],
        chunk: usize,
    ) {
        let ranges = crate::deepseek_v4_prefill_chunk_ranges(tokens.len(), chunk);
        let n = ranges.len();
        for (index, range) in ranges.into_iter().enumerate() {
            if index + 1 == n {
                session
                    .prefill_tokens(&backend.ctx, &tokens[range])
                    .unwrap();
            } else {
                session
                    .advance_tokens(&backend.ctx, &tokens[range])
                    .unwrap();
            }
        }
    }

    /// A warm start is exact: restoring a prompt-end snapshot (in place or
    /// into a fresh session) and prefilling the rest is bit-identical to
    /// continuing the original session. With DS4_SCHEDULE_REPORT set, also
    /// print how far continuing, a single cold chunk and token-by-token
    /// decode land from each other: DS4's packed and singleton schedules
    /// differ by cos ~0.997 (as llama.cpp's do; DEEPSEEK-V4-STRATEGY.md), so a
    /// near-tie greedy token can differ between a warm and a cold turn.
    #[test]
    #[ignore = "loads DeepSeek V4 (DSV4_GGUF); GPU"]
    fn gpu_ds4_prompt_snapshot_restore_equals_continuing() {
        let mut backend = gpu_backend("ds4-warm-start");
        let system = "You are a careful, pragmatic software engineering agent. Prefer small, verifiable steps. ".repeat(160);
        let user1 = serde_json::json!({"type":"message","role":"user","content":"List three prime numbers, one line."});
        let p1 = render_ids(&backend, &system, serde_json::json!([user1]));
        let p2 = render_ids(
            &backend,
            &system,
            serde_json::json!([user1,
                {"type":"message","role":"assistant","content":"2, 3, 5"},
                {"type":"message","role":"user","content":"Now three more, larger than 50, one line."}]),
        );
        assert!(p2.starts_with(&p1) && p2.len() > p1.len());
        let chunk = backend.prefill_chunk_tokens;
        let vocab = backend.vocab_size;
        let fresh = fresh_session;
        let release = release_session;
        let prefill =
            |backend: &DeepSeekV4Backend, session: &mut DeepSeekV4Session, tokens: &[u32]| {
                serve_prefill(backend, session, tokens, chunk)
            };
        let bits = |logits: &[f32]| logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>();

        let mut session = fresh(&mut backend);
        prefill(&backend, &mut session, &p1);
        let snapshot = session.capture_causal_snapshot().unwrap();
        prefill(&backend, &mut session, &p2[p1.len()..]);
        let continued = crate::copy_deepseek_v4_logits(&session, vocab, "continued").unwrap();
        session.restore_causal_snapshot(&snapshot).unwrap();
        prefill(&backend, &mut session, &p2[p1.len()..]);
        let in_place = crate::copy_deepseek_v4_logits(&session, vocab, "in place").unwrap();
        release(&mut backend, session);
        let mut session = fresh(&mut backend);
        session.restore_causal_snapshot(&snapshot).unwrap();
        prefill(&backend, &mut session, &p2[p1.len()..]);
        let restored = crate::copy_deepseek_v4_logits(&session, vocab, "restored").unwrap();
        release(&mut backend, session);
        assert_eq!(
            bits(&in_place),
            bits(&continued),
            "in-place restore differs from continuing"
        );
        assert_eq!(
            bits(&restored),
            bits(&continued),
            "fresh restore differs from continuing"
        );

        if std::env::var("DS4_SCHEDULE_REPORT").is_err() {
            return;
        }
        let compare = |label: &str, a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let max_abs = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "{label:<34} cos={:.6} max_abs={max_abs:.3}",
                dot / (na * nb)
            );
        };
        let mut session = fresh(&mut backend);
        prefill(&backend, &mut session, &p2);
        let cold = crate::copy_deepseek_v4_logits(&session, vocab, "cold").unwrap();
        release(&mut backend, session);
        let mut session = fresh(&mut backend);
        for &token in &p2 {
            session.forward_token(&backend.ctx, token).unwrap();
        }
        let serial = crate::copy_deepseek_v4_logits(&session, vocab, "serial").unwrap();
        release(&mut backend, session);
        compare("continued vs cold single chunk", &continued, &cold);
        compare("continued vs token-by-token", &continued, &serial);
        compare("cold single chunk vs token-by-token", &cold, &serial);
    }

    /// The retained serve probe case (/tmp/qwen-probe/probe.py, SYS_REPEAT=60,
    /// `chat-nothink`, 8 output tokens): turn-1 and turn-2 prompt ids.
    fn retained_turn2_ids(backend: &DeepSeekV4Backend) -> (Vec<u32>, Vec<u32>) {
        let para = "You are a careful, pragmatic software engineering agent working in a local \
            repository. Prefer small, verifiable steps. Read files before editing them. \
            Explain your reasoning briefly, cite file paths with line numbers, and never \
            invent APIs. When a command fails, read the error carefully before retrying. \
            Keep answers concise and avoid unnecessary preamble or summaries. ";
        let system = (0..60)
            .map(|i| format!("## Guideline {i}\n{para}"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let user1 = serde_json::json!({"type": "message", "role": "user",
            "content": "Name three prime numbers and say in one sentence what a prime is."});
        let answer1 = serde_json::json!({"id": "msg_resp_6ab450b40001", "type": "message",
            "role": "assistant", "status": "incomplete", "content": [{"type": "output_text",
            "text": "A prime number is a natural number greater", "annotations": []}]});
        let user2 = serde_json::json!({"type": "message", "role": "user",
            "content": "Name three more primes larger than 50, one line."});
        let p1 = render_ids(backend, &system, serde_json::json!([user1]));
        let p2 = render_ids(backend, &system, serde_json::json!([user1, answer1, user2]));
        assert!(p2.starts_with(&p1));
        // The retained serve logs: 4,578 / 4,602 prompt tokens, completed
        // boundary at 4,585 (7 decoded transitions).
        assert_eq!(
            (p1.len(), p2.len()),
            (4_578, 4_602),
            "not the retained case"
        );
        (p1, p2)
    }

    /// KL(p || q) of the softmaxes of two logit vectors.
    fn logits_kl(p: &[f32], q: &[f32]) -> f64 {
        let log_softmax = |x: &[f32]| {
            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = x.iter().map(|&v| (v as f64 - max).exp()).sum();
            x.iter()
                .map(|&v| v as f64 - max - sum.ln())
                .collect::<Vec<_>>()
        };
        let (lp, lq) = (log_softmax(p), log_softmax(q));
        lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
    }

    /// How the retained turn-2 distribution moves with the packed partition
    /// alone: the prompt split at explicit interior boundaries
    /// (`DS4_FLIP_PARTITIONS`, `;`-separated lists of `,`-separated
    /// positions; chunks at most 4096 tokens), each compared with serve's
    /// cold schedule by KL and top-3. On 2026-09-25 legal partitions ranged
    /// from KL 0 to 0.031 and several flipped greedy "Three" to "A".
    #[test]
    #[ignore = "loads DeepSeek V4 (DSV4_GGUF); GPU; ~20 s per partition"]
    fn gpu_ds4_turn2_partition_sweep() {
        let mut backend = gpu_backend("DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004");
        let (_, p2) = retained_turn2_ids(&backend);
        let vocab = backend.vocab_size;
        let partitions = std::env::var("DS4_FLIP_PARTITIONS")
            .unwrap_or_else(|_| "4096,4578;4096,4577;4096,4586;4096,4450;4096,4578,4580".into());
        let run = |backend: &mut DeepSeekV4Backend, boundaries: &[usize]| -> Vec<f32> {
            let mut session = fresh_session(backend);
            let mut starts = vec![0];
            starts.extend_from_slice(boundaries);
            for (index, &start) in starts.iter().enumerate() {
                let end = starts.get(index + 1).copied().unwrap_or(p2.len());
                assert!(
                    start < end
                        && end - start
                            <= qwen_llm::deepseek_v4_metal::DEEPSEEK_V4_PREFILL_MAX_TOKENS,
                    "chunk {start}..{end} must be nonempty and at most 4096 tokens"
                );
                if end == p2.len() {
                    session
                        .prefill_tokens(&backend.ctx, &p2[start..end])
                        .unwrap();
                } else {
                    session
                        .advance_tokens(&backend.ctx, &p2[start..end])
                        .unwrap();
                }
            }
            let logits = crate::copy_deepseek_v4_logits(&session, vocab, "sweep").unwrap();
            release_session(backend, session);
            logits
        };
        let reference = run(&mut backend, &[4_096]);
        for partition in partitions.split(';') {
            let boundaries: Vec<usize> = partition
                .split(',')
                .map(|value| value.trim().parse().unwrap())
                .collect();
            let logits = run(&mut backend, &boundaries);
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            let top = order[..3]
                .iter()
                .map(|&id| {
                    format!(
                        "{:?}={:.3}",
                        backend.tokenizer.decode(&[id as i32]),
                        logits[id]
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "boundaries={partition:<16} kl_vs_cold={:.5} top: {top}",
                logits_kl(&logits, &reference)
            );
        }
    }

    /// Diagnosis for the retained warm/cold turn-2 flip (PERF-LOG 2026-09-23,
    /// leverage map #1): the serve probe's exact two-turn chat case under
    /// every legal native schedule of the turn-2 prompt. Prints each
    /// schedule's top tokens and greedy continuation, and writes the prompt
    /// ids and raw last-position logits under `DS4_FLIP_DUMP` (default
    /// `/tmp/qwen-probe/ds4-flip`) for offline comparison with llama.cpp.
    #[test]
    #[ignore = "loads DeepSeek V4 (DSV4_GGUF); GPU; ~5 minutes"]
    fn gpu_ds4_turn2_schedule_report() {
        let dump = std::path::PathBuf::from(
            std::env::var("DS4_FLIP_DUMP").unwrap_or_else(|_| "/tmp/qwen-probe/ds4-flip".into()),
        );
        std::fs::create_dir_all(&dump).unwrap();
        let mut backend = gpu_backend("DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004");
        let (p1, p2) = retained_turn2_ids(&backend);
        let completed = p1.len() + 7;
        let chunk = backend.prefill_chunk_tokens;
        assert_eq!(
            chunk, 4_096,
            "the retained case ran at serve's default chunk; unset QWEN_DSV4_PREFILL_CHUNK_TOKENS"
        );
        let vocab = backend.vocab_size;
        let piece = |backend: &DeepSeekV4Backend, id: usize| {
            format!("{:?}", backend.tokenizer.decode(&[id as i32]))
        };
        let argmax = |logits: &[f32]| {
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        std::fs::write(
            dump.join("prompt_ids.json"),
            serde_json::json!({"p1": p1, "p2": p2, "completed": completed}).to_string(),
        )
        .unwrap();

        // Greedy continuation from the session's current logits, then report
        // and dump. Consumes the session.
        let finish = |backend: &mut DeepSeekV4Backend,
                      mut session: DeepSeekV4Session,
                      label: &str|
         -> Vec<f32> {
            let logits = crate::copy_deepseek_v4_logits(&session, vocab, label).unwrap();
            let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dump.join(format!("{label}.f32")), bytes).unwrap();
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            let top = order[..8]
                .iter()
                .map(|&id| format!("{}={:.3}", piece(backend, id), logits[id]))
                .collect::<Vec<_>>()
                .join(" ");
            let mut greedy = vec![argmax(&logits)];
            while greedy.len() < 8 {
                session
                    .forward_token(&backend.ctx, *greedy.last().unwrap() as u32)
                    .unwrap();
                let next = crate::copy_deepseek_v4_logits(&session, vocab, label).unwrap();
                greedy.push(argmax(&next));
            }
            let text = backend
                .tokenizer
                .decode(&greedy.iter().map(|&id| id as i32).collect::<Vec<_>>());
            eprintln!("{label:<22} top: {top}");
            eprintln!("{label:<22} greedy: {text:?}");
            release_session(backend, session);
            logits
        };

        // Turn 1 as serve ran it; its prompt-end boundary seeds S1/S2/S4.
        let mut session = fresh_session(&mut backend);
        serve_prefill(&backend, &mut session, &p1, chunk);
        let t1 = crate::copy_deepseek_v4_logits(&session, vocab, "t1").unwrap();
        let snapshot = session.capture_causal_snapshot().unwrap();
        release_session(&mut backend, session);
        let from_prompt_end = |backend: &mut DeepSeekV4Backend| {
            let mut session = fresh_session(backend);
            session.restore_causal_snapshot(&snapshot).unwrap();
            session
        };
        eprintln!(
            "t1 greedy first token {} (history has {})",
            piece(&backend, argmax(&t1)),
            piece(&backend, p2[p1.len()] as usize)
        );

        // S1: serve warm from the prompt-end snapshot (== cold partitioned at
        // the turn boundary, by the exact-restore test above).
        let mut session = from_prompt_end(&mut backend);
        serve_prefill(&backend, &mut session, &p2[p1.len()..], chunk);
        finish(&mut backend, session, "s1_prompt_end_warm");

        // S2: the live session: turn 1's 7 decoded transitions as singleton
        // forwards (they must be its greedy tokens), then the rest.
        let mut session = from_prompt_end(&mut backend);
        let mut expected = argmax(&t1);
        for &token in &p2[p1.len()..completed] {
            assert_eq!(
                token as usize,
                expected,
                "s2: turn 1 decoded {} but history has {}",
                piece(&backend, expected),
                piece(&backend, token as usize)
            );
            session.forward_token(&backend.ctx, token).unwrap();
            expected = argmax(&crate::copy_deepseek_v4_logits(&session, vocab, "s2").unwrap());
        }
        serve_prefill(&backend, &mut session, &p2[completed..], chunk);
        finish(&mut backend, session, "s2_live_continuation");

        // S4: the 24-token suffix as singleton forwards.
        let mut session = from_prompt_end(&mut backend);
        for &token in &p2[p1.len()..] {
            session.forward_token(&backend.ctx, token).unwrap();
        }
        finish(&mut backend, session, "s4_suffix_singleton");

        // S3/S5/S6: cold at serve's chunk and two smaller legal chunks.
        for (label, cold_chunk) in [
            ("s3_cold_4096", chunk),
            ("s5_cold_1024", 1_024),
            ("s6_cold_512", 512),
        ] {
            let mut session = fresh_session(&mut backend);
            serve_prefill(&backend, &mut session, &p2, cold_chunk);
            finish(&mut backend, session, label);
        }

        // S7: every prompt token as a singleton forward (~3 minutes).
        let mut session = fresh_session(&mut backend);
        for &token in &p2 {
            session.forward_token(&backend.ctx, token).unwrap();
        }
        finish(&mut backend, session, "s7_all_singleton");
    }

    #[test]
    fn only_open_reasoning_can_be_truncated_inside_reasoning() {
        use super::truncated_inside_reasoning as truncated;
        // Chat mode at the limit: extendable, keeps its completed snapshot.
        assert!(!truncated(false, true, || false));
        // Thinking mode at the limit without `</think>`: not extendable.
        assert!(truncated(true, true, || false));
        // Thinking mode that closed its reasoning, or stopped naturally.
        assert!(!truncated(true, true, || true));
        assert!(!truncated(true, false, || false));
    }

    use super::*;

    #[test]
    fn ephemeral_identity_fills_exactly_32_bytes() {
        // The startup panic this replaces (source 32 vs destination 16) only
        // surfaced when a 97 GB model finished loading.
        let ephemeral = ephemeral_content_id();
        assert_eq!(&ephemeral.as_bytes()[12..], b"qwen-serve-ephemeral");
        assert_eq!(
            &ephemeral.as_bytes()[..4],
            &std::process::id().to_le_bytes()
        );
    }

    #[test]
    fn sessions_bind_the_strong_identity_once_durable_resolves_it() {
        let ephemeral = ephemeral_content_id();
        let strong = DeepSeekV4ModelContentId::new([9; 32]);
        // Durable off or still resolving: the process-local stand-in.
        assert_eq!(session_content_id(None, ephemeral), ephemeral);
        // Resolved: never ephemeral again.
        assert_eq!(session_content_id(Some(strong), ephemeral), strong);
        assert_ne!(session_content_id(Some(strong), ephemeral), ephemeral);
    }

    #[test]
    fn only_long_strong_bound_boundaries_are_written_behind() {
        let ephemeral = ephemeral_content_id();
        let strong = DeepSeekV4ModelContentId::new([9; 32]);
        assert!(persistable(1024, 1024, strong, Some(strong)));
        assert!(!persistable(1023, 1024, strong, Some(strong)));
        assert!(!persistable(4096, 1024, ephemeral, Some(strong)));
        assert!(!persistable(4096, 1024, ephemeral, None));
        // A zero minimum still never persists an empty prefix.
        assert!(!persistable(0, 0, strong, Some(strong)));
        assert!(persistable(1, 0, strong, Some(strong)));
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
