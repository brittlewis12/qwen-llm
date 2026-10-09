//! GLM-5.3-Flash serve: borrowed resident weights, one live session kept
//! across requests and a RAM snapshot cache. A request continues the live
//! session when its prompt strictly extends exactly what the session
//! consumed (a replayed conversation does: the model ends a turn by
//! sampling `<|user|>`, the next turn's opener), or restores the longest
//! compatible snapshot when that reaches further; otherwise it prefills a
//! fresh session.
//!
//! Snapshots are captured where the renderer recorded boundaries: the end
//! of the shared instructions-and-tools prefix and, for Exact, the start of
//! the generation header. A request of a snapshot schedule splits its
//! prefill at those positions whether it hits, misses, finds the cache full
//! or is denied capture, so a restore continues exactly the trajectory a
//! miss would have run. Exact (`x_qwen.prefill_lineage: "exact"`) is
//! segmentation-invariant, so its snapshots also equal a cold run. Fast
//! snapshots split at the shared prefix only; they are on by default
//! (decision of 2026-10-08, PERF-LOG: the split against the unsplit Fast it
//! replaces showed no detectable quality change, while the preregistered
//! comparison against Exact was inconclusive) and `QWEN_GLM_FAST_SNAPSHOTS=0`
//! turns them off. A Fast request restores only a snapshot ending exactly
//! at its own verified cut, and publishes one only from a state on the
//! canonical schedule (packed chunks from 0, no other cut, no decoded
//! tokens); without a verified cut it neither restores nor captures.
//! Memory-pressure refusals of the transport or a new session release the
//! cache (except a snapshot being restored) and retry once.
use super::decode_loop;
use super::http::{
    BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink, PreparedResponse,
    PromptBoundaries,
};
use super::items::{ServeError, ServeRequest};
use super::render_glm5_next::{self as render, FAMILY};
use super::snapshot_cache::SnapshotCache;
use anyhow::{Context, Result};
use qwen_llm::gguf::GgufFile;
use qwen_llm::glm5_next::Glm5NextPreparedArtifact;
use qwen_llm::glm5_next_chat::{CHAT_STOPS, VerifiedChatProfile};
use qwen_llm::glm5_next_metal::{
    CapacityAdvice, DEFAULT_PREFILL_ROWS, Glm5NextMetalError, Glm5NextSession, Glm5NextSnapshot,
    Glm5NextWeights, PackedLineage, prefetch_retained_with_cancel, preflight_session,
    snapshot_bytes,
};
use qwen_llm::metal::MetalContext;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::sampling::Sampler;
use qwen_llm::snapshot_policy::EntryId;
use std::io;
use std::sync::Arc;
use std::time::Instant;

/// Default-on rollback lever for live-session and snapshot reuse.
const PREFIX_REUSE_ENV: &str = "QWEN_GLM_PREFIX_REUSE";
/// Default-on lever for Fast-lineage snapshots (shared-prefix split); `0`
/// restores unsplit Fast prefill with live-session reuse only.
const FAST_SNAPSHOTS_ENV: &str = "QWEN_GLM_FAST_SNAPSHOTS";

/// The prefill schedule a snapshot belongs to. A restore is valid only into
/// a request that splits its prefill the same way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Schedule {
    /// Exact lineage: cuts at the shared prefix end and the generation
    /// header start. Packed Exact equals serial decode, so any cuts give the
    /// same state.
    ExactV1,
    /// Fast lineage: one cut at the shared prefix end.
    FastSharedSplitV1,
}

impl Schedule {
    fn name(self) -> &'static str {
        match self {
            Self::ExactV1 => "exact_v1",
            Self::FastSharedSplitV1 => "fast_shared_split_v1",
        }
    }
}

/// What must match, beyond the token prefix, for a cached snapshot to be
/// restored (weights instance and policy version are fixed per process and
/// checked again by the engine).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CacheNamespace {
    lineage: PackedLineage,
    schedule: Schedule,
}

/// CPU startup facts: the admitted artifact, its verified chat profile and
/// the session geometry fixed for the server's lifetime.
pub(super) struct Prepared<'g> {
    artifact: Glm5NextPreparedArtifact<'g>,
    pub(super) capacity: usize,
    default_max: usize,
    prefill_rows: usize,
    profile: VerifiedChatProfile,
    /// Longest decoded token, for the tool block's byte budget.
    max_piece_bytes: usize,
}

impl<'g> Prepared<'g> {
    /// Inspects the GLM artifact and session limits using CPU metadata only.
    pub(super) fn new(
        gguf: &'g GgufFile,
        invocation: &crate::cli::ServeInvocation,
    ) -> Result<Self> {
        let artifact = Glm5NextPreparedArtifact::inspect(gguf)
            .with_context(|| format!("admit {FAMILY} artifact"))?;
        let limits = super::resolve_serve_limits(
            super::profile(ModelFamily::Glm5Next),
            Some(artifact.config().context_length as usize),
            invocation.max_context_tokens,
            invocation.max_tokens,
        )?
        .context("GLM-5.3-Flash profile has no fixed serve capacity")?;
        let (capacity, default_max) = (limits.context_tokens, limits.max_tokens);
        // There is no raw serve lane: without a verified profile nothing renders.
        let profile = artifact.chat_profile().map_err(|error| {
            anyhow::anyhow!("{FAMILY} serve renders verified text chat: {error}")
        })?;
        let prefill_rows = if artifact.packed_prefill() {
            DEFAULT_PREFILL_ROWS.min(capacity)
        } else {
            0
        };
        let max_piece_bytes = artifact.tokenizer().max_decoded_piece_bytes();
        crate::shutdown::checkpoint()?;
        Ok(Self {
            artifact,
            capacity,
            default_max,
            prefill_rows,
            profile,
            max_piece_bytes,
        })
    }

    pub(super) fn describe(&self) -> String {
        format!(
            "family=glm5_next input=verified_chat_and_tools renderer={} capacity={} default_max_tokens={} prefill_rows={}",
            self.profile.renderer, self.capacity, self.default_max, self.prefill_rows
        )
    }
}

/// Device admission, prefetch and load, before the listener accepts: a
/// session that cannot fit is refused with the capacity that would.
pub(super) fn load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    prepared: &Prepared<'_>,
) -> Result<Glm5NextWeights> {
    match preflight_session(
        ctx,
        gguf,
        prepared.artifact.model(),
        prepared.capacity,
        prepared.prefill_rows,
    ) {
        Ok(_) => {}
        Err(error @ Glm5NextMetalError::MemoryAdmission { advice, .. }) => {
            let advice = match advice {
                CapacityAdvice::Fits(n) => format!("pass --max-context-tokens {n} or less"),
                CapacityAdvice::NoneFits => {
                    "free device memory or use a smaller artifact".to_string()
                }
                CapacityAdvice::NotEvaluated => {
                    "no smaller capacity was evaluated; see the refusal reason".to_string()
                }
            };
            return Err(error).context(format!("admit GLM-5.3 serve session: {advice}"));
        }
        Err(error) => return Err(error).context("admit GLM-5.3 serve session"),
    }
    let prefetch =
        prefetch_retained_with_cancel(ctx, gguf, 0.98, &|| crate::shutdown::checkpoint().is_err())
            .context("prefetch GLM-5.3 retained windows")?;
    tracing::info!(
        target: "qwen_diag",
        "serve prefetch: family=glm5_next windows={} cold_windows={} bytes_read={} wall_ms={:.1}",
        prefetch.windows,
        prefetch.cold_windows,
        prefetch.bytes_read,
        prefetch.wall.as_secs_f64() * 1e3
    );
    crate::shutdown::checkpoint()?;
    Glm5NextWeights::load(ctx, gguf).context("load GLM-5.3 weights")
}

/// Performs device admission and model load after listener bind, then serves.
pub(super) fn start(
    prepared: Prepared<'_>,
    gguf: &GgufFile,
    invocation: &crate::cli::ServeInvocation,
    listening: super::Listening,
    idle_window: std::time::Duration,
) -> Result<()> {
    crate::shutdown::checkpoint()?;
    let ctx = MetalContext::new()?;
    let started = std::time::Instant::now();
    let weights = load(&ctx, gguf, &prepared)?;
    let load_ms = started.elapsed().as_secs_f64() * 1e3;
    let snapshot_cache_plan = super::SnapshotCachePlan::resolve(
        invocation.snapshot_cache_mib,
        invocation.snapshot_policy,
        ctx.memory_signals(),
    )?;
    let limits = prepared.describe();
    let mut backend = Glm5NextBackend::new(
        &ctx,
        &weights,
        prepared,
        listening.model_id.clone(),
        idle_window,
        snapshot_cache_plan,
    );
    tracing::info!(target: "qwen_diag", "serve limits: {limits} {}", backend.describe_snapshots());
    let warm_up_ms = backend.warm_up()?;
    tracing::info!(target: "qwen_diag", "serve startup: family=glm5_next load_ms={load_ms:.1} warm_up_ms={warm_up_ms:.1}");
    listening.serve(load_ms, &mut backend)
}

pub(super) struct Glm5NextBackend<'w, 'g> {
    ctx: &'w MetalContext,
    weights: &'w Glm5NextWeights,
    prepared: Prepared<'g>,
    model_id: String,
    session: Option<Glm5NextSession<'w>>,
    /// Exactly the tokens `session` has consumed, or empty when unknown.
    history: Vec<u32>,
    prefix_reuse: bool,
    idle_residency: super::idle_residency::IdleResidency,
    /// Packed-prefill arithmetic for new sessions (tests compare warm and
    /// cold paths under `Exact`, which matches serial decode bitwise).
    lineage: PackedLineage,
    cache: SnapshotCache<Glm5NextSnapshot, CacheNamespace>,
    pub(super) snapshot_cache_plan: super::SnapshotCachePlan,
    fast_snapshots: bool,
    /// The live session's whole state is a Fast packed prefill of its
    /// history in chunks from 0 (no other cuts, no decoded tokens), so a
    /// Fast capture continuing it would equal a miss's.
    live_canonical: bool,
    /// Bits of the last request's prefill logits (tests compare paths).
    #[cfg(test)]
    last_prefill_logits: Vec<u32>,
    /// Treat every capture admission as denied (tests).
    #[cfg(test)]
    deny_captures: bool,
}

/// Why a cut's snapshot was or was not cached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptureOutcome {
    Captured,
    Present,
    Ineligible,
    Denied,
    Failed,
    /// A Fast state not produced on the canonical schedule; not published.
    NonCanonical,
}

impl CaptureOutcome {
    fn name(self) -> &'static str {
        match self {
            Self::Captured => "captured",
            Self::Present => "present",
            Self::Ineligible => "ineligible",
            Self::Denied => "denied",
            Self::Failed => "failed",
            Self::NonCanonical => "noncanonical",
        }
    }
}

/// The snapshot schedule of a request of `lineage`, or `None` when its
/// prefill neither splits nor captures: no cache budget, or Fast with the
/// lever off.
fn snapshot_schedule(
    cache_bytes: u64,
    fast_snapshots: bool,
    lineage: PackedLineage,
) -> Option<Schedule> {
    if cache_bytes == 0 {
        return None;
    }
    match lineage {
        PackedLineage::Exact => Some(Schedule::ExactV1),
        PackedLineage::Fast if fast_snapshots => Some(Schedule::FastSharedSplitV1),
        PackedLineage::Fast => None,
    }
}

/// Token positions where a request of `schedule` splits its prefill: each
/// renderer boundary the schedule uses, mapped to a token position by
/// encoding the prompt up to it and requiring that to be a prefix of the
/// prompt's tokens. Sorted, unique and strictly inside the prompt; a
/// boundary that fails is dropped with a diagnostic (it depends only on the
/// prompt, so hits and misses still split alike).
fn cut_positions(
    encode: impl Fn(&str) -> Result<Vec<u32>, ServeError>,
    prompt: &str,
    tokens: &[u32],
    boundaries: Option<PromptBoundaries>,
    schedule: Schedule,
) -> Vec<usize> {
    let Some(boundaries) = boundaries else {
        return Vec::new();
    };
    let wanted = match schedule {
        Schedule::ExactV1 => [
            ("shared_prefix", boundaries.shared_prefix_end),
            ("generation_header", boundaries.generation_header_start),
        ],
        Schedule::FastSharedSplitV1 => [
            ("shared_prefix", boundaries.shared_prefix_end),
            ("generation_header", None),
        ],
    };
    let mut cuts = Vec::new();
    for (name, offset) in wanted {
        let Some(offset) = offset else { continue };
        let prefix = match prompt.get(..offset) {
            Some(prefix) => encode(prefix).ok(),
            None => None,
        };
        match prefix {
            Some(ids)
                if !ids.is_empty() && ids.len() < tokens.len() && tokens.starts_with(&ids) =>
            {
                cuts.push(ids.len());
            }
            _ => tracing::warn!(
                target: "qwen_diag",
                "serve: glm5_next {name} boundary at byte {offset} is not a token prefix of the prompt; not split there"
            ),
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

/// Capture the session's state (committed exactly `prefix`) into the cache
/// if it is new, fits and is admitted. Optional: every failure leaves the
/// request running on the same schedule.
fn capture_into(
    cache: &mut SnapshotCache<Glm5NextSnapshot, CacheNamespace>,
    session: &Glm5NextSession<'_>,
    ctx: &MetalContext,
    namespace: CacheNamespace,
    prefix: &[u32],
    estimate: Option<u64>,
    reserve: u64,
) -> CaptureOutcome {
    if cache.entry_for_in(&namespace, prefix).is_some() {
        return CaptureOutcome::Present;
    }
    let Some(entry_bytes) =
        estimate.and_then(|payload| cache.strict_eligibility_in(&namespace, prefix, payload))
    else {
        return CaptureOutcome::Ineligible;
    };
    let Some(required) = entry_bytes.checked_add(reserve) else {
        return CaptureOutcome::Ineligible;
    };
    if let Err((reason, signals)) = super::admit_snapshot_capture(
        required,
        || ctx.memory_signals(),
        |bytes| cache.evict_for(bytes),
    ) {
        tracing::warn!(
            target: "qwen_diag",
            "serve: glm5_next snapshot not captured at {}: {reason:?} (required {required} bytes, signals {signals:?})",
            prefix.len()
        );
        return CaptureOutcome::Denied;
    }
    let snapshot = match session.capture_snapshot() {
        Ok(snapshot) => snapshot,
        Err(error) => {
            tracing::warn!(
                target: "qwen_diag",
                "serve: glm5_next snapshot capture at {} failed: {error}",
                prefix.len()
            );
            return CaptureOutcome::Failed;
        }
    };
    let Some(bytes) = SnapshotCache::<Glm5NextSnapshot, CacheNamespace>::entry_bytes(
        prefix.len(),
        snapshot.bytes() as u64,
    ) else {
        return CaptureOutcome::Ineligible;
    };
    if cache.insert_strict_in(namespace, prefix.to_vec(), snapshot, bytes) {
        CaptureOutcome::Captured
    } else {
        CaptureOutcome::Ineligible
    }
}

/// A session-creation failure: a typed memory refusal takes the shared
/// status table (pressure 503, telemetry and size 500); anything else
/// (geometry, invalid weights, GPU) is a 500 server error.
fn session_error(error: Glm5NextMetalError) -> ServeError {
    match &error {
        Glm5NextMetalError::MemoryAdmission { denied, .. } => {
            super::transport_memory::memory_refusal(&format!("{FAMILY} session"), denied)
        }
        _ => ServeError::server_error(format!("{FAMILY} session: {error}")),
    }
}

/// A typed memory-pressure refusal (503 `memory_admission_denied`), which
/// releasing cached snapshots may relieve; telemetry, size and overflow
/// refusals are not retried.
fn is_pressure_refusal(error: &ServeError) -> bool {
    error.status == 503 && error.code == Some("memory_admission_denied")
}

/// Release every unpinned cached snapshot except `keep` (memory pressure:
/// the cache is optional, a refused request is not).
fn release_snapshots<V>(
    cache: &mut SnapshotCache<V, CacheNamespace>,
    keep: Option<EntryId>,
) -> qwen_llm::snapshot_policy::Evicted {
    let pinned = keep.filter(|&id| cache.pin(id));
    let released = cache.evict_for(u64::MAX);
    if let Some(id) = pinned {
        cache.unpin(id);
    }
    if !released.is_empty() {
        tracing::info!(
            target: "qwen_diag",
            "serve: glm5_next snapshot cache released for memory pressure; entries={} freed_bytes={}",
            released.ids.len(),
            released.bytes,
        );
    }
    released
}

/// `attempt`, retried once after releasing cached snapshots (except
/// `keep`) when it is refused for memory pressure and something was
/// released.
fn with_snapshot_release<T, V>(
    cache: &mut SnapshotCache<V, CacheNamespace>,
    keep: Option<EntryId>,
    mut attempt: impl FnMut() -> Result<T, ServeError>,
) -> Result<T, ServeError> {
    match attempt() {
        Err(error) if is_pressure_refusal(&error) => {
            if release_snapshots(cache, keep).is_empty() {
                return Err(error);
            }
            attempt()
        }
        other => other,
    }
}

/// A session of the serve geometry reading prompts with `lineage`.
fn new_session<'w>(
    ctx: &'w MetalContext,
    weights: &'w Glm5NextWeights,
    prepared: &Prepared<'_>,
    cpu_reserve: u64,
    lineage: PackedLineage,
) -> Result<Glm5NextSession<'w>, ServeError> {
    let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
        ctx,
        weights,
        prepared.capacity,
        prepared.prefill_rows,
        cpu_reserve,
    )
    .map_err(session_error)?;
    session.set_packed_lineage(lineage).map_err(session_error)?;
    Ok(session)
}

/// The lineage this request reads its prompt with: `x_qwen.prefill_lineage`
/// if sent, else the backend default.
fn request_lineage(request: &ServeRequest, default: PackedLineage) -> PackedLineage {
    match request.prefill_lineage {
        Some(super::items::PrefillLineage::Exact) => PackedLineage::Exact,
        Some(super::items::PrefillLineage::Fast) => PackedLineage::Fast,
        None => default,
    }
}

/// Tokens reused from the live session: its extending prefix, but only when
/// the session reads prompts with the requested lineage (a live session
/// never mixes lineages; a mismatch starts a fresh session).
fn reuse_len(extending: usize, live: Option<PackedLineage>, requested: PackedLineage) -> usize {
    match live {
        Some(live) if live == requested => extending,
        None => extending,
        Some(_) => 0,
    }
}

fn lineage_name(lineage: PackedLineage) -> &'static str {
    match lineage {
        PackedLineage::Fast => "fast",
        PackedLineage::Exact => "exact",
    }
}

impl<'w, 'g> Glm5NextBackend<'w, 'g> {
    pub(super) fn new(
        ctx: &'w MetalContext,
        weights: &'w Glm5NextWeights,
        prepared: Prepared<'g>,
        model_id: String,
        idle_residency: std::time::Duration,
        snapshot_cache_plan: super::SnapshotCachePlan,
    ) -> Self {
        Self {
            ctx,
            weights,
            prepared,
            model_id,
            session: None,
            history: Vec::new(),
            prefix_reuse: qwen_llm::env_flag::read_default_on(PREFIX_REUSE_ENV),
            idle_residency: super::idle_residency::IdleResidency::new("glm5_next", idle_residency),
            lineage: PackedLineage::default(),
            cache: SnapshotCache::new(snapshot_cache_plan.bytes, snapshot_cache_plan.policy),
            snapshot_cache_plan,
            fast_snapshots: qwen_llm::env_flag::read_default_on(FAST_SNAPSHOTS_ENV),
            live_canonical: false,
            #[cfg(test)]
            last_prefill_logits: Vec::new(),
            #[cfg(test)]
            deny_captures: false,
        }
    }

    fn deny_captures(&self) -> bool {
        #[cfg(test)]
        {
            self.deny_captures
        }
        #[cfg(not(test))]
        {
            false
        }
    }

    /// The `serve limits` fragment for snapshots: the schedules that run
    /// (none without a cache budget).
    pub(super) fn describe_snapshots(&self) -> String {
        let schedules = match (self.cache.max_bytes() > 0, self.fast_snapshots) {
            (false, _) => "none",
            (true, true) => "exact,fast_shared_split",
            (true, false) => "exact",
        };
        format!(
            "{} snapshot_schedules={schedules}",
            self.snapshot_cache_plan
        )
    }

    fn fresh_session(
        &self,
        cpu_reserve: u64,
        lineage: PackedLineage,
    ) -> Result<Glm5NextSession<'w>, ServeError> {
        new_session(self.ctx, self.weights, &self.prepared, cpu_reserve, lineage)
    }

    /// One short prefill on a throwaway session before the listener accepts,
    /// so the first request is not charged the weights' first GPU use.
    pub(super) fn warm_up(&mut self) -> Result<f64> {
        let started = Instant::now();
        let tokens = self
            .prepared
            .artifact
            .tokenizer()
            .encode("[gMASK]<sop>", false)?
            .into_iter()
            .map(|id| u32::try_from(id).context("warm-up token id"))
            .collect::<Result<Vec<_>>>()?;
        let mut session = self
            .fresh_session(0, self.lineage)
            .map_err(|e| anyhow::anyhow!(e.message))?;
        session
            .prefill_packed_with_checkpoint(self.ctx, &tokens, &mut || {
                crate::shutdown::checkpoint().map_err(|e| e.to_string())
            })
            .context("GLM-5.3 serve warm-up")?;
        drop(session);
        // The server is about to accept: open the idle-residency window.
        self.idle_residency.note_activity();
        Ok(started.elapsed().as_secs_f64() * 1e3)
    }

    /// One request: reuse (live session or snapshot), prefill split at the
    /// schedule's cuts with optional captures, decode.
    fn generate_with(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        boundaries: Option<PromptBoundaries>,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        self.idle_residency.before_request();
        // Everything a client can get wrong is refused before the live
        // session is touched, so a 400 keeps the cached conversation.
        let maximum = request
            .max_output_tokens
            .unwrap_or(self.prepared.default_max);
        let mut sampler = Sampler::new(render::sampling(request))
            .map_err(|e| ServeError::invalid_request(None, format!("{FAMILY} sampler: {e}")))?;
        let vocab_size = self.prepared.artifact.config().vocab_size;
        let tokenizer = self.prepared.artifact.tokenizer();
        // The renderer writes [gMASK]<sop>; the glm4 tokenizer adds nothing.
        let tokens = decode_loop::encode_checked(tokenizer, prompt, false, vocab_size, FAMILY)?;
        let required =
            decode_loop::required_forwards(FAMILY, tokens.len(), maximum, self.prepared.capacity)?;
        // A tool block's memory is admitted as it grows (the tools partition),
        // not reserved here at its rarely approached worst case.
        let reserve = sink.transport_reserve_bytes();
        // Cached snapshots are optional: released before a pressure refusal.
        with_snapshot_release(&mut self.cache, None, || {
            super::transport_memory::admit_resident_transport(
                reserve,
                MetalContext::process_limit_bytes_remaining(),
            )
        })?;
        sink.tick().map_err(BackendFailure::Aborted)?;

        // Taken before the session is used and republished only for a state
        // the session is known to hold, so a failure leaves no stale history.
        let history = std::mem::take(&mut self.history);
        let committed = self.session.as_ref().map_or(0, Glm5NextSession::position);
        // A request may ask for the Exact lineage (`x_qwen.prefill_lineage`);
        // a live session is reused only by requests of its own lineage.
        let lineage = request_lineage(request, self.lineage);
        let live_lineage = self
            .session
            .as_ref()
            .and_then(Glm5NextSession::packed_lineage);
        let live = reuse_len(
            decode_loop::extending_prefix(&history, &tokens, committed, self.prefix_reuse),
            live_lineage,
            lineage,
        );
        let mut schedule = snapshot_schedule(self.cache.max_bytes(), self.fast_snapshots, lineage);
        let boundary_t0 = Instant::now();
        let cuts = schedule.map_or_else(Vec::new, |schedule| {
            cut_positions(
                |text| decode_loop::encode_checked(tokenizer, text, false, vocab_size, FAMILY),
                prompt,
                &tokens,
                boundaries,
                schedule,
            )
        });
        // CPU cost of verifying boundaries (one prefix encode per cut).
        let boundary_ms = boundary_t0.elapsed().as_secs_f64() * 1e3;
        // Fast snapshots exist only at a verified shared-prefix cut: without
        // one this request neither restores nor captures.
        if schedule == Some(Schedule::FastSharedSplitV1) && cuts.is_empty() {
            schedule = None;
        }
        let namespace = schedule.map(|schedule| CacheNamespace { lineage, schedule });
        // Exact restores any compatible prefix (segmentation-invariant);
        // Fast only a snapshot ending exactly at this request's cut. Used
        // only when it reaches past the live extension (ties keep live).
        let hit = namespace
            .filter(|_| self.prefix_reuse)
            .and_then(|namespace| match namespace.schedule {
                Schedule::ExactV1 => self.cache.peek_best_prefix_in(&namespace, &tokens),
                Schedule::FastSharedSplitV1 => {
                    self.cache.peek_exact_in(&namespace, &tokens[..cuts[0]])
                }
            })
            .filter(|hit| hit.prefix_len > live);
        let restore_t0 = Instant::now();
        let mut restored: Option<EntryId> = None;
        // Whether the session's state at the current position equals a
        // packed prefill of the same tokens in chunks from 0 (Fast captures
        // publish only such states; Exact does not depend on it).
        let mut canonical;
        let matched = match hit {
            Some(hit) => {
                if live_lineage != Some(lineage) {
                    self.session = None;
                    self.session = Some(with_snapshot_release(
                        &mut self.cache,
                        Some(hit.id),
                        || new_session(self.ctx, self.weights, &self.prepared, reserve, lineage),
                    )?);
                }
                let session = self.session.as_mut().expect("session present");
                match session.restore_snapshot(&hit.value) {
                    Ok(()) => {
                        self.cache.touch(hit.id);
                        restored = Some(hit.id);
                        // Fast entries are captured only from canonical
                        // states (below).
                        canonical = lineage == PackedLineage::Fast;
                        hit.prefix_len
                    }
                    Err(error) => {
                        tracing::warn!(
                            target: "qwen_diag",
                            "serve: glm5_next snapshot restore at {} failed: {error}; prefilling a fresh session",
                            hit.prefix_len
                        );
                        // Release the failed snapshot before allocating, so
                        // a pressure retry can reclaim it if evicted.
                        drop(hit);
                        self.session = None;
                        self.session = Some(with_snapshot_release(&mut self.cache, None, || {
                            new_session(self.ctx, self.weights, &self.prepared, reserve, lineage)
                        })?);
                        canonical = true;
                        0
                    }
                }
                // The restored value's Arc drops here, before any capture
                // admission could need its memory released.
            }
            None if live > 0 => {
                canonical = self.live_canonical;
                live
            }
            None => {
                // Release the old session's memory before allocating its
                // successor; its buffers and this request's transport
                // allowance are admitted as one requirement.
                self.session = None;
                self.session = Some(with_snapshot_release(&mut self.cache, None, || {
                    new_session(self.ctx, self.weights, &self.prepared, reserve, lineage)
                })?);
                canonical = true;
                0
            }
        };
        let restore_ms = if restored.is_some() {
            restore_t0.elapsed().as_secs_f64() * 1e3
        } else {
            0.0
        };
        let reuse_source = match (restored.is_some(), matched) {
            (true, _) => "snapshot",
            (false, 0) => "none",
            (false, _) => "live",
        };
        // Whatever happens next, the live state's provenance is recomputed.
        self.live_canonical = false;

        let ctx = self.ctx;
        let prefill_t0 = Instant::now();
        let mut capture_ms = 0.0;
        let mut captures = Vec::new();
        // The entry this request restored, reused through the live session
        // or captured last stays pinned while a later cut is captured, so
        // that capture's admission cannot evict it.
        let mut protect = restored.or_else(|| {
            let namespace = namespace?;
            let cut = cuts.iter().copied().filter(|&cut| cut <= matched).max()?;
            self.cache.entry_for_in(&namespace, &tokens[..cut])
        });
        let rows = self.prepared.prefill_rows.max(1);
        let mut position = matched;
        let mut logits = Vec::new();
        let ends: Vec<usize> = cuts
            .iter()
            .copied()
            .filter(|&cut| cut > matched)
            .chain([tokens.len()])
            .collect();
        for end in ends {
            // Chunks of this segment start at `position`: they stay on the
            // from-0 grid only if it is a multiple of the chunk rows.
            canonical &= position % rows == 0;
            let session = self.session.as_mut().expect("session present");
            let mut checkpoint_abort: Option<io::Error> = None;
            let result =
                session.prefill_packed_with_checkpoint(ctx, &tokens[position..end], &mut || {
                    sink.tick().map_err(|error| {
                        checkpoint_abort = Some(error);
                        "transport aborted during GLM-5.3 prefill".into()
                    })
                });
            logits = match result {
                Ok(logits) => logits,
                Err(error) => {
                    if let (Glm5NextMetalError::Cancelled(_), Some(abort)) =
                        (&error, checkpoint_abort)
                    {
                        // Cancelled at a chunk boundary: the committed prefix
                        // is consistent, so a retry of this prompt resumes
                        // from it.
                        let position = session.position();
                        self.history = tokens[..position].to_vec();
                        self.live_canonical = canonical && lineage == PackedLineage::Fast;
                        return Err(BackendFailure::Aborted(abort));
                    }
                    self.session = None;
                    return Err(
                        ServeError::server_error(format!("{FAMILY} prefill: {error}")).into(),
                    );
                }
            };
            position = end;
            if end == tokens.len() {
                break;
            }
            let namespace = namespace.expect("cuts come from a snapshot schedule");
            let capture_t0 = Instant::now();
            let pinned = protect.filter(|&id| self.cache.pin(id));
            let outcome = if namespace.schedule == Schedule::FastSharedSplitV1 && !canonical {
                // Live history segmented differently (an earlier cut, a
                // cancellation off the chunk grid, decoded tokens): this
                // state is not the one a miss would publish.
                CaptureOutcome::NonCanonical
            } else if self.deny_captures() {
                CaptureOutcome::Denied
            } else {
                capture_into(
                    &mut self.cache,
                    self.session.as_ref().expect("session present"),
                    ctx,
                    namespace,
                    &tokens[..end],
                    snapshot_bytes(&self.weights.config, end as u64),
                    reserve,
                )
            };
            if let Some(id) = pinned {
                self.cache.unpin(id);
            }
            capture_ms += capture_t0.elapsed().as_secs_f64() * 1e3;
            protect = self
                .cache
                .entry_for_in(&namespace, &tokens[..end])
                .or(protect);
            captures.push(format!("{end}:{}", outcome.name()));
        }
        #[cfg(test)]
        {
            self.last_prefill_logits = logits.iter().map(|v| v.to_bits()).collect();
        }
        let session = self.session.as_mut().expect("session present");
        if let Err(abort) = sink.tick() {
            // The prompt is consumed but no retry can extend it (an equal
            // prompt has no fresh row): release the session now.
            self.session = None;
            return Err(BackendFailure::Aborted(abort));
        }
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3 - capture_ms;
        let generation = decode_loop::decode_serial(
            decode_loop::DecodeRequest {
                family: FAMILY,
                logits,
                max_tokens: maximum,
                stop_tokens: &CHAT_STOPS,
                vocab_size,
            },
            &mut sampler,
            tokenizer,
            sink,
            |token| Ok(session.forward(ctx, token)?),
        );
        let generation = match generation {
            Ok(generation) => generation,
            Err(failure) => {
                // A decode abort or failure clears the live session: a
                // retry's prompt never extends a partially generated history.
                self.session = None;
                return Err(failure);
            }
        };
        if session.position() != tokens.len() + generation.transitions {
            self.session = None;
            return Err(ServeError::server_error(format!(
                "{FAMILY} consumed-prefix accounting mismatch"
            ))
            .into());
        }
        let mut history = history;
        history.clear();
        history.extend_from_slice(&tokens);
        history.extend(
            generation.tokens[..generation.transitions]
                .iter()
                .map(|&token| token as u32),
        );
        self.history = history;
        tracing::info!(
            target: "qwen_diag",
            "serve phases: family=glm5_next prefill_ms={prefill_ms:.1} reused_tokens={matched} reuse_source={reuse_source} restore_ms={restore_ms:.1} boundary_ms={boundary_ms:.1} prefill_tokens={} decode_ms={:.1} required_forwards={required} capacity={} transitions={} lineage={} snapshot_schedule={} cuts={cuts:?} captures=[{}] snapshot_capture_ms={capture_ms:.1} snapshot_cache_bytes={} snapshot_cache_entries={}",
            tokens.len() - matched,
            generation.wall_ms,
            self.prepared.capacity,
            generation.transitions,
            lineage_name(lineage),
            schedule.map_or("none", Schedule::name),
            captures.join(","),
            self.cache.indexed_bytes(),
            self.cache.len(),
        );
        Ok(super::outcome::finish_generation(
            tokens.len(),
            &generation,
            matched,
            restore_ms,
        ))
    }
}

impl GenerationBackend for Glm5NextBackend<'_, '_> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn request_profile(&self) -> super::request_profile::RequestProfile {
        super::request_profile::RequestProfile::Glm5Next {
            default_max_tokens: self.prepared.default_max,
            capacity: self.prepared.capacity,
            max_piece_bytes: self.prepared.max_piece_bytes,
        }
    }

    fn render_prepared(
        &self,
        request: &ServeRequest,
    ) -> Result<(String, Option<PromptBoundaries>), ServeError> {
        self.request_profile().render_prepared(request)
    }

    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        self.generate_with(request, prompt, None, sink)
    }

    fn generate_prepared(
        &mut self,
        prepared: Arc<PreparedResponse>,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        self.generate_with(
            &prepared.request,
            &prepared.prompt,
            prepared.boundaries,
            sink,
        )
    }

    fn idle(&mut self) {
        super::log_expired_snapshots("glm5_next", &self.cache.sweep());
        let weights = self.weights;
        self.idle_residency
            .on_idle(self.ctx, || weights.retained_buffers());
    }

    fn request_finished(&mut self) {
        self.idle_residency.request_finished();
    }
    fn request_failed_on_server(&mut self) {
        self.idle_residency.request_failed_on_server();
    }

    fn shutdown(&mut self) {
        self.idle_residency.shutdown();
        self.session = None;
        self.history.clear();
    }
}

#[cfg(test)]
mod tests;
