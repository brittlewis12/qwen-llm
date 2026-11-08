//! Resident state and immutable checkpoint ownership for Flash-Next serve.
use super::*;

pub(super) struct Checkpoint<S = Qwen4ExpTextSnapshot> {
    pub(super) snapshot: S,
    pub(super) logits: Vec<f32>,
}
impl<S> std::ops::Deref for Checkpoint<S> {
    type Target = S;
    fn deref(&self) -> &Self::Target {
        &self.snapshot
    }
}

pub(super) struct Live {
    pub(super) tokens: Vec<u32>,
    pub(super) conversation: Option<super::super::checkpoint_ladder::Conversation>,
}

#[derive(Clone, Copy)]
pub(super) struct TurnConfig {
    pub(super) live: bool,
    pub(super) endpoint_checkpoint: bool,
}
/// Default-on ordinary Flash-Next live continuation and lazy completed capture;
/// 0 restores reset/restore and eager completed snapshots as well as pre-header checkpoints.
const TURNS: &str = "QWEN_SERVE_FN_TURNS";
/// Default-on resident workspace reuse, exact token keys and copy-on-branch
/// completed checkpoints. 0 retains the original workspace lifecycle.
const LIVE: &str = "QWEN_SERVE_FN_LIVE";
/// Default-on post-header checkpoints with exact endpoint logits for reused
/// 512..2048-row appends. Body/header arithmetic stays unchanged; 0 restores
/// pre-header transcript checkpoints and header replay on retries.
const ENDPOINT: &str = "QWEN_SERVE_FN_ENDPOINT";
impl TurnConfig {
    pub(super) fn from_env() -> Self {
        let on = qwen_llm::env_flag::read_default_on(TURNS);
        Self {
            live: on && qwen_llm::env_flag::read_default_on(LIVE),
            endpoint_checkpoint: on && qwen_llm::env_flag::read_default_on(ENDPOINT),
        }
    }
}

pub(super) fn live_prefix_len(
    tokens: &[u32],
    prompt: &[u32],
    position: usize,
    healthy: bool,
) -> Option<usize> {
    (healthy && !tokens.is_empty() && position == tokens.len() && prompt.starts_with(tokens))
        .then_some(position)
}

pub(super) fn endpoint_checkpoint(
    enabled: bool,
    matched: usize,
    total: usize,
    boundary: Option<usize>,
    rows: Option<usize>,
) -> bool {
    enabled
        && matched > 0
        && (512..=2048).contains(&total.saturating_sub(matched))
        && rows.is_some_and(|rows| total - matched <= rows)
        && boundary.is_some_and(|cut| cut > matched && cut < total && total - cut <= 8)
}

pub(super) fn transcript_split(
    transcript_at: Option<usize>,
    matched_tokens: usize,
) -> Option<usize> {
    transcript_at.filter(|&boundary| boundary > matched_tokens)
}

/// Return Flash-Next's single workspace after request work, including aborts.
pub(super) fn with_request_workspace<T, E, W, R>(
    outcome: Result<T, E>,
    workspace: W,
    restore: impl FnOnce(W) -> Result<(), R>,
    on_restore_failure: impl FnOnce(R),
) -> Result<T, E> {
    if let Err(error) = restore(workspace) {
        on_restore_failure(error);
    }
    outcome
}

pub(super) fn admit_flash_next_snapshot_capture(
    bytes: u64,
    signals: impl Fn() -> qwen_llm::metal::MetalMemorySignals,
    evict_for: impl FnOnce(u64) -> qwen_llm::snapshot_policy::Evicted,
) -> Result<
    (),
    (
        super::super::SnapshotCaptureDenial,
        qwen_llm::metal::MetalMemorySignals,
    ),
> {
    super::super::admit_snapshot_capture_for(FAMILY, bytes, signals, evict_for)
}

#[cfg(test)]
pub(super) fn restore_best_prefix<V, R, E: std::fmt::Debug>(
    cache: &mut SnapshotCache<V>,
    tokens: &[u32],
    runner: &mut R,
    restore: impl FnOnce(&mut R, &V) -> Result<usize, E>,
    reset: impl FnOnce(&mut R) -> Result<(), E>,
) -> Result<usize, E> {
    let hit = cache.peek_best_prefix_in(&(), tokens);
    restore_prefix_hit(cache, hit, runner, restore, reset)
}

pub(super) fn restore_prefix_hit<V, R, E: std::fmt::Debug>(
    cache: &mut SnapshotCache<V>,
    hit: Option<super::super::snapshot_cache::PrefixHit<V>>,
    runner: &mut R,
    restore: impl FnOnce(&mut R, &V) -> Result<usize, E>,
    reset: impl FnOnce(&mut R) -> Result<(), E>,
) -> Result<usize, E> {
    let Some(hit) = hit else {
        return Ok(0);
    };
    match restore(runner, &hit.value) {
        Ok(position) if position == hit.prefix_len => {
            cache.touch(hit.id);
            Ok(hit.prefix_len)
        }
        outcome => {
            tracing::warn!(
                "serve: qwen4exp restore of {} tokens failed, cold prefilling: {:?}",
                hit.prefix_len,
                outcome.err()
            );
            cache.remove(hit.id);
            reset(runner)?;
            Ok(0)
        }
    }
}

/// Snapshot the runner's committed state under `tokens`; true when cached.
/// Pinned entries are never evicted to admit it.
pub(super) fn capture_snapshot(
    cache: &mut SnapshotCache<Checkpoint>,
    ctx: &MetalContext,
    runner: &Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    boundary: &'static str,
    control_cpu_reserve: u64,
) -> bool {
    capture_snapshot_impl(
        cache,
        ctx,
        runner,
        tokens,
        boundary,
        control_cpu_reserve,
        None,
        &[],
    )
}

pub(super) fn capture_snapshot_impl(
    cache: &mut SnapshotCache<Checkpoint>,
    ctx: &MetalContext,
    runner: &Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    boundary: &'static str,
    control_cpu_reserve: u64,
    conversation: Option<super::super::checkpoint_ladder::Conversation>,
    logits: &[f32],
) -> bool {
    if cache.max_bytes() == 0 {
        return false;
    }
    if runner.next_position() != tokens.len() {
        tracing::warn!("serve: qwen4exp refused a checkpoint under a different token extent");
        return false;
    }
    let payload_bytes = match runner.snapshot_bytes() {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!("serve: qwen4exp {boundary} snapshot estimate failed: {error}");
            return false;
        }
    };
    let Some(value_bytes) = payload_bytes.checked_add(std::mem::size_of_val(logits) as u64) else {
        return false;
    };
    let Some(entry_bytes) = (match conversation {
        Some(key) => cache.ladder_eligibility_in(key, &(), tokens, value_bytes),
        None => cache.strict_eligibility(tokens, value_bytes),
    }) else {
        tracing::info!(
            target: "qwen_diag",
            "serve: qwen4exp {boundary} snapshot not cached (already present or over budget); payload_bytes={payload_bytes} cache_budget_bytes={}",
            cache.max_bytes(),
        );
        return false;
    };
    if let Err((reason, signals)) = admit_flash_next_snapshot_capture(
        entry_bytes.saturating_add(control_cpu_reserve),
        || ctx.memory_signals(),
        |deficit| cache.evict_for(deficit),
    ) {
        tracing::warn!(
            "serve: qwen4exp {boundary} snapshot denied by memory headroom; reason={reason:?} payload_bytes={payload_bytes} metal_current_bytes={} metal_recommended_bytes={} process_remaining_bytes={:?}",
            signals.current_allocated_bytes,
            signals.recommended_max_bytes,
            signals.process_limit_remaining_bytes,
        );
        return false;
    }
    match runner.capture_snapshot() {
        Ok(snapshot) => {
            debug_assert_eq!(snapshot.payload_bytes(), payload_bytes);
            let inserted = match conversation {
                Some(key) => cache.insert_ladder_in(
                    key,
                    (),
                    tokens.to_vec(),
                    Checkpoint {
                        snapshot,
                        logits: logits.to_vec(),
                    },
                    entry_bytes,
                ),
                None => cache.insert_strict(
                    tokens.to_vec(),
                    Checkpoint {
                        snapshot,
                        logits: logits.to_vec(),
                    },
                    entry_bytes,
                ),
            };
            if !inserted {
                tracing::warn!("serve: qwen4exp {boundary} snapshot rejected at cache insertion");
            }
            inserted
        }
        Err(error) => {
            tracing::warn!("serve: qwen4exp {boundary} snapshot capture failed: {error}");
            false
        }
    }
}

/// Endpoint observation is owned/accounted by the same immutable cache entry.
pub(super) fn capture_with_logits(
    cache: &mut SnapshotCache<Checkpoint>,
    ctx: &MetalContext,
    runner: &Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    boundary: &'static str,
    reserve: u64,
    logits: &[f32],
) -> bool {
    capture_snapshot_impl(cache, ctx, runner, tokens, boundary, reserve, None, logits)
}

/// An exact endpoint is usable only with its own observation. State-only
/// checkpoints retain the original strictly-shorter prefix rule.
pub(super) fn checkpoint_candidate<V>(
    cache: &mut SnapshotCache<V>,
    tokens: &[u32],
    exact: bool,
    has_logits: impl FnOnce(&V) -> bool,
) -> Option<super::super::snapshot_cache::PrefixHit<V>> {
    if exact {
        cache
            .peek_exact_in(&(), tokens)
            .filter(|hit| has_logits(&hit.value))
            .or_else(|| cache.peek_best_prefix_in(&(), tokens))
    } else {
        cache.peek_best_prefix_in(&(), tokens)
    }
}

#[cfg(test)]
#[path = "reuse_tests.rs"]
mod tests;
