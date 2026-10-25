use super::super::super::{admit_snapshot_capture_for, memory_relief, snapshot_cache::PrefixHit};
use super::*;
use qwen_llm::deepseek_v41::forward::StateError;

/// Pure causal compatibility check; neither a common prefix of a divergent
/// history nor an equal prompt without endpoint logits is reusable live state.
pub(super) fn live_prefix_len<S>(
    live: &Live<S>,
    ids: &[u32],
    position: usize,
    healthy: bool,
) -> Option<usize> {
    (healthy
        && position == live.tokens.len()
        && !live.tokens.is_empty()
        && ids.starts_with(&live.tokens)
        && (ids.len() > position || !live.logits.is_empty()))
    .then_some(position)
}

pub(super) fn discard_incompatible<S>(
    live: &mut Option<Live<S>>,
    ids: &[u32],
    describe: impl FnOnce(&S) -> (usize, bool),
) {
    if live.as_ref().is_some_and(|live| {
        let (position, healthy) = describe(&live.state);
        live_prefix_len(live, ids, position, healthy).is_none()
    }) {
        drop(live.take());
    }
}

pub(super) fn live_then_snapshot<S>(
    live: Option<Live<S>>,
    grow: impl FnOnce(&mut S) -> Result<(), StateError>,
    snapshot: impl FnOnce() -> Result<Option<Restored<S>>, ServeError>,
) -> Result<Option<Restored<S>>, ServeError> {
    if let Some(live) = live
        && let Some(live) = grow_live(live, grow)?
    {
        return Ok(Some(Restored {
            live,
            source: "live_session",
        }));
    }
    snapshot()
}

pub(super) fn snapshot_candidate<V>(
    cache: &mut SnapshotCache<V>,
    ids: &[u32],
    has_logits: impl FnOnce(&V) -> bool,
) -> Option<PrefixHit<V>> {
    cache
        .peek_exact_in(&(), ids)
        .filter(|hit| has_logits(&hit.value))
        .or_else(|| cache.peek_best_prefix_in(&(), ids))
}

pub(super) fn grow_live<S>(
    mut live: Live<S>,
    grow: impl FnOnce(&mut S) -> Result<(), StateError>,
) -> Result<Option<Live<S>>, ServeError> {
    match grow(&mut live.state) {
        Ok(()) => Ok(Some(live)),
        Err(error) if error.memory_admission().is_some() => {
            // Cold allocation need not coexist with the old KV and growth copy.
            drop(live);
            tracing::debug!("V4.1 live growth refused; trying snapshot then cold: {error}");
            Ok(None)
        }
        Err(error) => Err(state_error(error)),
    }
}

pub(super) fn restore_candidate<S, V>(
    cache: &mut SnapshotCache<V>,
    hit: PrefixHit<V>,
    create: impl FnOnce() -> Result<S, StateError>,
    restore: impl FnOnce(&mut S, &V) -> Result<(), StateError>,
) -> Result<Option<(S, PrefixHit<V>)>, ServeError> {
    let mut state = match create() {
        Ok(state) => state,
        Err(error) => {
            // Release the lookup pin before process-only cache relief. Without
            // actual relief, cold would repeat the same failed allocation.
            drop(hit);
            return memory_relief::retry(
                "deepseek_v41",
                Err(error),
                |result| {
                    result
                        .as_ref()
                        .err()?
                        .memory_admission()
                        .map(|denied| (denied.reason(), u64::MAX))
                },
                |bytes| cache.evict_for(bytes),
                || Ok(None),
            )
            .map_err(state_error);
        }
    };
    if let Err(error) = restore(&mut state, &hit.value) {
        // Allocation succeeded. Discard possibly poisoned restore state before
        // asking the normal cold path to allocate, without restore's extras.
        drop(state);
        drop(hit);
        if error.memory_admission().is_some() {
            tracing::debug!("V4.1 snapshot restore refused; cold prefilling: {error}");
            return Ok(None);
        }
        return Err(state_error(error));
    }
    cache.touch(hit.id);
    Ok(Some((state, hit)))
}

pub(super) fn capture_checkpoint<V>(
    cache: &mut SnapshotCache<V>,
    ids: &[u32],
    logits_bytes: u64,
    reserve: u64,
    signals: impl Fn() -> MetalMemorySignals,
    measure: impl FnOnce() -> Result<usize, StateError>,
    capture: impl FnOnce(usize) -> Result<V, StateError>,
) {
    cache.sweep();
    if cache.max_bytes() == 0 || cache.entry_for(ids).is_some() {
        return;
    }
    let payload_bytes = match measure() {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::debug!("V4.1 checkpoint measurement skipped: {error}");
            return;
        }
    };
    let Some(value_bytes) = (payload_bytes as u64).checked_add(logits_bytes) else {
        return;
    };
    let Some(entry_bytes) = cache.strict_eligibility(ids, value_bytes) else {
        return;
    };
    let Some(required) = entry_bytes.checked_add(reserve) else {
        return;
    };
    if let Err((reason, signals)) =
        admit_snapshot_capture_for("deepseek_v41", required, signals, |bytes| {
            cache.evict_for(bytes)
        })
    {
        tracing::warn!(
            target: "qwen_diag",
            "serve: deepseek_v41 snapshot not captured at {}: {reason:?} (required {required} bytes, signals {signals:?})",
            ids.len()
        );
        return;
    }
    match capture(payload_bytes) {
        Ok(checkpoint) => {
            cache.insert_strict(ids.to_vec(), checkpoint, entry_bytes);
        }
        Err(error) => tracing::debug!("V4.1 checkpoint skipped: {error}"),
    }
}
