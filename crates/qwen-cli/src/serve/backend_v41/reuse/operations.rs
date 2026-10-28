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

/// The single-slot rule [`LivePool::prepare`] keeps at a budget of one.
#[cfg(test)]
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

/// Idle live states kept for exact-prefix reuse, in recency order. In-flight
/// states are never here: `take` removes the chosen one and only a request's
/// successful, eligible commit returns it. A budget of one state is the single
/// live slot: before admission, an idle state incompatible with the next
/// request is dropped. Larger pools keep other conversations' states and
/// release them by LRU on commit and, under memory pressure, through
/// `release_idle` (never the one the admitting request may take).
pub(super) struct LivePool<S> {
    entries: Vec<PoolEntry<S>>,
    max_states: usize,
    max_bytes: Option<u64>,
    clock: u64,
}

struct PoolEntry<S> {
    live: Live<S>,
    bytes: u64,
    used: u64,
}

impl<S> LivePool<S> {
    pub(super) fn new(max_states: usize, max_bytes: Option<u64>) -> Self {
        Self {
            entries: Vec::new(),
            max_states: max_states.max(1),
            max_bytes,
            clock: 0,
        }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(super) fn tokens(&self) -> Vec<Vec<u32>> {
        self.entries
            .iter()
            .map(|entry| entry.live.tokens.clone())
            .collect()
    }

    fn compatible(
        entry: &PoolEntry<S>,
        ids: &[u32],
        describe: &impl Fn(&S) -> (usize, bool),
    ) -> bool {
        let (position, healthy) = describe(&entry.live.state);
        live_prefix_len(&entry.live, ids, position, healthy).is_some()
    }

    /// Non-allocating preparation before output admission.
    pub(super) fn prepare(&mut self, ids: &[u32], describe: impl Fn(&S) -> (usize, bool)) {
        if self.max_states == 1 {
            self.entries
                .retain(|entry| Self::compatible(entry, ids, &describe));
        } else {
            self.entries.retain(|entry| describe(&entry.live.state).1);
        }
    }

    /// Remove the compatible idle state with the longest prefix (most
    /// recently used among equals).
    pub(super) fn take(
        &mut self,
        ids: &[u32],
        describe: impl Fn(&S) -> (usize, bool),
    ) -> Option<Live<S>> {
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| Self::compatible(entry, ids, &describe))
            .max_by_key(|(_, entry)| (entry.live.tokens.len(), entry.used))
            .map(|(index, _)| index)?;
        Some(self.entries.remove(index).live)
    }

    /// Retain a committed state, then drop least recently used ones until
    /// both budgets hold. A state alone above the byte budget is not kept.
    pub(super) fn commit(&mut self, live: Live<S>, bytes: u64) {
        self.clock += 1;
        self.entries.push(PoolEntry {
            live,
            bytes,
            used: self.clock,
        });
        while self.entries.len() > self.max_states
            || self
                .max_bytes
                .is_some_and(|max| self.entries.iter().map(|e| e.bytes).sum::<u64>() > max)
        {
            self.remove_lru(|_| true);
        }
    }

    fn remove_lru(&mut self, eligible: impl Fn(&PoolEntry<S>) -> bool) -> Option<u64> {
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| eligible(entry))
            .min_by_key(|(_, entry)| entry.used)
            .map(|(index, _)| index)?;
        Some(self.entries.remove(index).bytes)
    }

    /// Memory relief: release idle states the admitting request (`protect`)
    /// could not take, least recently used first, until `bytes` are freed.
    /// Returns (states, bytes) released.
    pub(super) fn release_idle(
        &mut self,
        bytes: u64,
        protect: &[u32],
        describe: impl Fn(&S) -> (usize, bool),
    ) -> (usize, u64) {
        let (mut states, mut released) = (0, 0u64);
        while released < bytes {
            let Some(freed) = self.remove_lru(|entry| !Self::compatible(entry, protect, &describe))
            else {
                break;
            };
            states += 1;
            released = released.saturating_add(freed);
        }
        (states, released)
    }
}
