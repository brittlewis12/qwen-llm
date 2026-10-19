use super::super::super::{memory_relief, output_memory, snapshot_cache::PrefixHit};
use super::*;
use qwen_llm::deepseek_v41::forward::StateError;

pub(super) fn grow_live<S>(
    mut live: Live<S>,
    grow: impl FnOnce(&mut S) -> Result<(), StateError>,
) -> Result<Option<Live<S>>, ServeError> {
    match grow(&mut live.state) {
        Ok(()) => Ok(Some(live)),
        Err(error) if error.memory_admission().is_some() => {
            // Cold allocation need not coexist with the old KV and growth copy.
            drop(live);
            tracing::debug!("V4.1 live growth refused; cold prefilling: {error}");
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
                        .map(|denied| (denied.reason, u64::MAX))
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
    headroom: impl FnOnce() -> Option<u64>,
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
    // Include forward::capture_snapshot's CPU bookkeeping allowance as well
    // as the caller's outstanding reserve, without charging either to Metal.
    let Some(required) = entry_bytes
        .checked_add(4096)
        .and_then(|bytes| bytes.checked_add(reserve))
    else {
        return;
    };
    if let Err(error) = output_memory::admit_growth(required, 0, headroom()) {
        tracing::debug!("V4.1 checkpoint skipped: {}", error.message);
        return;
    }
    match capture(payload_bytes) {
        Ok(checkpoint) => {
            cache.insert_strict(ids.to_vec(), checkpoint, entry_bytes);
        }
        Err(error) => tracing::debug!("V4.1 checkpoint skipped: {error}"),
    }
}
