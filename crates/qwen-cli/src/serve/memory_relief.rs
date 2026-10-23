use qwen_llm::metal::MetalMemoryAdmissionReason;
use qwen_llm::snapshot_policy::Evicted;

/// Relieve process or host-availability pressure once. Device/host-ceiling
/// shortfalls cannot be relieved by releasing CPU snapshots.
///
/// The classifier keeps each operation's error shape local and supplies its
/// requested release size. Pass `u64::MAX` when that operation releases all
/// unpinned entries. The retry outcome may itself be a typed `Result`, so its
/// original refusal survives when no entry is evicted. The family labels the
/// shared log; cache-specific callbacks retain held-payload and keep rules.
pub(super) fn retry<T>(
    family: &str,
    first: T,
    classify: impl FnOnce(&T) -> Option<(MetalMemoryAdmissionReason, u64)>,
    release: impl FnOnce(u64) -> Evicted,
    retry: impl FnOnce() -> T,
) -> T {
    let Some((reason, requested_bytes)) = classify(&first) else {
        return first;
    };
    if !matches!(
        reason,
        MetalMemoryAdmissionReason::ProcessInsufficient
            | MetalMemoryAdmissionReason::HostAvailabilityInsufficient
    ) {
        return first;
    }
    let evicted = release(requested_bytes);
    if evicted.is_empty() {
        return first;
    }
    tracing::info!(
        target: "qwen_diag",
        "serve: memory relief family={family} released_entries={} released_bytes={} requested_bytes={requested_bytes}",
        evicted.ids.len(),
        evicted.bytes,
    );
    retry()
}

#[cfg(test)]
pub(super) fn test_evicted(bytes: u64) -> Evicted {
    let mut policy = qwen_llm::snapshot_policy::SnapshotPolicy::new(
        bytes,
        qwen_llm::snapshot_policy::SnapshotPolicyConfig::default(),
    );
    policy.insert(bytes, 1);
    policy.evict_for(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::snapshot_cache::SnapshotCache;
    use qwen_llm::snapshot_policy::SnapshotPolicyConfig;
    use std::sync::Arc;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum CacheState {
        Empty,
        AllHeld,
        SomeReleasable,
    }

    fn cache_for(state: CacheState) -> (SnapshotCache<u8>, Vec<Arc<u8>>) {
        let mut cache = SnapshotCache::new(1024, SnapshotPolicyConfig::LRU);
        let mut held = Vec::new();
        let count = match state {
            CacheState::Empty => 0,
            CacheState::AllHeld | CacheState::SomeReleasable => 2,
        };
        for index in 0..count {
            let value = Arc::new(index as u8);
            assert!(cache.insert_shared_strict_in((), vec![index as u32], value.clone(), 10));
            if state == CacheState::AllHeld || (state == CacheState::SomeReleasable && index == 0) {
                held.push(value);
            }
        }
        (cache, held)
    }

    #[test]
    fn relief_retry_table_covers_reasons_deficits_and_cache_states() {
        use MetalMemoryAdmissionReason as Reason;

        let reasons = [
            Reason::ProcessInsufficient,
            Reason::WorkingSetInsufficient,
            Reason::BothInsufficient,
            Reason::HostInsufficient,
            Reason::HostAvailabilityInsufficient,
            Reason::ProcessSignalUnavailable,
            Reason::HostSignalUnavailable,
            Reason::InvalidWorkingSetSignal,
            Reason::InvalidHostSignal,
            Reason::RequiredBytesOverflow,
            Reason::AdmittedWithProcessBudget,
            Reason::AdmittedWithHostBudget,
        ];
        for reason in reasons {
            for deficit in [None, Some(5)] {
                for cache_state in [
                    CacheState::Empty,
                    CacheState::AllHeld,
                    CacheState::SomeReleasable,
                ] {
                    let (mut cache, _held) = cache_for(cache_state);
                    let attempts = std::cell::Cell::new(0);
                    let requests = std::cell::RefCell::new(Vec::new());
                    let releases = std::cell::RefCell::new(Vec::new());
                    let result = retry(
                        "table",
                        {
                            attempts.set(1);
                            Err::<(), _>(reason)
                        },
                        |result| {
                            result
                                .as_ref()
                                .err()
                                .map(|reason| (*reason, deficit.unwrap_or(u64::MAX)))
                        },
                        |requested_bytes| {
                            requests.borrow_mut().push(requested_bytes);
                            let evicted = cache.evict_for(requested_bytes);
                            releases
                                .borrow_mut()
                                .push((evicted.ids.len(), evicted.bytes));
                            evicted
                        },
                        || {
                            attempts.set(attempts.get() + 1);
                            Ok(())
                        },
                    );

                    let relievable = matches!(
                        reason,
                        Reason::ProcessInsufficient | Reason::HostAvailabilityInsufficient
                    );
                    let some_releasable = cache_state == CacheState::SomeReleasable;
                    assert_eq!(requests.borrow().len(), usize::from(relievable));
                    if relievable {
                        assert_eq!(requests.borrow()[0], deficit.unwrap_or(u64::MAX));
                    }
                    assert_eq!(
                        attempts.get(),
                        if relievable && some_releasable { 2 } else { 1 },
                        "reason={reason:?} deficit={deficit:?} cache={cache_state:?}"
                    );
                    assert_eq!(result.is_ok(), relievable && some_releasable);
                    let expected_releases = if relievable && some_releasable {
                        vec![(1, 10)]
                    } else if relievable {
                        vec![(0, 0)]
                    } else {
                        Vec::new()
                    };
                    assert_eq!(releases.borrow().as_slice(), expected_releases);
                }
            }
        }
    }
}
