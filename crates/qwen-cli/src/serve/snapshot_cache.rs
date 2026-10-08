//! Serve-owned RAM snapshot cache for families without a `PrefixCache`
//! (DeepSeek V4, Flash-Next, GLM-5.3), keyed by exact token prefixes within
//! a compatibility namespace and governed by the shared [`SnapshotPolicy`]
//! under one byte budget. Values are shared on lookup so restoring a large
//! snapshot does not clone its state arenas.

use qwen_llm::snapshot_policy::{
    Clock, EntryId, Evicted, MonotonicClock, SnapshotPolicy, SnapshotPolicyConfig,
};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) struct SnapshotCache<V, N = ()> {
    entries: HashMap<EntryId, Entry<V, N>>,
    policy: SnapshotPolicy,
}

/// One cached snapshot: its compatibility namespace (what must match for a
/// restore to be valid beyond the token prefix, e.g. arithmetic lineage
/// and prefill schedule), its exact token key and the shared value.
struct Entry<V, N> {
    namespace: N,
    tokens: Vec<u32>,
    value: Arc<V>,
}

/// A compatible cached prefix found by [`SnapshotCache::peek_best_prefix_in`].
pub(crate) struct PrefixHit<V> {
    pub(crate) id: EntryId,
    pub(crate) prefix_len: usize,
    pub(crate) value: Arc<V>,
}

impl<V, N: PartialEq> SnapshotCache<V, N> {
    pub(crate) fn new(max_bytes: u64, config: SnapshotPolicyConfig) -> Self {
        Self::with_clock(max_bytes, config, Arc::new(MonotonicClock::new()))
    }

    pub(crate) fn with_clock(
        max_bytes: u64,
        config: SnapshotPolicyConfig,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            entries: HashMap::new(),
            policy: SnapshotPolicy::with_clock(max_bytes, config, clock),
        }
    }

    pub(crate) fn indexed_bytes(&self) -> u64 {
        self.policy.total_bytes()
    }

    pub(crate) fn max_bytes(&self) -> u64 {
        self.policy.max_bytes()
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Longest prefix of `tokens` cached in `namespace`, strictly shorter
    /// than the request (snapshots carry no observation, so at least one
    /// endpoint token must be prefilled to produce logits). Expires stale
    /// entries but records no hit: the caller decides whether to use it and
    /// then calls [`Self::touch`].
    pub(crate) fn peek_best_prefix_in(
        &mut self,
        namespace: &N,
        tokens: &[u32],
    ) -> Option<PrefixHit<V>> {
        self.sweep();
        let (&id, entry) = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry.namespace == *namespace
                    && entry.tokens.len() < tokens.len()
                    && tokens.starts_with(&entry.tokens)
            })
            .max_by_key(|(_, entry)| entry.tokens.len())?;
        Some(PrefixHit {
            id,
            prefix_len: entry.tokens.len(),
            value: Arc::clone(&entry.value),
        })
    }

    /// Record a use of an entry (frecency and idle expiry).
    pub(crate) fn touch(&mut self, id: EntryId) {
        self.policy.touch(id);
    }

    /// [`Self::peek_best_prefix_in`] that records the hit.
    pub(crate) fn best_prefix_in(
        &mut self,
        namespace: &N,
        tokens: &[u32],
    ) -> Option<(usize, Arc<V>)> {
        let hit = self.peek_best_prefix_in(namespace, tokens)?;
        self.touch(hit.id);
        Some((hit.prefix_len, hit.value))
    }

    /// Id of the entry cached in `namespace` under exactly `tokens`.
    pub(crate) fn entry_for_in(&self, namespace: &N, tokens: &[u32]) -> Option<EntryId> {
        self.entries
            .iter()
            .find(|(_, entry)| entry.namespace == *namespace && entry.tokens == tokens)
            .map(|(&id, _)| id)
    }

    /// Exclude an entry from eviction and expiry until [`Self::unpin`].
    /// Returns false when the entry is no longer indexed.
    pub(crate) fn pin(&mut self, id: EntryId) -> bool {
        self.policy.pin(id)
    }

    pub(crate) fn unpin(&mut self, id: EntryId) {
        self.policy.unpin(id);
    }

    pub(crate) fn entry_bytes(prefix_len: usize, value_bytes: u64) -> Option<u64> {
        value_bytes.checked_add((prefix_len as u64).checked_mul(size_of::<u32>() as u64)?)
    }

    /// Entry bytes if `tokens` is new in `namespace` and can fit by evicting
    /// only unpinned entries; never changes cache contents.
    pub(crate) fn strict_eligibility_in(
        &self,
        namespace: &N,
        tokens: &[u32],
        value_bytes: u64,
    ) -> Option<u64> {
        if self.entry_for_in(namespace, tokens).is_some() {
            return None;
        }
        let bytes = Self::entry_bytes(tokens.len(), value_bytes)?;
        self.policy.fits_strict(bytes).then_some(bytes)
    }

    pub(crate) fn insert_strict_in(
        &mut self,
        namespace: N,
        tokens: Vec<u32>,
        value: V,
        bytes: u64,
    ) -> bool {
        self.insert_shared_strict_in(namespace, tokens, Arc::new(value), bytes)
    }

    /// [`Self::insert_strict_in`] for a value the caller keeps using (e.g. a
    /// snapshot promoted from disk that is restored right after insertion,
    /// or one handed to a background writer).
    pub(crate) fn insert_shared_strict_in(
        &mut self,
        namespace: N,
        tokens: Vec<u32>,
        value: Arc<V>,
        bytes: u64,
    ) -> bool {
        self.sweep();
        if !self.policy.fits_strict(bytes) || self.entry_for_in(&namespace, &tokens).is_some() {
            return false;
        }
        let id = self.policy.insert(bytes, tokens.len() as u64);
        self.entries.insert(
            id,
            Entry {
                namespace,
                tokens,
                value,
            },
        );
        let evicted = self.policy.evict_to_budget(Some(id));
        self.drop_entries(&evicted);
        true
    }

    /// Drop entries past the policy's idle TTL or maximum age.
    pub(crate) fn sweep(&mut self) -> Evicted {
        let evicted = self.policy.sweep();
        self.drop_entries(&evicted);
        evicted
    }

    /// Evict unpinned entries by rank until `bytes` are released.
    pub(crate) fn evict_for(&mut self, bytes: u64) -> Evicted {
        let evicted = self.policy.evict_for(bytes);
        self.drop_entries(&evicted);
        evicted
    }

    fn drop_entries(&mut self, evicted: &Evicted) {
        for id in &evicted.ids {
            self.entries.remove(id);
        }
    }
}

/// The single-namespace API, for families whose snapshots are compatible
/// whenever their token prefixes match.
impl<V> SnapshotCache<V> {
    /// Longest cached prefix of `tokens`, strictly shorter than the request.
    /// Records a hit.
    pub(crate) fn best_prefix(&mut self, tokens: &[u32]) -> Option<(usize, Arc<V>)> {
        self.best_prefix_in(&(), tokens)
    }

    /// Id of the entry cached under exactly `tokens`.
    pub(crate) fn entry_for(&self, tokens: &[u32]) -> Option<EntryId> {
        self.entry_for_in(&(), tokens)
    }

    /// Entry bytes if `tokens` is new and can fit by evicting only unpinned
    /// entries; never changes cache contents.
    pub(crate) fn strict_eligibility(&self, tokens: &[u32], value_bytes: u64) -> Option<u64> {
        self.strict_eligibility_in(&(), tokens, value_bytes)
    }

    pub(crate) fn insert_strict(&mut self, tokens: Vec<u32>, value: V, bytes: u64) -> bool {
        self.insert_strict_in((), tokens, value, bytes)
    }

    /// [`Self::insert_strict`] for a value the caller keeps using.
    pub(crate) fn insert_shared_strict(
        &mut self,
        tokens: Vec<u32>,
        value: Arc<V>,
        bytes: u64,
    ) -> bool {
        self.insert_shared_strict_in((), tokens, value, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen_llm::snapshot_policy::FakeClock;
    use std::time::Duration;

    fn lru(max_bytes: u64) -> SnapshotCache<&'static str> {
        SnapshotCache::new(max_bytes, SnapshotPolicyConfig::LRU)
    }

    #[test]
    fn prefers_longest_strict_prefix() {
        for config in [SnapshotPolicyConfig::LRU, SnapshotPolicyConfig::default()] {
            let mut cache = SnapshotCache::new(1024, config);
            let bytes = cache.strict_eligibility(&[1], 5).unwrap();
            assert!(cache.insert_strict(vec![1], "short", bytes));
            let bytes = cache.strict_eligibility(&[1, 2], 4).unwrap();
            assert!(cache.insert_strict(vec![1, 2], "long", bytes));
            let (prefix_len, value) = cache.best_prefix(&[1, 2, 3]).unwrap();
            assert_eq!(prefix_len, 2);
            assert_eq!(*value, "long");
            assert!(cache.best_prefix(&[1, 2]).is_some_and(|hit| hit.0 == 1));
            assert!(cache.best_prefix(&[1]).is_none());
            assert!(cache.best_prefix(&[2, 1]).is_none());
            assert!(cache.strict_eligibility(&[1, 2], 4).is_none()); // duplicate
        }
    }

    #[test]
    fn accounts_bytes_and_evicts_lru() {
        let mut cache = lru(20);
        let bytes = cache.strict_eligibility(&[1], 6).unwrap();
        assert!(cache.insert_strict(vec![1], "first", bytes)); // 10 bytes with key
        let bytes = cache.strict_eligibility(&[2], 6).unwrap();
        assert!(cache.insert_strict(vec![2], "second", bytes));
        assert_eq!(cache.indexed_bytes(), 20);
        assert!(cache.best_prefix(&[1, 9]).is_some()); // first is now MRU
        let bytes = cache.strict_eligibility(&[3], 6).unwrap();
        assert!(cache.insert_strict(vec![3], "third", bytes));
        assert_eq!(cache.indexed_bytes(), 20);
        assert!(cache.best_prefix(&[2, 9]).is_none());
        assert!(cache.best_prefix(&[1, 9]).is_some());
        assert!(cache.strict_eligibility(&[4], 17).is_none());
        assert_eq!(cache.indexed_bytes(), 20);
    }

    #[test]
    fn eligibility_does_not_evict() {
        let mut cache = lru(20);
        let bytes = cache.strict_eligibility(&[1], 6).unwrap();
        assert!(cache.insert_strict(vec![1], "first", bytes));
        let bytes = cache.strict_eligibility(&[2], 6).unwrap();
        assert!(cache.insert_strict(vec![2], "second", bytes));

        assert!(cache.strict_eligibility(&[3], 6).is_some());
        assert!(cache.strict_eligibility(&[4], 17).is_none());
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.indexed_bytes(), 20);
        assert!(cache.best_prefix(&[1, 9]).is_some());
        assert!(cache.best_prefix(&[2, 9]).is_some());
    }

    /// A hot unrelated entry H, then a transcript T and a completed C from
    /// one request; the budget fits H+T or T+C but not all three. Frecency
    /// alone evicts the new, rarely used T; pinning T evicts H instead.
    #[test]
    fn pinned_transcript_survives_completed_admission_over_a_hot_entry() {
        for pin_transcript in [false, true] {
            let clock = FakeClock::default();
            let mut cache = SnapshotCache::with_clock(
                30,
                SnapshotPolicyConfig::default(),
                Arc::new(clock.clone()),
            );
            let bytes = cache.strict_eligibility(&[9], 6).unwrap(); // 10 with key
            assert!(cache.insert_strict(vec![9], "hot", bytes));
            for _ in 0..5 {
                assert!(cache.best_prefix(&[9, 0]).is_some());
            }
            let bytes = cache.strict_eligibility(&[1, 2], 6).unwrap(); // 14
            assert!(cache.insert_strict(vec![1, 2], "transcript", bytes));
            let transcript = cache.entry_for(&[1, 2]).unwrap();
            assert_eq!(cache.entry_for(&[1, 2, 3]), None);

            let pinned = pin_transcript.then(|| cache.pin(transcript));
            let bytes = cache.strict_eligibility(&[1, 2, 3], 3).unwrap(); // 15
            assert!(cache.insert_strict(vec![1, 2, 3], "completed", bytes));
            if pinned == Some(true) {
                cache.unpin(transcript);
            }

            assert!(cache.entry_for(&[1, 2, 3]).is_some());
            assert_eq!(cache.entry_for(&[1, 2]).is_some(), pin_transcript);
            assert_eq!(cache.entry_for(&[9]).is_some(), !pin_transcript);
        }
    }

    /// Identical tokens in two namespaces are two entries under one budget;
    /// lookup, exact lookup and duplicate detection never cross namespaces.
    #[test]
    fn namespaces_share_one_budget_and_never_cross() {
        let mut cache: SnapshotCache<&'static str, u8> =
            SnapshotCache::new(30, SnapshotPolicyConfig::LRU);
        let bytes = cache.strict_eligibility_in(&1, &[1, 2], 6).unwrap(); // 14 with key
        assert!(cache.insert_strict_in(1, vec![1, 2], "fast", bytes));
        assert!(cache.strict_eligibility_in(&1, &[1, 2], 6).is_none()); // duplicate
        let bytes = cache.strict_eligibility_in(&2, &[1, 2], 6).unwrap();
        assert!(cache.insert_strict_in(2, vec![1, 2], "exact", bytes));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.indexed_bytes(), 28);
        assert_eq!(*cache.best_prefix_in(&1, &[1, 2, 3]).unwrap().1, "fast");
        assert_eq!(*cache.best_prefix_in(&2, &[1, 2, 3]).unwrap().1, "exact");
        assert!(cache.best_prefix_in(&3, &[1, 2, 3]).is_none());
        assert!(cache.entry_for_in(&3, &[1, 2]).is_none());
        assert_ne!(
            cache.entry_for_in(&1, &[1, 2]),
            cache.entry_for_in(&2, &[1, 2])
        );
        // One budget: a third entry evicts the least recently used of either.
        let bytes = cache.strict_eligibility_in(&3, &[7], 6).unwrap();
        assert!(cache.insert_strict_in(3, vec![7], "other", bytes));
        assert!(cache.entry_for_in(&1, &[1, 2]).is_none());
        assert!(cache.entry_for_in(&2, &[1, 2]).is_some());
    }

    /// A peek records no use: under LRU the peeked entry is still the
    /// eviction victim until the caller touches it.
    #[test]
    fn peek_does_not_touch_until_the_hit_is_used() {
        for touch in [false, true] {
            let mut cache = lru(20);
            assert!(cache.insert_strict(vec![1], "first", 10));
            assert!(cache.insert_strict(vec![2], "second", 10));
            let hit = cache.peek_best_prefix_in(&(), &[1, 9]).unwrap();
            assert_eq!((hit.prefix_len, *hit.value), (1, "first"));
            if touch {
                cache.touch(hit.id);
            }
            drop(hit);
            assert!(cache.insert_strict(vec![3], "third", 10));
            assert_eq!(cache.entry_for(&[1]).is_some(), touch);
            assert_eq!(cache.entry_for(&[2]).is_some(), !touch);
        }
    }

    #[test]
    fn expiry_and_pressure_eviction_drop_values() {
        let clock = FakeClock::default();
        let mut cache = SnapshotCache::with_clock(
            1024,
            SnapshotPolicyConfig::default(),
            Arc::new(clock.clone()),
        );
        assert!(cache.insert_strict(vec![1], "a", 10));
        assert!(cache.insert_strict(vec![2], "b", 10));
        clock.advance(Duration::from_secs(3000));
        assert!(cache.best_prefix(&[1, 9]).is_some());
        clock.advance(Duration::from_secs(1000)); // [2] idle 4000 s, [1] 1000 s
        assert_eq!(cache.sweep().bytes, 10);
        assert!(cache.best_prefix(&[2, 9]).is_none());
        assert_eq!(cache.evict_for(1).bytes, 10);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.indexed_bytes(), 0);
    }
}
