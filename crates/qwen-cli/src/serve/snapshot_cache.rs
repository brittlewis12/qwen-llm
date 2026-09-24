//! Serve-owned RAM snapshot cache for families without a `PrefixCache`
//! (DeepSeek V4), keyed by exact token prefixes and governed by the shared
//! [`SnapshotPolicy`]. Values are shared on lookup so restoring a large
//! snapshot does not clone its state arenas.

use qwen_llm::snapshot_policy::{
    Clock, EntryId, Evicted, MonotonicClock, SnapshotPolicy, SnapshotPolicyConfig,
};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) struct SnapshotCache<V> {
    entries: HashMap<EntryId, (Vec<u32>, Arc<V>)>,
    policy: SnapshotPolicy,
}

impl<V> SnapshotCache<V> {
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

    /// Longest cached prefix of `tokens`, strictly shorter than the request
    /// (snapshots carry no observation, so at least one endpoint token must
    /// be prefilled to produce logits). Records a hit.
    pub(crate) fn best_prefix(&mut self, tokens: &[u32]) -> Option<(usize, Arc<V>)> {
        self.sweep();
        let (&id, (prefix, value)) = self
            .entries
            .iter()
            .filter(|(_, (prefix, _))| prefix.len() < tokens.len() && tokens.starts_with(prefix))
            .max_by_key(|(_, (prefix, _))| prefix.len())?;
        let hit = (prefix.len(), Arc::clone(value));
        self.policy.touch(id);
        Some(hit)
    }

    /// Id of the entry cached under exactly `tokens`.
    pub(crate) fn entry_for(&self, tokens: &[u32]) -> Option<EntryId> {
        self.entries
            .iter()
            .find(|(_, (prefix, _))| prefix == tokens)
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

    /// Entry bytes if `tokens` is new and can fit by evicting only unpinned
    /// entries; never changes cache contents.
    pub(crate) fn strict_eligibility(&self, tokens: &[u32], value_bytes: u64) -> Option<u64> {
        if self.entries.values().any(|(prefix, _)| prefix == tokens) {
            return None;
        }
        let bytes = Self::entry_bytes(tokens.len(), value_bytes)?;
        self.policy.fits_strict(bytes).then_some(bytes)
    }

    pub(crate) fn insert_strict(&mut self, tokens: Vec<u32>, value: V, bytes: u64) -> bool {
        self.insert_shared_strict(tokens, Arc::new(value), bytes)
    }

    /// [`Self::insert_strict`] for a value the caller keeps using (e.g. a
    /// snapshot promoted from disk that is restored right after insertion,
    /// or one handed to a background writer).
    pub(crate) fn insert_shared_strict(
        &mut self,
        tokens: Vec<u32>,
        value: Arc<V>,
        bytes: u64,
    ) -> bool {
        self.sweep();
        if !self.policy.fits_strict(bytes)
            || self.entries.values().any(|(prefix, _)| *prefix == tokens)
        {
            return false;
        }
        let id = self.policy.insert(bytes, tokens.len() as u64);
        self.entries.insert(id, (tokens, value));
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
