//! Serve-owned RAM snapshot cache for families without a `PrefixCache`
//! (DeepSeek V4, Flash-Next, GLM-5.3), keyed by exact token prefixes within
//! a compatibility namespace and governed by the shared [`SnapshotPolicy`]
//! under one byte budget. Values are shared on lookup so restoring a large
//! snapshot does not clone its state arenas.

use qwen_llm::snapshot_policy::{
    Clock, EntryId, Evicted, MonotonicClock, SnapshotPolicy, SnapshotPolicyConfig,
    SnapshotPolicyPinGuard,
};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) struct SnapshotCache<V, N = ()> {
    entries: HashMap<EntryId, Entry<V, N>>,
    policy: SnapshotPolicy,
    ladder: super::checkpoint_ladder::Ladder,
    tips: HashMap<super::checkpoint_ladder::Conversation, Vec<EntryId>>,
}

/// One cached snapshot: its compatibility namespace (what must match for a
/// restore to be valid beyond the token prefix, e.g. arithmetic lineage
/// and prefill schedule), its exact token key and the shared value.
struct Entry<V, N> {
    namespace: N,
    tokens: Vec<u32>,
    value: Arc<V>,
    bytes: u64,
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
            ladder: Default::default(),
            tips: Default::default(),
        }
    }

    pub(super) fn with_checkpoint_ladder(
        mut self,
        config: super::checkpoint_ladder::Config,
    ) -> Self {
        self.ladder = super::checkpoint_ladder::Ladder::new(config, self.max_bytes());
        self
    }

    pub(super) fn ladder_enabled(&self) -> bool {
        self.ladder.enabled()
    }
    pub(super) fn ladder_status(&self) -> (usize, u64, u64) {
        (self.ladder.len(), self.ladder.bytes(), self.ladder.budget())
    }

    /// Keep the latest transcript/completed pair in addition to the spaced
    /// ladder. Old turn tips must not consume the whole cache indefinitely.
    /// Pins/held restores defer pruning; pressure and expiry always apply.
    pub(super) fn record_tip(
        &mut self,
        conversation: Option<super::checkpoint_ladder::Conversation>,
        namespace: &N,
        tokens: &[u32],
    ) {
        if !self.ladder_enabled() || tokens.len() < 4096 {
            return;
        }
        let Some(key) = conversation else {
            return;
        };
        let Some(id) = self.entry_for_in(namespace, tokens) else {
            return;
        };
        // Reuse an already-paid copy at spaced turn boundaries. This keeps
        // coverage advancing during many short appends with ZERO extra copy.
        if !self.ladder.contains(id) {
            let bytes = self.entries[&id].bytes;
            if let Some(victims) = self.ladder_plan(key, namespace, tokens, bytes) {
                for victim in victims {
                    self.remove(victim);
                }
                self.ladder.inserted(id, key, tokens.len(), bytes);
            }
        }
        let tips = self.tips.entry(key).or_default();
        tips.retain(|&other| other != id);
        tips.push(id);
        // Keep the original deeper branch hot while serving an older fork.
        let mut deepest = tips.clone();
        deepest.sort_by_key(|id| self.entries.get(id).map_or(0, |e| e.tokens.len()));
        let deepest: Vec<_> = deepest.into_iter().rev().take(2).collect();
        let old = tips
            .iter()
            .take(tips.len().saturating_sub(2))
            .copied()
            .filter(|id| !deepest.contains(id))
            .collect::<Vec<_>>();
        for id in old {
            let covered = self.entries.get(&id).is_some_and(|tip| {
                self.entries.iter().any(|(anchor_id, anchor)| {
                    self.ladder.contains(*anchor_id)
                        && anchor.namespace == tip.namespace
                        && tip.tokens.starts_with(&anchor.tokens)
                })
            });
            if covered
                && !self.ladder.contains(id)
                && !self.policy.is_pinned(id)
                && self
                    .entries
                    .get(&id)
                    .is_some_and(|e| Arc::strong_count(&e.value) == 1)
            {
                self.remove(id);
            }
        }
    }

    fn sync_retention(&mut self) {
        self.ladder.retain(|id| self.entries.contains_key(&id));
        self.tips.retain(|_, ids| {
            ids.retain(|id| self.entries.contains_key(id));
            !ids.is_empty()
        });
    }

    fn ladder_plan(
        &self,
        conversation: super::checkpoint_ladder::Conversation,
        namespace: &N,
        tokens: &[u32],
        bytes: u64,
    ) -> Option<Vec<EntryId>> {
        let parent = self
            .entries
            .iter()
            .filter(|(id, e)| {
                self.ladder.contains(**id)
                    && e.namespace == *namespace
                    && tokens.starts_with(&e.tokens)
            })
            .map(|(_, e)| e.tokens.len())
            .max()
            .unwrap_or(0);
        let ranks = self.policy.ranked_best_first();
        self.ladder.plan(
            conversation,
            tokens.len(),
            bytes,
            parent,
            |id| {
                self.entries
                    .get(&id)
                    .is_some_and(|e| Arc::strong_count(&e.value) == 1)
                    && !self.policy.is_pinned(id)
            },
            |id| {
                ranks
                    .iter()
                    .position(|&other| other == id)
                    .unwrap_or(usize::MAX)
            },
        )
    }

    /// Pure admission probe: failed/denied copies never prune old coverage.
    pub(super) fn ladder_eligibility_in(
        &self,
        conversation: super::checkpoint_ladder::Conversation,
        namespace: &N,
        tokens: &[u32],
        payload: u64,
    ) -> Option<u64> {
        let bytes = self.strict_eligibility_in(namespace, tokens, payload)?;
        self.ladder_plan(conversation, namespace, tokens, bytes)?;
        Some(bytes)
    }

    pub(super) fn insert_ladder_in(
        &mut self,
        conversation: super::checkpoint_ladder::Conversation,
        namespace: N,
        tokens: Vec<u32>,
        value: V,
        bytes: u64,
    ) -> bool {
        self.sweep();
        let Some(victims) = self.ladder_plan(conversation, &namespace, &tokens, bytes) else {
            return false;
        };
        let position = tokens.len();
        // Payload is already copied. Insert first; prune only after success.
        // All entries, including these optional points, share the parent budget.
        let value = Arc::new(value);
        if !self.insert_shared_strict_in(namespace, tokens, Arc::clone(&value), bytes) {
            return false;
        }
        let id = *self
            .entries
            .iter()
            .find(|(_, entry)| Arc::ptr_eq(&entry.value, &value))
            .expect("inserted entry")
            .0;
        for victim in victims {
            self.remove(victim);
        }
        self.ladder.inserted(id, conversation, position, bytes);
        self.sync_retention();
        true
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

    /// The entry cached in `namespace` under exactly `tokens`, without
    /// recording a hit (expires stale entries first).
    pub(crate) fn peek_exact_in(&mut self, namespace: &N, tokens: &[u32]) -> Option<PrefixHit<V>> {
        self.sweep();
        let (&id, entry) = self
            .entries
            .iter()
            .find(|(_, entry)| entry.namespace == *namespace && entry.tokens == tokens)?;
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
                bytes,
            },
        );
        // Protect spaced history from a stream of ordinary tip snapshots when
        // both can coexist. Under pressure the normal policy wins instead.
        let protected: Vec<_> = self
            .entries
            .keys()
            .copied()
            .filter(|&id| self.ladder.contains(id) && !self.policy.is_pinned(id))
            .collect();
        // Use the policy's accounted payload bytes, not the token-key size.
        let evicted = {
            let mut pins = SnapshotPolicyPinGuard::new(&mut self.policy, protected);
            if pins.fits_strict(bytes) {
                Some(pins.evict_to_budget(Some(id)))
            } else {
                None
            }
        }
        .unwrap_or_else(|| self.policy.evict_to_budget(Some(id)));
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
        self.evict_for_except(bytes, None)
    }

    /// Evict unpinned entries while protecting one additional entry for the
    /// duration of this call.
    pub(crate) fn evict_for_except(&mut self, bytes: u64, keep: Option<EntryId>) -> Evicted {
        let held: Vec<EntryId> = self
            .entries
            .iter()
            .filter(|(_, entry)| Arc::strong_count(&entry.value) > 1)
            .map(|(&id, _)| id)
            .collect();
        let evicted = {
            let mut pins = SnapshotPolicyPinGuard::new(&mut self.policy, held);
            if let Some(keep) = keep {
                pins.pin(keep);
            }
            pins.evict_for(bytes)
        };
        self.drop_entries(&evicted);
        evicted
    }

    #[cfg(test)]
    pub(crate) fn pinned_bytes(&self) -> u64 {
        self.policy.pinned_bytes()
    }

    /// Remove a specific cached value when the caller proves it is unusable.
    pub(crate) fn remove(&mut self, id: EntryId) -> Option<u64> {
        let bytes = self.policy.remove(id)?;
        self.entries.remove(&id);
        self.sync_retention();
        Some(bytes)
    }

    fn drop_entries(&mut self, evicted: &Evicted) {
        for id in &evicted.ids {
            self.entries.remove(id);
        }
        self.sync_retention();
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
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    struct PanicClock(Arc<AtomicBool>);

    impl Clock for PanicClock {
        fn now(&self) -> Duration {
            assert!(
                !self.0.load(Ordering::Relaxed),
                "injected eviction clock panic"
            );
            Duration::ZERO
        }
    }

    fn lru(max_bytes: u64) -> SnapshotCache<&'static str> {
        SnapshotCache::new(max_bytes, SnapshotPolicyConfig::LRU)
    }

    #[test]
    fn ladder_retains_only_the_recent_tip_pair_and_spaced_anchors() {
        use super::super::checkpoint_ladder::{Config, Conversation};
        let key = Conversation::testing(1);
        let mut cache = lru(1_000_000).with_checkpoint_ladder(Config::testing(500_000));
        let anchor = vec![1; 4096];
        let bytes = cache.ladder_eligibility_in(key, &(), &anchor, 1).unwrap();
        assert!(cache.insert_ladder_in(key, (), anchor.clone(), "anchor", bytes));
        for i in 2..8 {
            let mut tokens = vec![1; 5000];
            tokens[4999] = i;
            assert!(cache.insert_strict(tokens.clone(), "tip", 20_001));
            cache.record_tip(Some(key), &(), &tokens);
        }
        assert_eq!(cache.len(), 3, "one anchor plus transcript/completed pair");
        assert!(cache.entry_for(&anchor).is_some());
        let mut recent = vec![1; 5000];
        recent[4999] = 6;
        assert!(cache.entry_for(&recent).is_some());
        recent[4999] = 7;
        assert!(cache.entry_for(&recent).is_some());
        cache.record_tip(Some(key), &(), &vec![9; 5000]);
        assert_eq!(cache.len(), 3, "failed publication does not prune tips");
        let mut extended = vec![1; 9000];
        extended[8999] = 8;
        assert!(cache.insert_strict(extended.clone(), "paid tip", 36_001));
        let before = cache.indexed_bytes();
        cache.record_tip(Some(key), &(), &extended);
        assert_eq!(
            cache.ladder_status().0,
            2,
            "spaced existing tip promoted without another payload"
        );
        assert!(
            cache.indexed_bytes() <= before,
            "promotion allocates no payload"
        );
        let mut tiny = lru(1_000_000).with_checkpoint_ladder(Config::testing(1));
        for i in 2..8 {
            let tokens = vec![i; 5000];
            assert!(tiny.insert_strict(tokens.clone(), "legacy", 20_001));
            tiny.record_tip(Some(key), &(), &tokens);
        }
        assert_eq!(
            tiny.len(),
            6,
            "insufficient coverage budget preserves legacy fallback"
        );
        let anchor = vec![1; 10000];
        let bytes = cache.ladder_eligibility_in(key, &(), &anchor, 1).unwrap();
        assert!(cache.insert_ladder_in(key, (), anchor, "deep anchor", bytes));
        let mut branches = Vec::new();
        for (len, marker) in [(12000, 2), (12001, 3), (6000, 4), (6001, 5)] {
            let mut tokens = vec![1; len];
            tokens[len - 1] = marker;
            assert!(cache.insert_strict(tokens.clone(), "branch tip", (len * 4 + 1) as u64));
            cache.record_tip(Some(key), &(), &tokens);
            branches.push(tokens);
        }
        for tokens in branches {
            assert!(
                cache.entry_for(&tokens).is_some(),
                "fork keeps original deep tip and recent pair"
            );
        }
    }

    #[test]
    fn ladder_shares_budget_preserves_history_and_respects_pins_and_held_arcs() {
        use super::super::checkpoint_ladder::{Config, Conversation};
        let key = Conversation::testing(1);
        let mut cache = lru(1_000_000).with_checkpoint_ladder(Config::testing(300_000));
        for position in [4096, 8192, 12288] {
            let tokens = vec![1; position];
            let bytes = cache
                .ladder_eligibility_in(key, &(), &tokens, 20_000)
                .unwrap();
            assert!(cache.insert_ladder_in(key, (), tokens, "checkpoint", bytes));
        }
        assert_eq!(cache.ladder_status().0, 3);
        let held = cache.peek_exact_in(&(), &vec![1; 4096]).unwrap();
        assert!(cache.pin(held.id));
        assert!(
            cache
                .ladder_eligibility_in(key, &(), &vec![1; 13000], 20_000)
                .is_none(),
            "spacing"
        );
        assert!(
            cache
                .ladder_eligibility_in(key, &(), &vec![2; 20000], 300_001)
                .is_none(),
            "aux budget"
        );
        assert_eq!(cache.ladder_status().0, 3, "failed probes never evict");
        assert!(cache.insert_strict(vec![3; 50000], "tip", 700_000));
        assert!(cache.indexed_bytes() <= cache.max_bytes());
        assert!(
            cache.entry_for(&vec![1; 4096]).is_some(),
            "explicit pins survive tip insertion"
        );
        cache.unpin(held.id);
        cache.evict_for(u64::MAX);
        assert_eq!(
            cache.ladder_status().0,
            1,
            "held Arc survives pressure; metadata swept"
        );
        drop(held);
        cache.evict_for(u64::MAX);
        assert_eq!(cache.ladder_status(), (0, 0, 300_000));
    }

    #[test]
    fn ladder_prunes_only_after_success_and_uses_exact_branch_keys() {
        use super::super::checkpoint_ladder::{Config, Conversation};
        let key = Conversation::testing(1);
        let mut cache = lru(2_000_000).with_checkpoint_ladder(Config::testing(500_000));
        for position in [4096, 8192, 12288, 16384, 20480, 24576] {
            // Different branches can overlap in coordinate; restore is still
            // authorized only by the full exact token prefix.
            let tokens = vec![position as u32; position];
            let bytes = cache.ladder_eligibility_in(key, &(), &tokens, 1).unwrap();
            assert!(cache.insert_ladder_in(key, (), tokens, "state", bytes));
            assert!(cache.ladder_status().0 <= 5);
            assert!(cache.ladder_status().1 <= 500_000);
        }
        assert!(cache.best_prefix(&vec![999; 30000]).is_none());
        assert!(cache.best_prefix(&vec![4096; 5000]).is_some());
        cache.evict_for(u64::MAX);
        assert_eq!(cache.ladder_status().0, 0);
    }

    #[test]
    fn eviction_panic_releases_held_and_keep_pins() {
        let panic_clock = Arc::new(AtomicBool::new(false));
        let mut cache = SnapshotCache::with_clock(
            100,
            SnapshotPolicyConfig::default(),
            Arc::new(PanicClock(Arc::clone(&panic_clock))),
        );
        assert!(cache.insert_strict(vec![1], "held", 10));
        let held_id = cache.entry_for(&[1]).unwrap();
        let held = cache.peek_exact_in(&(), &[1]).unwrap();
        assert!(cache.insert_strict(vec![2], "keep", 10));
        let keep_id = cache.entry_for(&[2]).unwrap();
        assert!(cache.insert_strict(vec![3], "victim", 10));
        let victim_id = cache.entry_for(&[3]).unwrap();

        panic_clock.store(true, Ordering::Relaxed);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache.evict_for_except(u64::MAX, Some(keep_id));
        }));
        assert!(result.is_err());
        panic_clock.store(false, Ordering::Relaxed);

        assert_eq!(cache.pinned_bytes(), 0);
        assert!(cache.entry_for(&[1]).is_some());
        assert!(cache.entry_for(&[2]).is_some());
        assert!(cache.entry_for(&[3]).is_some());
        drop(held);
        assert_eq!(cache.entry_for(&[1]), Some(held_id));
        assert_eq!(cache.entry_for(&[3]), Some(victim_id));
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
        assert_eq!(*cache.peek_exact_in(&1, &[1, 2]).unwrap().value, "fast");
        assert!(cache.peek_exact_in(&1, &[1]).is_none());
        assert!(cache.peek_exact_in(&3, &[1, 2]).is_none());
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

    #[test]
    fn pressure_eviction_skips_held_values_and_releases_unheld_values() {
        let mut cache = lru(30);
        let held = Arc::new("held");
        assert!(cache.insert_shared_strict(vec![1], Arc::clone(&held), 10));
        assert!(cache.insert_strict(vec![2], "unheld", 10));
        let held_id = cache.entry_for(&[1]).unwrap();

        let evicted = cache.evict_for(10);
        assert_eq!(evicted.bytes, 10);
        assert_eq!(evicted.ids.len(), 1);
        assert_eq!(cache.entry_for(&[1]), Some(held_id));
        assert!(cache.entry_for(&[2]).is_none());
        assert_eq!(cache.indexed_bytes(), 10);
        assert_eq!(*held, "held");
    }

    #[test]
    fn pressure_eviction_preserves_existing_pins_while_skipping_held_values() {
        let mut cache = lru(30);
        let held = Arc::new("held and pinned");
        assert!(cache.insert_shared_strict(vec![1], Arc::clone(&held), 10));
        assert!(cache.insert_strict(vec![2], "unheld", 10));
        let held_id = cache.entry_for(&[1]).unwrap();
        assert!(cache.pin(held_id));

        assert_eq!(cache.evict_for(10).bytes, 10);
        drop(held);
        assert!(cache.evict_for(u64::MAX).is_empty());
        assert_eq!(cache.entry_for(&[1]), Some(held_id));

        cache.unpin(held_id);
        assert_eq!(cache.evict_for(10).bytes, 10);
        assert!(cache.entry_for(&[1]).is_none());
    }

    #[test]
    fn pressure_eviction_returns_empty_when_all_payloads_are_held() {
        let mut cache = lru(30);
        let first = Arc::new("first");
        let second = Arc::new("second");
        assert!(cache.insert_shared_strict(vec![1], Arc::clone(&first), 10));
        assert!(cache.insert_shared_strict(vec![2], Arc::clone(&second), 10));

        assert!(cache.evict_for(u64::MAX).is_empty());
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.indexed_bytes(), 20);
    }
}
