use crate::metal_forward::{SessionSnapshot, SnapshotIdentity};
use crate::snapshot_policy::{
    Clock, EntryId, Evicted, MonotonicClock, SnapshotPolicy, SnapshotPolicyConfig,
};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PrefixCacheKey {
    identity: SnapshotIdentity,
    prefix_len: usize,
    prefix_hash: u64,
}

#[derive(Clone, Debug)]
pub struct PrefixCacheHit {
    pub snapshot: Arc<SessionSnapshot>,
    /// Canonical token prefix matched, including an emitted pending token.
    pub matched_prefix_len: usize,
    /// Tokens already represented in restored model state. The caller must
    /// replay the request from this offset, which consumes a pending token.
    pub restored_prefix_len: usize,
    pub exact: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixCacheStats {
    pub entries: usize,
    /// Payload bytes currently reachable from the cache index. Active restore
    /// Arcs may temporarily keep replaced or evicted payloads resident.
    pub indexed_bytes: u64,
    /// Budget for indexed payload bytes, not a process-RSS ceiling.
    pub max_indexed_bytes: u64,
}

/// Session snapshots are large (KV arenas + GDN state — tens to hundreds of
/// MiB each on 27B-class models), so an unbounded cache is a reliability
/// hazard in any long-running process. `PrefixCache` therefore carries a
/// byte budget enforced on insert by a [`SnapshotPolicy`]: LRU without
/// expiry by default, frecency with expiry for services.
///
/// `DEFAULT_MAX_BYTES` is deliberately generous (16 GiB) so existing bench
/// workflows never see eviction, while still bounding a runaway session.
/// Product callers should size it explicitly via [`PrefixCache::with_max_bytes`].
pub const DEFAULT_MAX_BYTES: u64 = 16 << 30;

struct Entry {
    id: EntryId,
    snapshot: Arc<SessionSnapshot>,
    /// Promoted from a durable store: an equivalent record is already on
    /// disk, so leaving RAM must not write it again.
    durable: bool,
}

pub struct PrefixCache {
    buckets: HashMap<PrefixCacheKey, Vec<Entry>>,
    keys: HashMap<EntryId, PrefixCacheKey>,
    policy: SnapshotPolicy,
    /// When set, entries released by budget eviction or expiry (not by
    /// pressure relief, replacement, or clear) are retained in `spilled`
    /// until the owner drains them with [`Self::take_spilled`].
    spill: bool,
    spilled: Vec<Arc<SessionSnapshot>>,
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::with_max_bytes(DEFAULT_MAX_BYTES)
    }
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// An LRU cache bounded at `max_bytes`. Inserting a snapshot larger than
    /// the budget evicts everything else and stores the oversized snapshot
    /// alone (the newest entry is never rejected: the caller just produced it
    /// and a cold cache would be strictly worse).
    pub fn with_max_bytes(max_bytes: u64) -> Self {
        Self::with_policy(
            max_bytes,
            SnapshotPolicyConfig::LRU,
            Arc::new(MonotonicClock::new()),
        )
    }

    pub fn with_policy(
        max_bytes: u64,
        config: SnapshotPolicyConfig,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            buckets: HashMap::new(),
            keys: HashMap::new(),
            policy: SnapshotPolicy::with_clock(max_bytes, config, clock),
            spill: false,
            spilled: Vec::new(),
        }
    }

    /// Retain entries that leave through budget eviction or expiry for a
    /// durable tier. Disabling drops anything not yet drained.
    pub fn set_spill(&mut self, enabled: bool) {
        self.spill = enabled;
        if !enabled {
            self.spilled.clear();
        }
    }

    /// Snapshots released since the last drain, oldest release first. Their
    /// payload stays resident until the caller drops them.
    pub fn take_spilled(&mut self) -> Vec<Arc<SessionSnapshot>> {
        std::mem::take(&mut self.spilled)
    }

    /// Indexed snapshots not already backed by a durable record, most
    /// valuable first (the reverse of eviction order).
    pub fn persist_candidates(&self) -> Vec<Arc<SessionSnapshot>> {
        self.policy
            .ranked_best_first()
            .into_iter()
            .filter_map(|id| self.entry(id))
            .filter(|entry| !entry.durable)
            .map(|entry| Arc::clone(&entry.snapshot))
            .collect()
    }

    fn entry(&self, id: EntryId) -> Option<&Entry> {
        let key = self.keys.get(&id)?;
        self.buckets.get(key)?.iter().find(|entry| entry.id == id)
    }

    pub fn total_bytes(&self) -> u64 {
        self.policy.total_bytes()
    }

    pub fn max_bytes(&self) -> u64 {
        self.policy.max_bytes()
    }

    pub fn len(&self) -> usize {
        self.policy.len()
    }

    pub fn is_empty(&self) -> bool {
        self.policy.is_empty()
    }

    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            entries: self.len(),
            indexed_bytes: self.total_bytes(),
            max_indexed_bytes: self.max_bytes(),
        }
    }

    pub fn policy_config(&self) -> SnapshotPolicyConfig {
        self.policy.config()
    }

    pub fn set_policy_config(&mut self, config: SnapshotPolicyConfig) {
        self.policy.set_config(config);
    }

    pub fn set_max_bytes(&mut self, max_bytes: u64) {
        self.policy.set_max_bytes(max_bytes);
        let newest = self.policy.most_recent();
        let evicted = self.policy.evict_to_budget(newest);
        self.release(&evicted, true);
    }

    pub fn clear(&mut self) {
        self.buckets.clear();
        self.keys.clear();
        self.policy.clear();
        self.spilled.clear();
    }

    /// Bytes held by pinned entries, which neither eviction nor expiry frees.
    pub fn pinned_bytes(&self) -> u64 {
        self.policy.pinned_bytes()
    }

    /// Exclude an entry from eviction and expiry until [`Self::unpin`].
    /// Returns false when the entry is no longer indexed.
    pub fn pin(&mut self, id: EntryId) -> bool {
        self.policy.pin(id)
    }

    pub fn unpin(&mut self, id: EntryId) {
        self.policy.unpin(id);
    }

    /// Drop entries past the policy's idle TTL or maximum age.
    pub fn sweep(&mut self) -> Evicted {
        let evicted = self.policy.sweep();
        self.release(&evicted, true);
        evicted
    }

    /// Evict unpinned entries by rank until `bytes` are released. This is
    /// memory-pressure relief, so released payloads are never retained for
    /// spilling: the caller needs the bytes back now.
    pub fn evict_for(&mut self, bytes: u64) -> Evicted {
        let evicted = self.policy.evict_for(bytes);
        self.release(&evicted, false);
        evicted
    }

    /// Check whether an entry can fit by evicting only unpinned entries,
    /// without changing cache contents.
    pub(crate) fn eligible_strict(&self, bytes: u64) -> bool {
        self.policy.fits_strict(bytes)
    }

    pub fn insert(&mut self, snap: SessionSnapshot) -> EntryId {
        self.insert_shared(Arc::new(snap))
    }

    /// Index `snap`, returning the id of the entry now representing its
    /// canonical prefix (an equivalent existing winner keeps its id).
    pub(crate) fn insert_shared(&mut self, snap: Arc<SessionSnapshot>) -> EntryId {
        self.sweep();
        let key = PrefixCacheKey {
            identity: snap.identity.clone(),
            prefix_len: snap.matched_prefix_len(),
            prefix_hash: hash_snapshot_prefix(&snap),
        };
        let bucket = self.buckets.entry(key.clone()).or_default();
        let equivalent = |entry: &Entry| same_canonical_prefix(&entry.snapshot, &snap);
        let existing_complete_logits = bucket
            .iter()
            .find(|entry| {
                equivalent(entry)
                    && entry.snapshot.pending_token.is_none()
                    && entry.snapshot.final_logits.is_some()
            })
            .map(|entry| entry.id);
        let incoming_complete_logits = snap.pending_token.is_none() && snap.final_logits.is_some();
        let same_representation = bucket
            .iter()
            .find(|entry| equivalent(entry) && same_snapshot_prefix(&entry.snapshot, &snap))
            .map(|entry| entry.id);

        let (id, replaced) = match (existing_complete_logits, same_representation) {
            // An equivalent complete-logits checkpoint dominates every other
            // representation of this boundary, including the incoming one.
            (Some(winner), _) => (winner, true),
            (None, Some(existing)) if !incoming_complete_logits => (existing, false),
            _ => {
                let id = self
                    .policy
                    .insert(snap.n_bytes(), snap.matched_prefix_len() as u64);
                (id, incoming_complete_logits)
            }
        };
        if replaced {
            let mut dropped = Vec::new();
            bucket.retain(|entry| {
                let keep = entry.id == id || !equivalent(entry);
                if !keep {
                    dropped.push(entry.id);
                }
                keep
            });
            for dropped in dropped {
                self.policy.remove(dropped);
                self.keys.remove(&dropped);
            }
        }
        if bucket.iter().any(|entry| entry.id == id) {
            self.policy.touch(id);
        } else {
            bucket.push(Entry {
                id,
                snapshot: snap,
                durable: false,
            });
            self.keys.insert(id, key);
        }
        let evicted = self.policy.evict_to_budget(Some(id));
        self.release(&evicted, true);
        id
    }

    /// Strict insertion of a snapshot decoded from a durable record. The
    /// entry is marked durable so a later eviction does not rewrite it; an
    /// equivalent entry that already won keeps its own marking.
    pub(crate) fn insert_durable_strict(&mut self, snap: Arc<SessionSnapshot>) -> Option<EntryId> {
        let id = self.insert_shared_strict(Arc::clone(&snap))?;
        let key = self.keys.get(&id).cloned();
        if let Some(entry) = key
            .and_then(|key| self.buckets.get_mut(&key))
            .and_then(|bucket| bucket.iter_mut().find(|entry| entry.id == id))
            .filter(|entry| Arc::ptr_eq(&entry.snapshot, &snap))
        {
            entry.durable = true;
        }
        Some(id)
    }

    /// Strict insertion: `None` when the snapshot cannot fit without evicting
    /// pinned entries; never retains an oversized snapshot alone.
    pub(crate) fn insert_shared_strict(&mut self, snap: Arc<SessionSnapshot>) -> Option<EntryId> {
        if !self.eligible_strict(snap.n_bytes()) {
            return None;
        }
        // Deduplicate or replace the canonical boundary before enforcing the
        // budget. Reserving the full incoming size first can evict unrelated
        // entries even when an equivalent complete snapshot already wins.
        let id = self.insert_shared(snap);
        debug_assert!(self.total_bytes() <= self.max_bytes());
        Some(id)
    }

    /// Drop released entries from the index; `spillable` releases (budget
    /// eviction, expiry) are retained for the durable tier when enabled.
    fn release(&mut self, evicted: &Evicted, spillable: bool) {
        for id in &evicted.ids {
            let Some(key) = self.keys.remove(id) else {
                continue;
            };
            if let Some(bucket) = self.buckets.get_mut(&key) {
                if let Some(index) = bucket.iter().position(|entry| entry.id == *id) {
                    let entry = bucket.remove(index);
                    if spillable && self.spill && !entry.durable {
                        self.spilled.push(entry.snapshot);
                    }
                }
                if bucket.is_empty() {
                    self.buckets.remove(&key);
                }
            }
        }
    }

    pub fn lookup_longest(
        &mut self,
        identity: &SnapshotIdentity,
        request_tokens: &[i32],
    ) -> Option<PrefixCacheHit> {
        self.lookup_and_touch(identity, request_tokens, false)
    }

    /// Find the longest prefix that can produce prompt-final logits. An exact
    /// state-only checkpoint is insufficient, but an exact checkpoint with a
    /// pending token can replay that one required transition.
    pub fn lookup_longest_for_completion(
        &mut self,
        identity: &SnapshotIdentity,
        request_tokens: &[i32],
    ) -> Option<PrefixCacheHit> {
        self.lookup_and_touch(identity, request_tokens, true)
    }

    /// Lookup without recording a hit. Callers sweep first if expiry matters.
    pub(crate) fn peek_longest_for_completion(
        &self,
        identity: &SnapshotIdentity,
        request_tokens: &[i32],
    ) -> Option<PrefixCacheHit> {
        self.lookup_longest_impl(identity, request_tokens, true)
            .map(|(_, hit)| hit)
    }

    fn lookup_longest_impl(
        &self,
        identity: &SnapshotIdentity,
        request_tokens: &[i32],
        exact_requires_logits: bool,
    ) -> Option<(EntryId, PrefixCacheHit)> {
        if request_tokens.is_empty() {
            return None;
        }
        let prefix_hashes = prefix_hashes(request_tokens);
        for prefix_len in (1..=request_tokens.len()).rev() {
            let key = PrefixCacheKey {
                identity: identity.clone(),
                prefix_len,
                prefix_hash: prefix_hashes[prefix_len - 1],
            };
            let hit = self.buckets.get(&key).and_then(|bucket| {
                bucket
                    .iter()
                    .filter(|entry| {
                        let snap = &entry.snapshot;
                        snapshot_matches_request(snap, request_tokens, prefix_len)
                            && (!exact_requires_logits
                                || prefix_len != request_tokens.len()
                                || snap.pending_token.is_some()
                                || snap.final_logits.is_some())
                    })
                    .max_by_key(|entry| entry.snapshot.prefix_len())
            });
            if let Some(entry) = hit {
                let snap = Arc::clone(&entry.snapshot);
                return Some((
                    entry.id,
                    PrefixCacheHit {
                        restored_prefix_len: snap.prefix_len(),
                        snapshot: snap,
                        matched_prefix_len: prefix_len,
                        exact: prefix_len == request_tokens.len(),
                    },
                ));
            }
        }
        None
    }

    fn lookup_and_touch(
        &mut self,
        identity: &SnapshotIdentity,
        request_tokens: &[i32],
        exact_requires_logits: bool,
    ) -> Option<PrefixCacheHit> {
        self.sweep();
        let (id, hit) =
            self.lookup_longest_impl(identity, request_tokens, exact_requires_logits)?;
        self.policy.touch(id);
        Some(hit)
    }

    /// Record a use of the indexed entry holding `snapshot`, returning its id
    /// (none once the entry has left the index).
    pub(crate) fn touch_shared(&mut self, snapshot: &Arc<SessionSnapshot>) -> Option<EntryId> {
        let id = self.entry_id_of(snapshot);
        if let Some(id) = id {
            self.policy.touch(id);
        }
        id
    }

    /// Id of the indexed entry holding exactly this `snapshot` allocation.
    pub(crate) fn entry_id_of(&self, snapshot: &Arc<SessionSnapshot>) -> Option<EntryId> {
        let key = PrefixCacheKey {
            identity: snapshot.identity.clone(),
            prefix_len: snapshot.matched_prefix_len(),
            prefix_hash: hash_snapshot_prefix(snapshot),
        };
        self.buckets.get(&key).and_then(|bucket| {
            bucket
                .iter()
                .find(|entry| Arc::ptr_eq(&entry.snapshot, snapshot))
                .map(|entry| entry.id)
        })
    }
}

const HASH_SEED: u64 = 0xcbf29ce484222325;
const HASH_PRIME: u64 = 0x100000001b3;

fn hash_tokens(tokens: &[i32]) -> u64 {
    let mut h = HASH_SEED;
    for &tok in tokens {
        h = hash_token(h, tok);
    }
    h
}

fn hash_snapshot_prefix(snapshot: &SessionSnapshot) -> u64 {
    let mut h = hash_tokens(&snapshot.prefix_tokens);
    if let Some(token) = snapshot.pending_token {
        h = hash_token(h, token);
    }
    h
}

fn hash_token(mut hash: u64, token: i32) -> u64 {
    hash ^= (token as u32 as u64).wrapping_add(0x9e3779b97f4a7c15);
    hash.wrapping_mul(HASH_PRIME)
}

fn same_snapshot_prefix(a: &SessionSnapshot, b: &SessionSnapshot) -> bool {
    a.prefix_tokens == b.prefix_tokens && a.pending_token == b.pending_token
}

fn same_canonical_prefix(a: &SessionSnapshot, b: &SessionSnapshot) -> bool {
    a.matched_prefix_len() == b.matched_prefix_len()
        && a.prefix_tokens.iter().copied().chain(a.pending_token).eq(b
            .prefix_tokens
            .iter()
            .copied()
            .chain(b.pending_token))
}

fn snapshot_matches_request(
    snapshot: &SessionSnapshot,
    request_tokens: &[i32],
    matched_prefix_len: usize,
) -> bool {
    let consumed = snapshot.prefix_len();
    matched_prefix_len == snapshot.matched_prefix_len()
        && snapshot.prefix_tokens == request_tokens[..consumed]
        && snapshot
            .pending_token
            .is_none_or(|token| request_tokens.get(consumed) == Some(&token))
}

fn prefix_hashes(tokens: &[i32]) -> Vec<u64> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut h = HASH_SEED;
    for &tok in tokens {
        h = hash_token(h, tok);
        out.push(h);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{PrefixCache, hash_tokens};
    use crate::metal_forward::{
        SNAPSHOT_LAYOUT_VERSION, SessionSnapshot, SnapshotIdentity, SnapshotKvStorageKind,
    };
    use crate::snapshot_policy::{FakeClock, SnapshotPolicyConfig};
    use std::sync::Arc;
    use std::time::Duration;

    fn ident(model_id: u64) -> SnapshotIdentity {
        SnapshotIdentity {
            model_id,
            tokenizer_id: 7,
            layout_version: SNAPSHOT_LAYOUT_VERSION,
            n_attn_layers: 2,
            n_gdn_layers: 3,
            kv_dim_elements: 4,
            kv_bytes_per_token: 8,
            kv_storage_kind: SnapshotKvStorageKind::F16,
            gdn_state_elements_per_layer: 5,
            gdn_conv_elements_per_layer: 6,
        }
    }

    fn snap(identity: SnapshotIdentity, prefix: &[i32], bytes: usize) -> SessionSnapshot {
        let half = bytes / 2;
        SessionSnapshot {
            identity,
            prefix_tokens: prefix.to_vec(),
            pending_token: None,
            kv_n_pos: vec![prefix.len(), prefix.len()],
            kv_k_arena: vec![1; half],
            kv_v_arena: vec![2; bytes - half],
            gdn_conv_arena: vec![3; 8],
            gdn_state_arena: vec![4; 12],
            final_logits: Some(vec![0.1, 0.2, 0.3]),
            capture_tail: None,
        }
    }

    #[test]
    fn hash_changes_with_tokens() {
        assert_ne!(hash_tokens(&[1, 2, 3]), hash_tokens(&[1, 2, 4]));
    }

    #[test]
    fn transient_prompt_capture_has_a_budget_and_failure_boundary() {
        let id = ident(1);
        let prior = Arc::new(snap(id.clone(), &[10], 96));
        let prompt = Arc::new(snap(id.clone(), &[10, 11], 192));
        let mut completed = snap(id.clone(), &[10, 11], 192);
        completed.pending_token = Some(12);
        completed.final_logits = None;
        let completed = Arc::new(completed);

        for both_fit in [false, true] {
            let budget = if both_fit {
                prompt.n_bytes() + completed.n_bytes()
            } else {
                prompt.n_bytes().max(completed.n_bytes())
            };
            let mut captured = PrefixCache::with_max_bytes(budget);
            let mut elided = PrefixCache::with_max_bytes(budget);
            for cache in [&mut captured, &mut elided] {
                assert!(cache.insert_shared_strict(Arc::clone(&prior)).is_some());
                assert!(cache.eligible_strict(prompt.n_bytes()));
                assert!(cache.eligible_strict(completed.n_bytes()));
            }
            assert!(captured.insert_shared_strict(Arc::clone(&prompt)).is_some());

            // If generation aborts before publishing completion, a retry differs.
            assert_eq!(
                captured
                    .peek_longest_for_completion(&id, &[10, 11])
                    .unwrap()
                    .restored_prefix_len,
                2
            );
            assert_eq!(
                elided
                    .peek_longest_for_completion(&id, &[10, 11])
                    .unwrap()
                    .restored_prefix_len,
                1
            );

            for cache in [&mut captured, &mut elided] {
                assert!(cache.insert_shared_strict(Arc::clone(&completed)).is_some());
                let hit = cache
                    .peek_longest_for_completion(&id, &[10, 11, 12, 13])
                    .unwrap();
                assert!(Arc::ptr_eq(&hit.snapshot, &completed));
            }
            if both_fit {
                assert_eq!(
                    captured
                        .peek_longest_for_completion(&id, &[10, 11])
                        .unwrap()
                        .restored_prefix_len,
                    2
                );
                assert_eq!(
                    elided
                        .peek_longest_for_completion(&id, &[10, 11])
                        .unwrap()
                        .restored_prefix_len,
                    1
                );
            } else {
                assert_eq!(captured.len(), 1);
                assert_eq!(elided.len(), 1);
                assert!(
                    captured
                        .peek_longest_for_completion(&id, &[10, 11])
                        .is_none()
                );
                assert!(elided.peek_longest_for_completion(&id, &[10, 11]).is_none());
            }
        }
    }

    #[test]
    fn lookup_prefers_longest_prefix() {
        let id = ident(1);
        let mut cache = PrefixCache::new();
        cache.insert(snap(id.clone(), &[10, 11], 32));
        cache.insert(snap(id.clone(), &[10, 11, 12], 48));
        let hit = cache.lookup_longest(&id, &[10, 11, 12, 13]).expect("hit");
        assert_eq!(hit.matched_prefix_len, 3);
        assert!(!hit.exact);
        assert_eq!(hit.snapshot.prefix_tokens, vec![10, 11, 12]);
    }

    #[test]
    fn shared_insert_indexes_without_cloning_snapshot_arenas() {
        let id = ident(1);
        let snapshot = Arc::new(snap(id.clone(), &[10, 11], 32));
        let mut cache = PrefixCache::new();
        cache.insert_shared(Arc::clone(&snapshot));

        let hit = cache.lookup_longest(&id, &[10, 11, 12]).expect("hit");
        assert!(Arc::ptr_eq(&snapshot, &hit.snapshot));
    }

    #[test]
    fn completion_lookup_falls_back_from_exact_state_without_logits() {
        let id = ident(1);
        let mut cache = PrefixCache::new();
        cache.insert(snap(id.clone(), &[10, 11], 32));
        let mut exact_without_logits = snap(id.clone(), &[10, 11, 12], 48);
        exact_without_logits.final_logits = None;
        cache.insert(exact_without_logits);

        let state_hit = cache
            .lookup_longest(&id, &[10, 11, 12])
            .expect("state-only exact hit");
        assert!(state_hit.exact);
        assert!(state_hit.snapshot.final_logits.is_none());

        let completion_hit = cache
            .lookup_longest_for_completion(&id, &[10, 11, 12])
            .expect("shorter completion-capable hit");
        assert_eq!(completion_hit.matched_prefix_len, 2);
        assert!(!completion_hit.exact);
        assert!(completion_hit.snapshot.final_logits.is_some());

        assert!(cache.lookup_longest_for_completion(&id, &[99]).is_none());
    }

    #[test]
    fn pending_terminal_token_matches_logically_and_replays_physically() {
        let id = ident(1);
        let mut cache = PrefixCache::new();
        let mut checkpoint = snap(id.clone(), &[10, 11], 32);
        checkpoint.pending_token = Some(12);
        checkpoint.final_logits = None;
        cache.insert(checkpoint);

        let exact = cache
            .lookup_longest_for_completion(&id, &[10, 11, 12])
            .expect("pending exact hit");
        assert!(exact.exact);
        assert_eq!(exact.matched_prefix_len, 3);
        assert_eq!(exact.restored_prefix_len, 2);

        let extension = cache
            .lookup_longest_for_completion(&id, &[10, 11, 12, 13])
            .expect("pending extension hit");
        assert!(!extension.exact);
        assert_eq!(extension.matched_prefix_len, 3);
        assert_eq!(extension.restored_prefix_len, 2);
        assert!(
            cache
                .lookup_longest_for_completion(&id, &[10, 11, 99])
                .is_none()
        );

        cache.insert(snap(id.clone(), &[10, 11, 12], 48));
        let fully_consumed = cache
            .lookup_longest_for_completion(&id, &[10, 11, 12, 13])
            .expect("fully consumed state");
        assert_eq!(fully_consumed.matched_prefix_len, 3);
        assert_eq!(fully_consumed.restored_prefix_len, 3);
    }

    #[test]
    fn pending_and_complete_without_logits_remain_complementary() {
        let id = ident(1);
        let mut cache = PrefixCache::new();
        let mut pending = snap(id.clone(), &[10, 11], 32);
        pending.pending_token = Some(12);
        pending.final_logits = None;
        let mut complete = snap(id.clone(), &[10, 11, 12], 64);
        complete.final_logits = None;
        cache.insert(pending);
        cache.insert(complete);
        assert_eq!(cache.len(), 2);

        let exact = cache
            .lookup_longest_for_completion(&id, &[10, 11, 12])
            .expect("pending reconstructs exact logits");
        assert_eq!(exact.restored_prefix_len, 2);

        let extension = cache
            .lookup_longest_for_completion(&id, &[10, 11, 12, 13])
            .expect("complete state accelerates extension");
        assert_eq!(extension.restored_prefix_len, 3);
    }

    #[test]
    fn retained_hit_survives_replacement_and_clear_outside_index_accounting() {
        let id = ident(1);
        let mut cache = PrefixCache::new();
        let mut old = snap(id.clone(), &[10, 11], 32);
        old.final_logits = None;
        cache.insert(old);

        let hit = cache.lookup_longest(&id, &[10, 11]).expect("old hit");
        let old_bytes = hit.snapshot.n_bytes();
        assert!(hit.snapshot.final_logits.is_none());

        let mut replacement = snap(id, &[10, 11], 64);
        replacement.final_logits = Some(vec![4.0, 5.0, 6.0]);
        cache.insert(replacement);
        assert_ne!(cache.stats().indexed_bytes, old_bytes);
        cache.clear();

        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().indexed_bytes, 0);
        assert_eq!(hit.snapshot.prefix_tokens, [10, 11]);
        assert!(hit.snapshot.final_logits.is_none());
    }

    #[test]
    fn lookup_filters_by_identity() {
        let id_a = ident(1);
        let id_b = ident(2);
        let mut cache = PrefixCache::new();
        cache.insert(snap(id_a.clone(), &[1, 2, 3], 40));
        assert!(cache.lookup_longest(&id_b, &[1, 2, 3, 4]).is_none());
        assert!(cache.lookup_longest(&id_a, &[1, 2, 3, 4]).is_some());
    }

    #[test]
    fn insert_keeps_first_equal_capability_snapshot() {
        let id = ident(1);
        let mut cache = PrefixCache::new();
        cache.insert(snap(id.clone(), &[1, 2, 3], 32));
        let before = cache.total_bytes();
        cache.insert(snap(id.clone(), &[1, 2, 3], 96));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_bytes(), before);
        let hit = cache.lookup_longest(&id, &[1, 2, 3]).expect("exact hit");
        assert!(hit.exact);
    }

    #[test]
    fn complete_logits_checkpoint_dominates_pending_representation() {
        let id = ident(1);
        for complete_first in [false, true] {
            let complete = snap(id.clone(), &[10, 11, 12], 64);
            let mut pending = snap(id.clone(), &[10, 11], 32);
            pending.pending_token = Some(12);
            pending.final_logits = None;
            let max_bytes = complete.n_bytes().max(pending.n_bytes());
            let mut cache = PrefixCache::with_max_bytes(max_bytes);
            if complete_first {
                cache.insert(complete);
                cache.insert(pending);
            } else {
                cache.insert(pending);
                cache.insert(complete);
            }

            assert_eq!(cache.len(), 1);
            let hit = cache
                .lookup_longest_for_completion(&id, &[10, 11, 12, 13])
                .expect("dominant complete checkpoint");
            assert_eq!(hit.matched_prefix_len, 3);
            assert_eq!(hit.restored_prefix_len, 3);
            assert!(hit.snapshot.final_logits.is_some());
        }
    }

    fn snap_bytes(prefix: &[i32], bytes: usize) -> u64 {
        // Mirror of SessionSnapshot::n_bytes accounting for our fixture:
        // arenas (kv_k + kv_v = `bytes`, conv 8, state 12) + logits.
        let s = snap(ident(1), prefix, bytes);
        s.n_bytes()
    }

    #[test]
    fn eviction_respects_byte_budget_and_lru_order() {
        let id = ident(1);
        let one = snap_bytes(&[1], 64);
        // Budget fits exactly two entries.
        let mut cache = PrefixCache::with_max_bytes(2 * one);
        cache.insert(snap(id.clone(), &[1], 64));
        cache.insert(snap(id.clone(), &[2], 64));
        assert_eq!(cache.len(), 2);
        // Touch [1] so [2] becomes LRU.
        assert!(cache.lookup_longest(&id, &[1]).is_some());
        cache.insert(snap(id.clone(), &[3], 64));
        assert_eq!(cache.len(), 2);
        assert!(cache.total_bytes() <= cache.max_bytes());
        // [2] was least-recently used and must be gone; [1] and [3] survive.
        assert!(cache.lookup_longest(&id, &[2]).is_none());
        assert!(cache.lookup_longest(&id, &[1]).is_some());
        assert!(cache.lookup_longest(&id, &[3]).is_some());
    }

    #[test]
    fn pre_admission_peek_does_not_promote_lru() {
        let id = ident(1);
        let one = snap_bytes(&[1], 64);
        let mut cache = PrefixCache::with_max_bytes(2 * one);
        cache.insert(snap(id.clone(), &[1], 64));
        cache.insert(snap(id.clone(), &[2], 64));
        assert!(cache.peek_longest_for_completion(&id, &[1]).is_some());
        cache.insert(snap(id.clone(), &[3], 64));
        assert!(cache.lookup_longest(&id, &[1]).is_none());
        assert!(cache.lookup_longest(&id, &[2]).is_some());
        assert!(cache.lookup_longest(&id, &[3]).is_some());
    }

    #[test]
    fn oversized_snapshot_is_kept_alone() {
        let id = ident(1);
        let mut cache = PrefixCache::with_max_bytes(1); // absurdly small
        cache.insert(snap(id.clone(), &[1, 2], 64));
        // The newest entry is never rejected, even over budget.
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup_longest(&id, &[1, 2]).is_some());
        // A second insert evicts the first (still keeps the newest).
        cache.insert(snap(id.clone(), &[3, 4], 64));
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup_longest(&id, &[1, 2]).is_none());
        assert!(cache.lookup_longest(&id, &[3, 4]).is_some());
    }

    #[test]
    fn strict_insertion_rejects_oversized_and_evicts_lru() {
        let id = ident(1);
        let one = snap_bytes(&[1], 64);
        let mut cache = PrefixCache::with_max_bytes(2 * one);
        cache.insert(snap(id.clone(), &[1], 64));
        cache.insert(snap(id.clone(), &[2], 64));
        assert!(cache.lookup_longest(&id, &[1]).is_some());
        assert!(
            cache
                .insert_shared_strict(Arc::new(snap(id.clone(), &[3], 64)))
                .is_some()
        );
        assert!(cache.lookup_longest(&id, &[2]).is_none());
        assert!(
            cache
                .insert_shared_strict(Arc::new(snap(id.clone(), &[4], (2 * one + 1) as usize)))
                .is_none()
        );
        assert!(cache.total_bytes() <= cache.max_bytes());
    }

    #[test]
    fn strict_equivalent_winner_does_not_evict_unrelated_lru() {
        let id = ident(1);
        let complete = snap(id.clone(), &[1], 64);
        let unrelated = snap(id.clone(), &[2], 64);
        let budget = complete.n_bytes() + unrelated.n_bytes();
        let mut cache = PrefixCache::with_max_bytes(budget);
        cache.insert(complete.clone());
        cache.insert(unrelated);

        assert!(cache.insert_shared_strict(Arc::new(complete)).is_some());
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup_longest(&id, &[1]).is_some());
        assert!(cache.lookup_longest(&id, &[2]).is_some());
    }

    #[test]
    fn strict_eligibility_never_evicts() {
        let id = ident(1);
        let one = snap_bytes(&[1], 64);
        let mut cache = PrefixCache::with_max_bytes(one);
        cache.insert(snap(id.clone(), &[1], 64));
        assert!(cache.eligible_strict(one));
        assert!(!cache.eligible_strict(one + 1));
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup_longest(&id, &[1]).is_some());
    }

    #[test]
    fn default_budget_is_generous() {
        // Guard against accidentally shipping a tiny default that would
        // silently change bench behavior.
        const { assert!(super::DEFAULT_MAX_BYTES >= (8u64 << 30)) };
    }

    #[test]
    fn stats_resize_and_clear_reflect_memory_policy() {
        let id = ident(1);
        let mut cache = PrefixCache::with_max_bytes(1 << 20);
        cache.insert(snap(id.clone(), &[1], 64));
        cache.insert(snap(id.clone(), &[2], 64));
        let before = cache.stats();
        assert_eq!(before.entries, 2);
        assert!(before.indexed_bytes > 0);

        cache.set_max_bytes(1);
        let after = cache.stats();
        assert_eq!(after.entries, 1);
        assert_eq!(after.max_indexed_bytes, 1);

        cache.clear();
        let cleared = cache.stats();
        assert_eq!(cleared.entries, 0);
        assert_eq!(cleared.indexed_bytes, 0);
        assert_eq!(cleared.max_indexed_bytes, 1);
    }

    fn frecency_cache(max_bytes: u64) -> (PrefixCache, FakeClock) {
        let clock = FakeClock::default();
        let config = SnapshotPolicyConfig {
            idle_ttl: Duration::from_secs(3600),
            ..SnapshotPolicyConfig::default()
        };
        (
            PrefixCache::with_policy(max_bytes, config, Arc::new(clock.clone())),
            clock,
        )
    }

    #[test]
    fn expiry_drops_the_index_arc() {
        let id = ident(1);
        let (mut cache, clock) = frecency_cache(1 << 20);
        let snapshot = Arc::new(snap(id.clone(), &[1, 2], 64));
        cache.insert_shared(Arc::clone(&snapshot));
        assert_eq!(Arc::strong_count(&snapshot), 2);
        clock.advance(Duration::from_secs(3601));
        assert_eq!(cache.sweep().ids.len(), 1);
        assert_eq!(Arc::strong_count(&snapshot), 1);
        assert_eq!(cache.stats().indexed_bytes, 0);
        // Lookups sweep lazily too.
        cache.insert(snap(id.clone(), &[3], 64));
        clock.advance(Duration::from_secs(3601));
        assert!(cache.lookup_longest(&id, &[3]).is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn pinned_entry_is_retained_by_strict_insert_and_counted_in_eligibility() {
        let id = ident(1);
        let one = snap_bytes(&[1], 64);
        let (mut cache, _) = frecency_cache(2 * one);
        let transcript = cache.insert(snap(id.clone(), &[1], 64));
        cache.insert(snap(id.clone(), &[2], 64));
        assert!(cache.pin(transcript));
        assert_eq!(cache.pinned_bytes(), one);
        assert!(!cache.eligible_strict(one + 1));
        let completed = cache
            // One more prefix token (4 bytes) with a 4-byte-smaller arena.
            .insert_shared_strict(Arc::new(snap(id.clone(), &[1, 5], 60)))
            .expect("fits by evicting the unpinned entry");
        cache.unpin(transcript);
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup_longest(&id, &[2]).is_none());
        assert!(cache.lookup_longest(&id, &[1]).is_some());
        assert_ne!(completed, transcript);
    }

    #[test]
    fn equivalent_insert_returns_the_winning_entry_id() {
        let id = ident(1);
        let mut cache = PrefixCache::new();
        let first = cache.insert(snap(id.clone(), &[1, 2], 64));
        assert_eq!(cache.insert(snap(id.clone(), &[1, 2], 96)), first);
    }

    #[test]
    fn spill_retains_budget_and_expiry_releases_only() {
        let id = ident(1);
        let one = snap_bytes(&[1], 64);
        let (mut cache, clock) = frecency_cache(2 * one);
        // Disabled by default: nothing is retained.
        cache.insert(snap(id.clone(), &[1], 64));
        cache.insert(snap(id.clone(), &[2], 64));
        cache.insert(snap(id.clone(), &[3], 64));
        assert!(cache.take_spilled().is_empty());

        cache.set_spill(true);
        cache.insert(snap(id.clone(), &[4], 64)); // budget eviction
        let spilled = cache.take_spilled();
        assert_eq!(spilled.len(), 1);
        assert!(cache.take_spilled().is_empty(), "drain empties the buffer");

        clock.advance(Duration::from_secs(3601)); // idle expiry
        assert_eq!(cache.sweep().ids.len(), 2);
        assert_eq!(cache.take_spilled().len(), 2);

        // Pressure relief frees memory now; it never retains payloads.
        cache.insert(snap(id.clone(), &[5], 64));
        assert_eq!(cache.evict_for(1).ids.len(), 1);
        assert!(cache.take_spilled().is_empty());

        // Replacement by a dominating representation is not a release.
        let mut pending = snap(id.clone(), &[6], 32);
        pending.pending_token = Some(7);
        pending.final_logits = None;
        cache.insert(pending);
        cache.insert(snap(id.clone(), &[6, 7], 32));
        assert!(cache.take_spilled().is_empty());

        cache.set_spill(false);
        cache.insert(snap(id.clone(), &[8], 64));
        cache.insert(snap(id.clone(), &[9], 64));
        assert!(cache.take_spilled().is_empty());
    }

    #[test]
    fn durable_promotions_are_not_spilled_or_persist_candidates() {
        let id = ident(1);
        let one = snap_bytes(&[1], 64);
        let (mut cache, _) = frecency_cache(2 * one);
        cache.set_spill(true);
        let promoted = Arc::new(snap(id.clone(), &[1], 64));
        cache
            .insert_durable_strict(Arc::clone(&promoted))
            .expect("fits");
        cache.insert(snap(id.clone(), &[2], 64));
        let candidates = cache.persist_candidates();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].prefix_tokens, [2]);
        // Evict both by budget: only the captured one is spilled.
        cache.set_max_bytes(0);
        cache.insert(snap(id.clone(), &[3], 64));
        let spilled: Vec<_> = cache
            .take_spilled()
            .iter()
            .map(|snapshot| snapshot.prefix_tokens.clone())
            .collect();
        assert_eq!(spilled, [vec![2]]);
        assert!(
            cache
                .insert_durable_strict(Arc::new(snap(id, &[4], 64)))
                .is_none()
        );
    }

    #[test]
    fn persist_candidates_follow_rank() {
        let id = ident(1);
        let (mut cache, _) = frecency_cache(1 << 20);
        let cold = cache.insert(snap(id.clone(), &[1], 64));
        cache.insert(snap(id.clone(), &[2], 64));
        for _ in 0..3 {
            assert!(cache.lookup_longest(&id, &[2]).is_some());
        }
        let order: Vec<_> = cache
            .persist_candidates()
            .iter()
            .map(|snapshot| snapshot.prefix_tokens.clone())
            .collect();
        assert_eq!(order, [vec![2], vec![1]]);
        assert!(cache.pin(cold));
        assert_eq!(
            cache.persist_candidates().len(),
            2,
            "pins do not hide entries"
        );
    }

    #[test]
    fn evict_for_releases_unpinned_bytes() {
        let id = ident(1);
        let one = snap_bytes(&[1], 64);
        let (mut cache, _) = frecency_cache(10 * one);
        let pinned = cache.insert(snap(id.clone(), &[1], 64));
        cache.pin(pinned);
        cache.insert(snap(id.clone(), &[2], 64));
        cache.insert(snap(id.clone(), &[3], 64));
        let evicted = cache.evict_for(one + 1);
        assert_eq!(evicted.bytes, 2 * one);
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup_longest(&id, &[1]).is_some());
    }
}
