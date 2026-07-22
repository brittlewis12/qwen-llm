use crate::metal_forward::{SessionSnapshot, SnapshotIdentity};
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
/// byte budget and evicts least-recently-used entries on insert.
///
/// `DEFAULT_MAX_BYTES` is deliberately generous (16 GiB) so existing bench
/// workflows never see eviction, while still bounding a runaway session.
/// Product callers should size it explicitly via [`PrefixCache::with_max_bytes`].
pub const DEFAULT_MAX_BYTES: u64 = 16 << 30;

pub struct PrefixCache {
    buckets: HashMap<PrefixCacheKey, Vec<Arc<SessionSnapshot>>>,
    total_bytes: u64,
    max_bytes: u64,
    /// Monotonic logical clock for LRU accounting. Bumped on insert and on
    /// lookup hit; per-entry stamps live in `last_used`.
    clock: u64,
    /// Last-used stamp per (key, prefix_tokens) entry, keyed by the same
    /// bucket key plus the index within the bucket's Vec. Rebuilt lazily on
    /// eviction; kept as a parallel map to avoid widening `SessionSnapshot`.
    last_used: HashMap<(PrefixCacheKey, usize), u64>,
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

    /// A cache bounded at `max_bytes`. Inserting a snapshot larger than the
    /// budget evicts everything else and stores the oversized snapshot alone
    /// (the newest entry is never rejected: the caller just produced it and
    /// a cold cache would be strictly worse).
    pub fn with_max_bytes(max_bytes: u64) -> Self {
        Self {
            buckets: HashMap::new(),
            total_bytes: 0,
            max_bytes,
            clock: 0,
            last_used: HashMap::new(),
        }
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    pub fn len(&self) -> usize {
        self.buckets.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            entries: self.len(),
            indexed_bytes: self.total_bytes,
            max_indexed_bytes: self.max_bytes,
        }
    }

    pub fn set_max_bytes(&mut self, max_bytes: u64) {
        self.max_bytes = max_bytes;
        self.evict_to_budget();
    }

    pub fn clear(&mut self) {
        self.buckets.clear();
        self.last_used.clear();
        self.total_bytes = 0;
    }

    pub fn insert(&mut self, snap: SessionSnapshot) {
        let key = PrefixCacheKey {
            identity: snap.identity.clone(),
            prefix_len: snap.matched_prefix_len(),
            prefix_hash: hash_snapshot_prefix(&snap),
        };
        self.clock += 1;
        let stamp = self.clock;
        let snap = Arc::new(snap);
        let bucket = self.buckets.remove(&key).unwrap_or_default();
        let old_bucket_bytes: u64 = bucket.iter().map(|entry| entry.n_bytes()).sum();
        self.total_bytes -= old_bucket_bytes;
        let mut entries: Vec<(Arc<SessionSnapshot>, u64)> = bucket
            .into_iter()
            .enumerate()
            .map(|(idx, entry)| {
                let old_stamp = self.last_used.remove(&(key.clone(), idx)).unwrap_or(0);
                (entry, old_stamp)
            })
            .collect();

        let equivalent: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter(|(_, (entry, _))| same_canonical_prefix(entry, &snap))
            .map(|(idx, _)| idx)
            .collect();
        let existing_complete_logits = equivalent.iter().copied().find(|&idx| {
            let entry = &entries[idx].0;
            entry.pending_token.is_none() && entry.final_logits.is_some()
        });
        let incoming_complete_logits = snap.pending_token.is_none() && snap.final_logits.is_some();

        if let Some(winner) = existing_complete_logits {
            entries[winner].1 = stamp;
            entries = entries
                .into_iter()
                .enumerate()
                .filter(|(idx, (entry, _))| *idx == winner || !same_canonical_prefix(entry, &snap))
                .map(|(_, entry)| entry)
                .collect();
        } else if incoming_complete_logits {
            entries.retain(|(entry, _)| !same_canonical_prefix(entry, &snap));
            entries.push((snap, stamp));
        } else if let Some(existing) = equivalent
            .iter()
            .copied()
            .find(|&idx| same_snapshot_prefix(&entries[idx].0, &snap))
        {
            entries[existing].1 = stamp;
        } else {
            entries.push((snap, stamp));
        }

        if !entries.is_empty() {
            let mut bucket = Vec::with_capacity(entries.len());
            for (idx, (entry, entry_stamp)) in entries.into_iter().enumerate() {
                self.total_bytes += entry.n_bytes();
                self.last_used.insert((key.clone(), idx), entry_stamp);
                bucket.push(entry);
            }
            self.buckets.insert(key, bucket);
        }
        self.evict_to_budget();
    }

    /// Evict least-recently-used entries until `total_bytes <= max_bytes`,
    /// never evicting the most-recently-stamped entry (the one just
    /// inserted/updated).
    fn evict_to_budget(&mut self) {
        while self.total_bytes > self.max_bytes && self.len() > 1 {
            // Find the (key, idx) with the smallest stamp, excluding the max.
            let newest = self.last_used.values().copied().max().unwrap_or(0);
            let victim = self
                .last_used
                .iter()
                .filter(|&(_, &stamp)| stamp != newest)
                .min_by_key(|&(_, &stamp)| stamp)
                .map(|(k, _)| k.clone());
            let Some((vkey, vidx)) = victim else { break };
            let Some(bucket) = self.buckets.get_mut(&vkey) else {
                self.last_used.remove(&(vkey, vidx));
                continue;
            };
            if vidx >= bucket.len() {
                self.last_used.remove(&(vkey, vidx));
                continue;
            }
            let removed = bucket.swap_remove(vidx);
            self.total_bytes -= removed.n_bytes();
            self.last_used.remove(&(vkey.clone(), vidx));
            // swap_remove moved the former last element into vidx; fix its stamp key.
            let moved_from = bucket.len();
            if let Some(stamp) = self.last_used.remove(&(vkey.clone(), moved_from)) {
                self.last_used.insert((vkey.clone(), vidx), stamp);
            }
            if bucket.is_empty() {
                self.buckets.remove(&vkey);
            }
        }
    }

    pub fn lookup_longest(
        &mut self,
        identity: &SnapshotIdentity,
        request_tokens: &[i32],
    ) -> Option<PrefixCacheHit> {
        self.lookup_longest_impl(identity, request_tokens, false)
    }

    /// Find the longest prefix that can produce prompt-final logits. An exact
    /// state-only checkpoint is insufficient, but an exact checkpoint with a
    /// pending token can replay that one required transition.
    pub fn lookup_longest_for_completion(
        &mut self,
        identity: &SnapshotIdentity,
        request_tokens: &[i32],
    ) -> Option<PrefixCacheHit> {
        self.lookup_longest_impl(identity, request_tokens, true)
    }

    fn lookup_longest_impl(
        &mut self,
        identity: &SnapshotIdentity,
        request_tokens: &[i32],
        exact_requires_logits: bool,
    ) -> Option<PrefixCacheHit> {
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
            let hit_idx = self.buckets.get(&key).and_then(|bucket| {
                bucket
                    .iter()
                    .enumerate()
                    .filter(|(_, snap)| {
                        snapshot_matches_request(snap, request_tokens, prefix_len)
                            && (!exact_requires_logits
                                || prefix_len != request_tokens.len()
                                || snap.pending_token.is_some()
                                || snap.final_logits.is_some())
                    })
                    .max_by_key(|(_, snap)| snap.prefix_len())
                    .map(|(idx, _)| idx)
            });
            if let Some(idx) = hit_idx {
                // Bump LRU stamp so hot prefixes survive eviction pressure.
                self.clock += 1;
                let stamp = self.clock;
                self.last_used.insert((key.clone(), idx), stamp);
                let snap = Arc::clone(&self.buckets[&key][idx]);
                return Some(PrefixCacheHit {
                    restored_prefix_len: snap.prefix_len(),
                    snapshot: snap,
                    matched_prefix_len: prefix_len,
                    exact: prefix_len == request_tokens.len(),
                });
            }
        }
        None
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
    use crate::metal_forward::{SNAPSHOT_LAYOUT_VERSION, SessionSnapshot, SnapshotIdentity};

    fn ident(model_id: u64) -> SnapshotIdentity {
        SnapshotIdentity {
            model_id,
            tokenizer_id: 7,
            layout_version: SNAPSHOT_LAYOUT_VERSION,
            n_attn_layers: 2,
            n_gdn_layers: 3,
            kv_dim_elements: 4,
            kv_bytes_per_token: 8,
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
        }
    }

    #[test]
    fn hash_changes_with_tokens() {
        assert_ne!(hash_tokens(&[1, 2, 3]), hash_tokens(&[1, 2, 4]));
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
    fn default_budget_is_generous() {
        // Guard against accidentally shipping a tiny default that would
        // silently change bench behavior.
        assert!(super::DEFAULT_MAX_BYTES >= (8u64 << 30));
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
}
