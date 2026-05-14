use crate::metal_forward::{SessionSnapshot, SnapshotIdentity};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PrefixCacheKey {
    identity: SnapshotIdentity,
    prefix_len: usize,
    prefix_hash: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct PrefixCacheHit<'a> {
    pub snapshot: &'a SessionSnapshot,
    pub matched_prefix_len: usize,
    pub exact: bool,
}

#[derive(Default)]
pub struct PrefixCache {
    buckets: HashMap<PrefixCacheKey, Vec<SessionSnapshot>>,
    total_bytes: u64,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    pub fn len(&self) -> usize {
        self.buckets.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    pub fn insert(&mut self, snap: SessionSnapshot) {
        let key = PrefixCacheKey {
            identity: snap.identity.clone(),
            prefix_len: snap.prefix_len(),
            prefix_hash: hash_tokens(&snap.prefix_tokens),
        };
        let bucket = self.buckets.entry(key).or_default();
        if let Some(existing) = bucket
            .iter_mut()
            .find(|s| s.prefix_tokens == snap.prefix_tokens)
        {
            self.total_bytes = self.total_bytes + snap.n_bytes() - existing.n_bytes();
            *existing = snap;
        } else {
            self.total_bytes += snap.n_bytes();
            bucket.push(snap);
        }
    }

    pub fn lookup_longest<'a>(
        &'a self,
        identity: &SnapshotIdentity,
        request_tokens: &[i32],
    ) -> Option<PrefixCacheHit<'a>> {
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
            if let Some(bucket) = self.buckets.get(&key) {
                for snap in bucket {
                    if snap.prefix_tokens == request_tokens[..prefix_len] {
                        return Some(PrefixCacheHit {
                            snapshot: snap,
                            matched_prefix_len: prefix_len,
                            exact: prefix_len == request_tokens.len(),
                        });
                    }
                }
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
        h ^= (tok as u32 as u64).wrapping_add(0x9e3779b97f4a7c15);
        h = h.wrapping_mul(HASH_PRIME);
    }
    h
}

fn prefix_hashes(tokens: &[i32]) -> Vec<u64> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut h = HASH_SEED;
    for &tok in tokens {
        h ^= (tok as u32 as u64).wrapping_add(0x9e3779b97f4a7c15);
        h = h.wrapping_mul(HASH_PRIME);
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
    fn lookup_filters_by_identity() {
        let id_a = ident(1);
        let id_b = ident(2);
        let mut cache = PrefixCache::new();
        cache.insert(snap(id_a.clone(), &[1, 2, 3], 40));
        assert!(cache.lookup_longest(&id_b, &[1, 2, 3, 4]).is_none());
        assert!(cache.lookup_longest(&id_a, &[1, 2, 3, 4]).is_some());
    }

    #[test]
    fn insert_replaces_exact_prefix_and_updates_bytes() {
        let id = ident(1);
        let mut cache = PrefixCache::new();
        cache.insert(snap(id.clone(), &[1, 2, 3], 32));
        let before = cache.total_bytes();
        cache.insert(snap(id.clone(), &[1, 2, 3], 96));
        assert_eq!(cache.len(), 1);
        assert!(cache.total_bytes() > before);
        let hit = cache.lookup_longest(&id, &[1, 2, 3]).expect("exact hit");
        assert!(hit.exact);
    }
}
