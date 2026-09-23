//! Eviction and expiry policy shared by the process-local snapshot caches.
//!
//! The policy owns only per-entry metadata keyed by a stable [`EntryId`];
//! caches keep the payloads and drop them for every id the policy releases.
//!
//! - **Frecency.** An entry's score starts at 1 and each hit adds 1 after
//!   decaying the old score with half-life `H`:
//!   `score_now = score * 2^(-(now - last_access) / H)`.
//! - **Victims** are the unpinned entries with the lowest
//!   `score_now * value / bytes` (reusable prefix tokens per retained byte),
//!   ties broken by older access. `H = 0` is the recency limit of that rank
//!   and is implemented exactly as LRU; a very large `H` tends to
//!   frequency-per-byte instead.
//! - **Expiry.** [`SnapshotPolicy::sweep`] releases unpinned entries idle
//!   longer than `idle_ttl` or older than `max_age` (zero disables either).
//! - **Pins** exclude an entry from eviction and expiry until unpinned.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Monotonic time source, as an offset from an arbitrary fixed origin.
pub trait Clock: Send + Sync {
    fn now(&self) -> Duration;
}

pub struct MonotonicClock(Instant);

impl MonotonicClock {
    pub fn new() -> Self {
        Self(Instant::now())
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MonotonicClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
}

/// Manually advanced clock for tests; clones share one time.
#[derive(Clone, Default)]
pub struct FakeClock(Arc<AtomicU64>);

impl FakeClock {
    pub fn advance(&self, by: Duration) {
        self.0.fetch_add(by.as_nanos() as u64, Ordering::Relaxed);
    }
}

impl Clock for FakeClock {
    fn now(&self) -> Duration {
        Duration::from_nanos(self.0.load(Ordering::Relaxed))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotPolicyConfig {
    /// Frecency half-life; zero ranks by recency alone (LRU).
    pub half_life: Duration,
    /// Expire unpinned entries idle longer than this; zero disables.
    pub idle_ttl: Duration,
    /// Expire unpinned entries older than this; zero disables.
    pub max_age: Duration,
}

impl SnapshotPolicyConfig {
    /// Pure LRU without expiry: the behavior of bench and CLI caches.
    pub const LRU: Self = Self {
        half_life: Duration::ZERO,
        idle_ttl: Duration::ZERO,
        max_age: Duration::ZERO,
    };
}

impl Default for SnapshotPolicyConfig {
    /// Long-running service defaults.
    fn default() -> Self {
        Self {
            half_life: Duration::from_secs(600),
            idle_ttl: Duration::from_secs(3600),
            max_age: Duration::from_secs(86_400),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct EntryId(u64);

/// Entries the policy released; the owning cache must drop their payloads.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Evicted {
    pub ids: Vec<EntryId>,
    pub bytes: u64,
}

impl Evicted {
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

struct Meta {
    bytes: u64,
    value: u64,
    inserted_at: Duration,
    last_access: Duration,
    /// Access order; breaks ties between equal timestamps.
    seq: u64,
    score: f64,
    pins: u32,
}

pub struct SnapshotPolicy {
    config: SnapshotPolicyConfig,
    clock: Arc<dyn Clock>,
    max_bytes: u64,
    total_bytes: u64,
    next_id: u64,
    seq: u64,
    entries: HashMap<EntryId, Meta>,
}

impl SnapshotPolicy {
    pub fn new(max_bytes: u64, config: SnapshotPolicyConfig) -> Self {
        Self::with_clock(max_bytes, config, Arc::new(MonotonicClock::new()))
    }

    pub fn with_clock(max_bytes: u64, config: SnapshotPolicyConfig, clock: Arc<dyn Clock>) -> Self {
        Self {
            config,
            clock,
            max_bytes,
            total_bytes: 0,
            next_id: 0,
            seq: 0,
            entries: HashMap::new(),
        }
    }

    pub fn config(&self) -> SnapshotPolicyConfig {
        self.config
    }

    pub fn set_config(&mut self, config: SnapshotPolicyConfig) {
        self.config = config;
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Callers evict afterwards with [`Self::evict_to_budget`].
    pub fn set_max_bytes(&mut self, max_bytes: u64) {
        self.max_bytes = max_bytes;
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    pub fn pinned_bytes(&self) -> u64 {
        self.entries
            .values()
            .filter(|meta| meta.pins > 0)
            .map(|meta| meta.bytes)
            .sum()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, id: EntryId) -> bool {
        self.entries.contains_key(&id)
    }

    /// Whether `bytes` can be admitted by evicting only unpinned entries.
    pub fn fits_strict(&self, bytes: u64) -> bool {
        bytes
            .checked_add(self.pinned_bytes())
            .is_some_and(|required| required <= self.max_bytes)
    }

    /// Track a new entry. `value` is its reusable prefix length in tokens.
    pub fn insert(&mut self, bytes: u64, value: u64) -> EntryId {
        let id = EntryId(self.next_id);
        self.next_id += 1;
        self.seq += 1;
        let now = self.clock.now();
        self.entries.insert(
            id,
            Meta {
                bytes,
                value,
                inserted_at: now,
                last_access: now,
                seq: self.seq,
                score: 1.0,
                pins: 0,
            },
        );
        self.total_bytes += bytes;
        id
    }

    /// Record a hit.
    pub fn touch(&mut self, id: EntryId) {
        let now = self.clock.now();
        let half_life = self.config.half_life;
        self.seq += 1;
        let seq = self.seq;
        if let Some(meta) = self.entries.get_mut(&id) {
            meta.score = decayed(meta, now, half_life) + 1.0;
            meta.last_access = now;
            meta.seq = seq;
        }
    }

    pub fn remove(&mut self, id: EntryId) -> Option<u64> {
        let meta = self.entries.remove(&id)?;
        self.total_bytes -= meta.bytes;
        Some(meta.bytes)
    }

    pub fn pin(&mut self, id: EntryId) -> bool {
        self.entries
            .get_mut(&id)
            .map(|meta| meta.pins += 1)
            .is_some()
    }

    pub fn unpin(&mut self, id: EntryId) {
        if let Some(meta) = self.entries.get_mut(&id) {
            meta.pins = meta.pins.saturating_sub(1);
        }
    }

    /// Most recently inserted or touched entry.
    pub fn most_recent(&self) -> Option<EntryId> {
        self.entries
            .iter()
            .max_by_key(|(_, meta)| meta.seq)
            .map(|(&id, _)| id)
    }

    /// Evict until within budget, never releasing pinned entries or `protect`.
    pub fn evict_to_budget(&mut self, protect: Option<EntryId>) -> Evicted {
        let mut evicted = Evicted::default();
        while self.total_bytes > self.max_bytes {
            let Some(victim) = self.victim(protect) else {
                break;
            };
            self.release(victim, &mut evicted);
        }
        evicted
    }

    /// Evict by rank until at least `bytes` are released or only pinned
    /// entries remain (memory-pressure relief).
    pub fn evict_for(&mut self, bytes: u64) -> Evicted {
        let mut evicted = Evicted::default();
        while evicted.bytes < bytes {
            let Some(victim) = self.victim(None) else {
                break;
            };
            self.release(victim, &mut evicted);
        }
        evicted
    }

    /// Release unpinned entries past the idle TTL or maximum age.
    pub fn sweep(&mut self) -> Evicted {
        let now = self.clock.now();
        let SnapshotPolicyConfig {
            idle_ttl, max_age, ..
        } = self.config;
        let expired: Vec<EntryId> = self
            .entries
            .iter()
            .filter(|(_, meta)| {
                meta.pins == 0
                    && ((!idle_ttl.is_zero() && now.saturating_sub(meta.last_access) > idle_ttl)
                        || (!max_age.is_zero() && now.saturating_sub(meta.inserted_at) > max_age))
            })
            .map(|(&id, _)| id)
            .collect();
        let mut evicted = Evicted::default();
        for id in expired {
            self.release(id, &mut evicted);
        }
        evicted
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_bytes = 0;
    }

    fn release(&mut self, id: EntryId, evicted: &mut Evicted) {
        if let Some(bytes) = self.remove(id) {
            evicted.ids.push(id);
            evicted.bytes += bytes;
        }
    }

    fn victim(&self, protect: Option<EntryId>) -> Option<EntryId> {
        let now = self.clock.now();
        let half_life = self.config.half_life;
        self.entries
            .iter()
            .filter(|(id, meta)| meta.pins == 0 && Some(**id) != protect)
            .map(|(&id, meta)| {
                // Zero half-life is the recency limit: every priority ties.
                let priority = if half_life.is_zero() {
                    0.0
                } else {
                    decayed(meta, now, half_life) * meta.value as f64 / meta.bytes.max(1) as f64
                };
                (priority, meta.last_access, meta.seq, id)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)))
            .map(|(.., id)| id)
    }
}

fn decayed(meta: &Meta, now: Duration, half_life: Duration) -> f64 {
    if half_life.is_zero() {
        return meta.score;
    }
    let elapsed = now.saturating_sub(meta.last_access).as_secs_f64();
    meta.score * (-elapsed / half_life.as_secs_f64()).exp2()
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    fn policy(max_bytes: u64, config: SnapshotPolicyConfig) -> (SnapshotPolicy, FakeClock) {
        let clock = FakeClock::default();
        (
            SnapshotPolicy::with_clock(max_bytes, config, Arc::new(clock.clone())),
            clock,
        )
    }

    fn frecency(half_life_secs: u32) -> SnapshotPolicyConfig {
        SnapshotPolicyConfig {
            half_life: half_life_secs * S,
            ..SnapshotPolicyConfig::LRU
        }
    }

    #[test]
    fn decay_orders_old_hot_against_new_cold() {
        // A score of 4 decays to 2 after one half-life and 0.5 after three;
        // a fresh entry scores 1.
        for (idle_secs, hot_survives) in [(600, true), (1800, false)] {
            let (mut policy, clock) = policy(20, frecency(600));
            let hot = policy.insert(10, 100);
            for _ in 0..3 {
                policy.touch(hot);
            }
            clock.advance(idle_secs * S);
            let cold = policy.insert(10, 100);
            let newest = policy.insert(10, 100);
            let expected = if hot_survives { cold } else { hot };
            assert_eq!(policy.evict_to_budget(Some(newest)).ids, [expected]);
        }
    }

    #[test]
    fn victim_is_lowest_value_per_byte() {
        let (mut policy, _) = policy(300, frecency(600));
        let dense = policy.insert(100, 1000); // 10 tokens/byte
        let sparse = policy.insert(100, 100); // 1 token/byte
        let older = policy.insert(100, 500);
        let newest = policy.insert(100, 1);
        assert_eq!(policy.evict_to_budget(Some(newest)).ids, [sparse]);
        assert!(policy.contains(dense) && policy.contains(older));
    }

    #[test]
    fn equal_rank_ties_break_to_older_access() {
        let (mut policy, _) = policy(20, frecency(600));
        let first = policy.insert(10, 10);
        let second = policy.insert(10, 10);
        let newest = policy.insert(10, 10);
        assert_eq!(policy.evict_to_budget(Some(newest)).ids, [first]);
        assert!(policy.contains(second));
    }

    #[test]
    fn zero_half_life_is_lru_by_last_access() {
        let (mut policy, clock) = policy(30, SnapshotPolicyConfig::LRU);
        let a = policy.insert(10, 1);
        let b = policy.insert(10, 1_000_000); // value cannot save it
        let c = policy.insert(10, 1);
        for _ in 0..10 {
            policy.touch(b); // nor can frequency
        }
        clock.advance(S);
        policy.touch(a);
        policy.touch(c);
        let d = policy.insert(10, 1);
        assert_eq!(policy.evict_to_budget(Some(d)).ids, [b]);
        policy.touch(a);
        let e = policy.insert(10, 1);
        assert_eq!(policy.evict_to_budget(Some(e)).ids, [c]);
    }

    #[test]
    fn idle_expiry_at_ttl_boundary_and_touch_extends() {
        let config = SnapshotPolicyConfig {
            idle_ttl: 3600 * S,
            ..SnapshotPolicyConfig::LRU
        };
        let (mut policy, clock) = policy(100, config);
        let idle = policy.insert(10, 1);
        let used = policy.insert(10, 1);
        clock.advance(1800 * S);
        policy.touch(used);
        clock.advance(1799 * S);
        assert!(policy.sweep().is_empty()); // idle 3599 s
        clock.advance(2 * S);
        assert_eq!(policy.sweep().ids, [idle]); // idle 3601 s
        assert!(policy.contains(used));
        assert_eq!(policy.total_bytes(), 10);
    }

    #[test]
    fn max_age_expires_even_hot_entries() {
        let config = SnapshotPolicyConfig {
            max_age: 100 * S,
            ..SnapshotPolicyConfig::LRU
        };
        let (mut policy, clock) = policy(100, config);
        let hot = policy.insert(10, 1);
        for _ in 0..99 {
            clock.advance(S);
            policy.touch(hot);
        }
        assert!(policy.sweep().is_empty());
        clock.advance(2 * S);
        assert_eq!(policy.sweep().ids, [hot]);
    }

    #[test]
    fn zero_ttl_and_age_disable_expiry() {
        let (mut policy, clock) = policy(100, SnapshotPolicyConfig::LRU);
        policy.insert(10, 1);
        clock.advance(Duration::from_secs(10_000_000));
        assert!(policy.sweep().is_empty());
    }

    #[test]
    fn pinned_entries_survive_expiry_and_eviction() {
        let config = SnapshotPolicyConfig {
            idle_ttl: 10 * S,
            ..frecency(600)
        };
        let (mut policy, clock) = policy(20, config);
        let pinned = policy.insert(10, 1);
        assert!(policy.pin(pinned));
        clock.advance(60 * S);
        assert!(policy.sweep().is_empty());

        let other = policy.insert(10, 1000);
        let newest = policy.insert(10, 1000);
        assert_eq!(policy.evict_to_budget(Some(newest)).ids, [other]);
        assert!(policy.contains(pinned));
        assert!(!policy.fits_strict(11));
        assert!(policy.fits_strict(10));

        policy.unpin(pinned);
        assert_eq!(policy.sweep().ids, [pinned]);
    }

    #[test]
    fn eviction_stops_when_only_pinned_or_protected_remain() {
        let (mut policy, _) = policy(5, frecency(600));
        let pinned = policy.insert(10, 1);
        policy.pin(pinned);
        let newest = policy.insert(10, 1);
        assert!(policy.evict_to_budget(Some(newest)).is_empty());
        assert_eq!(policy.total_bytes(), 20);
    }

    #[test]
    fn evict_for_releases_lowest_rank_until_enough() {
        let (mut policy, _) = policy(1000, frecency(600));
        let low = policy.insert(10, 1);
        let mid = policy.insert(10, 5);
        let high = policy.insert(10, 50);
        let pinned = policy.insert(10, 0);
        policy.pin(pinned);
        let evicted = policy.evict_for(15);
        assert_eq!(evicted.ids, [low, mid]);
        assert_eq!(evicted.bytes, 20);
        let evicted = policy.evict_for(1000);
        assert_eq!(evicted.ids, [high]);
        assert_eq!(policy.len(), 1);
        assert!(policy.evict_for(1).is_empty());
    }
}
