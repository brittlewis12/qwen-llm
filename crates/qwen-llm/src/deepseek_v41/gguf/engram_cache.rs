//! Bounded packed-row cache for the streamed Engram tables.
//!
//! Rows are immutable, so a cache of the serialized bytes is exact: a hit
//! decodes through the same codec call as a disk read. Capacity is measured
//! in packed payload bytes, not rows, because the row stride depends on the
//! table encoding. Eviction is CLOCK (second chance), which keeps the
//! bookkeeping O(1) per row without an allocation per hit.

use std::collections::HashMap;

/// Table row identity. Layer is the logical model layer, row is local to that
/// layer's table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct RowKey {
    pub(super) layer: usize,
    pub(super) row: u64,
}

struct Slot {
    key: RowKey,
    data: Box<[u8]>,
    referenced: bool,
}

/// A fixed-capacity row cache. All mutation is serialized by the caller's
/// lock; the type itself owns no lock so tests can drive it directly.
pub(super) struct RowCache {
    capacity_bytes: usize,
    bytes: usize,
    entries: HashMap<RowKey, usize>,
    slots: Vec<Slot>,
    hand: usize,
    hits: u64,
    misses: u64,
}

/// Current cache footprint and hit counts. Hits and misses count actual row
/// reads; advisory probes use [`RowCache::contains`] and are not counted.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RowCacheStats {
    pub(super) capacity_bytes: usize,
    pub(super) bytes: usize,
    pub(super) rows: usize,
    pub(super) hits: u64,
    pub(super) misses: u64,
}

impl RowCache {
    pub(super) fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            bytes: 0,
            entries: HashMap::new(),
            slots: Vec::new(),
            hand: 0,
            hits: 0,
            misses: 0,
        }
    }

    pub(super) fn stats(&self) -> RowCacheStats {
        RowCacheStats {
            capacity_bytes: self.capacity_bytes,
            bytes: self.bytes,
            rows: self.slots.len(),
            hits: self.hits,
            misses: self.misses,
        }
    }

    /// Look up a row without counting a read. Used by the advisory path.
    pub(super) fn contains(&mut self, key: RowKey) -> bool {
        let Some(&index) = self.entries.get(&key) else {
            return false;
        };
        self.slots[index].referenced = true;
        true
    }

    /// Look up a row and mark it for a second chance. The returned slice
    /// remains valid until the next insert; callers copy it before dropping
    /// the lock.
    pub(super) fn get(&mut self, key: RowKey) -> Option<&[u8]> {
        let Some(&index) = self.entries.get(&key) else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        let slot = &mut self.slots[index];
        slot.referenced = true;
        self.hits = self.hits.saturating_add(1);
        Some(&slot.data)
    }

    /// Insert a row's packed bytes. A row larger than the whole cache is not
    /// stored. Rows for one table share a stride, so a repeated key has the
    /// same length and is refreshed in place.
    pub(super) fn insert(&mut self, key: RowKey, data: &[u8]) {
        if data.len() > self.capacity_bytes {
            return;
        }
        if let Some(&index) = self.entries.get(&key) {
            let slot = &mut self.slots[index];
            slot.referenced = true;
            if slot.data.len() == data.len() {
                slot.data.copy_from_slice(data);
            }
            return;
        }
        self.make_room(data.len());
        if self.bytes.saturating_add(data.len()) > self.capacity_bytes {
            return;
        }
        let index = self.slots.len();
        self.bytes = self.bytes.saturating_add(data.len());
        self.slots.push(Slot {
            key,
            data: data.into(),
            referenced: false,
        });
        self.entries.insert(key, index);
    }

    fn make_room(&mut self, need: usize) {
        while self.bytes.saturating_add(need) > self.capacity_bytes && !self.slots.is_empty() {
            self.evict_one();
        }
    }

    fn evict_one(&mut self) {
        loop {
            if self.slots.is_empty() {
                self.hand = 0;
                return;
            }
            if self.hand >= self.slots.len() {
                self.hand = 0;
            }
            if self.slots[self.hand].referenced {
                self.slots[self.hand].referenced = false;
                self.hand = (self.hand + 1) % self.slots.len();
                continue;
            }
            let removed = self.slots.swap_remove(self.hand);
            self.entries.remove(&removed.key);
            self.bytes = self.bytes.saturating_sub(removed.data.len());
            if self.hand >= self.slots.len() {
                self.hand = 0;
            } else {
                let moved = self.slots[self.hand].key;
                self.entries.insert(moved, self.hand);
            }
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(row: u64) -> RowKey {
        RowKey { layer: 0, row }
    }

    #[test]
    fn hits_return_the_inserted_bytes() {
        let mut cache = RowCache::new(32);
        cache.insert(key(1), b"packed-row");
        assert_eq!(cache.get(key(1)), Some(&b"packed-row"[..]));
        assert_eq!(cache.get(key(2)), None);
        assert_eq!(cache.stats().bytes, 10);
    }

    #[test]
    fn eviction_stays_within_capacity_and_keeps_remaining_rows_intact() {
        let mut cache = RowCache::new(3 * 4);
        for row in 0..3 {
            cache.insert(key(row), &[row as u8; 4]);
        }
        assert_eq!(cache.stats().rows, 3);
        assert_eq!(cache.stats().bytes, 12);
        cache.insert(key(3), &[3; 4]);
        assert_eq!(cache.stats().bytes, 12);
        assert_eq!(cache.stats().rows, 3);
        assert_eq!(cache.get(key(3)), Some(&[3; 4][..]));
        assert_eq!(cache.get(key(0)), None);
        for row in 1..=3 {
            assert_eq!(cache.get(key(row)), Some(&[row as u8; 4][..]));
        }
    }

    #[test]
    fn second_chance_retains_a_referenced_row() {
        let mut cache = RowCache::new(2 * 4);
        cache.insert(key(0), &[0; 4]);
        cache.insert(key(1), &[1; 4]);
        assert_eq!(cache.get(key(0)), Some(&[0; 4][..]));
        cache.insert(key(2), &[2; 4]);
        assert_eq!(cache.get(key(0)), Some(&[0; 4][..]));
        assert_eq!(cache.get(key(1)), None);
        assert_eq!(cache.get(key(2)), Some(&[2; 4][..]));
    }

    #[test]
    fn oversized_row_is_not_cached() {
        let mut cache = RowCache::new(3);
        cache.insert(key(0), &[0; 4]);
        assert_eq!(cache.stats().rows, 0);
        assert_eq!(cache.stats().bytes, 0);
        assert_eq!(cache.get(key(0)), None);
    }
}
