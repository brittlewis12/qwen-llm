//! Test-private experiment. No public knob, new execution cut or workspace reserve.
use super::super::snapshot_cache::PrefixHit;
use super::*;

/// Proven state on the full-row schedule starting at this verified shared cut.
/// An absent origin cannot be recovered from arithmetic alignment alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ChunkOrigin {
    shared_cut: usize,
    rows: usize,
    precision: FastPrecision,
    committed: usize,
}

impl ChunkOrigin {
    pub(super) fn from_shared(
        shared_cut: usize,
        rows: usize,
        precision: FastPrecision,
    ) -> Option<Self> {
        (shared_cut > 0 && rows > 0).then_some(Self {
            shared_cut,
            rows,
            precision,
            committed: shared_cut,
        })
    }

    pub(super) fn restored(
        shared_cut: usize,
        rows: usize,
        precision: FastPrecision,
        committed: usize,
    ) -> Option<Self> {
        let mut origin = Self::from_shared(shared_cut, rows, precision)?;
        let suffix = committed.checked_sub(shared_cut)?;
        if suffix == 0 || suffix % rows != 0 {
            return None;
        }
        origin.committed = committed;
        Some(origin)
    }

    pub(super) fn matches(
        self,
        shared_cut: usize,
        rows: usize,
        precision: FastPrecision,
        committed: usize,
    ) -> bool {
        self.shared_cut == shared_cut
            && self.rows == rows
            && self.precision == precision
            && self.committed == committed
    }

    pub(super) fn at(self, committed: usize) -> bool {
        self.committed == committed
    }

    pub(super) fn advance(origin: &mut Option<Self>, start: usize, end: usize) {
        *origin = origin.and_then(|mut value| {
            let relative = start.checked_sub(value.shared_cut)?;
            if value.committed != start
                || relative % value.rows != 0
                || end.checked_sub(start)? != value.rows
            {
                return None;
            }
            value.committed = end;
            Some(value)
        });
    }
}

/// Only the selected cache namespace or an exact saved live origin proves
/// the suffix schedule. An aligned position by itself is not provenance.
pub(super) fn resume_origin(
    saved: Option<ChunkOrigin>,
    shared_cut: usize,
    rows: usize,
    precision: FastPrecision,
    matched: usize,
    restored: Option<Schedule>,
) -> Option<ChunkOrigin> {
    match restored {
        Some(Schedule::FastSharedSplitV1) if matched == shared_cut => {
            ChunkOrigin::from_shared(shared_cut, rows, precision)
        }
        Some(Schedule::FastChunkCheckpointV1 {
            shared_cut: source_cut,
            precision: source_precision,
        }) if source_cut == shared_cut && source_precision == precision => {
            ChunkOrigin::restored(shared_cut, rows, precision, matched)
        }
        None if matched > 0 => {
            saved.filter(|origin| origin.matches(shared_cut, rows, precision, matched))
        }
        _ => None,
    }
}

/// Last existing FULL suffix chunk strictly before the endpoint, even when the
/// prompt ends exactly on the relative grid. Never introduce a new chunk split.
pub(super) fn target(prompt: usize, shared_cut: usize, rows: usize) -> Option<usize> {
    if rows == 0 || shared_cut == 0 {
        return None;
    }
    let chunks = prompt.checked_sub(shared_cut)?.checked_sub(1)? / rows;
    let cut = shared_cut.checked_add(rows.checked_mul(chunks)?)?;
    (cut > shared_cut && cut < prompt).then_some(cut)
}

pub(super) fn namespace(
    shared: CacheNamespace,
    shared_cut: usize,
    precision: FastPrecision,
) -> Option<CacheNamespace> {
    (shared.schedule == Schedule::FastSharedSplitV1
        && shared.lineage == PackedLineage::Fast
        && shared.rows > 0
        && shared_cut > 0)
        .then_some(CacheNamespace {
            schedule: Schedule::FastChunkCheckpointV1 {
                shared_cut,
                precision,
            },
            ..shared
        })
}

/// Consume all candidates here, so the unselected Arc is gone before admission.
/// Equal lengths prefer live, then the incumbent shared/Exact candidate.
pub(super) fn choose<V>(
    live: usize,
    incumbent: Option<PrefixHit<V>>,
    deep: Option<PrefixHit<V>>,
) -> (Option<PrefixHit<V>>, bool) {
    let incumbent = incumbent.filter(|hit| hit.prefix_len > live);
    let deep = deep.filter(|hit| hit.prefix_len > live);
    if deep
        .as_ref()
        .is_some_and(|hit| hit.prefix_len > incumbent.as_ref().map_or(0, |hit| hit.prefix_len))
    {
        (deep, true)
    } else {
        (incumbent, false)
    }
}

/// Pin IDs rather than retaining snapshot Arcs across optional capture admission.
pub(super) fn with_pins<V, T>(
    cache: &mut SnapshotCache<V, CacheNamespace>,
    ids: [Option<EntryId>; 2],
    f: impl FnOnce(&mut SnapshotCache<V, CacheNamespace>) -> T,
) -> T {
    let first = ids[0].filter(|&id| cache.pin(id));
    let second = ids[1]
        .filter(|id| Some(*id) != first)
        .filter(|&id| cache.pin(id));
    struct Pins<'a, V> {
        cache: &'a mut SnapshotCache<V, CacheNamespace>,
        ids: [Option<EntryId>; 2],
    }
    impl<V> Drop for Pins<'_, V> {
        fn drop(&mut self) {
            for id in self.ids.into_iter().flatten() {
                self.cache.unpin(id);
            }
        }
    }
    let guard = Pins {
        cache,
        ids: [first, second],
    };
    f(&mut *guard.cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_formula_checked_and_strict() {
        assert_eq!(target(41420, 51, 2048), Some(41011));
        assert_eq!(target(51 + 2 * 2048, 51, 2048), Some(51 + 2048));
        assert_eq!(target(51 + 2048 + 1, 51, 2048), Some(51 + 2048));
        for (p, s, r) in [
            (51, 51, 2048),
            (50, 51, 2048),
            (2099, 51, 2048),
            (usize::MAX, 51, 0),
            (100, 0, 8),
        ] {
            assert_eq!(target(p, s, r), None);
        }
        let c = target(usize::MAX, 51, 2048).unwrap();
        assert!(c < usize::MAX && (c - 51) % 2048 == 0);
    }

    #[test]
    fn provenance_cannot_recover_after_partial_alignment() {
        let mut origin = ChunkOrigin::from_shared(51, 2048, FastPrecision::Half);
        ChunkOrigin::advance(&mut origin, 51, 2099);
        assert!(origin.unwrap().matches(51, 2048, FastPrecision::Half, 2099));
        // Cancellation before the next chunk preserves exactly this origin.
        assert!(!origin.unwrap().at(2100));
        ChunkOrigin::advance(&mut origin, 2099, 2199);
        assert!(origin.is_none());
        ChunkOrigin::advance(&mut origin, 2199, 4147); // +1948: aligned again, still invalid.
        ChunkOrigin::advance(&mut origin, 4147, 6195);
        assert!(origin.is_none());
        let restored = ChunkOrigin::restored(51, 2048, FastPrecision::Half, 41011).unwrap();
        assert!(!restored.matches(52, 2048, FastPrecision::Half, 41011));
        assert!(!restored.matches(51, 512, FastPrecision::Half, 41011));
        assert!(!restored.matches(51, 2048, FastPrecision::F32, 41011));
        assert!(ChunkOrigin::restored(51, 2048, FastPrecision::Half, 41012).is_none());
        assert!(ChunkOrigin::restored(51, 2048, FastPrecision::Half, 51).is_none());
    }

    #[test]
    fn lookup_namespace_strict_prefix_and_ties() {
        let shared = CacheNamespace {
            lineage: PackedLineage::Fast,
            schedule: Schedule::FastSharedSplitV1,
            rows: 8,
        };
        let ns = namespace(shared, 3, FastPrecision::Half).unwrap();
        let mut cache = SnapshotCache::new(1024, Default::default());
        let tokens = vec![1; 20];
        assert!(cache.insert_strict_in(shared, tokens[..3].to_vec(), (), 10));
        assert!(cache.insert_strict_in(ns, tokens[..19].to_vec(), (), 10));
        assert!(cache.peek_best_prefix_in(&ns, &tokens[..19]).is_none());
        for wrong in [
            namespace(shared, 4, FastPrecision::Half).unwrap(),
            namespace(shared, 3, FastPrecision::F32).unwrap(),
            CacheNamespace { rows: 16, ..ns },
        ] {
            assert!(cache.peek_best_prefix_in(&wrong, &tokens).is_none());
        }
        let mut wrong_tokens = tokens.clone();
        wrong_tokens[10] = 2;
        assert!(cache.peek_best_prefix_in(&ns, &wrong_tokens).is_none());
        for (live, expected, deep_selected) in
            [(0, Some(19), true), (19, None, false), (20, None, false)]
        {
            let incumbent = cache.peek_exact_in(&shared, &tokens[..3]);
            let deep = cache.peek_best_prefix_in(&ns, &tokens);
            let (chosen, actual_deep) = choose(live, incumbent, deep);
            assert_eq!(chosen.map(|hit| hit.prefix_len), expected);
            assert_eq!(actual_deep, deep_selected);
        }
        assert!(
            namespace(
                CacheNamespace {
                    lineage: PackedLineage::Exact,
                    ..shared
                },
                3,
                FastPrecision::Half
            )
            .is_none()
        );
        assert!(namespace(CacheNamespace { rows: 0, ..shared }, 3, FastPrecision::Half).is_none());
        // Losing candidates do not hold Arcs through capture admission.
        let (hit, _) = choose(
            19,
            cache.peek_exact_in(&shared, &tokens[..3]),
            cache.peek_best_prefix_in(&ns, &tokens),
        );
        assert!(hit.is_none());
        cache.evict_for(u64::MAX);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn source_origin_requires_shared_restore_or_saved_proof() {
        let half = FastPrecision::Half;
        let deep = Schedule::FastChunkCheckpointV1 {
            shared_cut: 51,
            precision: half,
        };
        let shared =
            resume_origin(None, 51, 2048, half, 51, Some(Schedule::FastSharedSplitV1)).unwrap();
        assert!(shared.at(51));
        assert!(resume_origin(None, 51, 2048, half, 51, None).is_none());
        assert!(resume_origin(None, 51, 2048, half, 2099, None).is_none());
        let origin = resume_origin(None, 51, 2048, half, 2099, Some(deep)).unwrap();
        assert_eq!(
            resume_origin(Some(origin), 51, 2048, half, 2099, None),
            Some(origin)
        );
        // Reset, wrong committed position, wrong origin, Exact and precision
        // changes cannot launder a previously valid origin.
        assert!(resume_origin(Some(origin), 51, 2048, half, 0, None).is_none());
        assert!(resume_origin(Some(origin), 51, 2048, half, 2100, None).is_none());
        assert!(resume_origin(Some(origin), 52, 2048, half, 2099, Some(deep)).is_none());
        assert!(
            resume_origin(Some(origin), 51, 2048, half, 2099, Some(Schedule::ExactV1)).is_none()
        );
        assert!(
            resume_origin(Some(origin), 51, 2048, FastPrecision::F32, 2099, Some(deep)).is_none()
        );
    }

    #[test]
    fn optional_capture_refusal_preserves_protected_entries_and_releases_pins() {
        let shared = CacheNamespace {
            lineage: PackedLineage::Fast,
            schedule: Schedule::FastSharedSplitV1,
            rows: 8,
        };
        let ns = namespace(shared, 3, FastPrecision::Half).unwrap();
        let mut cache = SnapshotCache::new(128, Default::default());
        assert!(cache.insert_strict_in(shared, vec![1, 2, 3], (), 32));
        assert!(cache.insert_strict_in(ns, vec![1; 11], (), 64));
        let s = cache.entry_for_in(&shared, &[1, 2, 3]);
        let d = cache.entry_for_in(&ns, &[1; 11]);
        assert_eq!(cache.strict_eligibility_in(&ns, &[1; 19], 20), Some(96));
        with_pins(&mut cache, [s, d], |cache| {
            assert_eq!(cache.pinned_bytes(), 96);
            assert!(cache.strict_eligibility_in(&ns, &[1; 19], 20).is_none());
            cache.evict_for(u64::MAX);
            assert_eq!(cache.len(), 2);
        });
        assert_eq!(cache.pinned_bytes(), 0);
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_pins(&mut cache, [s, d], |_| panic!("optional capture unwound"));
        }));
        assert!(failed.is_err());
        assert_eq!(cache.pinned_bytes(), 0);
        with_pins(&mut cache, [s, s], |cache| {
            assert_eq!(cache.pinned_bytes(), 32)
        });
        assert_eq!(cache.pinned_bytes(), 0);
    }
}
