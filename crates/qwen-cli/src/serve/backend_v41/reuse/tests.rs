use super::*;
use qwen_llm::deepseek_v41::forward::StateError;
use qwen_llm::metal::{MemoryAdmissionDenied, MetalMemoryAdmissionReason as Reason};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

fn capture_signals(process_limit_remaining_bytes: Option<u64>) -> MetalMemorySignals {
    MetalMemorySignals {
        host_budget: Some(qwen_llm::metal::HostMemoryBudget {
            physical_bytes: 1024 << 30,
            conservative_available_bytes: 1024 << 30,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
        recommended_max_bytes: u64::MAX,
        current_allocated_bytes: 0,
        process_limit_remaining_bytes,
    }
}

fn denied(reason: Reason) -> StateError {
    StateError::MemoryAdmissionDenied(
        MemoryAdmissionDenied::new(
            reason,
            Some(4096),
            MetalMemorySignals {
                host_budget: Some(qwen_llm::metal::HostMemoryBudget {
                    physical_bytes: 1024 << 30,
                    conservative_available_bytes: 1024 << 30,
                    external_gap_bytes: 0,
                    private_shadow_bytes: 0,
                }),
                recommended_max_bytes: 1024,
                current_allocated_bytes: 0,
                process_limit_remaining_bytes: Some(1024),
            },
            Some(1024),
        )
        .expect("test reason refuses admission"),
    )
}

struct Tracked {
    tokens: Vec<u32>,
    drops: Rc<Cell<usize>>,
}
impl Drop for Tracked {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

struct GrowthFixture {
    live: Option<Live<Tracked>>,
    pressure: bool,
}

#[test]
fn live_selection_is_pure_causal_and_live_first_then_snapshot_on_growth_refusal() {
    let drops = Rc::new(Cell::new(0));
    let make = || Live {
        state: Tracked {
            tokens: vec![1, 2],
            drops: drops.clone(),
        },
        tokens: vec![1, 2],
        logits: vec![7.],
    };
    let live = make();
    for (ids, position, healthy, expected) in [
        (vec![1, 2, 3, 4], 2, true, Some(2)),
        (vec![1, 2], 2, true, Some(2)),
        (vec![1, 9, 3], 2, true, None),
        (vec![1], 2, true, None),
        (vec![1, 2, 3], 1, true, None),
        (vec![1, 2, 3], 2, false, None),
    ] {
        assert_eq!(
            operations::live_prefix_len(&live, &ids, position, healthy),
            expected
        );
        assert_eq!(live.tokens, [1, 2]);
        assert_eq!(live.state.tokens, [1, 2]);
        assert_eq!(drops.get(), 0);
    }
    drop(live);
    for (pressure, snapshot_available) in [(false, true), (true, true), (true, false)] {
        let before = drops.get();
        let restored = operations::live_then_snapshot(
            Some(make()),
            |_| {
                if pressure {
                    Err(denied(Reason::ProcessInsufficient))
                } else {
                    Ok(())
                }
            },
            || {
                assert!(pressure, "cheap live wins even over a longer snapshot");
                assert_eq!(drops.get(), before + 1, "old owner released before restore");
                Ok(snapshot_available.then(|| Restored {
                    live: Live {
                        tokens: vec![1, 2, 3],
                        ..make()
                    },
                    source: "snapshot_ram",
                }))
            },
        )
        .unwrap();
        if let Some(restored) = restored {
            assert_eq!(
                restored.source,
                if pressure {
                    "snapshot_ram"
                } else {
                    "live_session"
                }
            );
            assert_eq!(restored.live.tokens.len(), if pressure { 3 } else { 2 });
        } else {
            assert!(
                pressure && !snapshot_available,
                "cold only after both sources"
            );
        }
    }
}

#[test]
fn snapshot_selection_uses_longest_causal_prefix_or_equal_with_logits_without_mutating_payload() {
    for exact_logits in [false, true] {
        let mut cache = SnapshotCache::new(1024, Default::default());
        assert!(cache.insert_strict(vec![1], (vec![1], false), 20));
        assert!(cache.insert_strict(vec![1, 2, 3], (vec![1, 2, 3], false), 40));
        assert!(cache.insert_strict(vec![1, 2, 3, 4], (vec![1, 2, 3, 4], exact_logits), 50));
        let bytes = cache.indexed_bytes();
        let hit =
            operations::snapshot_candidate(&mut cache, &[1, 2, 3, 4], |value| value.1).unwrap();
        assert_eq!(hit.prefix_len, if exact_logits { 4 } else { 3 });
        assert_eq!(hit.value.0, [1, 2, 3, 4][..hit.prefix_len]);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.indexed_bytes(), bytes);
    }
}

#[test]
fn snapshot_zero_budget_and_expiry_do_not_retire_compatible_live() {
    use qwen_llm::snapshot_policy::{FakeClock, SnapshotPolicyConfig};
    for budget in [0, 1024] {
        let clock = FakeClock::default();
        let mut cache = SnapshotCache::with_clock(
            budget,
            SnapshotPolicyConfig {
                idle_ttl: Duration::from_secs(1),
                ..Default::default()
            },
            Arc::new(clock.clone()),
        );
        cache.insert_strict(vec![1], 7u8, 20);
        let drops = Rc::new(Cell::new(0));
        let mut live = Some(Live {
            state: Tracked {
                tokens: vec![1],
                drops: drops.clone(),
            },
            tokens: vec![1],
            logits: vec![],
        });
        clock.advance(Duration::from_secs(2));
        cache.sweep();
        operations::discard_incompatible(&mut live, &[1, 2], |state| (state.tokens.len(), true));
        assert_eq!(cache.len(), 0);
        assert_eq!(drops.get(), 0);
        assert!(live.is_some());
        drop(live);
        assert_eq!(drops.get(), 1);
    }
}

thread_local! {
    static IDLE_BYTES: Cell<u64> = const { Cell::new(0) };
}

struct IdleAllocation(u64);
impl Drop for IdleAllocation {
    fn drop(&mut self) {
        IDLE_BYTES.with(|bytes| bytes.set(bytes.get() - self.0));
    }
}

fn idle_headroom() -> Option<HostMemoryHeadroom> {
    IDLE_BYTES.with(|bytes| finite_headroom((1 << 30) - bytes.get()))
}

struct IdleFixture(Option<Live<IdleAllocation>>);
impl Reuse<IdleAllocation> for IdleFixture {
    fn prepare(&mut self, ids: &[u32]) {
        operations::discard_incompatible(&mut self.0, ids, |_| (1, true));
    }
    fn take(
        &mut self,
        _: &[u32],
        _: usize,
        _: u64,
    ) -> Result<Option<Restored<IdleAllocation>>, ServeError> {
        Ok(self.0.take().map(|live| Restored {
            live,
            source: "live_session",
        }))
    }
    fn checkpoint(&mut self, _: &IdleAllocation, _: &[u32], _: &[f32], _: u64) {}
    fn commit(&mut self, live: Live<IdleAllocation>) {
        self.0 = Some(live);
    }
    fn capacity(&self, _: &IdleAllocation, required: usize) -> usize {
        required
    }
}

#[test]
fn incompatible_idle_owner_drops_before_output_admission_compatible_owner_survives() {
    struct Sink;
    impl GenerationSink for Sink {
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            Ok(())
        }
    }
    let mut settings = Settings::new(
        super::super::tokenizer_fixture::tokenizer(None),
        128,
        1,
        SamplingConfig::default(),
        256,
        "test".into(),
    )
    .unwrap();
    settings.headroom = idle_headroom;
    let request = settings
        .profile()
        .parse(&serde_json::json!({"model":"test", "input":"ab", "max_output_tokens":1}))
        .unwrap();
    for compatible in [false, true] {
        let bytes = if compatible { 1024 } else { (1 << 30) - 1 };
        IDLE_BYTES.with(|live| live.set(bytes));
        let mut fixture = IdleFixture(Some(Live {
            state: IdleAllocation(bytes),
            tokens: vec![if compatible { 97 } else { 99 }],
            logits: vec![],
        }));
        if !compatible {
            assert!(
                super::super::execution::admit_output(&mut fixture, 1 << 20, idle_headroom)
                    .is_err(),
                "obsolete resident would falsely refuse"
            );
            assert!(fixture.0.is_some(), "snapshot relief is not live policy");
        }
        let result = settings
            .execute(
                &request,
                "ab",
                None,
                &mut Sink,
                |_, _, _| {
                    assert_eq!(IDLE_BYTES.with(Cell::get), 0);
                    Ok(())
                },
                |_| {
                    assert!(!compatible);
                    Ok(IdleAllocation(0))
                },
                &mut fixture,
                |_, tokens, _, _| {
                    assert_eq!(tokens, if compatible { &[98][..] } else { &[97, 98][..] });
                    assert_eq!(
                        IDLE_BYTES.with(Cell::get),
                        if compatible { bytes } else { 0 }
                    );
                    let mut logits = vec![0.; 257];
                    logits[256] = 100.;
                    Ok(logits)
                },
                |_, _, _| panic!("EOS is not consumed"),
                || Duration::ZERO,
            )
            .unwrap();
        assert_eq!(
            result.stats.unwrap().matched_tokens,
            usize::from(compatible)
        );
        drop(fixture);
        assert_eq!(
            IDLE_BYTES.with(Cell::get),
            0,
            "only actual owner Drop releases fake allocation"
        );
    }
}
impl Reuse<Tracked> for GrowthFixture {
    fn take(
        &mut self,
        ids: &[u32],
        required: usize,
        _: u64,
    ) -> Result<Option<Restored<Tracked>>, ServeError> {
        let live = self.live.take().unwrap();
        assert!(ids.starts_with(&live.tokens));
        assert!(required > live.tokens.len());
        operations::grow_live(live, |_| {
            Err(if self.pressure {
                denied(Reason::ProcessInsufficient)
            } else {
                StateError::Invalid("bad live state".into())
            })
        })
        .map(|live| {
            live.map(|live| Restored {
                live,
                source: "live_session",
            })
        })
    }
    fn checkpoint(&mut self, _: &Tracked, _: &[u32], _: &[f32], _: u64) {}
    fn commit(&mut self, live: Live<Tracked>) {
        self.live = Some(live);
    }
    fn capacity(&self, _: &Tracked, required: usize) -> usize {
        required
    }
}

#[test]
fn growth_pressure_drops_old_state_before_normal_cold_allocation_but_invalid_is_fatal() {
    struct Sink;
    impl GenerationSink for Sink {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            Ok(())
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut settings = Settings::new(
        super::super::tokenizer_fixture::tokenizer(None),
        128,
        1,
        SamplingConfig::default(),
        256,
        "test".into(),
    )
    .unwrap();
    settings.headroom = || finite_headroom(1 << 30);
    let request = settings
        .profile()
        .parse(&serde_json::json!({"model":"test", "input":"ab", "max_output_tokens":1}))
        .unwrap();
    for pressure in [true, false] {
        let drops = Rc::new(Cell::new(0));
        let mut fixture = GrowthFixture {
            live: Some(Live {
                state: Tracked {
                    tokens: vec![97],
                    drops: drops.clone(),
                },
                tokens: vec![97],
                logits: vec![],
            }),
            pressure,
        };
        let creates = Cell::new(0);
        let admissions = Cell::new(0);
        let result = settings.execute(
            &request,
            "ab",
            None,
            &mut Sink,
            |_, _, _| {
                assert_eq!(drops.get(), 1);
                admissions.set(admissions.get() + 1);
                Ok(())
            },
            |_| {
                assert_eq!(
                    drops.get(),
                    1,
                    "old state must not coexist with cold allocation"
                );
                creates.set(creates.get() + 1);
                Ok(Tracked {
                    tokens: vec![],
                    drops: drops.clone(),
                })
            },
            &mut fixture,
            |state, tokens, _, with_logits| {
                assert!(with_logits);
                assert_eq!(tokens, [97, 98], "cold must replay the whole prompt");
                state.tokens.extend_from_slice(tokens);
                let mut logits = vec![0.; 257];
                logits[256] = 100.;
                Ok(logits)
            },
            |_, _, _| panic!("EOS is not consumed"),
            || Duration::ZERO,
        );
        assert_eq!(creates.get(), usize::from(pressure));
        assert_eq!(admissions.get(), usize::from(pressure));
        if pressure {
            let stats = result.unwrap().stats.unwrap();
            assert_eq!(stats.matched_tokens, 0);
            assert_eq!(stats.phases.unwrap()["reuse_source"], "none");
            assert_eq!(fixture.live.as_ref().unwrap().tokens, [97, 98]);
        } else {
            assert!(
                matches!(result, Err(BackendFailure::Serve(error)) if error.status == 500 && error.message.contains("bad live state"))
            );
            assert!(fixture.live.is_none());
        }
    }
}

#[test]
fn successful_growth_retains_live_tokens_and_logits() {
    let drops = Rc::new(Cell::new(0));
    let live = Live {
        state: Tracked {
            tokens: vec![1],
            drops: drops.clone(),
        },
        tokens: vec![1],
        logits: vec![2.],
    };
    let grown = operations::grow_live(live, |_| Ok(())).unwrap().unwrap();
    assert_eq!(drops.get(), 0);
    assert_eq!(grown.tokens, [1]);
    assert_eq!(grown.logits, [2.]);
}

#[test]
fn snapshot_allocation_pressure_only_allows_cold_retry_after_actual_process_relief() {
    for reason in [
        Some(Reason::ProcessInsufficient),
        Some(Reason::WorkingSetInsufficient),
        Some(Reason::BothInsufficient),
        Some(Reason::ProcessSignalUnavailable),
        None,
    ] {
        for held in [false, true] {
            let mut cache = SnapshotCache::new(1024, Default::default());
            let value = Arc::new(7u8);
            let weak = Arc::downgrade(&value);
            assert!(cache.insert_shared_strict(vec![1], value.clone(), 20));
            let _held = held.then_some(value);
            let hit = cache.peek_best_prefix_in(&(), &[1, 2]).unwrap();
            let creates = Cell::new(0);
            let result = operations::restore_candidate::<Tracked, _>(
                &mut cache,
                hit,
                || {
                    creates.set(creates.get() + 1);
                    Err(match reason {
                        Some(reason) => denied(reason),
                        None => StateError::Invalid("bad allocation".into()),
                    })
                },
                |_, _| panic!("restore after failed allocation"),
            );
            assert_eq!(
                creates.get(),
                1,
                "never repeat the same failed allocation here"
            );
            let relieved = reason == Some(Reason::ProcessInsufficient) && !held;
            assert_eq!(cache.len(), usize::from(!relieved));
            assert_eq!(
                weak.upgrade().is_none(),
                relieved,
                "retry requires released payload ownership"
            );
            if relieved {
                assert!(result.unwrap().is_none());
            } else {
                let error = result.err().expect("original refusal must survive");
                assert_eq!(
                    error.status,
                    if reason.is_none() || reason == Some(Reason::ProcessSignalUnavailable) {
                        500
                    } else {
                        503
                    }
                );
            }
        }
    }
}

#[test]
fn restore_pressure_discards_candidate_before_cold_but_nonmemory_restore_errors_propagate() {
    for pressure in [true, false] {
        let mut cache = SnapshotCache::new(1024, Default::default());
        assert!(cache.insert_strict(vec![1], 7u8, 20));
        let hit = cache.peek_best_prefix_in(&(), &[1, 2]).unwrap();
        let drops = Rc::new(Cell::new(0));
        let creates = Cell::new(0);
        let result = operations::restore_candidate(
            &mut cache,
            hit,
            || {
                creates.set(creates.get() + 1);
                Ok(Tracked {
                    tokens: vec![],
                    drops: drops.clone(),
                })
            },
            |state, _| {
                state.tokens.push(99);
                Err(if pressure {
                    denied(Reason::ProcessInsufficient)
                } else {
                    StateError::Invalid("bad snapshot".into())
                })
            },
        );
        assert_eq!(creates.get(), 1);
        assert_eq!(drops.get(), 1, "possibly partial restore is released");
        assert_eq!(cache.len(), 1, "independent prefix remains correct");
        if pressure {
            assert!(result.unwrap().is_none());
        } else {
            assert!(result.err().unwrap().message.contains("bad snapshot"));
        }
    }
}

#[test]
fn successful_snapshot_restore_keeps_candidate_and_prefix() {
    let mut cache = SnapshotCache::new(1024, Default::default());
    assert!(cache.insert_strict(vec![1], 7u8, 20));
    let hit = cache.peek_best_prefix_in(&(), &[1, 2]).unwrap();
    let restored = operations::restore_candidate(
        &mut cache,
        hit,
        || Ok(0usize),
        |state, &value| {
            *state = value as usize;
            Ok(())
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(restored.0, 7);
    assert_eq!(restored.1.prefix_len, 1);
    assert_eq!(cache.len(), 1);
}

#[test]
fn full_snapshot_payload_key_and_logits_must_fit_before_capture() {
    // 64 payload + 12 token-key + 8 logit bytes.
    for budget in [0, 12, 20, 83, 84] {
        let mut cache = SnapshotCache::new(budget, Default::default());
        let measures = Cell::new(0);
        let captures = Cell::new(0);
        operations::capture_checkpoint(
            &mut cache,
            &[1, 2, 3],
            8,
            0,
            || capture_signals(Some(0)),
            || {
                measures.set(measures.get() + 1);
                Ok(64)
            },
            |bytes| {
                captures.set(captures.get() + 1);
                assert_eq!(bytes, 64);
                Ok(7u8)
            },
        );
        assert_eq!(measures.get(), usize::from(budget != 0));
        assert_eq!(captures.get(), usize::from(budget == 84));
        assert_eq!(cache.indexed_bytes(), if budget == 84 { 84 } else { 0 });
        if budget == 84 {
            operations::capture_checkpoint(
                &mut cache,
                &[1, 2, 3],
                8,
                0,
                || panic!("existing checkpoint need not be readmitted"),
                || panic!("existing checkpoint need not be remeasured"),
                |_| panic!("duplicate capture"),
            );
        }
    }
    let mut cache: SnapshotCache<u8> = SnapshotCache::new(u64::MAX, Default::default());
    operations::capture_checkpoint(
        &mut cache,
        &[1],
        u64::MAX,
        0,
        || panic!("overflow must precede admission"),
        || Ok(1),
        |_| panic!("overflow capture"),
    );
    assert_eq!(cache.len(), 0);
}

#[test]
fn capture_admission_checks_full_entry_plus_reserve_and_overflow() {
    let entry_peak = 64 + 12 + 8 + crate::serve::SNAPSHOT_CAPTURE_HEADROOM_BYTES;
    for (reserve, headroom, expected) in [
        (0, Some(entry_peak), true),
        (100, Some(entry_peak), false),
        (100, Some(entry_peak + 100), true),
        (100, Some(entry_peak + 99), false),
        (100, None, false),
        (100, Some(0), true),
        (u64::MAX, Some(0), false),
    ] {
        let mut cache = SnapshotCache::new(1 << 20, Default::default());
        assert!(cache.insert_strict(vec![9], 3u8, 5));
        let _held = cache.best_prefix(&[9, 10]).unwrap().1;
        let captures = Cell::new(0);
        operations::capture_checkpoint(
            &mut cache,
            &[1, 2, 3],
            8,
            reserve,
            || {
                assert_ne!(reserve, u64::MAX, "overflow precedes telemetry");
                capture_signals(headroom)
            },
            || Ok(64),
            |_| {
                captures.set(captures.get() + 1);
                Ok(7)
            },
        );
        assert_eq!(captures.get(), usize::from(expected));
        assert_eq!(cache.indexed_bytes(), 5 + if expected { 84 } else { 0 });
        assert_eq!(*cache.best_prefix(&[9, 10]).unwrap().1, 3);
    }
}

#[test]
fn capture_process_pressure_evicts_unpinned_payload_then_rechecks_once() {
    let required = 84 + 100 + crate::serve::SNAPSHOT_CAPTURE_HEADROOM_BYTES;
    for held in [false, true] {
        for relieved in [false, true] {
            let mut cache = SnapshotCache::new(1024, Default::default());
            let old = Arc::new(3u8);
            let weak = Arc::downgrade(&old);
            assert!(cache.insert_shared_strict(vec![9], old.clone(), 20));
            let _held = held.then_some(old);
            let observations = Cell::new(0);
            let captures = Cell::new(0);
            operations::capture_checkpoint(
                &mut cache,
                &[1, 2, 3],
                8,
                100,
                || {
                    observations.set(observations.get() + 1);
                    capture_signals(Some(if relieved && weak.upgrade().is_none() {
                        required
                    } else {
                        required - 1
                    }))
                },
                || Ok(64),
                |_| {
                    assert!(weak.upgrade().is_none(), "payload released before capture");
                    assert_eq!(observations.get(), 2, "admission rechecked before capture");
                    captures.set(captures.get() + 1);
                    Ok(7)
                },
            );
            let captured = !held && relieved;
            assert_eq!(observations.get(), if held { 1 } else { 2 });
            assert_eq!(captures.get(), usize::from(captured));
            assert_eq!(weak.upgrade().is_none(), !held);
            assert_eq!(cache.len(), usize::from(held || captured));
            assert_eq!(cache.entry_for(&[1, 2, 3]).is_some(), captured);
        }
    }
}

#[test]
fn capture_metal_telemetry_and_overflow_declines_do_not_evict_or_capture() {
    let required = 84 + crate::serve::SNAPSHOT_CAPTURE_HEADROOM_BYTES;
    let good = capture_signals(Some(0));
    for (reserve, signals) in [
        (
            0,
            MetalMemorySignals {
                host_budget: Some(qwen_llm::metal::HostMemoryBudget {
                    physical_bytes: 1024 << 30,
                    conservative_available_bytes: 1024 << 30,
                    external_gap_bytes: 0,
                    private_shadow_bytes: 0,
                }),
                recommended_max_bytes: required - 1,
                ..good
            },
        ),
        (
            0,
            MetalMemorySignals {
                host_budget: Some(qwen_llm::metal::HostMemoryBudget {
                    physical_bytes: 1024 << 30,
                    conservative_available_bytes: 1024 << 30,
                    external_gap_bytes: 0,
                    private_shadow_bytes: 0,
                }),
                recommended_max_bytes: 0,
                ..good
            },
        ),
        (
            0,
            MetalMemorySignals {
                host_budget: Some(qwen_llm::metal::HostMemoryBudget {
                    physical_bytes: 1024 << 30,
                    conservative_available_bytes: 1024 << 30,
                    external_gap_bytes: 0,
                    private_shadow_bytes: 0,
                }),
                recommended_max_bytes: 1,
                current_allocated_bytes: 2,
                ..good
            },
        ),
        (0, capture_signals(None)),
        (u64::MAX - 84, good),
        (u64::MAX, good),
    ] {
        let mut cache = SnapshotCache::new(1024, Default::default());
        assert!(cache.insert_strict(vec![9], 3u8, 5));
        let observations = Cell::new(0);
        operations::capture_checkpoint(
            &mut cache,
            &[1, 2, 3],
            8,
            reserve,
            || {
                observations.set(observations.get() + 1);
                signals
            },
            || Ok(64),
            |_| panic!("declined capture must not allocate"),
        );
        assert_eq!(observations.get(), usize::from(reserve != u64::MAX));
        assert_eq!(cache.indexed_bytes(), 5);
        assert!(cache.entry_for(&[9]).is_some());
    }
}

#[test]
fn optional_capture_decline_preserves_request_schedule_and_live_state() {
    struct Fixture<'a> {
        cache: SnapshotCache<u8>,
        reserve: &'a Cell<u64>,
        cuts: Vec<usize>,
        live: Option<Live<usize>>,
        process_pressure: bool,
    }
    impl Reuse<usize> for Fixture<'_> {
        fn take(
            &mut self,
            _: &[u32],
            _: usize,
            _: u64,
        ) -> Result<Option<Restored<usize>>, ServeError> {
            Ok(None)
        }
        fn checkpoint(&mut self, state: &usize, ids: &[u32], logits: &[f32], reserve: u64) {
            assert_eq!(*state, ids.len());
            assert_eq!(reserve, self.reserve.get());
            let logits_bytes = (logits.len() * size_of::<f32>()) as u64;
            let snapshot_peak = 64 + ids.len() as u64 * 4 + logits_bytes;
            assert!(
                snapshot_peak <= reserve,
                "snapshot alone fits available headroom"
            );
            self.cuts.push(*state);
            operations::capture_checkpoint(
                &mut self.cache,
                ids,
                logits_bytes,
                reserve,
                || {
                    let available = reserve + crate::serve::SNAPSHOT_CAPTURE_HEADROOM_BYTES;
                    if self.process_pressure {
                        capture_signals(Some(available))
                    } else {
                        MetalMemorySignals {
                            host_budget: Some(qwen_llm::metal::HostMemoryBudget {
                                physical_bytes: 1024 << 30,
                                conservative_available_bytes: 1024 << 30,
                                external_gap_bytes: 0,
                                private_shadow_bytes: 0,
                            }),
                            recommended_max_bytes: available,
                            current_allocated_bytes: 0,
                            process_limit_remaining_bytes: Some(0),
                        }
                    }
                },
                || Ok(64),
                |_| panic!("optional snapshot must leave shared headroom plus caller reserve"),
            );
        }
        fn commit(&mut self, live: Live<usize>) {
            self.live = Some(live);
        }
        fn capacity(&self, _: &usize, required: usize) -> usize {
            required
        }
    }
    struct Sink;
    impl GenerationSink for Sink {
        fn piece(&mut self, _: &[u8]) -> io::Result<()> {
            Ok(())
        }
        fn tick(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut settings = Settings::new(
        super::super::tokenizer_fixture::tokenizer(None),
        128,
        1,
        SamplingConfig::default(),
        256,
        "test".into(),
    )
    .unwrap();
    settings.headroom = || finite_headroom(1 << 30);
    let request = settings
        .profile()
        .parse(&serde_json::json!({
            "model":"test", "input":"abc", "max_output_tokens":1,
        }))
        .unwrap();
    for (boundaries, process_pressure) in [
        Some(super::super::super::http::PromptBoundaries {
            shared_prefix_end: Some(1),
            generation_header_start: Some(2),
        }),
        None,
    ]
    .into_iter()
    .flat_map(|boundaries| [false, true].map(|pressure| (boundaries, pressure)))
    {
        let reserve = Cell::new(0);
        let mut fixture = Fixture {
            cache: SnapshotCache::new(1 << 20, Default::default()),
            reserve: &reserve,
            cuts: vec![],
            live: None,
            process_pressure,
        };
        assert!(fixture.cache.insert_strict(vec![9], 3, 5));
        let mut chunks = vec![];
        let outcome = settings
            .execute(
                &request,
                "abc",
                boundaries,
                &mut Sink,
                |_, _, bytes| {
                    reserve.set(bytes);
                    Ok(())
                },
                |_| Ok(0),
                &mut fixture,
                |state, tokens, _, with_logits| {
                    *state += tokens.len();
                    chunks.push(*state);
                    assert_eq!(with_logits, *state == 3);
                    let mut logits = if with_logits { vec![0.; 257] } else { vec![] };
                    if with_logits {
                        logits[256] = 100.;
                    }
                    Ok(logits)
                },
                |_, _, _| panic!("EOS must not be consumed"),
                || Duration::ZERO,
            )
            .unwrap();
        assert_eq!(
            chunks,
            if boundaries.is_some() {
                vec![1, 2, 3]
            } else {
                vec![3]
            }
        );
        assert_eq!(
            fixture.cuts,
            if boundaries.is_some() {
                vec![1, 2]
            } else {
                vec![3]
            }
        );
        assert_eq!(outcome.usage.output_tokens, 1);
        assert_eq!(fixture.live.unwrap().tokens, [97, 98, 99]);
        assert_eq!(fixture.cache.len(), usize::from(!process_pressure));
        if !process_pressure {
            assert_eq!(*fixture.cache.best_prefix(&[9, 10]).unwrap().1, 3);
        }
    }
}
