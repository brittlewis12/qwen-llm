use super::*;
use qwen_llm::deepseek_v41::forward::StateError;
use qwen_llm::metal::{MemoryAdmissionDenied, MetalMemoryAdmissionReason as Reason};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

fn denied(reason: Reason) -> StateError {
    StateError::MemoryAdmissionDenied(MemoryAdmissionDenied {
        reason,
        required_bytes: Some(4096),
        signals: MetalMemorySignals {
            recommended_max_bytes: 1024,
            current_allocated_bytes: 0,
            process_limit_remaining_bytes: Some(1024),
        },
        working_set_headroom_bytes: Some(1024),
    })
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
impl Reuse<Tracked> for GrowthFixture {
    fn take(
        &mut self,
        ids: &[u32],
        required: usize,
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
    fn checkpoint(&mut self, _: &Tracked, _: &[u32], _: &[f32]) {}
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
    settings.headroom = || Some(0);
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
            |_, _| {
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
        || Ok(1),
        |_| panic!("overflow capture"),
    );
    assert_eq!(cache.len(), 0);
}
