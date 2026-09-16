use crate::metal::{KernelEncoder, MetalContext, MetalTimestampSampleBuffer};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Instant;

pub(crate) const TARGET_LAYER: u32 = 39;
const SAMPLE_CAPACITY: usize = 64;

#[derive(Clone, Debug)]
pub(crate) struct Record {
    pub name: &'static str,
    pub start: usize,
    pub end: usize,
    pub host_ms: f64,
    pub completed: bool,
}

pub(crate) struct Bank {
    samples: MetalTimestampSampleBuffer,
    records: RefCell<Vec<Record>>,
}

impl Bank {
    pub fn new(ctx: &MetalContext) -> Rc<Self> {
        Rc::new(Self {
            samples: ctx
                .timestamp_dispatch_sample_buffer(SAMPLE_CAPACITY)
                .unwrap(),
            records: RefCell::new(Vec::with_capacity(SAMPLE_CAPACITY / 2)),
        })
    }

    pub fn resolve(&self, ctx: &MetalContext) -> (Vec<Record>, Result<Vec<u64>, String>) {
        let records = self.records.borrow().clone();
        let raw = ctx
            .resolve_timestamp_samples(&self.samples, records.len() * 2)
            .map_err(|e| e.to_string());
        (records, raw)
    }
}

thread_local! {
    static BANK: RefCell<Option<Rc<Bank>>> = const { RefCell::new(None) };
    static LAYER: Cell<Option<u32>> = const { Cell::new(None) };
}

pub(crate) fn with_bank<R>(bank: &Rc<Bank>, f: impl FnOnce() -> R) -> R {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            BANK.with(|slot| slot.borrow_mut().take());
        }
    }
    BANK.with(|slot| {
        assert!(slot.borrow().is_none(), "child profiles cannot nest");
        bank.records.borrow_mut().clear();
        *slot.borrow_mut() = Some(bank.clone());
    });
    let _restore = Restore;
    f()
}

pub(crate) struct LayerScope {
    previous: Option<u32>,
    _tag: Option<crate::metal::DispatchCensusTagGuard>,
}

pub(crate) fn layer(layer: u32) -> LayerScope {
    LayerScope {
        previous: LAYER.with(|slot| slot.replace(Some(layer))),
        _tag: crate::metal::dispatch_census_tag_scope(|| format!("native.layer{layer}")),
    }
}

impl Drop for LayerScope {
    fn drop(&mut self) {
        LAYER.with(|slot| slot.set(self.previous));
    }
}

pub(crate) struct Span<'a> {
    bank: Rc<Bank>,
    encoder: &'a KernelEncoder,
    ordinal: usize,
    started: Instant,
}

fn begin<'a>(encoder: &'a KernelEncoder, name: &'static str) -> Option<Span<'a>> {
    BANK.with(|slot| {
        let bank = slot.borrow().as_ref()?.clone();
        let ordinal = bank.records.borrow().len();
        assert!((ordinal + 1) * 2 <= SAMPLE_CAPACITY);
        bank.records.borrow_mut().push(Record {
            name,
            start: ordinal * 2,
            end: ordinal * 2 + 1,
            host_ms: 0.0,
            completed: false,
        });
        encoder.sample_counters(&bank.samples, ordinal * 2, true);
        Some(Span {
            bank,
            encoder,
            ordinal,
            started: Instant::now(),
        })
    })
}

pub(crate) fn command(encoder: &KernelEncoder) -> Option<Span<'_>> {
    begin(encoder, "command")
}

pub(crate) fn span<'a>(encoder: &'a KernelEncoder, name: &'static str) -> Option<Span<'a>> {
    if LAYER.with(|slot| slot.get()) != Some(TARGET_LAYER) {
        return None;
    }
    begin(encoder, name)
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        let host_ms = self.started.elapsed().as_secs_f64() * 1e3;
        self.encoder
            .sample_counters(&self.bank.samples, self.ordinal * 2 + 1, true);
        let mut records = self.bank.records.borrow_mut();
        records[self.ordinal].host_ms = host_ms;
        records[self.ordinal].completed = true;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Coverage {
    pub envelope: u64,
    pub sum: u64,
    pub union: u64,
    pub overlap: u64,
    pub gaps: u64,
}

pub(crate) fn coverage(envelope: (u64, u64), pairs: &[(u64, u64)]) -> Option<Coverage> {
    let width = envelope.1.checked_sub(envelope.0).filter(|n| *n > 0)?;
    let mut ordered = pairs.to_vec();
    ordered.sort_unstable();
    let (mut sum, mut union, mut end) = (0u64, 0u64, envelope.0);
    for (a, b) in ordered {
        if a < envelope.0 || b > envelope.1 || b <= a {
            return None;
        }
        sum = sum.checked_add(b - a)?;
        if b > end {
            union = union.checked_add(b - a.max(end))?;
        }
        end = end.max(b);
    }
    Some(Coverage {
        envelope: width,
        sum,
        union,
        overlap: sum - union,
        gaps: width - union,
    })
}

#[test]
fn child_interval_coverage_handles_overlap_and_gaps() {
    assert_eq!(
        coverage((10, 30), &[(20, 25), (12, 22)]),
        Some(Coverage {
            envelope: 20,
            sum: 15,
            union: 13,
            overlap: 2,
            gaps: 7
        })
    );
    assert_eq!(
        coverage((10, 30), &[(10, 20), (20, 30)]),
        Some(Coverage {
            envelope: 20,
            sum: 20,
            union: 20,
            overlap: 0,
            gaps: 0
        })
    );
    assert_eq!(coverage((10, 30), &[(12, 12)]), None);
    assert_eq!(coverage((10, 30), &[(9, 12)]), None);
    assert_eq!(coverage((10, 30), &[(29, 31)]), None);
    assert_eq!(coverage((30, 10), &[]), None);
}
