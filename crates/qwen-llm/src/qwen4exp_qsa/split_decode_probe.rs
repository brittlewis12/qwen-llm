//! Test-only observation of split decode attention. A witness can record
//! which steps took the split path and, for A/B arms, force the incumbent
//! single-pass kernel on a workspace that would otherwise split.
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Record {
    pub layer: u32,
    pub position: usize,
    pub ids: usize,
    pub split: bool,
}

struct Binding {
    split: bool,
    records: Rc<RefCell<Vec<Record>>>,
}

thread_local! {
    static BINDING: RefCell<Option<Binding>> = const { RefCell::new(None) };
}

/// Run `f` with split decode forced to `split` on every eligible step, and
/// return the eligible steps observed.
pub(crate) fn with_probe<R>(split: bool, f: impl FnOnce() -> R) -> (R, Vec<Record>) {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            BINDING.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }
    let records = Rc::new(RefCell::new(Vec::new()));
    BINDING.with(|slot| {
        assert!(slot.borrow().is_none(), "split probes cannot nest");
        *slot.borrow_mut() = Some(Binding {
            split,
            records: records.clone(),
        });
    });
    let _restore = Restore;
    let value = f();
    let records = records.borrow().clone();
    (value, records)
}

/// Called on every eligible step; returns whether the step should split.
pub(super) fn observe(position: usize, ids: usize) -> bool {
    BINDING.with(|slot| {
        let binding = slot.borrow();
        let Some(binding) = binding.as_ref() else {
            return true;
        };
        binding.records.borrow_mut().push(Record {
            layer: super::QWEN4EXP_QSA_CAPTURE_LAYER
                .with(|layer| layer.get().expect("native layer scope")),
            position,
            ids,
            split: binding.split,
        });
        binding.split
    })
}
