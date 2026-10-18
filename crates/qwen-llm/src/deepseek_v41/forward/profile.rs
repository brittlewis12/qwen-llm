//! Opt-in synchronous forward measurements. CPU intervals are inclusive wall
//! times, not additive CPU work or claims about CPU/GPU overlap.
use serde::Serialize;
use std::{cell::RefCell, marker::PhantomData, rc::Rc, time::Instant};

thread_local! {
    static ACTIVE: RefCell<Option<Vec<Operation>>> = const { RefCell::new(None) };
}

#[derive(Debug, Serialize)]
pub struct Operation {
    pub operation: &'static str,
    /// [input, output, experts, selected_slots]; dense uses [input, output, 0, 1].
    pub dimensions: [usize; 4],
    pub count: usize,
    pub completed_submits: usize,
    pub invalid_gpu_timestamps: usize,
    /// Sum of valid completed-command durations only; None if none are valid.
    pub gpu_ms: Option<f64>,
    pub encode_ms: f64,
    pub commit_wait_ms: f64,
    pub wall_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub token_wall_ms: f64,
    pub operations: Vec<Operation>,
}

/// Thread-bound scope; dropping on errors/unwind clears profiling. No nesting.
pub struct Scope {
    start: Instant,
    _thread: PhantomData<Rc<()>>,
}
impl Scope {
    pub fn begin() -> Result<Self, String> {
        ACTIVE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_some() {
                return Err("nested forward profiling scope".into());
            }
            *slot = Some(Vec::new());
            Ok(Self {
                start: Instant::now(),
                _thread: PhantomData,
            })
        })
    }
    pub fn finish(self) -> Report {
        Report {
            token_wall_ms: self.start.elapsed().as_secs_f64() * 1000.0,
            operations: ACTIVE.with(|slot| slot.borrow_mut().take().unwrap()),
        }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|slot| *slot.borrow_mut() = None);
    }
}

pub(super) fn start() -> Option<Instant> {
    ACTIVE.with(|slot| slot.borrow().is_some().then(Instant::now))
}
pub(super) struct Wall(pub &'static str, pub [usize; 4], pub Option<Instant>);
impl Drop for Wall {
    fn drop(&mut self) {
        if let Some(start) = self.2 {
            record(self.0, self.1, |op| {
                op.count += 1;
                op.wall_ms += start.elapsed().as_secs_f64() * 1000.0;
            });
        }
    }
}
fn record(name: &'static str, dimensions: [usize; 4], update: impl FnOnce(&mut Operation)) {
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(ops) = slot.as_mut() else { return };
        let index = ops
            .iter()
            .position(|op| op.operation == name && op.dimensions == dimensions)
            .unwrap_or_else(|| {
                ops.push(Operation {
                    operation: name,
                    dimensions,
                    count: 0,
                    completed_submits: 0,
                    invalid_gpu_timestamps: 0,
                    gpu_ms: None,
                    encode_ms: 0.0,
                    commit_wait_ms: 0.0,
                    wall_ms: 0.0,
                });
                ops.len() - 1
            });
        update(&mut ops[index]);
    });
}
pub(super) fn submitted(
    name: &'static str,
    dims: [usize; 4],
    encode_ms: f64,
    wait_ms: f64,
    completed: bool,
    start: f64,
    end: f64,
) {
    record(name, dims, |op| {
        op.encode_ms += encode_ms;
        op.commit_wait_ms += wait_ms;
        if completed {
            op.completed_submits += 1;
            if start.is_finite() && end.is_finite() && start > 0.0 && end > start {
                *op.gpu_ms.get_or_insert(0.0) += (end - start) * 1000.0;
            } else {
                op.invalid_gpu_timestamps += 1;
            }
        }
    });
}
