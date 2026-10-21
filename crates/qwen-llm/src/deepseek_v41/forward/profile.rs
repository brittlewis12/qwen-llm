//! Opt-in synchronous forward measurements. CPU intervals are inclusive wall
//! times, not additive CPU work or claims about CPU/GPU overlap.
use serde::Serialize;
use std::{cell::RefCell, marker::PhantomData, rc::Rc, time::Instant};

pub mod decode;

#[derive(Default)]
struct Active {
    operations: Vec<Operation>,
    indices: std::collections::BTreeMap<(&'static str, [usize; 4]), usize>,
    decode_intervals: Vec<decode::Interval>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
    static STAGES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
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
    pub process_pageins: Option<u64>,
    pub process_disk_read_bytes: Option<u64>,
    pub operations: Vec<Operation>,
    pub decode_intervals: Vec<decode::Interval>,
}

/// Thread-bound scope; dropping on errors/unwind clears profiling. No nesting.
pub struct Scope {
    start: Instant,
    before: Option<crate::pid_metrics::PidSnapshot>,
    _thread: PhantomData<Rc<()>>,
}
impl Scope {
    pub fn begin_stages() -> Result<Self, String> {
        let scope = Self::begin()?;
        STAGES.with(|enabled| enabled.set(true));
        Ok(scope)
    }
    pub fn begin() -> Result<Self, String> {
        ACTIVE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_some() {
                return Err("nested forward profiling scope".into());
            }
            *slot = Some(Active::default());
            Ok(Self {
                before: crate::pid_metrics::PidSnapshot::now().ok(),
                start: Instant::now(),
                _thread: PhantomData,
            })
        })
    }
    pub fn finish(self) -> Report {
        let token_wall_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        let after = crate::pid_metrics::PidSnapshot::now().ok();
        let delta = self.before.as_ref().zip(after.as_ref());
        let active = ACTIVE.with(|slot| slot.borrow_mut().take().unwrap());
        Report {
            token_wall_ms,
            process_pageins: delta
                .and_then(|(before, after)| after.pageins.checked_sub(before.pageins)),
            process_disk_read_bytes: delta.and_then(|(before, after)| {
                after.diskio_bytesread.checked_sub(before.diskio_bytesread)
            }),
            operations: active.operations,
            decode_intervals: active.decode_intervals,
        }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|slot| *slot.borrow_mut() = None);
        STAGES.with(|enabled| enabled.set(false));
    }
}

pub(super) fn stages() -> bool {
    STAGES.with(|enabled| enabled.get())
}

pub(super) fn gpu_interval(name: &'static str, rows: usize, start: u64, end: u64) {
    gpu_interval_dimensions(name, [rows, 0, 0, 0], start, end);
}

fn gpu_interval_dimensions(name: &'static str, dims: [usize; 4], start: u64, end: u64) {
    record(name, dims, |op| {
        op.count += 1;
        if start > 0 && end > start && end != u64::MAX {
            *op.gpu_ms.get_or_insert(0.0) += (end - start) as f64 / 1e6;
        } else {
            op.invalid_gpu_timestamps += 1;
        }
    });
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
        let Some(active) = slot.as_mut() else { return };
        let index = *active.indices.entry((name, dimensions)).or_insert_with(|| {
            active.operations.push(Operation {
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
            active.operations.len() - 1
        });
        update(&mut active.operations[index]);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_records_preserve_order_and_merge_without_position_scans() {
        let scope = Scope::begin_stages().unwrap();
        for position in 8192..8224 {
            for layer in 0..40 {
                gpu_interval_dimensions("fixture", [position, layer, 1, 0], 100, 1100);
            }
        }
        gpu_interval_dimensions("fixture", [8192, 0, 1, 0], 1100, 2100);
        gpu_interval_dimensions("invalid", [0; 4], 0, 0);
        let report = scope.finish();
        assert_eq!(report.operations.len(), 1281);
        assert_eq!(report.operations[0].dimensions, [8192, 0, 1, 0]);
        assert_eq!(report.operations[0].count, 2);
        assert_eq!(report.operations[0].gpu_ms, Some(0.002));
        assert_eq!(report.operations[1280].gpu_ms, None);
        assert_eq!(report.operations[1280].invalid_gpu_timestamps, 1);
        assert!(!stages());
    }

    #[test]
    fn scope_nesting_failure_and_unwind_clear_stage_state() {
        let unwind = std::panic::catch_unwind(|| {
            let _scope = Scope::begin_stages().unwrap();
            assert!(Scope::begin().is_err());
            assert!(stages());
            panic!("fixture unwind");
        });
        assert!(unwind.is_err());
        assert!(!stages());
        let report = Scope::begin().unwrap().finish();
        assert!(report.operations.is_empty());
        assert!(report.decode_intervals.is_empty());
    }
}
