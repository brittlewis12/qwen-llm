//! Opt-in synchronous forward measurements. CPU intervals are inclusive wall
//! times, not additive CPU work or claims about CPU/GPU overlap.
use serde::Serialize;
use std::{cell::RefCell, marker::PhantomData, rc::Rc, time::Instant};

/// Unsampled pair diagnostics: <=8192 rows, <=256 layers, one profile at a time.
/// The producer inventory is bounded by chunks and decoder layers, not runs.
pub const PAIRED_OPERATION_CAPACITY: usize = 4096;

thread_local! {
    static ACTIVE: RefCell<Option<Vec<Operation>>> = const { RefCell::new(None) };
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
            *slot = Some(Vec::new());
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
        Report {
            token_wall_ms,
            process_pageins: delta
                .map(|(before, after)| after.pageins.saturating_sub(before.pageins)),
            process_disk_read_bytes: delta.map(|(before, after)| {
                after
                    .diskio_bytesread
                    .saturating_sub(before.diskio_bytesread)
            }),
            operations: ACTIVE.with(|slot| slot.borrow_mut().take().unwrap()),
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

/// Per-layer sampled GPU interval; the layer lands in `dimensions[1]`.
pub(super) fn gpu_interval_layer(
    name: &'static str,
    rows: usize,
    layer: usize,
    start: u64,
    end: u64,
) {
    record(name, [rows, layer, 0, 0], |op| {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_production_command_inventory_fits_profile_capacity_without_metal() {
        use super::super::{
            Config,
            ced_plan::{CedPlan, CedRequest, chunk_views},
            prefill,
        };
        use crate::deepseek_v41::{engram::HashConfig, gguf::Schedule};
        let chunk = prefill::DEFAULT_CHUNK_ROWS;
        for layers in [40, 256] {
            let mut schedule = Schedule::default();
            schedule.layers = layers;
            schedule.compression_ratios.resize(layers, 1);
            for window in [1, Config::default().window, 8192] {
                let config = Config {
                    window,
                    ..Config::default()
                };
                for n in [2, 128, 2048, 2049, 8191, 8192] {
                    let plan = CedPlan::shrinking(
                        &config,
                        &schedule,
                        &HashConfig::default(),
                        &CedRequest::cold(n, n),
                    )
                    .unwrap();
                    let scope = Scope::begin().unwrap();
                    let mut commands = 0;
                    let mut emit = |role, dimensions| {
                        submitted(role, dimensions, 1.0, 2.0, true, 1.0, 1.001);
                        commands += 1;
                    };
                    // Match the production encoder, seed and decoder submissions,
                    // including the genuine Full scalar tail during compaction.
                    for view in chunk_views(0..n, chunk).unwrap() {
                        let r = view.absolute_rows;
                        if r.len() > 1 {
                            drop(Wall("ced_encoder_host_prepare", [0; 4], start()));
                            drop(Wall("ced_encoder_engram_prepare", [0; 4], start()));
                            drop(Wall("ced_block_rows_allocate", [0; 4], start()));
                        }
                        emit(
                            if r.len() == 1 {
                                "ced_encoder_scalar"
                            } else {
                                "ced_encoder_rows"
                            },
                            [r.start, r.len(), plan.publication_layer, 0],
                        );
                    }
                    let scalar_tail =
                        prefill::original_scalar_tail(n, super::super::prefill::DEFAULT_CHUNK_ROWS);
                    for layer in &plan.decoder {
                        let mut fuse = layer.fuse_seed_with_packed_output(n);
                        if !fuse {
                            for view in chunk_views(layer.seed_rows.clone(), chunk).unwrap() {
                                let r = view.absolute_rows;
                                emit("ced_local_seed", [layer.layer, r.start, r.len(), 0]);
                            }
                        }
                        for view in chunk_views(
                            layer.output_rows.start..scalar_tail.unwrap_or(layer.output_rows.end),
                            chunk,
                        )
                        .unwrap()
                        {
                            let r = view.absolute_rows;
                            drop(Wall("ced_block_rows_allocate", [0; 4], start()));
                            emit(
                                if fuse {
                                    "ced_seed_decoder_rows"
                                } else {
                                    "ced_decoder_rows"
                                },
                                [
                                    layer.layer,
                                    r.start,
                                    r.len(),
                                    if fuse { layer.seed_rows.len() } else { 0 },
                                ],
                            );
                            fuse = false;
                        }
                        if let Some(position) = scalar_tail {
                            emit("ced_decoder_scalar", [layer.layer, position, 1, 0]);
                        }
                    }
                    drop(emit);
                    let report = scope.finish();
                    assert_eq!(
                        commands,
                        super::super::ced::qualification::production_command_count(&plan).unwrap()
                    );
                    assert_eq!(
                        report
                            .operations
                            .iter()
                            .map(|op| op.completed_submits)
                            .sum::<usize>(),
                        commands
                    );
                    assert_eq!(
                        report.operations.len(),
                        commands + if n > 1 { 3 } else { 0 }
                    );
                    let producer_bound =
                        n.div_ceil(chunk) + layers * (2 * n.div_ceil(chunk) + 1) + 4;
                    assert!(report.operations.len() <= producer_bound);
                    assert!(producer_bound <= PAIRED_OPERATION_CAPACITY);
                    assert!(report.operations.capacity() <= PAIRED_OPERATION_CAPACITY);

                    let scope = Scope::begin().unwrap();
                    for view in chunk_views(0..n, chunk).unwrap() {
                        let r = view.absolute_rows;
                        if r.len() == 1 {
                            submitted(
                                "gpu_step",
                                [r.start, 0, layers, 0],
                                1.0,
                                2.0,
                                true,
                                1.0,
                                1.001,
                            );
                            drop(Wall("gpu_step", [r.start, 0, layers, 0], start()));
                        } else {
                            drop(Wall("prefill_host_prepare", [0; 4], start()));
                            drop(Wall("prefill_engram_prepare", [0; 4], start()));
                            submitted(
                                "prefill_chunk_rows",
                                [r.start, layers, r.len(), 0],
                                1.0,
                                2.0,
                                true,
                                1.0,
                                1.001,
                            );
                        }
                    }
                    let report = scope.finish();
                    assert_eq!(
                        report
                            .operations
                            .iter()
                            .map(|op| op.completed_submits)
                            .sum::<usize>(),
                        n.div_ceil(chunk)
                    );
                    assert_eq!(report.operations.len(), n.div_ceil(chunk) + 2);
                    assert!(report.operations.capacity() <= PAIRED_OPERATION_CAPACITY);
                }
            }
        }
    }

    #[test]
    fn unsampled_scope_is_cleared_on_error_and_unwind() {
        let run = || -> Result<(), String> {
            let _scope = Scope::begin()?;
            assert!(!stages());
            assert!(Scope::begin().is_err());
            Err("profile error fixture".into())
        };
        assert!(run().is_err());
        let result = std::panic::catch_unwind(|| {
            let _scope = Scope::begin().unwrap();
            panic!("profile unwind fixture");
        });
        assert!(result.is_err());
        let scope = Scope::begin().unwrap();
        assert!(!stages());
        assert!(scope.finish().operations.is_empty());
    }
}
