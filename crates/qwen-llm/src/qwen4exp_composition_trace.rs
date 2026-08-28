use crate::metal::{KernelEncoder, MetalContext, MetalError, MetalTensor, encode_copy_offset_f32};
use crate::tensor::GgmlType;
use objc2_metal::{MTLDevice, MTLResource};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Qwen4ExpCompositionTracePhase {
    LayerInput,
    AttentionInput,
    MixerOutput,
    AttentionOutput,
    FfnInput,
    MoeOutput,
    LayerOutput,
    PleOutput,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Qwen4ExpCompositionTraceStage {
    pub layer: u32,
    pub phase: Qwen4ExpCompositionTracePhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen4ExpCompositionTraceRecord {
    pub stage: Qwen4ExpCompositionTraceStage,
    pub width: usize,
    pub ordinal: usize,
}

#[derive(Clone)]
pub(crate) struct Qwen4ExpCompositionTraceBanks {
    pub target_position: usize,
    pub maximum_width: usize,
    pub maximum_records: usize,
    pub values: MetalTensor,
}

impl Qwen4ExpCompositionTraceBanks {
    pub(crate) fn new(
        ctx: &MetalContext,
        target_position: usize,
        maximum_width: usize,
        maximum_records: usize,
    ) -> Result<Self, MetalError> {
        assert!(maximum_width > 0);
        assert!(maximum_records > 0);
        let elements = maximum_width
            .checked_mul(maximum_records)
            .expect("composition trace allocation overflow");
        assert!(u32::try_from(elements).is_ok());
        Ok(Self {
            target_position,
            maximum_width,
            maximum_records,
            values: MetalTensor::zeros_f32(
                ctx,
                vec![maximum_width as u64, maximum_records as u64],
            )?,
        })
    }

    pub(crate) fn record_values(&self, record: Qwen4ExpCompositionTraceRecord) -> MetalTensor {
        assert!(record.ordinal < self.maximum_records);
        assert!(record.width <= self.maximum_width);
        self.values.view_subrange(
            (record.ordinal * self.maximum_width) as u64,
            vec![record.width as u64],
        )
    }
}

#[derive(Clone)]
struct Qwen4ExpCompositionTraceBinding {
    banks: Qwen4ExpCompositionTraceBanks,
    records: std::rc::Rc<std::cell::RefCell<Vec<Qwen4ExpCompositionTraceRecord>>>,
}

thread_local! {
    static QWEN4EXP_DIAGNOSTIC_EXECUTION_RANGE: std::cell::Cell<Option<(usize, usize)>> = const {
        std::cell::Cell::new(None)
    };
    static QWEN4EXP_COMPOSITION_TRACE: std::cell::RefCell<Option<Qwen4ExpCompositionTraceBinding>> = const {
        std::cell::RefCell::new(None)
    };
}

pub(crate) fn with_qwen4exp_composition_trace<R>(
    banks: &Qwen4ExpCompositionTraceBanks,
    f: impl FnOnce() -> R,
) -> (R, Vec<Qwen4ExpCompositionTraceRecord>) {
    struct RestoreCapture(Option<Qwen4ExpCompositionTraceBinding>);

    impl Drop for RestoreCapture {
        fn drop(&mut self) {
            QWEN4EXP_COMPOSITION_TRACE.with(|slot| {
                *slot.borrow_mut() = self.0.take();
            });
        }
    }

    QWEN4EXP_DIAGNOSTIC_EXECUTION_RANGE.with(|slot| {
        assert!(
            slot.get().is_none(),
            "composition trace cannot begin inside an execution range"
        );
    });
    QWEN4EXP_COMPOSITION_TRACE.with(|slot| {
        assert!(slot.borrow().is_none(), "composition traces cannot nest");
    });
    let records = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let previous = QWEN4EXP_COMPOSITION_TRACE.with(|slot| {
        slot.borrow_mut().replace(Qwen4ExpCompositionTraceBinding {
            banks: banks.clone(),
            records: records.clone(),
        })
    });
    debug_assert!(previous.is_none());
    let _restore = RestoreCapture(previous);
    let result = f();
    let records = records.borrow().clone();
    (result, records)
}

pub(crate) fn with_qwen4exp_diagnostic_execution_range<R>(
    start_position: usize,
    rows: usize,
    f: impl FnOnce() -> R,
) -> R {
    assert!(rows > 0);
    start_position
        .checked_add(rows)
        .expect("composition trace range overflow");

    struct RestoreRange(Option<(usize, usize)>);

    impl Drop for RestoreRange {
        fn drop(&mut self) {
            QWEN4EXP_DIAGNOSTIC_EXECUTION_RANGE.with(|slot| slot.set(self.0));
        }
    }

    QWEN4EXP_DIAGNOSTIC_EXECUTION_RANGE.with(|slot| {
        assert!(
            slot.get().is_none(),
            "diagnostic execution ranges cannot nest"
        );
    });
    let previous = QWEN4EXP_DIAGNOSTIC_EXECUTION_RANGE.with(|slot| {
        let previous = slot.get();
        slot.set(Some((start_position, rows)));
        previous
    });
    debug_assert!(previous.is_none());
    let _restore = RestoreRange(previous);
    f()
}

pub(crate) fn qwen4exp_diagnostic_execution_range() -> Option<(usize, usize)> {
    QWEN4EXP_DIAGNOSTIC_EXECUTION_RANGE.with(|slot| slot.get())
}

pub(crate) fn qwen4exp_composition_trace_active() -> bool {
    QWEN4EXP_COMPOSITION_TRACE.with(|slot| slot.borrow().is_some())
}

fn trace_error(detail: impl Into<String>) -> MetalError {
    MetalError::BadShape {
        kernel: "qwen4exp_composition_trace",
        detail: detail.into(),
    }
}

pub(crate) fn encode_qwen4exp_composition_trace(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    stage: Qwen4ExpCompositionTraceStage,
    source: &MetalTensor,
    row_width: usize,
) -> Result<(), MetalError> {
    QWEN4EXP_COMPOSITION_TRACE.with(|slot| {
        let binding = slot.borrow();
        let Some(binding) = binding.as_ref() else {
            return Ok(());
        };
        let (start_position, rows) = QWEN4EXP_DIAGNOSTIC_EXECUTION_RANGE
            .with(|slot| slot.get())
            .ok_or_else(|| trace_error("active capture has no execution range"))?;
        let end_position = start_position
            .checked_add(rows)
            .ok_or_else(|| trace_error("execution range overflow"))?;
        if binding.banks.target_position < start_position
            || binding.banks.target_position >= end_position
        {
            return Ok(());
        }
        let expected_elements = row_width
            .checked_mul(rows)
            .ok_or_else(|| trace_error("source element count overflow"))?;
        if source.dtype != GgmlType::F32 || source.n_elements() as usize != expected_elements {
            return Err(trace_error(format!(
                "source must be F32 with {expected_elements} elements, got {:?} {}",
                source.dtype,
                source.n_elements()
            )));
        }
        if source.buffer.device().registryID() != ctx.device.registryID() {
            return Err(trace_error("source belongs to a different Metal device"));
        }
        if row_width > binding.banks.maximum_width {
            return Err(trace_error(format!(
                "row width {row_width} exceeds capture width {}",
                binding.banks.maximum_width
            )));
        }
        let mut records = binding.records.borrow_mut();
        if records.iter().any(|record| record.stage == stage) {
            return Err(trace_error(format!("stage {stage:?} was captured twice")));
        }
        let ordinal = records.len();
        if ordinal >= binding.banks.maximum_records {
            return Err(trace_error(format!(
                "record {ordinal} exceeds capture capacity {}",
                binding.banks.maximum_records
            )));
        }
        let record = Qwen4ExpCompositionTraceRecord {
            stage,
            width: row_width,
            ordinal,
        };
        let destination = binding.banks.record_values(record);
        let source_row = binding.banks.target_position - start_position;
        let _tag = crate::metal::dispatch_census_tag_scope(|| {
            format!(
                "qwen4exp.composition_trace.layer{}.phase{:?}",
                stage.layer, stage.phase
            )
        });
        encode_copy_offset_f32(
            ctx,
            enc,
            source,
            source_row * row_width,
            &destination,
            row_width,
        )?;
        records.push(record);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCommandQueue;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    fn context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(ctx) => Some(ctx),
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => None,
            Err(error) => panic!("Metal initialization failed: {error}"),
        }
    }

    #[test]
    fn diagnostic_scopes_reject_nesting_without_losing_outer_bindings() {
        assert_eq!(qwen4exp_diagnostic_execution_range(), None);
        with_qwen4exp_diagnostic_execution_range(17, 3, || {
            let nested = catch_unwind(AssertUnwindSafe(|| {
                with_qwen4exp_diagnostic_execution_range(20, 1, || ());
            }));
            assert!(nested.is_err());
            assert_eq!(qwen4exp_diagnostic_execution_range(), Some((17, 3)));
        });
        assert_eq!(qwen4exp_diagnostic_execution_range(), None);

        let Some(ctx) = context() else { return };
        let banks = Qwen4ExpCompositionTraceBanks::new(&ctx, 17, 8, 2).unwrap();
        let outer = catch_unwind(AssertUnwindSafe(|| {
            with_qwen4exp_composition_trace(&banks, || {
                let nested = catch_unwind(AssertUnwindSafe(|| {
                    with_qwen4exp_composition_trace(&banks, || ());
                }));
                assert!(nested.is_err());
                assert!(qwen4exp_composition_trace_active());
                panic!("exercise outer unwind restoration");
            });
        }));
        assert!(outer.is_err());
        assert!(!qwen4exp_composition_trace_active());
    }

    #[test]
    fn inactive_composition_trace_is_a_no_op() {
        let Some(ctx) = context() else { return };
        let source = MetalTensor::zeros_i32(&ctx, vec![1]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        encode_qwen4exp_composition_trace(
            &ctx,
            &encoder,
            Qwen4ExpCompositionTraceStage {
                layer: 0,
                phase: Qwen4ExpCompositionTracePhase::LayerInput,
            },
            &source,
            usize::MAX,
        )
        .unwrap();
        assert!(crate::metal::dispatch_census_take().is_empty());
        encoder.end();
    }
}
