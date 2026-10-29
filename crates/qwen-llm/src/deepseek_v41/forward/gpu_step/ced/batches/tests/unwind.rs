use super::*;
use incumbent::{Action, Scope as Fault};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Submitted(usize),
    Terminal(usize),
    Release(&'static str, Option<usize>),
    SyntheticCancel,
    Checkpoint(&'static str, usize),
}

struct Observation {
    events: Vec<Event>,
    overflow: bool,
}

thread_local! {
    static TRACE: RefCell<Option<Observation>> = const { RefCell::new(None) };
    static STATUS: Cell<bool> = const { Cell::new(false) };
}

// Called from ownership destructors: no allocation, assertion or borrow panic.
fn record(event: Event) {
    let _ = TRACE.try_with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut()
            && let Some(trace) = slot.as_mut()
        {
            if trace.events.len() < trace.events.capacity() {
                trace.events.push(event);
            } else {
                trace.overflow = true;
            }
        }
    });
}

pub(in super::super::super) fn submitted(role: &str, first: usize) {
    if role == "ced_decoder_batch" {
        record(Event::Submitted(first));
    }
}

pub(in super::super::super) fn terminal(role: &str, first: usize) {
    if role == "ced_decoder_batch" {
        record(Event::Terminal(first));
    }
}

pub(in super::super::super) struct Release(&'static str, Option<usize>);

impl Release {
    pub(in super::super::super) fn new(resource: &'static str, first: Option<usize>) -> Self {
        Self(resource, first)
    }
}

impl Drop for Release {
    fn drop(&mut self) {
        record(Event::Release(self.0, self.1));
    }
}

struct Trace;

impl Trace {
    fn begin(status: bool) -> Self {
        TRACE.with(|slot| {
            assert!(slot.borrow().is_none());
            slot.replace(Some(Observation {
                events: Vec::with_capacity(64),
                overflow: false,
            }));
        });
        STATUS.with(|slot| assert!(!slot.replace(status)));
        Self
    }

    fn finish(&self, expected: usize, synthetic_cancel: bool, seam: Option<&'static str>) {
        let trace = TRACE.with(|slot| slot.take().unwrap());
        assert!(!trace.overflow);
        assert_order(&trace.events, expected);
        assert_eq!(
            trace
                .events
                .iter()
                .filter(|&&event| event == Event::SyntheticCancel)
                .count(),
            usize::from(synthetic_cancel)
        );
        if expected == 2 {
            assert_eq!(
                trace
                    .events
                    .iter()
                    .copied()
                    .filter(|event| matches!(event, Event::Submitted(_) | Event::Terminal(_)))
                    .take(2)
                    .collect::<Vec<_>>(),
                [Event::Submitted(2), Event::Submitted(4)],
                "successor committed before older terminal wait"
            );
        }
        if let Some(seam) = seam.filter(|role| role.starts_with("ced_pipeline_")) {
            let first = match seam {
                "ced_pipeline_first_encode" | "ced_pipeline_popped" | "ced_pipeline_waited" => 2,
                _ => 4,
            };
            assert_eq!(
                trace
                    .events
                    .iter()
                    .filter(|event| matches!(event, Event::Checkpoint(role, _) if *role == seam))
                    .copied()
                    .collect::<Vec<_>>(),
                [Event::Checkpoint(seam, first)]
            );
        }
        assert!(!STATUS.with(Cell::get), "status fault must be consumed");
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        TRACE.with(|slot| slot.take());
        STATUS.with(|slot| slot.set(false));
    }
}

fn assert_order(events: &[Event], expected: usize) {
    let mut submitted = std::collections::BTreeSet::new();
    let mut waited = std::collections::BTreeSet::new();
    let mut released = std::collections::BTreeSet::new();
    for event in events {
        match *event {
            Event::Submitted(first) => assert!(submitted.insert(first), "{events:?}"),
            Event::Terminal(first) => {
                assert!(submitted.contains(&first), "{events:?}");
                assert!(waited.insert(first), "no duplicate waits: {events:?}");
            }
            Event::Release(resource, first) => {
                if let Some(first) = first {
                    assert!(waited.contains(&first), "own status views: {events:?}");
                } else {
                    assert_eq!(submitted, waited, "{resource}: {events:?}");
                }
                assert!(released.insert((resource, first)), "{events:?}");
            }
            Event::SyntheticCancel | Event::Checkpoint(_, _) => {}
        }
    }
    assert_eq!(submitted.len(), expected, "{events:?}");
    assert_eq!(submitted, waited, "{events:?}");
    for resource in ["seed", "ledger", "pool/storage"] {
        assert!(released.contains(&(resource, None)), "{events:?}");
    }
    for first in submitted {
        assert!(released.contains(&("statuses", Some(first))), "{events:?}");
    }
}

pub(in super::super::super) fn checkpoint(
    role: &'static str,
    first: usize,
    operator_cancel: Option<&AtomicBool>,
) -> Result<()> {
    cancelled(operator_cancel)?;
    record(Event::Checkpoint(role, first));
    // Never inject into the process-wide operator stop flag. Model calls still
    // receive that real flag; only the checkpoint's synthetic action is local.
    let synthetic = AtomicBool::new(false);
    let result = incumbent::checkpoint(role, Some(&synthetic));
    if synthetic.load(Ordering::Relaxed) {
        record(Event::SyntheticCancel);
    }
    cancelled(operator_cancel)?;
    result
}

pub(in super::super::super) fn status_fault(statuses: &[StatusUnit]) -> Result<()> {
    if STATUS.with(|slot| slot.replace(false)) {
        // Synthetic host-ledger fault AFTER the real blocking/checked wait,
        // not a GPU fault. Exercise the unchanged production FFN validator.
        let ffn = statuses
            .iter()
            .find_map(|status| status.ffn.as_ref())
            .ok_or("CED pipeline status fault needs an FFN word")?;
        // This completed group's ledger slot is disjoint from its successor's.
        unsafe {
            *ffn.buffer
                .contents()
                .as_ptr()
                .cast::<i32>()
                .add((ffn.offset / 4) as usize) = -7;
        }
    }
    Ok(())
}

#[test]
fn release_trace_rejects_missing_duplicate_and_late_terminal_waits_cpu() {
    let healthy = [
        Event::Submitted(2),
        Event::Terminal(2),
        Event::Release("statuses", Some(2)),
        Event::Release("seed", None),
        Event::Release("ledger", None),
        Event::Release("pool/storage", None),
    ];
    assert_order(&healthy, 1);
    for bad in [
        vec![healthy[0], healthy[2], healthy[1]],
        vec![healthy[0], healthy[1], healthy[1]],
        vec![healthy[0], healthy[3], healthy[1]],
        healthy[..5].to_vec(),
    ] {
        assert!(catch_unwind(|| assert_order(&bad, 1)).is_err());
    }
}

fn assert_native_signal_scope() {
    assert!(std::env::var_os("V41_TEST_COOPERATIVE_SIGNAL_CHILD").is_none());
    let mut mask = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, std::ptr::null(), &mut mask) },
        0
    );
    for signal in [libc::SIGINT, libc::SIGTERM] {
        assert_eq!(unsafe { libc::sigismember(&mask, signal) }, 0);
    }
}

fn fixture(single: bool, body: impl FnOnce(&Forward<'_>)) {
    // Install BEFORE the lease/context, and destroy every GPU/autorelease
    // owner BEFORE explicit checked restoration. No GPU child or sent signal.
    let mut signals = metal::test_cancellation::CancellationGuard::install().unwrap();
    assert_native_signal_scope();
    {
        let _lease = reuse::require_debug_lease();
        let ctx = MetalContext::new().unwrap();
        objc2::rc::autoreleasepool(|_| {
            let weights = crate::deepseek_v41::forward::toy::Weights::new(&ctx);
            let embedding = incumbent::embeddings(&ctx);
            let mut model = incumbent::nonuniform_model(&weights, &ctx, &embedding);
            configure_four(&mut model, 65);
            if single {
                model.layers.truncate(4);
                model.schedule.layers = 4;
                model.schedule.compression_ratios.truncate(4);
                model.schedule.index_source_layers.retain(|&id| id < 4);
            }
            model.schedule.validate().unwrap();
            assert_eq!(override_layers(), None, "actual default batch selection");
            assert_eq!(batch_layers(), 2);
            assert_eq!(
                plan(&model, 640).decoder.len().div_ceil(2),
                if single { 1 } else { 2 }
            );
            admit(&model, 640);
            prefill::with_batched_f32(|| {
                objc2::rc::autoreleasepool(|_| {
                    let mut warm = model.new_state(640).unwrap();
                    crate::deepseek_v41::forward::ced::run(
                        &model,
                        &mut warm,
                        &incumbent::tokens(640),
                        Some(metal::test_cancellation::flag()),
                        Arithmetic::Production,
                    )
                    .unwrap();
                });
                metal::test_cancellation::check().unwrap();
                body(&model);
                metal::test_cancellation::check().unwrap();
            });
        });
    }
    signals.restore().unwrap();
    assert!(
        !metal::test_cancellation::flag().load(Ordering::Relaxed),
        "operator stop prevents fixture acceptance"
    );
}

fn case(single: bool, role: Option<&'static str>, action: Action, expected: usize) {
    fixture(single, |model| {
        let mut state = objc2::rc::autoreleasepool(|_| model.new_state(640).unwrap());
        let baseline = model.ctx.current_allocated_size();
        objc2::rc::autoreleasepool(|_| {
            let addresses = state.addresses;
            let fault = role.map(|role| Fault::new(role, action));
            let trace = Trace::begin(role.is_none());
            let result = catch_unwind(AssertUnwindSafe(|| {
                crate::deepseek_v41::forward::ced::run(
                    model,
                    &mut state,
                    &incumbent::tokens(640),
                    Some(metal::test_cancellation::flag()),
                    Arithmetic::Production,
                )
            }));
            metal::test_cancellation::check().unwrap();
            if let Some(fault) = &fault {
                assert_eq!(fault.hits(), 1, "{role:?} {action:?}");
            }
            if action == Action::Panic {
                assert!(result.is_err());
            } else {
                let error = result.unwrap().unwrap_err().to_string();
                let expected = match action {
                    Action::Completion => "injected discarded command buffer",
                    Action::Cancel => "cancelled",
                    _ if role.is_none() => "route status -7",
                    _ => role.unwrap(),
                };
                assert!(error.contains(expected), "{error}");
            }
            trace.finish(expected, action == Action::Cancel, role);
            drop(fault);
            assert!(state.is_poisoned() && !state.is_ready());
            assert_eq!(state.position(), 0);
            assert_eq!(state.addresses, addresses);
            assert_eq!(
                state.prefill_origin(),
                crate::deepseek_v41::forward::PrefillExecution::FullForward
            );
            assert!(
                state
                    .gpu_step
                    .as_ref()
                    .unwrap()
                    .attention
                    .ced_public_decoder_progress()
                    .is_none()
            );
            assert!(
                model
                    .step(&mut state, 0, Some(metal::test_cancellation::flag()))
                    .is_err()
            );
            assert!(
                model
                    .advance(&mut state, 0, Some(metal::test_cancellation::flag()))
                    .is_err()
            );
            assert!(
                model
                    .prefill(&mut state, &[0], Some(metal::test_cancellation::flag()))
                    .is_err()
            );
            assert!(model.capture_snapshot(&state).is_err());
            assert_eq!(state.addresses, addresses);
        });
        assert!(
            model.ctx.current_allocated_size() <= baseline,
            "transient resources released with poisoned state alive"
        );
        if action != Action::Cancel {
            objc2::rc::autoreleasepool(|_| healthy(model, single));
        }
    });
}

fn healthy(model: &Forward<'_>, single: bool) {
    metal::test_cancellation::check().unwrap();
    let trace = Trace::begin(false);
    let mut healthy = model.new_state(640).unwrap();
    let (_, report) = crate::deepseek_v41::forward::ced::run(
        model,
        &mut healthy,
        &incumbent::tokens(640),
        Some(metal::test_cancellation::flag()),
        Arithmetic::Production,
    )
    .unwrap();
    metal::test_cancellation::check().unwrap();
    assert!(healthy.is_ready() && healthy.position() == 640);
    assert!(report.fallback.is_none());
    report
        .counters
        .as_ref()
        .unwrap()
        .validate(&plan(model, 640))
        .unwrap();
    trace.finish(if single { 1 } else { 2 }, false, None);
}

#[test]
#[ignore = "self-leased <=128 MiB native default decoder pipeline; synthetic faults; MTL debug"]
fn default_pipeline_submitted_command_unwind_ownership_fault_matrix_gpu() {
    case(false, Some("ced_pipeline_first_encode"), Action::Panic, 0);
    for action in [Action::Error, Action::Panic] {
        case(false, Some("ced_pipeline_successor_encode"), action, 1);
        case(false, Some("ced_pipeline_before_adoption"), action, 2);
        case(false, Some("ced_pipeline_adopted"), action, 2);
    }
    for action in [Action::Error, Action::Panic] {
        case(false, Some("ced_pipeline_popped"), action, 2);
    }
    case(false, Some("ced_pipeline_waited"), Action::Panic, 2);
    case(true, Some("ced_pipeline_popped"), Action::Panic, 1);
    for single in [false, true] {
        case(
            single,
            Some("ced_decoder_batch"),
            Action::Completion,
            if single { 1 } else { 2 },
        );
        case(single, None, Action::Error, if single { 1 } else { 2 });
    }
}

#[test]
#[ignore = "self-leased <=128 MiB native pipeline; local synthetic cancel; no follow-up GPU work"]
fn default_pipeline_successor_encode_cancel_ownership_gpu() {
    case(
        false,
        Some("ced_pipeline_successor_encode"),
        Action::Cancel,
        1,
    );
}

#[test]
#[ignore = "self-leased <=128 MiB native pipeline; local synthetic cancel; no follow-up GPU work"]
fn default_pipeline_adopted_cancel_ownership_gpu() {
    case(false, Some("ced_pipeline_adopted"), Action::Cancel, 2);
}
