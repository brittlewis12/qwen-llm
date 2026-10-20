use super::*;
use crate::serve::{
    control, http, owner_activity, request_profile::RequestProfile, transport, transport_memory,
};
use qwen_llm::metal::{MetalMemorySignals, evaluate_metal_memory_admission_with_cpu_bytes};
use std::cell::{Cell, RefCell};

struct ContractBackend {
    control: u64,
    calls: usize,
}

impl GenerationBackend for ContractBackend {
    fn model_id(&self) -> &str {
        "test"
    }
    fn request_profile(&self) -> RequestProfile {
        RequestProfile::test_qwen()
    }
    fn generate(
        &mut self,
        _: &ServeRequest,
        _: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        self.calls += 1;
        let complete = sink.transport_reserve_bytes();
        assert_eq!(complete, transport::BUFFER_RESERVE_BYTES + self.control);
        let reserve = cpu_reserve(complete, 7).unwrap();
        let boundary = transport::BUFFER_RESERVE_BYTES + self.control + 7 * 4;
        assert_eq!(
            reserve, boundary,
            "the control allowance must be counted once"
        );
        transport_memory::admit_resident_transport(reserve, Some(boundary)).unwrap();
        let denied =
            transport_memory::admit_resident_transport(reserve, Some(boundary - 1)).unwrap_err();
        assert_eq!(
            (denied.status, denied.code),
            (503, Some("memory_admission_denied"))
        );
        assert!(
            transport_memory::admit_resident_transport(reserve, Some(boundary - self.control))
                .is_err()
        );
        Err(ServeError::server_error("reservation boundary checked").into())
    }
}

#[test]
fn prepared_transport_includes_static_or_history_control_reserve_once_at_boundary() {
    // Exercise the real Connection -> HttpProxy -> OwnerSink contract used by
    // control::handle, rather than a fake sink independently adding constants.
    for history in [false, true] {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let _client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (stream, _) = listener.accept().unwrap();
        let activity = owner_activity::OwnerActivity::default();
        let gate = control::ExecutionGate::default();
        let extra = control::cpu_reserve(history);
        transport::inject_proxy_headroom(|| Some(0));
        let mut connection = transport::Connection::prepared(
            stream,
            "test".into(),
            RequestProfile::test_qwen(),
            None,
            activity.admission().try_admit().unwrap(),
            http::HttpRequest {
                method: "POST".into(),
                path: "/v1/responses".into(),
                host: Some("localhost".into()),
                origin: None,
                body: br#"{"model":"test","input":"hi","max_output_tokens":7}"#.to_vec(),
            },
            gate.reserve().unwrap(),
            extra,
        )
        .unwrap();
        let mut backend = ContractBackend {
            control: extra,
            calls: 0,
        };
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        while !connection
            .advance(&mut backend, || {
                anyhow::ensure!(
                    Instant::now() < deadline,
                    "transport contract test timed out"
                );
                Ok(())
            })
            .unwrap()
        {}
        connection.stop_and_join().unwrap();
        assert_eq!(backend.calls, 1);
    }
}

fn signals(budget: u64, resident: u64) -> MetalMemorySignals {
    MetalMemorySignals {
        recommended_max_bytes: budget,
        current_allocated_bytes: resident,
        process_limit_remaining_bytes: Some(budget - resident),
    }
}

fn price(rows: usize) -> Result<prefill_plan::Price, ServeError> {
    Ok(prefill_plan::Price {
        buffers: rows as u64,
        admission: rows as u64,
    })
}

fn admit_runner(bytes: u64, cpu: u64, signals: MetalMemorySignals) -> Result<(), ServeError> {
    let decision = evaluate_metal_memory_admission_with_cpu_bytes(bytes, cpu, 0, signals, true);
    if let Some(denied) = decision.refusal() {
        return Err(adapter_serve_error(Glm5NextDFlashError::Target(
            Glm5NextMetalError::MemoryAdmission {
                denied,
                budget_bytes: signals.recommended_max_bytes,
                advice: CapacityAdvice::NotEvaluated,
            },
        )));
    }
    Ok(())
}

struct Target<'a> {
    resident: &'a Cell<u64>,
    drops: &'a RefCell<Vec<usize>>,
    rows: usize,
}
impl Drop for Target<'_> {
    fn drop(&mut self) {
        self.resident.set(self.resident.get() - self.rows as u64);
        self.drops.borrow_mut().push(self.rows);
    }
}

#[test]
fn runner_pressure_drops_target_and_replans_2048_to_512_before_prefill() {
    let candidates = packed_candidates(2048, 16).unwrap();
    let resident = Cell::new(0);
    let drops = RefCell::new(Vec::new());
    let attempts = RefCell::new(Vec::new());
    let prefills = Cell::new(0);
    let reserve = 17;
    let budget = 2200;
    let runner = 1600;
    let (result, plan) = with_admission_retry(
        &candidates,
        |rows| {
            assert_eq!(
                resident.get(),
                0,
                "failed target must drop before replanning"
            );
            prefill_plan::select(rows, signals(budget, 0), reserve, None, price)
        },
        |rows| {
            attempts.borrow_mut().push(rows);
            resident.set(rows as u64);
            let _target = Target {
                resident: &resident,
                drops: &drops,
                rows,
            };
            admit_runner(runner, reserve, signals(budget, resident.get()))?;
            prefills.set(prefills.get() + 1);
            Ok(Ok::<_, BackendFailure>(rows))
        },
    )
    .unwrap();
    assert_eq!(result.unwrap(), 512);
    assert_eq!(plan.rows, 512);
    assert_eq!(*attempts.borrow(), [2048, 1024, 512]);
    assert_eq!(*drops.borrow(), [2048, 1024, 512]);
    assert_eq!(prefills.get(), 1);
    assert_eq!(resident.get(), 0);
}

#[test]
fn session_planning_can_skip_wide_rows_before_allocation() {
    let mut attempts = Vec::new();
    let (_, plan) = with_admission_retry(
        &packed_candidates(2048, 16).unwrap(),
        |rows| prefill_plan::select(rows, signals(700, 0), 17, None, price),
        |rows| {
            attempts.push(rows);
            admit_runner(100, 17, signals(700, rows as u64))?;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(attempts, [512]);
    assert_eq!(plan.rows, 512);
}

#[test]
fn telemetry_refusal_is_immediate_500_in_plan_or_runner_admission() {
    for in_plan in [true, false] {
        let resident = Cell::new(0);
        let drops = RefCell::new(Vec::new());
        let plans = Cell::new(0);
        let attempts = Cell::new(0);
        let error = with_admission_retry(
            &packed_candidates(2048, 16).unwrap(),
            |rows| {
                plans.set(plans.get() + 1);
                let mut s = signals(4096, 0);
                if in_plan {
                    s.process_limit_remaining_bytes = None;
                }
                prefill_plan::select(rows, s, 17, None, price)
            },
            |rows| {
                attempts.set(attempts.get() + 1);
                resident.set(rows as u64);
                let _target = Target {
                    resident: &resident,
                    drops: &drops,
                    rows,
                };
                let mut s = signals(4096, resident.get());
                s.process_limit_remaining_bytes = None;
                admit_runner(100, 17, s)
            },
        )
        .unwrap_err();
        assert_eq!(
            (error.status, error.code),
            (500, Some("memory_signal_unavailable"))
        );
        assert_eq!(plans.get(), 1);
        assert_eq!(attempts.get(), usize::from(!in_plan));
        assert_eq!(drops.borrow().len(), usize::from(!in_plan));
        assert_eq!(resident.get(), 0);
    }
}

#[test]
fn pressure_never_crosses_verification_floor_or_falls_back_to_serial() {
    let candidates = packed_candidates(32, 12).unwrap();
    assert_eq!(candidates, [32, 16, 12]);
    assert_eq!(packed_candidates(7, 7).unwrap(), [7]);
    assert!(packed_candidates(8, 0).is_err());
    assert!(packed_candidates(8, 16).is_err());
    let mut attempts = Vec::new();
    let error = with_admission_retry(
        &candidates,
        |rows| prefill_plan::select(rows, signals(4096, 0), 17, None, price),
        |rows| {
            attempts.push(rows);
            admit_runner(4096, 17, signals(4096, rows as u64))
        },
    )
    .unwrap_err();
    assert_eq!(attempts, candidates);
    assert_eq!(error.status, 503);
}

#[test]
fn pressure_after_admission_is_not_retried() {
    let mut attempts = 0;
    let (execution, _) = with_admission_retry(
        &packed_candidates(2048, 16).unwrap(),
        |rows| prefill_plan::select(rows, signals(4096, 0), 17, None, price),
        |_| {
            attempts += 1;
            let error = admit_runner(4096, 17, signals(4096, 2048)).unwrap_err();
            Ok(Err::<(), BackendFailure>(error.into()))
        },
    )
    .unwrap();
    assert!(matches!(execution, Err(BackendFailure::Serve(error)) if error.status == 503));
    assert_eq!(attempts, 1);
}
