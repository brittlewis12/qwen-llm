use super::*;
use crate::serve::jobs::state::{Counters, JobState, StopReason};
use crate::serve::native::{CpuFixture, MAX_TOKEN_PIECE_BYTES, execute};

fn paused(capacity: usize) -> (Sink, Receiver<Event>) {
    let (sender, receiver) = sync_channel(capacity);
    (
        Sink {
            control: ExecutionControl::default(),
            server: Default::default(),
            sender,
            failure: Arc::default(),
            budget: Arc::default(),
            array_budget: Arc::default(),
            array_limit: ARRAY_QUEUE_BYTES,
            staging_started: AtomicBool::new(false),
            staging_hook: None,
            after_record: None,
        },
        receiver,
    )
}

#[test]
fn array_permits_cover_dequeued_payload_and_backpressure_blocks_dependent_rows() {
    let (mut sink, incoming) = paused(4);
    sink.array_limit = 8;
    sink.array(
        json!({"key":"source","quantity":"source_residual"}),
        &[1., -0.],
        Phase::Prefill,
        &Counters::default(),
    )
    .unwrap();
    let event = incoming.recv().unwrap();
    assert_eq!(sink.array_budget.used.load(Ordering::Acquire), 8);
    assert!(
        sink.array(
            json!({"key":"scores"}),
            &[2.],
            Phase::Prefill,
            &Counters::default()
        )
        .is_err()
    );
    sink.record(
        json!({"kind":"readout","retained":{"source_key":"source","logits_key":"scores"}}),
        Phase::Prefill,
        &Counters::default(),
    );
    assert!(incoming.try_recv().is_err());
    let Event::Array(_, payload, _, _, _) = event else {
        panic!("array event")
    };
    assert_eq!(
        payload.bytes,
        [1.0f32.to_le_bytes(), (-0.0f32).to_le_bytes()].concat()
    );
    assert_eq!(sink.array_budget.used.load(Ordering::Acquire), 8);
    drop(payload);
    assert_eq!(sink.array_budget.used.load(Ordering::Acquire), 0);
    assert_eq!(sink.array_budget.peak.load(Ordering::Acquire), 8);
}

#[test]
fn invalid_arrays_and_disconnected_writer_release_all_byte_permits() {
    for values in [
        vec![],
        vec![f32::NAN],
        vec![f32::INFINITY],
        vec![0.; crate::serve::jobs::store::MAX_ARRAY_BYTES / 4 + 1],
    ] {
        let (sink, _incoming) = paused(1);
        assert!(
            sink.array(
                json!({"key":"bad"}),
                &values,
                Phase::Prefill,
                &Counters::default()
            )
            .is_err()
        );
        assert_eq!(sink.array_budget.used.load(Ordering::Acquire), 0);
    }
    let (sink, incoming) = paused(1);
    drop(incoming);
    assert!(
        sink.array(
            json!({"key":"gone"}),
            &[1.],
            Phase::Prefill,
            &Counters::default()
        )
        .is_err()
    );
    assert_eq!(sink.array_budget.used.load(Ordering::Acquire), 0);
    assert_eq!(sink.budget.used.load(Ordering::Acquire), 0);
}

#[test]
fn stalled_store_write_keeps_dequeued_array_charged_until_storage_finishes() {
    let fixture = CpuFixture::new();
    let request = fixture.request("array-hold");
    let mut prepared = fixture
        .profile
        .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
        .ok()
        .unwrap();
    prepared.readouts.archive_bytes = 8;
    let id = fixture
        .store
        .accept_with_archive("array-hold", &request, true, 8)
        .unwrap()
        .status
        .id;
    let writer = Writer::spawn(
        fixture.store.clone(),
        id.clone(),
        fixture.store.control(&id).unwrap(),
        &prepared,
        Default::default(),
    )
    .unwrap();
    writer.wait_ready().unwrap();
    let budget = writer.sink().array_budget.clone();
    std::thread::scope(|scope| {
        let (locked, ready) = sync_channel(1);
        let (release, wait) = sync_channel(1);
        let store = &fixture.store;
        let id = &id;
        let hold = scope.spawn(move || {
            store.with_writer_locked(id, || {
                locked.send(()).unwrap();
                wait.recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            })
        });
        ready
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        writer
            .sink()
            .array(
                json!({"key":"held","quantity":"source_residual"}),
                &[1., 2.],
                Phase::Prefill,
                &prepared.counters(),
            )
            .unwrap();
        let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut queued = 0;
        while queued < EVENTS {
            assert!(
                std::time::Instant::now() < until,
                "writer must dequeue the first array"
            );
            if writer
                .sink()
                .sender
                .try_send(Event::Progress(Phase::Prefill, prepared.counters()))
                .is_ok()
            {
                queued += 1;
            } else {
                std::thread::yield_now();
            }
        }
        // An array plus EVENTS messages cannot fit: the array is now in the
        // blocked store call, not merely retained by the channel.
        assert_eq!(budget.used.load(Ordering::Acquire), 8);
        release.send(()).unwrap();
        hold.join().unwrap();
    });
    writer
        .finish(Outcome::interrupted(prepared.counters()))
        .unwrap();
    assert_eq!(budget.used.load(Ordering::Acquire), 0);
}

fn fitted_job() -> (
    CpuFixture,
    crate::linear_transport::tests::Fixture,
    Prepared,
    Writer,
    String,
) {
    let mut fixture = CpuFixture::new();
    let (files, registry) = crate::serve::native::registry::tests::fitted_fixture();
    let profile = Arc::get_mut(&mut fixture.profile).unwrap();
    profile.plain_readouts = true;
    profile.registry = Some(registry);
    let mut request = fixture.request("fitted");
    request["preconditions"]["asset_identities"] =
        json!({"fit":fixture.profile.registry.as_ref().unwrap().asset("fit").unwrap()["identity"]});
    request["diagnostics"] = json!({"directions":[],"operations":[],"readouts":[
        {"id":"fit","lens":"fit","mode":"full_vocabulary","top_k":2,
            "scope":{"layers":{"kind":"values","values":[0]},"prefill":{"kind":"values","values":[0]}}}]});
    let prepared = fixture
        .profile
        .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
        .ok()
        .unwrap();
    let id = fixture
        .store
        .accept("fitted", &request, true)
        .unwrap()
        .status
        .id;
    let writer = Writer::spawn(
        fixture.store.clone(),
        id.clone(),
        fixture.store.control(&id).unwrap(),
        &prepared,
        Default::default(),
    )
    .unwrap();
    assert!(matches!(writer.wait_ready().unwrap(), Readiness::Execute));
    (fixture, files, prepared, writer, id)
}

const STAGING_RESERVE: u64 = super::super::registry::STAGING_OVERHEAD_BYTES + 8;

#[test]
fn staging_transfers_once_and_writer_remains_available_for_normal_settlement() {
    let (fixture, _files, prepared, writer, id) = fitted_job();
    let mut forwards = 0;
    let outcome =
        crate::serve::native::with_staged(&prepared, writer.sink(), STAGING_RESERVE, |staged| {
            assert_eq!(staged.matrices.len(), 1);
            assert_eq!(staged.matrices.values().next().unwrap().len(), 8);
            assert!(
                writer
                    .sink()
                    .stage(&prepared.staging, STAGING_RESERVE)
                    .is_err()
            );
            execute::run_tokens(
                &prepared,
                writer.sink(),
                &[],
                |_, _, _| {
                    forwards += 1;
                    Ok(vec![0., 1.])
                },
                |_| Ok(b"x".to_vec()),
            )
        });
    assert!(forwards > 0);
    writer.finish(outcome).unwrap();
    let status = fixture.store.status(&id).unwrap();
    assert_eq!(status.state, JobState::Completed);
    assert!(status.result.complete);
}

#[test]
fn stage_fault_cancel_shutdown_and_post_handoff_cancel_prevent_every_forward() {
    use crate::serve::native::staging::Point;
    for case in ["hash", "cancel", "shutdown", "handoff"] {
        let (fixture, files, prepared, mut writer, id) = fitted_job();
        if case == "hash" {
            std::fs::write(files.0.join("transport.f16le"), vec![0; 16]).unwrap();
        }
        let control = writer.sink().control.clone();
        let server = writer.sink().server.clone();
        let checks = Arc::new(AtomicUsize::new(0));
        let observed = checks.clone();
        writer.staging_hook(Arc::new(move |point| {
            observed.fetch_add(1, Ordering::Relaxed);
            if case == "cancel" && point == Point::Checkpoint {
                control.cancel();
            }
            if case == "shutdown" && point == Point::Checkpoint {
                server.close();
            }
            if case == "handoff" && point == Point::Handoff {
                control.cancel();
            }
            Ok(())
        }));
        let outcome =
            crate::serve::native::with_staged(&prepared, writer.sink(), STAGING_RESERVE, |_| {
                panic!("no forward after {case}")
            });
        assert!(checks.load(Ordering::Relaxed) > 0);
        assert_eq!(outcome.counters.consumed_prompt_tokens, 0);
        let expected = match case {
            "hash" => JobState::Failed,
            "shutdown" => JobState::Interrupted,
            _ => JobState::Cancelled,
        };
        if case == "hash" {
            assert_eq!(
                outcome.error.as_ref().unwrap().code,
                "diagnostic_preparation_failed"
            );
        }
        writer.finish(outcome).unwrap();
        let status = fixture.store.status(&id).unwrap();
        assert_eq!(status.state, expected);
        assert!(status.result.complete);
        assert!(status.result.error.is_none());
    }
}

#[test]
fn publication_failure_cannot_discard_a_waiting_staging_response() {
    let (fixture, _files, prepared, writer, id) = fitted_job();
    fixture
        .store
        .fail_once(crate::serve::jobs::store::FaultPoint::RecordSync);
    writer
        .sink()
        .record(json!({"kind":"test"}), Phase::Prefill, &prepared.counters());
    let (request, response, _guard) =
        crate::serve::native::staging::request(&prepared.staging, STAGING_RESERVE).unwrap();
    writer
        .sink()
        .sender
        .try_send(Event::Stage(Box::new(request)))
        .ok()
        .unwrap();
    let failed = response
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    assert!(failed.is_err());
    writer
        .finish(Outcome::interrupted(prepared.counters()))
        .unwrap();
    assert!(fixture.store.status(&id).unwrap().result.error.is_some());
}

#[test]
fn staging_worker_panic_disconnects_wait_and_restart_settles_without_execution() {
    let (mut fixture, _files, prepared, mut writer, id) = fitted_job();
    writer.staging_hook(Arc::new(|_| panic!("synthetic staging worker panic")));
    let outcome =
        crate::serve::native::with_staged(&prepared, writer.sink(), STAGING_RESERVE, |_| {
            panic!("no execution")
        });
    assert_eq!(outcome.reason, StopReason::ExecutionError);
    assert!(writer.finish(outcome).is_err());
    let replacement = Arc::new(
        JobStore::open(
            &fixture.root.join("replacement"),
            crate::serve::jobs::store::Limits::default(),
        )
        .unwrap(),
    );
    drop(std::mem::replace(&mut fixture.store, replacement));
    let reopened = JobStore::open(
        &fixture.root.join("jobs"),
        crate::serve::jobs::store::Limits::default(),
    )
    .unwrap();
    assert_eq!(reopened.status(&id).unwrap().state, JobState::Interrupted);
}

#[test]
fn owner_unwind_after_handoff_joins_without_fabricating_user_cancellation() {
    let (fixture, _files, prepared, mut writer, id) = fitted_job();
    writer.staging_hook(Arc::new(|point| {
        if point == crate::serve::native::staging::Point::Handoff {
            panic!("synthetic owner unwind");
        }
        Ok(())
    }));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::serve::native::with_staged(&prepared, writer.sink(), STAGING_RESERVE, |_| {
                panic!("no execution")
            })
        }))
        .is_err()
    );
    drop(writer);
    let status = fixture.store.status(&id).unwrap();
    assert_eq!(status.state, JobState::Interrupted);
    assert!(!status.cancel_requested);
    assert!(status.result.complete);
}

#[test]
fn late_record_backpressure_does_not_rewrite_successful_terminal_sampling() {
    let (sink, records) = paused(1);
    let prepared = Prepared {
        prompt: vec![0],
        sampling: qwen_llm::sampling::SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        },
        max_tokens: 1,
        record: Vec::new(),
        readouts: Default::default(),
        interventions: Default::default(),
        staging: Default::default(),
    };
    let outcome = execute::run_tokens(
        &prepared,
        &sink,
        &[],
        |_, _, _| Ok(vec![1.0]),
        |_| Ok(b"x".to_vec()),
    );
    assert_eq!(outcome.reason, StopReason::TokenLimit);
    assert_eq!(outcome.counters.sampled_tokens, 1);
    assert_eq!(outcome.counters.consumed_generated_tokens, 0);
    assert!(sink.control.is_cancelled());
    assert_eq!(
        sink.failure.lock().unwrap().as_ref().unwrap().code,
        "artifact_backpressure"
    );
    drop(records);
    assert_eq!(sink.budget.used.load(Ordering::Acquire), 0);
}

#[test]
fn borrowed_sample_records_are_bounded_without_a_byte_array_value_tree() {
    for byte in [0u8, 255, b'"', b'\\'] {
        let (sink, records) = paused(2);
        let prepared = Prepared {
            prompt: vec![0],
            sampling: qwen_llm::sampling::SamplingConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: 0,
            },
            max_tokens: 1,
            record: Vec::new(),
            readouts: Default::default(),
            interventions: Default::default(),
            staging: Default::default(),
        };
        let outcome = execute::run_tokens(
            &prepared,
            &sink,
            &[],
            |_, _, _| Ok(vec![1.0]),
            |_| Ok(vec![byte; MAX_TOKEN_PIECE_BYTES]),
        );
        assert_eq!(outcome.reason, StopReason::TokenLimit);
        assert!(!sink.control.is_cancelled());
        let Event::Progress(_, _) = records.recv().unwrap() else {
            panic!("prefill progress")
        };
        let Event::Record(
            bytes,
            permit,
            _,
            Counters {
                sampled_tokens: 1, ..
            },
        ) = records.recv().unwrap()
        else {
            panic!("terminal sample")
        };
        assert!(bytes.len() < RECORD_BYTES - 128);
        let record: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            record["piece_bytes"].as_array().unwrap().len(),
            MAX_TOKEN_PIECE_BYTES
        );
        drop(permit);
        assert_eq!(sink.budget.used.load(Ordering::Acquire), 0);
    }
}

#[test]
fn queued_cancellation_settles_without_starting_or_reclassifying_execution() {
    let fixture = CpuFixture::new();
    let (id, prepared, control) = fixture.prepare("cancel-before-start");
    fixture.store.cancel(&id).unwrap();
    let writer = Writer::spawn(
        Arc::clone(&fixture.store),
        id.clone(),
        control,
        &prepared,
        Default::default(),
    )
    .unwrap();
    assert!(matches!(writer.wait_ready().unwrap(), Readiness::Settled));
    writer.join_settled().unwrap();
    let status = fixture.store.status(&id).unwrap();
    assert_eq!(status.state, JobState::Cancelled);
    assert_eq!(status.generation.counters.consumed_prompt_tokens, 0);
}

#[test]
fn cancellation_between_signal_and_publication_wins_writer_start() {
    let fixture = CpuFixture::new();
    let (id, prepared, control) = fixture.prepare("cancel-race");
    std::thread::scope(|scope| {
        let (signalled, wait) = sync_channel(1);
        let (release, paused) = sync_channel(1);
        let store = &fixture.store;
        let job_id = &id;
        let cancelling = scope.spawn(move || {
            store.cancel_paused(job_id, || {
                signalled.send(()).unwrap();
                paused
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            })
        });
        wait.recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert!(control.is_cancelled());
        assert_eq!(fixture.store.status(&id).unwrap().state, JobState::Queued);
        let writer = Writer::spawn(
            Arc::clone(&fixture.store),
            id.clone(),
            control,
            &prepared,
            Default::default(),
        )
        .unwrap();
        assert!(matches!(writer.wait_ready().unwrap(), Readiness::Settled));
        writer.join_settled().unwrap();
        let status = fixture.store.status(&id).unwrap();
        assert_eq!(status.state, JobState::Cancelled);
        assert_eq!(status.generation.error, None);
        assert_eq!(status.generation.counters.consumed_prompt_tokens, 0);
        release.send(()).unwrap();
        assert_eq!(cancelling.join().unwrap().unwrap(), status);
    });
}

#[test]
fn prepared_record_failure_is_durable_and_never_reports_model_success() {
    let fixture = CpuFixture::new();
    let (id, prepared, control) = fixture.prepare("failed-publication");
    fixture
        .store
        .fail_once(crate::serve::jobs::store::FaultPoint::RecordSync);
    let writer = Writer::spawn(
        Arc::clone(&fixture.store),
        id.clone(),
        control,
        &prepared,
        Default::default(),
    )
    .unwrap();
    assert!(writer.wait_ready().is_err());
    assert!(writer.join_settled().is_err());
    let status = fixture.store.status(&id).unwrap();
    assert_eq!(status.state, JobState::Failed);
    assert_eq!(status.generation.counters.consumed_prompt_tokens, 0);
    assert_eq!(
        status.generation.error.unwrap().code,
        "artifact_prepare_failed"
    );
}
