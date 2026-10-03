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
        },
        receiver,
    )
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
