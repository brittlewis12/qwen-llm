use super::*;
use crate::serve::jobs::state::JobState;
use crate::serve::native::{CpuFixture, run_cpu_readouts};

#[test]
fn metadata_batches_do_not_cross_array_barriers_or_publish_dependents_after_failure() {
    for fail_array in [false, true] {
        let mut fixture = CpuFixture::new();
        let request = fixture.request("barrier");
        let prepared = fixture
            .profile
            .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
            .map_err(|e| e.error.message)
            .unwrap();
        let id = fixture
            .store
            .accept_with_archive("barrier", &request, true, 8)
            .unwrap()
            .status
            .id;
        let (sink, incoming) = super::tests::paused(128);
        let counters = prepared.counters();
        for n in 1..=EVENTS - 2 {
            sink.record(json!({"kind":"test","n":n}), Phase::Prefill, &counters);
        }
        sink.array(
            json!({"key":"source","quantity":"source_residual"}),
            &[1., 2.],
            Phase::Prefill,
            &counters,
        )
        .unwrap();
        sink.record(
            json!({"kind":"test","source_key":"source"}),
            Phase::Prefill,
            &counters,
        );
        let state = worker::State {
            store: fixture.store.clone(),
            id: id.clone(),
            control: sink.control.clone(),
            server: sink.server.clone(),
            failure: sink.failure.clone(),
            budget: sink.budget.clone(),
            array_budget: sink.array_budget.clone(),
            waits: sink.waits.clone(),
        };
        drop(sink);
        if fail_array {
            fixture
                .store
                .fail_once(crate::serve::jobs::store::FaultPoint::ArrayPartialWrite);
        }
        let (started, _ready) = sync_channel(1);
        let (terminal, outcome) = sync_channel(1);
        terminal
            .send(Outcome::interrupted(counters.clone()))
            .unwrap();
        drop(terminal);
        state
            .run(
                prepared.record.clone(),
                counters,
                incoming,
                started,
                outcome,
            )
            .unwrap();
        let records = fixture.store.result(&id, None, 256).unwrap().records;
        assert_eq!(records[1]["n"], 1);
        assert_eq!(records[EVENTS - 2]["n"], EVENTS - 2);
        if fail_array {
            assert_eq!(records.len(), EVENTS - 1);
            assert!(fixture.store.status(&id).unwrap().result.error.is_some());
        } else {
            assert_eq!(records[EVENTS - 1]["kind"], "retained_array");
            assert_eq!(records[EVENTS]["source_key"], "source");
            assert_eq!(
                records[EVENTS + 1]["artifact_writer"]["metadata_batches"],
                2
            );
        }
        let replacement = Arc::new(
            JobStore::open(&fixture.root.join("replacement"), Default::default()).unwrap(),
        );
        drop(std::mem::replace(&mut fixture.store, replacement));
        let reopened = JobStore::open(&fixture.root.join("jobs"), Default::default()).unwrap();
        assert_eq!(reopened.result(&id, None, 256).unwrap().records, records);
        assert_eq!(
            std::fs::metadata(fixture.root.join("jobs").join(id).join("arrays.bin"))
                .unwrap()
                .len(),
            if fail_array { 0 } else { 8 }
        );
    }
}

#[test]
fn coalesced_progress_still_refuses_invalid_intermediate_transitions() {
    let fixture = CpuFixture::new();
    let (id, prepared, _control) = fixture.prepare("bad-progress");
    let (sink, incoming) = super::tests::paused(128);
    for consumed_prompt_tokens in [1, 0, 2] {
        sink.progress(
            Phase::Prefill,
            &Counters {
                consumed_prompt_tokens,
                ..prepared.counters()
            },
        );
    }
    let state = worker::State {
        store: fixture.store.clone(),
        id: id.clone(),
        control: sink.control.clone(),
        server: sink.server.clone(),
        failure: sink.failure.clone(),
        budget: sink.budget.clone(),
        array_budget: sink.array_budget.clone(),
        waits: sink.waits.clone(),
    };
    drop(sink);
    let (started, _ready) = sync_channel(1);
    let (terminal, outcome) = sync_channel(1);
    terminal
        .send(Outcome::interrupted(Counters {
            consumed_prompt_tokens: 2,
            ..prepared.counters()
        }))
        .unwrap();
    drop(terminal);
    state
        .run(
            prepared.record.clone(),
            prepared.counters(),
            incoming,
            started,
            outcome,
        )
        .unwrap();
    assert_eq!(
        fixture
            .store
            .status(&id)
            .unwrap()
            .result
            .error
            .unwrap()
            .code,
        "artifact_write_failed"
    );
    assert_eq!(
        fixture.store.result(&id, None, 128).unwrap().records.len(),
        1
    );
}

#[test]
fn maximum_admitted_readout_scope_completes_with_a_stalled_writer() {
    let mut fixture = CpuFixture::new();
    let profile = Arc::get_mut(&mut fixture.profile).unwrap();
    profile.plain_readouts = true;
    profile.layers = 64;
    let mut request = fixture.request("maximum-scope");
    request["input"]["messages"][0]["content"] = json!("a".repeat(80));
    request["preconditions"]["asset_identities"] = json!({"plain":"cpu-fixture"});
    request["diagnostics"] = json!({"directions":[],"operations":[],"readouts":(0..4).map(|i|json!({
        "id":format!("r{i}"),"lens":"plain","mode":"full_vocabulary","top_k":16,
        "scope":{"layers":{"kind":"all"},"prefill":{"kind":"range","start":0,"end":63}}
    })).collect::<Vec<_>>()});
    let prepared = fixture
        .profile
        .prepare(&crate::serve::lens_http::input::Request::parse(&request).unwrap())
        .map_err(|e| e.error.message)
        .unwrap();
    assert_eq!(
        prepared.readouts.output_rows,
        super::super::readouts::MAX_ROWS
    );
    assert_eq!(
        prepared.readouts.output_scores,
        super::super::readouts::MAX_SCORES
    );
    assert_eq!(
        prepared.readouts.head_evaluations,
        super::super::readouts::MAX_HEAD_EVALUATIONS
    );
    let id = fixture
        .store
        .accept("maximum-scope", &request, true)
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
    let waits = writer.sink().waits.clone();
    let tokenizer = fixture.profile.tokenizer.clone();
    let producer = fixture.store.with_writer_locked(&id, || {
        let producer = std::thread::spawn(move || {
            let outcome = run_cpu_readouts(&prepared, writer.sink(), &tokenizer);
            writer.finish(outcome).unwrap();
        });
        super::tests::wait_for_backpressure(&waits);
        assert!(!producer.is_finished());
        producer
    });
    producer.join().unwrap();
    let status = fixture.store.status(&id).unwrap();
    assert_eq!(status.state, JobState::Completed);
    assert!(status.result.complete && status.result.error.is_none());
    let mut cursor = None;
    let mut rows = 0;
    let mut scores = 0;
    let mut terminal = None;
    loop {
        let page = fixture.store.result(&id, cursor.as_deref(), 256).unwrap();
        for record in page.records {
            if record["kind"] == "readout" {
                rows += 1;
                scores += record["scores"].as_array().unwrap().len();
            }
            if record["kind"] == "generation_terminal" {
                terminal = Some(record);
            }
        }
        cursor = page.next_cursor;
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(rows, 16384);
    assert_eq!(scores, 262144);
    let terminal = terminal.unwrap();
    assert!(
        terminal["artifact_writer"]["backpressure_waits"]
            .as_u64()
            .unwrap()
            > 0
    );
    let batches = terminal["artifact_writer"]["metadata_batches"]
        .as_u64()
        .unwrap();
    assert!(batches > 0);
    eprintln!(
        "maximum-scope publication: {rows} rows, {scores} scores, {batches} metadata batches"
    );
    assert!(
        terminal["artifact_writer"]["peak_record_bytes"]
            .as_u64()
            .unwrap()
            <= QUEUE_BYTES as u64
    );
}
